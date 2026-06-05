"""Inference engine for Cola-DLM chat generation.

Wraps the block-by-block latent generation pipeline into a streaming
interface with text-based stop detection.
"""

from dataclasses import dataclass, field

import torch
from typing import Optional

from .modeling_cola_dit import ColaDiTModel
from .modeling_cola_vae import ColaTextVAEModel
from .attention_utils import create_na_block_causal_mask  # noqa: F401 (kept for future use)
from .inference import sample_with_strategies


@dataclass
class BlockOutput:
    """Output from one generation block."""
    token_ids: list[int]
    text: str
    step: int
    is_prompt: list[bool] = field(default_factory=list)
    is_intermediate: bool = False


def _shape_tensor(lens, device):
    return torch.tensor([[l] for l in lens], dtype=torch.long, device=device)


class ColaEngine:
    """Block-by-block generation engine for Cola-DLM.

    Usage::

        engine = ColaEngine(dit, vae, tokenizer)
        for block_text in engine.generate("User: hello\\nAssistant: "):
            print(block_text, end="", flush=True)
    """

    def __init__(self, dit: ColaDiTModel, vae: ColaTextVAEModel, tokenizer):
        self.dit = dit
        self.vae = vae
        self.tokenizer = tokenizer
        self.block_size = dit.block_size
        self.patch_size = vae.patch_size
        self.latent_dim = vae.config.latent_dim
        self.device = next(dit.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 128,
        timestep_num: int = 16,
        guidance_scale: float = 7.0,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        block_size: Optional[int] = None,
        stop_text: str = "■",
        stop_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        stream_steps: int = 0,
        T: float = 1000.0,
        seed: int = None,
    ):
        """Generate text block-by-block, yielding decoded text per block.

        Args:
            prompt_ids: tokenized prompt (list of int token IDs)
            stop_text: stop generation when decoded text contains this string
        """
        dit = self.dit
        vae = self.vae
        device = self.device
        bs = block_size or self.block_size
        patch_size = self.patch_size
        chunk = patch_size * bs

        dit.eval()
        vae.eval()

        if seed is not None:
            torch.manual_seed(seed)

        scale = vae.scaling_factor
        shift = vae.shifting_factor

        # Pad prompt to multiple of chunk
        if pad_token_id is None and stop_text is not None:
            pad_token_id = self.tokenizer.encode(stop_text).ids[0]
        ids = list(prompt_ids)
        prompt_len = len(ids)
        pad_len = (chunk - len(ids) % chunk) % chunk
        ids = ids + [pad_token_id] * pad_len

        input_ids = torch.tensor(ids, dtype=torch.long, device=device)

        # Encode prompt through VAE and normalize to DiT latent space
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc = vae.encode([input_ids])
        latents = ((enc.latents_list[0] - shift) * scale).float()

        # Determine first-block repaint setup (decision at latent-patch level)
        n_prompt_latents = (prompt_len + patch_size - 1) // patch_size
        latent_remainder = n_prompt_latents % bs

        if latent_remainder > 0:
            # First gen block partially overlaps with prompt
            first_block_start = (n_prompt_latents // bs) * bs
            first_block_latents = latents[first_block_start : first_block_start + bs].clone()
            flat_mask = torch.zeros(bs, dtype=torch.bool, device=device)
            flat_mask[:latent_remainder] = True
            prefix_latents = latents[:first_block_start]
            trim_tokens = prompt_len - first_block_start * patch_size
            has_repaint = True
        else:
            prefix_latents = latents[:n_prompt_latents]
            first_block_latents = None
            flat_mask = None
            has_repaint = False
            trim_tokens = 0

        # Enable KV cache
        for block in dit.blocks:
            block.set_kv_cache(True)
        vae.set_kv_cache(True)

        # Prefill KV cache with prefix
        prefix_len = prefix_latents.shape[0]
        if prefix_len > 0:
            txt_shape_prefix = _shape_tensor([prefix_len], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _ = dit(
                    txt=prefix_latents.to(torch.bfloat16),
                    txt_shape=txt_shape_prefix,
                    txt_q_shape=txt_shape_prefix,
                    timestep=torch.zeros(prefix_len, device=device, dtype=torch.bfloat16),
                    update_kv=True,
                    use_kv_cache=True,
                )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _ = vae.decode(
                    z=prefix_latents,
                    txt_shape=txt_shape_prefix,
                    txt_q_shape=txt_shape_prefix,
                    update_kv=True,
                )

        # Timesteps for Euler integration
        timesteps = torch.linspace(T, 0, timestep_num + 1, dtype=torch.float32)

        txt_shape_cum = _shape_tensor([prefix_len], device)
        txt_q_shape_block = _shape_tensor([bs], device)
        context_ids = None
        generated_text = ""

        cfg_scale = 1.0 if prefix_len == 0 else guidance_scale
        trunc_ids = [t for t in [stop_token_id, pad_token_id] if t is not None]

        step = 0
        try:
            while True:
                txt_shape_cum = txt_shape_cum + bs

                # Sample noise
                txt = torch.randn(bs, self.latent_dim, device=device)

                # Euler denoising loop
                for euler_step, (t_curr, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
                    ts_batch = torch.full((bs,), t_curr, device=device)
                    dt = (float(t_curr) - float(t_next)) / max(T, 1.0)

                    if step == 0 and has_repaint:
                        ts_batch[flat_mask] = 0
                        txt[flat_mask] = first_block_latents[flat_mask]

                    txt_bf16 = txt.to(torch.bfloat16)
                    ts_bf16 = ts_batch.to(torch.bfloat16)

                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        drift_cond = dit(
                            txt=txt_bf16,
                            txt_shape=txt_shape_cum,
                            txt_q_shape=txt_q_shape_block,
                            timestep=ts_bf16,
                            update_kv=False,
                            use_kv_cache=True,
                        ).txt_sample

                        drift_uncond = dit(
                            txt=txt_bf16,
                            txt_shape=txt_q_shape_block,
                            txt_q_shape=txt_q_shape_block,
                            timestep=ts_bf16,
                            update_kv=False,
                            use_kv_cache=False,
                        ).txt_sample

                    s = cfg_scale if step == 0 else guidance_scale
                    drift = s * (drift_cond - drift_uncond) + drift_uncond
                    txt_next = txt - drift * dt

                    if step == 0 and has_repaint:
                        txt_next[flat_mask] = first_block_latents[flat_mask]

                    txt = txt_next

                    # Intermediate x0 streaming
                    if (stream_steps > 0
                            and (euler_step + 1) % stream_steps == 0
                            and (euler_step + 1) < timestep_num):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            inter_decoded = vae.decode(
                                z=txt,
                                txt_shape=txt_shape_cum,
                                txt_q_shape=txt_q_shape_block,
                                update_kv=False,
                            )
                        inter_logits = inter_decoded.view(1, bs * patch_size, -1)
                        inter_ids = inter_logits.argmax(dim=-1)[0].tolist()
                        if step == 0 and has_repaint:
                            inter_gen_ids = inter_ids[trim_tokens:]
                        else:
                            inter_gen_ids = inter_ids
                        for tid in trunc_ids:
                            if tid in inter_gen_ids:
                                inter_gen_ids = inter_gen_ids[:inter_gen_ids.index(tid)]
                        inter_text = self.tokenizer.decode(inter_gen_ids) if inter_gen_ids else ""
                        yield BlockOutput(
                            token_ids=inter_gen_ids,
                            text=inter_text,
                            step=step,
                            is_intermediate=True,
                        )

                # Decode block via VAE
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    decoded = vae.decode(
                        z=txt,
                        txt_shape=txt_shape_cum,
                        txt_q_shape=txt_q_shape_block,
                        update_kv=True,
                    )
                decoded_logits = decoded.view(1, bs * patch_size, -1)

                block_ids = sample_with_strategies(
                    decoded_logits,
                    generated_ids=context_ids,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )

                if context_ids is None:
                    context_ids = block_ids
                else:
                    context_ids = torch.cat([context_ids, block_ids], dim=1)

                # Decode to text — trim prompt tokens from first block
                all_block_ids = block_ids[0].tolist()
                if step == 0 and has_repaint:
                    is_prompt = [True] * trim_tokens + [False] * (len(all_block_ids) - trim_tokens)
                    gen_token_ids = all_block_ids[trim_tokens:]
                else:
                    is_prompt = [False] * len(all_block_ids)
                    gen_token_ids = all_block_ids

                # Truncate at stop/pad token if found
                should_stop = False
                first_trunc = len(gen_token_ids)
                for tid in trunc_ids:
                    if tid in gen_token_ids:
                        first_trunc = min(first_trunc, gen_token_ids.index(tid))
                if first_trunc < len(gen_token_ids):
                    gen_token_ids = gen_token_ids[:first_trunc]
                    should_stop = True

                block_text = self.tokenizer.decode(gen_token_ids)

                # Check text-based stop
                if stop_text and stop_text in (generated_text + block_text):
                    block_text = block_text[: block_text.index(stop_text)] if stop_text in block_text else block_text
                    should_stop = True

                # Update KV cache (skip if stopping)
                if not should_stop:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _ = dit(
                            txt=txt.to(torch.bfloat16),
                            txt_shape=txt_shape_cum,
                            txt_q_shape=txt_q_shape_block,
                            timestep=torch.zeros(bs, device=device, dtype=torch.bfloat16),
                            update_kv=True,
                            use_kv_cache=True,
                        )

                yield BlockOutput(
                    token_ids=gen_token_ids,
                    text=block_text,
                    step=step,
                    is_prompt=is_prompt,
                )

                if should_stop:
                    break

                generated_text += block_text
                step += 1
                cfg_scale = guidance_scale

                if step * bs * patch_size >= max_new_tokens:
                    break

        finally:
            for block in dit.blocks:
                block.set_kv_cache(False)
            vae.set_kv_cache(False)
