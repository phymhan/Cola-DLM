# Architecture

> **English** · [中文](architecture_zh.md)

This document describes the Cola DLM pipeline at the level of paper notation, tensor shapes, attention conventions, and the block-wise prior-transport loop. Implementation lives under [`cola_dlm/`](../cola_dlm/). For the underlying theory, see Section 2 ("Continuous Latent Diffusion Language Model") of the paper [arXiv:2605.06548](https://arxiv.org/abs/2605.06548).

## 0. Hierarchical latent-variable framing

Cola DLM defines a **hierarchical latent-variable language model** over a discrete text sequence `x` and a continuous latent sequence `z_0 ∈ R^d`:

```
p(x, z_0) = p_theta(x | z_0) * p_psi(z_0),
p(x)      = ∫ p_theta(x | z_0) * p_psi(z_0) dz_0.
```

(Eq. 2.1.1 in the paper.) The encoder `q_phi(z_0 | x)` is **not** part of the generative model — it is an inference model used for variational training and, at inference time, for prefix encoding. The latent prior is realized as a continuous-flow prior with base distribution `p_1 = N(0, I)` and learned vector field `v_psi(z_t, t)`:

```
z_1 ~ p_1,    dz_t/dt = v_psi(z_t, t),    z_0 = Phi^psi_{0<-1}(z_1).
```

(Eq. 2.1.2.) For sequences, `z_0` is decomposed into `B` blocks `z_0 = (z_0^(1), ..., z_0^(B))` and the prior factorizes block-causally:

```
p_psi(z_0) = p_psi(z_0^(1)) * ∏_{b≥2} p_psi(z_0^(b) | z_0^(<b))
```

(Eq. 2.1.4.) This block factorization is exactly what the block-causal attention mask in [`cola_dlm/attention_utils.py`](../cola_dlm/attention_utils.py) enforces, and it is the reason the generation loop in [`cola_dlm/inference.py`](../cola_dlm/inference.py) is block-by-block.

The repository ships the trained checkpoint corresponding to the **2000 EFLOPs** entry on the paper's RQ4 scaling curve. All training-time code paths (Stage 1 / Stage 2 losses, Flow Matching solver, reference-encoder regularizer) are summarized in §9 below for reference; the open-source release is inference-only.

## 1. Inference pipeline (prefix encode → block-wise prior transport → conditional decode)

Given a batch of prompts, the open-source inference path implements the three steps of Eq. 2.2.4–2.2.6 of the paper:

1. **Tokenise** each prompt with the project tokenizer (BPE / WordPiece over a 100k-vocab).
2. **Per-sample block align** — pad each sequence to a multiple of `patch_size * block_size` tokens using `pad_token_id`. **No batch-level `max_len` padding** is performed: the sequences remain variable-length.
3. **Prefix encode** — `ColaTextVAEModel.encode` realizes `z^pre ~ q_phi(z^pre | x^pre)` and maps each `(L_i,)` token sequence to a continuous latent `(n_i, latent_dim)`, with `n_i = L_i / patch_size`.
4. **Classify** every latent position as `1` (prompt), `2` (to-generate), or `3` (hard pad). Pad tail is then dropped so only `1/2` latents flow into the DiT.
5. **Block-wise prior transport** — `ColaDiTModel` realizes one block of `Phi^psi_{0←1}` per generation step under the visibility constraint `V_b = {sg(z_0^(<b)), z_t^(b)}` (Eq. 2.2.3), refining the latents one `block_size` chunk at a time under classifier-free guidance.
6. **Conditional decode** — `ColaTextVAEModel.decode` realizes `hat x^res ~ p_theta(x^res | z^pre, hat z_0^(1:B))` and converts each freshly transported block into token logits.
7. **Sample** the next tokens (argmax / top-k / top-p / repetition penalty) and append to the per-sample output buffer.
8. Repeat 5–7 until every sample emits an EOS (`eos_token_id` / `im_end_token_id`) or the generation budget `max_new_tokens` is exhausted.

The full loop is implemented in [`cola_dlm/inference.py::generate_task_repaint_inference`](../cola_dlm/inference.py).

## 2. NA flatten-concat layout

All models and attention kernels speak the same **no-padding** ("NA") layout:

- Per-sample tensors are concatenated along a single sequence axis, e.g. `txt: (L_total, c)` with `L_total = sum(n_i)`.
- A companion tensor `txt_shape: (B, 1)` carries per-sample K-side lengths.
- `txt_q_shape: (B, 1)` carries per-sample Q-side lengths: either equal to `txt_shape[i]` (prefix self-attention) or `block_size` (block-wise prior-transport step).

### Block-causal mask (visible set V_b)

[`cola_dlm/attention_utils.py::create_na_block_causal_mask`](../cola_dlm/attention_utils.py) produces an additive block-diagonal / block-causal mask `(1, 1, L_q_total, L_k_total)` in one pass from `txt_shape` and `txt_q_shape`. It is the implementation of the visible set `V_b` from Eq. 2.2.3:

- Attention is **blocked** across samples (no cross-sample leakage).
- Within a sample, Q block `b_q` attends to K blocks `b_k ≤ b_q` (block-causal across blocks; bidirectional within a block).
- The mask value is `0` at allowed positions and `dtype.min` elsewhere, safe to add onto `QK^T` before the softmax.

## 3. `ColaTextVAEModel`

[`cola_dlm/modeling_cola_vae.py`](../cola_dlm/modeling_cola_vae.py) implements both the inference encoder `q_phi(z_0 | x)` and the conditional decoder `p_theta(x | z_0)` of Eq. 2.1.1. It is a block-causal Transformer with:

- Token embedding → `encoder_num_blocks` self-attention blocks → LayerNorm → projection to `2 * latent_dim` (mean + log-variance).
- A matching decoder of `decoder_num_blocks` blocks back to a vocab-size projection head.
- SwiGLU MLP, RoPE positional encoding (`rope_theta=500000`, full-precision option), optional QK-normalisation, optional `clip_qkv`, configurable block-causal / fully-causal / bidirectional masks.
- A [`DiagonalGaussianDistribution`](../cola_dlm/modeling_cola_vae.py) wrapper so `encode()` can return either deterministic posteriors (inference) or sampled latents (training-style).

### Public API

```python
enc = vae.encode(input_ids_list)      # → TextVAEEncoderOutput(latents_list, latent_dists)
# latents_list[i]: (n_i, latent_dim) — realizes z^pre ~ q_phi(z^pre | x^pre).
out = vae.decode(
    latents,           # (L_total, latent_dim) — z^pre and/or hat z_0^(1:B)
    txt_shape,         # (B, 1) — per-sample K lengths
    txt_q_shape,       # (B, 1) — per-sample Q lengths
    update_kv=True,    # write through the VAE decoder KV cache
    use_kv_cache=True,
)
# out.logits: (L_q_total, vocab_size) — realizes hat x^res ~ p_theta(x^res | z^pre, hat z_0^(1:B)).
```

### Latent scaling

The trainer normalises VAE latents by `(x - shifting_factor) * scaling_factor`; these constants are stored on the config and restored automatically by `from_pretrained`. They control the geometry of the latent space `z_0 ∈ R^d` over which the DiT prior `p_psi` is fit.

## 4. `ColaDiTModel`

[`cola_dlm/modeling_cola_dit.py`](../cola_dlm/modeling_cola_dit.py) implements the **block-causal DiT prior `p_psi(z_0)`** of Eq. 2.1.4. At training time the DiT learns the vector field `v_psi(z_t, t; z_0^(<b))` of Eq. 2.1.7 via conditional Flow Matching; at inference time each forward call realizes one block of the integrator `Phi^psi_{0←1}` (Eq. 2.1.2 / 2.2.4):

- Patchify latents → linear projection to `txt_dim`.
- `num_layers` transformer blocks, each with:
  - RMSNorm pre-norm, SwiGLU FFN (`expand_ratio × txt_dim`).
  - **AdaLN conditioning** on a sinusoidal timestep embedding (matches `diffusers` `get_timestep_embedding(flip_sin_to_cos=False, downscale_freq_shift=0)`).
  - RoPE over `rope_dim` channels (split head).
  - Per-sample NA block-causal attention enforcing the visible set `V_b`; K/V are cached between blocks when `use_kv_cache=True`.
- Final RMSNorm + linear projection back to `txt_out_channels`.

### Public API

```python
out = dit(
    txt,               # (L_q_total, in_channels) — z_t^(b) (current noisy block) for each sample
    txt_shape,         # (B, 1) K lengths   — covers z_0^(<b) (clean history) + z_t^(b)
    txt_q_shape,       # (B, 1) Q lengths   — == block_size during generation
    timestep,          # (B,) or scalar     — Flow Matching time t
    update_kv=False,   # True only when committing the new block to the prior KV cache
    use_kv_cache=True,
)
# out.sample: (L_q_total, out_channels) — one prior-transport step's drift, used by the
# integrator update x_{t-Δ} = x_t - Δ/T * pred (see Section 6).
```

`ColaDiTModel.blocks[i].set_kv_cache(bool)` toggles the cache at any layer depth.

## 5. Classifier-free guidance

Inside [`generate_task_repaint_inference`](../cola_dlm/inference.py) each prior-transport step runs:

- **Conditional pass** — Q = current block, K/V = `[prefix_cache | current_block]`. Attention sees the prompt (`z^pre` plus committed history `hat z_0^(<b)`).
- **Unconditional pass** — K/V = `current_block` only (empty prefix). Attention sees no prompt.

Guidance is fused via `pred = uncond + guidance_scale * (cond - uncond)`. When a sample's prompt is shorter than `block_size` (empty prefix for its first block), guidance is automatically switched off for that sample's first step to avoid amplifying bf16 kernel noise. See [`docs/inference.md`](inference.md) for the gory details.

## 6. Timestep schedule

`timesteps = torch.linspace(T, 0, timestep_num + 1)` with `T=1000.0` by default. The DiT predicts the velocity field `v_psi`, and the Euler-style integrator update `x_{t-Δ} = x_t - Δ/T * pred` numerically realizes Eq. 2.1.2 within one block.

## 7. KV caching contract

`ColaDiTModel`:

- `block.set_kv_cache(True)` allocates an initially empty cache on first use.
- `update_kv=True` **appends** the Q projections for the current block to the cache; `update_kv=False` treats the cache as read-only.
- The cache is a flat `(L_total, heads, head_dim)` tensor per layer, split back into per-sample tensors via `txt_shape`.

`ColaTextVAEModel.decode` follows the same convention: call `vae.set_kv_cache(True)` before a generation run and `vae.set_kv_cache(False)` on teardown.

## 8. Reference configuration (1.8B DiT + 500M VAE)

The canonical config defaults for the reference checkpoint are embedded in `ColaDiTConfig` and `ColaTextVAEConfig`. The released checkpoint matches the **2000 EFLOPs** point on the paper's RQ4 scaling curve, comprising a ~500M-parameter Text VAE and a ~1.8B-parameter block-causal DiT prior, for a total of roughly 2B parameters — strictly matched to the autoregressive and LLaDA baselines in the paper.

## 9. Training stages (for reference)

Training code is **not** released; this section is provided so readers can map the released weights back to the training pipeline of the paper.

### Stage 1 — Text VAE pretraining (Eq. 2.2.1)

Establishes a stable text↔latent correspondence so that subsequent prior modeling has something well-formed to fit. The objective combines reconstruction, a BERT-style masking loss `L_mask`, and a KL regularizer to a base prior `p_base`:

```
L_VAE = - E_{q_phi(z_0|x)}[log p_theta(x | z_0)]
        + beta * KL(q_phi(z_0|x) || p_base(z_0))
        + lambda_mask * L_mask.
```

Both the VAE encoder and decoder are strictly causal and do **not** compress the sequence length — the goal at this stage is a stable representation, not the final prior.

### Stage 2 — Joint VAE + block-causal DiT pretraining (Eq. 2.2.3)

Learns the final latent prior `p_psi(z_0)` while keeping the VAE trainable under regularizers that prevent representation drift. For block `b`, the visible set is `V_b = {sg(z_0^(<b)), z_t^(b)}`, enforcing bidirectional attention within a block and causal dependence across blocks (exactly the mask in §2). The objective is

```
L_stage2 = lambda_VAE * ( - E_{q_phi}[log p_theta] + beta * E_{q_phi}[log q_phi]
                          + lambda_mask * L_mask )
         + lambda_fm * L_FM
         + lambda_ref * E[ KL( q_phi(z_0 | x) || q_phi_ref(z_0 | x) ) ],
```

with the conditional Flow Matching loss

```
L_FM = sum_{b=1..B} E_{t, z_0, z_1} [ || v_psi(z_t^(b), t; z_0^(<b)) - u_t^(b)(z_0, z_1) ||_2^2 ].
```

(Eq. 2.1.7 / 2.2.3.) The first group preserves the autoencoding structure with regularized latent learning, the second term learns the block-level conditional prior, and the third term suppresses latent drift via a KL to a frozen reference encoder `q_phi_ref`. The block-causal DiT shipped in this repo is the resulting prior model.

### From training to inference

At inference, all three components — the encoder `q_phi`, the prior `p_psi`, and the decoder `p_theta` — are frozen, and the open-source code paths in [`cola_dlm/`](../cola_dlm/) realize Eq. 2.2.4–2.2.6 verbatim under the NA flatten-concat layout.
