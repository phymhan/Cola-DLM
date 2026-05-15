# Inference guide

> **English** · [中文](inference_zh.md)

This guide describes how to run the open-source Cola DLM inference pipeline. The pipeline is the direct implementation of the three-step inference algorithm of the paper [*Continuous Latent Diffusion Language Model*](https://arxiv.org/abs/2605.06548): **(i)** prefix encode `z^pre ~ q_phi(z^pre | x^pre)` (Eq. 2.2.4), **(ii)** block-wise prior transport `hat z_0^(b) = Phi^psi_{0←1}(eps^(b); z^pre, hat z_0^(<b))` with `eps^(b) ~ N(0, I)` (Eq. 2.2.5), and **(iii)** conditional decoding `hat x^res ~ p_theta(x^res | z^pre, hat z_0^(1:B))` (Eq. 2.2.6).

There are two ways to run the pipeline, in order of increasing control:

1. **Shell script** — `scripts/run_benchmark.sh`.
2. **Python API** — `from cola_dlm import ColaDiTModel, ColaTextVAEModel, generate_task_repaint_inference`.

## 1. Shell script

Place your HuggingFace-format model weights under `hf_models/cola_dlm/` (with `cola_dit/` and `cola_vae/` subdirectories) and `tokenizer.json` under `hf_models/`. Then:

```bash
# Batch evaluation (8 tasks)
bash scripts/run_benchmark.sh

# Override paths, GPU count, or task list
DIT_PATH=... VAE_PATH=... NUM_GPUS=4 TASKS="lambada mmlu" bash scripts/run_benchmark.sh
```

The script honours environment variables to override model paths, batch size, sampling settings, and GPU count. The full list is documented at the top of the script.

## 2. Module CLI

The module CLI is a thin wrapper around the Python API:

```bash
python -m cola_dlm.inference \
    --dit_path hf_models/cola_dlm/cola_dit \
    --vae_path hf_models/cola_dlm/cola_vae \
    --tokenizer_path hf_models/tokenizer.json \
    --input_jsonl generate_task_data/lambada.jsonl \
    --output_dir eval_output/my_run \
    --task_name lambada \
    --batch_size 70 \
    --max_samples 1000 \
    --max_new_tokens 32 \
    --timestep_num 16 \
    --guidance_scale 7.0 \
    --temperature 0.0 \
    --eos_token_id 100257 \
    --im_end_token_id 100265
```

### Key CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--timestep_num` | 16 | Number of denoising steps per block. |
| `--guidance_scale` | 7.0 | CFG scale (1.0 = off, 7.0 = strong). |
| `--max_new_tokens` | 32 | Generation budget per prompt. |
| `--temperature` | 0.0 | 0 = greedy; 0.8 ≈ creative. |
| `--top_k` / `--top_p` | 50 / 0.9 | Truncation samplers (ignored when `temperature=0`). |

## 3. Python API

```python
from cola_dlm import ColaDiTModel, ColaTextVAEModel, generate_task_repaint_inference
import torch
from tokenizers import Tokenizer

device = torch.device("cuda")
dit = ColaDiTModel.from_pretrained("hf_models/cola_dlm/cola_dit").to(device)
vae = ColaTextVAEModel.from_pretrained("hf_models/cola_dlm/cola_vae").to(device)
tokenizer = Tokenizer.from_file("hf_models/tokenizer.json")

prompts = [
    {"question": "Question: What is the capital of France? Answer:"},
    {"question": "Question: 2 + 2 equals? Answer:"},
]

results = generate_task_repaint_inference(
    dit=dit,
    vae=vae,
    tokenizer=tokenizer,
    prompts=prompts,
    task_name="lambada",
    device=device,
    T=1000.0,
    timestep_num=16,
    guidance_scale=7.0,
    max_new_tokens=32,
    temperature=0.0,
    pad_token_id=100277,
    eos_token_id=100257,
    im_end_token_id=100265,
)
for r in results:
    print(r["prompt"], "→", r["generate"])
```

`generate_task_repaint_inference` is batch-safe, manages DiT / VAE KV caches automatically, and returns a list of dicts with keys `id`, `prompt`, `generate`, `ground_truth` (and optionally `choices`).

## 4. Notes on prompt length, CFG, and the first generation block

The block-wise prior transport step `hat z_0^(b) = Phi^psi_{0←1}(eps^(b); z^pre, hat z_0^(<b))` (Eq. 2.2.5 of the paper) is realized by two DiT forward passes per timestep: a **conditional** pass that attends to the prompt history, and an **unconditional** pass with an empty prefix. CFG fuses the two via `pred = uncond + guidance_scale * (cond - uncond)`.

When a tokenised prompt is shorter than `block_size` (typically 16), the prefix latent is empty for `b = 1`, so the conditional and unconditional paths become mathematically identical — `inference.py` automatically falls back to `guidance_scale=1.0` **for that sample's first block only**. From the second block onward every sample has a non-empty prefix (the just-generated `hat z_0^(<b)`) and the configured `guidance_scale` is restored for the whole batch.

Practical consequence: short prompts still work, but the first block relies on clean-guidance (prompt positions pinned to ground truth) instead of amplified CFG. For cleaner outputs, wrap short prompts in a QA template so the tokenised form is `>= block_size` tokens:

```text
Question: What is the capital of France? Answer:
```

## 5. Multi-GPU data-parallel evaluation

`scripts/run_benchmark.sh` spawns one `python -m cola_dlm.inference` process per GPU, sharding the dataset with `--rank / --world_size`. Per-rank JSONL outputs are merged at the end into a single `<task>.jsonl` file.

