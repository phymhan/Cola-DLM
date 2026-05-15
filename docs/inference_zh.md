# 推理指南

> [English](inference.md) · **中文**

本指南介绍如何运行 Cola DLM 的开源推理流水线。整个流水线严格对应论文 [*Continuous Latent Diffusion Language Model*](https://arxiv.org/abs/2605.06548) 的三步推理算法：**(i)** 前缀编码 `z^pre ~ q_phi(z^pre | x^pre)`（论文式 2.2.4）；**(ii)** 分块先验传输 `hat z_0^(b) = Phi^psi_{0←1}(eps^(b); z^pre, hat z_0^(<b))`，其中 `eps^(b) ~ N(0, I)`（式 2.2.5）；**(iii)** 条件解码 `hat x^res ~ p_theta(x^res | z^pre, hat z_0^(1:B))`（式 2.2.6）。

按控制粒度由低到高，共有两种推理方式：

1. **Shell 脚本** — `scripts/run_benchmark.sh`。
2. **Python API** — `from cola_dlm import ColaDiTModel, ColaTextVAEModel, generate_task_repaint_inference`。

## 1. Shell 脚本

将 HuggingFace 格式的模型权重放到 `hf_models/cola_dlm/`（包含 `cola_dit/` 和 `cola_vae/` 子目录），`tokenizer.json` 放在 `hf_models/` 下，然后：

```bash
# 批量评测（8 个任务）
bash scripts/run_benchmark.sh

# 覆盖路径、GPU 数量或任务列表
DIT_PATH=... VAE_PATH=... NUM_GPUS=4 TASKS="lambada mmlu" bash scripts/run_benchmark.sh
```

脚本支持通过环境变量覆盖模型路径、batch size、采样参数与 GPU 数量。完整变量列表见脚本顶部注释。

## 2. 模块级 CLI

模块 CLI 是 Python API 的轻量封装：

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

### 关键 CLI 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--timestep_num` | 16 | 每个 block 的降噪步数。 |
| `--guidance_scale` | 7.0 | CFG 强度（1.0 = 关闭，7.0 = 较强）。 |
| `--max_new_tokens` | 32 | 每个 prompt 的生成预算。 |
| `--temperature` | 0.0 | 0 = greedy；0.8 约为 "有创造力"。 |
| `--top_k` / `--top_p` | 50 / 0.9 | 截断采样（`temperature=0` 时会被忽略）。 |

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

`generate_task_repaint_inference` 原生支持 batch，自动管理 DiT / VAE 的 KV cache，返回一个 dict 列表，key 至少包括 `id`、`prompt`、`generate`、`ground_truth`（MC 任务还会带 `choices`）。

## 4. 关于 prompt 长度、CFG 与首生成 block 的说明

分块先验传输步 `hat z_0^(b) = Phi^psi_{0←1}(eps^(b); z^pre, hat z_0^(<b))`（论文式 2.2.5）在每个 timestep 由两次 DiT 前向构成：一次**条件前向**（attend 到 prompt 历史），一次**无条件前向**（空前缀）。CFG 把二者按 `pred = uncond + guidance_scale * (cond - uncond)` 融合。

当 prompt 分词后长度短于 `block_size`（默认 16）时，`b = 1` 的前缀隐变量为空，条件与无条件两条前向在数学上完全相同 —— 因此 `inference.py` 会 **仅对该样本的首 block** 自动将 `guidance_scale` 回退为 `1.0`。从第二个 block 开始，所有样本都有非空前缀（即刚生成的 `hat z_0^(<b)`），CFG 会被恢复为你设定的强度。

实际影响：短 prompt 仍然可用，但首 block 只能靠 "clean-guidance"（prompt 位置被钉到 ground truth）而非放大的 CFG 提供指导。要获得更干净的输出，请将短 prompt 包在 QA 模板里，让分词后 ≥ `block_size`，例如：

```text
Question: What is the capital of France? Answer:
```

## 5. 多 GPU 数据并行评测

`scripts/run_benchmark.sh` 会为每张 GPU 启动一个 `python -m cola_dlm.inference` 进程，以 `--rank / --world_size` 对数据集做 stride 切片。每个 rank 输出独立 JSONL，最终合并为单个 `<task>.jsonl`。

