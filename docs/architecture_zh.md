# 架构

> [English](architecture.md) · **中文**

本文从论文符号、张量 shape、注意力约定与分块先验传输循环四个角度描述 Cola DLM 的推理流水线。实现代码位于 [`cola_dlm/`](../cola_dlm/)。完整理论见论文 [arXiv:2605.06548](https://arxiv.org/abs/2605.06548) 第 2 节 "Continuous Latent Diffusion Language Model"。

## 0. 层次化隐变量框架

Cola DLM 在离散文本 `x` 与连续隐序列 `z_0 ∈ R^d` 上定义一个**层次化隐变量语言模型**：

```
p(x, z_0) = p_theta(x | z_0) * p_psi(z_0),
p(x)      = ∫ p_theta(x | z_0) * p_psi(z_0) dz_0.
```

（论文式 2.1.1）。其中编码器 `q_phi(z_0 | x)` **不属于**生成模型本身，仅作为变分推断与（推理时）前缀编码使用。隐先验由一个连续流（continuous flow）刻画，基础分布 `p_1 = N(0, I)`，可学习向量场 `v_psi(z_t, t)`：

```
z_1 ~ p_1,    dz_t/dt = v_psi(z_t, t),    z_0 = Phi^psi_{0<-1}(z_1).
```

（式 2.1.2）。对序列而言，将 `z_0` 沿序列维分解为 `B` 个 block：`z_0 = (z_0^(1), ..., z_0^(B))`，先验自然写成块因果分解：

```
p_psi(z_0) = p_psi(z_0^(1)) * ∏_{b≥2} p_psi(z_0^(b) | z_0^(<b))
```

（式 2.1.4）。这正是 [`cola_dlm/attention_utils.py`](../cola_dlm/attention_utils.py) 中分块因果 mask 所强制的可见性约束，也是 [`cola_dlm/inference.py`](../cola_dlm/inference.py) 采用 block-by-block 生成循环的根本原因。

仓库发布的 checkpoint 对应论文 RQ4 scaling 曲线中训练量最大的 **2000 EFLOPs** 节点。所有训练阶段的代码路径（Stage 1 / Stage 2 损失、Flow Matching 求解器、reference-encoder 正则）在 §9 给出概要，仅供参考；本仓库本身只发布推理代码。

## 1. 推理流水线（前缀编码 → 分块先验传输 → 条件解码）

对一个 prompt batch，本仓库的推理路径严格对应论文式 2.2.4–2.2.6 的三步流程：

1. **分词**：使用项目 tokenizer（BPE / WordPiece，词表约 10 万）。
2. **按样本对齐**：每条序列末尾 pad 到 `patch_size * block_size` 的整数倍，pad 使用 `pad_token_id`。**不做 batch 级别的 `max_len` padding**，序列保持变长。
3. **前缀编码**：`ColaTextVAEModel.encode` 实现 `z^pre ~ q_phi(z^pre | x^pre)`，将 `(L_i,)` 的 token 序列编码为连续隐向量 `(n_i, latent_dim)`，其中 `n_i = L_i / patch_size`。
4. **标签分类**：给每个 latent 位置打 `1`（prompt）、`2`（待生成）、`3`（硬 pad）。尾部的 `3` 会被裁掉，后续流程只传 `1/2` 两类。
5. **分块先验传输**：`ColaDiTModel` 在可见集 `V_b = {sg(z_0^(<b)), z_t^(b)}` 约束下（式 2.2.3），每步实现一个 block 的 `Phi^psi_{0←1}`，按 `block_size` 分块、在 CFG 下迭代地传输隐空间的先验。
6. **条件解码**：`ColaTextVAEModel.decode` 实现 `hat x^res ~ p_theta(x^res | z^pre, hat z_0^(1:B))`，把刚刚传输得到的 latent block 解码回 token logits。
7. **采样**：argmax / top-k / top-p / repetition penalty，选下一段 token 并追加到对应样本的输出 buffer。
8. 重复 5–7，直到所有样本都遇到 EOS（`eos_token_id` / `im_end_token_id`），或达到 `max_new_tokens` 预算。

完整循环在 [`cola_dlm/inference.py::generate_task_repaint_inference`](../cola_dlm/inference.py)。

## 2. NA flatten-concat 布局

所有模型与注意力 kernel 都使用统一的 **无填充（NA）** 布局：

- 样本级张量沿单一序列轴拼接，例如 `txt: (L_total, c)`，其中 `L_total = sum(n_i)`。
- 伴随张量 `txt_shape: (B, 1)` 记录每个样本的 **K 侧** 长度。
- `txt_q_shape: (B, 1)` 记录每个样本的 **Q 侧** 长度：等于 `txt_shape[i]`（前缀自注意力）或 `block_size`（分块先验传输的一个 query block）。

### 分块因果 mask（可见集 V_b）

[`cola_dlm/attention_utils.py::create_na_block_causal_mask`](../cola_dlm/attention_utils.py) 直接由 `txt_shape` / `txt_q_shape` 一次性构造 `(1, 1, L_q_total, L_k_total)` 的加性 mask，对应论文式 2.2.3 的 `V_b`：

- 样本之间完全 **阻断**（防止跨样本泄漏）。
- 样本内 Q 所在 block `b_q` 只能看到 K 的 block `b_k ≤ b_q`（块间因果，块内双向）。
- 允许位置填 `0`，其余填 `dtype.min`，可以安全地在 softmax 前加到 `QK^T` 上。

## 3. `ColaTextVAEModel`

[`cola_dlm/modeling_cola_vae.py`](../cola_dlm/modeling_cola_vae.py) 同时实现式 2.1.1 中的推断编码器 `q_phi(z_0 | x)` 与条件解码器 `p_theta(x | z_0)`，是一个分块因果 Transformer：

- Token embedding → `encoder_num_blocks` 层自注意力 → LayerNorm → 投影到 `2 * latent_dim`（mean + log-variance）。
- 对称的 `decoder_num_blocks` 层 decoder，最后投影回 vocab 维度。
- SwiGLU MLP；RoPE 位置编码（`rope_theta=500000`，可选 full-precision）；可选 QK-norm / `clip_qkv`；可配置 block-causal / full-causal / bidirectional 三种 mask。
- 使用 [`DiagonalGaussianDistribution`](../cola_dlm/modeling_cola_vae.py) 封装后验，`encode()` 可返回确定性后验（推理）或采样 latent（类训练形式）。

### 公开 API

```python
enc = vae.encode(input_ids_list)      # → TextVAEEncoderOutput(latents_list, latent_dists)
# latents_list[i]: (n_i, latent_dim) — 实现 z^pre ~ q_phi(z^pre | x^pre)。
out = vae.decode(
    latents,           # (L_total, latent_dim) — z^pre 与/或 hat z_0^(1:B)
    txt_shape,         # (B, 1) — 每样本 K 长度
    txt_q_shape,       # (B, 1) — 每样本 Q 长度
    update_kv=True,    # 是否写入 VAE decoder KV cache
    use_kv_cache=True,
)
# out.logits: (L_q_total, vocab_size) — 实现 hat x^res ~ p_theta(x^res | z^pre, hat z_0^(1:B))。
```

### Latent 缩放

训练时 VAE latent 使用 `(x - shifting_factor) * scaling_factor` 归一化；这两个常数保存在 config 中，`from_pretrained` 时自动加载。它们决定了后续 DiT 先验 `p_psi` 所拟合的隐空间 `z_0 ∈ R^d` 的几何形状。

## 4. `ColaDiTModel`

[`cola_dlm/modeling_cola_dit.py`](../cola_dlm/modeling_cola_dit.py) 实现式 2.1.4 的**分块因果 DiT 先验 `p_psi(z_0)`**。训练时 DiT 通过条件 Flow Matching 学习向量场 `v_psi(z_t, t; z_0^(<b))`（式 2.1.7）；推理时每次前向都对应积分子 `Phi^psi_{0←1}` 在某一 block 上的一步（式 2.1.2 / 2.2.4）：

- 对 latent 做 patchify，再线性投影到 `txt_dim`。
- 共 `num_layers` 层 transformer block，每层包含：
  - RMSNorm pre-norm；SwiGLU FFN（扩展比 `expand_ratio × txt_dim`）。
  - **AdaLN 条件化**：输入是 sinusoidal timestep embedding（与 `diffusers.get_timestep_embedding(flip_sin_to_cos=False, downscale_freq_shift=0)` 对齐）。
  - 对 `rope_dim` 个通道做 RoPE（split head）。
  - 每样本的 NA 分块因果注意力，强制 `V_b` 可见性约束；`use_kv_cache=True` 时跨 block 缓存 K/V。
- 最后 RMSNorm + 线性投影回 `txt_out_channels`。

### 公开 API

```python
out = dit(
    txt,               # (L_q_total, in_channels) — 当前样本的噪声块 z_t^(b)
    txt_shape,         # (B, 1) K 长度 — 覆盖干净历史 z_0^(<b) + 当前 z_t^(b)
    txt_q_shape,       # (B, 1) Q 长度（生成阶段 == block_size）
    timestep,          # (B,) 或标量 — Flow Matching 时间 t
    update_kv=False,   # 只有需要把当前 block 提交到先验 KV cache 时才 True
    use_kv_cache=True,
)
# out.sample: (L_q_total, out_channels) — 一步先验传输的漂移项，
# 由积分子 x_{t-Δ} = x_t - Δ/T * pred 使用（见 §6）。
```

`ColaDiTModel.blocks[i].set_kv_cache(bool)` 可以在任意层级开关 KV cache。

## 5. Classifier-Free Guidance

[`generate_task_repaint_inference`](../cola_dlm/inference.py) 每个先验传输步做两次前向：

- **条件前向**：Q = 当前 block，K/V = `[prefix_cache | current_block]`，可见 prompt（即 `z^pre` 与已提交的历史 `hat z_0^(<b)`）。
- **无条件前向**：K/V = 仅 `current_block`（空前缀），不可见 prompt。

二者按 `pred = uncond + guidance_scale * (cond - uncond)` 融合。当一条样本的 prompt 短于 `block_size`（即首 block 的前缀为空）时，会对该样本的第一步 **自动将 `guidance_scale` 置为 1.0**，避免放大 bf16 kernel 的数值误差。细节见 [`docs/inference_zh.md`](inference_zh.md)。

## 6. 时间步 schedule

`timesteps = torch.linspace(T, 0, timestep_num + 1)`，默认 `T=1000.0`。DiT 预测速度场 `v_psi`，Euler 风格的更新规则 `x_{t-Δ} = x_t - Δ/T * pred` 在每个 block 内数值实现式 2.1.2。

## 7. KV Cache 约定

`ColaDiTModel`：

- `block.set_kv_cache(True)` 在首次使用时分配一个初始为空的 cache。
- `update_kv=True` 把当前 block 的 Q 投影 **追加** 到 cache；`update_kv=False` 则把 cache 当成只读。
- Cache 在每一层都是一个扁平的 `(L_total, heads, head_dim)` 张量，按 `txt_shape` 切回各个样本。

`ColaTextVAEModel.decode` 遵循同样的约定：生成前调用 `vae.set_kv_cache(True)`，结束后调用 `vae.set_kv_cache(False)` 释放。

## 8. 参考配置（1.8B DiT + 500M VAE）

参考 checkpoint 的规范化 config 默认值已嵌入 `ColaDiTConfig` 和 `ColaTextVAEConfig` 中。本次开源对应论文 RQ4 scaling 曲线中训练量最大的 **2000 EFLOPs** 节点：约 5 亿参数的 Text VAE + 约 18 亿参数的分块因果 DiT 先验，总规模约 2B 参数，与论文中严格匹配的自回归 / LLaDA baseline 在量级上对齐。

## 9. 训练阶段（参考说明）

训练代码**未**随仓库一同发布；本节给出训练流程概要，便于读者把开源权重映射回论文的训练管线。

### Stage 1：Text VAE 预训练（式 2.2.1）

目标是先建立稳定的"文本↔隐空间"对应关系，让后续的先验建模有一个良好刻度的对象可拟合。损失同时包含重构、BERT 风格 mask 损失 `L_mask` 与对基础先验 `p_base` 的 KL 正则：

```
L_VAE = - E_{q_phi(z_0|x)}[log p_theta(x | z_0)]
        + beta * KL(q_phi(z_0|x) || p_base(z_0))
        + lambda_mask * L_mask.
```

VAE 的 encoder / decoder 均严格因果，且**不**对序列长度做压缩 —— 这一阶段优先建立稳定表示，而非最终先验。

### Stage 2：VAE + 分块因果 DiT 联合预训练（式 2.2.3）

学习最终的隐先验 `p_psi(z_0)`，与此同时通过正则项让 VAE 仍然可训但不会出现表示漂移。block `b` 的可见集为 `V_b = {sg(z_0^(<b)), z_t^(b)}`，强制块内双向、块间因果（即 §2 的 mask）。损失为

```
L_stage2 = lambda_VAE * ( - E_{q_phi}[log p_theta] + beta * E_{q_phi}[log q_phi]
                          + lambda_mask * L_mask )
         + lambda_fm * L_FM
         + lambda_ref * E[ KL( q_phi(z_0 | x) || q_phi_ref(z_0 | x) ) ],
```

其中条件 Flow Matching 损失为

```
L_FM = sum_{b=1..B} E_{t, z_0, z_1} [ || v_psi(z_t^(b), t; z_0^(<b)) - u_t^(b)(z_0, z_1) ||_2^2 ].
```

（式 2.1.7 / 2.2.3）。第一组保持 autoencoding 结构与正则化的 latent 学习；第二项学习块级条件先验；第三项通过对一个冻结的参考编码器 `q_phi_ref` 求 KL 来抑制隐空间漂移。本仓库发布的分块因果 DiT 即对应这一阶段最终训练得到的先验模型。

### 从训练到推理

推理时三个组件 —— 编码器 `q_phi`、先验 `p_psi`、解码器 `p_theta` —— 均被冻结，[`cola_dlm/`](../cola_dlm/) 中的开源代码路径在 NA flatten-concat 布局下严格实现式 2.2.4–2.2.6。
