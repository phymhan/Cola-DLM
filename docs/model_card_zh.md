# Model Card：Cola DLM

> [English](model_card.md) · **中文**

本 Model Card 遵循 Mitchell 等人（2019）*Model Cards for Model Reporting* 的结构，描述与本仓库一同发布的参考 `cola-dlm` checkpoint，以及对应的论文 [*Continuous Latent Diffusion Language Model*](https://arxiv.org/abs/2605.06548)。

## 论文信息

- **论文标题**：Continuous Latent Diffusion Language Model
- **作者**：Hongcan Guo, Qinyu Zhao, Yian Zhao, Shen Nie, Rui Zhu, Qiushan Guo, Feng Wang, Tao Yang, Hengshuang Zhao, Guoqiang Wei, Yan Zeng（ByteDance Seed 等）
- **arXiv**：[arxiv.org/abs/2605.06548](https://arxiv.org/abs/2605.06548)
- **HuggingFace Daily Paper**：[huggingface.co/papers/2605.06548](https://huggingface.co/papers/2605.06548)
- **项目主页**：[hongcanguo.github.io/Cola-DLM](https://hongcanguo.github.io/Cola-DLM/)

## 模型基础信息

- **模型名称**：Cola DLM（Continuous Latent Diffusion Language Model，连续隐空间扩散语言模型）
- **版本**：`0.1.0`
- **发布日期**：2026
- **方法路线**：层次化的连续隐空间扩散语言模型 —— 由 Text VAE 编码器 `q_phi(z_0 | x)` + 条件解码器 `p_theta(x | z_0)`，配合一个由 Flow Matching 学习的分块因果 DiT 先验 `p_psi(z_0)` 组成。扩散在这里用于**隐空间先验传输**（论文式 2.1.4），而不是用来恢复 token。
- **模型结构**：两个协同模块
  - `ColaTextVAEModel` —— 约 5 亿参数的 Text VAE，实现 `q_phi` 与 `p_theta`（4 encoder + 4 decoder block，`dim=1536`，`ffn_dim=6144`，`latent_dim=16`）。
  - `ColaDiTModel` —— 约 18 亿参数的 1-D 扩散 Transformer，实现分块因果隐先验 `p_psi`（24 层，`txt_dim=emb_dim=2048`，16 头 × 128 head_dim，`expand_ratio=4`）。
- **训练量对应节点**：本次开源的权重对应论文 RQ4 scaling 曲线中训练量最大的 **2000 EFLOPs** 节点；总参数量约 2B，与论文中严格匹配的自回归 / LLaDA baseline 在量级上对齐。
- **许可证**：[Apache License 2.0](../LICENSE)。
- **框架**：PyTorch 2.1+；HuggingFace Transformers 4.40+。
- **数值精度**：master 权重保持在 fp32；CUDA 前向在 `torch.autocast(dtype=torch.bfloat16)` 下执行。
- **Tokenizer**：OLMo 2 tokenizer（10 万词表的 BPE）；`pad_token_id=100277`，`eos_token_id=100257`，`im_end_token_id=100265`。

## 预期用途

### 主要用例

- 层次化 / 连续隐空间文本扩散语言模型的研究。
- 闭域问答与零 / 少样本 benchmark 评测（LAMBADA、MMLU、HellaSwag、OBQA、RACE、SIQA、SQuAD、Story Cloze 等）—— 即论文 RQ4 中的 8 项任务。
- 研究"隐空间先验传输"式文本生成的性质（CFG 可控性、分块 re-painting、隐空间语义结构等）。

### 非预期用途

- 安全 / 决策关键场景（医疗、法律、金融等建议）。
- 生成违反法律、平台政策或科研伦理的内容。
- **不建议** 作为直接替代的聊天机器人：预训练目标是层次化隐先验建模，而非指令对话。

## 影响因素

- 训练语料以英文为主；其他语言性能未经验证，预期较差。
- 生成质量对 prompt 长度相对于 `block_size`（默认 16）非常敏感：短于一个 block 的 prompt 会让首 block 自动关闭 CFG，质量略有下降。缓解方式见 [`docs/inference_zh.md`](inference_zh.md)。

## 指标

零样本 8 项 benchmark 准确率，batch size 70，`timestep_num=16`，`guidance_scale=7.0`，贪心解码（`temperature=0.0`），`max_new_tokens=32`，每任务 1000 样本，checkpoint 为 `global_step_300000`（即论文 RQ4 scaling 曲线中训练量最大的 2000 EFLOPs 节点）：

| 任务          | 准确率（%） |
|---------------|-------------|
| LAMBADA       | 50.80       |
| MMLU          | 19.30       |
| OBQA          | 23.00       |
| HellaSwag     | 10.70       |
| RACE          | 19.60       |
| SIQA          | 28.90       |
| SQuAD         | 30.90       |
| Story Cloze   | 30.77       |
| **Tasks Average** | **26.75** |

> **关于开源模型与准确率说明：**
> 当前开源的模型权重对应论文 RQ4 scaling 曲线中训练量最大的 **2000 EFLOPs** checkpoint。由于论文中评测使用的内部模型架构与本仓库基于 HuggingFace Transformers 重构的开源架构存在细微差异，各任务的准确率数值会有小幅波动，但整体趋势与论文报告一致。此外，本仓库测出的 **Tasks Average（26.75%）高于论文中报告的最终平均水平**。

完整评测协议见 [`eval_output/accuracy_summary.csv`](../eval_output/accuracy_summary.csv) 与 [`scripts/run_benchmark.sh`](../scripts/run_benchmark.sh)。

## 训练数据

- 大规模公开英文网页 + 图书文本，过滤、去重与质量筛选后得到。训练语料的组成与混合比例在论文中给出，**不随本仓库一起发布**。
- 本仓库只发布模型权重与推理代码，不分发训练语料本身。

## 训练流程概要

Cola DLM 采用两阶段训练，本节给出概要（训练代码不在本仓库中，开源版本仅含推理）：

1. **Stage 1：Text VAE 预训练**（论文式 2.2.1 中的 `L_VAE`）：重构 + BERT 风格 mask + 对基础先验的 KL 正则，建立稳定的"文本↔隐空间"对应关系。
2. **Stage 2：Text VAE + 分块因果 Text DiT 联合预训练**（论文式 2.2.3 中的 `L_stage2`）：用条件 Flow Matching 损失 `L_FM` 学习分块因果先验 `p_psi`，同时附带一个正则化的 autoencoding 项与一个 reference-encoder KL 用于抑制隐空间漂移。

完整目标函数见 [`docs/architecture_zh.md`](architecture_zh.md) §9。

## 伦理与社会影响

- **偏见（Bias）**：与任何在互联网文本上训练的大模型一样，Cola DLM 会反映甚至放大训练数据中的社会偏见（刻板印象、性别假设、少数群体代表不足等）。下游使用者在面向用户的产品中部署前必须进行审计。
- **有害内容与幻觉**：模型可能产生事实错误、攻击性或有害文本，尤其在对抗式 prompting 下；请勿将输出视为权威信息。
- **隐私**：训练语料未在文档级别做深度去重（仅标准哈希），稀有文本存在记忆风险。
- **误用风险**：模型可能被滥用于生成垃圾信息、谣言、钓鱼文本或针对真实人物的非同意内容。使用者必须遵守当地法律与平台条款。

## 注意事项与建议

- Cola DLM 是一个 **研究型产物**。它没有经过指令微调 / RLHF，输出通常比不上生产级聊天机器人。
- 为了获得更好的质量，请将短 prompt 封装成 QA 模板（`"Question: ... Answer:"`），并保证 token 数 ≥ `block_size`。
- 大规模部署时，请使用独立的分类器监控输出的安全 / 偏见 / 事实性，并向最终用户披露模型局限。

## 引用

完整 BibTeX（论文 + 开源软件）见 README 的 [引用](../README_zh.md#引用) 章节。

## 联系方式

请在 <https://github.com/your-org/cola-dlm/issues> 开 issue。
