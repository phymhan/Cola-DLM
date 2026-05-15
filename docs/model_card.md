# Model Card: Cola DLM

> **English** · [中文](model_card_zh.md)

This model card follows the structure suggested by Mitchell et al. (2019) *Model Cards for Model Reporting*. It describes the reference `cola-dlm` checkpoint released alongside this repository and the paper [*Continuous Latent Diffusion Language Model*](https://arxiv.org/abs/2605.06548).

## Paper

- **Title:** Continuous Latent Diffusion Language Model
- **Authors:** Hongcan Guo, Qinyu Zhao, Yian Zhao, Shen Nie, Rui Zhu, Qiushan Guo, Feng Wang, Tao Yang, Hengshuang Zhao, Guoqiang Wei, Yan Zeng (ByteDance Seed et al.)
- **arXiv:** [arxiv.org/abs/2605.06548](https://arxiv.org/abs/2605.06548)
- **HuggingFace daily paper:** [huggingface.co/papers/2605.06548](https://huggingface.co/papers/2605.06548)
- **Project page:** [hongcanguo.github.io/Cola-DLM](https://hongcanguo.github.io/Cola-DLM/)

## Model details

- **Model name:** Cola DLM (Continuous Latent Diffusion Language Model)
- **Version:** `0.1.0`
- **Release date:** 2026
- **Method:** Hierarchical continuous latent diffusion language model — a Text VAE encoder `q_phi(z_0 | x)` + conditional decoder `p_theta(x | z_0)` paired with a block-causal Diffusion Transformer prior `p_psi(z_0)` learned by Flow Matching. Diffusion is used as **latent prior transport** (Eq. 2.1.4 of the paper), not token-level observation recovery.
- **Architecture:** Two cooperating modules
  - `ColaTextVAEModel` — ~500M parameter Text VAE realizing `q_phi` and `p_theta` (4 encoder + 4 decoder blocks, `dim=1536`, `ffn_dim=6144`, `latent_dim=16`).
  - `ColaDiTModel` — ~1.8B parameter 1-D Diffusion Transformer realizing the block-causal latent prior `p_psi` (24 layers, `txt_dim=emb_dim=2048`, 16 heads × 128 head_dim, `expand_ratio=4`).
- **Training-compute checkpoint:** This release ships the **2000 EFLOPs** entry on the paper's RQ4 scaling curve — the largest-compute checkpoint in the paper. The total of ~2B parameters is strictly matched against the autoregressive and LLaDA baselines used in the same scaling comparison.
- **License:** [Apache License 2.0](../LICENSE).
- **Framework:** PyTorch 2.1+; HuggingFace Transformers 4.40+.
- **Precision:** Master weights kept in fp32; forward passes use `torch.autocast(dtype=torch.bfloat16)` on CUDA.
- **Tokenizer:** OLMo 2 tokenizer (BPE over a 100,278-entry vocabulary); `pad_token_id=100277`, `eos_token_id=100257`, `im_end_token_id=100265`.

## Intended use

### Primary use cases

- Research on hierarchical / continuous latent text diffusion language models.
- Closed-domain question answering and zero-/few-shot benchmark evaluation (e.g. LAMBADA, MMLU, HellaSwag, OBQA, RACE, SIQA, SQuAD, Story Cloze) — exactly the 8 tasks reported in the paper's RQ4.
- Studying properties of latent-prior-transport text generation (controllability via CFG, block-wise re-painting, latent-space semantic structure, etc.).

### Out-of-scope uses

- Safety-critical decision-making (medical, legal, financial advice).
- Generation of content that violates applicable laws, platform policies, or research-ethics norms.
- Use as a drop-in chatbot without task-specific fine-tuning — the pretraining objective is hierarchical latent prior modeling, not instruction-following dialogue.

## Factors

- The model was pretrained primarily on English text; performance on other languages is untested and expected to be weak.
- Generation quality depends strongly on prompt length relative to `block_size` (typically 16): prompts shorter than one block auto-disable classifier-free guidance for their first block and may produce weaker outputs. See [`docs/inference.md`](inference.md) for the mitigation.

## Metrics

Accuracy on 8 zero-shot benchmarks, batch size 70, `timestep_num=16`, `guidance_scale=7.0`, greedy decoding (`temperature=0.0`), `max_new_tokens=32`, 1000 samples per task, checkpoint `global_step_300000` (the 2000 EFLOPs entry on the paper's RQ4 scaling curve):

| Task       | Accuracy (%) |
|------------|--------------|
| LAMBADA    | 50.80        |
| MMLU       | 19.30        |
| OBQA       | 23.00        |
| HellaSwag  | 10.70        |
| RACE       | 19.60        |
| SIQA       | 28.90        |
| SQuAD      | 30.90        |
| Story Cloze| 30.77        |
| **Tasks Average** | **26.75** |

> **Note on open-source model and accuracy:**
> The released model weights correspond to the **2000 EFLOPs** entry on the paper's RQ4 scaling curve — the largest training-compute checkpoint reported. Because the internal architecture used for evaluation in the paper differs slightly from the open-source HuggingFace Transformers-based implementation in this repository, per-task accuracy numbers may exhibit minor fluctuations, but the overall trend is consistent with the paper. Notably, the **Tasks Average (26.75%) measured here is slightly higher than the final average reported in the paper**.

See [`eval_output/accuracy_summary.csv`](../eval_output/accuracy_summary.csv) and [`scripts/run_benchmark.sh`](../scripts/run_benchmark.sh) for the exact protocol.

## Training data

- Large-scale public web and book text, filtered for English, deduplicated and quality-filtered. The corpus composition and mix ratios are documented in the paper; they are not included in this repository.
- The training corpus is **not** redistributed as part of this repo; only the model weights and inference code are released.

## Training procedure (overview)

Cola DLM is trained in two stages, summarized here for reference (training code is not in this repo; the open-source release is inference-only):

1. **Stage 1 — Text VAE pretraining** (`L_VAE` in Eq. 2.2.1): reconstruction + BERT-style masking + KL regularization to a base prior, establishing a stable text↔latent correspondence.
2. **Stage 2 — Joint Text VAE + block-causal Text DiT pretraining** (`L_stage2` in Eq. 2.2.3): conditional Flow Matching loss `L_FM` for the block-causal prior `p_psi`, plus a regularized autoencoding term and a reference-encoder KL that suppresses latent drift.

See [`docs/architecture.md`](architecture.md) §9 for the explicit objectives.

## Ethical considerations

- **Bias.** Like any large-scale language model trained on internet text, Cola DLM will reflect and can amplify societal biases present in the training data (stereotypes, gendered assumptions, under-representation of minority dialects, etc.). Downstream users must audit outputs before deploying in user-facing products.
- **Toxicity & hallucinations.** The model may produce factually incorrect, offensive, or harmful text, especially under adversarial prompting. Do not treat its outputs as authoritative.
- **Privacy.** Training data was not de-duplicated at the document level beyond standard hashing; memorisation of rare text is possible.
- **Misuse.** The model could be misused for generating spam, misinformation, phishing content, or non-consensual text about real people. Users must comply with local laws and platform terms of service.

## Caveats & recommendations

- Cola DLM is a **research artifact**. It ships without an instruction-tuning / RLHF stage, and its outputs will often underperform production chatbots.
- For best quality, wrap short prompts in QA-style templates (`"Question: ... Answer:"`) and use at least `block_size` tokens.
- When deploying at scale, monitor output for safety / bias / factuality using independent classifiers, and surface disclaimers to end users.

## Citation

See the [README](../README.md#citation) for the BibTeX entries (paper + open-source software).

## Contact

Open an issue at <https://github.com/your-org/cola-dlm/issues>.
