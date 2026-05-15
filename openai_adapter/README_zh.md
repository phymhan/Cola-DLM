# OpenAI 兼容 HTTP Adapter

[English](README.md) · [中文](README_zh.md)

本目录提供一个轻量 HTTP 服务，用 OpenAI 兼容的 Chat Completions 接口暴露 Cola DLM：

```text
POST /v1/chat/completions
```

Adapter 会把传入的 chat messages 转成 Cola DLM 已有推理流程所需的 prompt，并把生成结果返回到：

```text
choices[0].message.content
```

## 安装

在仓库根目录执行：

```bash
pip install -e .
pip install -r openai_adapter/requirements.txt
```

默认模型文件布局如下：

```text
hf_models/
├── cola_dlm/
│   ├── cola_dit/
│   └── cola_vae/
└── tokenizer.json
```

如果模型文件放在其他位置，可以通过环境变量指定路径。

## 启动服务

```bash
export COLA_DIT_PATH=hf_models/cola_dlm/cola_dit
export COLA_VAE_PATH=hf_models/cola_dlm/cola_vae
export COLA_TOKENIZER_PATH=hf_models/tokenizer.json
export COLA_MODEL_NAME=cola-dlm
export COLA_API_KEY=change-me

uvicorn openai_adapter.server:app --host 0.0.0.0 --port 8000
```

也可以运行：

```bash
python -m openai_adapter.server
```

生产部署时，建议每张 GPU 启动一个 worker，除非你明确希望每个 worker 都各自加载一份模型。

## 请求示例

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me" \
  -d '{
    "model": "cola-dlm",
    "messages": [
      {
        "role": "user",
        "content": "Question: What is the capital of France? Answer:"
      }
    ],
    "temperature": 0,
    "max_tokens": 32,
    "stream": false
  }'
```

返回格式示例：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1778823653,
  "model": "cola-dlm",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Paris"
      },
      "finish_reason": "stop"
    }
  ]
}
```

## 支持的接口

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

当前 adapter 只支持非流式生成。请求中如果设置 `stream: true`，服务会返回 OpenAI 风格的错误响应。

## 配置项

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `COLA_DIT_PATH` | `hf_models/cola_dlm/cola_dit` | Cola DiT checkpoint 目录。 |
| `COLA_VAE_PATH` | `hf_models/cola_dlm/cola_vae` | Cola VAE checkpoint 目录。 |
| `COLA_TOKENIZER_PATH` | `hf_models/tokenizer.json` | `tokenizer.json` 路径。 |
| `COLA_MODEL_NAME` | `cola-dlm` | HTTP API 暴露的模型 id。 |
| `COLA_API_KEY` | 未设置 | 可选 Bearer token；未设置时不校验鉴权。 |
| `COLA_DEVICE` | `auto` | 可设置为 `auto`、`cuda`、`cuda:0` 或 `cpu`。 |
| `COLA_DEFAULT_MAX_NEW_TOKENS` | `32` | 请求未传 `max_tokens` 时使用的默认生成长度。 |
| `COLA_MAX_NEW_TOKENS_LIMIT` | `4096` | 请求 `max_tokens` 的上限。 |
| `COLA_TIMESTEP_NUM` | `16` | Diffusion 积分步数。 |
| `COLA_GUIDANCE_SCALE` | `7.0` | Classifier-free guidance scale。 |
| `COLA_TOP_K` | `50` | 默认 top-k 采样参数。 |
| `COLA_TOP_P` | `0.9` | 默认 top-p 采样参数。 |
| `COLA_REPETITION_PENALTY` | `1.1` | 采样时使用的 repetition penalty。 |
| `COLA_PAD_TOKEN_ID` | `100277` | Pad token id。 |
| `COLA_EOS_TOKEN_ID` | `100257` | EOS token id；设为空可禁用。 |
| `COLA_IM_END_TOKEN_ID` | `100265` | Chat end token id；设为空可禁用。 |

## 说明

- 如果请求只包含一条 user message，adapter 会直接把这条内容作为 prompt。
- 如果请求包含多轮 messages，adapter 会把它们格式化成带角色前缀的简单 transcript，并以 `Assistant:` 结尾。
- 推理过程会使用可变 KV cache，所以同一个服务进程内会串行执行生成请求。
- 如果需要 HTTPS、请求限流或额外鉴权，可以把服务放在常用的反向代理或负载均衡后面。

