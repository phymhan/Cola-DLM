# OpenAI-Compatible HTTP Adapter

[English](README.md) · [中文](README_zh.md)

This folder provides a lightweight HTTP service that exposes Cola DLM through an OpenAI-compatible Chat Completions endpoint:

```text
POST /v1/chat/completions
```

The adapter converts incoming chat messages into the existing Cola DLM inference pipeline and returns the generated answer in:

```text
choices[0].message.content
```

## Install

From the repository root:

```bash
pip install -e .
pip install -r openai_adapter/requirements.txt
```

Prepare model files in the default layout:

```text
hf_models/
├── cola_dlm/
│   ├── cola_dit/
│   └── cola_vae/
└── tokenizer.json
```

Or point the service to custom paths with environment variables.

## Start The Service

```bash
export COLA_DIT_PATH=hf_models/cola_dlm/cola_dit
export COLA_VAE_PATH=hf_models/cola_dlm/cola_vae
export COLA_TOKENIZER_PATH=hf_models/tokenizer.json
export COLA_MODEL_NAME=cola-dlm
export COLA_API_KEY=change-me

uvicorn openai_adapter.server:app --host 0.0.0.0 --port 8000
```

You can also run:

```bash
python -m openai_adapter.server
```

For production, run a single worker per GPU unless you intentionally want each worker to load its own copy of the model.

## Request Example

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

Response shape:

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

## Supported Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

The adapter currently supports non-streaming completions only. Requests with `stream: true` return an OpenAI-shaped error response.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `COLA_DIT_PATH` | `hf_models/cola_dlm/cola_dit` | Path to the Cola DiT checkpoint directory. |
| `COLA_VAE_PATH` | `hf_models/cola_dlm/cola_vae` | Path to the Cola VAE checkpoint directory. |
| `COLA_TOKENIZER_PATH` | `hf_models/tokenizer.json` | Path to `tokenizer.json`. |
| `COLA_MODEL_NAME` | `cola-dlm` | Model id exposed by the HTTP API. |
| `COLA_API_KEY` | unset | Optional bearer token. If unset, requests are accepted without auth. |
| `COLA_DEVICE` | `auto` | `auto`, `cuda`, `cuda:0`, or `cpu`. |
| `COLA_DEFAULT_MAX_NEW_TOKENS` | `32` | Used when a request omits `max_tokens`. |
| `COLA_MAX_NEW_TOKENS_LIMIT` | `4096` | Upper bound for request `max_tokens`. |
| `COLA_TIMESTEP_NUM` | `16` | Diffusion integration steps. |
| `COLA_GUIDANCE_SCALE` | `7.0` | Classifier-free guidance scale. |
| `COLA_TOP_K` | `50` | Default top-k sampling value. |
| `COLA_TOP_P` | `0.9` | Default top-p sampling value. |
| `COLA_REPETITION_PENALTY` | `1.1` | Repetition penalty used by the sampler. |
| `COLA_PAD_TOKEN_ID` | `100277` | Pad token id. |
| `COLA_EOS_TOKEN_ID` | `100257` | EOS token id; set empty to disable. |
| `COLA_IM_END_TOKEN_ID` | `100265` | Chat end token id; set empty to disable. |

## Notes

- If the request contains a single user message, the adapter forwards that content directly as the prompt.
- If the request contains multiple messages, the adapter formats them as a simple role-tagged transcript ending with `Assistant:`.
- The model pipeline uses mutable KV caches during generation, so the adapter serializes generation calls inside one server process.
- Put the service behind your usual reverse proxy or load balancer if you need HTTPS, request limits, or additional authentication.

