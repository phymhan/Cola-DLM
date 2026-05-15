# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenAI-compatible HTTP adapter for Cola DLM.

The service exposes ``POST /v1/chat/completions`` and maps OpenAI-style
chat messages onto the repository's existing ``generate_task_repaint_inference``
function. The returned text is placed in ``choices[0].message.content``.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from cola_dlm import ColaDiTModel, ColaTextVAEModel, generate_task_repaint_inference


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_optional_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


@dataclass(frozen=True)
class AdapterSettings:
    dit_path: str
    vae_path: str
    tokenizer_path: str
    model_name: str = "cola-dlm"
    api_key: Optional[str] = None
    device: str = "auto"
    default_max_new_tokens: int = 32
    max_new_tokens_limit: int = 4096
    timestep_num: int = 16
    guidance_scale: float = 7.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    pad_token_id: int = 100277
    eos_token_id: Optional[int] = 100257
    im_end_token_id: Optional[int] = 100265

    @classmethod
    def from_env(cls) -> "AdapterSettings":
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return cls(
            dit_path=os.environ.get("COLA_DIT_PATH", os.path.join(repo_root, "hf_models/cola_dlm/cola_dit")),
            vae_path=os.environ.get("COLA_VAE_PATH", os.path.join(repo_root, "hf_models/cola_dlm/cola_vae")),
            tokenizer_path=os.environ.get("COLA_TOKENIZER_PATH", os.path.join(repo_root, "hf_models/tokenizer.json")),
            model_name=os.environ.get("COLA_MODEL_NAME", "cola-dlm"),
            api_key=os.environ.get("COLA_API_KEY") or None,
            device=os.environ.get("COLA_DEVICE", "auto"),
            default_max_new_tokens=_env_int("COLA_DEFAULT_MAX_NEW_TOKENS", 32),
            max_new_tokens_limit=_env_int("COLA_MAX_NEW_TOKENS_LIMIT", 4096),
            timestep_num=_env_int("COLA_TIMESTEP_NUM", 16),
            guidance_scale=_env_float("COLA_GUIDANCE_SCALE", 7.0),
            top_k=_env_int("COLA_TOP_K", 50),
            top_p=_env_float("COLA_TOP_P", 0.9),
            repetition_penalty=_env_float("COLA_REPETITION_PENALTY", 1.1),
            pad_token_id=_env_int("COLA_PAD_TOKEN_ID", 100277),
            eos_token_id=_env_optional_int("COLA_EOS_TOKEN_ID", 100257),
            im_end_token_id=_env_optional_int("COLA_IM_END_TOKEN_ID", 100265),
        )


class ChatMessage(BaseModel):
    role: str
    content: Any = ""

    class Config:
        extra = "allow"


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="cola-dlm")
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: Optional[int] = 4096
    stream: bool = False
    top_p: Optional[float] = None
    n: int = 1
    top_k: Optional[int] = None

    class Config:
        extra = "allow"


class ColaOpenAIAdapter:
    """Small synchronous wrapper around the Cola DLM inference pipeline."""

    def __init__(self, settings: AdapterSettings):
        self.settings = settings
        self.device = self._resolve_device(settings.device)
        self._lock = threading.Lock()

        print(f"[openai-adapter] loading DiT from {settings.dit_path}")
        self.dit = ColaDiTModel.from_pretrained(settings.dit_path).to(self.device)
        self.dit.eval()

        print(f"[openai-adapter] loading VAE from {settings.vae_path}")
        self.vae = ColaTextVAEModel.from_pretrained(settings.vae_path).to(self.device)
        self.vae.eval()

        print(f"[openai-adapter] loading tokenizer from {settings.tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(settings.tokenizer_path)
        print(f"[openai-adapter] ready on device={self.device}")

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested COLA_DEVICE={device_name!r}, but CUDA is not available.")
        return torch.device(device_name)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
    ) -> str:
        with self._lock:
            results = generate_task_repaint_inference(
                dit=self.dit,
                vae=self.vae,
                tokenizer=self.tokenizer,
                prompts=[{"id": 0, "question": prompt}],
                task_name="openai",
                device=self.device,
                T=1000.0,
                timestep_num=self.settings.timestep_num,
                guidance_scale=self.settings.guidance_scale,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=self.settings.top_k if top_k is None else top_k,
                top_p=self.settings.top_p if top_p is None else top_p,
                repetition_penalty=self.settings.repetition_penalty,
                pad_token_id=self.settings.pad_token_id,
                eos_token_id=self.settings.eos_token_id,
                im_end_token_id=self.settings.im_end_token_id,
            )
        return str(results[0].get("generate", ""))


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
        return "\n".join(part for part in parts if part)
    return str(content)


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    if not messages:
        raise HTTPException(status_code=400, detail="messages must contain at least one item")

    user_messages = [m for m in messages if m.role == "user"]
    if len(messages) == 1 and user_messages:
        return _message_content_to_text(user_messages[0].content)

    lines: list[str] = []
    for message in messages:
        text = _message_content_to_text(message.content).strip()
        if not text:
            continue
        role = message.role.lower()
        if role == "system":
            lines.append(f"System: {text}")
        elif role == "assistant":
            lines.append(f"Assistant: {text}")
        elif role == "user":
            lines.append(f"User: {text}")
        else:
            lines.append(f"{message.role}: {text}")

    if not lines:
        raise HTTPException(status_code=400, detail="messages content must not be empty")
    if messages[-1].role != "assistant":
        lines.append("Assistant:")
    return "\n".join(lines)


def _openai_error(status_code: int, message: str, error_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )


def _check_api_key(expected_key: Optional[str], authorization: Optional[str]) -> None:
    if not expected_key:
        return
    if authorization != f"Bearer {expected_key}":
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


settings = AdapterSettings.from_env()
app = FastAPI(title="Cola DLM OpenAI-Compatible Adapter", version="0.1.0")


@app.on_event("startup")
def load_model() -> None:
    app.state.runner = ColaOpenAIAdapter(settings)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": settings.model_name}


@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    _check_api_key(settings.api_key, authorization)
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_name,
                "object": "model",
                "created": created,
                "owned_by": "cola-dlm",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    _check_api_key(settings.api_key, authorization)

    if request.stream:
        return _openai_error(400, "stream=true is not supported by this adapter yet")
    if request.n != 1:
        return _openai_error(400, "Only n=1 is supported")
    if request.model and request.model != settings.model_name:
        return _openai_error(404, f"Unknown model {request.model!r}; expected {settings.model_name!r}", "not_found_error")

    prompt = _messages_to_prompt(request.messages)
    max_new_tokens = request.max_tokens or settings.default_max_new_tokens
    max_new_tokens = max(1, min(int(max_new_tokens), settings.max_new_tokens_limit))

    try:
        content = await run_in_threadpool(
            app.state.runner.generate,
            prompt,
            max_new_tokens,
            float(request.temperature),
            request.top_p,
            request.top_k,
        )
    except Exception as exc:  # pragma: no cover - keeps HTTP errors OpenAI-shaped at runtime.
        return _openai_error(500, f"Generation failed: {exc}", "server_error")

    created = int(time.time())
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": settings.model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
    }
    return JSONResponse(content=response)


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("openai_adapter.server:app", host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
