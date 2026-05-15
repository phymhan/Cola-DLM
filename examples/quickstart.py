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

"""Minimal end-to-end Cola DLM inference example.

Usage:
    python examples/quickstart.py \\
        --dit_path hf_models/cola_dlm/cola_dit \\
        --vae_path hf_models/cola_dlm/cola_vae \\
        --tokenizer_path hf_models/tokenizer.json

This script loads a Cola DLM checkpoint (Text VAE + block-causal DiT
prior) and runs the paper's three-step inference algorithm on a small
batch of prompts via ``generate_task_repaint_inference``: (i) prefix
encode ``z^pre ~ q_phi(z^pre | x^pre)`` (Eq. 2.2.4), (ii) block-wise
latent prior transport ``hat z_0^(b) = Phi^psi_{0<-1}(eps^(b); z^pre,
hat z_0^(<b))`` (Eq. 2.2.5), and (iii) conditional decoding
``hat x^res ~ p_theta(x^res | z^pre, hat z_0^(1:B))`` (Eq. 2.2.6). The
generated continuations are printed back to stdout.

See the paper *Continuous Latent Diffusion Language Model*
(arXiv:2605.06548, https://arxiv.org/abs/2605.06548) for the underlying
theory.

For the full CLI with multi-GPU data-parallel sharding and benchmark
JSONL inputs, use ``python -m cola_dlm.inference`` (see
``docs/inference.md``).
"""

from __future__ import annotations

import argparse
import sys

import torch
from tokenizers import Tokenizer

from cola_dlm import (
    ColaDiTModel,
    ColaTextVAEModel,
    generate_task_repaint_inference,
)

PROMPTS: list[str] = [
    "Question: What is the capital of France? Answer:",
    "Question: Who painted the Mona Lisa? Answer:",
    "Question: What is the boiling point of water at sea level in Celsius? Answer:",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dit_path", required=True, help="Path to the converted Cola DiT directory.")
    parser.add_argument("--vae_path", required=True, help="Path to the converted Cola VAE directory.")
    parser.add_argument("--tokenizer_path", required=True, help="Path to tokenizer.json.")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--timestep_num", type=int, default=16)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--pad_token_id", type=int, default=100277)
    parser.add_argument("--eos_token_id", type=int, default=100257)
    parser.add_argument("--im_end_token_id", type=int, default=100265)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[quickstart] device: {device}")

    print(f"[quickstart] loading DiT from {args.dit_path}")
    dit = ColaDiTModel.from_pretrained(args.dit_path).to(device)
    dit.eval()

    print(f"[quickstart] loading VAE from {args.vae_path}")
    vae = ColaTextVAEModel.from_pretrained(args.vae_path).to(device)
    vae.eval()

    print(f"[quickstart] loading tokenizer from {args.tokenizer_path}")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    prompts = [{"question": p} for p in PROMPTS]

    print(f"[quickstart] generating {len(prompts)} sample(s)")
    results = generate_task_repaint_inference(
        dit=dit,
        vae=vae,
        tokenizer=tokenizer,
        prompts=prompts,
        task_name="lambada",
        device=device,
        T=1000.0,
        timestep_num=args.timestep_num,
        guidance_scale=args.guidance_scale,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        pad_token_id=args.pad_token_id,
        eos_token_id=args.eos_token_id,
        im_end_token_id=args.im_end_token_id,
    )

    print("=" * 70)
    for i, r in enumerate(results, 1):
        print(f"[{i}] prompt:    {r['prompt']}")
        print(f"    generated: {r['generate']}")
        print("-" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
