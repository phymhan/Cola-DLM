"""
Interactive multi-turn chat CLI for Cola-DLM.

Usage:
    python scripts/cola_chat_cli.py
    python scripts/cola_chat_cli.py --dit-path cola_sft_checkpoints/my_run/dit_step_001000
    python scripts/cola_chat_cli.py --prompt "What is the capital of France?"
"""

import argparse
import sys
import torch
from tokenizers import Tokenizer

from cola_dlm import ColaDiTModel, ColaTextVAEModel
from cola_dlm.engine import ColaEngine

parser = argparse.ArgumentParser(description="Cola-DLM Chat CLI")
parser.add_argument("--dit-path", type=str, default="hf_models/cola_dlm/cola_dit")
parser.add_argument("--vae-path", type=str, default="hf_models/cola_dlm/cola_vae")
parser.add_argument("--tokenizer-path", type=str, default="hf_models/tokenizer.json")
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument("--timestep-num", type=int, default=16)
parser.add_argument("--guidance-scale", type=float, default=7.0)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--top-p", type=float, default=0.9)
parser.add_argument("--block-size", type=int, default=None, help="Override block size for generation")
parser.add_argument("--chat-format", type=str, default="text", choices=["text", "chatml"])
parser.add_argument("--prompt", type=str, default=None, help="Single-shot prompt (non-interactive)")
parser.add_argument("--system-prompt", type=str, default="",
                    help="System prompt prepended to conversation (default: none)")
parser.add_argument("--stream-steps", type=int, default=0,
                    help="Stream intermediate x0 every N denoising steps (0=disabled)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading models...")
dit = ColaDiTModel.from_pretrained(args.dit_path).to(device).eval()
vae = ColaTextVAEModel.from_pretrained(args.vae_path).to(device).eval()
tokenizer = Tokenizer.from_file(args.tokenizer_path)

# Auto-detect SFT config from model, CLI args override
chat_format = args.chat_format or getattr(dit.config, "sft_chat_format", "text")
pad_with_stop = getattr(dit.config, "sft_pad_with_stop", False)
stop_token_id = getattr(dit.config, "sft_stop_token_id", None)

if chat_format == "text":
    STOP_TEXT = "■"
    STOP_TOKEN_ID = stop_token_id or 47774
    PAD_TOKEN_ID = 47774
else:
    STOP_TEXT = None
    STOP_TOKEN_ID = stop_token_id or 100265
    PAD_TOKEN_ID = 100265 if pad_with_stop else 100277

engine = ColaEngine(dit, vae, tokenizer)
print(f"DiT: {args.dit_path}")
print(f"VAE: {args.vae_path}")
print(f"Format: {args.chat_format}, Stop: {repr(STOP_TEXT)}")


def build_prompt_text(conversation_history):
    parts = []
    if args.system_prompt:
        parts.append(f"System: {args.system_prompt}\n")
    for role, content in conversation_history:
        if role == "user":
            parts.append(f"User: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}■\n")
    parts.append("Assistant: ")
    return "".join(parts)


def build_prompt_chatml(conversation_history):
    parts = []
    if args.system_prompt:
        parts.append(f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n")
    for role, content in conversation_history:
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def build_prompt(conversation_history):
    if args.chat_format == "text":
        return build_prompt_text(conversation_history)
    return build_prompt_chatml(conversation_history)


def generate_response(prompt_text):
    prompt_ids = tokenizer.encode(prompt_text).ids
    response = ""
    printed = 0
    stop_tokens = [s for s in [STOP_TEXT, "User:", "\nUser"] if s is not None]
    done = False
    intermediate_len = 0
    is_tty = sys.stdout.isatty()
    for block in engine.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        timestep_num=args.timestep_num,
        guidance_scale=args.guidance_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        block_size=args.block_size,
        stop_text=STOP_TEXT,
        stop_token_id=STOP_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        stream_steps=args.stream_steps,
        seed=args.seed,
    ):
        if block.is_intermediate:
            if not is_tty:
                continue
            if intermediate_len > 0:
                sys.stdout.write('\b' * intermediate_len + ' ' * intermediate_len + '\b' * intermediate_len)
            preview = block.text.replace('\n', '').replace('\r', '')
            sys.stdout.write(preview)
            sys.stdout.flush()
            intermediate_len = len(preview)
        else:
            if intermediate_len > 0 and is_tty:
                sys.stdout.write('\b' * intermediate_len + ' ' * intermediate_len + '\b' * intermediate_len)
                intermediate_len = 0
            response += block.text
            for st in stop_tokens:
                if st in response:
                    response = response[: response.index(st)]
                    done = True
                    break
            new_text = response[printed:]
            if new_text:
                print(new_text, end="", flush=True)
            printed = len(response)
            if done:
                break
    print()
    return response.strip()


if args.prompt:
    history = [("user", args.prompt)]
    prompt_text = build_prompt(history)
    print(f"\nAssistant: ", end="")
    response = generate_response(prompt_text)
    exit(0)

print("\nCola-DLM Chat")
print(f"Commands: 'clear' to reset, 'quit' to exit")
print("-" * 50)

conversation_history = []

while True:
    try:
        user_input = input("\nUser: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit"):
        print("Bye!")
        break
    if user_input.lower() == "clear":
        conversation_history = []
        print("(conversation cleared)")
        continue

    conversation_history.append(("user", user_input))
    prompt_text = build_prompt(conversation_history)

    print(f"\nAssistant: ", end="")
    response = generate_response(prompt_text)
    conversation_history.append(("assistant", response))
