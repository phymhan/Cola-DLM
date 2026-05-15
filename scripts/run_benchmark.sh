#!/usr/bin/env bash
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
# ============================================================================
# Cola Benchmark Evaluation
#
# Evaluates on 8 tasks: lambada, mmlu, obqa, hellaswag, race, siqa, squad,
# story_cloze using pre-converted HuggingFace model weights.
#
# Usage:
#   bash scripts/run_benchmark.sh
#
# Override defaults via environment variables:
#   DIT_PATH=... VAE_PATH=... NUM_GPUS=1 bash scripts/run_benchmark.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COLA_INFER_PER_SAMPLE_NOISE_SEED="${COLA_INFER_PER_SAMPLE_NOISE_SEED:-66}"

# ---------- configurable paths ----------
DIT_PATH="${DIT_PATH:-${REPO_DIR}/hf_models/cola_dlm/cola_dit}"
VAE_PATH="${VAE_PATH:-${REPO_DIR}/hf_models/cola_dlm/cola_vae}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_DIR}/hf_models/tokenizer.json}"
TASK_DATA_DIR="${TASK_DATA_DIR:-${REPO_DIR}/generate_task_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/eval_output/tasks_default}"

# ---------- inference parameters ----------
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
TIMESTEP_NUM="${TIMESTEP_NUM:-16}"
BATCH_SIZE="${BATCH_SIZE:-20}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_K="${TOP_K:-50}"
TOP_P="${TOP_P:-0.9}"
PAD_TOKEN_ID="${PAD_TOKEN_ID:-100277}"
EOS_TOKEN_ID="${EOS_TOKEN_ID:-100257}"
IM_END_TOKEN_ID="${IM_END_TOKEN_ID:-100265}"

# ---------- multi-GPU data-parallel ----------
NUM_GPUS="${NUM_GPUS:-8}"

# ---------- task list ----------
TASKS_DEFAULT=(
    "lambada"
    "obqa"
    "hellaswag"
    "mmlu"
    "race"
    "siqa"
    "squad"
    "story_cloze"
)
if [ -n "${TASKS:-}" ]; then
    # shellcheck disable=SC2206
    TASKS=(${TASKS})
else
    TASKS=("${TASKS_DEFAULT[@]}")
fi

# ---------- pre-flight checks ----------
if [ ! -d "$DIT_PATH" ]; then
    echo "ERROR: DiT model directory not found: $DIT_PATH"
    exit 1
fi
if [ ! -d "$VAE_PATH" ]; then
    echo "ERROR: VAE model directory not found: $VAE_PATH"
    exit 1
fi
if [ ! -f "$TOKENIZER_PATH" ]; then
    echo "ERROR: Tokenizer not found: $TOKENIZER_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================="
echo " Cola Benchmark Evaluation"
echo "============================================="
echo "  DiT model:       $DIT_PATH"
echo "  VAE model:       $VAE_PATH"
echo "  Tokenizer:       $TOKENIZER_PATH"
echo "  Task data:       $TASK_DATA_DIR"
echo "  Output:          $OUTPUT_DIR"
echo "  Guidance scale:  $GUIDANCE_SCALE"
echo "  Timestep num:    $TIMESTEP_NUM"
echo "  Batch size:      $BATCH_SIZE"
echo "  Max samples:     $MAX_SAMPLES"
echo "  Max new tokens:  $MAX_NEW_TOKENS"
echo "  Temperature:     $TEMPERATURE"
echo "  Num GPUs:        $NUM_GPUS"
echo "  Tasks:           ${TASKS[*]}"
echo "============================================="
echo ""

for TASK in "${TASKS[@]}"; do
    JSONL_PATH="${TASK_DATA_DIR}/${TASK}.jsonl"

    if [ ! -f "$JSONL_PATH" ]; then
        echo "[SKIP] Task data not found: $JSONL_PATH"
        continue
    fi

    echo "----------------------------------------------"
    echo " Task: ${TASK}  (${NUM_GPUS} GPUs)"
    echo " Data: ${JSONL_PATH}"
    echo "----------------------------------------------"

    if [ "$NUM_GPUS" -gt 1 ]; then
        PIDS=()
        FAIL=0
        for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
            CUDA_VISIBLE_DEVICES=$GPU_ID python -m cola_dlm.inference \
                --dit_path "$DIT_PATH" \
                --vae_path "$VAE_PATH" \
                --tokenizer_path "$TOKENIZER_PATH" \
                --input_jsonl "$JSONL_PATH" \
                --output_dir "$OUTPUT_DIR" \
                --task_name "$TASK" \
                --batch_size "$BATCH_SIZE" \
                --max_samples "$MAX_SAMPLES" \
                --max_new_tokens "$MAX_NEW_TOKENS" \
                --timestep_num "$TIMESTEP_NUM" \
                --guidance_scale "$GUIDANCE_SCALE" \
                --temperature "$TEMPERATURE" \
                --top_k "$TOP_K" \
                --top_p "$TOP_P" \
                --pad_token_id "$PAD_TOKEN_ID" \
                --eos_token_id "$EOS_TOKEN_ID" \
                --im_end_token_id "$IM_END_TOKEN_ID" \
                --rank "$GPU_ID" \
                --world_size "$NUM_GPUS" &
            PIDS+=($!)
        done

        for PID in "${PIDS[@]}"; do
            wait "$PID" || FAIL=1
        done

        if [ "$FAIL" -ne 0 ]; then
            echo "[ERROR] Some GPU processes failed for task ${TASK}"
            exit 1
        fi

        # Merge per-rank shard files into single output
        > "${OUTPUT_DIR}/${TASK}.jsonl"
        for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
            SHARD_FILE="${OUTPUT_DIR}/${TASK}_rank${GPU_ID}.jsonl"
            if [ -f "$SHARD_FILE" ]; then
                cat "$SHARD_FILE" >> "${OUTPUT_DIR}/${TASK}.jsonl"
                rm -f "$SHARD_FILE"
            fi
        done
    else
        python -m cola_dlm.inference \
            --dit_path "$DIT_PATH" \
            --vae_path "$VAE_PATH" \
            --tokenizer_path "$TOKENIZER_PATH" \
            --input_jsonl "$JSONL_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --task_name "$TASK" \
            --batch_size "$BATCH_SIZE" \
            --max_samples "$MAX_SAMPLES" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --timestep_num "$TIMESTEP_NUM" \
            --guidance_scale "$GUIDANCE_SCALE" \
            --temperature "$TEMPERATURE" \
            --top_k "$TOP_K" \
            --top_p "$TOP_P" \
            --pad_token_id "$PAD_TOKEN_ID" \
            --eos_token_id "$EOS_TOKEN_ID" \
            --im_end_token_id "$IM_END_TOKEN_ID"
    fi

    echo "[DONE] ${TASK} -> ${OUTPUT_DIR}/${TASK}.jsonl"
    echo ""
done

echo "============================================="
echo " All benchmarks complete!"
echo " Results saved to: $OUTPUT_DIR"
echo "============================================="
ls -lh "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "(no output files found)"
