#!/usr/bin/env bash
# Run GGU 2D/3D quality evaluation with sensible defaults.
#
# Required env: CSV_PATH (path to GGU_<Model>_results.csv from inference)
# Optional env: MODEL_PATH, DATA_DIR, OUTPUT, SUBTASKS, GPU_IDS,
#               BATCH_SIZE, MAX_NEW_TOKENS, MODEL_NAME
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-32B-Instruct}"
CSV_PATH="${CSV_PATH:?CSV_PATH env var is required}"
DATA_DIR="${DATA_DIR:-../data/Gen_Guided_Und}"
OUTPUT="${OUTPUT:-evaluate_ggu_quality_results.json}"
SUBTASKS="${SUBTASKS:-2D_Spatial,3D_Spatial}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"

cd "$(dirname "$0")"

NUM_GPUS=$(echo "$GPU_IDS" | awk -F, '{print NF}')

CMD=(python evaluate_ggu_quality.py
  --model-path "$MODEL_PATH"
  --csv-path "$CSV_PATH"
  --data-dir "$DATA_DIR"
  --output "$OUTPUT"
  --subtasks "$SUBTASKS"
  --gpu-ids "$GPU_IDS"
  --num-gpus "$NUM_GPUS"
  --batch-size "$BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS")

if [[ -n "${MODEL_NAME:-}" ]]; then
  CMD+=(--model-name "$MODEL_NAME")
fi

echo "[run] ${CMD[*]}"
exec "${CMD[@]}"
