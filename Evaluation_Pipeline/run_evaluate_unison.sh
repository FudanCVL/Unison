#!/usr/bin/env bash
# Run Unison evaluation pipeline.
#
# Without arguments: evaluates all models below sequentially.
# With MODEL_NAME set: evaluates that single model only.
#
# Required env vars:
#   API_KEY     - DashScope API key (or set DASHSCOPE_API_KEY)
#
# Optional:
#   MODEL_NAME  - single model to evaluate (skips the all-models loop)
#   RESULT_DIR  - path to model result dir (default: ../Inference_Pipeline/result/$MODEL_NAME)
#   DATA_DIR    - path to benchmark data dir (default: ../data)
#   INFERENCE_BASE_DIR - Inference_Pipeline root (default: ../Inference_Pipeline)
#   TASKS       - comma-separated tasks (default: per-model list below)
#   MAX_WORKERS - parallel judge threads (default: 8)
#   OUTPUT      - output JSON path (default: eval_${MODEL_NAME}.json)

set -euo pipefail

API_KEY="${API_KEY:-${DASHSCOPE_API_KEY:-}}"
if [[ -z "$API_KEY" ]]; then
  echo "ERROR: API_KEY or DASHSCOPE_API_KEY is required" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/../data}"
INFERENCE_BASE_DIR="${INFERENCE_BASE_DIR:-${SCRIPT_DIR}/../Inference_Pipeline}"
MAX_WORKERS="${MAX_WORKERS:-8}"

cd "$SCRIPT_DIR"

run_model() {
  local model="$1"
  local tasks="$2"
  local result_dir="${RESULT_DIR:-${SCRIPT_DIR}/../Inference_Pipeline/result/${model}}"
  local output="${OUTPUT:-${SCRIPT_DIR}/eval_${model}.json}"

  echo "========================================"
  echo "Evaluating: $model  tasks=$tasks"
  echo "========================================"
  python evaluate_unison.py \
    --result-dir        "$result_dir" \
    --data-dir          "$DATA_DIR" \
    --inference-base-dir "$INFERENCE_BASE_DIR" \
    --api-key           "$API_KEY" \
    --tasks             "$tasks" \
    --max-workers       "$MAX_WORKERS" \
    --output            "$output"
  echo "Done: $model -> $output"
}

# Single-model mode (MODEL_NAME is set)
if [[ -n "${MODEL_NAME:-}" ]]; then
  TASKS="${TASKS:-IC,UGG,GGU,ME}"
  run_model "$MODEL_NAME" "$TASKS"
  exit 0
fi

# All-models mode
# run_model "Janus-Pro-7B"  "IC,GGU,UGG"
# run_model "SEED-X-17B"    "IC,GGU,UGG,ME"
# run_model "TokenFlow"     "IC,GGU,UGG"

run_model "UniWorld-V1"   "IC,GGU,UGG,ME"
# run_model "OmniGen2"      "IC,GGU,UGG,ME"
# run_model "ILLUME-plus-7B" "IC,GGU,UGG,ME"
# run_model "BAGEL-7B-MoT" "IC,GGU,UGG,ME"
# run_model "Show-o2-7B" "IC,GGU,UGG"

# run_model "BAGEL-7B-MoT" "IC,GGU,UGG,ME"


echo "========================================"
echo "All evaluations complete."

echo "Aggregating results..."
python "$SCRIPT_DIR/aggregate_results.py"
