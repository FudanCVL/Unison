#!/usr/bin/env bash
# Run the Unison evaluation pipeline.
#
# Single entry point: drives evaluate_unison.py over one or more models, scoring
# whichever subset of the four tasks you choose. Configure everything via env vars.
#
# Pick models and tasks:
#   MODELS   - comma-separated model name(s); each maps to <RESULT_DIR>/<MODEL>.
#              (default: UniWorld-V1). MODEL_NAME is accepted as a single-model alias.
#   TASKS    - comma-separated subset of IC,UGG,GGU,ME to score (default: IC,UGG,GGU,ME).
#
# Judge backend:
#   JUDGE_BACKEND      - 'local' (default, trained Qwen3-VL) or 'api' (DashScope Qwen3-VL-Plus)
#   GPU_IDS            - GPUs for the local judge, e.g. '0-7' or '0,1,2' (default: 0-1)
#   LOCAL_JUDGE_MODEL  - path to the local judge / Unison-Judge weights (default: ./judge_model)
#   LOCAL_JUDGE_IO_LOG - local judge per-call I/O log path (default: judge_io_local.csv)
#   API_KEY            - DashScope key (or DASHSCOPE_API_KEY); required only for JUDGE_BACKEND=api
#
# Paths / misc:
#   DATA_DIR           - benchmark data dir / Unison-data (default: ../data)
#   INFERENCE_BASE_DIR - Inference_Pipeline root (default: ../Inference_Pipeline)
#   RESULT_DIR         - result dir for the model(s) (default: $INFERENCE_BASE_DIR/result/<MODEL>)
#   OUTPUT             - output JSON path (default: eval_<MODEL>.json)
#   MAX_WORKERS        - parallel judge threads (default: 8)
#   IMGEDIT_BBOX_MODE  - ImgEdit bbox handling: 'full' (default) embeds the GT region
#                        scope; 'noscope' drops it so ONLY ME/misalignment uses bbox
#
# Examples:
#   MODELS=UniWorld-V1 ./run_evaluate_unison.sh                       # one model, all tasks, local judge
#   MODELS=BAGEL-7B-MoT TASKS=IC,GGU ./run_evaluate_unison.sh          # choose tasks
#   MODELS="BAGEL-7B-MoT,Janus-Pro-7B" ./run_evaluate_unison.sh        # several models
#   JUDGE_BACKEND=api API_KEY=sk-... MODELS=UniWorld-V1 ./run_evaluate_unison.sh   # API judge

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load local secrets if present (not committed)
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a; source "${SCRIPT_DIR}/.env"; set +a
fi

MODELS="${MODELS:-${MODEL_NAME:-UniWorld-V1}}"
TASKS="${TASKS:-IC,UGG,GGU,ME}"

JUDGE_BACKEND="${JUDGE_BACKEND:-local}"
GPU_IDS="${GPU_IDS:-0-1}"
LOCAL_JUDGE_MODEL="${LOCAL_JUDGE_MODEL:-${SCRIPT_DIR}/judge_model}"
LOCAL_JUDGE_IO_LOG="${LOCAL_JUDGE_IO_LOG:-${SCRIPT_DIR}/judge_io_local.csv}"

DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/../data}"
INFERENCE_BASE_DIR="${INFERENCE_BASE_DIR:-${SCRIPT_DIR}/../Inference_Pipeline}"
MAX_WORKERS="${MAX_WORKERS:-8}"
IMGEDIT_BBOX_MODE="${IMGEDIT_BBOX_MODE:-full}"

API_KEY="${API_KEY:-${DASHSCOPE_API_KEY:-}}"
if [[ "$JUDGE_BACKEND" == "api" && -z "$API_KEY" ]]; then
  echo "ERROR: API_KEY or DASHSCOPE_API_KEY is required for JUDGE_BACKEND=api" >&2
  exit 1
fi

# api backend: write the judge_io.csv capture set (unchanged behavior).
# local backend: never write judge_io.csv; the local I/O log goes to python via --judge-io-log.
if [[ "$JUDGE_BACKEND" == "api" ]]; then
  export JUDGE_IO_LOG="${JUDGE_IO_LOG-${SCRIPT_DIR}/judge_io.csv}"
  echo "Judge I/O log (api): ${JUDGE_IO_LOG:-<disabled>}"
else
  unset JUDGE_IO_LOG || true
  echo "Judge I/O log (local): ${LOCAL_JUDGE_IO_LOG:-<disabled>}"
fi

cd "$SCRIPT_DIR"

run_model() {
  local model="$1"
  local tasks="$2"
  local result_dir="${RESULT_DIR:-${INFERENCE_BASE_DIR}/result/${model}}"
  local output="${OUTPUT:-${SCRIPT_DIR}/eval_${model}.json}"

  echo "========================================"
  echo "Evaluating: $model  tasks=$tasks  backend=$JUDGE_BACKEND"
  echo "========================================"

  local backend_args=(--judge-backend "$JUDGE_BACKEND")
  if [[ "$JUDGE_BACKEND" == "local" ]]; then
    backend_args+=(--local-model-path "$LOCAL_JUDGE_MODEL" --gpu-ids "$GPU_IDS" \
                   --judge-io-log "$LOCAL_JUDGE_IO_LOG")
  else
    backend_args+=(--api-key "$API_KEY")
  fi

  python evaluate_unison.py \
    --result-dir         "$result_dir" \
    --data-dir           "$DATA_DIR" \
    --inference-base-dir "$INFERENCE_BASE_DIR" \
    --tasks              "$tasks" \
    --max-workers        "$MAX_WORKERS" \
    --output             "$output" \
    --imgedit-bbox-mode  "$IMGEDIT_BBOX_MODE" \
    "${backend_args[@]}"
  echo "Done: $model -> $output"
}

IFS=',' read -ra _models <<< "$MODELS"
for m in "${_models[@]}"; do
  m="$(echo "$m" | xargs)"   # trim whitespace
  [[ -z "$m" ]] && continue
  run_model "$m" "$TASKS"
done

echo "========================================"
echo "All evaluations complete."
echo "Aggregating results..."
python "$SCRIPT_DIR/aggregate_results.py"
