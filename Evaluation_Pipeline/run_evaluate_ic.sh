SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="$SCRIPT_DIR"

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-32B-Instruct"}
CSV_PATH=${CSV_PATH:-"../Inference_Pipeline/result/BAGEL-7B-MoT/IC/IC_BAGEL-7B-MoT_results.csv"}
OUTPUT=${OUTPUT:-"evaluate_ic_results.json"}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
TEMPERATURE=${TEMPERATURE:-0.0}
DEVICE=${DEVICE:-"auto"}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
MAX_ITEMS=${MAX_ITEMS:-""}
IMAGES_BASE_DIR=${IMAGES_BASE_DIR:-""}

mkdir -p "$(dirname "$OUTPUT")"

cd "$PROJECT_ROOT"

EXTRA_ARGS=()
if [[ -n "$MAX_ITEMS" ]]; then
  EXTRA_ARGS+=(--max-items "$MAX_ITEMS")
fi
if [[ -n "$GPU_IDS" ]]; then
  EXTRA_ARGS+=(--gpu-ids "$GPU_IDS")
fi
if [[ -n "$IMAGES_BASE_DIR" ]]; then
  EXTRA_ARGS+=(--images-base-dir "$IMAGES_BASE_DIR")
fi


# Compute the number of GPUs
if [[ -n "$GPU_IDS" ]]; then
  NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
else
  NUM_GPUS=4
fi


while true; do
  python "$PROJECT_ROOT/evaluate_ic.py" \
    --model-path "$MODEL_PATH" \
    --csv-path "$CSV_PATH" \
    --output "$OUTPUT" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --num-gpus "$NUM_GPUS" \
    --device "$DEVICE" \
    "${EXTRA_ARGS[@]}" \
    "$@"

  sleep 10
done
