#!/bin/bash
# ============================================
# Configuration variables
# ============================================

# GPU IDs (required, comma-separated)
GPUS="0,1,2,3,4,5,6,7"

# Task list (optional; a single task as a string, or multiple comma-separated, e.g. "IC" or "IC,UGG,GGU,ME")


# Model list (optional; multiple models comma-separated, model name alone is fine, e.g. "BAGEL-7B-MoT,UniWorld-V1")
# See the *.json files under config/ for available models
# Each model runs in the conda_env specified in its config (isolated from each other)
# If empty, the default model list in infer.py is used (runs only in the base env)

# All four tasks supported
MODELS="BAGEL-7B-MoT,OmniGen2,UniWorld-V1,SEED-X-17B,ILLUME-plus-7B,ILLUME-plus-3B,Show-o2-7B,Show-o-1.3B,Show-o2-1.5B,Janus-Pro-7B,Janus-Pro-1B,TokenFlow,D-DiT"

TASKS="IC,UGG,GGU,ME"

# Test mode (optional; set to "true" or "1" to enable test mode, processing only 2 items per task)
TEST_MODE=""

# Number of items to sample per task (0=disabled; IC/UGG/ME take the first N, GGU takes the first N per sub-task)
# TEST_N=8

# Data directory (path relative to this script, or an absolute path)
DATA_DIR="../data"

# Result directory
RESULT_DIR="result"

# conda install path (used to source conda.sh)
CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"

# ============================================
# Argument validation
# ============================================

if [ -z "$GPUS" ]; then
    echo "Error: GPUS cannot be empty!"
    echo "Set the GPUS variable in the script, e.g. GPUS=\"0,1,2,3\""
    exit 1
fi

if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "Error: conda.sh not found: $CONDA_BASE/etc/profile.d/conda.sh"
    echo "Specify the conda install path via the CONDA_BASE environment variable"
    exit 1
fi

# ============================================
# Change to the script's directory
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================
# Log directory
# ============================================

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/run_${RUN_TIMESTAMP}.log"

# Send all subsequent output to both the terminal and the main log
exec > >(tee -a "$MAIN_LOG") 2>&1
echo "Log saved to: $MAIN_LOG"

# Models are already cached locally; force offline mode so from_pretrained reads only local files and makes no network requests.
# To download a new model for the first time, temporarily comment out these two lines.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Keep the mirror config for temporary downloads (comment out the two lines above first)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ============================================
# Normalize MODELS (model name alone is allowed)
# ============================================

NORMALIZED_MODELS=""
if [ -n "$MODELS" ]; then
    IFS=',' read -ra MODEL_ITEMS <<< "$MODELS"
    for item in "${MODEL_ITEMS[@]}"; do
        item="${item// /}"
        [ -z "$item" ] && continue
        if [[ "$item" == */* ]]; then
            normalized="$item"
        else
            base="${item%.json}"
            normalized="config/${base}.json"
        fi
        if [ ! -f "$normalized" ]; then
            echo "Error: model config file not found: $normalized"
            exit 1
        fi
        if [ -z "$NORMALIZED_MODELS" ]; then
            NORMALIZED_MODELS="$normalized"
        else
            NORMALIZED_MODELS="$NORMALIZED_MODELS,$normalized"
        fi
    done
fi

# ============================================
# Common arguments (everything except --models)
# ============================================

COMMON_ARGS="--gpus $GPUS"
[ -n "$TASKS" ] && COMMON_ARGS="$COMMON_ARGS --tasks $TASKS"
if [ "$TEST_MODE" = "true" ] || [ "$TEST_MODE" = "1" ]; then
    COMMON_ARGS="$COMMON_ARGS --test"
fi
[ -n "${TEST_N:-}" ] && [ "$TEST_N" -gt 0 ] 2>/dev/null && COMMON_ARGS="$COMMON_ARGS --test-n $TEST_N"
[ -n "$DATA_DIR" ]   && COMMON_ARGS="$COMMON_ARGS --data-dir $DATA_DIR"
[ -n "$RESULT_DIR" ] && COMMON_ARGS="$COMMON_ARGS --result-dir $RESULT_DIR"

# ============================================
# Print configuration
# ============================================

echo "============================================"
echo "Inference run configuration"
echo "============================================"
echo "GPU IDs: $GPUS"
echo "Task list: ${TASKS:-all tasks}"
echo "Model list: ${NORMALIZED_MODELS:-(default)}"
if [ "$TEST_MODE" = "true" ] || [ "$TEST_MODE" = "1" ]; then
    echo "Test mode: enabled (only 2 items per task)"
elif [ -n "${TEST_N:-}" ] && [ "$TEST_N" -gt 0 ] 2>/dev/null; then
    echo "Test mode: --test-n $TEST_N (IC/UGG/ME $TEST_N items each, GGU $TEST_N per sub-task)"
else
    echo "Test mode: disabled"
fi
echo "Data directory: $DATA_DIR"
echo "Result directory: $RESULT_DIR"
echo "conda install path: $CONDA_BASE"
echo "============================================"
echo ""

# ============================================
# Initialize conda
# ============================================

# Disable set -u errors triggered by conda activate
set +u
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ============================================
# Loop over models (each model uses its own conda env)
# ============================================

if [ -z "$NORMALIZED_MODELS" ]; then
    # No model specified: run directly in the current env (infer.py uses its default list)
    echo ">>> Running infer.py in the current env (no model specified)"
    python infer.py $COMMON_ARGS
    exit $?
fi


OVERALL_STATUS=0
IFS=',' read -ra MODEL_LIST <<< "$NORMALIZED_MODELS"
TOTAL=${#MODEL_LIST[@]}

for i in "${!MODEL_LIST[@]}"; do
    model_cfg="${MODEL_LIST[$i]}"
    seq=$((i + 1))

    # Read the conda_env field from config (fall back to base)
    env_name=$(python -c "
import json, sys
try:
    with open('$model_cfg') as f:
        cfg = json.load(f)
    print(cfg.get('conda_env', 'base'))
except Exception as e:
    print('base')
    sys.stderr.write(f'Warning: failed to read $model_cfg: {e}\n')
")

    model_name=$(basename "$model_cfg" .json)

    echo ""
    echo "============================================"
    echo "[$seq/$TOTAL] $model_name  (env: $env_name)"
    echo "============================================"

    MODEL_LOG="$LOG_DIR/${model_name}_${RUN_TIMESTAMP}.log"
    echo "Model log: $MODEL_LOG"

    # Activate the env in a subshell and run, writing a separate per-model log
    (
        conda activate "$env_name" || {
            echo "Error: failed to activate conda env [$env_name], skipping $model_name"
            exit 1
        }
        echo "Python: $(which python)"
        python infer.py $COMMON_ARGS --models "$model_cfg"
    ) 2>&1 | tee -a "$MODEL_LOG"
    rc=${PIPESTATUS[0]}

    if [ $rc -ne 0 ]; then
        echo "[$seq/$TOTAL] $model_name failed (exit $rc)"
        OVERALL_STATUS=$rc
    else
        echo "[$seq/$TOTAL] $model_name completed"
    fi
done

echo ""
echo "============================================"
