#!/bin/bash
# ============================================
# Download the missing weights for 4 models (ILLUME+ / SEED-X / Show-o2 / TokenFlow)
# Uses the hf-mirror mirror. Model groups download in parallel; within a group, downloads are sequential.
# Usage: ./download_weights.sh [model name...]   no arguments = all
# Logs: download_logs/<model>.log
# ============================================
set -u
export HF_ENDPOINT=https://hf-mirror.com

UM="/path/to/Unified_Models"
LOGDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download_logs"
mkdir -p "$LOGDIR"

HF() { huggingface-cli download "$@"; }

dl_illume() {
    local ck="$UM/ILLUME_plus/checkpoints"
    mkdir -p "$ck"
    HF ILLUME-MLLM/illume_plus-qwen2_5-7b   --local-dir "$ck/illume_plus-qwen2_5-7b"
    HF ILLUME-MLLM/dualvitok                --local-dir "$ck/dualvitok"
    HF ILLUME-MLLM/dualvitok-sdxl-decoder   --local-dir "$ck/dualvitok-sdxl-decoder"
    # Symlink the MLLM to the logdir path expected by the stage3 config
    local ld="$UM/ILLUME_plus/logdir/illume_plus_7b"
    mkdir -p "$ld"
    ln -sfn "$ck/illume_plus-qwen2_5-7b" "$ld/illume_plus-qwen2_5-7b_stage3"
    echo "[illume] done"
}

dl_seedx() {
    local pt="$UM/SEED-X/pretrained"
    mkdir -p "$pt"
    # Main bundle: internally already laid out as seed_x / seed_x_i / seed_x_edit / seed_detokenizer / cvlm_..._tokenizer
    HF AILab-CVC/SEED-X-17B --local-dir "$pt"
    # Base SDXL (used by the generation/editing detokenizer)
    HF stabilityai/stable-diffusion-xl-base-1.0 --local-dir "$pt/stable-diffusion-xl-base-1.0"
    # Qwen-VL-Chat (extract the ViT)
    HF Qwen/Qwen-VL-Chat --local-dir "$pt/Qwen-VL-Chat"
    # Extract qwen_vit_G.pt -> pretrained/QwenViT/ (run in the seedx env)
    cd "$UM/SEED-X" && "$HOME/anaconda3/envs/seedx/bin/python" src/tools/reload_qwen_vit.py 2>&1 \
        || echo "[seedx] reload_qwen_vit.py failed (manual handling required)"
    echo "[seedx] done"
}

dl_showo2() {
    # Cache with only config.json: clear and re-download to ensure the .bin weights are complete
    rm -rf "$HOME/.cache/huggingface/hub/models--showlab--show-o2-7B"
    HF showlab/show-o2-7B   # Download into the HF cache; from_pretrained resolves by repo id
    echo "[showo2] done"
}

dl_tokenflow() {
    # understanding 14B: the cached tokenizer.json is corrupted -> delete and re-download
    rm -rf "$HOME/.cache/huggingface/hub/models--ByteFlow-AI--TokenFlow-llava-qwen2.5-14B-finetuning"
    HF ByteFlow-AI/TokenFlow-llava-qwen2.5-14B-finetuning
    # generation t2i (repo name corrected)
    HF ByteFlow-AI/TokenFlow-t2i
    # VQ tokenizer .pt → pretrained_ckpts/
    local pc="$UM/TokenFlow/pretrained_ckpts"
    mkdir -p "$pc"
    HF ByteFlow-AI/TokenFlow tokenflow_clipb_32k_enhanced.pt tokenflow_siglip_32k.pt --local-dir "$pc"
    echo "[tokenflow] done"
}

run_group() {
    local name="$1"; local fn="$2"
    echo "[$(date '+%H:%M:%S')] START $name"
    ( $fn ) > "$LOGDIR/$name.log" 2>&1
    echo "[$(date '+%H:%M:%S')] END   $name (rc=$?)"
}

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(illume seedx showo2 tokenflow)

declare -A FN=( [illume]=dl_illume [seedx]=dl_seedx [showo2]=dl_showo2 [tokenflow]=dl_tokenflow )

pids=()
for m in "${MODELS[@]}"; do
    run_group "$m" "${FN[$m]}" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "All download groups completed. Logs: $LOGDIR/"
