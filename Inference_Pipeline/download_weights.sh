#!/bin/bash
# ============================================================================
# Download the model weights for every Unison inference backend.
#
# Usage:
#   ./download_weights.sh                       # download ALL models
#   ./download_weights.sh bagel janus7b showo1  # download selected groups
#   UMM=/data/Unified_Models ./download_weights.sh   # choose the local weight root
#
# Groups (one per model):
#   bagel  janus7b  janus1b  omnigen2  uniworld  seedx
#   showo1  showo2_7b  showo2_1_5b  tokenflow  illume7b  illume3b  ddit
#
# Requirements:
#   - huggingface-cli   (pip install -U "huggingface_hub[cli]")
#   - For GATED repos (FLUX.1-dev, Stable Diffusion 3): run `huggingface-cli login`
#     and accept the model license on its HF page first.
#   - Optional faster mirror: export HF_ENDPOINT=https://hf-mirror.com
#
# How weights are placed:
#   * Top-level model weights go to  $UMM/<Model>/...  — then edit that model's
#     config/<Model>.json `model_path` (replace the /path/to/Unified_Models
#     placeholder) to point at the downloaded directory.
#   * Auxiliary base models referenced by HuggingFace repo id from inside the
#     model code/config (phi-1_5, magvitv2, CLIP, Qwen2.5, ...) are downloaded
#     into the HF cache, where from_pretrained resolves them by repo id.
#
# NOTE: model *code* and per-model conda environments are NOT handled here.
#       Clone each upstream repository and create its conda env per the project
#       README before running inference. This script only fetches weights.
#
# Logs: download_logs/<group>.log
# ============================================================================
set -u
: "${HF_ENDPOINT:=https://hf-mirror.com}"; export HF_ENDPOINT

UMM="${UMM:-/path/to/Unified_Models}"
LOGDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download_logs"
mkdir -p "$LOGDIR"

HF() { huggingface-cli download "$@"; }

# -------------------- repo-id models (also placed under $UMM) --------------------

dl_bagel() {
    HF ByteDance-Seed/BAGEL-7B-MoT --local-dir "$UMM/Bagel/BAGEL-7B-MoT"
    echo "[bagel] done -> set model_path to $UMM/Bagel/BAGEL-7B-MoT (or keep the repo id)"
}

dl_janus7b() {
    HF deepseek-ai/Janus-Pro-7B --local-dir "$UMM/Janus-Pro-7B"
    echo "[janus7b] done -> model_path: $UMM/Janus-Pro-7B"
}

dl_janus1b() {
    HF deepseek-ai/Janus-Pro-1B --local-dir "$UMM/Janus-Pro-1B"
    echo "[janus1b] done -> model_path: $UMM/Janus-Pro-1B"
}

dl_omnigen2() {
    HF OmniGen2/OmniGen2 --local-dir "$UMM/OmniGen2/OmniGen2"
    echo "[omnigen2] done -> model_path: $UMM/OmniGen2/OmniGen2 (or keep the repo id)"
}

# -------------------- local-dir models (+ base models in cache) --------------------

dl_uniworld() {
    local w="$UMM/UniWorld/UniWorld-V1/model_weights/UniWorld-V1"
    HF LanguageBind/UniWorld-V1 --local-dir "$w"
    # Base models loaded alongside UniWorld (FLUX.1-dev is GATED -> requires login + license)
    HF black-forest-labs/FLUX.1-dev          --local-dir "$UMM/UniWorld/FLUX.1-dev" \
        || echo "[uniworld] FLUX.1-dev failed (gated: huggingface-cli login + accept license)"
    HF google/siglip2-so400m-patch16-512     --local-dir "$UMM/UniWorld/siglip2-so400m-patch16-512"
    echo "[uniworld] done -> model_path: $w"
}

dl_seedx() {
    local pt="$UMM/SEED-X/model_weights"
    mkdir -p "$pt"
    # Main bundle: internally laid out as seed_x / seed_x_i / seed_x_edit / seed_detokenizer / cvlm_..._tokenizer
    HF AILab-CVC/SEED-X-17B --local-dir "$pt"
    # Base SDXL (used by the generation/editing detokenizer)
    HF stabilityai/stable-diffusion-xl-base-1.0 --local-dir "$pt/stable-diffusion-xl-base-1.0"
    # Qwen-VL-Chat (the ViT is extracted from this)
    HF Qwen/Qwen-VL-Chat --local-dir "$pt/Qwen-VL-Chat"
    echo "[seedx] weights done -> model_path: $pt"
    echo "[seedx] NOTE: extract the ViT (qwen_vit_G.pt) via the SEED-X repo's"
    echo "        src/tools/reload_qwen_vit.py inside the seedx conda env."
}

dl_showo1() {
    # Show-o v1 (1.3B): all loaded by repo id -> HF cache.
    HF showlab/show-o-w-clip-vit-512x512
    HF microsoft/phi-1_5
    HF showlab/magvitv2
    HF openai/clip-vit-large-patch14-336
    echo "[showo1] weights done (HF cache, resolved by repo id)"
    echo "[showo1] NOTE: also clone https://github.com/showlab/Show-o and set"
    echo "        showo_project_path in config/Show-o-1.3B.json to that checkout."
}

dl_showo2_7b() {
    HF showlab/show-o2-7B
    HF Qwen/Qwen2.5-7B-Instruct
    echo "[showo2_7b] weights done (HF cache)"
    echo "[showo2_7b] NOTE: also fetch the Wan2.1 VAE (Wan2.1_VAE.pth) and set"
    echo "        vae_model_path; clone https://github.com/showlab/Show-o for showo_project_path."
}

dl_showo2_1_5b() {
    HF showlab/show-o2-1.5B --local-dir "$UMM/Show-o/show-o2-1.5B"
    HF Qwen/Qwen2.5-1.5B-Instruct
    echo "[showo2_1_5b] done -> model_path: $UMM/Show-o/show-o2-1.5B"
    echo "[showo2_1_5b] NOTE: also fetch the Wan2.1 VAE (Wan2.1_VAE.pth) and set vae_model_path."
}

dl_tokenflow() {
    # Understanding 14B
    HF ByteFlow-AI/TokenFlow-llava-qwen2.5-14B-finetuning
    # Generation t2i
    HF ByteFlow-AI/TokenFlow-t2i
    # VQ tokenizer .pt files -> pretrained_ckpts/
    local pc="$UMM/TokenFlow/pretrained_ckpts"
    mkdir -p "$pc"
    HF ByteFlow-AI/TokenFlow tokenflow_clipb_32k_enhanced.pt tokenflow_siglip_32k.pt --local-dir "$pc"
    echo "[tokenflow] done -> tokenizer_path: $pc/tokenflow_clipb_32k_enhanced.pt"
}

dl_illume7b() {
    local ck="$UMM/ILLUME_plus/checkpoints"
    mkdir -p "$ck"
    HF ILLUME-MLLM/illume_plus-qwen2_5-7b --local-dir "$ck/illume_plus-qwen2_5-7b"
    HF ILLUME-MLLM/dualvitok              --local-dir "$ck/dualvitok"
    HF ILLUME-MLLM/dualvitok-sdxl-decoder --local-dir "$ck/dualvitok-sdxl-decoder"
    # Symlink the MLLM to the logdir path expected by the stage3 config
    local ld="$UMM/ILLUME_plus/logdir/illume_plus_7b"
    mkdir -p "$ld"
    ln -sfn "$ck/illume_plus-qwen2_5-7b" "$ld/illume_plus-qwen2_5-7b_stage3"
    echo "[illume7b] done"
}

dl_illume3b() {
    local ck="$UMM/ILLUME_plus/checkpoints"
    mkdir -p "$ck"
    HF ILLUME-MLLM/illume_plus-qwen2_5-3b --local-dir "$ck/illume_plus-qwen2_5-3b"
    # DualViTok + SDXL decoder are shared with the 7B model (download once if absent)
    [ -d "$ck/dualvitok" ]              || HF ILLUME-MLLM/dualvitok              --local-dir "$ck/dualvitok"
    [ -d "$ck/dualvitok-sdxl-decoder" ] || HF ILLUME-MLLM/dualvitok-sdxl-decoder --local-dir "$ck/dualvitok-sdxl-decoder"
    local ld="$UMM/ILLUME_plus/logdir/illume_plus_3b"
    mkdir -p "$ld"
    ln -sfn "$ck/illume_plus-qwen2_5-3b" "$ld/illume_plus-qwen2_5-3b_stage3"
    echo "[illume3b] done"
}

dl_ddit() {
    HF JleeOfficial/dual_diff_sd3_512_base --local-dir "$UMM/Dual-Diffusion/dual_diff_sd3_512_base"
    HF JleeOfficial/dual_diff_sd3_512_sft  --local-dir "$UMM/Dual-Diffusion/dual_diff_sd3_512_sft"
    echo "[ddit] done -> model_path: $UMM/Dual-Diffusion/dual_diff_sd3_512_sft"
}

run_group() {
    local name="$1"; local fn="$2"
    echo "[$(date '+%H:%M:%S')] START $name"
    ( $fn ) > "$LOGDIR/$name.log" 2>&1
    echo "[$(date '+%H:%M:%S')] END   $name (rc=$?)  -> $LOGDIR/$name.log"
}

declare -A FN=(
    [bagel]=dl_bagel               [janus7b]=dl_janus7b        [janus1b]=dl_janus1b
    [omnigen2]=dl_omnigen2         [uniworld]=dl_uniworld      [seedx]=dl_seedx
    [showo1]=dl_showo1             [showo2_7b]=dl_showo2_7b    [showo2_1_5b]=dl_showo2_1_5b
    [tokenflow]=dl_tokenflow       [illume7b]=dl_illume7b      [illume3b]=dl_illume3b
    [ddit]=dl_ddit
)
ALL=(bagel janus7b janus1b omnigen2 uniworld seedx showo1 showo2_7b showo2_1_5b tokenflow illume7b illume3b ddit)

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=("${ALL[@]}")

if [ "$UMM" = "/path/to/Unified_Models" ]; then
    echo "WARNING: UMM is the placeholder '/path/to/Unified_Models'. Set UMM=/your/root first," >&2
    echo "         e.g.  UMM=/data/Unified_Models ./download_weights.sh" >&2
fi
echo "Weight root (UMM): $UMM"
echo "Downloading groups: ${MODELS[*]}"

pids=()
for m in "${MODELS[@]}"; do
    if [ -z "${FN[$m]:-}" ]; then echo "Unknown group: $m (skipped)" >&2; continue; fi
    run_group "$m" "${FN[$m]}" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "All download groups completed. Logs: $LOGDIR/"
