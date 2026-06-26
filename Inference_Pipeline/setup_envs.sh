#!/usr/bin/env bash
# ============================================================================
# setup_envs.sh — create one conda environment per Unison backend and install
# each model's upstream code, plus the shared base/judge environment.
#
# Every model backend has heavy, mutually-incompatible dependencies, so each
# runs in its OWN conda env (the `conda_env` field in config/<Model>.json).
# This script clones each upstream repository and installs it into the matching
# env, exactly the way `download_weights.sh` fetches the matching weights.
#
# Usage:
#   ./setup_envs.sh                          # set up EVERYTHING (base + all models)
#   ./setup_envs.sh base bagel uniworld      # set up selected groups only
#   UMM=/data/Unified_Models ./setup_envs.sh  # choose where upstream CODE is cloned
#
# Groups  (group -> conda env -> upstream repo):
#   base       -> unison      (this repo's requirements.txt: orchestrator + judge)
#   bagel      -> bagel       ByteDance-Seed/Bagel
#   janus      -> janus       deepseek-ai/Janus
#   omnigen2   -> omnigen2    VectorSpaceLab/OmniGen2
#   seedx      -> seedx       AILab-CVC/SEED-X
#   showo      -> showo2      showlab/Show-o          (Show-o-1.3B + Show-o2)
#   tokenflow  -> tokenflow   ByteVisionLab/TokenFlow
#   uniworld   -> univa       PKU-YuanGroup/UniWorld  (code in UniWorld/UniWorld-V1)
#   illume     -> illume      illume-unified-mllm/ILLUME_plus
#   ddit       -> d-dit       zijieli-Jlee/Dual-Diffusion
#
# Requirements:
#   - conda (Anaconda/Miniconda). Point CONDA_BASE at it if not in $HOME/anaconda3.
#   - git, and network access to GitHub / PyPI.
#   - The conda env names created here MUST match the `conda_env` field in the
#     model configs — keep them in sync if you rename anything.
#
# Notes:
#   - Idempotent: an existing conda env is reused, an existing clone is left as-is
#     (only `git clone` is skipped, pip install still runs so deps stay current).
#   - $UMM is the SAME third-party root used by download_weights.sh, so each repo
#     is cloned to exactly the path the config `*_project_path` fields expect
#     (e.g. $UMM/Bagel, $UMM/UniWorld/UniWorld-V1). After running both scripts,
#     replace the /path/to/Unified_Models placeholders in config/*.json with $UMM.
#   - flash-attn / flash_attn builds are best-effort; a failure is logged and the
#     group continues (most backends run without it, just slower).
#   - Per-group logs: setup_logs/<group>.log
# ============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Third-party model CODE root (shared with download_weights.sh weight root).
UMM="${UMM:-/path/to/Unified_Models}"
# conda install prefix (used to source conda.sh), same convention as run.sh.
CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
# Optional faster PyPI / HF mirrors (uncomment / export to use them).
: "${HF_ENDPOINT:=https://hf-mirror.com}"; export HF_ENDPOINT

LOGDIR="$SCRIPT_DIR/setup_logs"
mkdir -p "$LOGDIR"

# Light deps infer.py itself needs. run.sh launches infer.py INSIDE each model's
# env, so these must exist in every backend env (torch/transformers come from the
# upstream repo and are intentionally left untouched here).
ORCH_DEPS="pyyaml pandas tqdm pillow numpy"

if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "ERROR: conda.sh not found at $CONDA_BASE/etc/profile.d/conda.sh" >&2
    echo "       Set CONDA_BASE=/path/to/conda (the dir containing etc/profile.d/conda.sh)." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
set +u   # conda activate trips `set -u`

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

ensure_env() {  # ensure_env <name> <python_version>
    local name="$1" py="$2"
    if env_exists "$name"; then
        echo "[env] '$name' already exists — reusing"
    else
        echo "[env] creating '$name' (python=$py)"
        conda create -n "$name" "python=$py" -y
    fi
}

clone_repo() {  # clone_repo <git_url> <dest_dir>
    local url="$1" dest="$2"
    if [ -d "$dest/.git" ]; then
        echo "[git] $dest already cloned — skipping clone"
    else
        echo "[git] cloning $url -> $dest"
        mkdir -p "$(dirname "$dest")"
        git clone "$url" "$dest"
    fi
}

# Run install commands inside an env, from a working directory.
in_env() {  # in_env <env> <workdir> <bash_commands>
    local env="$1" workdir="$2" cmds="$3"
    ( conda activate "$env" \
        && echo "[python] $(which python)  ($(python --version 2>&1))" \
        && cd "$workdir" \
        && eval "$cmds" )
}

# ---------------------------------------------------------------------------
# Per-group setup functions
# ---------------------------------------------------------------------------

setup_base() {
    # Orchestrator (infer.py) + evaluation judge (evaluate_unison.py) env.
    ensure_env unison 3.10
    in_env unison "$REPO_ROOT" "pip install -r requirements.txt"
}

setup_bagel() {
    ensure_env bagel 3.10
    clone_repo https://github.com/ByteDance-Seed/Bagel.git "$UMM/Bagel"
    in_env bagel "$UMM/Bagel" "
        pip install -r requirements.txt
        pip install flash_attn==2.5.8 --no-build-isolation || echo '[bagel] flash_attn build failed (optional)'
        pip install -q $ORCH_DEPS
    "
}

setup_janus() {
    ensure_env janus 3.10
    clone_repo https://github.com/deepseek-ai/Janus.git "$UMM/Janus"
    in_env janus "$UMM/Janus" "
        pip install -e .
        pip install -q $ORCH_DEPS
    "
}

setup_omnigen2() {
    ensure_env omnigen2 3.11
    clone_repo https://github.com/VectorSpaceLab/OmniGen2.git "$UMM/OmniGen2"
    in_env omnigen2 "$UMM/OmniGen2" "
        pip install torch==2.6.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124
        pip install -r requirements.txt
        pip install flash-attn==2.7.4.post1 --no-build-isolation || echo '[omnigen2] flash-attn build failed (optional)'
        pip install -q $ORCH_DEPS
    "
}

setup_seedx() {
    ensure_env seedx 3.10
    clone_repo https://github.com/AILab-CVC/SEED-X.git "$UMM/SEED-X"
    in_env seedx "$UMM/SEED-X" "
        pip install -r requirements.txt
        pip install -q $ORCH_DEPS
    "
    echo "[seedx] NOTE: after downloading weights, extract the Qwen ViT (qwen_vit_G.pt)"
    echo "        with this repo's src/tools/reload_qwen_vit.py inside the 'seedx' env."
}

setup_showo() {
    # Env 'showo2' is shared by Show-o-1.3B and Show-o2 (see config/Show-o*.json).
    ensure_env showo2 3.10
    clone_repo https://github.com/showlab/Show-o.git "$UMM/Show-o"
    in_env showo2 "$UMM/Show-o" "
        pip install -r requirements.txt
        pip install -q $ORCH_DEPS
    "
    echo "[showo] NOTE: Show-o2 also needs the Wan2.1 VAE (Wan2.1_VAE.pth) — see"
    echo "        download_weights.sh; set vae_model_path in config/Show-o2-*.json."
}

setup_tokenflow() {
    ensure_env tokenflow 3.10
    clone_repo https://github.com/ByteVisionLab/TokenFlow.git "$UMM/TokenFlow"
    in_env tokenflow "$UMM/TokenFlow" "
        pip install -r t2i/requirements.txt
        ( cd i2t && pip install -e . )   # installs the 'llava' understanding package
        pip install -q $ORCH_DEPS
    "
}

setup_uniworld() {
    # Env 'univa'; the runnable code lives in the UniWorld-V1 subdir.
    ensure_env univa 3.10
    clone_repo https://github.com/PKU-YuanGroup/UniWorld.git "$UMM/UniWorld"
    in_env univa "$UMM/UniWorld/UniWorld-V1" "
        pip install -r requirements.txt
        pip install flash_attn --no-build-isolation || echo '[uniworld] flash_attn build failed (optional)'
        pip install -q $ORCH_DEPS
    "
}

setup_illume() {
    ensure_env illume 3.9
    clone_repo https://github.com/illume-unified-mllm/ILLUME_plus.git "$UMM/ILLUME_plus"
    in_env illume "$UMM/ILLUME_plus" "
        pip install -U pip setuptools
        ( cd ILLUME && pip install -e . )
        pip install flash-attn --no-build-isolation || echo '[illume] flash-attn build failed (optional)'
        pip install -q $ORCH_DEPS
    "
}

setup_ddit() {
    # Env 'd-dit'. Public code only; bring your own trained checkpoint (see README).
    ensure_env d-dit 3.10
    clone_repo https://github.com/zijieli-Jlee/Dual-Diffusion.git "$UMM/Dual-Diffusion"
    in_env d-dit "$UMM/Dual-Diffusion" "
        pip install -r requirements.txt
        pip install -q $ORCH_DEPS
    "
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

declare -A FN=(
    [base]=setup_base          [bagel]=setup_bagel        [janus]=setup_janus
    [omnigen2]=setup_omnigen2  [seedx]=setup_seedx        [showo]=setup_showo
    [tokenflow]=setup_tokenflow [uniworld]=setup_uniworld [illume]=setup_illume
    [ddit]=setup_ddit
)
ALL=(base bagel janus omnigen2 seedx showo tokenflow uniworld illume ddit)

GROUPS=("$@")
[ ${#GROUPS[@]} -eq 0 ] && GROUPS=("${ALL[@]}")

if [ "$UMM" = "/path/to/Unified_Models" ]; then
    echo "WARNING: UMM is the placeholder '/path/to/Unified_Models'. Set your own root," >&2
    echo "         e.g.  UMM=/data/Unified_Models ./setup_envs.sh" >&2
fi
echo "conda base:    $CONDA_BASE"
echo "code root UMM: $UMM"
echo "groups:       ${GROUPS[*]}"
echo

STATUS=0
for g in "${GROUPS[@]}"; do
    fn="${FN[$g]:-}"
    if [ -z "$fn" ]; then echo "Unknown group: $g (skipped)" >&2; continue; fi
    echo "============================================================"
    echo ">>> $g"
    echo "============================================================"
    if "$fn" 2>&1 | tee "$LOGDIR/$g.log"; then
        echo "<<< $g OK"
    else
        echo "<<< $g FAILED (see $LOGDIR/$g.log)" >&2
        STATUS=1
    fi
    echo
done

echo "Done. Logs: $LOGDIR/"
echo "Next: download weights (./download_weights.sh) and point each config/*.json"
echo "      model_path / *_project_path at \$UMM=$UMM."
exit $STATUS
