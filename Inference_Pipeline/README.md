# Inference Pipeline

Runs Unison benchmark inference across unified multimodal models on multiple GPUs. Each model executes in its own isolated conda environment.


## Setup

### 1. Environments

`setup_envs.sh` creates one conda env per model backend and clones each upstream repo. Set `UM` to the directory where upstream code and weights will live.

```bash
UM=/data/Unified_Models ./setup_envs.sh              # everything
UM=/data/Unified_Models ./setup_envs.sh base bagel   # selected groups
```

| Group | Conda env | Model(s) |
|-------|-----------|----------|
| `base` | `unison` | orchestrator + judge (this repo's `requirements.txt`) |
| `bagel` | `bagel` | BAGEL-7B-MoT |
| `janus` | `janus` | Janus-Pro-1B / 7B |
| `omnigen2` | `omnigen2` | OmniGen2 |
| `seedx` | `seedx` | SEED-X-17B |
| `showo` | `showo2` | Show-o-1.3B / Show-o2-1.5B / 7B |
| `tokenflow` | `tokenflow` | TokenFlow |
| `uniworld` | `univa` | UniWorld-V1 |
| `illume` | `illume` | ILLUME+-3B / 7B |
| `ddit` | `d-dit` | D-DiT |

### 2. Weights

```bash
UM=/data/Unified_Models ./download_weights.sh                 # all models
UM=/data/Unified_Models ./download_weights.sh bagel showo1   # selected
```

Then update `model_path` (and any `*_project_path`) in each `config/*.json` to point at `$UM/...`.

### 3. Data

Place Unison-data at `../data/` (relative to this directory), or set `DATA_DIR` in `run.sh`.


## Quick start

```bash
cd Inference_Pipeline

# Edit run.sh: set GPUS, MODELS, TASKS, DATA_DIR, RESULT_DIR
GPUS=0,1,2,3,4,5,6,7 MODELS=BAGEL-7B-MoT TASKS=IC,UGG,GGU,ME ./run.sh

```

Results land in `result/<ModelName>/<TaskID>/<TaskID>_<ModelName>_results.csv`.


## Configuration

### `run.sh` variables

| Variable | Description |
|----------|-------------|
| `GPUS` | Comma-separated GPU IDs (required) |
| `MODELS` | Model name(s) — resolves to `config/<Name>.json` |
| `TASKS` | Subset of `IC,UGG,GGU,ME` (default: all) |
| `DATA_DIR` | Path to Unison-data (default: `../data`) |
| `RESULT_DIR` | Output directory (default: `result`) |
| `TEST_MODE` | `true` to run 2 items per task |
| `TEST_N` | Run exactly N items per task (overrides `TEST_MODE`) |

### `config/<Model>.json` fields

| Field | Description |
|-------|-------------|
| `model_name` | Display name; also the result subdirectory |
| `model_path` | Local weight path or HuggingFace repo ID |
| `api_type` | Inference backend (see dispatch map below) |
| `conda_env` | Conda env this model runs in |
| `capabilities` | Supported modes: `understanding`, `generation`, `editing` |
| `inference_mode` | Default mode |
| `seed` | Random seed (default: 666) |

**Supported `api_type` values:** `bagel`, `janus`, `seed_x`, `tokenflow`, `showo`, `showo2`, `uniworld`, `omnigen2`, `illume`, `ddit`

## How `infer.py` works

1. Loads `tasks.yaml` and model configs from `config/`.
2. Validates all task data before starting.
3. Splits each task's dataset evenly across GPUs using `multiprocessing` (spawn).
4. Each GPU process runs the model adapter (`model_inference/<api_type>_inference.py`) over its shard.
5. Per-GPU temp CSVs are merged and deduplicated into the final results file.

Runs are **checkpoint/resume aware** — already-processed items are skipped on restart.

## Tasks

| ID | Name | Operations |
|----|------|------------|
| `IC` | Internal Consistency | `generation` → `understanding` (per-question VQA) |
| `UGG` | Understanding Guided Generation | `understanding` (bbox) → `editing` → `unify` |
| `GGU` | Generation Guided Understanding | `generation` → `understanding` → `unify` |
| `ME` | Mutual Enhancement | `unify` (u2g multi-turn) + `unify` (g2u multi-turn) |

Task prompts are defined in `tasks.yaml` via references to functions in `prompt_templates.py`.

## Adding a new model

1. Create `config/<ModelName>.json` with `model_name`, `model_path`, `api_type`, `conda_env`, and `capabilities`.
2. If `api_type` is new, add `model_inference/<api_type>_inference.py` exposing:
   ```python
   def <api_type>_inference_function(image_path, prompt, config, index=None, round_number=None) -> str
   ```
3. Run with `MODELS=<ModelName> ./run.sh`.
