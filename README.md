<div align="center">
<h1><img src="images/Unison-logo.png" alt="Unison" height="25"/>: Benchmarking Unified Multimodal Models via Synergistic Understanding and Generation</h3>
</div>

<p align="center"><b><a href="https://jinyuliuu.github.io">Jinyu Liu</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=kLY6SUMAAAAJ&hl=en&oi=ao">Xincheng Shuai</a><sup>1</sup>, <a href="https://henghuiding.com">Henghui Ding</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=f3_FP8AAAAAJ&hl=en&oi=ao">Yu-Gang Jiang</a><sup>1</sup></b></p>
<p align="center"><sup>1 </sup>Fudan University</p>

<div align="center">
<a href='https://arxiv.org/abs/xxxxx'><img src='https://img.shields.io/badge/ICML 2026-Unison-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://henghuiding.com/Unison'><img src='https://img.shields.io/badge/Project-Page-orange'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/datasets/FudanCVL/Unison"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark-Unison--Bench-green"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/FudanCVL/Unison-Judge"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-Unison--Judge-green"></a> &nbsp;&nbsp;&nbsp;&nbsp;
</div>

## Overview

***TL;DR: Unison evaluates Unified Multimodal Models (UMMs) by leveraging the synergy between understanding and generation across four dimensions. Unison-Judge, the automatic evaluation model, achieves an 88.7% alignment with human judgments.***


<p align="center"><img src="images/overview.png" alt="Unison Overview" width="100%"/></p>

We introduce **Unison**, a comprehensive benchmark comprising 2,169 high-quality unified task samples, designed to evaluate joint understanding and generation in unified multimodal models. Unison offers three key strengths: **1) Comprehensive Dimensions**: Unison encompasses internal consistency, understanding-guided generation, generation-guided understanding, and mutual enhancement to enable holistic evaluation. **2) Diagnostic Evaluation**: it provides both unified and decoupled tracks for understanding and generation, allowing fine-grained attribution of failure modes and quantitative analysis of the gains from unified modeling. **3) Human Alignment**: we also train Unison-Judge, an evaluation model well aligned with human judgments to achieve reliable assessment.

## News
- **[2026/06/25]** We release **Unison-Bench** and **Unison-Judge**.



- **`Inference_Pipeline/`** — runs benchmark inference across many unified models on multiple GPUs.
- **`Evaluation_Pipeline/`** — scores model outputs with a VLM judge: a local trained
  Qwen3-VL checkpoint (default) or the DashScope Qwen3-VL-Plus API.

> This is the open-source release of the inference and evaluation **code**. The
> companion artifacts are released separately:
>
> - **Unison-data** — the benchmark data (four tasks + the Judge Consistency set).
> - **Unison-Judge** — the trained local Qwen3-VL judge weights.
>
> Model backbone weights are not bundled either. See [Data](#data) and
> [Model weights](#model-weights) for how to obtain and wire everything up.

## Benchmark tasks

Unison defines four tasks that each probe the interplay between understanding and generation:

| Task | Name | What it measures |
|------|------|------------------|
| **IC**  | Internal Consistency        | Generate an image from a prompt, then answer VQA questions about it. |
| **UGG** | Understanding Guided Generation | Use image understanding (e.g. bounding-box extraction) to guide editing. |
| **GGU** | Generation Guided Understanding | Generate an image, then answer spatial-reasoning questions about it. |
| **ME**  | Mutual Enhancement          | Multi-turn dialogue where understanding and generation reinforce each other. |

## Repository layout

```
.
├── Inference_Pipeline/
│   ├── infer.py              # Multi-GPU parallel inference orchestrator
│   ├── tasks.yaml            # Task definitions (data dirs + prompt templates per operation)
│   ├── prompt_templates.py   # Functions that build prompts per task/operation
│   ├── run.sh                # Launch script (set GPUS / MODELS / TASKS)
│   ├── setup_envs.sh         # Create per-backend conda envs + clone upstream code
│   ├── download_weights.sh   # Helper to fetch missing model weights
│   ├── config/               # Per-model JSON configs (model path, api_type, hyperparams)
│   └── model_inference/      # One inference wrapper per model backend
├── Evaluation_Pipeline/
│   ├── evaluate_unison.py     # Single entry point — scores any subset of the four tasks
│   ├── run_evaluate_unison.sh # Launcher (pick models/tasks/judge backend via env vars)
│   ├── aggregate_results.py   # Merge per-model eval_*.json into a summary
│   ├── common/                # Judge clients (API + local Qwen3-VL), IO, normalization, geometry
│   └── tasks/                 # Per-task scoring logic (IC, UGG, GGU, ME)
├── requirements.txt
├── .env.example
└── LICENSE
```

## Installation

There are two layers to install: a **base environment** (the orchestrator,
judge, and data-handling code shared by both pipelines) and **one conda
environment per model backend** (BAGEL, Janus, SEED-X, Show-o/Show-o2, TokenFlow,
UniWorld, OmniGen2, ILLUME+, D-DiT). Each backend has heavy,
mutually-incompatible dependencies, so each runs in its own env, declared via the
`conda_env` field in that model's config.

### Automated setup (recommended)

`Inference_Pipeline/setup_envs.sh` creates every conda environment and installs
each backend's upstream code in one go. `UM` is the third-party **code** root —
the same root `download_weights.sh` uses for **weights** — so repos land exactly
where the configs expect them (e.g. `$UM/Bagel`, `$UM/UniWorld/UniWorld-V1`).

```bash
cd Inference_Pipeline

# Everything: the base/judge env (conda env `unison`) + all model envs
UM=/data/Unified_Models ./setup_envs.sh

# Or only selected groups (group -> conda env):
UM=/data/Unified_Models ./setup_envs.sh base bagel uniworld
```

| Group | conda env | Upstream repo (cloned into `$UM/…`) |
|-------|-----------|-------------------------------------|
| `base`      | `unison`    | — (installs this repo's `requirements.txt`) |
| `bagel`     | `bagel`     | `ByteDance-Seed/Bagel` → `Bagel` |
| `janus`     | `janus`     | `deepseek-ai/Janus` → `Janus` |
| `omnigen2`  | `omnigen2`  | `VectorSpaceLab/OmniGen2` → `OmniGen2` |
| `seedx`     | `seedx`     | `AILab-CVC/SEED-X` → `SEED-X` |
| `showo`     | `showo2`    | `showlab/Show-o` → `Show-o` (Show-o-1.3B + Show-o2) |
| `tokenflow` | `tokenflow` | `ByteVisionLab/TokenFlow` → `TokenFlow` |
| `uniworld`  | `univa`     | `PKU-YuanGroup/UniWorld` → `UniWorld` (code in `UniWorld/UniWorld-V1`) |
| `illume`    | `illume`    | `illume-unified-mllm/ILLUME_plus` → `ILLUME_plus` |
| `ddit`      | `d-dit`     | `zijieli-Jlee/Dual-Diffusion` → `Dual-Diffusion` |

The script needs `conda` (point `CONDA_BASE` at your install if it isn't
`~/anaconda3`) and `git`. It's idempotent — existing envs/clones are reused — and
writes per-group logs to `Inference_Pipeline/setup_logs/`. The env names it
creates must match each config's `conda_env`; keep them in sync if you rename
anything. `flash-attn` builds are best-effort (a failure is logged and skipped).

### Manual setup

Equivalent to the `base` group, if you'd rather not use the script for it:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then install each backend from its upstream repository (see the table above) into
a conda env named after that model's `conda_env`.

## Data

The pipelines expect a benchmark data directory, released separately as
**Unison-data**.

**Where to put it:** download/unpack Unison-data to the repo root as `data/`
(i.e. `Unison-OpenSource/data/`). That's the default both launch scripts point at
(`DATA_DIR=../data`) and what `evaluate_unison.py` auto-detects, so no flags are
needed:

```
Unison-OpenSource/
└── data/                       # <- put Unison-data here (== $DATA_DIR)
    ├── Internal_Consistency/   # IC:  prompts.txt + questions.json
    ├── Und_Guided_Gen/         # UGG: UGG.csv (+ referenced images)
    ├── Gen_Guided_Und/         # GGU: 2D_Spatial/ 3D_Spatial/ Complex_Relation/
    ├── Mutual_Enhancement/     # ME:  ME.csv (+ referenced images)
    └── Judge_Consistency/      # judge-validation set: items.jsonl + images/
```

To keep it elsewhere, pass `--data-dir /path/to/Unison-data` (or set the
`DATA_DIR` env var in the launch scripts).

The per-task layout, referenced by `tasks.yaml` and the evaluation scripts, is:

| Task | File(s) | Key fields |
|------|---------|-----------|
| IC  | `Internal_Consistency/prompts.txt` + `questions.json` | prompts indexed by line; questions keyed by `prompt_N` |
| UGG | `Und_Guided_Gen/UGG.csv`         | `image_path`, `instruction`, `operation`, `bbox`, `mask` |
| GGU | `Gen_Guided_Und/{2D_Spatial/2d_spatial.json, 3D_Spatial/spatial.json, Complex_Relation/complex_relation.json}` | `image_path`, `question`, `options`, `answer`, `description` (three spatial sub-categories, merged at load) |
| ME  | `Mutual_Enhancement/ME.csv`      | `image_path`, `operation`, `instruction`, `final_caption` |

The `Judge_Consistency/` directory (`items.jsonl` + `images/`) is the
human-vs-judge agreement study set used to validate the automatic judge.

## Model weights

Model configs in `Inference_Pipeline/config/*.json` reference local weight paths
using the placeholder root **`/path/to/Unified_Models/...`**. Edit each config to
point at your local checkout, e.g.:

```json
{
  "model_name": "UniWorld-V1",
  "model_path": "/path/to/Unified_Models/UniWorld/UniWorld-V1/model_weights/UniWorld-V1",
  "api_type": "uniworld",
  "conda_env": "univa",
  "capabilities": ["understanding", "generation", "editing"]
}
```

Which tasks actually run is controlled per launch via `--tasks` / the `TASKS`
variable in `run.sh` (filtering the task list in `tasks.yaml`), not by the model
config.

`download_weights.sh` fetches weights for **all** model backends (one download
group per model). Set the local weight root and pick models, e.g.:

```bash
UM=/data/Unified_Models ./download_weights.sh                 # everything
UM=/data/Unified_Models ./download_weights.sh bagel showo1    # selected groups
```

It places top-level weights under `$UM/<Model>/…` (point each config's
`model_path` there) and base models into the HF cache. Gated repos (FLUX.1-dev,
SD3) need `huggingface-cli login` + license acceptance; D-DiT has no public
single-repo release (bring your own checkpoint). Model *code* and per-model conda
envs are installed separately by `setup_envs.sh` (see [Installation](#installation));
run both scripts with the same `UM` so code and weights share one root.

### Judge weights (Unison-Judge)

The default (local) evaluation backend runs a trained **Qwen3-VL-8B** judge,
released separately as **Unison-Judge**.

**Where to put it:** download the Unison-Judge checkpoint into
`Evaluation_Pipeline/judge_model/`. That is the default `evaluate_unison.py`,
`run_evaluate_unison.sh`, and `--local-model-path` all use, so no flags are
needed:

```
Unison-OpenSource/
└── Evaluation_Pipeline/
    └── judge_model/            # <- put the Unison-Judge weights here
        ├── config.json
        ├── model-*.safetensors
        └── ...                 # tokenizer / processor files
```

To keep it elsewhere, point at it with `LOCAL_JUDGE_MODEL=/path/to/judge`
(launch script) or `--local-model-path /path/to/judge` (direct invocation). No
local judge weights are needed when using the `api` backend instead.

## Running inference

All commands run from `Inference_Pipeline/`:

```bash
# Standard multi-GPU run (model name resolves to config/<name>.json)
GPUS=0,1,2,3 MODELS=UniWorld-V1 ./run.sh

# Multiple models, specific tasks
GPUS=0,1,2,3 MODELS="BAGEL-7B-MoT,Janus-Pro-7B" TASKS="IC,GGU" ./run.sh

# Quick test mode (2 items per task)
GPUS=0 MODELS=UniWorld-V1 TEST_MODE=true ./run.sh

# Direct Python invocation
python infer.py --gpus 0,1 --models config/UniWorld-V1.json --data-dir ../data --result-dir result
```

Results are written to `result/<ModelName>/<TaskID>/<TaskID>_<ModelName>_results.csv`.
Runs are checkpoint/resume aware — already-processed items are skipped.

### How `infer.py` works

1. Loads `tasks.yaml` and the per-model JSON configs from `config/`.
2. Validates data for every requested task before starting.
3. Splits each task's data across GPUs using `multiprocessing` (spawn, required for CUDA).
4. Each GPU process runs the model's wrapper (selected by `api_type`) over its shard.
5. Per-GPU temp CSVs are merged into the final results after all processes finish.

### Adding a new model

1. Create `config/<ModelName>.json` with at least `model_name`, `model_path`, `api_type`, and `capabilities`.
2. If `api_type` is not a built-in backend, add
   `model_inference/<name>_inference.py` exposing
   `<name>_inference_function(image_path, prompt, config, index, round_number)`.
3. Run with `MODELS=<ModelName> ./run.sh`.

## Running evaluation

All commands run from `Evaluation_Pipeline/`. The judge has two backends, selected
with `--judge-backend` (env `JUDGE_BACKEND`):

- **`local`** (default) — a trained Qwen3-VL judge (the **Unison-Judge** weights)
  sharded across local GPUs. Set `GPU_IDS` / `--gpu-ids` and point
  `LOCAL_JUDGE_MODEL` / `--local-model-path` at the weights (default
  `./judge_model`).
- **`api`** — the DashScope Qwen3-VL-Plus endpoint; requires
  `DASHSCOPE_API_KEY` (or `--api-key`).

Pick which model(s) and task(s) to score with the `MODELS` and `TASKS` env vars:

```bash
# Local judge (default), all four tasks
GPU_IDS=0,1,2,3 LOCAL_JUDGE_MODEL=./judge_model \
MODELS=UniWorld-V1 ./run_evaluate_unison.sh

# Choose a subset of tasks
MODELS=BAGEL-7B-MoT TASKS=IC,GGU ./run_evaluate_unison.sh

# Several models at once
MODELS="BAGEL-7B-MoT,Janus-Pro-7B" ./run_evaluate_unison.sh

# API judge
export DASHSCOPE_API_KEY=sk-...     # or copy .env.example to .env
JUDGE_BACKEND=api MODELS=UniWorld-V1 ./run_evaluate_unison.sh

# Direct invocation (local judge)
python evaluate_unison.py \
    --result-dir         ../Inference_Pipeline/result/UniWorld-V1 \
    --data-dir           /path/to/Unison-data \
    --inference-base-dir ../Inference_Pipeline \
    --judge-backend      local \
    --local-model-path   ./judge_model \
    --gpu-ids            0,1,2,3 \
    --tasks              IC,UGG,GGU,ME \
    --output             eval_UniWorld-V1.json
```

`--result-dir` should contain the `IC/ UGG/ GGU/ ME/` subdirectories produced by
the inference pipeline. After evaluating several models, merge their per-model
JSON into one summary:

```bash
python aggregate_results.py   # -> evaluation_summary.json
```

## Configuration reference

| Config field | Purpose |
|--------------|---------|
| `model_name` | Display name; also the result subdirectory. |
| `model_path` | Local path or HuggingFace repo ID for the weights. |
| `api_type`   | Selects the inference backend: `bagel`, `janus`, `seed_x`, `tokenflow`, `showo`, `showo2`, `uniworld`, `omnigen2`, `illume`, `ddit`. |
| `capabilities` | Modes the model supports: any of `understanding`, `generation`, `editing`. |
| `conda_env`  | Conda environment the model runs in (keeps incompatible deps isolated). |
| `inference_mode` | Default mode: `understanding`, `generation`, or `editing`. |
| `seed`       | Random seed (default 666). |

## License

Released under the [MIT License](LICENSE).
