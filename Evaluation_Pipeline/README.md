# Evaluation Pipeline

Scores Unison inference results using a VLM judge. Supports two backends: a local trained **Unison-Judge** (Qwen3-VL-8B, default) or the **DashScope Qwen3-VL-Plus API**.

## Quick start

```bash
cd Evaluation_Pipeline

# Local judge (default) — all four tasks
GPU_IDS=0,1,2,3 MODELS=UniWorld-V1 ./run_evaluate_unison.sh

# Select specific tasks
MODELS=BAGEL-7B-MoT TASKS=IC,GGU ./run_evaluate_unison.sh

# Several models at once
MODELS="BAGEL-7B-MoT,Janus-Pro-7B" ./run_evaluate_unison.sh

# API judge
JUDGE_BACKEND=api DASHSCOPE_API_KEY=sk-... MODELS=UniWorld-V1 ./run_evaluate_unison.sh
```

Output is written to `eval_<ModelName>.json`. After evaluating multiple models, aggregate:

```bash
python aggregate_results.py   # -> evaluation_summary.json
```

## Setup

### Unison-Judge weights

Place the Unison-Judge checkpoint at `./judge_model/` (default path). Or point at it explicitly:

```bash
LOCAL_JUDGE_MODEL=/path/to/judge ./run_evaluate_unison.sh
```

No local weights needed when using `JUDGE_BACKEND=api`.

### Data

Place Unison-data at `../data/` (relative to this directory), or set `DATA_DIR`.

The evaluator reads the inference results produced by the Inference Pipeline from `../Inference_Pipeline/result/<ModelName>/`. Override with `RESULT_DIR`.

## `run_evaluate_unison.sh` variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS` | `UniWorld-V1` | Comma-separated model name(s) |
| `TASKS` | `IC,UGG,GGU,ME` | Tasks to score |
| `JUDGE_BACKEND` | `local` | `local` or `api` |
| `GPU_IDS` | `0-1` | GPUs for the local judge (e.g. `0,1,2,3` or `0-7`) |
| `LOCAL_JUDGE_MODEL` | `./judge_model` | Path to Unison-Judge weights |
| `DATA_DIR` | `../data` | Unison-data root |
| `INFERENCE_BASE_DIR` | `../Inference_Pipeline` | Inference Pipeline root |
| `RESULT_DIR` | `$INFERENCE_BASE_DIR/result/<MODEL>` | Inference results directory |
| `OUTPUT` | `eval_<MODEL>.json` | Output JSON path |
| `MAX_WORKERS` | `8` | Parallel judge threads |
| `DASHSCOPE_API_KEY` | — | Required for `JUDGE_BACKEND=api` |

## Direct invocation

```bash
python evaluate_unison.py \
    --result-dir         ../Inference_Pipeline/result/UniWorld-V1 \
    --data-dir           ../data \
    --inference-base-dir ../Inference_Pipeline \
    --judge-backend      local \
    --local-model-path   ./judge_model \
    --gpu-ids            0,1,2,3 \
    --tasks              IC,UGG,GGU,ME \
    --output             eval_UniWorld-V1.json
```

## Scoring metrics

Each task produces three scores (0–10 scale):

| Metric | Description |
|--------|-------------|
| `understanding_score` | Understanding-only track |
| `generation_score` | Generation-only track |
| `unified_score` | Unified track (joint understanding + generation) |

`aggregate_results.py` reads all `evaluation_results/*.csv` files and writes `evaluation_summary.json` with per-task and overall averages across all evaluated models.

## Directory layout

```
Evaluation_Pipeline/
├── evaluate_unison.py        # Main entry point
├── run_evaluate_unison.sh    # Launcher
├── aggregate_results.py      # Multi-model summary
├── common/
│   ├── judge.py              # API judge client (DashScope Qwen3-VL-Plus)
│   ├── local_judge.py        # Local judge driver (multi-GPU Qwen3-VL)
│   ├── local_qwenvl.py       # Qwen3-VL inference wrapper
│   ├── io.py                 # CSV I/O helpers
│   ├── normalize.py          # Answer normalization
│   └── geometry.py           # Bbox IoU utilities
├── tasks/
│   ├── evaluate_ic.py        # IC scorer
│   ├── evaluate_ugg.py       # UGG scorer
│   ├── evaluate_ggu.py       # GGU scorer
│   └── evaluate_me.py        # ME scorer
└── judge_model/              # <- place Unison-Judge weights here
```
