<div align="center">
<h1><img src="images/Unison-logo.png" alt="Unison" height="25"/>: Benchmarking Unified Multimodal Models via Synergistic Understanding and Generation</h3>
</div>

<p align="center"><b><a href="https://jinyuliuu.github.io">Jinyu Liu</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=kLY6SUMAAAAJ&hl=en&oi=ao">Xincheng Shuai</a><sup>1</sup>, <a href="https://henghuiding.com">Henghui Ding</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=f3_FP8AAAAAJ&hl=en&oi=ao">Yu-Gang Jiang</a><sup>1</sup></b></p>
<p align="center"><sup>1 </sup>Fudan University</p>

<div align="center">
<a href='https://arxiv.org/abs/xxxxx'><img src='https://img.shields.io/badge/arXiv-2603.15616-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://henghuiding.com/Unison'><img src='https://img.shields.io/badge/Project-Page-orange'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/datasets/FudanCVL/Unison"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Unison-Benchmark-green"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/FudanCVL/Unison-Judge"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Unison--Judge-Model-green"></a> &nbsp;&nbsp;&nbsp;&nbsp;
</div>

## News
- **[2026/06/01]** We release **Unison-Bench** and **Unison-Judge**.
- **[2026/05/01]** **Unison** has been accepted to ICML 2026.

<div align="center">
<img src="images/overview.png" alt="Unison Overview" width="100%"/>
</div>

---

**Unison** is a multimodal benchmark for evaluating *unified* image understanding
and generation models — models that both interpret and synthesize images within a
single architecture. This repository contains the two pipelines needed to
reproduce Unison results:

- **`Inference_Pipeline/`** — runs benchmark inference across many unified models on multiple GPUs.
- **`Evaluation_Pipeline/`** — scores model outputs using Qwen3-VL-Plus as an LLM/VLM judge.

> Benchmark data and model weights are **not** bundled — see [Data](#data) and
> [Model weights](#model-weights) for how to obtain and wire them up.

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
│   ├── download_weights.sh   # Helper to fetch missing model weights
│   ├── config/               # Per-model JSON configs (model path, api_type, hyperparams)
│   └── model_inference/      # One inference wrapper per model backend
├── Evaluation_Pipeline/
│   ├── evaluate_unison.py    # Unified entry point — scores all four tasks
│   ├── evaluate_ic.py        # Standalone local-VLM IC evaluation (optional)
│   ├── evaluate_ggu_quality.py
│   ├── aggregate_results.py  # Merge per-model eval_*.json into a summary
│   ├── common/               # Judge client, IO, normalization, geometry, aggregation
│   ├── tasks/                # Per-task scoring logic (IC, UGG, GGU, ME)
│   └── run_evaluate_*.sh     # Launch scripts
├── requirements.txt
├── .env.example
└── LICENSE
```

## Getting Started

### Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers the orchestration, judge, and data-handling layers.
**Each model backend** (BAGEL, Janus, SEED-X, Show-o2, TokenFlow, UniWorld,
OmniGen2, ILLUME+, D-DiT) has heavy, mutually-incompatible dependencies and is
expected to run inside its **own conda environment** — declared via the
`conda_env` field in that model's config. Install each backend from its upstream
repository before running it.

## Data

The pipelines expect a benchmark `data/` directory (not included). The default
layout, referenced by `tasks.yaml` and the evaluation scripts, is:

| Task | File(s) | Key fields |
|------|---------|-----------|
| IC  | `Internal_Consistency/prompts.txt` + `questions.json` | prompts indexed by line; questions keyed by `prompt_N` |
| UGG | `Und_Guided_Gen/UGG.csv`         | `image_path`, `instruction`, `operation`, `bbox`, `mask` |
| GGU | `Gen_Guided_Und/spatial.json`    | `image_path`, `question`, `options`, `answer`, `description` |
| ME  | `Mutual_Enhancement/ME.csv`      | `image_path`, `operation`, `instruction`, `final_caption` |

Point the pipelines at your data with `--data-dir` (or the `DATA_DIR` env var in
the launch scripts).

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

`download_weights.sh` is a convenience helper for fetching a few backends; review
and adjust its `UM` root before running.

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

All commands run from `Evaluation_Pipeline/`. The judge calls Qwen3-VL-Plus over
the DashScope OpenAI-compatible API — set your key first:

```bash
export DASHSCOPE_API_KEY=sk-...      # or copy .env.example to .env
```

```bash
# Evaluate one model across all four tasks
MODEL_NAME=UniWorld-V1 ./run_evaluate_unison.sh

# Direct invocation
python evaluate_unison.py \
    --result-dir         ../Inference_Pipeline/result/UniWorld-V1 \
    --data-dir           ../data \
    --inference-base-dir ../Inference_Pipeline \
    --api-key            "$DASHSCOPE_API_KEY" \
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
| `api_type`   | Selects the inference backend: `bagel`, `janus`, `seed_x`, `tokenflow`, `uniworld`, `showo2`, `omnigen2`, `illume`, `ddit`. |
| `capabilities` | Modes the model supports: any of `understanding`, `generation`, `editing`. |
| `conda_env`  | Conda environment the model runs in (keeps incompatible deps isolated). |
| `inference_mode` | Default mode: `understanding`, `generation`, or `editing`. |
| `seed`       | Random seed (default 666). |

---

## Acknowledgements

TODO

---

## Citation
```bibtex
@inproceedings{Unison,
  title={{Unison}: Benchmarking Unified Multimodal Models via Synergistic Understanding and Generation},
  author={Jinyu Liu, Xincheng Shuai, Henghui Ding and Yu-Gang Jiang},
  booktitle={ICML},
  year={2026}
}
```

## License

Released under the [MIT License](LICENSE).

