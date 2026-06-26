<div align="center">
<h1><img src="images/Unison-logo.png" alt="Unison" height="25"/>: Benchmarking Unified Multimodal Models via Synergistic Understanding and Generation</h3>
</div>

<p align="center"><b><a href="https://scholar.google.com/citations?user=oPK92GMAAAAJ&hl=en&authuser=1">Jinyu Liu</a>, <a href="https://scholar.google.com/citations?user=kLY6SUMAAAAJ&hl=en&oi=ao">Xincheng Shuai</a>, <a href="https://henghuiding.com">Henghui Ding</a>, <a href="https://scholar.google.com/citations?user=f3_FP8AAAAAJ&hl=en&oi=ao">Yu-Gang Jiang</a></b></p>
<p align="center">Fudan University</p>

<div align="center">
<a href='https://arxiv.org/abs/2606.26984'><img src='https://img.shields.io/badge/ICML 2026-Unison-b31b1b.svg?logo=arxiv&logoColor=B31B1B&labelColor=white'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://fudancvl.github.io/Unison'><img src='https://img.shields.io/badge/Website-Unison-orange?logo=googlechrome&logoColor=4285F4&labelColor=white'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/datasets/FudanCVL/Unison"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark-Unison-green?labelColor=white"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/FudanCVL/Unison-Judge"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Evaluator-Unison--Judge-blue?labelColor=white"></a> &nbsp;&nbsp;&nbsp;&nbsp;
</div>

## 📖 Overview

***TL;DR: Unison evaluates Unified Multimodal Models (UMMs) by leveraging the synergy between understanding and generation across four dimensions. Unison-Judge, the automatic evaluation model, achieves an 88.7% alignment with human judgments.***


<p align="center"><img src="images/overview.png" alt="Unison Overview" width="100%"/></p>

We introduce **Unison**, a comprehensive benchmark comprising 2,169 high-quality unified task samples, designed to evaluate joint understanding and generation in unified multimodal models. Unison offers three key strengths: **1) Comprehensive Dimensions**: Unison encompasses internal consistency, understanding-guided generation, generation-guided understanding, and mutual enhancement to enable holistic evaluation. **2) Diagnostic Evaluation**: it provides both unified and decoupled tracks for understanding and generation, allowing fine-grained attribution of failure modes and quantitative analysis of the gains from unified modeling. **3) Human Alignment**: we also train Unison-Judge, an evaluation model well aligned with human judgments to achieve reliable assessment.

## 🔥 Updates
- **[2026/06/25]** We release **Unison-Bench** and **Unison-Judge**.


## 📊 Evaluation Results

### Open-Source Unified Multimodal Models

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left" width="140">Model</th>
      <th rowspan="2" align="left">Params</th>
      <th colspan="3">Internal Consistency</th>
      <th colspan="3">Und.-Guided Gen.</th>
      <th colspan="3">Gen-Guided Und.</th>
      <th colspan="3">Mutual Enhancement</th>
      <th rowspan="2">Overall</th>
    </tr>
    <tr>
      <th>Und.</th><th>Gen.</th><th>Uni.</th>
      <th>Und.</th><th>Gen.</th><th>Uni.</th>
      <th>Und.</th><th>Gen.</th><th>Uni.</th>
      <th>Und.</th><th>Gen.</th><th>Uni.</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="left" nowrap><a href="https://github.com/showlab/Show-o">Show-o</a></td><td align="left">1.3B</td><td align="right">88.3</td><td align="right">64.7</td><td align="right">58.5</td><td align="right">8.90</td><td align="center">-</td><td align="center">-</td><td align="right">12.0</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td></tr>
    <tr><td align="left" nowrap><a href="https://github.com/deepseek-ai/Janus">Janus-Pro</a></td><td align="left">1.5B</td><td align="right">94.4</td><td align="right">47.1</td><td align="right">45.0</td><td align="right">0.3</td><td align="center">-</td><td align="center">-</td><td align="right">19.2</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td></tr>
    <tr><td align="left" nowrap><a href="https://github.com/showlab/Show-o/tree/main/show-o2">Show-o2</a></td><td align="left">1.5B</td><td align="right"><u>96.0</u></td><td align="right">67.9</td><td align="right">65.8</td><td align="right">26.7</td><td align="center">-</td><td align="center">-</td><td align="right">9.4</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td></tr>
    <tr><td align="left" nowrap><a href="https://github.com/zijieli-Jlee/Dual-Diffusion">D-DiT</a></td><td align="left">2B</td><td align="right">86.5</td><td align="right">65.0</td><td align="right">58.1</td><td align="right">0.2</td><td align="center">-</td><td align="center">-</td><td align="right">6.8</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td></tr>
    <tr><td align="left" nowrap><a href="https://github.com/illume-unified-mllm/ILLUME_plus">ILLUME+</a></td><td align="left">3B</td><td align="right">43.4</td><td align="right">19.9</td><td align="right">10.5</td><td align="right">10.3</td><td align="right">7.7</td><td align="right">9.0</td><td align="right">11.3</td><td align="right">30.1</td><td align="right">15.1</td><td align="right">1.0</td><td align="right">5.5</td><td align="right">3.2</td><td align="center">9.4</td></tr>
    <tr><td align="left" nowrap><a href="https://github.com/deepseek-ai/Janus">Janus-Pro</a></td><td align="left">7B</td><td align="right">95.7</td><td align="right">71.7</td><td align="right">69.8</td><td align="right">3.2</td><td align="center">-</td><td align="center">-</td><td align="right">15.1</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td></tr>
    <tr><td align="left" nowrap><a href="https://github.com/showlab/Show-o/tree/main/show-o2">Show-o2</a></td><td align="left">7B</td><td align="right"><strong>97.2</strong></td><td align="right">73.8</td><td align="right">72.5</td><td align="right">9.9</td><td align="center">-</td><td align="center">-</td><td align="right">9.2</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td></tr>
    <tr><td align="left" nowrap><a href="https://github.com/illume-unified-mllm/ILLUME_plus">ILLUME+</a></td><td align="left">7B</td><td align="right">80.2</td><td align="right">20.4</td><td align="right">16.7</td><td align="right">12.4</td><td align="right">10.4</td><td align="right">11.4</td><td align="right">11.3</td><td align="right">27.7</td><td align="right">13.9</td><td align="right">2.7</td><td align="right">6.8</td><td align="right">4.8</td><td align="center">11.7</td></tr>
    <tr><td align="left" nowrap><a href="https://github.com/VectorSpaceLab/OmniGen2">OmniGen2</a> 🥈</td><td align="left">7B</td><td align="right">92.3</td><td align="right"><u>79.0</u></td><td align="right"><u>74.5</u></td><td align="right"><u>61.3</u></td><td align="right"><u>42.6</u></td><td align="right"><u>52.0</u></td><td align="right">19.7</td><td align="right"><strong>41.9</strong></td><td align="right"><u>30.9</u></td><td align="right"><u>45.0</u></td><td align="right"><u>50.3</u></td><td align="right"><strong>47.7</strong></td><td align="center"><u>51.3</u></td></tr>
    <tr><td align="left" nowrap><a href="https://github.com/ByteVisionLab/TokenFlow">TokenFlow</a></td><td align="left">14B</td><td align="right">93.0</td><td align="right">47.1</td><td align="right">44.5</td><td align="right">20.1</td><td align="center">-</td><td align="center">-</td><td align="right">17.0</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td></tr>
    <tr><td align="left" nowrap><a href="https://github.com/ByteDance-Seed/Bagel">BAGEL</a> 🥇</td><td align="left">14B</td><td align="right"><u>96.0</u></td><td align="right"><strong>82.5</strong></td><td align="right"><strong>80.3</strong></td><td align="right">57.6</td><td align="right"><strong>78.1</strong></td><td align="right"><strong>67.9</strong></td><td align="right"><strong>28.2</strong></td><td align="right"><u>41.6</u></td><td align="right"><strong>32.0</strong></td><td align="right">7.2</td><td align="right"><strong>57.7</strong></td><td align="right"><u>32.5</u></td><td align="center"><strong>53.2</strong></td></tr>
    <tr><td align="left" nowrap><a href="https://github.com/AILab-CVC/SEED-X">SEED-X</a></td><td align="left">17B</td><td align="right">82.8</td><td align="right">38.9</td><td align="right">34.2</td><td align="right">18.6</td><td align="right">13.7</td><td align="right">16.1</td><td align="right">13.5</td><td align="right">27.4</td><td align="right">20.8</td><td align="right">0.2</td><td align="right">16.8</td><td align="right">8.5</td><td align="center">19.9</td></tr>
    <tr><td align="left" nowrap><a href="https://github.com/PKU-YuanGroup/UniWorld">UniWorld-V1</a> 🥉</td><td align="left">19B</td><td align="right">92.6</td><td align="right">68.5</td><td align="right">65.1</td><td align="right"><strong>63.4</strong></td><td align="right">26.4</td><td align="right">44.9</td><td align="right"><u>22.8</u></td><td align="right">32.0</td><td align="right">26.9</td><td align="right"><strong>46.4</strong></td><td align="right">16.2</td><td align="right">31.3</td><td align="center">42.1</td></tr>
  </tbody>
</table>

### Closed-Source Models

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left" width="140">Model</th>
      <th rowspan="2" align="left">Params</th>
      <th colspan="3">Internal Consistency</th>
      <th colspan="3">Und.-Guided Gen.</th>
      <th colspan="3">Gen-Guided Und.</th>
      <th colspan="3">Mutual Enhancement</th>
      <th rowspan="2">Overall</th>
    </tr>
    <tr>
      <th>Und.</th><th>Gen.</th><th>Uni.</th>
      <th>Und.</th><th>Gen.</th><th>Uni.</th>
      <th>Und.</th><th>Gen.</th><th>Uni.</th>
      <th>Und.</th><th>Gen.</th><th>Uni.</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="left" nowrap>Gemini 3 Pro</td><td align="left">-</td><td align="right">98.3</td><td align="right">88.1</td><td align="right">86.9</td><td align="right">71.0</td><td align="right">82.8</td><td align="right">76.9</td><td align="right">42.2</td><td align="right">46.5</td><td align="right">43.9</td><td align="right">65.3</td><td align="right">77.4</td><td align="right">71.4</td><td align="center">69.8</td></tr>
    <tr><td align="left" nowrap>GPT-5.2</td><td align="left">-</td><td align="right">98.6</td><td align="right">86.3</td><td align="right">84.7</td><td align="right">69.7</td><td align="right">85.7</td><td align="right">77.7</td><td align="right">44.8</td><td align="right">58.2</td><td align="right">52.7</td><td align="right">69.1</td><td align="right">71.2</td><td align="right">70.2</td><td align="center">71.3</td></tr>
  </tbody>
</table>

## 🛠️ Installation

There are two layers to install: a **base environment** (the orchestrator, judge, and data-handling code shared by both pipelines) and **one conda environment per model backend** (BAGEL, Janus, SEED-X, Show-o/Show-o2, TokenFlow, UniWorld, OmniGen2, ILLUME+, D-DiT). Each backend has heavy, mutually-incompatible dependencies, so each runs in its own env.

### Automated setup (recommended)

`Inference_Pipeline/setup_envs.sh` creates every conda environment and installs each backend's upstream code in one go. `UM` is the third-party **code** root — the same root `download_weights.sh` uses for **weights** — so repos land exactly where the configs expect them.

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
| `uniworld`  | `univa`     | `PKU-YuanGroup/UniWorld` → `UniWorld` |
| `illume`    | `illume`    | `illume-unified-mllm/ILLUME_plus` → `ILLUME_plus` |
| `ddit`      | `d-dit`     | `zijieli-Jlee/Dual-Diffusion` → `Dual-Diffusion` |

The script needs `conda` and `git`. It's idempotent — existing envs/clones are reused — and writes per-group logs to `Inference_Pipeline/setup_logs/`.

### Manual setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then install each backend from its upstream repository into a conda env named after that model's `conda_env`.

## 📦 Data

The pipelines expect a benchmark data directory, released separately as **Unison-data** on [HuggingFace](https://huggingface.co/datasets/FudanCVL/Unison).

**Where to put it:** download/unpack Unison-data to the repo root as `data/`. That is the default both launch scripts point at (`DATA_DIR=../data`), so no flags are needed:

```
Unison/
└── data/                       # <- put Unison-data here
    ├── Internal_Consistency/   # IC:  prompts.txt + questions.json
    ├── Und_Guided_Gen/         # UGG: UGG.csv (+ referenced images)
    ├── Gen_Guided_Und/         # GGU: 2D_Spatial/ 3D_Spatial/ Complex_Relation/
    ├── Mutual_Enhancement/     # ME:  ME.csv (+ referenced images)
    └── Judge_Consistency/      # judge-validation set: items.jsonl + images/
```

To keep it elsewhere, pass `--data-dir /path/to/Unison-data` or set the `DATA_DIR` env var in the launch scripts.

## ⚖️ Model Weights

### Benchmark model weights

Model configs in `Inference_Pipeline/config/*.json` reference local weight paths using the placeholder root `/path/to/Unified_Models/...`. Edit each config to point at your local checkout, e.g.:

```json
{
  "model_name": "UniWorld-V1",
  "model_path": "/path/to/Unified_Models/UniWorld/UniWorld-V1/model_weights/UniWorld-V1",
  "api_type": "uniworld",
  "conda_env": "univa",
  "capabilities": ["understanding", "generation", "editing"]
}
```

`download_weights.sh` fetches weights for all model backends. Set the local weight root and pick models:

```bash
UM=/data/Unified_Models ./download_weights.sh                 # everything
UM=/data/Unified_Models ./download_weights.sh bagel showo1    # selected groups
```

Gated repos (FLUX.1-dev, SD3) need `huggingface-cli login` + license acceptance. D-DiT has no public single-repo release — bring your own checkpoint. Run `setup_envs.sh` and `download_weights.sh` with the same `UM` so code and weights share one root.

### Judge weights (Unison-Judge)

The default evaluation backend runs a trained **Qwen3-VL-8B** judge, released separately as **Unison-Judge** on [HuggingFace](https://huggingface.co/FudanCVL/Unison-Judge).

**Where to put it:** download the checkpoint into `Evaluation_Pipeline/judge_model/`. That is the default path used by `evaluate_unison.py` and `run_evaluate_unison.sh`, so no flags are needed:

```
Unison/
└── Evaluation_Pipeline/
    └── judge_model/            # <- put Unison-Judge weights here
        ├── config.json
        ├── model-*.safetensors
        └── ...
```

To keep it elsewhere, set `LOCAL_JUDGE_MODEL=/path/to/judge` or pass `--local-model-path /path/to/judge`. No local judge weights are needed when using the `api` backend.

## 🚀 Inference and Evaluation

See [`Inference_Pipeline/README.md`](Inference_Pipeline/README.md) and [`Evaluation_Pipeline/README.md`](Evaluation_Pipeline/README.md) for the detailed guides.

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{liu2026unison,
  title     = {Unison: Benchmarking Unified Multimodal Models via Synergistic Understanding and Generation},
  author    = {Liu, Jinyu and Shuai, Xincheng and Ding, Henghui and Jiang, Yu-Gang},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## 📄 License

Released under the [MIT License](LICENSE).
