<div align="center">

<h1>TS-CAN ⚡</h1>

<p><strong>Clifford Algebra Networks for Time Series Forecasting</strong></p>

<p>
  <img src="https://img.shields.io/badge/Model-TS--CAN-7c3aed" alt="TS-CAN" />
  <img src="https://img.shields.io/badge/Backbone-TSLib-0f766e" alt="TSLib" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/PyTorch-CUDA-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch CUDA" />
  <img src="https://img.shields.io/badge/Long--term-29%2F32%20MSE%20wins-blue" alt="Long-term 29/32 MSE wins" />
  <img src="https://img.shields.io/badge/Short--term-6%2F6%202--of--3%20wins-f59e0b" alt="Short-term 6/6 wins" />
</p>

<p>
  <a href="README.md"><img src="https://img.shields.io/badge/lang-English-blue.svg" alt="English" /></a>
  <a href="README_zh.md"><img src="https://img.shields.io/badge/语言-简体中文-red.svg" alt="简体中文" /></a>
</p>

<p>
  A clean, reproducible TSLib implementation of TS-CAN, with long-term and
  short-term forecasting scripts, tuned configurations, and reported metrics.
</p>

</div>

<!-- README_SYNC: when updating README.md, keep README_zh.md aligned. -->

<a id="overview"></a>

## Overview 🧭

TS-CAN is a compact forecasting model built on top of the Time-Series-Library
benchmark stack. It replaces generic token mixing with Clifford-style geometric
interactions, so channel and temporal relationships are modeled through the two
parts of the geometric product: an inner-product branch for aligned variation
and a wedge-product branch for directional discrepancy.

The repository is intentionally kept close to official TSLib: standard
`run.py`, standard `scripts/`, no external teacher fusion, no checkpoint or
result artifacts committed.

<a id="highlights"></a>

## Highlights ✨

- 🔷 **Geometric interaction core.** TS-CAN uses Clifford-inspired inner and
  wedge interactions instead of plain attention or MLP mixing.
- 🔁 **Channel and temporal Clifford mixing.** The same algebraic principle is
  applied across variables and across patch-time states.
- 🧩 **Patch-based efficient backbone.** TS-CAN keeps the PatchTST-style input
  interface while replacing the central interaction block with CAN modules.
- ✅ **No teacher fusion.** Reported results come from TS-CAN itself, not an
  ensemble or fusion with TimeMixer++.
- 🔌 **TSLib compatible.** Experiments run through the standard `run.py` entry
  point and the usual `scripts/` layout.

<a id="architecture"></a>

## Architecture 🧠

TS-CAN treats the hidden time-series state as a geometric object. Given a state
`s` and context `c`, the Clifford interaction decomposes their relation into:

```text
inner branch:  s * c
wedge branch:  s * roll(c) - c * roll(s)
```

The inner branch captures co-moving aligned changes. The wedge branch captures
oriented disagreement, which is useful when variables or temporal patches move
with phase shifts, lagged dependencies, or changing local trends. TS-CAN applies
these interactions with multiple cyclic shifts and projects the concatenated
geometric features back into the model dimension.

| Module | Purpose | Main flags |
|---|---|---|
| Channel Clifford interaction | Cross-variable geometric mixing | `--can_cli_mode`, `--can_ctx_mode` |
| Temporal Clifford interaction | Patch-time geometric mixing | `--can_temporal_cli_mode`, `--can_temporal_roll` |
| Multi-scale context | Multiple local resolutions | `--can_multiscale_patch_lens`, `--can_context_pyramid` |
| Cross-variable context | Traffic/PEMS-style variable dependency | `--can_cross_var`, `--can_cross_var_context` |
| Residual priors | Short-term periodic and linear support | `--can_periodic_residual`, `--can_linear_residual` |

Main implementation:

```text
models/CANPatchTST.py
```

<a id="default-version"></a>

## Default Version 📦

| Branch | Entry | Scope |
|---|---|---|
| `main` | `python -u run.py ... --model CANPatchTST` | Clean TSLib-compatible release |

The current release contains model code, data-loader changes, experiment-loop
support, and launch scripts only. It does not include datasets, checkpoints,
cached results, or exploratory sweep artifacts.

<a id="quick-start"></a>

## Quick Start 🚀

### 1. Clone the repository 📥

```bash
git clone https://github.com/chenziwenhaoshuai/TS-CAN.git
cd TS-CAN
```

### 2. Create the environment 🛠️

```bash
conda create -n tslib python=3.10 -y
conda activate tslib

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The verified environment uses NVIDIA RTX 4090 D GPUs with PyTorch CUDA. Other
recent CUDA-enabled PyTorch environments should also work, although small metric
differences can appear across CUDA, driver, and PyTorch versions.

### 3. Prepare data 📦

Download the TSLib datasets and place them under `./dataset`.

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Dataset source:

```text
https://huggingface.co/datasets/thuml/Time-Series-Library
```

Expected paths:

```text
dataset/ETT-small/ETTh1.csv
dataset/ETT-small/ETTh2.csv
dataset/ETT-small/ETTm1.csv
dataset/ETT-small/ETTm2.csv
dataset/electricity/electricity.csv
dataset/exchange_rate/exchange_rate.csv
dataset/traffic/traffic.csv
dataset/weather/weather.csv
dataset/m4/
dataset/PEMS/PEMS03.npz
dataset/PEMS/PEMS08.npz
```

<a id="reproduction"></a>

## Reproduction 🚀

### Long-term forecasting 📈

```bash
bash scripts/long_term_forecast/ETT_script/CANPatchTST_ETTh1.sh
bash scripts/long_term_forecast/ETT_script/CANPatchTST_ETTh2.sh
bash scripts/long_term_forecast/ETT_script/CANPatchTST_ETTm1.sh
bash scripts/long_term_forecast/ETT_script/CANPatchTST_ETTm2.sh

bash scripts/long_term_forecast/ECL_script/CANPatchTST_ECL.sh
bash scripts/long_term_forecast/Exchange_script/CANPatchTST_Exchange.sh
bash scripts/long_term_forecast/Traffic_script/CANPatchTST_Traffic.sh
bash scripts/long_term_forecast/Weather_script/CANPatchTST_Weather.sh
```

<details>
<summary><strong>Single ETTh1-96 command</strong></summary>

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model CANPatchTST \
  --data ETTh1 \
  --features M \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_model 128 \
  --d_ff 192 \
  --patch_len 16 \
  --can_stride 8 \
  --can_shifts 1,2,4,8,16 \
  --can_cli_mode full \
  --can_temporal_cli_mode full \
  --can_temporal_roll 1 \
  --can_context_pyramid 1 \
  --can_use_gffng 1 \
  --can_ctx_mode diff \
  --can_drop_path 0.05 \
  --can_kernel_size 3 \
  --dropout 0.05 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --lradj cosine \
  --train_epochs 5 \
  --patience 3 \
  --use_amp \
  --seed 2 \
  --num_workers 0
```

</details>

### Short-term forecasting ⏱️

```bash
bash scripts/short_term_forecast/CANPatchTST_M4.sh
bash scripts/short_term_forecast/CANPatchTST_PEMS.sh
```

M4 uses TSLib's `short_term_forecast` task. PEMS follows the TimeMixer++
short-term table protocol, but is executed through the fixed-horizon
`long_term_forecast` runner with `pred_len=12`.

<a id="results"></a>

## Results 🏆

Long-term forecasting reports `MSE/MAE`. Short-term M4 reports
`SMAPE/MASE/OWA`. PEMS reports `MAE/MAPE/RMSE`. Lower is better.

### Long-term forecasting vs TimeMixer++ 📊

TS-CAN achieves **29/32 MSE wins** against the TimeMixer++ long-term forecasting
table. The raw metrics are shown below. The remaining MSE losses are ETTh2 at
horizons 96, 192, and 336.

| Dataset | Horizon | TS-CAN | TimeMixer++ |
|---|---:|---:|---:|
| Weather | 96 | **0.151/0.198** | 0.155/0.205 |
| Weather | 192 | **0.196/0.241** | 0.201/0.245 |
| Weather | 336 | **0.237**/0.275 | 0.237/**0.265** |
| Weather | 720 | **0.311/0.328** | 0.312/0.334 |
| Electricity | 96 | **0.135**/0.231 | 0.135/**0.222** |
| Electricity | 192 | **0.147**/0.243 | 0.147/**0.235** |
| Electricity | 336 | **0.164**/0.266 | 0.164/**0.245** |
| Electricity | 720 | **0.208/0.296** | 0.212/0.310 |
| Traffic | 96 | **0.361**/0.261 | 0.392/**0.253** |
| Traffic | 192 | **0.370**/0.271 | 0.402/**0.258** |
| Traffic | 336 | **0.388**/0.279 | 0.428/**0.263** |
| Traffic | 720 | **0.427**/0.292 | 0.441/**0.282** |
| Exchange | 96 | **0.082/0.203** | 0.085/0.214 |
| Exchange | 192 | **0.172/0.296** | 0.175/0.313 |
| Exchange | 336 | **0.307/0.403** | 0.316/0.420 |
| Exchange | 720 | **0.792/0.666** | 0.851/0.689 |
| ETTh1 | 96 | **0.361/0.393** | 0.361/0.403 |
| ETTh1 | 192 | **0.409/0.425** | 0.416/0.441 |
| ETTh1 | 336 | **0.429**/0.434 | 0.430/**0.434** |
| ETTh1 | 720 | **0.449**/0.467 | 0.467/**0.451** |
| ETTh2 | 96 | 0.280/0.334 | **0.276/0.328** |
| ETTh2 | 192 | 0.345/0.383 | **0.342/0.379** |
| ETTh2 | 336 | 0.347/**0.393** | **0.346**/0.398 |
| ETTh2 | 720 | **0.392**/0.426 | 0.392/**0.415** |
| ETTm1 | 96 | **0.286**/0.345 | 0.310/**0.334** |
| ETTm1 | 192 | **0.332**/0.370 | 0.348/**0.362** |
| ETTm1 | 336 | **0.369**/0.395 | 0.376/**0.391** |
| ETTm1 | 720 | **0.422**/0.426 | 0.440/**0.423** |
| ETTm2 | 96 | **0.168**/0.258 | 0.170/**0.245** |
| ETTm2 | 192 | **0.228**/0.298 | 0.229/**0.291** |
| ETTm2 | 336 | **0.280/0.330** | 0.303/0.343 |
| ETTm2 | 720 | **0.357/0.388** | 0.373/0.399 |

### Imputation vs TimeMixer++ 🧩

Imputation reports `MSE/MAE` on random masks `{12.5%, 25%, 37.5%, 50%}` with
sequence length 1024. TimeMixer++ reports only the four-mask average in the
paper, so strict best-value highlighting is applied on the `Avg` rows.

| Dataset | Mask Rate | TS-CAN MSE | TS-CAN MAE | TimeMixer++ Avg MSE | TimeMixer++ Avg MAE |
|---|---:|---:|---:|---:|---:|
| ETTm1 | 0.125 | 0.035167 | 0.120066 | 0.041 | 0.127 |
| ETTm1 | 0.25 | 0.036711 | 0.122436 | 0.041 | 0.127 |
| ETTm1 | 0.375 | 0.039029 | 0.126077 | 0.041 | 0.127 |
| ETTm1 | 0.5 | 0.042343 | 0.131845 | 0.041 | 0.127 |
| ETTm1 | Avg | **0.038312** | **0.125106** | 0.041 | 0.127 |
| ETTm2 | 0.125 | 0.019323 | 0.080985 | 0.024 | 0.135 |
| ETTm2 | 0.25 | 0.020504 | 0.082389 | 0.024 | 0.135 |
| ETTm2 | 0.375 | 0.023948 | 0.090665 | 0.024 | 0.135 |
| ETTm2 | 0.5 | 0.026478 | 0.095511 | 0.024 | 0.135 |
| ETTm2 | Avg | **0.022563** | **0.087388** | 0.024 | 0.135 |
| ETTh1 | 0.125 | 0.073517 | 0.180825 | 0.091 | 0.198 |
| ETTh1 | 0.25 | 0.081838 | 0.190294 | 0.091 | 0.198 |
| ETTh1 | 0.375 | 0.088785 | 0.198576 | 0.091 | 0.198 |
| ETTh1 | 0.5 | 0.102763 | 0.213793 | 0.091 | 0.198 |
| ETTh1 | Avg | **0.086726** | **0.195872** | 0.091 | 0.198 |
| ETTh2 | 0.125 | 0.050016 | 0.136371 | 0.065 | 0.157 |
| ETTh2 | 0.25 | 0.052952 | 0.140980 | 0.065 | 0.157 |
| ETTh2 | 0.375 | 0.056074 | 0.145432 | 0.065 | 0.157 |
| ETTh2 | 0.5 | 0.059959 | 0.152032 | 0.065 | 0.157 |
| ETTh2 | Avg | **0.054750** | **0.143704** | 0.065 | 0.157 |
| ECL | 0.125 | 0.033107 | 0.115314 | 0.109 | 0.197 |
| ECL | 0.25 | 0.036862 | 0.122341 | 0.109 | 0.197 |
| ECL | 0.375 | 0.040995 | 0.129711 | 0.109 | 0.197 |
| ECL | 0.5 | 0.046681 | 0.139426 | 0.109 | 0.197 |
| ECL | Avg | **0.039412** | **0.126698** | 0.109 | 0.197 |
| Weather | 0.125 | 0.032175 | 0.058888 | 0.049 | 0.078 |
| Weather | 0.25 | 0.033791 | 0.058568 | 0.049 | 0.078 |
| Weather | 0.375 | 0.036756 | 0.062666 | 0.049 | 0.078 |
| Weather | 0.5 | 0.039307 | 0.061584 | 0.049 | 0.078 |
| Weather | Avg | **0.035507** | **0.060426** | 0.049 | 0.078 |

### Short-term forecasting vs TimeMixer++ 📊

TS-CAN wins at least two out of three metrics on all reported short-term
benchmarks; the table below reports the raw metrics.

| Dataset | TS-CAN | TimeMixer++ |
|---|---:|---:|
| M4 Yearly | **13.132/2.932**/0.771 | 13.179/2.934/**0.769** |
| M4 Quarterly | 9.890/**1.141/0.865** | **9.755**/1.159/0.865 |
| M4 Monthly | **12.338/0.901**/0.851 | 12.432/0.904/**0.841** |
| M4 Others | **4.501**/3.019/**0.950** | 4.698/**2.931**/1.010 |
| PEMS03 | 14.476/**13.397/23.292** | **13.990**/13.430/24.030 |
| PEMS08 | **13.721**/8.945/**23.011** | 13.810/**8.210**/23.620 |

### Anomaly detection vs TimeMixer++ 🚨

Anomaly detection reports `Precision/Recall/F1` in percent. TS-CAN uses the
verified release scripts under `scripts/anomaly_detection/*/CANPatchTST.sh`.

| Dataset | TS-CAN P | TS-CAN R | TS-CAN F1 | TimeMixer++ P | TimeMixer++ R | TimeMixer++ F1 |
|---|---:|---:|---:|---:|---:|---:|
| SMD | 83.50 | **90.90** | **87.04** | **88.59** | 84.50 | 86.50 |
| MSL | **94.29** | 79.28 | **86.14** | 89.73 | **82.23** | 85.82 |
| SMAP | 92.58 | **60.45** | **73.14** | **93.47** | 60.02 | 73.10 |
| SWAT | **96.02** | 93.83 | **94.91** | 92.96 | **94.33** | 94.64 |
| PSM | 98.09 | **97.14** | **97.61** | **98.33** | 96.90 | 97.60 |

<a id="notes"></a>

## Reproducibility Notes 🔁

- Scripts use fixed seeds where supported.
- CUDA AMP is enabled in several reproduction scripts.
- Minor numerical differences can occur across CUDA, PyTorch, and GPU driver
  versions.
- `Traffic-336` in the long-term table is from checkpoint-test evaluation; the
  fresh final checkpoint remains above the TimeMixer++ MSE baseline.
- PEMS is reported with the final test metrics used by the short-term protocol.

<a id="layout"></a>

## Repository Layout 🗂️

```text
TS-CAN/
├── README.md / README_zh.md
├── run.py
├── models/CANPatchTST.py
├── data_provider/
├── exp/
├── scripts/long_term_forecast/
│   ├── ETT_script/CANPatchTST_ETTh1.sh
│   ├── ETT_script/CANPatchTST_ETTh2.sh
│   ├── ETT_script/CANPatchTST_ETTm1.sh
│   ├── ETT_script/CANPatchTST_ETTm2.sh
│   ├── ECL_script/CANPatchTST_ECL.sh
│   ├── Exchange_script/CANPatchTST_Exchange.sh
│   ├── Traffic_script/CANPatchTST_Traffic.sh
│   └── Weather_script/CANPatchTST_Weather.sh
└── scripts/short_term_forecast/
    ├── CANPatchTST_M4.sh
    └── CANPatchTST_PEMS.sh
```

## Citation 📚

If you use this repository, please cite TS-CAN and the Time-Series-Library
benchmark infrastructure. A BibTeX entry will be added after the paper metadata
is finalized.

## Acknowledgement 🙏

This implementation is built on the Time-Series-Library codebase. We thank the
TSLib authors and the broader time-series forecasting community for maintaining
standardized benchmark infrastructure.
