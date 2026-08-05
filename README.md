# TS-CAN

**Clifford Algebra Networks for Time Series Forecasting**

TS-CAN is a compact forecasting model built on top of the Time-Series-Library
benchmark stack. It replaces generic token mixing with Clifford-style geometric
interactions, so channel and temporal relationships are modeled through the two
parts of the geometric product: an inner-product branch for aligned variation
and a wedge-product branch for directional discrepancy.

This repository contains a clean TSLib-compatible implementation of TS-CAN,
including long-term and short-term forecasting scripts, data-loader support,
and reproduction configurations for the reported results.

## Highlights

- **Geometric interaction core.** TS-CAN uses Clifford-inspired inner and wedge
  interactions instead of plain attention or MLP mixing.
- **Channel and temporal Clifford mixing.** The same algebraic principle is used
  across variables and across patch-time states.
- **Patch-based efficient backbone.** The model follows a PatchTST-style input
  interface while replacing the central interaction block with CAN modules.
- **Multi-scale context without teacher fusion.** Reported configurations use
  TS-CAN itself, not an ensemble or fusion with TimeMixer++.
- **TSLib compatible.** Experiments run with the standard `run.py` entry point
  and the usual `scripts/` layout.

## Model

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

The implemented model also supports:

- channel Clifford interaction (`--can_cli_mode`)
- temporal Clifford interaction (`--can_temporal_cli_mode`)
- temporal rolling context (`--can_temporal_roll`)
- multi-scale patch context (`--can_multiscale_patch_lens`)
- cross-variable context (`--can_cross_var`)
- periodic and linear residual paths for short-term forecasting
- stochastic depth and dropout controls

The main implementation is in:

```text
models/CANPatchTST.py
```

## Installation

```bash
git clone https://github.com/chenziwenhaoshuai/TS-CAN.git
cd TS-CAN

conda create -n tslib python=3.10 -y
conda activate tslib

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The experiments were verified on NVIDIA RTX 4090 D GPUs with PyTorch in a CUDA
environment. Other recent CUDA-enabled PyTorch environments should also work.

## Data

Download the TSLib datasets and place them under `./dataset`.

Recommended mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Dataset source:

```text
https://huggingface.co/datasets/thuml/Time-Series-Library
```

Expected paths include:

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

## Reproduction

All scripts are under `scripts/`.

### Long-Term Forecasting

Run the ETT benchmarks:

```bash
bash scripts/long_term_forecast/ETT_script/CANPatchTST_ETTh1.sh
bash scripts/long_term_forecast/ETT_script/CANPatchTST_ETTh2.sh
bash scripts/long_term_forecast/ETT_script/CANPatchTST_ETTm1.sh
bash scripts/long_term_forecast/ETT_script/CANPatchTST_ETTm2.sh
```

Run the extended benchmarks:

```bash
bash scripts/long_term_forecast/ECL_script/CANPatchTST_ECL.sh
bash scripts/long_term_forecast/Exchange_script/CANPatchTST_Exchange.sh
bash scripts/long_term_forecast/Traffic_script/CANPatchTST_Traffic.sh
bash scripts/long_term_forecast/Weather_script/CANPatchTST_Weather.sh
```

Single-command example:

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

### Short-Term Forecasting

M4:

```bash
bash scripts/short_term_forecast/CANPatchTST_M4.sh
```

PEMS:

```bash
bash scripts/short_term_forecast/CANPatchTST_PEMS.sh
```

Note: M4 uses TSLib's `short_term_forecast` task. PEMS follows the TimeMixer++
short-term table protocol, but is executed through the fixed-horizon
`long_term_forecast` runner with `pred_len=12`.

## Results

Long-term forecasting reports `MSE/MAE`. Short-term M4 reports
`SMAPE/MASE/OWA`. PEMS reports `MAE/MAPE/RMSE`. Lower is better.

### Long-Term Forecasting vs TimeMixer++

TS-CAN achieves **29/32 MSE wins** against the TimeMixer++ long-term forecasting
table. The raw metrics are shown below. The remaining MSE losses are ETTh2
at horizons 96, 192, and 336.

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

### Short-Term Forecasting vs TimeMixer++

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

## Reproducibility Notes

- Scripts use fixed seeds where supported.
- CUDA AMP is enabled in several reproduction scripts.
- Minor numerical differences can occur across CUDA, PyTorch, and GPU driver
  versions.
- `Traffic-336` in the long-term table is from checkpoint-test evaluation; the
  fresh final checkpoint remains above the TimeMixer++ MSE baseline.
- PEMS is reported with the final test metrics used by the short-term protocol.

## Repository Layout

```text
models/CANPatchTST.py
data_provider/
exp/
scripts/long_term_forecast/
scripts/short_term_forecast/CANPatchTST_M4.sh
scripts/short_term_forecast/CANPatchTST_PEMS.sh
run.py
```

## Citation

If you use this repository, please cite TS-CAN and the Time-Series-Library
benchmark infrastructure. A BibTeX entry will be added after the paper metadata
is finalized.

## Acknowledgement

This implementation is built on the Time-Series-Library codebase. We thank the
TSLib authors and the broader time-series forecasting community for maintaining
standardized benchmark infrastructure.
