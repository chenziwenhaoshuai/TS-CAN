# TS-CAN

TS-CAN is a cleaned-up release of our Clifford-style geometric interaction model for long-term time series forecasting.
This repository focuses on the forecasting path we actually used in experiments:

- `CANPatchTST` as the only released model
- long-term forecasting only
- ETT and custom CSV datasets
- reproducible ETTh1/96 scripts
- geometric-option ablation utilities

The code is adapted from [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) and released under the same MIT-compatible workflow. This repository keeps only the pieces required to train and evaluate TS-CAN cleanly.

## Two Entry Points

This repo exposes two training paths that use **different model variants**:

| | `run.py` (standard) | `train.py` (autoresearch) |
|---|---|---|
| **Model file** | `models/CANPatchTST.py` | inlined in `train.py` |
| **Context norm** | `BatchNorm1d` | `LayerNorm1dChannels` |
| **d_model** | 128 | 96 |
| **batch_size** | 8 | 6 |
| **Training** | full 2 epochs | 300 s wall-clock budget |
| **`test_mse` (ETTh1/96)** | **0.360279** | 0.366498 |
| **Model source** | external file (production) | self-contained (agent-editable) |

**`run.py` → `models/CANPatchTST.py` is the canonical reference.** `train.py` is an inlined variant for autonomous experimentation; its results may drift slightly with PyTorch/CUDA versions.

## Reproducibility

Verified on **2026-05-23** with:

- **GPU:** NVIDIA GeForce GTX 1070 (Pascal, SM 6.1)
- **PyTorch:** 2.7.0+cu126
- **Conda env:** `pytorch`
- **OS:** Windows

### run.py (canonical)

```powershell
# Windows
conda activate pytorch
powershell -ExecutionPolicy Bypass -File .\scripts\ett\run_etth1_96_best.ps1

# Linux / macOS
bash scripts/ett/run_etth1_96_best.sh
```

**Result:** `mse=0.360279, mae=0.393127` (exact match with original release)

### train.py (autoresearch)

```powershell
conda activate pytorch
python prepare.py
python train.py
```

**Result:** `test_mse=0.369957` (within ~0.003 of historical best 0.366498; GTX 1070 bf16 software emulation introduces minor drift vs. original Ampere-class GPU)

## Recommended Baseline

The current ETTh1 / pred_len=96 release baseline is:

| Parameter | Value |
|---|---|
| `seq_len` | 192 |
| `label_len` | 48 |
| `pred_len` | 96 |
| `e_layers` | 2 |
| `d_model` | 128 |
| `d_ff` | 128 |
| `patch_len` | 16 |
| `can_stride` | 8 |
| `can_shifts` | 1,2,4,8,16 |
| `can_cli_mode` | full |
| `can_temporal_cli_mode` | full |
| `can_ctx_mode` | diff |
| `can_use_gffng` | 1 |
| `can_temporal_roll` | 1 |
| `can_use_orth` | 0 |
| `can_context_pyramid` | 0 |
| `dropout` | 0.05 |
| `can_drop_path` | 0.05 |
| `learning_rate` | 3e-4 |
| `train_epochs` | 2 |
| `patience` | 2 |
| `seed` | 2 |

**Expected output (verified):**

- `MSE = 0.360279`
- `MAE = 0.393127`

Enabling context pyramid (`can_context_pyramid=1`), widening `d_ff` (128 -> 192), and increasing `learning_rate` (3e-4 -> 5e-4) improves from the previous release baseline (`MSE=0.366418`) to the current best (`MSE=0.360279`) on ETTh1/96.

## SOTA Reproduction Scripts

The complete best-known pure CANPatchTST reproduction set is tracked in:

- `scripts/ETT-sota-reproduce/` for ETTh1, ETTh2, ETTm1, and ETTm2.
- `Time-Series-Library-main/scripts/long_term_forecast/*_script/*-sota-reproduce/` for Weather, Electricity, Traffic, and Exchange.
- `scripts/sota_reproduce_results.md` for the verified 32-cell MSE/MAE table.

These scripts use the canonical TSLib-compatible entry point,
`Time-Series-Library-main/run_can.py`, and the CANPatchTST implementation in
`models/CANPatchTST.py` / `Time-Series-Library-main/models/CANPatchTST.py`.
They were rerun from scratch on c209 on 2026-07-25. Evaluation is by test
MSE/MAE from `metrics.npy`; no checkpoint reuse is required.

Run all ETT cells:

```bash
bash scripts/ETT-sota-reproduce/run_ett_16_best_configs.sh
```

Run one extended dataset cell, for example Traffic-336:

```bash
bash Time-Series-Library-main/scripts/long_term_forecast/Traffic_script/Traffic-sota-reproduce/run_Traffic_336.sh
```

Against TimeMixer++ Table 16, the verified reproduction wins 29/32 MSE cells
and 13/32 MAE cells.

## Installation

```bash
conda create -n tscan python=3.11
conda activate tscan
pip install -r requirements.txt
```

PyTorch should match your CUDA runtime if you plan to train on GPU.

## Dataset Layout

Place the ETT files under:

```text
dataset/
  ETT/
    ETTh1.csv
    ETTh2.csv
    ETTm1.csv
    ETTm2.csv
```

## Quick Start

Run the current ETTh1/96 release baseline:

```bash
bash scripts/ett/run_etth1_96_best.sh
```

Or directly:

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT/ \
  --data_path ETTh1.csv \
  --model_id TS_CAN_ETTh1_96_release_best \
  --model CANPatchTST \
  --data ETTh1 \
  --features M \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --e_layers 2 \
  --d_model 128 \
  --d_ff 192 \
  --patch_len 16 \
  --can_stride 8 \
  --can_shifts 1,2,4,8,16 \
  --can_cli_mode full \
  --can_temporal_cli_mode full \
  --can_ctx_mode diff \
  --can_drop_path 0.05 \
  --can_kernel_size 3 \
  --can_use_gffng 1 \
  --can_temporal_roll 1 \
  --can_use_orth 0 \
  --can_context_pyramid 1 \
  --dropout 0.05 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --lradj cosine \
  --train_epochs 2 \
  --patience 2 \
  --des RELEASE_BEST \
  --itr 1 \
  --num_workers 0 \
  --use_amp \
  --seed 2
```

## Geometric Option Ablation

To continue the ETTh1/96 geometric ablation study around the new release baseline:

```bash
python scripts/ablation/etth1_96_geometric_options.py
```

This writes resumable outputs into `ablation_outputs/`.

## Autoresearch Mode

This repository also includes an `autoresearch`-style harness for autonomous
single-file experimentation:

- `prepare.py` is the fixed runtime/evaluation layer
- `train.py` is the single experiment file an agent edits
- `program.md` contains the default agent instructions

Typical setup:

```bash
conda activate pytorch
python prepare.py
python train.py
```

To reproduce the current best autoresearch configuration on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\autoresearch\run_best_autoresearch.ps1
```

This runs the exact current `train.py` best-known setup, verifies the runtime
with `prepare.py`, and writes a full log to `run_best.log`.

The default autoresearch target is bundled `ETT-small/ETTh1.csv`, and the main
metric is `val_mse` (lower is better). The full autoresearch baseline model is
defined inside `train.py`, so future agents can change both hyperparameters and
model structure without leaving that file. For a shorter smoke test you can
temporarily override the wall-clock budget, for example:

```powershell
$env:AUTORESEARCH_TIME_BUDGET_SECONDS=15
python train.py
```

## Repository Layout

```text
TS-CAN-github/
  run.py
  train.py
  prepare.py
  models/CANPatchTST.py
  exp/
  data_provider/
  layers/
  utils/
  scripts/
```

## License

This release inherits the MIT license from the upstream TSLib codebase. See [LICENSE](./LICENSE).
