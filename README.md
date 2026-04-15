# TS-CAN

TS-CAN is a cleaned-up release of our Clifford-style geometric interaction model for long-term time series forecasting.
This repository focuses on the forecasting path we actually used in experiments:

- `CANPatchTST` as the only released model
- long-term forecasting only
- ETT and custom CSV datasets
- reproducible ETTh1/96 scripts
- geometric-option ablation utilities

The code is adapted from [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) and released under the same MIT-compatible workflow. This repository keeps only the pieces required to train and evaluate TS-CAN cleanly.

## What Changed In This Release

Compared with the in-workspace research code, this release:

- removes unrelated models and tasks
- keeps only the forecasting pipeline needed by TS-CAN
- documents the current ETTh1/96 best setting as the recommended baseline
- includes a resumable geometric ablation script

## Recommended Baseline

The current ETTh1 / pred_len=96 release baseline is:

- `seq_len=192`
- `label_len=48`
- `pred_len=96`
- `e_layers=2`
- `d_model=128`
- `d_ff=128`
- `patch_len=16`
- `can_stride=8`
- `can_shifts=1,2,4,8,16`
- `can_cli_mode=full`
- `can_temporal_cli_mode=full`
- `can_ctx_mode=diff`
- `can_use_gffng=1`
- `can_temporal_roll=1`
- `can_use_orth=0`
- `can_context_pyramid=0`
- `dropout=0.05`
- `can_drop_path=0.05`
- `learning_rate=3e-4`
- `train_epochs=2`
- `patience=2`
- `seed=2`

In our local reproduction, this setting reached:

- `MSE=0.366418`
- `MAE=0.395711`

This slightly improved over the previous `can_cli_mode=inner` baseline (`MSE=0.367277`, `MAE=0.396796`) on ETTh1/96.

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
  --d_ff 128 \
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
  --can_context_pyramid 0 \
  --dropout 0.05 \
  --batch_size 8 \
  --learning_rate 0.00030 \
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
  models/CANPatchTST.py
  exp/
  data_provider/
  layers/
  utils/
  scripts/
```

## License

This release inherits the MIT license from the upstream TSLib codebase. See [LICENSE](./LICENSE).
