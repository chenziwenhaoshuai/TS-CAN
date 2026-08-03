#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSLIB_ROOT="${TSLIB_ROOT:-$(cd "$HERE/../../../.." && pwd)}"
PYTHON="${PYTHON:-python}"
GPU="${1:-0}"
OUT="$HERE/artifacts/reproduced/Daily"

mkdir -p "$OUT"
cd "$TSLIB_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

"$PYTHON" -u run.py \
  --task_name short_term_forecast --is_training 1 \
  --root_path ./dataset/m4 --model CANPatchTST --data m4 --features M \
  --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 \
  --can_cli_mode full --can_temporal_cli_mode full \
  --can_temporal_roll 1 --can_context_pyramid 1 --can_use_gffng 1 \
  --can_drop_path 0.0 --dropout 0.0 \
  --itr 1 --train_epochs 50 --patience 20 --loss MASE --num_workers 0 \
  --d_model 48 --d_ff 64 --e_layers 4 \
  --patch_len 7 --can_stride 1 --can_shifts 1,2,4,7 \
  --learning_rate 0.002 --batch_size 64 \
  --can_periodic_residual 1 --can_periods 7 \
  --can_periodic_alpha 0.015 --can_periodic_learnable 1 \
  --lradj cosine --seasonal_patterns Daily \
  --model_id m4_Daily_D40_mase_period7_lr002_reproduce \
  --des M4freq_D40_mase_period7_lr002_reproduce \
  2>&1 | tee "$OUT/train.log"

cp "$TSLIB_ROOT/m4_results/CANPatchTST/Daily_forecast.csv" "$OUT/Daily_forecast.csv"
"$PYTHON" "$HERE/evaluate_daily.py" "$OUT/Daily_forecast.csv" \
  --tslib-root "$TSLIB_ROOT" \
  --output "$OUT/metrics.json"
