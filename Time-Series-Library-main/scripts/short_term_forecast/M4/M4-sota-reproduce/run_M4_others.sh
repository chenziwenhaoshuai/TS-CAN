#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSLIB_ROOT="${TSLIB_ROOT:-$(cd "$HERE/../../../.." && pwd)}"
PYTHON="${PYTHON:-python}"
GPU="${1:-0}"
OUT="$HERE/artifacts/reproduced/Others"

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
  --itr 1 --train_epochs 50 --patience 20 --loss SMAPE --num_workers 0 \
  --d_model 32 --d_ff 64 --e_layers 4 \
  --patch_len 2 --can_stride 1 --can_shifts 1,2 \
  --learning_rate 0.0025 --batch_size 128 \
  --lradj cosine --seasonal_patterns Weekly --seed 2 \
  --model_id m4_Weekly_W16_w13_smape_lr0025_reproduce \
  --des M4freq_W16_w13_smape_lr0025_reproduce \
  2>&1 | tee "$OUT/Weekly_train.log"

cp "$TSLIB_ROOT/m4_results/CANPatchTST/Weekly_forecast.csv" "$OUT/Weekly_forecast.csv"

"$PYTHON" -u run.py \
  --task_name short_term_forecast --is_training 1 \
  --root_path ./dataset/m4 --model CANPatchTST --data m4 --features M \
  --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 \
  --can_cli_mode full --can_temporal_cli_mode full \
  --can_temporal_roll 1 --can_context_pyramid 1 --can_use_gffng 1 \
  --can_drop_path 0.0 --dropout 0.0 \
  --itr 1 --train_epochs 50 --patience 20 --loss MASE --num_workers 0 \
  --d_model 32 --d_ff 64 --e_layers 4 \
  --patch_len 4 --can_stride 1 --can_shifts 1,2,4 \
  --learning_rate 0.0022 --batch_size 128 \
  --lradj cosine --seasonal_patterns Daily --seed 2 \
  --model_id m4_Daily_D39_d37_mase_lr0022_drop0_reproduce \
  --des M4freq_D39_d37_mase_lr0022_drop0_reproduce \
  2>&1 | tee "$OUT/Daily_train.log"

cp "$TSLIB_ROOT/m4_results/CANPatchTST/Daily_forecast.csv" "$OUT/Daily_forecast.csv"

"$PYTHON" -u run.py \
  --task_name short_term_forecast --is_training 1 \
  --root_path ./dataset/m4 --model CANPatchTST --data m4 --features M \
  --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 \
  --can_cli_mode full --can_temporal_cli_mode full \
  --can_temporal_roll 1 --can_context_pyramid 1 --can_use_gffng 1 \
  --can_drop_path 0.0 --dropout 0.0 \
  --itr 1 --train_epochs 50 --patience 20 --loss SMAPE --num_workers 0 \
  --d_model 64 --d_ff 64 --e_layers 4 \
  --patch_len 8 --can_stride 4 --can_shifts 1,2,4,8 \
  --learning_rate 0.00045 --batch_size 64 \
  --lradj cosine --seasonal_patterns Hourly --seed 2 \
  --model_id m4_Hourly_H28_h24_smape_lr00045_reproduce \
  --des M4freq_H28_h24_smape_lr00045_reproduce \
  2>&1 | tee "$OUT/Hourly_train.log"

cp "$TSLIB_ROOT/m4_results/CANPatchTST/Hourly_forecast.csv" "$OUT/Hourly_forecast.csv"

"$PYTHON" "$HERE/evaluate_others.py" \
  --forecast-dir "$OUT" \
  --tslib-root "$TSLIB_ROOT" \
  --output "$OUT/metrics.json"
