#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSLIB_ROOT="${TSLIB_ROOT:-$(cd "$HERE/../../../.." && pwd)}"
PYTHON="${PYTHON:-python}"
GPU="${1:-0}"
OUT="$HERE/artifacts/reproduced/Monthly"

mkdir -p "$OUT"
cd "$TSLIB_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

"$PYTHON" -u run.py \
  --task_name short_term_forecast --is_training 1 \
  --root_path ./dataset/m4 --data_path ETTh1.csv \
  --model_id m4_Monthly_reproduce --model CANPatchTST --data m4 \
  --features M --seasonal_patterns Monthly \
  --enc_in 1 --dec_in 1 --c_out 1 \
  --d_layers 1 --factor 3 \
  --use_norm 1 --channel_independence 1 \
  --d_model 64 --d_ff 128 --e_layers 4 \
  --patch_len 6 --can_stride 3 \
  --can_shifts 1,2,3,6,12 --can_temporal_shifts 1,2,3,6 \
  --can_cli_mode full --can_temporal_cli_mode full \
  --can_ctx_mode diff --can_temporal_roll 1 --can_temporal_circular 0 \
  --can_use_gffng 1 --can_global_cli_mode inner --can_global_ctx_mode abs \
  --can_context_pyramid 1 --can_use_ffn 0 \
  --can_drop_path 0.001 --can_drop_path_schedule linear \
  --can_kernel_size 3 --can_init_values 0.00001 --can_beta_init 0.5 \
  --can_linear_residual 1 --can_linear_mode raw \
  --can_linear_individual 0 --can_linear_scale_init 0.003 \
  --can_periodic_residual 0 --can_cross_var 0 --can_var_attn 0 \
  --can_var_embed 0 --dropout 0.006 \
  --batch_size 128 --learning_rate 0.0038 --lradj cosine \
  --loss SMAPE --train_epochs 50 --patience 20 \
  --seed 2 --itr 1 --num_workers 0 --gpu 0 \
  --des M4freq_Monthly_reproduce \
  2>&1 | tee "$OUT/train.log"

cp m4_results/CANPatchTST/Monthly_forecast.csv "$OUT/Monthly_forecast.csv"
"$PYTHON" "$HERE/evaluate_monthly.py" "$OUT/Monthly_forecast.csv" \
  --tslib-root "$TSLIB_ROOT" \
  --output "$OUT/metrics.json"
