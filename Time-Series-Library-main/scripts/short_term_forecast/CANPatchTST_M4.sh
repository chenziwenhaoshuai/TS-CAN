#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

model_name=CANPatchTST

run_m4() {
  local pattern="$1"
  local d_ff="$2"
  python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns "$pattern" \
    --model_id "m4_${pattern}" \
    --model "$model_name" \
    --data m4 \
    --features M \
    --e_layers 4 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --batch_size 128 \
    --d_model 32 \
    --d_ff "$d_ff" \
    --patch_len 4 \
    --can_stride 2 \
    --can_shifts 1,2,4 \
    --can_cli_mode full \
    --can_temporal_cli_mode full \
    --can_temporal_roll 1 \
    --can_context_pyramid 1 \
    --can_use_gffng 1 \
    --can_drop_path 0.05 \
    --dropout 0.05 \
    --des CAN_short \
    --itr 1 \
    --learning_rate 0.01 \
    --train_epochs 50 \
    --patience 20 \
    --loss SMAPE \
    --num_workers 0
}

run_m4 Monthly 32
run_m4 Yearly 32
run_m4 Quarterly 64
run_m4 Daily 16
run_m4 Weekly 32
run_m4 Hourly 32
