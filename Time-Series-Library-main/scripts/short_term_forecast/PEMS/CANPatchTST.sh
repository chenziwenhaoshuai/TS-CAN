#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

model_name=CANPatchTST
seq_len=96
pred_len=12
learning_rate=0.003
d_model=128
d_ff=256
train_epochs=10
patience=10

run_pems() {
  local dataset="$1"
  local enc_in="$2"
  local batch_size="$3"
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path "${dataset}.npz" \
    --model_id "$dataset" \
    --model "$model_name" \
    --data PEMS \
    --features M \
    --seq_len "$seq_len" \
    --label_len 0 \
    --pred_len "$pred_len" \
    --e_layers 5 \
    --d_layers 1 \
    --factor 3 \
    --enc_in "$enc_in" \
    --dec_in "$enc_in" \
    --c_out "$enc_in" \
    --des CAN_short \
    --itr 1 \
    --use_norm 0 \
    --channel_independence 0 \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --patch_len 16 \
    --can_stride 8 \
    --can_shifts 1,2,4,8,16 \
    --can_cli_mode full \
    --can_temporal_cli_mode full \
    --can_temporal_roll 1 \
    --can_context_pyramid 1 \
    --can_use_gffng 1 \
    --can_drop_path 0.05 \
    --dropout 0.05 \
    --batch_size "$batch_size" \
    --learning_rate "$learning_rate" \
    --train_epochs "$train_epochs" \
    --patience "$patience" \
    --lradj type1 \
    --num_workers 0
}

run_pems PEMS03 358 4
run_pems PEMS04 307 4
run_pems PEMS07 883 2
run_pems PEMS08 170 8
