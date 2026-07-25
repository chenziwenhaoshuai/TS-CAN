#!/bin/bash
# TS-CAN ETTm1: seq_len=192, e2, dff=192, roll=1, pyramid=1, lr=5e-4 cosine, freq=t
# Results: 96=0.299, 192=0.336, 336=0.377, 720=0.429

export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")/../.."

for pred_len in 96 192 336 720; do
  python -u run_can.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_${pred_len} \
    --model CANPatchTST \
    --data ETTm1 \
    --features M \
    --seq_len 192 \
    --label_len 48 \
    --pred_len ${pred_len} \
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
    --freq t \
    --batch_size 8 \
    --learning_rate 0.0005 \
    --lradj cosine \
    --train_epochs 5 \
    --patience 3 \
    --des CAN_m1 \
    --itr 1 \
    --num_workers 0 \
    --use_amp \
    --seed 2
done
