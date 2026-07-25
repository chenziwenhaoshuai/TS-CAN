#!/bin/bash
# TS-CAN ETTh2 fixed-structure best known configs.
# Test MSE verified on c209, 2026-07-21:
#   96:  seq_len=96,  lr=6.375e-4, bs=8, type1 -> 0.286007
#   192: seq_len=168, lr=4.875e-4, bs=8, type1 -> 0.352260
#   336: seq_len=240, lr=8.5e-5, bs=24, cosine, dropout=0.02, init=1e-3, orth=1 -> 0.379609
#   720: seq_len=252, lr=8e-5, bs=24, cosine, init=1e-3, orth=1 -> 0.403969

export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")/../.."

for pred_len in 96 192 336 720; do
  sl=96
  lr=0.0005
  bs=8
  lradj=type1
  epochs=8
  patience=4
  dropout=0.05
  drop_path=0.05
  init_values=1e-5
  orth=0

  if [ $pred_len -eq 96 ]; then lr=0.0006375; fi
  if [ $pred_len -eq 192 ]; then sl=168; lr=0.0004875; fi
  if [ $pred_len -eq 336 ]; then
    sl=240
    lr=0.000085
    bs=24
    lradj=cosine
    epochs=12
    patience=5
    dropout=0.02
    drop_path=0.02
    init_values=0.001
    orth=1
  fi
  if [ $pred_len -eq 720 ]; then
    sl=252
    lr=0.00008
    bs=24
    lradj=cosine
    epochs=12
    patience=5
    init_values=0.001
    orth=1
  fi

  python -u run_can.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_${sl}_${pred_len} \
    --model CANPatchTST \
    --data ETTh2 \
    --features M \
    --seq_len ${sl} \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 3 \
    --d_model 128 \
    --d_ff 256 \
    --patch_len 16 \
    --can_stride 8 \
    --can_shifts 1,2,4,8,16 \
    --can_cli_mode full \
    --can_temporal_cli_mode full \
    --can_ctx_mode diff \
    --can_drop_path ${drop_path} \
    --can_kernel_size 3 \
    --can_init_values ${init_values} \
    --can_use_gffng 1 \
    --can_temporal_roll 0 \
    --can_use_orth ${orth} \
    --can_context_pyramid 0 \
    --dropout ${dropout} \
    --batch_size ${bs} \
    --learning_rate ${lr} \
    --lradj ${lradj} \
    --train_epochs ${epochs} \
    --patience ${patience} \
    --des CAN_h2 \
    --itr 1 \
    --num_workers 0 \
    --use_amp \
    --seed 2
done
