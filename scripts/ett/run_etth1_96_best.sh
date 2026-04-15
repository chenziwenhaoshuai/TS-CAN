#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT_PATH="${ROOT_PATH:-./dataset/ETT/}"
SEED="${SEED:-2}"

"${PYTHON_BIN}" -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "${ROOT_PATH}" \
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
  --seed "${SEED}"
