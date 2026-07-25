#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Dataset: Exchange
# Horizon: 720
# Verified test MSE/MAE: 0.7915571332, 0.6661531925
# TimeMixer++ target MSE/MAE: 0.8510000000, 0.6890000000
# Source run root: /home/c209/??/czw/project/TS-CAN-github/runs/extended_sota_r003_scout_20260723
# Source log: /home/c209/??/czw/project/TS-CAN-github/runs/extended_sota_r003_scout_20260723/logs/r001_0025_Exchange_720.log
python -u run_can.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model_id Exchange_192_720_r001_0025_Exchange_720 --model CANPatchTST --data custom --features M --freq d --seq_len 192 --label_len 48 --pred_len 720 --enc_in 8 --dec_in 8 --c_out 8 --e_layers 2 --d_model 128 --d_ff 192 --patch_len 16 --can_stride 8 --can_shifts 1,2,4,8,16 --can_cli_mode full --can_temporal_cli_mode full --can_ctx_mode diff --can_drop_path 0.05 --can_kernel_size 3 --can_use_gffng 1 --can_temporal_roll 1 --can_context_pyramid 1 --dropout 0.05 --batch_size 8 --learning_rate 0.0005 --lradj cosine --train_epochs 20 --patience 20 --num_workers 0 --itr 1 --seed 2 --des EXTSOTA_r001_0025_Exchange_720 --checkpoints ./checkpoints/Exchange-sota-reproduce --results ./results/Exchange-sota-reproduce --test_results ./test_results/Exchange-sota-reproduce --use_amp --max_train_steps 700
