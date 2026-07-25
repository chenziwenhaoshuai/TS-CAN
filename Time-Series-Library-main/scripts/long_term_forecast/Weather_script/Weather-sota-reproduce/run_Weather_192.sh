#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Dataset: Weather
# Horizon: 192
# Verified test MSE/MAE: 0.1961645037, 0.2454460561
# TimeMixer++ target MSE/MAE: 0.2010000000, 0.2450000000
# Source run root: /home/c209/??/czw/project/TS-CAN-github/runs/near_threshold_r004_20260723
# Source log: /home/c209/??/czw/project/TS-CAN-github/runs/near_threshold_r004_20260723/logs/w192_03.log
python -u run_can.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_336_192_w192_03 --model CANPatchTST --data custom --features M --freq t --seq_len 336 --label_len 48 --pred_len 192 --enc_in 21 --dec_in 21 --c_out 21 --e_layers 2 --d_model 128 --d_ff 192 --patch_len 16 --can_stride 8 --can_shifts 1,2,4,8,16 --can_cli_mode full --can_temporal_cli_mode full --can_ctx_mode diff --can_drop_path 0.05 --can_kernel_size 3 --can_use_gffng 1 --can_temporal_roll 1 --can_context_pyramid 1 --dropout 0.05 --batch_size 32 --learning_rate 0.0003 --lradj cosine --train_epochs 20 --patience 20 --max_train_steps 900 --num_workers 0 --itr 1 --seed 2 --des R004_w192_03 --checkpoints ./checkpoints/Weather-sota-reproduce --results ./results/Weather-sota-reproduce --test_results ./test_results/Weather-sota-reproduce --use_amp
