#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Dataset: Weather
# Horizon: 720
# Verified test MSE/MAE: 0.3111478984, 0.3290944993
# TimeMixer++ target MSE/MAE: 0.3120000000, 0.3340000000
# Source run root: /home/c209/??/czw/project/TS-CAN-github/runs/extended_sota_r014_refine_20260723
# Source log: /home/c209/??/czw/project/TS-CAN-github/runs/extended_sota_r014_refine_20260723/logs/r014_0002_Weather_720.log
python -u run_can.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_720_720_r014_0002_Weather_720 --model CANPatchTST --data custom --features M --freq t --seq_len 720 --label_len 48 --pred_len 720 --enc_in 21 --dec_in 21 --c_out 21 --e_layers 2 --d_model 128 --d_ff 192 --patch_len 16 --can_stride 8 --can_shifts 1,2,4,8,16 --can_cli_mode full --can_temporal_cli_mode full --can_ctx_mode diff --can_drop_path 0.05 --can_kernel_size 3 --can_use_gffng 1 --can_temporal_roll 1 --can_context_pyramid 1 --dropout 0.05 --batch_size 16 --learning_rate 0.00015 --lradj cosine --train_epochs 20 --patience 20 --num_workers 0 --itr 1 --seed 2 --des R011_r014_0002_Weather_720 --checkpoints ./checkpoints/Weather-sota-reproduce --results ./results/Weather-sota-reproduce --test_results ./test_results/Weather-sota-reproduce --can_periodic_residual 1 --can_periods 24,48,96,168 --can_periodic_alpha 0.16 --max_train_steps 1500 --use_amp
