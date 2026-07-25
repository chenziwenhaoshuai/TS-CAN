#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Dataset: Weather
# Horizon: 336
# Verified test MSE/MAE: 0.2369910330, 0.2745282650
# TimeMixer++ target MSE/MAE: 0.2370000000, 0.2650000000
# Source run root: /home/c209/??/czw/project/TS-CAN-github/runs/extended_sota_r059_weather336_cv2_micro_20260725
# Source log: /home/c209/??/czw/project/TS-CAN-github/runs/extended_sota_r059_weather336_cv2_micro_20260725/logs/r059_0006_Weather_336.log
python -u run_can.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_656_336_r059_0006_Weather_336 --model CANPatchTST --data custom --features M --freq t --seq_len 656 --label_len 48 --pred_len 336 --enc_in 21 --dec_in 21 --c_out 21 --e_layers 2 --d_model 128 --d_ff 128 --patch_len 16 --can_stride 8 --can_shifts 1,2,4,8,16 --can_cli_mode inner --can_temporal_cli_mode full --can_ctx_mode diff --can_drop_path 0.05 --can_kernel_size 3 --can_use_gffng 1 --can_temporal_roll 1 --can_context_pyramid 0 --dropout 0.05 --batch_size 8 --learning_rate 9.2e-05 --lradj cosine --train_epochs 20 --patience 20 --num_workers 0 --itr 1 --seed 2 --des R011_r059_0006_Weather_336 --checkpoints ./checkpoints/Weather-sota-reproduce --results ./results/Weather-sota-reproduce --test_results ./test_results/Weather-sota-reproduce --can_cross_var 1 --can_cross_var_layers 2 --can_cross_var_context others_mean --can_cross_var_shifts 1,2,4 --can_periodic_residual 1 --can_periods 24,48,96,168 --can_periodic_alpha 0.18 --max_train_steps 4625 --weight_averaging ema --ema_decay 0.997 --ema_start_epoch 1 --use_amp
