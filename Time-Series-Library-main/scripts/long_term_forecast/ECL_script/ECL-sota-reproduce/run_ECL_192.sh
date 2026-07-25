#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Dataset: ECL
# Horizon: 192
# Verified test MSE/MAE: 0.1468018889, 0.2430038899
# TimeMixer++ target MSE/MAE: 0.1470000000, 0.2350000000
# Source run root: /home/c209/??/czw/project/TS-CAN-github/runs/extended_sota_r027_ecl_weather_archive_close_20260724
# Source log: /home/c209/??/czw/project/TS-CAN-github/runs/extended_sota_r027_ecl_weather_archive_close_20260724/logs/r027_0003_ECL_192.log
python -u run_can.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_512_192_r027_0003_ECL_192 --model CANPatchTST --data custom --features M --freq h --seq_len 512 --label_len 48 --pred_len 192 --enc_in 321 --dec_in 321 --c_out 321 --e_layers 2 --d_model 128 --d_ff 128 --patch_len 12 --can_stride 6 --can_shifts 1,2,4,8,16 --can_cli_mode inner --can_temporal_cli_mode full --can_ctx_mode diff --can_drop_path 0.03 --can_kernel_size 3 --can_use_gffng 1 --can_temporal_roll 1 --can_context_pyramid 0 --dropout 0.015 --batch_size 2 --learning_rate 0.00015 --lradj cosine --train_epochs 30 --patience 8 --num_workers 0 --itr 1 --seed 2 --des R011_r027_0003_ECL_192 --checkpoints ./checkpoints/ECL-sota-reproduce --results ./results/ECL-sota-reproduce --test_results ./test_results/ECL-sota-reproduce --use_amp
