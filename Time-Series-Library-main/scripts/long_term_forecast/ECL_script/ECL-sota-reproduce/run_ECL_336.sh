#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Dataset: ECL
# Horizon: 336
# Verified test MSE/MAE: 0.1639380008, 0.2656868100
# TimeMixer++ target MSE/MAE: 0.1640000000, 0.2450000000
# Source run root: /home/c209/??/czw/project/TS-CAN-github/runs/extended_sota_r065_ecl336_epoch3_longcos_20260725
# Source log: /home/c209/??/czw/project/TS-CAN-github/runs/extended_sota_r065_ecl336_epoch3_longcos_20260725/logs/r065_0019_ECL_336.log
python -u run_can.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_512_336_r065_0019_ECL_336 --model CANPatchTST --data custom --features M --freq h --seq_len 512 --label_len 48 --pred_len 336 --enc_in 321 --dec_in 321 --c_out 321 --e_layers 2 --d_model 128 --d_ff 128 --patch_len 12 --can_stride 6 --can_shifts 1,2,4,8,16 --can_cli_mode inner --can_temporal_cli_mode full --can_ctx_mode diff --can_drop_path 0.05 --can_kernel_size 3 --can_use_gffng 1 --can_temporal_roll 1 --can_context_pyramid 0 --dropout 0.05 --batch_size 2 --learning_rate 0.0002 --lradj cosine --train_epochs 36 --patience 9 --num_workers 0 --itr 1 --seed 2 --des R011_r065_0019_ECL_336 --checkpoints ./checkpoints/ECL-sota-reproduce --results ./results/ECL-sota-reproduce --test_results ./test_results/ECL-sota-reproduce --stop_after_epochs 3 --can_beta_init 0.48 --use_amp
