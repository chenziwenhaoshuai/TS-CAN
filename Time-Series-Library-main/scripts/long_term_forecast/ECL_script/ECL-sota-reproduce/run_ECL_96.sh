#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Dataset: ECL
# Horizon: 96
# Verified test MSE/MAE: 0.1345473081, 0.2305439264
# TimeMixer++ target MSE/MAE: 0.1350000000, 0.2220000000
# Source run root: /home/c209/??/czw/project/TS-CAN-github/runs/extended_sota_r017_ecl_20260724
# Source log: /home/c209/??/czw/project/TS-CAN-github/runs/extended_sota_r017_ecl_20260724/logs/r017_0023_ECL_96.log
python -u run_can.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_336_96_r017_0023_ECL_96 --model CANPatchTST --data custom --features M --freq h --seq_len 336 --label_len 0 --pred_len 96 --enc_in 321 --dec_in 321 --c_out 321 --e_layers 3 --d_model 32 --d_ff 64 --patch_len 16 --can_stride 8 --can_shifts 1,2,4,8,16 --can_cli_mode full --can_temporal_cli_mode full --can_ctx_mode diff --can_drop_path 0.02 --can_kernel_size 3 --can_use_gffng 1 --can_temporal_roll 1 --can_context_pyramid 1 --dropout 0.05 --batch_size 16 --learning_rate 0.0018 --lradj type1 --train_epochs 20 --patience 20 --num_workers 0 --itr 1 --seed 2 --des R011_r017_0023_ECL_96 --checkpoints ./checkpoints/ECL-sota-reproduce --results ./results/ECL-sota-reproduce --test_results ./test_results/ECL-sota-reproduce --can_linear_residual 1 --can_linear_mode decomp --can_linear_scale_init 0.5 --max_train_steps 2000 --use_amp
