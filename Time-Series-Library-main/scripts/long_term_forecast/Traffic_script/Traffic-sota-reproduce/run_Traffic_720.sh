#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Dataset: Traffic
# Horizon: 720
# Verified test MSE/MAE: 0.4270187020, 0.2918797731
# TimeMixer++ target MSE/MAE: 0.4410000000, 0.2820000000
# Source run root: /home/c209/??/czw/project/TS-CAN-github/runs/extended_sota_r026_traffic_archive_first_20260724
# Source log: /home/c209/??/czw/project/TS-CAN-github/runs/extended_sota_r026_traffic_archive_first_20260724/logs/r026_0006_Traffic_720.log
python -u run_can.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_512_720_r026_0006_Traffic_720 --model CANPatchTST --data custom --features M --freq h --seq_len 512 --label_len 48 --pred_len 720 --enc_in 862 --dec_in 862 --c_out 862 --e_layers 2 --d_model 128 --d_ff 128 --patch_len 16 --can_stride 8 --can_shifts 1,2,4,8,16 --can_cli_mode inner --can_temporal_cli_mode full --can_ctx_mode diff --can_drop_path 0.02 --can_kernel_size 3 --can_use_gffng 1 --can_temporal_roll 1 --can_context_pyramid 0 --dropout 0.02 --batch_size 2 --learning_rate 0.00015 --lradj cosine --train_epochs 40 --patience 10 --num_workers 0 --itr 1 --seed 2 --des R011_r026_0006_Traffic_720 --checkpoints ./checkpoints/Traffic-sota-reproduce --results ./results/Traffic-sota-reproduce --test_results ./test_results/Traffic-sota-reproduce --use_amp
