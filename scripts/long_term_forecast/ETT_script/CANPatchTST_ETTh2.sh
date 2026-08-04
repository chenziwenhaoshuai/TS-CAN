#!/usr/bin/env bash
set -euo pipefail

# TS-CAN ETTh2: horizon-specific reproduced best configs.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

run_cell() {
  local pred_len="$1"
  shift
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model CANPatchTST \
    --data ETTh2 \
    --features M \
    --label_len 48 \
    --pred_len "${pred_len}" \
    --can_shifts 1,2,4,8,16 \
    --can_ctx_mode diff \
    --can_kernel_size 3 \
    --can_init_values 1e-5 \
    --can_beta_init 0.5 \
    --can_gamma_lr_scale 1.0 \
    --can_gamma_weight_decay 0.0 \
    --can_drop_path_schedule linear \
    --can_use_gffng 1 \
    --can_temporal_roll 1 \
    --can_use_orth 0 \
    --can_context_pyramid 0 \
    --optimizer adam \
    --weight_decay 0.0 \
    --warmup_epochs 1 \
    --weight_averaging none \
    --ema_decay 0.995 \
    --ema_start_epoch 1 \
    --loss_variable_weights "" \
    --loss_horizon_weight_start 0 \
    --loss_horizon_weight 1.0 \
    --loss_horizon_weight_mode step \
    --loss_volatility_weight 0.0 \
    --loss_level_weight 0.0 \
    --loss_variance_weight 0.0 \
    --loss_range_weight 0.0 \
    --loss_tail_bias_weight 0.0 \
    --loss_tail_bias_start 0 \
    --loss_tail_bias_variables "" \
    --loss_tail_level_weight 0.0 \
    --loss_tail_level_start 0 \
    --loss_tail_level_variables "" \
    --loss_tail_hard_weight 0.0 \
    --loss_tail_hard_start 0 \
    --loss_tail_hard_power 1.0 \
    --loss_tail_hard_clip 3.0 \
    --loss_tail_lowpass_weight 0.0 \
    --loss_tail_lowpass_start 0 \
    --loss_tail_lowpass_kernel 9 \
    --loss_tail_lowpass_variables "" \
    --vali_metric_mode all \
    --vali_metric_horizon_start 0 \
    --itr 1 \
    --num_workers 0 \
    --use_amp \
    --seed 2 \
    "$@"

}

for pred_len in 96 192 336 720; do
  case "${pred_len}" in
    96) run_cell 96 --model_id ETTh2_96_r275_0000 --des ETTh2_h2_best_r275_0000 --seq_len 112 --e_layers 3 --d_model 128 --d_ff 256 --patch_len 16 --can_stride 8 --can_cli_mode full --can_temporal_cli_mode full --can_temporal_roll 0 --can_context_pyramid 0 --can_init_values 1e-5 --can_beta_init 0.40 --can_drop_path 0.0475 --can_use_orth 0 --dropout 0.050 --batch_size 8 --learning_rate 0.000695 --optimizer adam --weight_decay 0.0 --weight_averaging ema --ema_decay 0.99770 --ema_start_epoch 2 --loss_variable_weights 1.28,0.94,0.84,1.48,1.24,0.58,0.84 --loss_tail_bias_weight 0.002 --loss_tail_bias_start 48 --lradj type1 --train_epochs 10 --patience 5 --max_train_steps 1480 ;;
    192) run_cell 192 --model_id ETTh2_192_r305_0000 --des ETTh2_h2_best_r305_0000 --seq_len 176 --e_layers 3 --d_model 160 --d_ff 304 --patch_len 16 --can_stride 8 --can_cli_mode full --can_temporal_cli_mode full --can_temporal_roll 0 --can_context_pyramid 0 --can_init_values 1e-5 --can_beta_init 0.50 --can_drop_path 0.058 --can_use_orth 0 --dropout 0.058 --batch_size 8 --learning_rate 0.000339 --optimizer adam --weight_decay 0.0 --weight_averaging ema --ema_decay 0.99716 --ema_start_epoch 2 --loss_tail_level_weight 0.006 --loss_tail_level_start 96 --vali_metric_mode tail --vali_metric_horizon_start 96 --lradj type1 --train_epochs 18 --patience 7 --max_train_steps 2230 ;;
    336) run_cell 336 --model_id ETTh2_336_r310_0002 --des ETTh2_h2_best_r310_0002 --seq_len 432 --label_len 144 --e_layers 3 --d_model 160 --d_ff 320 --patch_len 8 --can_stride 4 --can_cli_mode full --can_temporal_cli_mode full --can_temporal_roll 0 --can_context_pyramid 0 --can_init_values 5e-4 --can_beta_init 0.46 --can_drop_path 0.498 --can_use_orth 1 --dropout 0.508 --batch_size 5 --learning_rate 0.00000290 --optimizer adamw --weight_decay 0.0001 --weight_averaging ema --ema_decay 0.999270 --ema_start_epoch 1 --loss_variable_weights 1.52,0.84,0.00,2.04,1.56,0.000,1.06 --loss_variance_weight 0.84 --loss_tail_bias_weight 0.017 --loss_tail_bias_start 216 --loss_tail_bias_variables 1.70,1.30,0.74,2.02,1.62,0.18,0.84 --can_gamma_lr_scale 0.02 --lradj cosine --train_epochs 78 --patience 24 --max_train_steps 12800 ;;
    720) run_cell 720 --model_id ETTh2_720_r68_0006 --des ETTh2_h2_best_r68_0006 --seq_len 312 --e_layers 3 --d_model 128 --d_ff 256 --patch_len 8 --can_stride 4 --can_cli_mode full --can_temporal_cli_mode full --can_temporal_roll 0 --can_context_pyramid 0 --can_init_values 1e-3 --can_drop_path 0.060 --can_use_orth 1 --dropout 0.060 --batch_size 20 --learning_rate 0.0000388 --optimizer adam --weight_decay 0.0 --weight_averaging ema --ema_decay 0.99575 --ema_start_epoch 1 --loss_variable_weights 1.15,1.00,0.925,1.25,1.125,0.75,0.925 --loss_tail_hard_start 510 --loss_tail_hard_weight 0.18 --loss_tail_hard_power 1.0 --loss_tail_hard_clip 2.0 --lradj cosine --train_epochs 12 --patience 5 --max_train_steps 724 ;;
    *) echo "Unsupported horizon: ${pred_len}" >&2; exit 2 ;;
  esac
done
