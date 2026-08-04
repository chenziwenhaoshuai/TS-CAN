#!/usr/bin/env bash
set -euo pipefail

# TS-CAN ETTm2: horizon-specific reproduced best configs.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

run_cell() {
  local pred_len="$1"
  shift
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model CANPatchTST \
    --data ETTm2 \
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
    --seed 2 --freq t \
    "$@"

}

for pred_len in 96 192 336 720; do
  case "${pred_len}" in
    96) run_cell 96 --model_id ETTm2_96_R96_B --des WIN_PHASE3_R96_B --seq_len 512 --e_layers 2 --d_model 160 --d_ff 160 --patch_len 16 --can_stride 8 --can_cli_mode inner --can_temporal_cli_mode full --dropout 0.05 --can_drop_path 0.05 --batch_size 2 --learning_rate 0.0002 --lradj cosine --train_epochs 28 --patience 7 --max_train_steps 0 ;;
    192) run_cell 192 --model_id ETTm2_192_R192_C --des WIN_PHASE3_R192_C --seq_len 512 --e_layers 2 --d_model 160 --d_ff 160 --patch_len 16 --can_stride 8 --can_cli_mode inner --can_temporal_cli_mode inner --dropout 0.05 --can_drop_path 0.05 --batch_size 2 --learning_rate 0.0002 --lradj cosine --train_epochs 28 --patience 7 --max_train_steps 0 ;;
    336) run_cell 336 --model_id ETTm2_336_P336C01_screening --des WIN_UCANP_P336C01 --seq_len 512 --e_layers 2 --d_model 128 --d_ff 128 --patch_len 16 --can_stride 8 --can_cli_mode inner --can_temporal_cli_mode inner --dropout 0.1 --can_drop_path 0.1 --batch_size 4 --learning_rate 0.00025 --lradj type3 --train_epochs 2 --patience 2 --max_train_steps 0 ;;
    720) run_cell 720 --model_id ETTm2_720_R720_B --des WIN_PHASE3_R720_B --seq_len 512 --e_layers 2 --d_model 128 --d_ff 128 --patch_len 12 --can_stride 6 --can_cli_mode inner --can_temporal_cli_mode full --dropout 0.05 --can_drop_path 0.05 --batch_size 2 --learning_rate 0.0002 --lradj cosine --train_epochs 44 --patience 11 --max_train_steps 0 ;;
    *) echo "Unsupported horizon: ${pred_len}" >&2; exit 2 ;;
  esac
done
