#!/usr/bin/env bash
set -euo pipefail

# Single-cell runner for the historical winning CANPatchTST artifact configs.
#
# Target set:
#   ETTh1: 96, 192, 336, 720
#   ETTm1: 96, 192, 336, 720
#   ETTm2: 96, 192, 336, 720
#   ETTh2: 720
#
# Usage:
#   bash scripts/ETT-sota-reproduce/run_one_winning_artifact.sh ETTh1 336

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 DATASET PRED_LEN" >&2
  exit 2
fi

DATASET="$1"
PRED_LEN="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TSLIB_DIR="${REPO_ROOT}/Time-Series-Library-main"
RUN_ROOT="${RUN_ROOT:-runs/reproduce_winning_artifact_${DATASET}_${PRED_LEN}_$(date +%Y%m%d_%H%M%S)}"

freq_args=()
if [[ "${DATASET}" == ETTm* ]]; then
  freq_args=(--freq t)
fi

common_args=(
  --task_name long_term_forecast
  --is_training 1
  --root_path ./dataset/ETT-small/
  --data_path "${DATASET}.csv"
  --model CANPatchTST
  --data "${DATASET}"
  --features M
  --label_len 48
  --pred_len "${PRED_LEN}"
  --can_shifts 1,2,4,8,16
  --can_ctx_mode diff
  --can_kernel_size 3
  --can_init_values 1e-5
  --can_beta_init 0.5
  --can_gamma_lr_scale 1.0
  --can_gamma_weight_decay 0.0
  --can_drop_path_schedule linear
  --can_use_gffng 1
  --can_temporal_roll 1
  --can_use_orth 0
  --can_context_pyramid 0
  --optimizer adam
  --weight_decay 0.0
  --warmup_epochs 1
  --weight_averaging none
  --ema_decay 0.995
  --ema_start_epoch 1
  --loss_variable_weights ""
  --loss_horizon_weight_start 0
  --loss_horizon_weight 1.0
  --loss_horizon_weight_mode step
  --loss_volatility_weight 0.0
  --loss_level_weight 0.0
  --loss_variance_weight 0.0
  --loss_range_weight 0.0
  --loss_tail_bias_weight 0.0
  --loss_tail_bias_start 0
  --loss_tail_bias_variables ""
  --loss_tail_level_weight 0.0
  --loss_tail_level_start 0
  --loss_tail_level_variables ""
  --loss_tail_hard_weight 0.0
  --loss_tail_hard_start 0
  --loss_tail_hard_power 1.0
  --loss_tail_hard_clip 3.0
  --loss_tail_lowpass_weight 0.0
  --loss_tail_lowpass_start 0
  --loss_tail_lowpass_kernel 9
  --loss_tail_lowpass_variables ""
  --vali_metric_mode all
  --vali_metric_horizon_start 0
  --itr 1
  --num_workers 0
  --use_amp
  --seed 2
)

case "${DATASET}_${PRED_LEN}" in
  ETTh1_96)
    cell_args=(
      --model_id ETTh1_96_s192_ref_ep2_bs8_repeat --des WIN_TM16_s192_ref_ep2_bs8_repeat
      --seq_len 192 --e_layers 2 --d_model 128 --d_ff 192
      --patch_len 16 --can_stride 8 --can_cli_mode full --can_temporal_cli_mode full
      --can_context_pyramid 1 --dropout 0.05 --can_drop_path 0.05
      --batch_size 8 --learning_rate 0.0005 --lradj cosine
      --train_epochs 2 --patience 2 --max_train_steps 0
    )
    ;;
  ETTh1_192)
    cell_args=(
      --model_id ETTh1_192_P192C05_final_train --des WIN_UCANP_P192C05
      --seq_len 512 --e_layers 2 --d_model 128 --d_ff 128
      --patch_len 12 --can_stride 6 --can_cli_mode inner --can_temporal_cli_mode inner
      --dropout 0.05 --can_drop_path 0.05
      --batch_size 2 --learning_rate 0.0002 --lradj type3
      --train_epochs 24 --patience 6 --max_train_steps 0
    )
    ;;
  ETTh1_336)
    cell_args=(
      --model_id ETTh1_336_r84_0022 --des WIN_CAN_h1_336_focus_r84_0022
      --seq_len 336 --e_layers 2 --d_model 128 --d_ff 192
      --patch_len 16 --can_stride 8 --can_cli_mode full --can_temporal_cli_mode full
      --can_context_pyramid 1 --dropout 0.03 --can_drop_path 0.03
      --batch_size 8 --learning_rate 0.0003 --lradj cosine
      --train_epochs 5 --patience 3 --max_train_steps 900
    )
    ;;
  ETTh1_720)
    cell_args=(
      --model_id ETTh1_720_R720_D --des WIN_PHASE2_R720_D
      --seq_len 208 --e_layers 2 --d_model 160 --d_ff 160
      --patch_len 32 --can_stride 16 --can_cli_mode full --can_temporal_cli_mode inner
      --dropout 0.1 --can_drop_path 0.1
      --batch_size 2 --learning_rate 0.00013 --lradj cosine
      --train_epochs 44 --patience 11 --max_train_steps 0
    )
    ;;
  ETTh2_720)
    cell_args=(
      --model_id ETTh2_720_r68_0006 --des WIN_ETTh2_r68_0006
      --seq_len 312 --e_layers 3 --d_model 128 --d_ff 256
      --patch_len 8 --can_stride 4 --can_cli_mode full --can_temporal_cli_mode full
      --can_temporal_roll 0 --can_context_pyramid 0 --can_init_values 1e-3
      --can_use_orth 1 --dropout 0.060 --can_drop_path 0.060
      --batch_size 20 --learning_rate 0.0000388 --optimizer adam --weight_decay 0.0
      --weight_averaging ema --ema_decay 0.99575 --ema_start_epoch 1
      --loss_variable_weights 1.15,1.00,0.925,1.25,1.125,0.75,0.925
      --loss_tail_hard_start 510 --loss_tail_hard_weight 0.18
      --loss_tail_hard_power 1.0 --loss_tail_hard_clip 2.0
      --lradj cosine --train_epochs 12 --patience 5 --max_train_steps 724
    )
    ;;
  ETTm1_96)
    cell_args=(
      --model_id ETTm1_96_P96C05_full_compare --des WIN_UCANP_P96C05
      --seq_len 512 --e_layers 2 --d_model 128 --d_ff 128
      --patch_len 12 --can_stride 6 --can_cli_mode inner --can_temporal_cli_mode inner
      --dropout 0.05 --can_drop_path 0.05
      --batch_size 2 --learning_rate 0.0002 --lradj type3
      --train_epochs 24 --patience 6 --max_train_steps 0
    )
    ;;
  ETTm1_192)
    cell_args=(
      --model_id ETTm1_192_R192_G --des WIN_PHASE4_R192_G
      --seq_len 336 --e_layers 2 --d_model 160 --d_ff 160
      --patch_len 12 --can_stride 6 --can_cli_mode inner --can_temporal_cli_mode full
      --dropout 0.05 --can_drop_path 0.05
      --batch_size 2 --learning_rate 0.0002 --lradj cosine
      --train_epochs 30 --patience 8 --max_train_steps 0
    )
    ;;
  ETTm1_336)
    cell_args=(
      --model_id ETTm1_336_R336_E --des WIN_PHASE3_R336_E
      --seq_len 336 --e_layers 2 --d_model 128 --d_ff 128
      --patch_len 32 --can_stride 16 --can_cli_mode inner --can_temporal_cli_mode full
      --dropout 0.1 --can_drop_path 0.1
      --batch_size 2 --learning_rate 0.00013 --lradj cosine
      --train_epochs 36 --patience 9 --max_train_steps 0
    )
    ;;
  ETTm1_720)
    cell_args=(
      --model_id ETTm1_720_R720_D --des WIN_PHASE3_R720_D
      --seq_len 208 --e_layers 2 --d_model 160 --d_ff 160
      --patch_len 32 --can_stride 16 --can_cli_mode full --can_temporal_cli_mode inner
      --dropout 0.1 --can_drop_path 0.1
      --batch_size 2 --learning_rate 0.00013 --lradj cosine
      --train_epochs 44 --patience 11 --max_train_steps 0
    )
    ;;
  ETTm2_96)
    cell_args=(
      --model_id ETTm2_96_R96_B --des WIN_PHASE3_R96_B
      --seq_len 512 --e_layers 2 --d_model 160 --d_ff 160
      --patch_len 16 --can_stride 8 --can_cli_mode inner --can_temporal_cli_mode full
      --dropout 0.05 --can_drop_path 0.05
      --batch_size 2 --learning_rate 0.0002 --lradj cosine
      --train_epochs 28 --patience 7 --max_train_steps 0
    )
    ;;
  ETTm2_192)
    cell_args=(
      --model_id ETTm2_192_R192_C --des WIN_PHASE3_R192_C
      --seq_len 512 --e_layers 2 --d_model 160 --d_ff 160
      --patch_len 16 --can_stride 8 --can_cli_mode inner --can_temporal_cli_mode inner
      --dropout 0.05 --can_drop_path 0.05
      --batch_size 2 --learning_rate 0.0002 --lradj cosine
      --train_epochs 28 --patience 7 --max_train_steps 0
    )
    ;;
  ETTm2_336)
    cell_args=(
      --model_id ETTm2_336_P336C01_screening --des WIN_UCANP_P336C01
      --seq_len 512 --e_layers 2 --d_model 128 --d_ff 128
      --patch_len 16 --can_stride 8 --can_cli_mode inner --can_temporal_cli_mode inner
      --dropout 0.1 --can_drop_path 0.1
      --batch_size 4 --learning_rate 0.00025 --lradj type3
      --train_epochs 2 --patience 2 --max_train_steps 0
    )
    ;;
  ETTm2_720)
    cell_args=(
      --model_id ETTm2_720_R720_B --des WIN_PHASE3_R720_B
      --seq_len 512 --e_layers 2 --d_model 128 --d_ff 128
      --patch_len 12 --can_stride 6 --can_cli_mode inner --can_temporal_cli_mode full
      --dropout 0.05 --can_drop_path 0.05
      --batch_size 2 --learning_rate 0.0002 --lradj cosine
      --train_epochs 44 --patience 11 --max_train_steps 0
    )
    ;;
  *)
    echo "Unsupported winning artifact cell: ${DATASET}_${PRED_LEN}" >&2
    exit 2
    ;;
esac

cd "${TSLIB_DIR}"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/results" "${RUN_ROOT}/test_results"

log_path="${RUN_ROOT}/logs/${DATASET}_${PRED_LEN}.log"
echo "START ${DATASET}-${PRED_LEN} run_root=${RUN_ROOT}"

python -u run_can.py \
  "${common_args[@]}" \
  "${cell_args[@]}" \
  --checkpoints "${TSLIB_DIR}/${RUN_ROOT}/checkpoints" \
  --results "${TSLIB_DIR}/${RUN_ROOT}/results" \
  --test_results "${TSLIB_DIR}/${RUN_ROOT}/test_results" \
  "${freq_args[@]}" \
  > "${log_path}" 2>&1

echo "END ${DATASET}-${PRED_LEN}"
python "${SCRIPT_DIR}/summarize_run_root.py" "${TSLIB_DIR}/${RUN_ROOT}"
