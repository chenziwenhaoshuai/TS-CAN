#!/usr/bin/env bash
set -euo pipefail

# Single-cell runner for the 12 historical non-ETTh2 CANPatchTST cells.
#
# Usage:
#   bash scripts/ETT-sota-reproduce/run_one_table12.sh ETTh1 96

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 DATASET PRED_LEN" >&2
  exit 2
fi

DATASET="$1"
PRED_LEN="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TSLIB_DIR="${REPO_ROOT}/Time-Series-Library-main"
RUN_ROOT="${RUN_ROOT:-runs/reproduce_ett_table12_single_${DATASET}_${PRED_LEN}_$(date +%Y%m%d_%H%M%S)}"

case "${DATASET}_${PRED_LEN}" in
  ETTh1_96)  SEQ_LEN=192; FREQ=""; DES=CAN_h1 ;;
  ETTh1_192) SEQ_LEN=192; FREQ=""; DES=CAN_h1 ;;
  ETTh1_336) SEQ_LEN=336; FREQ=""; DES=CAN_h1 ;;
  ETTh1_720) SEQ_LEN=336; FREQ=""; DES=CAN_h1 ;;
  ETTm1_96)  SEQ_LEN=192; FREQ=t; DES=CAN_m1 ;;
  ETTm1_192) SEQ_LEN=192; FREQ=t; DES=CAN_m1 ;;
  ETTm1_336) SEQ_LEN=192; FREQ=t; DES=CAN_m1 ;;
  ETTm1_720) SEQ_LEN=192; FREQ=t; DES=CAN_m1 ;;
  ETTm2_96)  SEQ_LEN=192; FREQ=t; DES=CAN_m2 ;;
  ETTm2_192) SEQ_LEN=192; FREQ=t; DES=CAN_m2 ;;
  ETTm2_336) SEQ_LEN=192; FREQ=t; DES=CAN_m2 ;;
  ETTm2_720) SEQ_LEN=192; FREQ=t; DES=CAN_m2 ;;
  *)
    echo "Unsupported cell: ${DATASET}_${PRED_LEN}" >&2
    exit 2
    ;;
esac

freq_args=()
if [ -n "${FREQ}" ]; then
  freq_args=(--freq "${FREQ}")
fi

cd "${TSLIB_DIR}"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/results" "${RUN_ROOT}/test_results"

log_path="${RUN_ROOT}/logs/${DATASET}_${PRED_LEN}.log"
echo "START ${DATASET}-${PRED_LEN} seq=${SEQ_LEN} run_root=${RUN_ROOT}"

python -u run_can.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path "${DATASET}.csv" \
  --model_id "${DATASET}_96_${PRED_LEN}" \
  --model CANPatchTST \
  --data "${DATASET}" \
  --features M \
  --seq_len "${SEQ_LEN}" \
  --label_len 48 \
  --pred_len "${PRED_LEN}" \
  --e_layers 2 \
  --d_model 128 \
  --d_ff 192 \
  --patch_len 16 \
  --can_stride 8 \
  --can_shifts 1,2,4,8,16 \
  --can_cli_mode full \
  --can_temporal_cli_mode full \
  --can_ctx_mode diff \
  --can_drop_path 0.05 \
  --can_kernel_size 3 \
  --can_init_values 1e-5 \
  --can_use_gffng 1 \
  --can_temporal_roll 1 \
  --can_use_orth 0 \
  --can_context_pyramid 1 \
  --dropout 0.05 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --lradj cosine \
  --train_epochs 5 \
  --patience 3 \
  --des "${DES}" \
  --itr 1 \
  --num_workers 0 \
  --use_amp \
  --seed 2 \
  --checkpoints "${TSLIB_DIR}/${RUN_ROOT}/checkpoints" \
  --results "${TSLIB_DIR}/${RUN_ROOT}/results" \
  --test_results "${TSLIB_DIR}/${RUN_ROOT}/test_results" \
  "${freq_args[@]}" \
  > "${log_path}" 2>&1

echo "END ${DATASET}-${PRED_LEN}"
python "${SCRIPT_DIR}/summarize_run_root.py" "${TSLIB_DIR}/${RUN_ROOT}"
