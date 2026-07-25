#!/usr/bin/env bash
set -euo pipefail

# Run the 12 non-ETTh2 ETT cells from the historical table with the current
# canonical CANPatchTST TSLib runner. This is intentionally pure CANPatchTST:
# no TimeMixer, no teacher fusion, no checkpoint reuse.
#
# Usage from any directory:
#   bash scripts/ETT-sota-reproduce/run_table12_current_configs.sh
#
# Optional:
#   CUDA_VISIBLE_DEVICES=0 RUN_ROOT=runs/reproduce_ett_table12_manual bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TSLIB_DIR="${REPO_ROOT}/Time-Series-Library-main"
RUN_ROOT="${RUN_ROOT:-runs/ETT-sota-reproduce_$(date +%Y%m%d_%H%M%S)}"

cd "${TSLIB_DIR}"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/results" "${RUN_ROOT}/test_results"

run_one() {
  local dataset="$1"
  local pred_len="$2"
  local seq_len="$3"
  local freq="$4"
  local des="$5"
  local data_path="${dataset}.csv"
  local log_path="${RUN_ROOT}/logs/${dataset}_${pred_len}.log"
  local freq_args=()
  if [ -n "${freq}" ]; then
    freq_args=(--freq "${freq}")
  fi

  echo "START ${dataset}-${pred_len} seq=${seq_len} log=${log_path}"
  python -u run_can.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path "${data_path}" \
    --model_id "${dataset}_96_${pred_len}" \
    --model CANPatchTST \
    --data "${dataset}" \
    --features M \
    --seq_len "${seq_len}" \
    --label_len 48 \
    --pred_len "${pred_len}" \
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
    --des "${des}" \
    --itr 1 \
    --num_workers 0 \
    --use_amp \
    --seed 2 \
    --checkpoints "${TSLIB_DIR}/${RUN_ROOT}/checkpoints" \
    --results "${TSLIB_DIR}/${RUN_ROOT}/results" \
    --test_results "${TSLIB_DIR}/${RUN_ROOT}/test_results" \
    "${freq_args[@]}" \
    > "${log_path}" 2>&1
  echo "END ${dataset}-${pred_len}"
}

run_one ETTh1 96 192 "" CAN_h1
run_one ETTh1 192 192 "" CAN_h1
run_one ETTh1 336 336 "" CAN_h1
run_one ETTh1 720 336 "" CAN_h1

run_one ETTm1 96 192 t CAN_m1
run_one ETTm1 192 192 t CAN_m1
run_one ETTm1 336 192 t CAN_m1
run_one ETTm1 720 192 t CAN_m1

run_one ETTm2 96 192 t CAN_m2
run_one ETTm2 192 192 t CAN_m2
run_one ETTm2 336 192 t CAN_m2
run_one ETTm2 720 192 t CAN_m2

python - <<'PY'
from pathlib import Path
import csv
import numpy as np
import os

run_root = Path(os.environ.get("RUN_ROOT", "") or "runs")
print(f"Run root: {run_root}")
print("Read metrics from the run_root/results subdirectories. See current_repro_20260723.csv for the latest c209 baseline.")
PY
