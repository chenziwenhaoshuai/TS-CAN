#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE="${HERE}/../ETT-sota-reproduce-archive"
TSLIB_DIR="$(cd "${HERE}/../../../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-runs/reproduce_winning_artifacts_$(date +%Y%m%d_%H%M%S)}"

for cell in \
  "ETTh1 96" "ETTh1 192" "ETTh1 336" "ETTh1 720" \
  "ETTm1 96" "ETTm1 192" "ETTm1 336" "ETTm1 720" \
  "ETTm2 96" "ETTm2 192" "ETTm2 336" "ETTm2 720" \
  "ETTh2 720"; do
  read -r dataset pred_len <<< "${cell}"
  RUN_ROOT="${RUN_ROOT}" bash "${ARCHIVE}/run_one_winning_artifact.sh" "${dataset}" "${pred_len}"
done

python "${ARCHIVE}/summarize_run_root.py" "${TSLIB_DIR}/${RUN_ROOT}"
echo "Summary: ${TSLIB_DIR}/${RUN_ROOT}/summary_from_metrics.csv"
