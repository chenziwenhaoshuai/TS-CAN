#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE="${HERE}/../ETT-sota-reproduce-archive"
TSLIB_DIR="$(cd "${HERE}/../../../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-runs/reproduce_etth2_current_best_$(date +%Y%m%d_%H%M%S)}"

for pred_len in 96 192 336 720; do
  RUN_ROOT="${RUN_ROOT}" bash "${ARCHIVE}/run_one_etth2_best.sh" "${pred_len}"
done

python "${ARCHIVE}/summarize_run_root.py" "${TSLIB_DIR}/${RUN_ROOT}"
echo "Summary: ${TSLIB_DIR}/${RUN_ROOT}/summary_from_metrics.csv"
