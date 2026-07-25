#!/usr/bin/env bash
set -euo pipefail

# Run all historical winning CANPatchTST artifact configurations:
# H1/M1/M2 all horizons plus H2-720.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TSLIB_DIR="${REPO_ROOT}/Time-Series-Library-main"
RUN_ROOT="${RUN_ROOT:-runs/reproduce_winning_artifacts_$(date +%Y%m%d_%H%M%S)}"

cells=(
  "ETTh1 96"
  "ETTh1 192"
  "ETTh1 336"
  "ETTh1 720"
  "ETTm1 96"
  "ETTm1 192"
  "ETTm1 336"
  "ETTm1 720"
  "ETTm2 96"
  "ETTm2 192"
  "ETTm2 336"
  "ETTm2 720"
  "ETTh2 720"
)

for cell in "${cells[@]}"; do
  read -r dataset pred_len <<< "${cell}"
  RUN_ROOT="${RUN_ROOT}" bash "${SCRIPT_DIR}/run_one_winning_artifact.sh" "${dataset}" "${pred_len}"
done

python "${SCRIPT_DIR}/summarize_run_root.py" "${TSLIB_DIR}/${RUN_ROOT}"
echo "Summary: ${TSLIB_DIR}/${RUN_ROOT}/summary_from_metrics.csv"
