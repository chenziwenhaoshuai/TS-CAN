#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSLIB_ROOT="${TSLIB_ROOT:-$(cd "$HERE/../../../.." && pwd)}"
PYTHON="${PYTHON:-python}"
GPU="${1:-0}"
OUT="$HERE/artifacts/reproduced/Yearly"

mkdir -p "$OUT"
cd "$TSLIB_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

"$PYTHON" -u scripts/short_term_forecast/M4/can_m4_yearly_bridge.py \
  --gpu 0 \
  --trials Y283_drop00902_scale00696 \
  2>&1 | tee "$OUT/train.log"

cp m4_results_archive/can_m4_yearly_bridge/Yearly_Y283_drop00902_scale00696/Yearly_forecast.csv \
  "$OUT/Yearly_forecast.csv"
"$PYTHON" "$HERE/evaluate_yearly.py" "$OUT/Yearly_forecast.csv" \
  --tslib-root "$TSLIB_ROOT" \
  --output "$OUT/metrics.json"
