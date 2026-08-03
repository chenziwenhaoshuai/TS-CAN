#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSLIB_ROOT="${TSLIB_ROOT:-$(cd "$HERE/../../../.." && pwd)}"
PYTHON="${PYTHON:-python}"
GPU="${1:-0}"
OUT="$HERE/artifacts/reproduced/Quarterly"

mkdir -p "$OUT" "$TSLIB_ROOT/scripts/short_term_forecast/M4"
cp "$HERE/can_m4_quarterly_loss_schedule.py" "$TSLIB_ROOT/scripts/short_term_forecast/M4/can_m4_quarterly_loss_schedule.py"
cp "$HERE/can_m4_quarterly_composite_loss.py" "$TSLIB_ROOT/scripts/short_term_forecast/M4/can_m4_quarterly_composite_loss.py"
cp "$HERE/can_m4_quarterly_q980_refine.py" "$TSLIB_ROOT/scripts/short_term_forecast/M4/can_m4_quarterly_q980_refine.py"

cd "$TSLIB_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

M4_QUARTERLY_EPOCH_SUMMARY="m4_quarterly_q980_reproduce.csv" \
M4_QUARTERLY_EPOCH_ARCHIVE="can_m4_quarterly_q980_reproduce" \
"$PYTHON" -u scripts/short_term_forecast/M4/can_m4_quarterly_q980_refine.py \
  --gpu 0 --epochs 58 --trials QM02_scale0038 \
  2>&1 | tee "$OUT/train.log"

cp m4_results_archive/can_m4_quarterly_q980_reproduce/QM02_scale0038/epoch_058/Quarterly_forecast.csv \
  "$OUT/Quarterly_forecast.csv"
"$PYTHON" "$HERE/evaluate_quarterly.py" "$OUT/Quarterly_forecast.csv" \
  --tslib-root "$TSLIB_ROOT" \
  --output "$OUT/metrics.json"
