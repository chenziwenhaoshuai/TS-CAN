#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$HERE/../../../.." && pwd)}"
PYTHON="${PYTHON:-python}"
TRIAL="P3B44_02_bias0450"
SELECTED_EPOCH="012"
SUMMARY_NAME="pems03_bias044_mape_refine_reproduce_aligned.csv"
OUT="$HERE/artifacts/reproduced"

cd "$ROOT"
mkdir -p "$OUT" scripts/short_term_forecast/PEMS

cp "$HERE/can_pems03_bias044_mape_refine.py" \
  scripts/short_term_forecast/PEMS/can_pems03_bias044_mape_refine.py

"$PYTHON" -u scripts/short_term_forecast/PEMS/can_pems03_bias044_mape_refine.py \
  --gpu "$GPU" \
  --trials "$TRIAL" \
  --summary-name "$SUMMARY_NAME" \
  2>&1 | tee "$OUT/train.log"

SOURCE_DIR="$ROOT/pems_results_archive/can_pems03_epoch_scan/$TRIAL/epoch_$SELECTED_EPOCH"
cp "$SOURCE_DIR/pred.npy" "$OUT/pred.npy"
cp "$SOURCE_DIR/true.npy" "$OUT/true.npy"
cp "$SOURCE_DIR/metrics.npy" "$OUT/metrics.npy"
cp "$ROOT/short_term_results/$SUMMARY_NAME" "$OUT/$SUMMARY_NAME"

"$PYTHON" "$HERE/evaluate_pems03.py" \
  --pred "$OUT/pred.npy" \
  --true "$OUT/true.npy" \
  --trial "$TRIAL" \
  --selected-epoch "$((10#$SELECTED_EPOCH))" \
  --output "$OUT/metrics.json"
