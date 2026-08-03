#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$HERE/../../../.." && pwd)}"
PYTHON="${PYTHON:-python}"
TRIAL="P8W35_10_bs6_lr00118_swa_s16_reproduce"
SELECTED_EPOCH="018"
SUMMARY_NAME="P8W35_10_bs6_lr00118_swa_s16_reproduce.csv"
OUT="$HERE/artifacts/reproduced"

mkdir -p "$OUT"
cd "$ROOT"

"$PYTHON" -u scripts/short_term_forecast/PEMS/can_pems_epoch_scan.py \
  --dataset PEMS08 \
  --batch-size 6 \
  --can-beta-init 0.5 \
  --can-cli-mode full \
  --can-context-pyramid 0 \
  --can-cross-var 1 \
  --can-cross-var-context others_mean \
  --can-cross-var-layers 1 \
  --can-cross-var-shifts 1,2,4,8,16 \
  --can-ctx-mode diff \
  --can-drop-path 0.0 \
  --can-global-cli-mode inner \
  --can-global-ctx-mode abs \
  --can-init-values 1e-05 \
  --can-kernel-size 3 \
  --can-linear-mode raw \
  --can-linear-residual 0 \
  --can-linear-scale-init 0.5 \
  --multiscale-main-bias 0.1 \
  --multiscale-patch-lens 6,12,24 \
  --can-periodic-alpha 0.03 \
  --can-periodic-learnable 1 \
  --can-periodic-residual 1 \
  --can-periods 12,24 \
  --can-shifts 1,2,4,8 \
  --can-stride 4 \
  --can-temporal-cli-mode inner \
  --can-temporal-roll 1 \
  --can-use-gffng 1 \
  --can-var-attn 0 \
  --d-ff 160 \
  --d-model 80 \
  --dropout 0.0 \
  --e-layers 3 \
  --ema-decay 0.995 \
  --ema-start-epoch 1 \
  --gpu "$GPU" \
  --huber-delta 1.0 \
  --label-len 0 \
  --learning-rate 0.00118 \
  --loss MAE \
  --loss-mse-weight 0.5 \
  --lradj cosine \
  --optimizer adam \
  --patch-len 8 \
  --pred-len 12 \
  --seq-len 96 \
  --swa-end-epoch 0 \
  --swa-start-epoch 16 \
  --train-epochs 34 \
  --use-norm 0 \
  --warmup-epochs 1 \
  --weight-averaging swa \
  --weight-decay 0.0 \
  --summary-name "$SUMMARY_NAME" \
  --trial "$TRIAL" \
  2>&1 | tee "$OUT/train.log"

SOURCE_DIR="$ROOT/pems_results_archive/can_pems08_epoch_scan/$TRIAL/epoch_$SELECTED_EPOCH"
cp "$SOURCE_DIR/pred.npy" "$OUT/pred.npy"
cp "$SOURCE_DIR/true.npy" "$OUT/true.npy"
cp "$SOURCE_DIR/metrics.npy" "$OUT/metrics.npy"
cp "$ROOT/short_term_results/$SUMMARY_NAME" "$OUT/$SUMMARY_NAME"

"$PYTHON" "$HERE/evaluate_pems08.py" \
  --pred "$OUT/pred.npy" \
  --true "$OUT/true.npy" \
  --trial "$TRIAL" \
  --selected-epoch "$((10#$SELECTED_EPOCH))" \
  --output "$OUT/metrics.json"
