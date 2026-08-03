#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"

bash "$HERE/run_M4_monthly.sh" "$GPU"
bash "$HERE/run_M4_yearly.sh" "$GPU"
bash "$HERE/run_M4_daily.sh" "$GPU"
bash "$HERE/run_M4_others.sh" "$GPU"
bash "$HERE/run_M4_quarterly.sh" "$GPU"
