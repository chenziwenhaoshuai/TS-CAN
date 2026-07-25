#!/usr/bin/env python3
"""Summarize metrics.npy files under a CANPatchTST reproduction run root."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np


def parse_setting(path: Path) -> tuple[str, int]:
    text = str(path)
    dataset = ""
    for candidate in ("ETTh1", "ETTh2", "ETTm1", "ETTm2"):
        if candidate in text:
            dataset = candidate
            break
    pred_len = -1
    match = re.search(r"_pl(96|192|336|720)_", text)
    if match:
        pred_len = int(match.group(1))
    return dataset, pred_len


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: summarize_run_root.py RUN_ROOT", file=sys.stderr)
        return 2

    run_root = Path(sys.argv[1]).resolve()
    rows = []
    for metrics_path in sorted((run_root / "results").glob("*/metrics.npy")):
        dataset, pred_len = parse_setting(metrics_path)
        metrics = np.load(metrics_path)
        rows.append(
            {
                "dataset": dataset,
                "pred_len": pred_len,
                "mae": float(metrics[0]),
                "mse": float(metrics[1]),
                "rmse": float(metrics[2]),
                "mape": float(metrics[3]),
                "mspe": float(metrics[4]),
                "metrics_path": str(metrics_path),
            }
        )

    out_path = run_root / "summary_from_metrics.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "pred_len", "mae", "mse", "rmse", "mape", "mspe", "metrics_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote={out_path}")
    for row in rows:
        print(
            f"{row['dataset']},{row['pred_len']},mse={row['mse']:.9f},"
            f"mae={row['mae']:.9f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
