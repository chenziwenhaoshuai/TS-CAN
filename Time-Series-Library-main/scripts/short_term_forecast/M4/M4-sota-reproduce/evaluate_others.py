#!/usr/bin/env python3
"""Evaluate weighted M4 Others from Weekly, Daily, and Hourly forecasts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = {"SMAPE": 4.698, "MASE": 2.931, "OWA": 1.010}
PATTERNS = ["Weekly", "Daily", "Hourly"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-dir", type=Path, default=Path.cwd())
    parser.add_argument("--tslib-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    root = (args.tslib_root or here.parents[4]).resolve()
    forecast_dir = args.forecast_dir.resolve()
    output = (args.output or forecast_dir / "metrics.json").resolve()

    dataset_dir = root / "dataset" / "m4"
    if not (dataset_dir / "M4-info.csv").exists():
        raise FileNotFoundError(f"M4 dataset is missing under {dataset_dir}")

    forecasts = {pattern: forecast_dir / f"{pattern}_forecast.csv" for pattern in PATTERNS}
    missing = [str(path) for path in forecasts.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing forecast files: {missing}")

    sys.path.insert(0, str(root))
    from data_provider.m4 import M4Dataset
    from scripts.short_term_forecast.summarize_can_short import (
        clean_series,
        grouped_values,
        mase_one,
        smape_2,
    )

    training_set = M4Dataset.load(training=True, dataset_file=str(dataset_dir))
    test_set = M4Dataset.load(training=False, dataset_file=str(dataset_dir))
    train_values = clean_series(training_set.values)
    test_values = clean_series(test_set.values)
    naive_values = clean_series(
        pd.read_csv(dataset_dir / "submission-Naive2.csv").values[:, 1:].astype(np.float64)
    )

    per_subset = {}
    total_count = 0
    weighted_smape = 0.0
    weighted_mase = 0.0
    weighted_naive_smape = 0.0
    weighted_naive_mase = 0.0

    for pattern in PATTERNS:
        forecast = pd.read_csv(forecasts[pattern]).values.astype(np.float64)
        target = grouped_values(test_values, test_set.groups, pattern)
        insample = grouped_values(train_values, training_set.groups, pattern)
        naive = grouped_values(naive_values, test_set.groups, pattern)
        frequency = int(training_set.frequencies[training_set.groups == pattern][0])

        target_matrix = np.vstack(target)
        naive_matrix = np.vstack(naive)
        smape = smape_2(forecast, target_matrix)
        naive_smape = smape_2(naive_matrix, target_matrix)
        mase = float(np.mean([
            mase_one(forecast[i], insample[i], target[i], frequency)
            for i in range(len(target))
        ]))
        naive_mase = float(np.mean([
            mase_one(naive[i], insample[i], target[i], frequency)
            for i in range(len(target))
        ]))
        owa = 0.5 * (smape / naive_smape + mase / naive_mase)
        count = len(target)

        per_subset[pattern] = {
            "count": count,
            "SMAPE": smape,
            "MASE": mase,
            "OWA": owa,
            "naive_SMAPE": naive_smape,
            "naive_MASE": naive_mase,
            "forecast": str(forecasts[pattern]),
        }
        total_count += count
        weighted_smape += smape * count
        weighted_mase += mase * count
        weighted_naive_smape += naive_smape * count
        weighted_naive_mase += naive_mase * count

    smape = weighted_smape / total_count
    mase = weighted_mase / total_count
    naive_smape = weighted_naive_smape / total_count
    naive_mase = weighted_naive_mase / total_count
    owa = 0.5 * (smape / naive_smape + mase / naive_mase)
    metrics = {"SMAPE": smape, "MASE": mase, "OWA": owa}
    wins = {metric: metrics[metric] < target for metric, target in TARGET.items()}

    result = {
        "scope": "M4 weighted Others aggregate over Weekly, Daily, and Hourly",
        "forecast_dir": str(forecast_dir),
        "metrics": metrics,
        "target": TARGET,
        "wins": {**wins, "count": sum(wins.values())},
        "status": "full_win" if all(wins.values()) else f"accepted_{sum(wins.values())}_of_3",
        "per_subset": per_subset,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
