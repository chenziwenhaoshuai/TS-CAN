#!/usr/bin/env python3
"""Evaluate an archived M4-Yearly forecast with the project evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


TARGET = {"SMAPE": 13.179, "MASE": 2.934, "OWA": 0.769}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("forecast", type=Path)
    parser.add_argument("--tslib-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    root = (args.tslib_root or here.parents[4]).resolve()
    forecast = args.forecast.resolve()
    output = (args.output or forecast.with_name("metrics.json")).resolve()

    if not forecast.exists():
        raise FileNotFoundError(forecast)
    if not (root / "dataset/m4/M4-info.csv").exists():
        raise FileNotFoundError(f"M4 dataset is missing under {root / 'dataset/m4'}")

    sys.path.insert(0, str(root))
    from scripts.short_term_forecast.M4.search_can_frequency import evaluate_pattern

    metrics = evaluate_pattern(root, "Yearly", forecast)
    wins = {
        metric: float(metrics[metric]) < target
        for metric, target in TARGET.items()
    }
    result = {
        "forecast": str(forecast),
        "metrics": {metric: float(metrics[metric]) for metric in TARGET},
        "target": TARGET,
        "wins": {**wins, "count": sum(wins.values())},
        "status": "full_win" if all(wins.values()) else f"accepted_{sum(wins.values())}_of_3",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
