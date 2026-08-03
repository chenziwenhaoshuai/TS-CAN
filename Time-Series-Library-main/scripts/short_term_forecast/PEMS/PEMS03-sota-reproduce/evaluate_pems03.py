#!/usr/bin/env python3
"""Independently evaluate the archived PEMS03 prediction arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TARGET = {"MAE": 13.99, "MAPE": 13.43, "RMSE": 24.03}


def clipped_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(pred - true)))
    mse = float(np.mean((pred - true) ** 2))
    rmse = float(np.sqrt(mse))
    ratio = np.abs((pred - true) / true)
    ratio = np.where(ratio > 5, 0, ratio)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": float(np.mean(ratio)) * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="artifacts/pred.npy")
    parser.add_argument("--true", default="artifacts/true.npy")
    parser.add_argument("--output")
    parser.add_argument("--trial", default="P3B44_02_bias0450")
    parser.add_argument("--selected-epoch", type=int, default=8)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    pred_path = Path(args.pred)
    true_path = Path(args.true)
    if not pred_path.is_absolute():
        pred_path = here / pred_path
    if not true_path.is_absolute():
        true_path = here / true_path

    pred = np.load(pred_path)
    true = np.load(true_path)
    metrics = clipped_metrics(pred, true)
    wins = {
        "MAE": metrics["MAE"] < TARGET["MAE"],
        "MAPE": metrics["MAPE"] < TARGET["MAPE"],
        "RMSE": metrics["RMSE"] < TARGET["RMSE"],
    }
    result = {
        "status": "accepted_2_of_3" if sum(wins.values()) >= 2 else "not_accepted",
        "dataset": "PEMS03",
        "trial": args.trial,
        "selected_epoch": args.selected_epoch,
        "shape_pred": list(pred.shape),
        "shape_true": list(true.shape),
        "metrics": metrics,
        "target": TARGET,
        "wins": {**wins, "count": sum(wins.values())},
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = here / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
