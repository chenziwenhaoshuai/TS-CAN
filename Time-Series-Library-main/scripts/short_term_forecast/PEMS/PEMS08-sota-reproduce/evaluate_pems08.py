#!/usr/bin/env python3
"""Independently evaluate PEMS08 prediction arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TARGET = {
    "MAE": 13.81,
    "MAPE": 8.21,
    "RMSE": 23.62,
}


def pems_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err * err))
    rmse = float(np.sqrt(mse))
    ratio = np.abs(err / true)
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
    parser.add_argument("--trial", default="P8W35_10_bs6_lr00118_swa_s16_reproduce")
    parser.add_argument("--selected-epoch", type=int, default=18)
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
    metrics = pems_metrics(pred, true)
    wins = {name: metrics[name] < TARGET[name] for name in TARGET}
    payload = {
        "dataset": "PEMS08",
        "trial": args.trial,
        "epoch": args.selected_epoch,
        "metrics": metrics,
        "target_TimeMixerPP": TARGET,
        "wins": wins,
        "win_count": int(sum(wins.values())),
        "status": "accepted_2_of_3" if sum(wins.values()) >= 2 else "not_accepted",
        "prediction_shape": list(pred.shape),
        "true_shape": list(true.shape),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = here / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
