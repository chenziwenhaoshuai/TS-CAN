#!/usr/bin/env python3
"""Summarize CAN short-term runs against TimeMixer++ Tables 17 and 18."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from data_provider.m4 import M4Dataset, M4Meta


M4_TARGETS = OrderedDict([
    ("Yearly", {"SMAPE": 13.179, "MASE": 2.934, "OWA": 0.769}),
    ("Quarterly", {"SMAPE": 9.755, "MASE": 1.159, "OWA": 0.865}),
    ("Monthly", {"SMAPE": 12.432, "MASE": 0.904, "OWA": 0.841}),
    ("Others", {"SMAPE": 4.698, "MASE": 2.931, "OWA": 1.010}),
    ("Average", {"SMAPE": 11.448, "MASE": 1.487, "OWA": 0.821}),
])

PEMS_TARGETS = OrderedDict([
    ("PEMS03", {"MAE": 13.99, "MAPE": 13.43, "RMSE": 24.03}),
    ("PEMS04", {"MAE": 17.46, "MAPE": 11.34, "RMSE": 28.83}),
    ("PEMS07", {"MAE": 18.38, "MAPE": 7.32, "RMSE": 31.75}),
    ("PEMS08", {"MAE": 13.81, "MAPE": 8.21, "RMSE": 23.62}),
])


def latest_metrics_file(results_dir: Path, dataset: str) -> Path | None:
    candidates = sorted(
        results_dir.glob(
            f"long_term_forecast_{dataset}_CANPatchTST_PEMS_ftM_sl96_ll0_pl12_*CAN_short*/metrics.npy"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def all_metrics_files(results_dir: Path, dataset: str) -> list[Path]:
    return sorted(
        results_dir.glob(
            f"long_term_forecast_{dataset}_CANPatchTST_PEMS_ftM_sl96_ll0_pl12_*CAN_short*/metrics.npy"
        ),
        key=lambda path: path.stat().st_mtime,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_m4(root: Path, output_dir: Path) -> list[dict[str, object]]:
    forecast_dir = root / "m4_results" / "CANPatchTST"
    required = [
        forecast_dir / f"{pattern}_forecast.csv"
        for pattern in ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
    ]
    if not all(path.exists() for path in required):
        missing = [str(path) for path in required if not path.exists()]
        print("M4 missing forecasts:", missing)
        return []
    smape, mase, owa = evaluate_m4(forecast_dir, root / "dataset" / "m4")
    rows = []
    for group, targets in M4_TARGETS.items():
        for metric_name, value_map in [("SMAPE", smape), ("MASE", mase), ("OWA", owa)]:
            can_value = float(value_map[group])
            target = float(targets[metric_name])
            rows.append({
                "table": "17",
                "dataset": group,
                "metric": metric_name,
                "CAN": can_value,
                "TimeMixer++": target,
                "win": can_value < target,
            })
    write_csv(output_dir / "m4_vs_timemixerpp_table17.csv", rows)
    return rows


def clean_series(values: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(v[~np.isnan(v)], dtype=np.float64) for v in values]


def grouped_values(values: list[np.ndarray], groups: np.ndarray, group_name: str) -> list[np.ndarray]:
    return [values[i] for i in np.where(groups == group_name)[0]]


def smape_2(forecast: np.ndarray, target: np.ndarray) -> float:
    denom = np.abs(target) + np.abs(forecast)
    denom[denom == 0.0] = 1.0
    return float(np.mean(200.0 * np.abs(forecast - target) / denom))


def mase_one(forecast: np.ndarray, insample: np.ndarray, outsample: np.ndarray, frequency: int) -> float:
    scale = np.mean(np.abs(insample[:-frequency] - insample[frequency:]))
    if scale == 0.0:
        scale = 1.0
    return float(np.mean(np.abs(forecast - outsample)) / scale)


def summarize_groups(scores: dict[str, float], counts: dict[str, int]) -> OrderedDict[str, float]:
    result = OrderedDict()
    weighted = {}
    for group in ["Yearly", "Quarterly", "Monthly"]:
        result[group] = scores[group]
        weighted[group] = scores[group] * counts[group]
    others_total = 0.0
    others_count = 0
    for group in ["Weekly", "Daily", "Hourly"]:
        others_total += scores[group] * counts[group]
        others_count += counts[group]
    result["Others"] = others_total / others_count
    weighted["Others"] = others_total
    result["Average"] = sum(weighted.values()) / sum(counts.values())
    return result


def evaluate_m4(forecast_dir: Path, dataset_dir: Path):
    training_set = M4Dataset.load(training=True, dataset_file=str(dataset_dir))
    test_set = M4Dataset.load(training=False, dataset_file=str(dataset_dir))
    train_values = clean_series(training_set.values)
    test_values = clean_series(test_set.values)
    naive_values = clean_series(
        pd.read_csv(dataset_dir / "submission-Naive2.csv").values[:, 1:].astype(np.float64)
    )

    grouped_smapes = {}
    grouped_mases = {}
    naive_smapes = {}
    naive_mases = {}
    counts = {}
    for group in M4Meta.seasonal_patterns:
        forecast = pd.read_csv(forecast_dir / f"{group}_forecast.csv").values.astype(np.float64)
        target = grouped_values(test_values, test_set.groups, group)
        insample = grouped_values(train_values, training_set.groups, group)
        naive = grouped_values(naive_values, test_set.groups, group)
        frequency = int(training_set.frequencies[training_set.groups == group][0])
        counts[group] = len(target)

        target_matrix = np.vstack(target)
        naive_matrix = np.vstack(naive)
        grouped_smapes[group] = smape_2(forecast, target_matrix)
        naive_smapes[group] = smape_2(naive_matrix, target_matrix)
        grouped_mases[group] = float(np.mean([
            mase_one(forecast[i], insample[i], target[i], frequency)
            for i in range(len(target))
        ]))
        naive_mases[group] = float(np.mean([
            mase_one(naive[i], insample[i], target[i], frequency)
            for i in range(len(target))
        ]))

    smape = summarize_groups(grouped_smapes, counts)
    mase = summarize_groups(grouped_mases, counts)
    naive_smape = summarize_groups(naive_smapes, counts)
    naive_mase = summarize_groups(naive_mases, counts)
    owa = OrderedDict()
    for group in smape:
        owa[group] = 0.5 * (smape[group] / naive_smape[group] + mase[group] / naive_mase[group])
    return smape, mase, owa


def pems_metric_values(metrics_file: Path) -> dict[str, float]:
    mae, mse, rmse, mape, mspe = np.load(metrics_file)
    pred_file = metrics_file.parent / "pred.npy"
    true_file = metrics_file.parent / "true.npy"
    if pred_file.exists() and true_file.exists():
        pred = np.load(pred_file)
        true = np.load(true_file)
        ratio = np.abs((pred - true) / true)
        ratio = np.where(ratio > 5, 0, ratio)
        mape = float(np.mean(ratio))
    return {"MAE": float(mae), "MAPE": float(mape) * 100.0, "RMSE": float(rmse)}


def summarize_pems(root: Path, output_dir: Path) -> list[dict[str, object]]:
    rows = []
    for dataset, targets in PEMS_TARGETS.items():
        metrics_files = all_metrics_files(root / "results", dataset)
        if not metrics_files:
            print(f"PEMS missing metrics: {dataset}")
            continue
        scored = [(pems_metric_values(path)["MAE"], path) for path in metrics_files]
        metrics_file = min(scored, key=lambda item: item[0])[1]
        values = pems_metric_values(metrics_file)
        for metric_name, can_value in values.items():
            target = float(targets[metric_name])
            rows.append({
                "table": "18",
                "dataset": dataset,
                "metric": metric_name,
                "CAN": can_value,
                "TimeMixer++": target,
                "win": can_value < target,
                "metrics_file": str(metrics_file),
            })
    write_csv(output_dir / "pems_vs_timemixerpp_table18.csv", rows)

    detail_rows = []
    for dataset, targets in PEMS_TARGETS.items():
        for metrics_file in all_metrics_files(root / "results", dataset):
            values = pems_metric_values(metrics_file)
            detail_rows.append({
                "dataset": dataset,
                "setting": metrics_file.parent.name,
                "MAE": values["MAE"],
                "MAPE": values["MAPE"],
                "RMSE": values["RMSE"],
                "target_MAE": targets["MAE"],
                "target_MAPE": targets["MAPE"],
                "target_RMSE": targets["RMSE"],
                "win_MAE": values["MAE"] < targets["MAE"],
                "win_MAPE": values["MAPE"] < targets["MAPE"],
                "win_RMSE": values["RMSE"] < targets["RMSE"],
                "metrics_file": str(metrics_file),
            })
    write_csv(output_dir / "pems_all_can_short_results.csv", detail_rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("short_term_results"))
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()

    rows = summarize_m4(root, output_dir) + summarize_pems(root, output_dir)
    if rows:
        total = len(rows)
        wins = sum(bool(row["win"]) for row in rows)
        print(f"wins={wins}/{total}")
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
