#!/usr/bin/env python3
"""Parallel from-scratch verifier for ETT-sota-reproduce scripts.

This driver runs the existing per-cell shell wrappers with a shared RUN_ROOT,
one process per available GPU, and writes a compact summary from metrics.npy.
It does not alter the model or per-cell hyperparameters.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CELLS = [
    ("ETTh1", 96, "winning_single/run_ETTh1_96.sh"),
    ("ETTh1", 192, "winning_single/run_ETTh1_192.sh"),
    ("ETTh1", 336, "winning_single/run_ETTh1_336.sh"),
    ("ETTh1", 720, "winning_single/run_ETTh1_720.sh"),
    ("ETTh2", 96, "single_h2_best/run_ETTh2_96.sh"),
    ("ETTh2", 192, "single_h2_best/run_ETTh2_192.sh"),
    ("ETTh2", 336, "single_h2_best/run_ETTh2_336.sh"),
    ("ETTh2", 720, "single_h2_best/run_ETTh2_720.sh"),
    ("ETTm1", 96, "winning_single/run_ETTm1_96.sh"),
    ("ETTm1", 192, "winning_single/run_ETTm1_192.sh"),
    ("ETTm1", 336, "winning_single/run_ETTm1_336.sh"),
    ("ETTm1", 720, "winning_single/run_ETTm1_720.sh"),
    ("ETTm2", 96, "winning_single/run_ETTm2_96.sh"),
    ("ETTm2", 192, "winning_single/run_ETTm2_192.sh"),
    ("ETTm2", 336, "winning_single/run_ETTm2_336.sh"),
    ("ETTm2", 720, "winning_single/run_ETTm2_720.sh"),
]

TMPP_MSE = {
    ("ETTh1", 96): 0.361,
    ("ETTh1", 192): 0.416,
    ("ETTh1", 336): 0.430,
    ("ETTh1", 720): 0.467,
    ("ETTh2", 96): 0.276,
    ("ETTh2", 192): 0.342,
    ("ETTh2", 336): 0.346,
    ("ETTh2", 720): 0.392,
    ("ETTm1", 96): 0.310,
    ("ETTm1", 192): 0.348,
    ("ETTm1", 336): 0.376,
    ("ETTm1", 720): 0.440,
    ("ETTm2", 96): 0.170,
    ("ETTm2", 192): 0.229,
    ("ETTm2", 336): 0.303,
    ("ETTm2", 720): 0.373,
}


@dataclass
class Running:
    dataset: str
    pred_len: int
    wrapper: str
    gpu: str
    process: subprocess.Popen
    log_handle: object
    started: float
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_setting(metrics_path: Path) -> tuple[str, int]:
    text = str(metrics_path)
    dataset = ""
    for candidate in ("ETTh1", "ETTh2", "ETTm1", "ETTm2"):
        if candidate in text:
            dataset = candidate
            break
    match = re.search(r"_pl(96|192|336|720)_", text)
    return dataset, int(match.group(1)) if match else -1


def existing_completed(summary_path: Path) -> set[tuple[str, int]]:
    if not summary_path.exists():
        return set()
    with summary_path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["dataset"], int(row["pred_len"]))
            for row in csv.DictReader(handle)
            if row.get("returncode") == "0"
        }


def append_row(summary_path: Path, row: dict[str, object]) -> None:
    fieldnames = [
        "dataset",
        "pred_len",
        "trial_id",
        "mse",
        "mae",
        "tmpp_mse",
        "mse_win",
        "returncode",
        "gpu",
        "elapsed_s",
        "log_path",
        "metrics_path",
        "wrapper",
        "timestamp",
    ]
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def find_metrics(run_root: Path, dataset: str, pred_len: int, started: float) -> Path | None:
    candidates = []
    for metrics_path in (run_root / "results").glob("*/metrics.npy"):
        parsed_dataset, parsed_pred_len = parse_setting(metrics_path)
        if parsed_dataset == dataset and parsed_pred_len == pred_len:
            candidates.append(metrics_path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    script_dir = repo / "scripts" / "ETT-sota-reproduce"
    tslib_dir = repo / "Time-Series-Library-main"
    run_root = tslib_dir / "runs" / args.run_name
    log_dir = run_root / "driver_logs"
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "summary_verify.csv"
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]

    completed = existing_completed(summary_path) if args.resume else set()
    queue = [(d, p, w) for d, p, w in CELLS if (d, p) not in completed]
    running: list[Running] = []

    print(f"repo={repo}", flush=True)
    print(f"run_root={run_root}", flush=True)
    print(f"completed={len(completed)} queue={len(queue)} gpus={gpus}", flush=True)

    while queue or running:
        available = [gpu for gpu in gpus if gpu not in {item.gpu for item in running}]
        while available and queue:
            gpu = available.pop(0)
            dataset, pred_len, wrapper = queue.pop(0)
            trial_id = f"verify_{dataset}_{pred_len}"
            log_path = log_dir / f"{trial_id}.log"
            env = os.environ.copy()
            conda_bin = "/home/c209/anaconda3/envs/pytorch2/bin"
            if Path(conda_bin).exists():
                env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["RUN_ROOT"] = f"runs/{args.run_name}/{trial_id}"
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                ["bash", str(script_dir / wrapper)],
                cwd=str(repo),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running.append(
                Running(dataset, pred_len, wrapper, gpu, process, handle, time.time(), log_path)
            )
            print(f"[launch] {trial_id} gpu={gpu} wrapper={wrapper}", flush=True)

        for item in list(running):
            ret = item.process.poll()
            if ret is None:
                continue
            item.log_handle.close()
            elapsed = time.time() - item.started
            cell_run_root = run_root / f"verify_{item.dataset}_{item.pred_len}"
            metrics_path = find_metrics(cell_run_root, item.dataset, item.pred_len, item.started)
            mse = ""
            mae = ""
            if metrics_path is not None:
                metrics = np.load(metrics_path)
                mae = f"{float(metrics[0]):.10f}"
                mse = f"{float(metrics[1]):.10f}"
            append_row(
                summary_path,
                {
                    "dataset": item.dataset,
                    "pred_len": item.pred_len,
                    "trial_id": f"verify_{item.dataset}_{item.pred_len}",
                    "mse": mse,
                    "mae": mae,
                    "tmpp_mse": f"{TMPP_MSE[(item.dataset, item.pred_len)]:.10f}",
                    "mse_win": int(bool(mse) and float(mse) < TMPP_MSE[(item.dataset, item.pred_len)]),
                    "returncode": ret,
                    "gpu": item.gpu,
                    "elapsed_s": f"{elapsed:.1f}",
                    "log_path": str(item.log_path),
                    "metrics_path": str(metrics_path or ""),
                    "wrapper": item.wrapper,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            print(
                f"[done] verify_{item.dataset}_{item.pred_len} ret={ret} "
                f"mse={mse} mae={mae} elapsed={elapsed:.1f}s",
                flush=True,
            )
            running.remove(item)

        time.sleep(5)

    print(f"summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
