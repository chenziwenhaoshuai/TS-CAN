#!/usr/bin/env python3
"""Loss-schedule bridge scan for M4 Quarterly.

Static SMAPE training gives the best SMAPE front but weak MASE; static MASE
training gives the best MASE/OWA front but weak SMAPE.  This script keeps the
model unchanged and switches among existing TSLib losses at epoch boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "M4_QUARTERLY_EPOCH_SUMMARY",
    "m4_quarterly_loss_schedule.csv",
)
os.environ.setdefault(
    "M4_QUARTERLY_EPOCH_ARCHIVE",
    "can_m4_quarterly_loss_schedule",
)

from data_provider.m4 import M4Meta  # noqa: E402
from scripts.short_term_forecast.M4 import can_m4_quarterly_pareto_epoch_scan as base  # noqa: E402


base.SUMMARY_NAME = os.environ["M4_QUARTERLY_EPOCH_SUMMARY"]
base.ARCHIVE_NAME = os.environ["M4_QUARTERLY_EPOCH_ARCHIVE"]


LOW_SMAPE_Q524 = {
    "d_model": 64,
    "d_ff": 128,
    "e_layers": 4,
    "patch_len": 4,
    "can_stride": 2,
    "can_shifts": "1,2,4",
    "learning_rate": 0.0044,
    "batch_size": 96,
    "dropout": 0.003,
    "can_drop_path": 0.0,
    "loss": "SMAPE",
    "lradj": "cosine",
    "can_kernel_size": 5,
    "can_periodic_residual": 1,
    "can_periods": "4",
    "can_periodic_alpha": 0.02,
    "can_periodic_image": 1,
    "can_periodic_image_top_k": 2,
    "can_periodic_image_dim": 16,
    "can_periodic_image_layers": 1,
    "can_periodic_image_shifts": "1,2,4",
    "can_periodic_image_scale_init": 0.004,
}

LOW_MASE_Q721 = {
    "d_model": 64,
    "d_ff": 128,
    "e_layers": 4,
    "patch_len": 3,
    "can_stride": 2,
    "can_shifts": "1,2,4",
    "learning_rate": 0.00445,
    "batch_size": 96,
    "dropout": 0.002,
    "can_drop_path": 0.0,
    "loss": "MASE",
    "lradj": "cosine",
    "can_periodic_residual": 1,
    "can_periods": "4",
    "can_periodic_alpha": 0.03,
    "can_periodic_image": 1,
    "can_periodic_image_top_k": 2,
    "can_periodic_image_dim": 16,
    "can_periodic_image_layers": 1,
    "can_periodic_image_shifts": "1,2,4",
    "can_periodic_image_scale_init": 0.01,
}

LOW_MASE_Q343 = {
    **LOW_MASE_Q721,
    "patch_len": 4,
    "can_stride": 2,
    "learning_rate": 0.0046,
    "dropout": 0.002,
}

LOW_MASE_Q387 = {
    **LOW_MASE_Q343,
    "dropout": 0.004,
}


def cfg(base_cfg: dict[str, object], schedule: str, **kwargs: object) -> dict[str, object]:
    values = {**base_cfg, **kwargs}
    values["loss_schedule"] = schedule
    values["loss"] = schedule.split(",")[0].split(":")[0]
    return values


TRIALS = {
    # Start from the low-SMAPE front, then use MASE to repair scaling.
    "Q755_q524_s20_m40": cfg(LOW_SMAPE_Q524, "SMAPE:20,MASE:40"),
    "Q756_q524_s30_m30": cfg(LOW_SMAPE_Q524, "SMAPE:30,MASE:30"),
    "Q757_q524_s40_m20": cfg(LOW_SMAPE_Q524, "SMAPE:40,MASE:20"),
    "Q758_q524_s15_m45": cfg(LOW_SMAPE_Q524, "SMAPE:15,MASE:45"),
    "Q759_q524_s45_m15": cfg(LOW_SMAPE_Q524, "SMAPE:45,MASE:15"),
    "Q760_q524_s30_mape10_m20": cfg(LOW_SMAPE_Q524, "SMAPE:30,MAPE:10,MASE:20"),

    # Start from the low-MASE front, then use SMAPE late to pull percentage error.
    "Q761_p3_m20_s40": cfg(LOW_MASE_Q721, "MASE:20,SMAPE:40"),
    "Q762_p3_m30_s30": cfg(LOW_MASE_Q721, "MASE:30,SMAPE:30"),
    "Q763_p3_m40_s20": cfg(LOW_MASE_Q721, "MASE:40,SMAPE:20"),
    "Q764_p3_m45_s15": cfg(LOW_MASE_Q721, "MASE:45,SMAPE:15"),
    "Q765_p3_m30_s10_m20": cfg(LOW_MASE_Q721, "MASE:30,SMAPE:10,MASE:20"),
    "Q766_p3_s20_m40": cfg(LOW_MASE_Q721, "SMAPE:20,MASE:40"),

    # Same schedules on the original patch4 low-MASE basin.
    "Q767_q343_m30_s30": cfg(LOW_MASE_Q343, "MASE:30,SMAPE:30"),
    "Q768_q343_m40_s20": cfg(LOW_MASE_Q343, "MASE:40,SMAPE:20"),
    "Q769_q343_s20_m40": cfg(LOW_MASE_Q343, "SMAPE:20,MASE:40"),
    "Q770_q387_m30_s30": cfg(LOW_MASE_Q387, "MASE:30,SMAPE:30"),
    "Q771_q387_m40_s20": cfg(LOW_MASE_Q387, "MASE:40,SMAPE:20"),
    "Q772_q387_s20_m40": cfg(LOW_MASE_Q387, "SMAPE:20,MASE:40"),
}


def parse_schedule(text: str) -> list[tuple[str, int]]:
    parts: list[tuple[str, int]] = []
    for chunk in text.split(","):
        name, count = chunk.split(":", 1)
        parts.append((name.strip().upper(), int(count)))
    return parts


def loss_for_epoch(schedule: list[tuple[str, int]], epoch_index: int) -> str:
    cursor = 0
    for name, count in schedule:
        cursor += count
        if epoch_index < cursor:
            return name
    return schedule[-1][0]


def run_epoch_scan(trial_id: str, overrides: dict[str, object], epochs: int, stop_on_win: bool) -> bool:
    from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
    from scripts.short_term_forecast.M4.search_can_frequency import evaluate_pattern
    from utils.tools import adjust_learning_rate

    schedule = parse_schedule(str(overrides["loss_schedule"]))
    args = base.build_args(overrides)
    args.train_epochs = epochs
    args.model_id = f"m4_Quarterly_{trial_id}"
    args.des = f"M4freq_{trial_id}"
    base.set_seed(int(args.seed))
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")

    exp = Exp_Short_Term_Forecast(args)
    train_data, train_loader = exp._get_data(flag="train")
    vali_data, vali_loader = exp._get_data(flag="val")
    del train_data, vali_data

    model_optim = exp._select_optimizer()
    criteria = {
        name: exp._select_criterion(name)
        for name in sorted({name for name, _ in schedule})
    }
    summary = ROOT / "short_term_results" / base.SUMMARY_NAME
    archive = ROOT / "m4_results_archive" / base.ARCHIVE_NAME / trial_id
    ckpt_dir = archive / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_gap = float("inf")
    best_hit = False
    for epoch in range(int(args.train_epochs)):
        active_loss = loss_for_epoch(schedule, epoch)
        criterion = criteria[active_loss]
        start_time = time.time()
        exp.model.train()
        losses = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            del batch_x_mark
            model_optim.zero_grad()
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)
            outputs = exp.model(batch_x, None, dec_inp, None)
            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(exp.device)
            batch_y_mark = batch_y_mark[:, -args.pred_len:, f_dim:].to(exp.device)
            loss = criterion(batch_x, args.frequency_map, outputs, batch_y, batch_y_mark)
            losses.append(float(loss.item()))
            loss.backward()
            model_optim.step()

        vali_loss = exp.vali(train_loader, vali_loader, criterion)
        forecast = archive / f"epoch_{epoch + 1:03d}" / "Quarterly_forecast.csv"
        base.make_forecast(exp, forecast)
        metrics = base.metric_fields(evaluate_pattern(ROOT, "Quarterly", forecast))
        checkpoint = ckpt_dir / f"epoch_{epoch + 1:03d}.pth"
        is_best = bool(float(metrics["max_gap"]) < best_gap)
        if is_best:
            best_gap = float(metrics["max_gap"])
            torch.save(exp.model.state_dict(), checkpoint)
            shutil.copy2(forecast, archive / "best_Quarterly_forecast.csv")
            shutil.copy2(checkpoint, ckpt_dir / "best_by_test_gap.pth")

        row = {
            "pattern": "Quarterly",
            "trial": trial_id,
            "epoch": epoch + 1,
            "status": "ok",
            "elapsed_sec": round(time.time() - start_time, 2),
            "train_loss": float(np.mean(losses)),
            "vali_loss": float(vali_loss),
            "forecast": str(forecast),
            "checkpoint": str(checkpoint) if is_best else "",
            "best_so_far": is_best,
            "config_json": json.dumps({**overrides, "active_loss": active_loss}, sort_keys=True),
            **metrics,
        }
        base.append_row(summary, row)
        print(row, flush=True)
        if int(row["wins"]) >= 2:
            best_hit = True
            print("TWO_PLUS_WIN_STOP", row, flush=True)
            if stop_on_win:
                return True

        adjust_learning_rate(model_optim, epoch + 1, args)

    del exp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_hit


def run_trial(trial_id: str, overrides: dict[str, object], epochs: int, stop_on_win: bool) -> bool:
    original_history = M4Meta.history_size["Quarterly"]
    history_size = overrides.get("m4_history_size")
    if history_size is not None:
        M4Meta.history_size["Quarterly"] = float(history_size)
    try:
        return run_epoch_scan(trial_id, overrides, epochs, stop_on_win)
    finally:
        M4Meta.history_size["Quarterly"] = original_history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--trials", nargs="*", default=list(TRIALS))
    parser.add_argument("--stop-on-win", action="store_true", default=False)
    args = parser.parse_args()

    for trial_id in args.trials:
        overrides = dict(TRIALS[trial_id])
        overrides["gpu"] = int(args.gpu)
        hit = run_trial(trial_id, overrides, args.epochs, args.stop_on_win)
        if hit and args.stop_on_win:
            break


if __name__ == "__main__":
    main()
