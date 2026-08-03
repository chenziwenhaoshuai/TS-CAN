#!/usr/bin/env python3
"""Composite-loss bridge scan for M4 Quarterly.

This keeps the CANPatchTST structure fixed and combines existing M4 losses.
The goal is to preserve the low-SMAPE front while still optimizing the MASE
component that dominates OWA near the current Q755 frontier.
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
    "m4_quarterly_composite_loss.csv",
)
os.environ.setdefault(
    "M4_QUARTERLY_EPOCH_ARCHIVE",
    "can_m4_quarterly_composite_loss",
)

from data_provider.m4 import M4Meta  # noqa: E402
from scripts.short_term_forecast.M4 import can_m4_quarterly_pareto_epoch_scan as base  # noqa: E402
from scripts.short_term_forecast.M4 import can_m4_quarterly_loss_schedule as scheduled  # noqa: E402


base.SUMMARY_NAME = os.environ["M4_QUARTERLY_EPOCH_SUMMARY"]
base.ARCHIVE_NAME = os.environ["M4_QUARTERLY_EPOCH_ARCHIVE"]


class WeightedCriterion:
    def __init__(self, named_criteria: dict[str, torch.nn.Module], weights: dict[str, float]):
        self.named_criteria = named_criteria
        total = sum(float(v) for v in weights.values())
        if total <= 0:
            raise ValueError("Composite weights must sum to a positive value")
        self.weights = {name: float(value) / total for name, value in weights.items()}

    def __call__(self, insample, freq, forecast, target, mask):
        loss = None
        for name, weight in self.weights.items():
            value = self.named_criteria[name](insample, freq, forecast, target, mask)
            term = value * weight
            loss = term if loss is None else loss + term
        return loss


def cfg(schedule: str = "SMAPE:20,OWA:40", **kwargs: object) -> dict[str, object]:
    values = {**scheduled.LOW_SMAPE_Q524, **kwargs}
    values["loss_schedule"] = schedule
    values["loss"] = schedule.split(",", 1)[0].split(":", 1)[0]
    return values


COMPOSITES: dict[str, dict[str, float]] = {
    # OWA ~= 0.045406786 * SMAPE + 0.364595642 * MASE.
    # The weights are normalized inside WeightedCriterion.
    "OWA": {"SMAPE": 0.045406786, "MASE": 0.364595642},
    "OWA_S1": {"SMAPE": 0.055, "MASE": 0.355},
    "OWA_S2": {"SMAPE": 0.065, "MASE": 0.345},
    "OWA_S3": {"SMAPE": 0.075, "MASE": 0.335},
    "OWA_M1": {"SMAPE": 0.040, "MASE": 0.380},
    "OWA_M2": {"SMAPE": 0.035, "MASE": 0.400},
    "OWA_BAL": {"SMAPE": 0.050, "MASE": 0.300},
    "OWA_MAPE": {"SMAPE": 0.050, "MASE": 0.340, "MAPE": 0.020},
}


TRIALS = {
    # Direct bridge from the verified Q755 schedule.
    "Q900_s20_owa": cfg("SMAPE:20,OWA:40"),
    "Q901_s18_owa": cfg("SMAPE:18,OWA:42"),
    "Q902_s22_owa": cfg("SMAPE:22,OWA:38"),
    "Q903_s20_s1": cfg("SMAPE:20,OWA_S1:40"),
    "Q904_s20_s2": cfg("SMAPE:20,OWA_S2:40"),
    "Q905_s20_s3": cfg("SMAPE:20,OWA_S3:40"),
    "Q906_s20_m1": cfg("SMAPE:20,OWA_M1:40"),
    "Q907_s20_m2": cfg("SMAPE:20,OWA_M2:40"),
    "Q908_s20_bal": cfg("SMAPE:20,OWA_BAL:40"),
    "Q909_s20_mape": cfg("SMAPE:20,OWA_MAPE:40"),

    # Small parameter perturbations on the same 60-epoch trajectory.
    "Q910_owa_lr00434": cfg("SMAPE:20,OWA:40", learning_rate=0.00434),
    "Q911_owa_lr00446": cfg("SMAPE:20,OWA:40", learning_rate=0.00446),
    "Q912_owa_drop0025": cfg("SMAPE:20,OWA:40", dropout=0.0025),
    "Q913_owa_alpha018": cfg("SMAPE:20,OWA:40", can_periodic_alpha=0.018),
    "Q914_owa_alpha022": cfg("SMAPE:20,OWA:40", can_periodic_alpha=0.022),

    # Check whether deterministic kernels stabilize or hurt the frontier.
    "Q920_exact_det": cfg("SMAPE:20,MASE:40", deterministic=1),
    "Q921_owa_det": cfg("SMAPE:20,OWA:40", deterministic=1),
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


def set_deterministic(enabled: bool) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(enabled)
    try:
        torch.use_deterministic_algorithms(bool(enabled), warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(bool(enabled))


def build_criteria(exp, schedule: list[tuple[str, int]]) -> dict[str, object]:
    primitive_names = {"SMAPE", "MASE", "MAPE"}
    active_names = {name for name, _ in schedule}
    needed_primitives = set(active_names & primitive_names)
    for name in active_names:
        needed_primitives.update(COMPOSITES.get(name, {}).keys())
    primitive = {name: exp._select_criterion(name) for name in sorted(needed_primitives)}

    criteria: dict[str, object] = {}
    for name in active_names:
        if name in primitive:
            criteria[name] = primitive[name]
        elif name in COMPOSITES:
            criteria[name] = WeightedCriterion(primitive, COMPOSITES[name])
        else:
            raise KeyError(f"Unknown loss schedule component: {name}")
    return criteria


def run_epoch_scan(trial_id: str, overrides: dict[str, object], epochs: int, stop_on_win: bool) -> bool:
    from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
    from scripts.short_term_forecast.M4.search_can_frequency import evaluate_pattern
    from utils.tools import adjust_learning_rate

    deterministic = bool(int(overrides.get("deterministic", 0) or 0))
    set_deterministic(deterministic)
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
    criteria = build_criteria(exp, schedule)
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
        hit = run_trial(trial_id, overrides, int(args.epochs), args.stop_on_win)
        if hit and args.stop_on_win:
            break


if __name__ == "__main__":
    main()
