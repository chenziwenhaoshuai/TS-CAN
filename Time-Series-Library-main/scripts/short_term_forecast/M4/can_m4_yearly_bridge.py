#!/usr/bin/env python3
"""Bridge M4-Yearly between the low-MASE and low-SMAPE near-winners."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

local_scripts = types.ModuleType("scripts")
local_scripts.__path__ = [str(ROOT / "scripts")]
sys.modules["scripts"] = local_scripts


BASE = {
    "d_model": 32,
    "d_ff": 64,
    "e_layers": 4,
    "patch_len": 2,
    "can_stride": 1,
    "can_shifts": "1,2",
    "learning_rate": 0.0030,
    "batch_size": 96,
    "dropout": 0.00895,
    "can_drop_path": 0.001,
    "loss": "SMAPE",
    "lradj": "cosine",
    "train_epochs": 50,
    "patience": 20,
    "can_linear_residual": 1,
    "can_linear_mode": "decomp",
    "can_linear_scale_init": 0.007,
}


TRIALS = [
    ("Yearly", "Y280_drop00902_lr00299", {**BASE, "dropout": 0.00902, "learning_rate": 0.00299}),
    ("Yearly", "Y281_drop00902_lr00298", {**BASE, "dropout": 0.00902, "learning_rate": 0.00298}),
    ("Yearly", "Y282_drop00902_scale00698", {**BASE, "dropout": 0.00902, "can_linear_scale_init": 0.00698}),
    ("Yearly", "Y283_drop00902_scale00696", {**BASE, "dropout": 0.00902, "can_linear_scale_init": 0.00696}),
    ("Yearly", "Y284_drop00902_lr00299_scale00698", {**BASE, "dropout": 0.00902, "learning_rate": 0.00299, "can_linear_scale_init": 0.00698}),
    ("Yearly", "Y285_drop00902_lr00298_scale00696", {**BASE, "dropout": 0.00902, "learning_rate": 0.00298, "can_linear_scale_init": 0.00696}),
    ("Yearly", "Y286_drop00903_lr00299_scale00698", {**BASE, "dropout": 0.00903, "learning_rate": 0.00299, "can_linear_scale_init": 0.00698}),
    ("Yearly", "Y287_drop00901_lr00299_scale00698", {**BASE, "dropout": 0.00901, "learning_rate": 0.00299, "can_linear_scale_init": 0.00698}),
    ("Yearly", "Y288_drop00902_bs88_lr00299", {**BASE, "dropout": 0.00902, "learning_rate": 0.00299, "batch_size": 88}),
    ("Yearly", "Y289_drop00902_bs104_lr00299", {**BASE, "dropout": 0.00902, "learning_rate": 0.00299, "batch_size": 104}),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--trials", nargs="*")
    args = parser.parse_args()

    from scripts.short_term_forecast.M4.search_can_frequency import append_row, run_trial

    selected = set(args.trials or [])
    root = ROOT.resolve()
    archive = (root / "m4_results_archive/can_m4_yearly_bridge").resolve()
    summary = root / "short_term_results/m4_yearly_bridge.csv"

    for pattern, trial_id, overrides in TRIALS:
        if selected and trial_id not in selected:
            continue
        row = run_trial(root, archive, pattern, trial_id, overrides, args.gpu)
        append_row(summary, row)
        print(row, flush=True)
        if row.get("win_SMAPE") is True and row.get("win_MASE") is True and row.get("win_OWA") is True:
            print("FULL_WIN_STOP", row, flush=True)
            break


if __name__ == "__main__":
    main()
