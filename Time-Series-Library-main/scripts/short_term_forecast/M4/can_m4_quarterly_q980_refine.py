#!/usr/bin/env python3
"""Focused Q980 refinement for M4 Quarterly.

The current Quarterly frontier is Q980:
SMAPE:20,MASE:30,OWA:5,MASE:5 with OWA just above the TimeMixer++ target.
This script keeps the model code fixed and only perturbs existing training and
model-control arguments around that basin.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "M4_QUARTERLY_EPOCH_SUMMARY",
    "m4_quarterly_q980_refine.csv",
)
os.environ.setdefault(
    "M4_QUARTERLY_EPOCH_ARCHIVE",
    "can_m4_quarterly_q980_refine",
)

from scripts.short_term_forecast.M4 import can_m4_quarterly_composite_loss as comp  # noqa: E402


comp.base.SUMMARY_NAME = os.environ["M4_QUARTERLY_EPOCH_SUMMARY"]
comp.base.ARCHIVE_NAME = os.environ["M4_QUARTERLY_EPOCH_ARCHIVE"]


def cfg(schedule: str, **kwargs: object) -> dict[str, object]:
    return comp.cfg(schedule, **kwargs)


TRIALS = {
    # Exact Q980 replay plus local learning-rate/dropout perturbations.
    "Q1000_q980_replay": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5"),
    "Q1001_lr00434": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5", learning_rate=0.00434),
    "Q1002_lr00436": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5", learning_rate=0.00436),
    "Q1003_lr00438": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5", learning_rate=0.00438),
    "Q1004_lr00442": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5", learning_rate=0.00442),
    "Q1005_lr00446": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5", learning_rate=0.00446),
    "Q1006_drop0026": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5", dropout=0.0026),
    "Q1007_drop0028": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5", dropout=0.0028),
    "Q1008_drop0032": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5", dropout=0.0032),
    "Q1009_drop0034": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5", dropout=0.0034),
    "QM02_scale0038": cfg(
        "SMAPE:20,MASE:30,OWA:5,MASE:5",
        dropout=0.0032,
        can_periodic_image_scale_init=0.0038,
    ),

    # Small schedule shifts around the epoch-58 Q980 apex.
    "Q1010_m28_o5_m7": cfg("SMAPE:20,MASE:28,OWA:5,MASE:7"),
    "Q1011_m29_o5_m6": cfg("SMAPE:20,MASE:29,OWA:5,MASE:6"),
    "Q1012_m31_o5_m4": cfg("SMAPE:20,MASE:31,OWA:5,MASE:4"),
    "Q1013_m32_o5_m3": cfg("SMAPE:20,MASE:32,OWA:5,MASE:3"),
    "Q1014_m30_o4_m6": cfg("SMAPE:20,MASE:30,OWA:4,MASE:6"),
    "Q1015_m30_o6_m4": cfg("SMAPE:20,MASE:30,OWA:6,MASE:4"),
    "Q1016_m30_o3_m7": cfg("SMAPE:20,MASE:30,OWA:3,MASE:7"),
    "Q1017_m30_o7_m3": cfg("SMAPE:20,MASE:30,OWA:7,MASE:3"),

    # Periodic residual and periodic image scale micro-perturbations.
    "Q1020_alpha016": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5", can_periodic_alpha=0.016),
    "Q1021_alpha018": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5", can_periodic_alpha=0.018),
    "Q1022_alpha022": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5", can_periodic_alpha=0.022),
    "Q1023_alpha024": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5", can_periodic_alpha=0.024),
    "Q1024_scale003": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5", can_periodic_image_scale_init=0.003),
    "Q1025_scale005": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5", can_periodic_image_scale_init=0.005),
    "Q1026_scale006": cfg("SMAPE:20,MASE:30,OWA:5,MASE:5", can_periodic_image_scale_init=0.006),
    "Q1027_alpha018_scale005": cfg(
        "SMAPE:20,MASE:30,OWA:5,MASE:5",
        can_periodic_alpha=0.018,
        can_periodic_image_scale_init=0.005,
    ),

    # Use the lower-SMAPE composite front briefly, then repair MASE.
    "Q1030_s1_25_owa5_m10": cfg("SMAPE:20,OWA_S1:25,OWA:5,MASE:10"),
    "Q1031_s1_30_m10": cfg("SMAPE:20,OWA_S1:30,MASE:10"),
    "Q1032_s1_35_m5": cfg("SMAPE:20,OWA_S1:35,MASE:5"),
    "Q1033_s1_28_owa4_m8": cfg("SMAPE:20,OWA_S1:28,OWA:4,MASE:8"),
    "Q1034_s2_25_owa5_m10": cfg("SMAPE:20,OWA_S2:25,OWA:5,MASE:10"),
    "Q1035_s2_30_m10": cfg("SMAPE:20,OWA_S2:30,MASE:10"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=62)
    parser.add_argument("--trials", nargs="*", default=list(TRIALS))
    parser.add_argument("--stop-on-win", action="store_true", default=False)
    args = parser.parse_args()

    for trial_id in args.trials:
        overrides = dict(TRIALS[trial_id])
        overrides["gpu"] = int(args.gpu)
        hit = comp.run_trial(trial_id, overrides, int(args.epochs), args.stop_on_win)
        if hit and args.stop_on_win:
            break


if __name__ == "__main__":
    main()
