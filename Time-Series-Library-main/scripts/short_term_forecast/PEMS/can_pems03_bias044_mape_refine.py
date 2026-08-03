#!/usr/bin/env python3
"""PEMS03 bias=0.44 MAPE refinement.

This sweep keeps CANPatchTST code fixed and only combines existing
``can_pems_epoch_scan.py`` command-line knobs. It targets the historical best
score basin:

P3X_01_full_bias044_lr0022_bs10, epoch 10
MAE 14.4871, MAPE 13.4457, RMSE 23.4221

The goal is to reduce the small MAPE gap without losing the strong MAE/RMSE
profile.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


BASE = {
    "dataset": "PEMS03",
    "d_model": 64,
    "d_ff": 128,
    "e_layers": 3,
    "patch_len": 12,
    "can_stride": 6,
    "learning_rate": 0.0022,
    "lradj": "cosine",
    "batch_size": 10,
    "dropout": 0.005,
    "can_drop_path": 0.005,
    "can_cli_mode": "full",
    "can_temporal_cli_mode": "inner",
    "can_context_pyramid": 0,
    "can_multiscale_patch_lens": "8,16",
    "can_multiscale_main_bias": 0.44,
    "can_cross_var": 1,
    "can_cross_var_layers": 1,
    "can_cross_var_context": "others_mean",
    "can_cross_var_shifts": "1,2,4,8,16",
    "can_periodic_residual": 0,
    "loss": "MAE",
    "train_epochs": 12,
    "start_epoch": 7,
    "end_epoch": 12,
}


def cfg(**kwargs: object) -> dict[str, object]:
    values = dict(BASE)
    values.update(kwargs)
    return values


TRIALS = {
    # Very narrow interpolation from the low-MAE bias=0.44 basin toward the
    # MAPE-winning bias=0.48 basin.
    "P3B44_00_bias0442": cfg(can_multiscale_main_bias=0.442),
    "P3B44_01_bias0445": cfg(can_multiscale_main_bias=0.445),
    "P3B44_02_bias0450": cfg(can_multiscale_main_bias=0.450),
    "P3B44_03_bias0455": cfg(can_multiscale_main_bias=0.455),
    "P3B44_04_bias0460": cfg(can_multiscale_main_bias=0.460),
    "P3B44_05_bias0465": cfg(can_multiscale_main_bias=0.465),
    # Preserve the bias=0.44 basin and perturb optimization only.
    "P3B44_10_lr00218": cfg(learning_rate=0.00218),
    "P3B44_11_lr00222": cfg(learning_rate=0.00222),
    "P3B44_12_reg004": cfg(dropout=0.004, can_drop_path=0.004),
    "P3B44_13_reg006": cfg(dropout=0.006, can_drop_path=0.006),
    "P3B44_14_beta045": cfg(can_beta_init=0.45),
    "P3B44_15_beta055": cfg(can_beta_init=0.55),
    "P3B44_16_gamma075": cfg(can_gamma_lr_scale=0.75),
    "P3B44_17_gamma125": cfg(can_gamma_lr_scale=1.25),
    # Light MAPE objectives. These are deliberately mild because pure MAPE
    # losses previously moved too far away from the low-MAE basin.
    "P3B44_20_evalmapemae015": cfg(loss="evalmapemae", loss_mse_weight=0.15),
    "P3B44_21_evalmapemae025": cfg(loss="evalmapemae", loss_mse_weight=0.25),
    "P3B44_22_origmapemae015": cfg(loss="origmapemae", loss_mse_weight=0.15),
    "P3B44_23_origmapemae025": cfg(loss="origmapemae", loss_mse_weight=0.25),
    # Horizon tilts around the last half of the 12-step short horizon.
    "P3B44_30_tail098": cfg(loss_horizon_weight=0.98, loss_horizon_weight_start=6),
    "P3B44_31_tail102": cfg(loss_horizon_weight=1.02, loss_horizon_weight_start=6),
    "P3B44_32_tail098_bias045": cfg(
        can_multiscale_main_bias=0.45,
        loss_horizon_weight=0.98,
        loss_horizon_weight_start=6,
    ),
    "P3B44_33_tail102_bias045": cfg(
        can_multiscale_main_bias=0.45,
        loss_horizon_weight=1.02,
        loss_horizon_weight_start=6,
    ),
    # Small batch-noise changes near the same basin.
    "P3B44_40_bs9": cfg(batch_size=9),
    "P3B44_41_bs11": cfg(batch_size=11),
}


def run_trial(trial: str, values: dict[str, object], gpu: str, summary_name: str) -> None:
    cmd = [
        sys.executable,
        "-u",
        "scripts/short_term_forecast/PEMS/can_pems_epoch_scan.py",
        "--dataset",
        str(values.pop("dataset")),
        "--gpu",
        str(gpu),
        "--trial",
        trial,
        "--summary-name",
        summary_name,
    ]
    for key, value in values.items():
        cli_key = {
            "can_multiscale_patch_lens": "multiscale-patch-lens",
            "can_multiscale_main_bias": "multiscale-main-bias",
        }.get(key, key.replace("_", "-"))
        cmd.extend([f"--{cli_key}", str(value)])
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--trials", nargs="*", default=list(TRIALS))
    parser.add_argument("--summary-name", default="pems03_bias044_mape_refine.csv")
    args = parser.parse_args()

    for trial in args.trials:
        run_trial(trial, dict(TRIALS[trial]), args.gpu, args.summary_name)


if __name__ == "__main__":
    main()
