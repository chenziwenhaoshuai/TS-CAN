#!/usr/bin/env python
import csv
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "ablation_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PYTHON_EXE = sys.executable


def base_cfg():
    return {
        "task_name": "long_term_forecast",
        "is_training": 1,
        "root_path": "./dataset/ETT/",
        "data_path": "ETTh1.csv",
        "model": "CANPatchTST",
        "data": "ETTh1",
        "features": "M",
        "seq_len": 192,
        "label_len": 48,
        "pred_len": 96,
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "e_layers": 2,
        "d_model": 128,
        "d_ff": 128,
        "patch_len": 16,
        "can_stride": 8,
        "can_shifts": "1,2,4,8,16",
        "can_cli_mode": "full",
        "can_temporal_cli_mode": "full",
        "can_ctx_mode": "diff",
        "can_drop_path": 0.05,
        "can_kernel_size": 3,
        "can_use_gffng": 1,
        "can_temporal_roll": 1,
        "can_use_orth": 0,
        "can_context_pyramid": 0,
        "dropout": 0.05,
        "batch_size": 8,
        "learning_rate": 0.00030,
        "lradj": "cosine",
        "train_epochs": 2,
        "patience": 2,
        "des": "ABLATE_GEOM_RELEASE",
        "itr": 1,
        "num_workers": 0,
        "use_amp": True,
        "seed": 2,
    }


def variants():
    return [
        ("baseline_release_best", {}),
        ("temporal_cli_inner", {"can_temporal_cli_mode": "inner"}),
        ("temporal_cli_wedge", {"can_temporal_cli_mode": "wedge"}),
        ("temporal_cli_adaptive", {"can_temporal_cli_mode": "adaptive"}),
        ("ctx_abs", {"can_ctx_mode": "abs"}),
        ("no_global_branch", {"can_use_gffng": 0}),
        ("no_temporal_roll", {"can_temporal_roll": 0}),
        ("orthogonal_context", {"can_use_orth": 1}),
        ("context_pyramid", {"can_context_pyramid": 1}),
        ("shifts_1248", {"can_shifts": "1,2,4,8"}),
        ("shifts_135711", {"can_shifts": "1,3,5,7,11"}),
    ]


def to_cli_args(cfg):
    args = ["run.py"]
    for key, value in cfg.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend([flag, str(value)])
    return args


def parse_metrics(log_text):
    mse = None
    mae = None
    for line in log_text.splitlines():
        if "mse:" in line and "mae:" in line:
            mse = float(line.split("mse:")[1].split(",")[0])
            mae = float(line.split("mae:")[1].split(",")[0])
    return mse, mae


def main():
    raw_path = OUTPUT_DIR / "etth1_96_geometric_option_ablation.csv"
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    done = set()
    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                done.add(row["variant"])

    cfg0 = base_cfg()
    for variant, update in variants():
        if variant in done:
            continue
        cfg = dict(cfg0)
        cfg.update(update)
        cfg["model_id"] = f"ABLATE_{variant}"
        cmd = [PYTHON_EXE, *to_cli_args(cfg)]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        log_text = proc.stdout + "\n" + proc.stderr
        (log_dir / f"{variant}.log").write_text(log_text, encoding="utf-8", errors="ignore")
        mse, mae = parse_metrics(log_text)
        if proc.returncode != 0 or mse is None:
            raise RuntimeError(f"variant {variant} failed")
        rows.append(
            {
                "variant": variant,
                "mse": f"{mse:.12f}",
                "mae": f"{mae:.12f}",
                "seconds": f"{time.time() - t0:.1f}",
            }
        )
        with raw_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["variant", "mse", "mae", "seconds"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved: {raw_path}")


if __name__ == "__main__":
    main()
