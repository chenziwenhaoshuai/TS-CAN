"""
Read-only runtime utilities for autoresearch-style TS-CAN experiments.

The idea matches karpathy/autoresearch:
- `prepare.py` stays fixed during the agent loop
- `train.py` is the single editable experiment file
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.tools import dotdict

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / "ETT-small"
DEFAULT_DATA_PATH = os.environ.get("AUTORESEARCH_DATA_PATH", "ETTh1.csv")
DEFAULT_DATASET = os.environ.get("AUTORESEARCH_DATASET", Path(DEFAULT_DATA_PATH).stem)

TIME_BUDGET_SECONDS = int(os.environ.get("AUTORESEARCH_TIME_BUDGET_SECONDS", "300"))
STARTUP_GRACE_STEPS = int(os.environ.get("AUTORESEARCH_STARTUP_GRACE_STEPS", "2"))
DEFAULT_SEED = int(os.environ.get("AUTORESEARCH_SEED", "2"))

BASE_ARGS: Dict[str, object] = {
    "task_name": "long_term_forecast",
    "is_training": 1,
    "model_id": "TS_CAN_autoresearch",
    "model": "CANPatchTST",
    "data": DEFAULT_DATASET,
    "root_path": str(Path(os.environ.get("AUTORESEARCH_DATA_ROOT", DEFAULT_DATA_ROOT))),
    "data_path": DEFAULT_DATA_PATH,
    "features": os.environ.get("AUTORESEARCH_FEATURES", "M"),
    "target": os.environ.get("AUTORESEARCH_TARGET", "OT"),
    "freq": os.environ.get("AUTORESEARCH_FREQ", "h"),
    "checkpoints": "./checkpoints/",
    "results": "./results/",
    "test_results": "./test_results/",
    "seq_len": 192,
    "label_len": 48,
    "pred_len": 96,
    "inverse": False,
    "d_model": 128,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 128,
    "moving_avg": 25,
    "dropout": 0.05,
    "embed": "timeF",
    "activation": "gelu",
    "factor": 1,
    "distil": True,
    "expand": 2,
    "d_conv": 4,
    "top_k": 5,
    "num_kernels": 6,
    "seasonal_patterns": "Monthly",
    "patch_len": 16,
    "can_stride": 8,
    "can_shifts": "1,2,4,8,16",
    "can_cli_mode": "full",
    "can_temporal_cli_mode": "full",
    "can_ctx_mode": "diff",
    "can_drop_path": 0.05,
    "can_kernel_size": 3,
    "can_init_values": 1e-5,
    "can_use_gffng": 1,
    "can_temporal_roll": 1,
    "can_beta_init": 0.5,
    "can_use_orth": 0,
    "can_context_pyramid": 0,
    "num_workers": 0,
    "itr": 1,
    "train_epochs": 2,
    "batch_size": 8,
    "patience": 2,
    "learning_rate": 3e-4,
    "des": "AUTORESEARCH",
    "loss": "MSE",
    "lradj": "cosine",
    "use_amp": False,
    "use_dtw": False,
    "use_gpu": True,
    "gpu": 0,
    "gpu_type": "cuda",
    "use_multi_gpu": False,
    "devices": "0",
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
    "seed": DEFAULT_SEED,
}


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_data_path(root_path: str, data_path: str) -> Path:
    return Path(root_path).expanduser().resolve() / data_path


def infer_io_dims(root_path: str, data_path: str, features: str, target: str) -> Tuple[int, int, int]:
    csv_path = _resolve_data_path(root_path, data_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}. "
            "Run `python prepare.py` to verify your data path configuration."
        )

    frame = pd.read_csv(csv_path, nrows=1)
    value_columns = [column for column in frame.columns if column != "date"]
    if target not in value_columns:
        raise ValueError(f"Target column `{target}` not found in {csv_path.name}.")

    multivariate_dim = len(value_columns)
    if features == "S":
        return 1, 1, 1
    if features == "MS":
        return multivariate_dim, multivariate_dim, 1
    return multivariate_dim, multivariate_dim, multivariate_dim


def build_args(overrides: Optional[Dict[str, object]] = None) -> dotdict:
    args = dict(BASE_ARGS)
    if overrides:
        args.update(overrides)

    enc_in, dec_in, c_out = infer_io_dims(
        root_path=str(args["root_path"]),
        data_path=str(args["data_path"]),
        features=str(args["features"]),
        target=str(args["target"]),
    )
    args["enc_in"] = enc_in
    args["dec_in"] = dec_in
    args["c_out"] = c_out
    return dotdict(args)


def select_device() -> torch.device:
    requested = os.environ.get("AUTORESEARCH_DEVICE", "auto").lower()

    if requested in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataloader(args: dotdict, flag: str):
    return data_provider(args, flag)


def make_dataloaders(args: dotdict):
    train_data, train_loader = make_dataloader(args, "train")
    val_data, val_loader = make_dataloader(args, "val")
    test_data, test_loader = make_dataloader(args, "test")
    return {
        "train": (train_data, train_loader),
        "val": (val_data, val_loader),
        "test": (test_data, test_loader),
    }


def _autocast_context(device: torch.device, amp_enabled: bool, amp_dtype: Optional[torch.dtype]):
    enabled = amp_enabled and device.type == "cuda" and amp_dtype is not None
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enabled)


def forward_batch(
    model: torch.nn.Module,
    batch,
    args: dotdict,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
):
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    decoder_input = torch.zeros_like(batch_y[:, -args.pred_len :, :], device=device)
    decoder_input = torch.cat([batch_y[:, : args.label_len, :], decoder_input], dim=1)

    with _autocast_context(device, amp_enabled, amp_dtype):
        outputs = model(batch_x, batch_x_mark, decoder_input, batch_y_mark)

    feature_dim = -1 if args.features == "MS" else 0
    outputs = outputs[:, -args.pred_len :, feature_dim:]
    targets = batch_y[:, -args.pred_len :, feature_dim:]
    return outputs, targets


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    args: dotdict,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    preds = []
    trues = []
    model.eval()

    for batch in loader:
        outputs, targets = forward_batch(
            model=model,
            batch=batch,
            args=args,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        preds.append(outputs.detach().cpu().numpy())
        trues.append(targets.detach().cpu().numpy())

    preds_np = np.concatenate(preds, axis=0)
    trues_np = np.concatenate(trues, axis=0)
    mae, mse, rmse, mape, mspe = metric(preds_np, trues_np)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "mspe": float(mspe),
    }


def verify_environment() -> Dict[str, object]:
    args = build_args()
    dataset_file = _resolve_data_path(args.root_path, args.data_path)
    return {
        "repo_root": str(REPO_ROOT),
        "dataset_file": str(dataset_file),
        "dataset_name": args.data,
        "time_budget_seconds": TIME_BUDGET_SECONDS,
        "device": str(select_device()),
        "enc_in": args.enc_in,
        "c_out": args.c_out,
    }


def main() -> None:
    env = verify_environment()
    print("Autoresearch runtime is ready.")
    print(f"repo_root:           {env['repo_root']}")
    print(f"dataset_name:        {env['dataset_name']}")
    print(f"dataset_file:        {env['dataset_file']}")
    print(f"time_budget_seconds: {env['time_budget_seconds']}")
    print(f"default_device:      {env['device']}")
    print(f"enc_in:              {env['enc_in']}")
    print(f"c_out:               {env['c_out']}")


if __name__ == "__main__":
    main()
