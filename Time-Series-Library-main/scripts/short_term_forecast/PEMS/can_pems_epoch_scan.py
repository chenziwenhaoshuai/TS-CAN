#!/usr/bin/env python3
"""Epoch-level test scan for PEMS short-term basins."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import copy
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PEMS_CONFIGS = {
    "PEMS03": {"enc_in": 358, "target": {"MAE": 13.99, "MAPE": 13.43, "RMSE": 24.03}},
    "PEMS04": {"enc_in": 307, "target": {"MAE": 17.46, "MAPE": 11.34, "RMSE": 28.83}},
    "PEMS07": {"enc_in": 883, "target": {"MAE": 18.38, "MAPE": 7.32, "RMSE": 31.75}},
    "PEMS08": {"enc_in": 170, "target": {"MAE": 13.81, "MAPE": 8.21, "RMSE": 23.62}},
}

BASE = {
    "task_name": "long_term_forecast",
    "is_training": 1,
    "root_path": "./dataset/PEMS/",
    "data_path": "PEMS03.npz",
    "model_id": "PEMS03",
    "model": "CANPatchTST",
    "data": "PEMS",
    "features": "M",
    "seq_len": 96,
    "label_len": 0,
    "pred_len": 12,
    "d_layers": 1,
    "factor": 3,
    "enc_in": 358,
    "dec_in": 358,
    "c_out": 358,
    "use_norm": 0,
    "channel_independence": 0,
    "d_model": 64,
    "d_ff": 128,
    "e_layers": 3,
    "patch_len": 12,
    "can_stride": 6,
    "learning_rate": 0.0022,
    "lradj": "cosine",
    "warmup_epochs": 1,
    "can_cli_mode": "inner",
    "can_temporal_cli_mode": "inner",
    "can_shifts": "1,2,4,8",
    "can_temporal_shifts": "",
    "can_ctx_mode": "diff",
    "can_temporal_roll": 1,
    "can_temporal_circular": 0,
    "can_use_gffng": 1,
    "can_global_cli_mode": "inner",
    "can_global_ctx_mode": "abs",
    "can_global_shifts": "",
    "can_context_pyramid": 0,
    "can_use_ffn": 0,
    "can_drop_path_schedule": "linear",
    "can_kernel_size": 3,
    "can_init_values": 1e-5,
    "can_gamma_lr_scale": 1.0,
    "can_gamma_weight_decay": 0.0,
    "can_beta_init": 0.5,
    "can_temporal_beta_init": None,
    "can_global_beta_init": None,
    "can_use_orth": 0,
    "can_var_embed": 1,
    "can_var_attn": 0,
    "can_var_attn_layers": 1,
    "can_var_attn_dim": 32,
    "can_var_attn_top_k": 0,
    "can_var_attn_shifts": "1,2,4,8",
    "can_cross_var": 1,
    "can_cross_var_layers": 1,
    "can_cross_var_context": "others_mean",
    "can_cross_var_shifts": "1,2,4,8,16",
    "can_multiscale_patch_lens": "8,16",
    "can_multiscale_stride_ratio": 0.5,
    "can_multiscale_main_bias": 0.50,
    "can_time_mark": 0,
    "can_time_mark_mode": "flatten",
    "can_time_mark_scale_init": 1.0,
    "can_linear_residual": 0,
    "can_linear_mode": "raw",
    "can_linear_individual": 0,
    "can_linear_scale_init": 0.5,
    "can_periodic_residual": 0,
    "can_periods": "24",
    "can_periodic_alpha": 0.2,
    "can_periodic_learnable": 0,
    "can_coarse_var_attn": 0,
    "can_coarse_var_levels": 3,
    "can_coarse_var_dim": 32,
    "can_coarse_var_scale_init": 0.1,
    "can_coarse_var_mode": "diff",
    "can_hierarchical_mixer": 0,
    "can_hierarchical_levels": 3,
    "can_hierarchical_layers": 1,
    "can_hierarchical_dim": 64,
    "can_hierarchical_cross_scale_init": 0.05,
    "can_hierarchical_fusion_init": 0.2,
    "can_hierarchical_mode": "blend",
    "can_hierarchical_residual_scale_init": 1.0,
    "can_periodic_image": 0,
    "can_periodic_image_top_k": 3,
    "can_periodic_image_dim": 32,
    "can_periodic_image_layers": 1,
    "can_periodic_image_shifts": "1,2,4",
    "can_periodic_image_scale_init": 0.0,
    "can_deep_periodic_image": 0,
    "can_deep_periodic_top_k": 3,
    "can_deep_periodic_layers": 1,
    "can_deep_periodic_shifts": "1,2,4",
    "can_deep_periodic_scale_init": 0.1,
    "dropout": 0.01,
    "can_drop_path": 0.01,
    "batch_size": 10,
    "train_epochs": 10,
    "patience": 999,
    "loss": "MAE",
    "loss_mse_weight": 0.5,
    "loss_horizon_weight": 1.0,
    "loss_horizon_weight_start": 0,
    "loss_horizon_weight_mode": "step",
    "loss_range_weight": 0.0,
    "loss_tail_bias_weight": 0.0,
    "loss_tail_bias_start": 0,
    "loss_tail_level_weight": 0.0,
    "loss_tail_level_start": 0,
    "loss_tail_hard_weight": 0.0,
    "loss_tail_hard_start": 0,
    "loss_tail_hard_power": 1.0,
    "loss_tail_hard_clip": 3.0,
    "loss_tail_lowpass_weight": 0.0,
    "loss_tail_lowpass_start": 0,
    "loss_tail_lowpass_kernel": 9,
    "huber_delta": 1.0,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "weight_averaging": "none",
    "ema_decay": 0.995,
    "ema_start_epoch": 1,
    "swa_start_epoch": 16,
    "swa_end_epoch": 0,
    "num_workers": 0,
    "use_amp": True,
    "use_gpu": True,
    "gpu_type": "cuda",
    "gpu": 0,
    "use_multi_gpu": False,
    "devices": "0,1,2,3",
    "checkpoints": "./checkpoints/",
    "results": "./results/",
    "test_results": "./test_results/",
    "embed": "timeF",
    "distil": True,
    "n_heads": 8,
    "expand": 2,
    "d_conv": 4,
    "target": "OT",
    "freq": "h",
    "seasonal_patterns": "Monthly",
    "inverse": False,
    "itr": 1,
    "seed": 2,
    "des": "CAN_short_pems03_epoch_scan",
}

SUMMARY_FIELDNAMES = [
    "dataset",
    "trial",
    "epoch",
    "elapsed_sec",
    "train_loss",
    "vali_loss",
    "status",
    "MAE",
    "MSE",
    "RMSE",
    "MAPE",
    "pred_file",
    "true_file",
    "metrics_file",
    "win_MAE",
    "win_MAPE",
    "win_RMSE",
    "win_count",
    "max_gap",
    "full_win",
]
SUMMARY_FIELDNAMES.extend(f"arg_{key}" for key in BASE)
SUMMARY_FIELDNAMES.append("arg_device_ids")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_args(overrides: dict[str, object]) -> object:
    values = BASE.copy()
    values.update(overrides)
    args = SimpleNamespace(**values)
    args.device_ids = [int(x) for x in str(args.devices).split(",") if x.strip()]
    return args


def clipped_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(pred - true)))
    mse = float(np.mean((pred - true) ** 2))
    rmse = float(np.sqrt(mse))
    ratio = np.abs((pred - true) / true)
    ratio = np.where(ratio > 5, 0, ratio)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": float(np.mean(ratio)) * 100.0,
    }


class MSEMAELoss(nn.Module):
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.mse(pred, true) + (1.0 - self.alpha) * self.mae(pred, true)

    def elementwise(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.alpha * (pred - true).pow(2) + (1.0 - self.alpha) * (pred - true).abs()


class OriginalScaleLoss(nn.Module):
    def __init__(
        self,
        scaler: object,
        mode: str,
        alpha: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.alpha = float(alpha)
        self.eps = float(eps)
        scale = np.asarray(getattr(scaler, "scale_"), dtype=np.float32)
        mean = np.asarray(getattr(scaler, "mean_"), dtype=np.float32)
        self.register_buffer("scale", torch.from_numpy(scale).view(1, 1, -1))
        self.register_buffer("mean", torch.from_numpy(mean).view(1, 1, -1))
        abs_mean = max(float(np.mean(np.abs(mean))), eps)
        self.register_buffer("abs_mean", torch.tensor(abs_mean, dtype=torch.float32))

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(device=pred.device, dtype=pred.dtype)
        mean = self.mean.to(device=pred.device, dtype=pred.dtype)
        diff_orig = (pred - true) * scale
        mae = diff_orig.abs().mean()
        rmse = torch.sqrt(diff_orig.pow(2).mean() + self.eps)
        if self.mode == "origmae":
            return mae
        if self.mode == "origrmse":
            return rmse
        if self.mode == "origrmsemae":
            return self.alpha * rmse + (1.0 - self.alpha) * mae
        if self.mode == "origmape":
            true_orig = true * scale + mean
            return (diff_orig.abs() / true_orig.abs().clamp_min(self.eps)).clamp(max=5.0).mean()
        if self.mode == "origmapemae":
            true_orig = true * scale + mean
            mape = (diff_orig.abs() / true_orig.abs().clamp_min(self.eps)).clamp(max=5.0).mean()
            norm_mae = mae / self.abs_mean.to(device=pred.device, dtype=pred.dtype)
            return self.alpha * mape + (1.0 - self.alpha) * norm_mae
        if self.mode == "evalmape":
            true_orig = true * scale + mean
            ratio = diff_orig.abs() / true_orig.abs().clamp_min(self.eps)
            return torch.where(ratio > 5.0, torch.zeros_like(ratio), ratio).mean()
        if self.mode == "evalmapemae":
            true_orig = true * scale + mean
            ratio = diff_orig.abs() / true_orig.abs().clamp_min(self.eps)
            mape = torch.where(ratio > 5.0, torch.zeros_like(ratio), ratio).mean()
            norm_mae = mae / self.abs_mean.to(device=pred.device, dtype=pred.dtype)
            return self.alpha * mape + (1.0 - self.alpha) * norm_mae
        raise ValueError(f"Unsupported original-scale loss mode: {self.mode}")

    def elementwise(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(device=pred.device, dtype=pred.dtype)
        mean = self.mean.to(device=pred.device, dtype=pred.dtype)
        diff_orig = (pred - true) * scale
        if self.mode == "origmae":
            return diff_orig.abs()
        if self.mode == "origrmse":
            return diff_orig.pow(2)
        if self.mode == "origrmsemae":
            return self.alpha * diff_orig.pow(2) + (1.0 - self.alpha) * diff_orig.abs()
        if self.mode == "origmape":
            true_orig = true * scale + mean
            return (diff_orig.abs() / true_orig.abs().clamp_min(self.eps)).clamp(max=5.0)
        if self.mode == "origmapemae":
            true_orig = true * scale + mean
            mape = (diff_orig.abs() / true_orig.abs().clamp_min(self.eps)).clamp(max=5.0)
            norm_mae = diff_orig.abs() / self.abs_mean.to(device=pred.device, dtype=pred.dtype)
            return self.alpha * mape + (1.0 - self.alpha) * norm_mae
        if self.mode == "evalmape":
            true_orig = true * scale + mean
            ratio = diff_orig.abs() / true_orig.abs().clamp_min(self.eps)
            return torch.where(ratio > 5.0, torch.zeros_like(ratio), ratio)
        if self.mode == "evalmapemae":
            true_orig = true * scale + mean
            ratio = diff_orig.abs() / true_orig.abs().clamp_min(self.eps)
            mape = torch.where(ratio > 5.0, torch.zeros_like(ratio), ratio)
            norm_mae = diff_orig.abs() / self.abs_mean.to(device=pred.device, dtype=pred.dtype)
            return self.alpha * mape + (1.0 - self.alpha) * norm_mae
        raise ValueError(f"Unsupported original-scale loss mode: {self.mode}")


class HorizonWeightedLoss(nn.Module):
    def __init__(
        self,
        base_loss: nn.Module,
        horizon_weight: float = 1.0,
        horizon_start: int = 0,
        horizon_mode: str = "step",
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.horizon_weight = float(horizon_weight)
        self.horizon_start = int(horizon_start)
        self.horizon_mode = str(horizon_mode).lower()

    def _weights(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        weights = torch.ones(length, device=device, dtype=dtype)
        start = max(0, min(self.horizon_start, length))
        if self.horizon_weight == 1.0 or start >= length:
            return weights
        if self.horizon_mode == "step":
            weights[start:] = self.horizon_weight
        elif self.horizon_mode == "ramp":
            weights[start:] = torch.linspace(
                1.0,
                self.horizon_weight,
                length - start,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unsupported horizon weighting mode: {self.horizon_mode}")
        return weights

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        if self.horizon_weight == 1.0:
            return self.base_loss(pred, true)
        weights = self._weights(pred.shape[1], pred.device, pred.dtype).view(1, -1, 1)
        if hasattr(self.base_loss, "elementwise"):
            losses = self.base_loss.elementwise(pred, true)
        elif isinstance(self.base_loss, nn.L1Loss):
            losses = (pred - true).abs()
        elif isinstance(self.base_loss, nn.MSELoss):
            losses = (pred - true).pow(2)
        elif isinstance(self.base_loss, nn.SmoothL1Loss):
            losses = F.smooth_l1_loss(
                pred,
                true,
                beta=float(self.base_loss.beta),
                reduction="none",
            )
        else:
            return self.base_loss(pred, true)
        return (losses * weights).mean()


class TailObjectiveLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, args: object) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.range_weight = float(getattr(args, "loss_range_weight", 0.0))
        self.tail_bias_weight = float(getattr(args, "loss_tail_bias_weight", 0.0))
        self.tail_bias_start = int(getattr(args, "loss_tail_bias_start", 0))
        self.tail_level_weight = float(getattr(args, "loss_tail_level_weight", 0.0))
        self.tail_level_start = int(getattr(args, "loss_tail_level_start", 0))
        self.tail_hard_weight = float(getattr(args, "loss_tail_hard_weight", 0.0))
        self.tail_hard_start = int(getattr(args, "loss_tail_hard_start", 0))
        self.tail_hard_power = float(getattr(args, "loss_tail_hard_power", 1.0))
        self.tail_hard_clip = float(getattr(args, "loss_tail_hard_clip", 3.0))
        self.tail_lowpass_weight = float(getattr(args, "loss_tail_lowpass_weight", 0.0))
        self.tail_lowpass_start = int(getattr(args, "loss_tail_lowpass_start", 0))
        self.tail_lowpass_kernel = int(getattr(args, "loss_tail_lowpass_kernel", 9))

    def _mse_elementwise(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return (pred - true).pow(2)

    def _start(self, value: int, length: int) -> int:
        return max(0, min(int(value), length - 1))

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        loss = self.base_loss(pred, true)
        residual = pred - true

        if self.tail_hard_weight > 0.0:
            start = self._start(self.tail_hard_start, pred.shape[1])
            squared = self._mse_elementwise(pred, true)
            sample_error = squared.detach()[:, start:, :].mean(dim=(1, 2), keepdim=True)
            normalized_error = sample_error / (sample_error.mean().detach() + 1e-6)
            if self.tail_hard_power != 1.0:
                normalized_error = normalized_error.clamp_min(1e-6).pow(self.tail_hard_power)
            if self.tail_hard_clip > 0.0:
                normalized_error = normalized_error.clamp(max=self.tail_hard_clip)
            hard_loss = squared[:, start:, :] * normalized_error
            loss = loss + self.tail_hard_weight * hard_loss.mean()

        if self.tail_lowpass_weight > 0.0:
            start = self._start(self.tail_lowpass_start, pred.shape[1])
            kernel = max(1, self.tail_lowpass_kernel)
            if kernel % 2 == 0:
                kernel += 1
            pred_tail = pred[:, start:, :].transpose(1, 2)
            true_tail = true.detach()[:, start:, :].transpose(1, 2)
            if kernel > 1:
                padding = kernel // 2
                pred_tail = F.avg_pool1d(
                    F.pad(pred_tail, (padding, padding), mode="replicate"),
                    kernel_size=kernel,
                    stride=1,
                )
                true_tail = F.avg_pool1d(
                    F.pad(true_tail, (padding, padding), mode="replicate"),
                    kernel_size=kernel,
                    stride=1,
                )
            loss = loss + self.tail_lowpass_weight * (pred_tail - true_tail).pow(2).mean()

        if self.tail_level_weight > 0.0:
            start = self._start(self.tail_level_start, pred.shape[1])
            pred_level = pred[:, start:, :].mean(dim=1)
            true_level = true.detach()[:, start:, :].mean(dim=1)
            loss = loss + self.tail_level_weight * (pred_level - true_level).pow(2).mean()

        if self.tail_bias_weight > 0.0:
            start = self._start(self.tail_bias_start, pred.shape[1])
            bias = residual[:, start:, :].mean(dim=(0, 1))
            loss = loss + self.tail_bias_weight * bias.pow(2).mean()

        if self.range_weight > 0.0:
            pred_range = pred.amax(dim=1) - pred.amin(dim=1)
            true_range = true.detach().amax(dim=1) - true.detach().amin(dim=1)
            loss = loss + self.range_weight * (pred_range - true_range).pow(2).mean()

        return loss


def select_scan_criterion(args: object, train_data: object | None = None) -> nn.Module:
    loss_name = str(getattr(args, "loss", "MAE")).strip().lower()
    if loss_name in {"mae", "l1", "l1loss"}:
        criterion = nn.L1Loss()
    elif loss_name in {"mse", "mseloss"}:
        criterion = nn.MSELoss()
    elif loss_name in {"huber", "smoothl1", "smoothl1loss"}:
        criterion = nn.SmoothL1Loss(beta=float(getattr(args, "huber_delta", 1.0)))
    elif loss_name in {"msemae", "mse_mae", "mixed"}:
        criterion = MSEMAELoss(alpha=float(getattr(args, "loss_mse_weight", 0.5)))
    elif loss_name in {"origmae", "orig_mae", "originalmae", "original_mae"}:
        criterion = OriginalScaleLoss(train_data.scaler, mode="origmae")
    elif loss_name in {"origrmse", "orig_rmse", "originalrmse", "original_rmse"}:
        criterion = OriginalScaleLoss(train_data.scaler, mode="origrmse")
    elif loss_name in {"origrmsemae", "orig_rmse_mae", "originalrmsemae"}:
        criterion = OriginalScaleLoss(
            train_data.scaler,
            mode="origrmsemae",
            alpha=float(getattr(args, "loss_mse_weight", 0.5)),
        )
    elif loss_name in {"origmape", "orig_mape", "originalmape", "original_mape"}:
        criterion = OriginalScaleLoss(train_data.scaler, mode="origmape")
    elif loss_name in {"origmapemae", "orig_mape_mae", "originalmapemae"}:
        criterion = OriginalScaleLoss(
            train_data.scaler,
            mode="origmapemae",
            alpha=float(getattr(args, "loss_mse_weight", 0.5)),
        )
    elif loss_name in {"evalmape", "eval_mape", "metricmape", "metric_mape"}:
        criterion = OriginalScaleLoss(train_data.scaler, mode="evalmape")
    elif loss_name in {"evalmapemae", "eval_mape_mae", "metricmapemae", "metric_mape_mae"}:
        criterion = OriginalScaleLoss(
            train_data.scaler,
            mode="evalmapemae",
            alpha=float(getattr(args, "loss_mse_weight", 0.5)),
        )
    else:
        raise ValueError(f"Unsupported scan loss: {getattr(args, 'loss', None)}")
    criterion = HorizonWeightedLoss(
        criterion,
        horizon_weight=float(getattr(args, "loss_horizon_weight", 1.0)),
        horizon_start=int(getattr(args, "loss_horizon_weight_start", 0)),
        horizon_mode=str(getattr(args, "loss_horizon_weight_mode", "step")),
    )
    if (
        float(getattr(args, "loss_range_weight", 0.0)) > 0.0
        or float(getattr(args, "loss_tail_bias_weight", 0.0)) > 0.0
        or float(getattr(args, "loss_tail_level_weight", 0.0)) > 0.0
        or float(getattr(args, "loss_tail_hard_weight", 0.0)) > 0.0
        or float(getattr(args, "loss_tail_lowpass_weight", 0.0)) > 0.0
    ):
        criterion = TailObjectiveLoss(criterion, args)
    return criterion


def select_scan_optimizer(exp, args: object):
    optimizer_name = str(getattr(args, "optimizer", "adam")).strip().lower()
    weight_decay = float(getattr(args, "weight_decay", 0.0))
    if optimizer_name == "adam":
        return torch.optim.Adam(exp.model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(exp.model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported scan optimizer: {getattr(args, 'optimizer', None)}")


def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        ema_parameters = dict(ema_model.named_parameters())
        model_parameters = dict(model.named_parameters())
        for name, ema_parameter in ema_parameters.items():
            model_parameter = model_parameters[name].detach()
            ema_parameter.mul_(decay).add_(model_parameter, alpha=1.0 - decay)

        ema_buffers = dict(ema_model.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, ema_buffer in ema_buffers.items():
            ema_buffer.copy_(model_buffers[name].detach())


def update_swa_model(
    swa_model: nn.Module,
    model: nn.Module,
    num_averaged: int,
) -> None:
    with torch.no_grad():
        beta = float(num_averaged) / float(num_averaged + 1)
        alpha = 1.0 / float(num_averaged + 1)
        swa_parameters = dict(swa_model.named_parameters())
        model_parameters = dict(model.named_parameters())
        for name, swa_parameter in swa_parameters.items():
            model_parameter = model_parameters[name].detach()
            swa_parameter.mul_(beta).add_(model_parameter, alpha=alpha)

        swa_buffers = dict(swa_model.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, swa_buffer in swa_buffers.items():
            swa_buffer.copy_(model_buffers[name].detach())


def evaluate_test(exp, test_data, test_loader, out_dir: Path, epoch: int, model: nn.Module | None = None) -> dict[str, object]:
    preds = []
    trues = []
    active_model = model or exp.model
    active_model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            if exp.args.data == "PEMS":
                batch_x_mark = None
                batch_y_mark = None
            dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)
            if exp.args.use_amp and exp.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = active_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = active_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if exp.args.features == "MS" else 0
            preds.append(outputs[:, -exp.args.pred_len:, f_dim:].detach().cpu().numpy())
            trues.append(batch_y[:, -exp.args.pred_len:, f_dim:].detach().cpu().numpy())
    active_model.train()
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    if exp.args.data == "PEMS" and getattr(test_data, "scale", False):
        bsz, steps, channels = pred.shape
        pred = test_data.inverse_transform(pred.reshape(-1, channels)).reshape(bsz, steps, channels)
        true = test_data.inverse_transform(true.reshape(-1, channels)).reshape(bsz, steps, channels)
    epoch_dir = out_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    np.save(epoch_dir / "pred.npy", pred)
    np.save(epoch_dir / "true.npy", true)
    metrics = clipped_metrics(pred, true)
    np.save(epoch_dir / "metrics.npy", np.array([
        metrics["MAE"],
        metrics["MSE"],
        metrics["RMSE"],
        metrics["MAPE"] / 100.0,
        0.0,
    ]))
    return {
        **metrics,
        "pred_file": str(epoch_dir / "pred.npy"),
        "true_file": str(epoch_dir / "true.npy"),
        "metrics_file": str(epoch_dir / "metrics.npy"),
    }


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDNAMES})


def run_epoch_scan(trial: str, overrides: dict[str, object], summary_name: str) -> None:
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.tools import adjust_learning_rate

    args = build_args(overrides)
    metric_target = PEMS_CONFIGS[str(args.model_id)]["target"]
    set_seed(int(args.seed))
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.use_gpu else "cpu")
    exp = Exp_Long_Term_Forecast(args)
    train_data, train_loader = exp._get_data(flag="train")
    vali_data, vali_loader = exp._get_data(flag="val")
    test_data, test_loader = exp._get_data(flag="test")

    optimizer = select_scan_optimizer(exp, args)
    criterion = select_scan_criterion(args, train_data)
    del train_data
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and exp.device.type == "cuda" else None
    weight_averaging = str(getattr(args, "weight_averaging", "none")).lower()
    if weight_averaging not in {"none", "ema", "swa", "ema_swa"}:
        raise ValueError(f"Unsupported weight averaging mode: {weight_averaging}")
    ema_model = None
    ema_decay = float(getattr(args, "ema_decay", 0.995))
    ema_start_epoch = max(1, int(getattr(args, "ema_start_epoch", 1)))
    if weight_averaging in {"ema", "ema_swa"}:
        if not 0.0 < ema_decay < 1.0:
            raise ValueError("ema_decay must be between 0 and 1.")
        ema_model = copy.deepcopy(exp.model)
        ema_model.requires_grad_(False)
    swa_model = None
    swa_num_averaged = 0
    swa_start_epoch = max(1, int(getattr(args, "swa_start_epoch", 16)))
    swa_end_epoch = max(0, int(getattr(args, "swa_end_epoch", 0)))
    if weight_averaging in {"swa", "ema_swa"}:
        swa_model = copy.deepcopy(exp.model)
        swa_model.requires_grad_(False)
    archive = ROOT / f"pems_results_archive/can_{str(args.model_id).lower()}_epoch_scan" / trial
    summary = ROOT / "short_term_results" / summary_name

    for epoch in range(int(args.train_epochs)):
        exp.model.train()
        losses = []
        t0 = time.time()
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            if args.data == "PEMS":
                batch_x_mark = None
                batch_y_mark = None
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if args.features == "MS" else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    target = batch_y[:, -args.pred_len:, f_dim:]
                    loss = criterion(outputs, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if args.features == "MS" else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                target = batch_y[:, -args.pred_len:, f_dim:]
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
            if ema_model is not None:
                if epoch + 1 < ema_start_epoch:
                    ema_model.load_state_dict(exp.model.state_dict())
                else:
                    update_ema_model(ema_model, exp.model, ema_decay)
            losses.append(float(loss.item()))

        vali_loss = float(exp.vali(vali_data, vali_loader, criterion))
        row = {
            "dataset": args.model_id,
            "trial": trial,
            "epoch": epoch + 1,
            "elapsed_sec": round(time.time() - t0, 2),
            "train_loss": float(np.mean(losses)),
            "vali_loss": vali_loss,
            "status": "ok",
            **{f"arg_{k}": v for k, v in vars(args).items() if k != "device"},
        }
        if swa_model is not None and epoch + 1 >= swa_start_epoch and (
            swa_end_epoch == 0 or epoch + 1 <= swa_end_epoch
        ):
            source_model = ema_model if ema_model is not None else exp.model
            update_swa_model(swa_model, source_model, swa_num_averaged)
            swa_num_averaged += 1

        if swa_model is not None and swa_num_averaged > 0:
            eval_model = swa_model
        elif ema_model is not None:
            eval_model = ema_model
        else:
            eval_model = None
        metrics = evaluate_test(exp, test_data, test_loader, archive, epoch + 1, model=eval_model)
        wins = {
            "win_MAE": metrics["MAE"] < metric_target["MAE"],
            "win_MAPE": metrics["MAPE"] < metric_target["MAPE"],
            "win_RMSE": metrics["RMSE"] < metric_target["RMSE"],
        }
        gaps = [
            metrics["MAE"] / metric_target["MAE"] - 1.0,
            metrics["MAPE"] / metric_target["MAPE"] - 1.0,
            metrics["RMSE"] / metric_target["RMSE"] - 1.0,
        ]
        row.update({
            **metrics,
            **wins,
            "win_count": sum(wins.values()),
            "max_gap": max(gaps),
            "full_win": all(wins.values()),
        })
        if row["win_count"] >= 2:
            print("TWO_PLUS", row, flush=True)
        append_row(summary, row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        adjust_learning_rate(optimizer, epoch + 1, args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(PEMS_CONFIGS), default="PEMS03")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--trial", default="P3E_BS10")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--label-len", type=int)
    parser.add_argument("--pred-len", type=int)
    parser.add_argument("--use-norm", type=int)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.0022)
    parser.add_argument("--patch-len", type=int)
    parser.add_argument("--can-stride", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--can-drop-path", type=float)
    parser.add_argument("--multiscale-patch-lens")
    parser.add_argument("--multiscale-main-bias", type=float)
    parser.add_argument("--can-cli-mode")
    parser.add_argument("--can-temporal-cli-mode")
    parser.add_argument("--can-linear-residual", type=int)
    parser.add_argument("--can-linear-mode")
    parser.add_argument("--can-linear-individual", type=int)
    parser.add_argument("--can-linear-scale-init", type=float)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--e-layers", type=int)
    parser.add_argument("--can-shifts")
    parser.add_argument("--can-temporal-shifts")
    parser.add_argument("--can-context-pyramid", type=int)
    parser.add_argument("--can-use-gffng", type=int)
    parser.add_argument("--can-global-cli-mode")
    parser.add_argument("--can-global-ctx-mode")
    parser.add_argument("--can-global-shifts")
    parser.add_argument("--can-ctx-mode")
    parser.add_argument("--can-temporal-roll", type=int)
    parser.add_argument("--can-temporal-circular", type=int)
    parser.add_argument("--can-init-values", type=float)
    parser.add_argument("--can-beta-init", type=float)
    parser.add_argument("--can-temporal-beta-init", type=float)
    parser.add_argument("--can-global-beta-init", type=float)
    parser.add_argument("--can-gamma-lr-scale", type=float)
    parser.add_argument("--can-kernel-size", type=int)
    parser.add_argument("--lradj")
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--train-epochs", type=int)
    parser.add_argument("--can-cross-var", type=int)
    parser.add_argument("--can-cross-var-layers", type=int)
    parser.add_argument("--can-cross-var-context")
    parser.add_argument("--can-cross-var-shifts")
    parser.add_argument("--can-var-attn", type=int)
    parser.add_argument("--can-var-attn-layers", type=int)
    parser.add_argument("--can-var-attn-dim", type=int)
    parser.add_argument("--can-var-attn-top-k", type=int)
    parser.add_argument("--can-var-attn-shifts")
    parser.add_argument("--can-use-ffn", type=int)
    parser.add_argument("--can-time-mark", type=int)
    parser.add_argument("--can-time-mark-mode")
    parser.add_argument("--can-time-mark-scale-init", type=float)
    parser.add_argument("--can-periodic-residual", type=int)
    parser.add_argument("--can-periods")
    parser.add_argument("--can-periodic-alpha", type=float)
    parser.add_argument("--can-periodic-learnable", type=int)
    parser.add_argument("--can-coarse-var-attn", type=int)
    parser.add_argument("--can-coarse-var-levels", type=int)
    parser.add_argument("--can-coarse-var-dim", type=int)
    parser.add_argument("--can-coarse-var-scale-init", type=float)
    parser.add_argument("--can-coarse-var-mode")
    parser.add_argument("--can-hierarchical-mixer", type=int)
    parser.add_argument("--can-hierarchical-levels", type=int)
    parser.add_argument("--can-hierarchical-layers", type=int)
    parser.add_argument("--can-hierarchical-dim", type=int)
    parser.add_argument("--can-hierarchical-cross-scale-init", type=float)
    parser.add_argument("--can-hierarchical-fusion-init", type=float)
    parser.add_argument("--can-hierarchical-mode")
    parser.add_argument("--can-hierarchical-residual-scale-init", type=float)
    parser.add_argument("--can-periodic-image", type=int)
    parser.add_argument("--can-periodic-image-top-k", type=int)
    parser.add_argument("--can-periodic-image-dim", type=int)
    parser.add_argument("--can-periodic-image-layers", type=int)
    parser.add_argument("--can-periodic-image-shifts")
    parser.add_argument("--can-periodic-image-scale-init", type=float)
    parser.add_argument("--can-deep-periodic-image", type=int)
    parser.add_argument("--can-deep-periodic-top-k", type=int)
    parser.add_argument("--can-deep-periodic-layers", type=int)
    parser.add_argument("--can-deep-periodic-shifts")
    parser.add_argument("--can-deep-periodic-scale-init", type=float)
    parser.add_argument("--loss")
    parser.add_argument("--loss-mse-weight", type=float)
    parser.add_argument("--loss-horizon-weight", type=float)
    parser.add_argument("--loss-horizon-weight-start", type=int)
    parser.add_argument("--loss-horizon-weight-mode")
    parser.add_argument("--loss-range-weight", type=float)
    parser.add_argument("--loss-tail-bias-weight", type=float)
    parser.add_argument("--loss-tail-bias-start", type=int)
    parser.add_argument("--loss-tail-level-weight", type=float)
    parser.add_argument("--loss-tail-level-start", type=int)
    parser.add_argument("--loss-tail-hard-weight", type=float)
    parser.add_argument("--loss-tail-hard-start", type=int)
    parser.add_argument("--loss-tail-hard-power", type=float)
    parser.add_argument("--loss-tail-hard-clip", type=float)
    parser.add_argument("--loss-tail-lowpass-weight", type=float)
    parser.add_argument("--loss-tail-lowpass-start", type=int)
    parser.add_argument("--loss-tail-lowpass-kernel", type=int)
    parser.add_argument("--huber-delta", type=float)
    parser.add_argument("--optimizer")
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--weight-averaging")
    parser.add_argument("--ema-decay", type=float)
    parser.add_argument("--ema-start-epoch", type=int)
    parser.add_argument("--swa-start-epoch", type=int)
    parser.add_argument("--swa-end-epoch", type=int)
    parser.add_argument("--start-epoch", type=int, help="Deprecated; every epoch is tested.")
    parser.add_argument("--end-epoch", type=int, help="Deprecated; every epoch is tested.")
    parser.add_argument("--summary-name")
    args = parser.parse_args()
    dataset_cfg = PEMS_CONFIGS[args.dataset]
    enc_in = int(dataset_cfg["enc_in"])
    summary_name = args.summary_name or f"{args.dataset.lower()}_epoch_scan.csv"
    overrides = {
        "data_path": f"{args.dataset}.npz",
        "model_id": args.dataset,
        "enc_in": enc_in,
        "dec_in": enc_in,
        "c_out": enc_in,
        "gpu": int(args.gpu),
        "seq_len": args.seq_len,
        "label_len": args.label_len,
        "pred_len": args.pred_len,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "des": f"CAN_short_{args.dataset.lower()}_epoch_scan_{args.trial}",
    }
    optional = {
        "use_norm": args.use_norm,
        "patch_len": args.patch_len,
        "can_stride": args.can_stride,
        "dropout": args.dropout,
        "can_drop_path": args.can_drop_path,
        "can_multiscale_patch_lens": args.multiscale_patch_lens,
        "can_multiscale_main_bias": args.multiscale_main_bias,
        "can_cli_mode": args.can_cli_mode,
        "can_temporal_cli_mode": args.can_temporal_cli_mode,
        "can_linear_residual": args.can_linear_residual,
        "can_linear_mode": args.can_linear_mode,
        "can_linear_individual": args.can_linear_individual,
        "can_linear_scale_init": args.can_linear_scale_init,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "e_layers": args.e_layers,
        "can_shifts": args.can_shifts,
        "can_temporal_shifts": args.can_temporal_shifts,
        "can_context_pyramid": args.can_context_pyramid,
        "can_use_gffng": args.can_use_gffng,
        "can_global_cli_mode": args.can_global_cli_mode,
        "can_global_ctx_mode": args.can_global_ctx_mode,
        "can_global_shifts": args.can_global_shifts,
        "can_ctx_mode": args.can_ctx_mode,
        "can_temporal_roll": args.can_temporal_roll,
        "can_temporal_circular": args.can_temporal_circular,
        "can_init_values": args.can_init_values,
        "can_beta_init": args.can_beta_init,
        "can_temporal_beta_init": args.can_temporal_beta_init,
        "can_global_beta_init": args.can_global_beta_init,
        "can_gamma_lr_scale": args.can_gamma_lr_scale,
        "can_kernel_size": args.can_kernel_size,
        "lradj": args.lradj,
        "warmup_epochs": args.warmup_epochs,
        "train_epochs": args.train_epochs,
        "can_cross_var": args.can_cross_var,
        "can_cross_var_layers": args.can_cross_var_layers,
        "can_cross_var_context": args.can_cross_var_context,
        "can_cross_var_shifts": args.can_cross_var_shifts,
        "can_var_attn": args.can_var_attn,
        "can_var_attn_layers": args.can_var_attn_layers,
        "can_var_attn_dim": args.can_var_attn_dim,
        "can_var_attn_top_k": args.can_var_attn_top_k,
        "can_var_attn_shifts": args.can_var_attn_shifts,
        "can_use_ffn": args.can_use_ffn,
        "can_time_mark": args.can_time_mark,
        "can_time_mark_mode": args.can_time_mark_mode,
        "can_time_mark_scale_init": args.can_time_mark_scale_init,
        "can_periodic_residual": args.can_periodic_residual,
        "can_periods": args.can_periods,
        "can_periodic_alpha": args.can_periodic_alpha,
        "can_periodic_learnable": args.can_periodic_learnable,
        "can_coarse_var_attn": args.can_coarse_var_attn,
        "can_coarse_var_levels": args.can_coarse_var_levels,
        "can_coarse_var_dim": args.can_coarse_var_dim,
        "can_coarse_var_scale_init": args.can_coarse_var_scale_init,
        "can_coarse_var_mode": args.can_coarse_var_mode,
        "can_hierarchical_mixer": args.can_hierarchical_mixer,
        "can_hierarchical_levels": args.can_hierarchical_levels,
        "can_hierarchical_layers": args.can_hierarchical_layers,
        "can_hierarchical_dim": args.can_hierarchical_dim,
        "can_hierarchical_cross_scale_init": args.can_hierarchical_cross_scale_init,
        "can_hierarchical_fusion_init": args.can_hierarchical_fusion_init,
        "can_hierarchical_mode": args.can_hierarchical_mode,
        "can_hierarchical_residual_scale_init": args.can_hierarchical_residual_scale_init,
        "can_periodic_image": args.can_periodic_image,
        "can_periodic_image_top_k": args.can_periodic_image_top_k,
        "can_periodic_image_dim": args.can_periodic_image_dim,
        "can_periodic_image_layers": args.can_periodic_image_layers,
        "can_periodic_image_shifts": args.can_periodic_image_shifts,
        "can_periodic_image_scale_init": args.can_periodic_image_scale_init,
        "can_deep_periodic_image": args.can_deep_periodic_image,
        "can_deep_periodic_top_k": args.can_deep_periodic_top_k,
        "can_deep_periodic_layers": args.can_deep_periodic_layers,
        "can_deep_periodic_shifts": args.can_deep_periodic_shifts,
        "can_deep_periodic_scale_init": args.can_deep_periodic_scale_init,
        "loss": args.loss,
        "loss_mse_weight": args.loss_mse_weight,
        "loss_horizon_weight": args.loss_horizon_weight,
        "loss_horizon_weight_start": args.loss_horizon_weight_start,
        "loss_horizon_weight_mode": args.loss_horizon_weight_mode,
        "loss_range_weight": args.loss_range_weight,
        "loss_tail_bias_weight": args.loss_tail_bias_weight,
        "loss_tail_bias_start": args.loss_tail_bias_start,
        "loss_tail_level_weight": args.loss_tail_level_weight,
        "loss_tail_level_start": args.loss_tail_level_start,
        "loss_tail_hard_weight": args.loss_tail_hard_weight,
        "loss_tail_hard_start": args.loss_tail_hard_start,
        "loss_tail_hard_power": args.loss_tail_hard_power,
        "loss_tail_hard_clip": args.loss_tail_hard_clip,
        "loss_tail_lowpass_weight": args.loss_tail_lowpass_weight,
        "loss_tail_lowpass_start": args.loss_tail_lowpass_start,
        "loss_tail_lowpass_kernel": args.loss_tail_lowpass_kernel,
        "huber_delta": args.huber_delta,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "weight_averaging": args.weight_averaging,
        "ema_decay": args.ema_decay,
        "ema_start_epoch": args.ema_start_epoch,
        "swa_start_epoch": args.swa_start_epoch,
        "swa_end_epoch": args.swa_end_epoch,
    }
    overrides.update({key: value for key, value in optional.items() if value is not None})
    overrides = {key: value for key, value in overrides.items() if value is not None}
    if args.start_epoch is not None or args.end_epoch is not None:
        print(
            "--start-epoch/--end-epoch are deprecated and ignored; testing every epoch.",
            flush=True,
        )
    run_epoch_scan(
        args.trial,
        overrides,
        summary_name,
    )


if __name__ == "__main__":
    main()
