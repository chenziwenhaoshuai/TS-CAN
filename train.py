"""
Autoresearch experiment entrypoint for TS-CAN.

This is the only file the agent should modify during the experiment loop.
The full TS-CAN baseline model is intentionally defined here so architecture
changes also stay in one file.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    STARTUP_GRACE_STEPS,
    TIME_BUDGET_SECONDS,
    build_args,
    evaluate_loader,
    forward_batch,
    make_dataloaders,
    select_device,
    set_random_seed,
)

# ---------------------------------------------------------------------------
# Editable experiment section
# ---------------------------------------------------------------------------

SEED = 2
USE_AMP = True
USE_TORCH_COMPILE = False

DATASET_OVERRIDES: Dict[str, object] = {
    "data": "ETTh1",
    "data_path": "ETTh1.csv",
    "features": "M",
    "target": "OT",
    "seq_len": 192,
    "label_len": 48,
    "pred_len": 96,
}

MODEL_OVERRIDES: Dict[str, object] = {
    "d_model": 96,
    "n_heads": 8,
    "e_layers": 2,
    "d_ff": 128,
    "dropout": 0.05,
    "patch_len": 16,
    "can_stride": 6,
    "can_shifts": "1,2,4,8,16",
    "can_cli_mode": "full",
    "can_temporal_cli_mode": "full",
    "can_ctx_mode": "diff",
    "can_drop_path": 0.05,
    "can_kernel_size": 3,
    "can_init_values": 1e-4,
    "can_use_gffng": 1,
    "can_temporal_roll": 1,
    "can_use_orth": 0,
    "can_context_pyramid": 0,
}

OPTIMIZER_NAME = "adamw"
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_ACCUM_STEPS = 1
GRAD_CLIP_NORM: Optional[float] = None

BATCH_SIZE = 8
NUM_WORKERS = 0
WARMUP_RATIO = 0.05
MIN_LR_SCALE = 0.2

# ---------------------------------------------------------------------------


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, patch_len: int, stride: int, padding: int, dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class LayerNorm1dChannels(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)


def parse_shift_list(shift_string: str):
    shifts = []
    for token in str(shift_string).split(","):
        token = token.strip()
        if token:
            shifts.append(int(token))
    return shifts if shifts else [1, 2, 4, 8]


def orthogonalize_context(state: torch.Tensor, context: torch.Tensor, eps: float = 1e-6):
    dot = (state * context).sum(dim=1, keepdim=True)
    norm = state.pow(2).sum(dim=1, keepdim=True).clamp_min(eps)
    return context - (dot / norm) * state


class CliffordChannelInteraction1D(nn.Module):
    def __init__(self, dim: int, cli_mode: str = "full", ctx_mode: str = "diff", shifts=None):
        super().__init__()
        self.dim = dim
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.shifts = shifts if shifts is not None else [1, 2, 4, 8]

        branch_dim = dim * len(self.shifts)
        if self.cli_mode == "adaptive":
            self.proj_inner = nn.Conv1d(branch_dim, dim, kernel_size=1)
            self.proj_wedge = nn.Conv1d(branch_dim, dim, kernel_size=1)
            self.mix_gate = nn.Conv1d(dim * 2, dim, kernel_size=1)
        else:
            if self.cli_mode == "full":
                cat_dim = branch_dim * 2
            elif self.cli_mode in ("wedge", "inner"):
                cat_dim = branch_dim
            else:
                raise ValueError(f"Invalid cli_mode: {self.cli_mode}")
            self.proj = nn.Conv1d(cat_dim, dim, kernel_size=1)

    def _make_context(self, state: torch.Tensor, context: torch.Tensor):
        if self.ctx_mode == "diff":
            return context - state
        if self.ctx_mode == "abs":
            return context
        raise ValueError(f"Invalid ctx_mode: {self.ctx_mode}")

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        c = self._make_context(state, context)
        inner_feats = []
        wedge_feats = []

        for shift in self.shifts:
            c_shift = torch.roll(c, shifts=shift, dims=1)
            s_shift = torch.roll(state, shifts=shift, dims=1)
            inner_feats.append(F.silu(state * c_shift))
            wedge_feats.append(state * c_shift - c * s_shift)

        if self.cli_mode == "adaptive":
            inner_out = self.proj_inner(torch.cat(inner_feats, dim=1))
            wedge_out = self.proj_wedge(torch.cat(wedge_feats, dim=1))
            alpha = torch.sigmoid(self.mix_gate(torch.cat([state, c], dim=1)))
            return alpha * inner_out + (1.0 - alpha) * wedge_out

        if self.cli_mode == "inner":
            out = torch.cat(inner_feats, dim=1)
        elif self.cli_mode == "wedge":
            out = torch.cat(wedge_feats, dim=1)
        else:
            out = torch.cat(wedge_feats + inner_feats, dim=1)
        return self.proj(out)


class CliffordTemporalInteraction1D(nn.Module):
    def __init__(self, dim: int, cli_mode: str = "inner", ctx_mode: str = "diff", shifts=None):
        super().__init__()
        self.dim = dim
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.shifts = shifts if shifts is not None else [1, 2, 4, 8]

        branch_dim = dim * len(self.shifts)
        if self.cli_mode == "adaptive":
            self.proj_inner = nn.Conv1d(branch_dim, dim, kernel_size=1)
            self.proj_wedge = nn.Conv1d(branch_dim, dim, kernel_size=1)
            self.mix_gate = nn.Conv1d(dim * 2, dim, kernel_size=1)
        else:
            if self.cli_mode == "full":
                cat_dim = branch_dim * 2
            elif self.cli_mode in ("wedge", "inner"):
                cat_dim = branch_dim
            else:
                raise ValueError(f"Invalid cli_mode: {self.cli_mode}")
            self.proj = nn.Conv1d(cat_dim, dim, kernel_size=1)

    @staticmethod
    def causal_shift_right(x: torch.Tensor, shift: int) -> torch.Tensor:
        if shift <= 0:
            return x
        x = F.pad(x, (shift, 0))
        return x[..., :-shift]

    def _make_context(self, state: torch.Tensor, context: torch.Tensor):
        if self.ctx_mode == "diff":
            return context - state
        if self.ctx_mode == "abs":
            return context
        raise ValueError(f"Invalid ctx_mode: {self.ctx_mode}")

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        c = self._make_context(state, context)
        inner_feats = []
        wedge_feats = []

        for shift in self.shifts:
            c_shift = self.causal_shift_right(c, shift)
            s_shift = self.causal_shift_right(state, shift)
            inner_feats.append(F.silu(state * c_shift))
            wedge_feats.append(state * c_shift - c * s_shift)

        if self.cli_mode == "adaptive":
            inner_out = self.proj_inner(torch.cat(inner_feats, dim=1))
            wedge_out = self.proj_wedge(torch.cat(wedge_feats, dim=1))
            alpha = torch.sigmoid(self.mix_gate(torch.cat([state, c], dim=1)))
            return alpha * inner_out + (1.0 - alpha) * wedge_out

        if self.cli_mode == "inner":
            out = torch.cat(inner_feats, dim=1)
        elif self.cli_mode == "wedge":
            out = torch.cat(wedge_feats, dim=1)
        else:
            out = torch.cat(wedge_feats + inner_feats, dim=1)
        return self.proj(out)


class CANPatchBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        cli_mode: str,
        temporal_cli_mode: str,
        ctx_mode: str,
        channel_shifts,
        temporal_shifts,
        kernel_size: int = 3,
        drop_path_rate: float = 0.0,
        init_values: float = 1e-5,
        use_global_context: bool = True,
        beta_init: float = 0.5,
        enable_temporal_interaction: bool = True,
        use_orthogonal: bool = False,
        use_context_pyramid: bool = False,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.norm = LayerNorm1dChannels(dim)
        self.get_state = nn.Conv1d(dim, dim, kernel_size=1)
        self.context_base = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=False),
            nn.Conv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=(kernel_size // 2) * 2,
                dilation=2,
                groups=dim,
                bias=False,
            ),
            LayerNorm1dChannels(dim),
            nn.SiLU(),
        )
        self.use_context_pyramid = use_context_pyramid
        if self.use_context_pyramid:
            hidden = max(4, dim // 4)
            self.context_dilated = nn.Sequential(
                nn.Conv1d(
                    dim,
                    dim,
                    kernel_size=kernel_size,
                    padding=(kernel_size // 2) * 3,
                    dilation=3,
                    groups=dim,
                    bias=False,
                ),
                LayerNorm1dChannels(dim),
                nn.SiLU(),
            )
            self.context_coarse_proj = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
            self.context_weight_mlp = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(dim, hidden, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(hidden, 3, kernel_size=1),
            )

        self.use_orthogonal = use_orthogonal
        self.channel_interaction = CliffordChannelInteraction1D(
            dim=dim,
            cli_mode=cli_mode,
            ctx_mode=ctx_mode,
            shifts=channel_shifts,
        )

        self.enable_temporal_interaction = enable_temporal_interaction
        if self.enable_temporal_interaction:
            self.temporal_interaction = CliffordTemporalInteraction1D(
                dim=dim,
                cli_mode=temporal_cli_mode,
                ctx_mode=ctx_mode,
                shifts=temporal_shifts,
            )
            self.temporal_beta = nn.Parameter(torch.tensor(beta_init))

        self.use_global_context = use_global_context
        if self.use_global_context:
            self.global_interaction = CliffordChannelInteraction1D(
                dim=dim,
                cli_mode="inner",
                ctx_mode="abs",
                shifts=channel_shifts,
            )
            self.global_beta = nn.Parameter(torch.tensor(beta_init))

        self.gate_fc = nn.Conv1d(dim * 2, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1), init_values))
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def build_context(self, x_ln: torch.Tensor) -> torch.Tensor:
        base = self.context_base(x_ln)
        if not self.use_context_pyramid:
            return base

        dilated = self.context_dilated(x_ln)
        coarse = F.avg_pool1d(x_ln, kernel_size=2, stride=2, ceil_mode=True)
        coarse = self.context_coarse_proj(coarse)
        coarse = F.interpolate(coarse, size=x_ln.shape[-1], mode="linear", align_corners=False)

        weights = self.context_weight_mlp(x_ln).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        w0 = weights[:, 0].view(-1, 1, 1)
        w1 = weights[:, 1].view(-1, 1, 1)
        w2 = weights[:, 2].view(-1, 1, 1)
        return w0 * base + w1 * dilated + w2 * coarse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x_ln = self.norm(x)

        state = self.get_state(x_ln)
        context_local = self.build_context(x_ln)
        if self.use_orthogonal:
            context_local = orthogonalize_context(state, context_local)

        geom = self.channel_interaction(state, context_local)
        if self.enable_temporal_interaction:
            geom = geom + self.temporal_beta * self.temporal_interaction(state, context_local)

        if self.use_global_context:
            context_global = x_ln.mean(dim=-1, keepdim=True).expand_as(x_ln)
            if self.use_orthogonal:
                context_global = orthogonalize_context(state, context_global)
            geom = geom + self.global_beta * self.global_interaction(state, context_global)

        gate = torch.sigmoid(self.gate_fc(torch.cat([x_ln, geom], dim=1)))
        mixed = F.silu(x_ln) + gate * geom
        return shortcut + self.drop_path(self.gamma * mixed)


class FlattenHead(nn.Module):
    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class ForecastModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        patch_len = getattr(configs, "patch_len", 16)
        stride = getattr(configs, "can_stride", max(1, patch_len // 2))
        padding = stride

        raw_shifts = parse_shift_list(getattr(configs, "can_shifts", "1,2,4,8"))
        patch_num = int((configs.seq_len - patch_len) / stride + 2)
        channel_shifts = [shift for shift in raw_shifts if shift < self.d_model] or [1]
        temporal_shifts = [shift for shift in raw_shifts if shift < patch_num] or [1]

        cli_mode = getattr(configs, "can_cli_mode", "full")
        temporal_cli_mode = getattr(configs, "can_temporal_cli_mode", "inner")
        ctx_mode = getattr(configs, "can_ctx_mode", "diff")
        kernel_size = getattr(configs, "can_kernel_size", 3)
        can_drop_path = getattr(configs, "can_drop_path", 0.1)
        init_values = getattr(configs, "can_init_values", 1e-5)
        beta_init = getattr(configs, "can_beta_init", 0.5)
        use_global_context = bool(getattr(configs, "can_use_gffng", 1))
        enable_temporal_interaction = bool(getattr(configs, "can_temporal_roll", 1))
        use_orthogonal = bool(getattr(configs, "can_use_orth", 0))
        use_context_pyramid = bool(getattr(configs, "can_context_pyramid", 0))

        self.patch_embedding = PatchEmbedding(
            d_model=configs.d_model,
            patch_len=patch_len,
            stride=stride,
            padding=padding,
            dropout=configs.dropout,
        )

        drop_rates = torch.linspace(0, can_drop_path, configs.e_layers).tolist()
        self.blocks = nn.ModuleList(
            [
                CANPatchBlock(
                    dim=configs.d_model,
                    cli_mode=cli_mode,
                    temporal_cli_mode=temporal_cli_mode,
                    ctx_mode=ctx_mode,
                    channel_shifts=channel_shifts,
                    temporal_shifts=temporal_shifts,
                    kernel_size=kernel_size,
                    drop_path_rate=drop_rates[index],
                    init_values=init_values,
                    use_global_context=use_global_context,
                    beta_init=beta_init,
                    enable_temporal_interaction=enable_temporal_interaction,
                    use_orthogonal=use_orthogonal,
                    use_context_pyramid=use_context_pyramid,
                )
                for index in range(configs.e_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(configs.d_model)
        head_nf = configs.d_model * patch_num
        self.head = FlattenHead(self.enc_in, head_nf, self.pred_len, head_dropout=configs.dropout)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = enc_out.transpose(1, 2)

        for block in self.blocks:
            enc_out = block(enc_out)

        enc_out = self.final_norm(enc_out.transpose(1, 2)).transpose(1, 2)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[1], enc_out.shape[2]))
        dec_out = self.head(enc_out).permute(0, 2, 1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        return self.forecast(x_enc)[:, -self.pred_len :, :]


def build_model(args) -> nn.Module:
    return ForecastModel(args)


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    if OPTIMIZER_NAME.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=BETAS,
            weight_decay=WEIGHT_DECAY,
        )
    if OPTIMIZER_NAME.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=BETAS,
            weight_decay=WEIGHT_DECAY,
        )
    if OPTIMIZER_NAME.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY,
        )
    raise ValueError(f"Unsupported optimizer: {OPTIMIZER_NAME}")


def build_args_for_run():
    overrides = {
        **DATASET_OVERRIDES,
        **MODEL_OVERRIDES,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "use_amp": USE_AMP,
    }
    return build_args(overrides)


def maybe_compile_model(model: nn.Module) -> nn.Module:
    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        return torch.compile(model)
    return model


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def resolve_amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    if not USE_AMP or device.type != "cuda":
        return None
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def lr_multiplier(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    if WARMUP_RATIO > 0 and progress < WARMUP_RATIO:
        return max(progress / WARMUP_RATIO, 1e-3)

    tail_progress = 0.0
    if WARMUP_RATIO < 1.0:
        tail_progress = (progress - WARMUP_RATIO) / max(1.0 - WARMUP_RATIO, 1e-8)
    cosine = 0.5 * (1.0 + math.cos(math.pi * tail_progress))
    return MIN_LR_SCALE + (1.0 - MIN_LR_SCALE) * cosine


def set_learning_rate(optimizer: torch.optim.Optimizer, progress: float) -> float:
    multiplier = lr_multiplier(progress)
    current_lr = LEARNING_RATE * multiplier
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return current_lr


def train_step(
    model: nn.Module,
    batch_iterator,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    args,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> float:
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0

    for _ in range(GRAD_ACCUM_STEPS):
        batch = next(batch_iterator)
        outputs, targets = forward_batch(
            model=model,
            batch=batch,
            args=args,
            device=device,
            amp_enabled=amp_dtype is not None,
            amp_dtype=amp_dtype,
        )
        loss = criterion(outputs, targets) / GRAD_ACCUM_STEPS
        loss.backward()
        total_loss += loss.detach().item()

    if GRAD_CLIP_NORM is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    return total_loss


def main() -> None:
    t_start = time.perf_counter()
    set_random_seed(SEED)
    device = select_device()
    amp_dtype = resolve_amp_dtype(device)

    args = build_args_for_run()
    loaders = make_dataloaders(args)
    _, train_loader = loaders["train"]
    _, val_loader = loaders["val"]
    _, test_loader = loaders["test"]

    model = build_model(args).to(device)
    model = maybe_compile_model(model)
    optimizer = build_optimizer(model)
    criterion = nn.MSELoss()
    batch_iterator = cycle_loader(train_loader)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    num_params = sum(parameter.numel() for parameter in model.parameters())
    smooth_loss = 0.0
    steady_training_time = 0.0
    step = 0

    print(f"Time budget: {TIME_BUDGET_SECONDS}s")
    print(f"Using device: {device}")
    print(f"AMP dtype: {amp_dtype}")

    while True:
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_start = time.perf_counter()

        progress = 0.0 if TIME_BUDGET_SECONDS <= 0 else steady_training_time / TIME_BUDGET_SECONDS
        current_lr = set_learning_rate(optimizer, progress)
        train_loss = train_step(
            model=model,
            batch_iterator=batch_iterator,
            optimizer=optimizer,
            criterion=criterion,
            args=args,
            device=device,
            amp_dtype=amp_dtype,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_time = time.perf_counter() - step_start

        if step >= STARTUP_GRACE_STEPS:
            steady_training_time += step_time

        if math.isnan(train_loss) or math.isinf(train_loss):
            raise RuntimeError("Training diverged: loss became NaN or Inf.")

        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1.0 - ema_beta) * train_loss
        debiased_loss = smooth_loss / (1.0 - ema_beta ** (step + 1))
        pct_done = 100.0 * min(progress, 1.0)
        remaining = max(0.0, TIME_BUDGET_SECONDS - steady_training_time)

        print(
            f"\rstep {step:05d} ({pct_done:5.1f}%) | "
            f"loss: {debiased_loss:.6f} | lr: {current_lr:.6g} | "
            f"dt: {step_time * 1000:.0f}ms | remaining: {remaining:.1f}s",
            end="",
            flush=True,
        )

        step += 1
        if step > STARTUP_GRACE_STEPS and steady_training_time >= TIME_BUDGET_SECONDS:
            break

    print()

    val_metrics = evaluate_loader(
        model=model,
        loader=val_loader,
        args=args,
        device=device,
        amp_enabled=amp_dtype is not None,
        amp_dtype=amp_dtype,
    )
    test_metrics = evaluate_loader(
        model=model,
        loader=test_loader,
        args=args,
        device=device,
        amp_enabled=amp_dtype is not None,
        amp_dtype=amp_dtype,
    )

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device=device) / 1024.0 / 1024.0

    total_seconds = time.perf_counter() - t_start
    print("---")
    print(f"val_mse:          {val_metrics['mse']:.6f}")
    print(f"val_mae:          {val_metrics['mae']:.6f}")
    print(f"test_mse:         {test_metrics['mse']:.6f}")
    print(f"test_mae:         {test_metrics['mae']:.6f}")
    print(f"training_seconds: {steady_training_time:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.3f}")
    print(f"batch_size:       {BATCH_SIZE}")
    print(f"optimizer:        {OPTIMIZER_NAME}")


if __name__ == "__main__":
    main()
