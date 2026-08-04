import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import PatchEmbedding


def drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class LayerNorm1dChannels(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)


def parse_shift_list(shift_string):
    shifts = []
    for token in str(shift_string).split(','):
        token = token.strip()
        if token:
            shifts.append(int(token))
    return shifts if shifts else [1, 2, 4, 8]


def parse_int_list(value):
    values = []
    for token in str(value).split(','):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def orthogonalize_context(state, context, eps=1e-6):
    dot = (state * context).sum(dim=1, keepdim=True)
    norm = state.pow(2).sum(dim=1, keepdim=True).clamp_min(eps)
    return context - (dot / norm) * state


class CliffordChannelInteraction1D(nn.Module):
    def __init__(self, dim, cli_mode='full', ctx_mode='diff', shifts=None):
        super().__init__()
        self.dim = dim
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.shifts = shifts if shifts is not None else [1, 2, 4, 8]

        branch_dim = dim * len(self.shifts)
        if self.cli_mode == 'adaptive':
            self.proj_inner = nn.Conv1d(branch_dim, dim, kernel_size=1)
            self.proj_wedge = nn.Conv1d(branch_dim, dim, kernel_size=1)
            self.mix_gate = nn.Conv1d(dim * 2, dim, kernel_size=1)
        else:
            if self.cli_mode == 'full':
                cat_dim = branch_dim * 2
            elif self.cli_mode in ('wedge', 'inner'):
                cat_dim = branch_dim
            else:
                raise ValueError(f'Invalid cli_mode: {self.cli_mode}')
            self.proj = nn.Conv1d(cat_dim, dim, kernel_size=1)

    def _make_context(self, state, context):
        if self.ctx_mode == 'diff':
            return context - state
        if self.ctx_mode == 'abs':
            return context
        raise ValueError(f'Invalid ctx_mode: {self.ctx_mode}')

    def forward(self, state, context):
        c = self._make_context(state, context)
        inner_feats = []
        wedge_feats = []

        for shift in self.shifts:
            c_shift = torch.roll(c, shifts=shift, dims=1)
            s_shift = torch.roll(state, shifts=shift, dims=1)
            inner_feats.append(F.silu(state * c_shift))
            wedge_feats.append(state * c_shift - c * s_shift)

        if self.cli_mode == 'adaptive':
            inner_out = self.proj_inner(torch.cat(inner_feats, dim=1))
            wedge_out = self.proj_wedge(torch.cat(wedge_feats, dim=1))
            alpha = torch.sigmoid(self.mix_gate(torch.cat([state, c], dim=1)))
            return alpha * inner_out + (1.0 - alpha) * wedge_out

        if self.cli_mode == 'inner':
            out = torch.cat(inner_feats, dim=1)
        elif self.cli_mode == 'wedge':
            out = torch.cat(wedge_feats, dim=1)
        else:
            out = torch.cat(wedge_feats + inner_feats, dim=1)
        return self.proj(out)


class CliffordTemporalInteraction1D(nn.Module):
    def __init__(self, dim, cli_mode='inner', ctx_mode='diff', shifts=None, circular=False):
        super().__init__()
        self.dim = dim
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.shifts = shifts if shifts is not None else [1, 2, 4, 8]
        self.circular = circular

        branch_dim = dim * len(self.shifts)
        if self.cli_mode == 'adaptive':
            self.proj_inner = nn.Conv1d(branch_dim, dim, kernel_size=1)
            self.proj_wedge = nn.Conv1d(branch_dim, dim, kernel_size=1)
            self.mix_gate = nn.Conv1d(dim * 2, dim, kernel_size=1)
        else:
            if self.cli_mode == 'full':
                cat_dim = branch_dim * 2
            elif self.cli_mode in ('wedge', 'inner'):
                cat_dim = branch_dim
            else:
                raise ValueError(f'Invalid cli_mode: {self.cli_mode}')
            self.proj = nn.Conv1d(cat_dim, dim, kernel_size=1)

    @staticmethod
    def causal_shift_right(x, shift):
        if shift <= 0:
            return x
        x = F.pad(x, (shift, 0))
        return x[..., :-shift]

    def _make_context(self, state, context):
        if self.ctx_mode == 'diff':
            return context - state
        if self.ctx_mode == 'abs':
            return context
        raise ValueError(f'Invalid ctx_mode: {self.ctx_mode}')

    def forward(self, state, context):
        c = self._make_context(state, context)
        inner_feats = []
        wedge_feats = []

        for shift in self.shifts:
            if self.circular:
                c_shift = torch.roll(c, shifts=shift, dims=-1)
                s_shift = torch.roll(state, shifts=shift, dims=-1)
            else:
                c_shift = self.causal_shift_right(c, shift)
                s_shift = self.causal_shift_right(state, shift)
            inner_feats.append(F.silu(state * c_shift))
            wedge_feats.append(state * c_shift - c * s_shift)

        if self.cli_mode == 'adaptive':
            inner_out = self.proj_inner(torch.cat(inner_feats, dim=1))
            wedge_out = self.proj_wedge(torch.cat(wedge_feats, dim=1))
            alpha = torch.sigmoid(self.mix_gate(torch.cat([state, c], dim=1)))
            return alpha * inner_out + (1.0 - alpha) * wedge_out

        if self.cli_mode == 'inner':
            out = torch.cat(inner_feats, dim=1)
        elif self.cli_mode == 'wedge':
            out = torch.cat(wedge_feats, dim=1)
        else:
            out = torch.cat(wedge_feats + inner_feats, dim=1)
        return self.proj(out)


class CANPatchBlock(nn.Module):
    def __init__(self, dim, cli_mode, temporal_cli_mode, ctx_mode, channel_shifts, temporal_shifts, kernel_size=3,
                 drop_path_rate=0.0, init_values=1e-5, use_global_context=True, beta_init=0.5,
                 temporal_beta_init=None, global_beta_init=None,
                 global_cli_mode='inner', global_ctx_mode='abs', global_shifts=None,
                 enable_temporal_interaction=True, use_orthogonal=False, use_context_pyramid=False,
                 use_ffn=False, d_ff=None, temporal_circular=False):
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
                bias=False
            ),
            nn.BatchNorm1d(dim),
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
                    bias=False
                ),
                nn.BatchNorm1d(dim),
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
            shifts=channel_shifts
        )

        self.enable_temporal_interaction = enable_temporal_interaction
        if self.enable_temporal_interaction:
            self.temporal_interaction = CliffordTemporalInteraction1D(
                dim=dim,
                cli_mode=temporal_cli_mode,
                ctx_mode=ctx_mode,
                shifts=temporal_shifts,
                circular=temporal_circular
            )
            temporal_beta = beta_init if temporal_beta_init is None else temporal_beta_init
            self.temporal_beta = nn.Parameter(torch.tensor(temporal_beta))

        self.use_global_context = use_global_context
        if self.use_global_context:
            self.global_interaction = CliffordChannelInteraction1D(
                dim=dim,
                cli_mode=global_cli_mode,
                ctx_mode=global_ctx_mode,
                shifts=global_shifts if global_shifts is not None else channel_shifts
            )
            global_beta = beta_init if global_beta_init is None else global_beta_init
            self.global_beta = nn.Parameter(torch.tensor(global_beta))

        self.gate_fc = nn.Conv1d(dim * 2, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1), init_values))
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.use_ffn = use_ffn
        if self.use_ffn:
            ffn_dim = d_ff if d_ff is not None else dim * 2
            self.ffn = nn.Sequential(
                nn.Conv1d(dim, ffn_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(ffn_dim, dim, kernel_size=1),
            )
            self.ffn_gamma = nn.Parameter(torch.full((1, dim, 1), init_values))

    def build_context(self, x_ln):
        base = self.context_base(x_ln)
        if not self.use_context_pyramid:
            return base

        dilated = self.context_dilated(x_ln)
        coarse = F.avg_pool1d(x_ln, kernel_size=2, stride=2, ceil_mode=True)
        coarse = self.context_coarse_proj(coarse)
        coarse = F.interpolate(coarse, size=x_ln.shape[-1], mode='linear', align_corners=False)

        weights = self.context_weight_mlp(x_ln).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        w0 = weights[:, 0].view(-1, 1, 1)
        w1 = weights[:, 1].view(-1, 1, 1)
        w2 = weights[:, 2].view(-1, 1, 1)
        return w0 * base + w1 * dilated + w2 * coarse

    def forward(self, x):
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
        if self.use_ffn:
            mixed = mixed + self.ffn_gamma * self.ffn(x_ln)
        return shortcut + self.drop_path(self.gamma * mixed)


class CrossVariableCliffordBlock(nn.Module):
    def __init__(self, dim, shifts, context_mode='others_mean', drop_path_rate=0.0, init_values=1e-5):
        super().__init__()
        if context_mode not in ('mean', 'others_mean'):
            raise ValueError(f'Invalid cross-variable context mode: {context_mode}')
        self.context_mode = context_mode
        self.norm = LayerNorm1dChannels(dim)
        self.get_state = nn.Conv1d(dim, dim, kernel_size=1)
        self.get_context = nn.Conv1d(dim, dim, kernel_size=1)
        self.interaction = CliffordChannelInteraction1D(
            dim=dim,
            cli_mode='full',
            ctx_mode='diff',
            shifts=shifts
        )
        self.gate_fc = nn.Conv1d(dim * 2, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1), init_values))
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        batch_size, n_vars, dim, patch_num = x.shape
        flat = x.reshape(batch_size * n_vars, dim, patch_num)
        x_ln = self.norm(flat).reshape(batch_size, n_vars, dim, patch_num)

        if self.context_mode == 'others_mean' and n_vars > 1:
            context = (x_ln.sum(dim=1, keepdim=True) - x_ln) / (n_vars - 1)
        else:
            context = x_ln.mean(dim=1, keepdim=True).expand_as(x_ln)

        state = self.get_state(x_ln.reshape(batch_size * n_vars, dim, patch_num))
        context = self.get_context(context.reshape(batch_size * n_vars, dim, patch_num))
        geom = self.interaction(state, context)
        gate = torch.sigmoid(self.gate_fc(torch.cat([flat, geom], dim=1)))
        mixed = F.silu(flat) + gate * geom
        out = flat + self.drop_path(self.gamma * mixed)
        return out.reshape(batch_size, n_vars, dim, patch_num)


class VariableAttentionCliffordBlock(nn.Module):
    def __init__(self, dim, shifts, attn_dim=32, top_k=0, drop_path_rate=0.0, init_values=1e-5):
        super().__init__()
        self.top_k = int(top_k)
        self.norm = LayerNorm1dChannels(dim)
        self.query = nn.Linear(dim, attn_dim, bias=False)
        self.key = nn.Linear(dim, attn_dim, bias=False)
        self.get_state = nn.Conv1d(dim, dim, kernel_size=1)
        self.get_context = nn.Conv1d(dim, dim, kernel_size=1)
        self.interaction = CliffordChannelInteraction1D(
            dim=dim,
            cli_mode='full',
            ctx_mode='diff',
            shifts=shifts
        )
        self.gate_fc = nn.Conv1d(dim * 2, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1), init_values))
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        batch_size, n_vars, dim, patch_num = x.shape
        flat = x.reshape(batch_size * n_vars, dim, patch_num)
        x_ln = self.norm(flat).reshape(batch_size, n_vars, dim, patch_num)

        summary = x_ln.mean(dim=-1)
        query = F.normalize(self.query(summary), dim=-1)
        key = F.normalize(self.key(summary), dim=-1)
        scores = torch.matmul(query, key.transpose(1, 2)) / (query.shape[-1] ** 0.5)
        if self.top_k > 0 and self.top_k < n_vars:
            top_values, top_indices = torch.topk(scores, self.top_k, dim=-1)
            masked = scores.new_full(scores.shape, -torch.inf)
            scores = masked.scatter(-1, top_indices, top_values)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum('bij,bjdp->bidp', weights, x_ln)

        state = self.get_state(flat)
        context = self.get_context(context.reshape(batch_size * n_vars, dim, patch_num))
        geom = self.interaction(state, context)
        gate = torch.sigmoid(self.gate_fc(torch.cat([flat, geom], dim=1)))
        mixed = F.silu(flat) + gate * geom
        out = flat + self.drop_path(self.gamma * mixed)
        return out.reshape(batch_size, n_vars, dim, patch_num)


class TimePatchEmbedding(nn.Module):
    def __init__(self, time_dim, d_model, patch_len, stride, padding, mode='flatten', scale_init=1.0):
        super().__init__()
        if mode not in ('flatten', 'mean'):
            raise ValueError(f'Invalid time patch mode: {mode}')
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        self.mode = mode
        input_dim = time_dim * patch_len if mode == 'flatten' else time_dim
        self.proj = nn.Linear(input_dim, d_model, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x_mark):
        x_mark = x_mark.transpose(1, 2)
        x_mark = F.pad(x_mark, (0, self.padding), mode='replicate')
        x_mark = x_mark.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_mark = x_mark.permute(0, 2, 1, 3)
        if self.mode == 'mean':
            x_mark = x_mark.mean(dim=-1)
        else:
            x_mark = x_mark.flatten(start_dim=-2)
        return self.scale * self.proj(x_mark)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size

    def forward(self, x):
        padding = (self.kernel_size - 1) // 2
        x_channels = x.transpose(1, 2)
        x_padded = F.pad(x_channels, (padding, padding), mode='replicate')
        trend = F.avg_pool1d(x_padded, kernel_size=self.kernel_size, stride=1)
        trend = trend.transpose(1, 2)
        return x - trend, trend


class LinearForecastResidual(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, mode='raw', individual=False, moving_avg=25, scale_init=0.5):
        super().__init__()
        if mode not in ('raw', 'decomp'):
            raise ValueError(f'Invalid linear residual mode: {mode}')
        self.mode = mode
        self.individual = individual
        self.n_vars = n_vars
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        if mode == 'decomp':
            self.decomp = SeriesDecomposition(moving_avg)
        branch_count = 2 if mode == 'decomp' else 1
        if individual:
            self.projections = nn.ModuleList(
                [
                    nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(n_vars)])
                    for _ in range(branch_count)
                ]
            )
        else:
            self.projections = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(branch_count)]
            )

    def _project(self, x, branch):
        x = x.transpose(1, 2)
        if self.individual:
            outputs = [self.projections[branch][i](x[:, i]) for i in range(self.n_vars)]
            return torch.stack(outputs, dim=-1)
        return self.projections[branch](x).transpose(1, 2)

    def forward(self, x):
        if self.mode == 'decomp':
            seasonal, trend = self.decomp(x)
            out = self._project(seasonal, 0) + self._project(trend, 1)
        else:
            out = self._project(x, 0)
        return self.scale * out


class PeriodicForecastResidual(nn.Module):
    def __init__(self, pred_len, periods, alpha=0.2, learnable_alpha=False):
        super().__init__()
        self.pred_len = pred_len
        self.periods = periods
        alpha = min(max(float(alpha), 1e-4), 1.0 - 1e-4)
        alpha_logit = torch.logit(torch.tensor(alpha))
        if learnable_alpha:
            self.alpha_logit = nn.Parameter(alpha_logit)
        else:
            self.register_buffer('alpha_logit', alpha_logit)
        if len(periods) > 1:
            self.period_logits = nn.Parameter(torch.zeros(len(periods)))
        else:
            self.register_buffer('period_logits', torch.zeros(1))

    def _repeat_period(self, x, period):
        pattern = x[:, -period:, :]
        repeats = (self.pred_len + period - 1) // period
        return pattern.repeat(1, repeats, 1)[:, :self.pred_len, :]

    def forward(self, x, forecast):
        periodic = torch.stack(
            [self._repeat_period(x, period) for period in self.periods],
            dim=0
        )
        weights = torch.softmax(self.period_logits, dim=0).view(-1, 1, 1, 1)
        periodic = (weights * periodic).sum(dim=0)
        alpha = torch.sigmoid(self.alpha_logit)
        return forecast + alpha * (periodic - forecast)


class CoarseVariableAttention(nn.Module):
    def __init__(self, seq_len, d_model, levels=3, attn_dim=32, scale_init=0.1, mode='diff'):
        super().__init__()
        if mode not in ('abs', 'diff'):
            raise ValueError(f'Invalid coarse variable attention mode: {mode}')
        self.levels = max(0, int(levels))
        self.mode = mode
        coarse_len = seq_len
        for _ in range(self.levels):
            coarse_len = max(1, coarse_len // 2)
        self.query = nn.Linear(coarse_len, attn_dim, bias=False)
        self.key = nn.Linear(coarse_len, attn_dim, bias=False)
        self.value = nn.Linear(coarse_len, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.attn_scale = attn_dim ** -0.5

    def forward(self, x):
        coarse = x.transpose(1, 2)
        for _ in range(self.levels):
            coarse = F.avg_pool1d(coarse, kernel_size=2, stride=2)
        query = self.query(coarse)
        key = self.key(coarse)
        value = self.value(coarse)
        attention = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) * self.attn_scale,
            dim=-1
        )
        context = torch.matmul(attention, value)
        if self.mode == 'diff':
            context = context - value
        return self.scale * self.norm(context)


class PeriodicAxisCliffordBlock(nn.Module):
    def __init__(self, dim, shifts, kernel_size=3, init_values=1e-3, use_orthogonal=True):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.shifts = shifts if shifts else [1]
        self.padding = kernel_size // 2
        self.use_orthogonal = use_orthogonal
        self.norm = LayerNorm1dChannels(dim)
        self.state_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.context_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            groups=dim,
            bias=False
        )
        self.context_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.branch_projections = nn.ModuleList(
            [nn.Conv1d(dim * 2, dim, kernel_size=1) for _ in self.shifts]
        )
        self.gate = nn.Conv1d(dim * 2, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1), init_values))

    def forward(self, x):
        if x.shape[-1] <= 1:
            return x
        shortcut = x
        normalized = self.norm(x)
        state = self.state_proj(normalized)
        padded = F.pad(
            normalized,
            (self.padding, self.padding),
            mode='circular'
        )
        context = self.context_proj(self.context_conv(padded))
        if self.use_orthogonal:
            context = orthogonalize_context(state, context)

        branches = []
        length = x.shape[-1]
        for shift, projection in zip(self.shifts, self.branch_projections):
            effective_shift = shift % length
            if effective_shift == 0:
                continue
            context_shift = torch.roll(context, shifts=effective_shift, dims=-1)
            state_shift = torch.roll(state, shifts=effective_shift, dims=-1)
            inner = F.silu(state * context_shift)
            wedge = state * context_shift - context * state_shift
            branches.append(projection(torch.cat([wedge, inner], dim=1)))
        if not branches:
            return shortcut

        geometry = torch.stack(branches, dim=0).mean(dim=0)
        gate = torch.sigmoid(self.gate(torch.cat([normalized, geometry], dim=1)))
        return shortcut + self.gamma * gate * geometry


class PeriodicImageCliffordRefiner(nn.Module):
    def __init__(
        self,
        seq_len,
        d_model=32,
        top_k=3,
        layers=1,
        shifts=None,
        kernel_size=3,
        init_values=1e-3,
        scale_init=0.0,
        use_orthogonal=True
    ):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = max(1, int(top_k))
        self.input_projection = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        axis_shifts = shifts if shifts else [1, 2, 4]
        layer_count = max(1, int(layers))
        self.within_period_blocks = nn.ModuleList(
            [
                PeriodicAxisCliffordBlock(
                    d_model,
                    axis_shifts,
                    kernel_size=kernel_size,
                    init_values=init_values,
                    use_orthogonal=use_orthogonal
                )
                for _ in range(layer_count)
            ]
        )
        self.across_period_blocks = nn.ModuleList(
            [
                PeriodicAxisCliffordBlock(
                    d_model,
                    axis_shifts,
                    kernel_size=kernel_size,
                    init_values=init_values,
                    use_orthogonal=use_orthogonal
                )
                for _ in range(layer_count)
            ]
        )
        self.output_norm = LayerNorm1dChannels(d_model)
        self.output_projection = nn.Conv1d(d_model, 1, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def select_periods(self, x):
        with torch.no_grad():
            spectrum = torch.fft.rfft(x.float(), dim=1).abs()
            amplitudes = spectrum.mean(dim=(0, 2))
            amplitudes[0] = 0.0
            count = min(self.top_k, max(1, amplitudes.numel() - 1))
            values, frequencies = torch.topk(amplitudes, count)
            periods = torch.div(
                self.seq_len + frequencies - 1,
                frequencies.clamp_min(1),
                rounding_mode='floor'
            )
            periods = periods.clamp(2, self.seq_len)
            weights = torch.softmax(values, dim=0)
        return periods.tolist(), weights.to(dtype=x.dtype, device=x.device)

    @staticmethod
    def to_image(feature, period):
        length = feature.shape[-1]
        columns = (length + period - 1) // period
        padded_length = period * columns
        if padded_length > length:
            feature = F.pad(feature, (0, padded_length - length))
        image = feature.reshape(
            feature.shape[0],
            feature.shape[1],
            columns,
            period
        )
        return image.permute(0, 1, 3, 2).contiguous(), length

    @staticmethod
    def to_sequence(image, length):
        sequence = image.permute(0, 1, 3, 2).contiguous()
        return sequence.reshape(sequence.shape[0], sequence.shape[1], -1)[..., :length]

    def refine_image(self, feature, period):
        image, length = self.to_image(feature, period)
        for within_block, across_block in zip(
            self.within_period_blocks,
            self.across_period_blocks
        ):
            batch, dim, rows, columns = image.shape
            within = image.permute(0, 3, 1, 2).reshape(
                batch * columns,
                dim,
                rows
            )
            within = within_block(within)
            within = within.reshape(batch, columns, dim, rows)
            within = within.permute(0, 2, 3, 1).contiguous()

            across = image.permute(0, 2, 1, 3).reshape(
                batch * rows,
                dim,
                columns
            )
            across = across_block(across)
            across = across.reshape(batch, rows, dim, columns)
            across = across.permute(0, 2, 1, 3).contiguous()
            image = 0.5 * (within + across)
        return self.to_sequence(image, length)

    def forward(self, x):
        batch_size, seq_len, n_vars = x.shape
        if seq_len != self.seq_len:
            raise ValueError(
                f'Periodic image branch expected seq_len={self.seq_len}, got {seq_len}'
            )
        flat = x.permute(0, 2, 1).reshape(batch_size * n_vars, 1, seq_len)
        feature = self.input_projection(flat)
        periods, weights = self.select_periods(x)
        refined = [
            self.refine_image(feature, int(period))
            for period in periods
        ]
        mixed = sum(
            weight * value for weight, value in zip(weights, refined)
        )
        correction = self.output_projection(self.output_norm(mixed))
        correction = correction.reshape(
            batch_size,
            n_vars,
            seq_len
        ).permute(0, 2, 1)
        return self.scale * correction


class DeepPeriodicImageCliffordMixer(nn.Module):
    def __init__(
        self,
        dim,
        top_k=3,
        layers=1,
        shifts=None,
        kernel_size=3,
        init_values=1e-2,
        scale_init=0.1,
        use_orthogonal=True
    ):
        super().__init__()
        self.top_k = max(1, int(top_k))
        axis_shifts = shifts if shifts else [1, 2, 4]
        layer_count = max(1, int(layers))
        self.within_period_blocks = nn.ModuleList(
            [
                PeriodicAxisCliffordBlock(
                    dim,
                    axis_shifts,
                    kernel_size=kernel_size,
                    init_values=init_values,
                    use_orthogonal=use_orthogonal
                )
                for _ in range(layer_count)
            ]
        )
        self.across_period_blocks = nn.ModuleList(
            [
                PeriodicAxisCliffordBlock(
                    dim,
                    axis_shifts,
                    kernel_size=kernel_size,
                    init_values=init_values,
                    use_orthogonal=use_orthogonal
                )
                for _ in range(layer_count)
            ]
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def select_periods(self, x):
        token_count = x.shape[-1]
        with torch.no_grad():
            spectrum = torch.fft.rfft(x.float(), dim=-1).abs()
            amplitudes = spectrum.mean(dim=(0, 1))
            amplitudes[0] = 0.0
            count = min(self.top_k, max(1, amplitudes.numel() - 1))
            values, frequencies = torch.topk(amplitudes, count)
            periods = torch.div(
                token_count + frequencies - 1,
                frequencies.clamp_min(1),
                rounding_mode='floor'
            ).clamp(2, token_count)
            weights = torch.softmax(values, dim=0)
        return periods.tolist(), weights.to(dtype=x.dtype, device=x.device)

    @staticmethod
    def to_image(feature, period):
        length = feature.shape[-1]
        columns = (length + period - 1) // period
        padded_length = period * columns
        if padded_length > length:
            feature = F.pad(feature, (0, padded_length - length))
        image = feature.reshape(
            feature.shape[0],
            feature.shape[1],
            columns,
            period
        )
        return image.permute(0, 1, 3, 2).contiguous(), length

    @staticmethod
    def to_sequence(image, length):
        sequence = image.permute(0, 1, 3, 2).contiguous()
        return sequence.reshape(sequence.shape[0], sequence.shape[1], -1)[..., :length]

    def refine_image(self, feature, period):
        image, length = self.to_image(feature, period)
        for within_block, across_block in zip(
            self.within_period_blocks,
            self.across_period_blocks
        ):
            batch, dim, rows, columns = image.shape
            within = image.permute(0, 3, 1, 2).reshape(
                batch * columns,
                dim,
                rows
            )
            within = within_block(within)
            within = within.reshape(batch, columns, dim, rows)
            within = within.permute(0, 2, 3, 1).contiguous()

            across = image.permute(0, 2, 1, 3).reshape(
                batch * rows,
                dim,
                columns
            )
            across = across_block(across)
            across = across.reshape(batch, rows, dim, columns)
            across = across.permute(0, 2, 1, 3).contiguous()
            image = 0.5 * (within + across)
        return self.to_sequence(image, length)

    def forward(self, x):
        if x.shape[-1] <= 2:
            return x
        periods, weights = self.select_periods(x)
        refined = [
            self.refine_image(x, int(period))
            for period in periods
        ]
        mixed = sum(
            weight * value for weight, value in zip(weights, refined)
        )
        return x + self.scale * (mixed - x)


class CrossScaleCliffordFusion(nn.Module):
    def __init__(self, dim, shifts, scale_init=0.05):
        super().__init__()
        self.state_norm = LayerNorm1dChannels(dim)
        self.context_norm = LayerNorm1dChannels(dim)
        self.state_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.context_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.interaction = CliffordChannelInteraction1D(
            dim=dim,
            cli_mode='full',
            ctx_mode='diff',
            shifts=shifts
        )
        self.gate = nn.Conv1d(dim * 2, dim, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, state, context):
        state_norm = self.state_norm(state)
        context_norm = self.context_norm(context)
        geom = self.interaction(
            self.state_proj(state_norm),
            self.context_proj(context_norm)
        )
        gate = torch.sigmoid(self.gate(torch.cat([state_norm, geom], dim=1)))
        return state + self.scale * gate * geom


class HierarchicalCliffordMixer(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        d_model,
        levels,
        layers,
        moving_avg,
        shifts,
        cli_mode,
        temporal_cli_mode,
        ctx_mode,
        kernel_size,
        drop_path,
        init_values,
        beta_init,
        use_global_context,
        use_orthogonal,
        use_context_pyramid,
        temporal_circular,
        cross_scale_init,
        output_mode='blend'
    ):
        super().__init__()
        if output_mode not in ('blend', 'residual'):
            raise ValueError(f'Invalid hierarchical output mode: {output_mode}')
        self.pred_len = pred_len
        self.levels = max(1, int(levels))
        self.output_mode = output_mode
        self.scale_lengths = [
            max(1, seq_len // (2 ** level))
            for level in range(self.levels + 1)
        ]
        self.input_projection = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        self.scale_embedding = nn.Parameter(
            torch.zeros(self.levels + 1, 1, d_model, 1)
        )
        nn.init.trunc_normal_(self.scale_embedding, std=0.02)

        min_length = self.scale_lengths[-1]
        temporal_shifts = [shift for shift in shifts if shift < min_length]
        if not temporal_shifts:
            temporal_shifts = [1]
        channel_shifts = [shift for shift in shifts if shift < d_model]
        if not channel_shifts:
            channel_shifts = [1]

        layer_count = max(1, int(layers))
        dpr = torch.linspace(0, drop_path, layer_count).tolist()
        self.blocks = nn.ModuleList(
            [
                CANPatchBlock(
                    dim=d_model,
                    cli_mode=cli_mode,
                    temporal_cli_mode=temporal_cli_mode,
                    ctx_mode=ctx_mode,
                    channel_shifts=channel_shifts,
                    temporal_shifts=temporal_shifts,
                    kernel_size=kernel_size,
                    drop_path_rate=dpr[index],
                    init_values=init_values,
                    use_global_context=use_global_context,
                    beta_init=beta_init,
                    enable_temporal_interaction=True,
                    use_orthogonal=use_orthogonal,
                    use_context_pyramid=use_context_pyramid,
                    use_ffn=False,
                    temporal_circular=temporal_circular
                )
                for index in range(layer_count)
            ]
        )

        self.season_down = nn.ModuleList()
        self.trend_up = nn.ModuleList()
        self.season_fusion = nn.ModuleList()
        self.trend_fusion = nn.ModuleList()
        for _ in self.blocks:
            self.season_down.append(
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv1d(
                                d_model,
                                d_model,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=d_model,
                                bias=False
                            ),
                            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
                            nn.SiLU(),
                        )
                        for _ in range(self.levels)
                    ]
                )
            )
            self.trend_up.append(
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv1d(
                                d_model,
                                d_model,
                                kernel_size=3,
                                padding=1,
                                groups=d_model,
                                bias=False
                            ),
                            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
                            nn.SiLU(),
                        )
                        for _ in range(self.levels)
                    ]
                )
            )
            self.season_fusion.append(
                nn.ModuleList(
                    [
                        CrossScaleCliffordFusion(
                            d_model,
                            channel_shifts,
                            scale_init=cross_scale_init
                        )
                        for _ in range(self.levels)
                    ]
                )
            )
            self.trend_fusion.append(
                nn.ModuleList(
                    [
                        CrossScaleCliffordFusion(
                            d_model,
                            channel_shifts,
                            scale_init=cross_scale_init
                        )
                        for _ in range(self.levels)
                    ]
                )
            )

        kernels = []
        for level in range(self.levels + 1):
            scale_kernel = max(3, int(round(moving_avg / (2 ** level))))
            if scale_kernel % 2 == 0:
                scale_kernel += 1
            kernels.append(scale_kernel)
        self.decomposition_kernels = kernels
        self.output_norms = nn.ModuleList(
            [LayerNorm1dChannels(d_model) for _ in self.scale_lengths]
        )
        self.temporal_heads = nn.ModuleList(
            [nn.Linear(length, pred_len) for length in self.scale_lengths]
        )
        self.output_projection = nn.Linear(d_model, 1)
        if self.output_mode == 'residual':
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)
        self.scale_logits = nn.Parameter(torch.zeros(self.levels + 1))

    @staticmethod
    def decompose(x, kernel_size):
        padding = (kernel_size - 1) // 2
        trend = F.avg_pool1d(
            F.pad(x, (padding, padding), mode='replicate'),
            kernel_size=kernel_size,
            stride=1
        )
        return x - trend, trend

    def build_scales(self, x):
        scales = [x]
        for _ in range(self.levels):
            scales.append(F.avg_pool1d(scales[-1], kernel_size=2, stride=2))
        return scales

    def forward(self, x):
        batch_size, seq_len, n_vars = x.shape
        raw = x.permute(0, 2, 1).reshape(batch_size * n_vars, 1, seq_len)
        raw_scales = self.build_scales(raw)
        features = []
        statistics = []
        for level, raw_scale in enumerate(raw_scales):
            mean = raw_scale.mean(dim=-1, keepdim=True)
            stdev = torch.sqrt(
                raw_scale.var(dim=-1, keepdim=True, unbiased=False) + 1e-5
            )
            normalized = (raw_scale - mean) / stdev
            features.append(
                self.input_projection(normalized) + self.scale_embedding[level]
            )
            statistics.append((mean, stdev))

        for layer_index, block in enumerate(self.blocks):
            encoded = [block(feature) for feature in features]
            seasonal = []
            trend = []
            for feature, kernel in zip(encoded, self.decomposition_kernels):
                season_part, trend_part = self.decompose(feature, kernel)
                seasonal.append(season_part)
                trend.append(trend_part)

            mixed_seasonal = [seasonal[0]]
            for level in range(1, self.levels + 1):
                context = self.season_down[layer_index][level - 1](
                    mixed_seasonal[-1]
                )
                if context.shape[-1] != seasonal[level].shape[-1]:
                    context = F.interpolate(
                        context,
                        size=seasonal[level].shape[-1],
                        mode='linear',
                        align_corners=False
                    )
                mixed_seasonal.append(
                    self.season_fusion[layer_index][level - 1](
                        seasonal[level],
                        context
                    )
                )

            mixed_trend = list(trend)
            for level in range(self.levels - 1, -1, -1):
                context = F.interpolate(
                    mixed_trend[level + 1],
                    size=trend[level].shape[-1],
                    mode='linear',
                    align_corners=False
                )
                context = self.trend_up[layer_index][level](context)
                mixed_trend[level] = self.trend_fusion[layer_index][level](
                    trend[level],
                    context
                )

            features = [
                norm(season_part + trend_part)
                for norm, season_part, trend_part in zip(
                    self.output_norms,
                    mixed_seasonal,
                    mixed_trend
                )
            ]

        predictions = []
        for feature, head, (mean, stdev) in zip(
            features,
            self.temporal_heads,
            statistics
        ):
            prediction = head(feature).transpose(1, 2)
            prediction = self.output_projection(prediction).squeeze(-1)
            if self.output_mode == 'blend':
                prediction = prediction * stdev.squeeze(1) + mean.squeeze(1)
            prediction = prediction.reshape(
                batch_size,
                n_vars,
                self.pred_len
            ).permute(0, 2, 1)
            predictions.append(prediction)

        weights = torch.softmax(self.scale_logits, dim=0)
        return sum(
            weight * prediction
            for weight, prediction in zip(weights, predictions)
        )


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.use_norm = bool(getattr(configs, 'use_norm', 1))

        patch_len = getattr(configs, 'patch_len', 16)
        stride = getattr(configs, 'can_stride', max(1, patch_len // 2))
        padding = stride

        raw_shifts = parse_shift_list(getattr(configs, 'can_shifts', '1,2,4,8'))
        temporal_shift_string = str(getattr(configs, 'can_temporal_shifts', '')).strip()
        raw_temporal_shifts = (
            parse_shift_list(temporal_shift_string) if temporal_shift_string else raw_shifts
        )
        patch_num = int((configs.seq_len - patch_len) / stride + 2)
        channel_shifts = [s for s in raw_shifts if s < self.d_model]
        temporal_shifts = [s for s in raw_temporal_shifts if s < patch_num]
        if not channel_shifts:
            channel_shifts = [1]
        if not temporal_shifts:
            temporal_shifts = [1]

        cli_mode = getattr(configs, 'can_cli_mode', 'full')
        temporal_cli_mode = getattr(configs, 'can_temporal_cli_mode', 'inner')
        ctx_mode = getattr(configs, 'can_ctx_mode', 'diff')
        global_cli_mode = getattr(configs, 'can_global_cli_mode', 'inner')
        global_ctx_mode = getattr(configs, 'can_global_ctx_mode', 'abs')
        global_shift_string = str(getattr(configs, 'can_global_shifts', '')).strip()
        if global_shift_string:
            global_shifts = [s for s in parse_shift_list(global_shift_string) if s < self.d_model]
            if not global_shifts:
                global_shifts = [1]
        else:
            global_shifts = channel_shifts
        kernel_size = getattr(configs, 'can_kernel_size', 3)
        can_drop_path = getattr(configs, 'can_drop_path', 0.1)
        drop_path_schedule = getattr(
            configs, 'can_drop_path_schedule', 'linear'
        )
        self.gamma_lr_scale = float(
            getattr(configs, 'can_gamma_lr_scale', 1.0)
        )
        self.gamma_weight_decay = float(
            getattr(configs, 'can_gamma_weight_decay', 0.0)
        )
        init_values = getattr(configs, 'can_init_values', 1e-5)
        beta_init = getattr(configs, 'can_beta_init', 0.5)
        temporal_beta_init = getattr(configs, 'can_temporal_beta_init', None)
        global_beta_init = getattr(configs, 'can_global_beta_init', None)
        if temporal_beta_init is None:
            temporal_beta_init = beta_init
        if global_beta_init is None:
            global_beta_init = beta_init
        use_global_context = bool(getattr(configs, 'can_use_gffng', 1))
        enable_temporal_interaction = bool(getattr(configs, 'can_temporal_roll', 1))
        temporal_circular = bool(getattr(configs, 'can_temporal_circular', 0))
        use_orthogonal = bool(getattr(configs, 'can_use_orth', 0))
        use_context_pyramid = bool(getattr(configs, 'can_context_pyramid', 0))
        use_ffn = bool(getattr(configs, 'can_use_ffn', 0))
        d_ff_val = getattr(configs, 'd_ff', 192)
        use_cross_var = bool(getattr(configs, 'can_cross_var', 0))
        cross_var_layers = int(getattr(configs, 'can_cross_var_layers', 1))
        cross_var_context = getattr(configs, 'can_cross_var_context', 'others_mean')
        cross_var_shifts = [
            s for s in parse_shift_list(getattr(configs, 'can_cross_var_shifts', '1,2,4,8,16'))
            if s < self.d_model
        ]
        if not cross_var_shifts:
            cross_var_shifts = [1]
        use_variable_attention = bool(getattr(configs, 'can_var_attn', 0))
        variable_attention_layers = int(getattr(configs, 'can_var_attn_layers', 1))
        variable_attention_dim = int(getattr(configs, 'can_var_attn_dim', 32))
        variable_attention_top_k = int(getattr(configs, 'can_var_attn_top_k', 0))
        variable_attention_shifts = [
            s for s in parse_shift_list(getattr(configs, 'can_var_attn_shifts', '1,2,4,8'))
            if s < self.d_model
        ]
        if not variable_attention_shifts:
            variable_attention_shifts = [1]
        use_variable_embedding = bool(getattr(configs, 'can_var_embed', 0))
        use_time_mark = bool(getattr(configs, 'can_time_mark', 0))
        time_mark_mode = getattr(configs, 'can_time_mark_mode', 'flatten')
        time_mark_scale_init = float(getattr(configs, 'can_time_mark_scale_init', 1.0))
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        time_dim = freq_map.get(str(getattr(configs, 'freq', 'h')).lower(), 4)
        use_linear_residual = bool(getattr(configs, 'can_linear_residual', 0))
        linear_mode = getattr(configs, 'can_linear_mode', 'raw')
        linear_individual = bool(getattr(configs, 'can_linear_individual', 0))
        linear_scale_init = float(getattr(configs, 'can_linear_scale_init', 0.5))
        moving_avg = int(getattr(configs, 'moving_avg', 25))
        use_periodic_residual = bool(getattr(configs, 'can_periodic_residual', 0))
        periodic_periods = [
            period for period in parse_int_list(getattr(configs, 'can_periods', '24'))
            if 0 < period <= self.seq_len
        ]
        if not periodic_periods:
            periodic_periods = [min(24, self.seq_len)]
        periodic_alpha = float(getattr(configs, 'can_periodic_alpha', 0.2))
        periodic_learnable = bool(getattr(configs, 'can_periodic_learnable', 0))
        use_coarse_var_attention = bool(getattr(configs, 'can_coarse_var_attn', 0))
        coarse_var_levels = int(getattr(configs, 'can_coarse_var_levels', 3))
        coarse_var_dim = int(getattr(configs, 'can_coarse_var_dim', 32))
        coarse_var_scale_init = float(getattr(configs, 'can_coarse_var_scale_init', 0.1))
        coarse_var_mode = getattr(configs, 'can_coarse_var_mode', 'diff')
        use_hierarchical_mixer = bool(getattr(configs, 'can_hierarchical_mixer', 0))
        hierarchical_levels = int(getattr(configs, 'can_hierarchical_levels', 3))
        hierarchical_layers = int(getattr(configs, 'can_hierarchical_layers', 1))
        hierarchical_dim = int(getattr(configs, 'can_hierarchical_dim', 64))
        hierarchical_cross_scale_init = float(
            getattr(configs, 'can_hierarchical_cross_scale_init', 0.05)
        )
        hierarchical_fusion_init = float(
            getattr(configs, 'can_hierarchical_fusion_init', 0.2)
        )
        hierarchical_mode = getattr(configs, 'can_hierarchical_mode', 'blend')
        hierarchical_residual_scale_init = float(
            getattr(configs, 'can_hierarchical_residual_scale_init', 1.0)
        )
        use_periodic_image = bool(getattr(configs, 'can_periodic_image', 0))
        periodic_image_top_k = int(getattr(configs, 'can_periodic_image_top_k', 3))
        periodic_image_dim = int(getattr(configs, 'can_periodic_image_dim', 32))
        periodic_image_layers = int(getattr(configs, 'can_periodic_image_layers', 1))
        periodic_image_shift_string = str(
            getattr(configs, 'can_periodic_image_shifts', '1,2,4')
        ).strip()
        periodic_image_shifts = parse_shift_list(periodic_image_shift_string)
        periodic_image_scale_init = float(
            getattr(configs, 'can_periodic_image_scale_init', 0.0)
        )
        use_deep_periodic_image = bool(
            getattr(configs, 'can_deep_periodic_image', 0)
        )
        deep_periodic_top_k = int(
            getattr(configs, 'can_deep_periodic_top_k', 3)
        )
        deep_periodic_layers = int(
            getattr(configs, 'can_deep_periodic_layers', 1)
        )
        deep_periodic_shifts = parse_shift_list(
            getattr(configs, 'can_deep_periodic_shifts', '1,2,4')
        )
        deep_periodic_scale_init = float(
            getattr(configs, 'can_deep_periodic_scale_init', 0.1)
        )

        self.patch_embedding = PatchEmbedding(
            d_model=configs.d_model,
            patch_len=patch_len,
            stride=stride,
            padding=padding,
            dropout=configs.dropout
        )
        self.use_time_mark = use_time_mark
        if self.use_time_mark:
            self.time_patch_embedding = TimePatchEmbedding(
                time_dim=time_dim,
                d_model=configs.d_model,
                patch_len=patch_len,
                stride=stride,
                padding=padding,
                mode=time_mark_mode,
                scale_init=time_mark_scale_init
            )
        self.use_linear_residual = use_linear_residual
        if self.use_linear_residual:
            self.linear_residual = LinearForecastResidual(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                n_vars=self.enc_in,
                mode=linear_mode,
                individual=linear_individual,
                moving_avg=moving_avg,
                scale_init=linear_scale_init
            )
        self.use_periodic_residual = use_periodic_residual
        if self.use_periodic_residual:
            self.periodic_residual = PeriodicForecastResidual(
                pred_len=self.pred_len,
                periods=periodic_periods,
                alpha=periodic_alpha,
                learnable_alpha=periodic_learnable
            )
        self.use_coarse_var_attention = use_coarse_var_attention
        if self.use_coarse_var_attention:
            self.coarse_var_attention = CoarseVariableAttention(
                seq_len=self.seq_len,
                d_model=configs.d_model,
                levels=coarse_var_levels,
                attn_dim=coarse_var_dim,
                scale_init=coarse_var_scale_init,
                mode=coarse_var_mode
            )
        self.use_hierarchical_mixer = use_hierarchical_mixer
        if self.use_hierarchical_mixer:
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_state = (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None
            )
            self.hierarchical_mixer = HierarchicalCliffordMixer(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                d_model=hierarchical_dim,
                levels=hierarchical_levels,
                layers=hierarchical_layers,
                moving_avg=moving_avg,
                shifts=raw_shifts,
                cli_mode=cli_mode,
                temporal_cli_mode=temporal_cli_mode,
                ctx_mode=ctx_mode,
                kernel_size=kernel_size,
                drop_path=can_drop_path,
                init_values=init_values,
                beta_init=beta_init,
                use_global_context=use_global_context,
                use_orthogonal=use_orthogonal,
                use_context_pyramid=use_context_pyramid,
                temporal_circular=temporal_circular,
                cross_scale_init=hierarchical_cross_scale_init,
                output_mode=hierarchical_mode
            )
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            self.hierarchical_mode = hierarchical_mode
            if self.hierarchical_mode == 'blend':
                hierarchical_fusion_init = min(
                    max(hierarchical_fusion_init, 1e-4),
                    1.0 - 1e-4
                )
                self.hierarchical_fusion_logit = nn.Parameter(
                    torch.logit(torch.tensor(hierarchical_fusion_init))
                )
            elif self.hierarchical_mode == 'residual':
                self.hierarchical_residual_scale = nn.Parameter(
                    torch.tensor(hierarchical_residual_scale_init)
                )
            else:
                raise ValueError(
                    f'Invalid hierarchical mode: {self.hierarchical_mode}'
                )
        self.use_periodic_image = use_periodic_image
        if self.use_periodic_image:
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_state = (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None
            )
            self.periodic_image_refiner = PeriodicImageCliffordRefiner(
                seq_len=self.seq_len,
                d_model=periodic_image_dim,
                top_k=periodic_image_top_k,
                layers=periodic_image_layers,
                shifts=periodic_image_shifts,
                kernel_size=kernel_size,
                init_values=init_values,
                scale_init=periodic_image_scale_init,
                use_orthogonal=use_orthogonal
            )
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

        def make_blocks(scale_patch_num):
            scale_temporal_shifts = [s for s in raw_temporal_shifts if s < scale_patch_num]
            if not scale_temporal_shifts:
                scale_temporal_shifts = [1]
            if drop_path_schedule == 'linear':
                dpr = torch.linspace(
                    0, can_drop_path, configs.e_layers
                ).tolist()
            elif drop_path_schedule == 'uniform':
                dpr = [can_drop_path] * configs.e_layers
            else:
                raise ValueError(
                    f'Invalid CAN drop path schedule: {drop_path_schedule}'
                )
            return nn.ModuleList(
                [
                    CANPatchBlock(
                        dim=configs.d_model,
                        cli_mode=cli_mode,
                        temporal_cli_mode=temporal_cli_mode,
                        ctx_mode=ctx_mode,
                        channel_shifts=channel_shifts,
                        temporal_shifts=scale_temporal_shifts,
                        kernel_size=kernel_size,
                        drop_path_rate=dpr[i],
                        init_values=init_values,
                        use_global_context=use_global_context,
                        beta_init=beta_init,
                        temporal_beta_init=temporal_beta_init,
                        global_beta_init=global_beta_init,
                        global_cli_mode=global_cli_mode,
                        global_ctx_mode=global_ctx_mode,
                        global_shifts=global_shifts,
                        enable_temporal_interaction=enable_temporal_interaction,
                        use_orthogonal=use_orthogonal,
                        use_context_pyramid=use_context_pyramid,
                        use_ffn=use_ffn,
                        d_ff=d_ff_val,
                        temporal_circular=temporal_circular
                    )
                    for i in range(configs.e_layers)
                ]
            )

        self.blocks = make_blocks(patch_num)
        self.use_deep_periodic_image = use_deep_periodic_image
        if self.use_deep_periodic_image:
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_state = (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None
            )
            self.deep_periodic_mixer = DeepPeriodicImageCliffordMixer(
                dim=configs.d_model,
                top_k=deep_periodic_top_k,
                layers=deep_periodic_layers,
                shifts=deep_periodic_shifts,
                kernel_size=kernel_size,
                init_values=init_values,
                scale_init=deep_periodic_scale_init,
                use_orthogonal=use_orthogonal
            )
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        self.final_norm = nn.LayerNorm(configs.d_model)
        self.use_cross_var = use_cross_var
        self.use_variable_embedding = use_variable_embedding
        if self.use_variable_embedding:
            self.variable_embedding = nn.Parameter(torch.zeros(1, self.enc_in, configs.d_model, 1))
            nn.init.trunc_normal_(self.variable_embedding, std=0.02)
        if self.use_cross_var:
            cross_dpr = torch.linspace(0, can_drop_path, max(1, cross_var_layers)).tolist()
            self.cross_var_blocks = nn.ModuleList(
                [
                    CrossVariableCliffordBlock(
                        dim=configs.d_model,
                        shifts=cross_var_shifts,
                        context_mode=cross_var_context,
                        drop_path_rate=cross_dpr[i],
                        init_values=init_values
                    )
                    for i in range(cross_var_layers)
                ]
            )
            self.cross_var_final_norm = nn.LayerNorm(configs.d_model)
        self.use_variable_attention = use_variable_attention
        if self.use_variable_attention:
            var_attn_dpr = torch.linspace(0, can_drop_path, max(1, variable_attention_layers)).tolist()
            self.variable_attention_blocks = nn.ModuleList(
                [
                    VariableAttentionCliffordBlock(
                        dim=configs.d_model,
                        shifts=variable_attention_shifts,
                        attn_dim=variable_attention_dim,
                        top_k=variable_attention_top_k,
                        drop_path_rate=var_attn_dpr[i],
                        init_values=init_values
                    )
                    for i in range(variable_attention_layers)
                ]
            )
            self.variable_attention_final_norm = nn.LayerNorm(configs.d_model)

        head_nf = configs.d_model * patch_num
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.head = FlattenHead(self.enc_in, head_nf, self.pred_len, head_dropout=configs.dropout)
        else:
            raise NotImplementedError('CANPatchTST currently supports forecasting tasks only.')

        multiscale_patch_lens = parse_int_list(getattr(configs, 'can_multiscale_patch_lens', ''))
        multiscale_patch_lens = [
            value for value in multiscale_patch_lens
            if 1 < value <= self.seq_len and value != patch_len
        ]
        multiscale_patch_lens = list(dict.fromkeys(multiscale_patch_lens))
        self.multiscale_patch_lens = multiscale_patch_lens
        self.multiscale_embeddings = nn.ModuleList()
        self.multiscale_blocks = nn.ModuleList()
        self.multiscale_norms = nn.ModuleList()
        self.multiscale_heads = nn.ModuleList()
        stride_ratio = float(getattr(configs, 'can_multiscale_stride_ratio', 0.5))
        for scale_patch_len in self.multiscale_patch_lens:
            scale_stride = max(1, int(round(scale_patch_len * stride_ratio)))
            scale_patch_num = int((self.seq_len - scale_patch_len) / scale_stride + 2)
            self.multiscale_embeddings.append(
                PatchEmbedding(
                    d_model=configs.d_model,
                    patch_len=scale_patch_len,
                    stride=scale_stride,
                    padding=scale_stride,
                    dropout=configs.dropout
                )
            )
            self.multiscale_blocks.append(make_blocks(scale_patch_num))
            self.multiscale_norms.append(nn.LayerNorm(configs.d_model))
            self.multiscale_heads.append(
                FlattenHead(
                    self.enc_in,
                    configs.d_model * scale_patch_num,
                    self.pred_len,
                    head_dropout=configs.dropout
                )
            )
        if self.multiscale_patch_lens:
            fusion_bias = float(getattr(configs, 'can_multiscale_main_bias', 0.0))
            fusion_logits = torch.zeros(1 + len(self.multiscale_patch_lens))
            fusion_logits[0] = fusion_bias
            self.multiscale_fusion_logits = nn.Parameter(fusion_logits)

    @staticmethod
    def encode_scale(x_enc, patch_embedding, blocks, final_norm, head):
        enc_out, n_vars = patch_embedding(x_enc)
        enc_out = enc_out.transpose(1, 2)
        for block in blocks:
            enc_out = block(enc_out)
        enc_out = final_norm(enc_out.transpose(1, 2)).transpose(1, 2)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[1], enc_out.shape[2]))
        return head(enc_out).permute(0, 2, 1)

    def forecast(self, x_enc, x_mark_enc=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        else:
            means = None
            stdev = None
        normalized_x_enc = x_enc
        if self.use_periodic_image:
            if self.training:
                devices = (
                    [normalized_x_enc.device.index]
                    if normalized_x_enc.is_cuda else []
                )
                with torch.random.fork_rng(devices=devices):
                    periodic_image_correction = self.periodic_image_refiner(
                        normalized_x_enc
                    )
            else:
                periodic_image_correction = self.periodic_image_refiner(
                    normalized_x_enc
                )
            normalized_x_enc = normalized_x_enc + periodic_image_correction
            x_enc = normalized_x_enc
        linear_out = self.linear_residual(x_enc) if self.use_linear_residual else None
        variable_context = (
            self.coarse_var_attention(x_enc)
            if self.use_coarse_var_attention else None
        )

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        if variable_context is not None:
            variable_context = variable_context.reshape(-1, self.d_model).unsqueeze(-1)
            enc_out = enc_out.transpose(1, 2) + variable_context
            enc_out = enc_out.transpose(1, 2)
        if self.use_time_mark and x_mark_enc is not None:
            time_out = self.time_patch_embedding(x_mark_enc)
            time_out = time_out.repeat_interleave(n_vars, dim=0)
            enc_out = enc_out + time_out
        enc_out = enc_out.transpose(1, 2)

        for block in self.blocks:
            enc_out = block(enc_out)
        if self.use_deep_periodic_image:
            if self.training:
                devices = (
                    [enc_out.device.index]
                    if enc_out.is_cuda else []
                )
                with torch.random.fork_rng(devices=devices):
                    enc_out = self.deep_periodic_mixer(enc_out)
            else:
                enc_out = self.deep_periodic_mixer(enc_out)

        enc_out = self.final_norm(enc_out.transpose(1, 2)).transpose(1, 2)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[1], enc_out.shape[2]))
        if self.use_variable_embedding:
            enc_out = enc_out + self.variable_embedding[:, :n_vars]
        if self.use_cross_var:
            for block in self.cross_var_blocks:
                enc_out = block(enc_out)
            enc_out = self.cross_var_final_norm(
                enc_out.permute(0, 1, 3, 2)
            ).permute(0, 1, 3, 2)
        if self.use_variable_attention:
            for block in self.variable_attention_blocks:
                enc_out = block(enc_out)
            enc_out = self.variable_attention_final_norm(
                enc_out.permute(0, 1, 3, 2)
            ).permute(0, 1, 3, 2)
        dec_out = self.head(enc_out).permute(0, 2, 1)
        if self.multiscale_patch_lens:
            scale_outputs = [dec_out]
            for embedding, blocks, norm, head in zip(
                self.multiscale_embeddings,
                self.multiscale_blocks,
                self.multiscale_norms,
                self.multiscale_heads
            ):
                scale_outputs.append(
                    self.encode_scale(x_enc, embedding, blocks, norm, head)
                )
            fusion_weights = torch.softmax(self.multiscale_fusion_logits, dim=0)
            dec_out = sum(
                weight * output for weight, output in zip(fusion_weights, scale_outputs)
            )
        if linear_out is not None:
            dec_out = dec_out + linear_out
        if self.use_periodic_residual:
            dec_out = self.periodic_residual(normalized_x_enc, dec_out)
        hierarchical_out = None
        if self.use_hierarchical_mixer:
            if self.training:
                devices = (
                    [normalized_x_enc.device.index]
                    if normalized_x_enc.is_cuda else []
                )
                with torch.random.fork_rng(devices=devices):
                    hierarchical_out = self.hierarchical_mixer(
                        normalized_x_enc
                    )
            else:
                hierarchical_out = self.hierarchical_mixer(
                    normalized_x_enc
                )
        if hierarchical_out is not None:
            if self.hierarchical_mode == 'blend':
                hierarchical_weight = torch.sigmoid(
                    self.hierarchical_fusion_logit
                )
                dec_out = dec_out + hierarchical_weight * (
                    hierarchical_out - dec_out
                )
            else:
                dec_out = dec_out + (
                    self.hierarchical_residual_scale * hierarchical_out
                )

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def optimizer_param_groups(self, base_lr):
        if (
            self.gamma_lr_scale == 1.0
            and self.gamma_weight_decay == 0.0
        ):
            return self.parameters()

        gamma_parameters = []
        other_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.endswith('.gamma'):
                gamma_parameters.append(parameter)
            else:
                other_parameters.append(parameter)

        groups = [
            {
                'params': other_parameters,
                'lr': base_lr,
                'base_lr': base_lr,
                'group_name': 'main',
            }
        ]
        if gamma_parameters:
            gamma_lr = base_lr * self.gamma_lr_scale
            groups.append(
                {
                    'params': gamma_parameters,
                    'lr': gamma_lr,
                    'base_lr': gamma_lr,
                    'group_name': 'gamma',
                    'weight_decay': self.gamma_weight_decay,
                }
            )
        return groups

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :]
        raise NotImplementedError('CANPatchTST currently supports forecasting tasks only.')
