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
    def __init__(self, dim, cli_mode='inner', ctx_mode='diff', shifts=None):
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
                 enable_temporal_interaction=True, use_orthogonal=False, use_context_pyramid=False):
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
                shifts=temporal_shifts
            )
            self.temporal_beta = nn.Parameter(torch.tensor(beta_init))

        self.use_global_context = use_global_context
        if self.use_global_context:
            self.global_interaction = CliffordChannelInteraction1D(
                dim=dim,
                cli_mode='inner',
                ctx_mode='abs',
                shifts=channel_shifts
            )
            self.global_beta = nn.Parameter(torch.tensor(beta_init))

        self.gate_fc = nn.Conv1d(dim * 2, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1), init_values))
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

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
        return shortcut + self.drop_path(self.gamma * mixed)


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

        patch_len = getattr(configs, 'patch_len', 16)
        stride = getattr(configs, 'can_stride', max(1, patch_len // 2))
        padding = stride

        raw_shifts = parse_shift_list(getattr(configs, 'can_shifts', '1,2,4,8'))
        patch_num = int((configs.seq_len - patch_len) / stride + 2)
        channel_shifts = [s for s in raw_shifts if s < self.d_model]
        temporal_shifts = [s for s in raw_shifts if s < patch_num]
        if not channel_shifts:
            channel_shifts = [1]
        if not temporal_shifts:
            temporal_shifts = [1]

        cli_mode = getattr(configs, 'can_cli_mode', 'full')
        temporal_cli_mode = getattr(configs, 'can_temporal_cli_mode', 'inner')
        ctx_mode = getattr(configs, 'can_ctx_mode', 'diff')
        kernel_size = getattr(configs, 'can_kernel_size', 3)
        can_drop_path = getattr(configs, 'can_drop_path', 0.1)
        init_values = getattr(configs, 'can_init_values', 1e-5)
        beta_init = getattr(configs, 'can_beta_init', 0.5)
        use_global_context = bool(getattr(configs, 'can_use_gffng', 1))
        enable_temporal_interaction = bool(getattr(configs, 'can_temporal_roll', 1))
        use_orthogonal = bool(getattr(configs, 'can_use_orth', 0))
        use_context_pyramid = bool(getattr(configs, 'can_context_pyramid', 0))

        self.patch_embedding = PatchEmbedding(
            d_model=configs.d_model,
            patch_len=patch_len,
            stride=stride,
            padding=padding,
            dropout=configs.dropout
        )

        dpr = torch.linspace(0, can_drop_path, configs.e_layers).tolist()
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
                    drop_path_rate=dpr[i],
                    init_values=init_values,
                    use_global_context=use_global_context,
                    beta_init=beta_init,
                    enable_temporal_interaction=enable_temporal_interaction,
                    use_orthogonal=use_orthogonal,
                    use_context_pyramid=use_context_pyramid
                )
                for i in range(configs.e_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(configs.d_model)

        head_nf = configs.d_model * patch_num
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.head = FlattenHead(self.enc_in, head_nf, self.pred_len, head_dropout=configs.dropout)
        else:
            raise NotImplementedError('CANPatchTST currently supports forecasting tasks only.')

    def forecast(self, x_enc):
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

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        raise NotImplementedError('CANPatchTST currently supports forecasting tasks only.')
