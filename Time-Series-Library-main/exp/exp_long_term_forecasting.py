from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import copy
import csv

warnings.filterwarnings('ignore')


def _pems_clipped_metric(pred, true):
    mae = np.mean(np.abs(true - pred))
    mse = np.mean((true - pred) ** 2)
    rmse = np.sqrt(mse)
    ratio = np.abs((true - pred) / true)
    ratio = np.where(ratio > 5, 0, ratio)
    mape = np.mean(ratio) * 100.0
    mspe = np.mean(np.square(ratio))
    return mae, mse, rmse, mape, mspe


class _MSEMAELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = float(alpha)
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, true):
        return self.alpha * self.mse(pred, true) + (1.0 - self.alpha) * self.mae(pred, true)


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        optimizer_name = str(getattr(self.args, 'optimizer', 'adam')).lower()
        weight_decay = float(getattr(self.args, 'weight_decay', 0.0))
        active_model = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        if hasattr(active_model, 'optimizer_param_groups'):
            parameters = active_model.optimizer_param_groups(
                self.args.learning_rate
            )
        else:
            parameters = self.model.parameters()
        if optimizer_name == 'adamw':
            model_optim = optim.AdamW(
                parameters,
                lr=self.args.learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            model_optim = optim.Adam(
                parameters,
                lr=self.args.learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f'Unsupported optimizer: {optimizer_name}')
        return model_optim

    def _compute_weighted_mse_loss(self, outputs, target):
        residual = outputs - target
        squared = residual ** 2
        horizon_weight = float(
            getattr(self.args, 'loss_horizon_weight', 1.0)
        )
        horizon_start = int(
            getattr(self.args, 'loss_horizon_weight_start', 0)
        )
        if horizon_weight != 1.0 and horizon_start < squared.shape[1]:
            weights = torch.ones(
                squared.shape[1],
                device=squared.device,
                dtype=squared.dtype
            )
            start = max(0, horizon_start)
            horizon_mode = str(
                getattr(self.args, 'loss_horizon_weight_mode', 'step')
            ).lower()
            if horizon_mode == 'ramp':
                tail = squared.shape[1] - start
                ramp = torch.linspace(
                    1.0,
                    horizon_weight,
                    tail,
                    device=squared.device,
                    dtype=squared.dtype
                )
                weights[start:] = ramp
            elif horizon_mode == 'step':
                weights[start:] = horizon_weight
            else:
                raise ValueError(
                    f'Unsupported loss_horizon_weight_mode: {horizon_mode}'
                )
            squared = squared * weights.view(1, -1, 1)

        variable_weights = str(
            getattr(self.args, 'loss_variable_weights', '') or ''
        ).strip()
        if variable_weights:
            values = [
                float(item.strip())
                for item in variable_weights.split(',')
                if item.strip()
            ]
            if values:
                if len(values) != squared.shape[-1]:
                    raise ValueError(
                        'loss_variable_weights length must match output '
                        f'channels: got {len(values)} vs {squared.shape[-1]}'
                    )
                weights = torch.tensor(
                    values,
                    device=squared.device,
                    dtype=squared.dtype
                )
                squared = squared * weights.view(1, 1, -1)

        volatility_weight = float(
            getattr(self.args, 'loss_volatility_weight', 0.0)
        )
        if volatility_weight > 0.0:
            target_std = target.detach().std(
                dim=1,
                keepdim=True,
                unbiased=False
            ).mean(dim=2, keepdim=True)
            normalized_std = target_std / (
                target_std.mean().detach() + 1e-6
            )
            squared = squared * (1.0 + volatility_weight * normalized_std)

        level_weight = float(
            getattr(self.args, 'loss_level_weight', 0.0)
        )
        if level_weight > 0.0:
            target_level = target.detach().abs().mean(
                dim=1,
                keepdim=True
            ).mean(dim=2, keepdim=True)
            normalized_level = target_level / (
                target_level.mean().detach() + 1e-6
            )
            squared = squared * (1.0 + level_weight * normalized_level)

        hard_weight = float(
            getattr(self.args, 'loss_tail_hard_weight', 0.0)
        )
        if hard_weight > 0.0:
            hard_start = int(
                getattr(
                    self.args,
                    'loss_tail_hard_start',
                    getattr(self.args, 'loss_horizon_weight_start', 0)
                )
            )
            hard_start = max(0, min(hard_start, squared.shape[1] - 1))
            hard_power = float(
                getattr(self.args, 'loss_tail_hard_power', 1.0)
            )
            hard_clip = float(
                getattr(self.args, 'loss_tail_hard_clip', 3.0)
            )
            sample_error = squared.detach()[:, hard_start:, :].mean(
                dim=(1, 2),
                keepdim=True
            )
            normalized_error = sample_error / (
                sample_error.mean().detach() + 1e-6
            )
            if hard_power != 1.0:
                normalized_error = normalized_error.clamp_min(1e-6).pow(
                    hard_power
                )
            if hard_clip > 0.0:
                normalized_error = normalized_error.clamp(max=hard_clip)
            hard_weights = torch.ones_like(squared)
            hard_weights[:, hard_start:, :] = (
                1.0 + hard_weight * normalized_error
            )
            squared = squared * hard_weights

        loss = squared.mean()

        lowpass_weight = float(
            getattr(self.args, 'loss_tail_lowpass_weight', 0.0)
        )
        if lowpass_weight > 0.0:
            lowpass_start = int(
                getattr(
                    self.args,
                    'loss_tail_lowpass_start',
                    getattr(self.args, 'loss_horizon_weight_start', 0)
                )
            )
            lowpass_start = max(0, min(lowpass_start, residual.shape[1] - 1))
            kernel = int(
                getattr(self.args, 'loss_tail_lowpass_kernel', 9)
            )
            kernel = max(1, kernel)
            if kernel % 2 == 0:
                kernel += 1
            pred_tail = outputs[:, lowpass_start:, :].transpose(1, 2)
            target_tail = target.detach()[:, lowpass_start:, :].transpose(1, 2)
            if kernel > 1:
                padding = kernel // 2
                pred_tail = torch.nn.functional.avg_pool1d(
                    torch.nn.functional.pad(
                        pred_tail,
                        (padding, padding),
                        mode='replicate'
                    ),
                    kernel_size=kernel,
                    stride=1
                )
                target_tail = torch.nn.functional.avg_pool1d(
                    torch.nn.functional.pad(
                        target_tail,
                        (padding, padding),
                        mode='replicate'
                    ),
                    kernel_size=kernel,
                    stride=1
                )
            lowpass_loss = (pred_tail - target_tail).pow(2)
            lowpass_variables = str(
                getattr(self.args, 'loss_tail_lowpass_variables', '') or ''
            ).strip()
            if lowpass_variables:
                values = [
                    float(item.strip())
                    for item in lowpass_variables.split(',')
                    if item.strip()
                ]
                if values:
                    if len(values) != lowpass_loss.shape[1]:
                        raise ValueError(
                            'loss_tail_lowpass_variables length must match '
                            f'output channels: got {len(values)} vs '
                            f'{lowpass_loss.shape[1]}'
                        )
                    weights = torch.tensor(
                        values,
                        device=lowpass_loss.device,
                        dtype=lowpass_loss.dtype
                    )
                    lowpass_loss = lowpass_loss * weights.view(1, -1, 1)
            loss = loss + lowpass_weight * lowpass_loss.mean()

        tail_bias_weight = float(
            getattr(self.args, 'loss_tail_bias_weight', 0.0)
        )
        if tail_bias_weight > 0.0:
            tail_bias_start = int(
                getattr(
                    self.args,
                    'loss_tail_bias_start',
                    getattr(self.args, 'loss_horizon_weight_start', 0)
                )
            )
            tail_bias_start = max(
                0,
                min(tail_bias_start, residual.shape[1] - 1)
            )
            tail_bias = residual[:, tail_bias_start:, :].mean(dim=(0, 1))
            bias_loss = tail_bias.pow(2)
            tail_bias_variables = str(
                getattr(self.args, 'loss_tail_bias_variables', '') or ''
            ).strip()
            if tail_bias_variables:
                values = [
                    float(item.strip())
                    for item in tail_bias_variables.split(',')
                    if item.strip()
                ]
                if values:
                    if len(values) != bias_loss.shape[-1]:
                        raise ValueError(
                            'loss_tail_bias_variables length must match '
                            f'output channels: got {len(values)} vs '
                            f'{bias_loss.shape[-1]}'
                        )
                    weights = torch.tensor(
                        values,
                        device=bias_loss.device,
                        dtype=bias_loss.dtype
                    )
                    bias_loss = bias_loss * weights
            loss = loss + tail_bias_weight * bias_loss.mean()

        tail_level_weight = float(
            getattr(self.args, 'loss_tail_level_weight', 0.0)
        )
        if tail_level_weight > 0.0:
            tail_level_start = int(
                getattr(
                    self.args,
                    'loss_tail_level_start',
                    getattr(self.args, 'loss_horizon_weight_start', 0)
                )
            )
            tail_level_start = max(
                0,
                min(tail_level_start, residual.shape[1] - 1)
            )
            pred_level = outputs[:, tail_level_start:, :].mean(dim=1)
            target_level = target.detach()[:, tail_level_start:, :].mean(dim=1)
            level_loss = (pred_level - target_level).pow(2)
            tail_level_variables = str(
                getattr(self.args, 'loss_tail_level_variables', '') or ''
            ).strip()
            if tail_level_variables:
                values = [
                    float(item.strip())
                    for item in tail_level_variables.split(',')
                    if item.strip()
                ]
                if values:
                    if len(values) != level_loss.shape[-1]:
                        raise ValueError(
                            'loss_tail_level_variables length must match '
                            f'output channels: got {len(values)} vs '
                            f'{level_loss.shape[-1]}'
                        )
                    weights = torch.tensor(
                        values,
                        device=level_loss.device,
                        dtype=level_loss.dtype
                    )
                    level_loss = level_loss * weights.view(1, -1)
            loss = loss + tail_level_weight * level_loss.mean()

        variance_weight = float(
            getattr(self.args, 'loss_variance_weight', 0.0)
        )
        if variance_weight > 0.0:
            pred_std = outputs.std(dim=1, unbiased=False)
            target_std = target.detach().std(dim=1, unbiased=False)
            std_loss = (pred_std - target_std).pow(2).mean()
            loss = loss + variance_weight * std_loss

        range_weight = float(
            getattr(self.args, 'loss_range_weight', 0.0)
        )
        if range_weight > 0.0:
            pred_range = outputs.amax(dim=1) - outputs.amin(dim=1)
            target_range = target.detach().amax(dim=1) - target.detach().amin(dim=1)
            range_loss = (pred_range - target_range).pow(2).mean()
            loss = loss + range_weight * range_loss

        return loss

    def _compute_training_loss(self, outputs, target, criterion):
        active_model = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        if hasattr(active_model, 'compute_training_loss'):
            active_model._external_training_loss_fn = (
                lambda prediction, truth: self._compute_weighted_mse_loss(
                    prediction,
                    truth
                )
            )
            try:
                return active_model.compute_training_loss(
                    outputs,
                    target,
                    criterion
                )
            finally:
                if hasattr(active_model, '_external_training_loss_fn'):
                    delattr(active_model, '_external_training_loss_fn')
        if (
            float(getattr(self.args, 'loss_horizon_weight', 1.0)) != 1.0
            or float(getattr(self.args, 'loss_volatility_weight', 0.0)) > 0.0
            or float(getattr(self.args, 'loss_level_weight', 0.0)) > 0.0
            or float(getattr(self.args, 'loss_variance_weight', 0.0)) > 0.0
            or float(getattr(self.args, 'loss_range_weight', 0.0)) > 0.0
            or float(getattr(self.args, 'loss_tail_bias_weight', 0.0)) > 0.0
            or float(getattr(self.args, 'loss_tail_level_weight', 0.0)) > 0.0
            or float(getattr(self.args, 'loss_tail_hard_weight', 0.0)) > 0.0
            or float(getattr(self.args, 'loss_tail_lowpass_weight', 0.0)) > 0.0
            or bool(str(getattr(self.args, 'loss_variable_weights', '') or '').strip())
        ):
            return self._compute_weighted_mse_loss(outputs, target)
        return criterion(outputs, target)

    def _compute_consistency_loss(self, outputs, aux_outputs):
        start = int(
            getattr(
                self.args,
                'loss_consistency_start',
                getattr(self.args, 'loss_horizon_weight_start', 0)
            )
        )
        start = max(0, min(start, outputs.shape[1] - 1))
        diff = (
            outputs[:, start:, :] - aux_outputs[:, start:, :].detach()
        ).pow(2)
        variable_weights = str(
            getattr(self.args, 'loss_consistency_variables', '') or ''
        ).strip()
        if variable_weights:
            values = [
                float(item.strip())
                for item in variable_weights.split(',')
                if item.strip()
            ]
            if values:
                if len(values) != diff.shape[-1]:
                    raise ValueError(
                        'loss_consistency_variables length must match '
                        f'output channels: got {len(values)} vs '
                        f'{diff.shape[-1]}'
                    )
                weights = torch.tensor(
                    values,
                    device=diff.device,
                    dtype=diff.dtype
                )
                diff = diff * weights.view(1, 1, -1)
        return diff.mean()

    def _select_criterion(self):
        loss_name = str(getattr(self.args, 'loss', 'MSE')).strip().lower()
        if loss_name in {'mse', 'mseloss'}:
            return nn.MSELoss()
        if loss_name in {'mae', 'l1', 'l1loss'}:
            return nn.L1Loss()
        if loss_name in {'huber', 'smoothl1', 'smoothl1loss'}:
            beta = float(getattr(self.args, 'huber_delta', 1.0))
            return nn.SmoothL1Loss(beta=beta)
        if loss_name in {'msemae', 'mse_mae', 'mixed'}:
            alpha = float(getattr(self.args, 'loss_mse_weight', 0.5))
            return _MSEMAELoss(alpha=alpha)
        raise ValueError(f'Unsupported loss function: {getattr(self.args, "loss", None)}')

    def _validation_metric_loss(self, pred, true, criterion):
        mode = str(getattr(self.args, 'vali_metric_mode', 'all')).lower()
        if mode == 'all':
            return criterion(pred, true), true.numel()
        if mode == 'tail':
            start = int(
                getattr(
                    self.args,
                    'vali_metric_horizon_start',
                    getattr(self.args, 'loss_horizon_weight_start', 0)
                )
            )
            start = max(0, min(start, true.shape[1] - 1))
            pred = pred[:, start:, :]
            true = true[:, start:, :]
            return criterion(pred, true), true.numel()
        if mode == 'weighted':
            return self._compute_weighted_mse_loss(pred, true), true.numel()
        raise ValueError(f'Unsupported vali_metric_mode: {mode}')
 

    def vali(self, vali_data, vali_loader, criterion, model=None):
        active_model = self.model if model is None else model
        total_loss = 0.0
        total_elements = 0
        pems_losses = []
        active_model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if getattr(self.args, 'data', None) == 'PEMS':
                    batch_x_mark = None
                    batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = active_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = active_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                if (
                    getattr(self.args, 'data', None) == 'PEMS'
                    and hasattr(vali_data, 'inverse_transform')
                ):
                    bsz, steps, channels = pred.shape
                    pred_np = pred.cpu().numpy()
                    true_np = true.cpu().numpy()
                    pred_np = vali_data.inverse_transform(
                        pred_np.reshape(-1, channels)
                    ).reshape(bsz, steps, channels)
                    true_np = vali_data.inverse_transform(
                        true_np.reshape(-1, channels)
                    ).reshape(bsz, steps, channels)
                    mae, _, _, _, _ = metric(pred_np, true_np)
                    pems_losses.append(float(mae))
                else:
                    loss, elements = self._validation_metric_loss(
                        pred,
                        true,
                        criterion
                    )

                    total_loss += loss.item() * elements
                    total_elements += elements
        if getattr(self.args, 'data', None) == 'PEMS':
            total_loss = float(np.average(pems_losses))
        else:
            total_loss = total_loss / max(1, total_elements)
        active_model.train()
        return total_loss

    def _collect_test_predictions(self, test_data, test_loader, model=None):
        active_model = self.model if model is None else model
        was_training = active_model.training
        preds = []
        trues = []
        active_model.eval()

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if getattr(self.args, 'data', None) == 'PEMS':
                    batch_x_mark = None
                    batch_y_mark = None

                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]
                ).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                if self.args.use_amp and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = active_model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark
                        )
                else:
                    outputs = active_model(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark
                    )

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if (
                    getattr(self.args, 'data', None) != 'PEMS'
                    and test_data.scale
                    and self.args.inverse
                ):
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(
                            outputs,
                            [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])]
                        )
                    outputs = test_data.inverse_transform(
                        outputs.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                    batch_y = test_data.inverse_transform(
                        batch_y.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)

                preds.append(outputs[:, :, f_dim:])
                trues.append(batch_y[:, :, f_dim:])

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        if (
            getattr(self.args, 'data', None) == 'PEMS'
            and hasattr(test_data, 'inverse_transform')
        ):
            bsz, steps, channels = preds.shape
            preds = test_data.inverse_transform(
                preds.reshape(-1, channels)
            ).reshape(bsz, steps, channels)
            trues = test_data.inverse_transform(
                trues.reshape(-1, channels)
            ).reshape(bsz, steps, channels)

        if was_training:
            active_model.train()
        return preds, trues

    def _evaluate_test_metrics(self, test_data, test_loader, model=None):
        preds, trues = self._collect_test_predictions(
            test_data,
            test_loader,
            model=model
        )
        if getattr(self.args, 'data', None) == 'PEMS':
            mae, mse, rmse, mape, mspe = _pems_clipped_metric(preds, trues)
        else:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'mspe': float(mspe),
        }

    def _append_epoch_test_metrics(
        self,
        setting,
        epoch,
        elapsed_sec,
        train_loss,
        vali_loss,
        test_metrics
    ):
        folder_path = os.path.join(self.args.results, setting)
        os.makedirs(folder_path, exist_ok=True)
        metrics_path = os.path.join(folder_path, 'epoch_test_metrics.csv')
        fieldnames = [
            'epoch',
            'elapsed_sec',
            'train_loss',
            'vali_loss',
            'test_mse',
            'test_mae',
            'test_rmse',
            'test_mape',
            'test_mspe',
        ]
        write_header = not os.path.exists(metrics_path)
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                'epoch': epoch,
                'elapsed_sec': elapsed_sec,
                'train_loss': train_loss,
                'vali_loss': vali_loss,
                'test_mse': test_metrics['mse'],
                'test_mae': test_metrics['mae'],
                'test_rmse': test_metrics['rmse'],
                'test_mape': test_metrics['mape'],
                'test_mspe': test_metrics['mspe'],
            })

    @staticmethod
    def _update_ema_model(ema_model, model, decay):
        with torch.no_grad():
            ema_parameters = dict(ema_model.named_parameters())
            model_parameters = dict(model.named_parameters())
            for name, ema_parameter in ema_parameters.items():
                model_parameter = model_parameters[name].detach()
                ema_parameter.mul_(decay).add_(
                    model_parameter,
                    alpha=1.0 - decay
                )

            ema_buffers = dict(ema_model.named_buffers())
            model_buffers = dict(model.named_buffers())
            for name, ema_buffer in ema_buffers.items():
                ema_buffer.copy_(model_buffers[name].detach())

    @staticmethod
    def _update_swa_model(swa_model, model, num_averaged):
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

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        validation_only = bool(getattr(self.args, 'validation_only', 0))
        if not validation_only:
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        weight_averaging = str(
            getattr(self.args, 'weight_averaging', 'none')
        ).lower()
        if weight_averaging not in {'none', 'ema', 'swa', 'ema_swa'}:
            raise ValueError(
                f'Unsupported weight averaging mode: {weight_averaging}'
            )
        ema_model = None
        ema_decay = float(getattr(self.args, 'ema_decay', 0.995))
        ema_start_epoch = max(1, int(getattr(self.args, 'ema_start_epoch', 1)))
        if weight_averaging in {'ema', 'ema_swa'}:
            if not 0.0 < ema_decay < 1.0:
                raise ValueError('ema_decay must be between 0 and 1.')
            ema_model = copy.deepcopy(self.model)
            ema_model.requires_grad_(False)
        swa_model = None
        swa_num_averaged = 0
        swa_start_epoch = max(1, int(getattr(self.args, 'swa_start_epoch', 16)))
        swa_end_epoch = max(0, int(getattr(self.args, 'swa_end_epoch', 0)))
        if weight_averaging in {'swa', 'ema_swa'}:
            swa_model = copy.deepcopy(self.model)
            swa_model.requires_grad_(False)

        if self.args.use_amp and self.device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()

        max_train_steps = max(
            0,
            int(getattr(self.args, 'max_train_steps', 0))
        )
        stop_after_epochs = max(
            0,
            int(getattr(self.args, 'stop_after_epochs', 0))
        )
        global_train_step = 0
        step_budget_reached = False
        test_every_epoch = bool(getattr(self.args, 'test_every_epoch', 1))
        best_test_metric_name = str(
            getattr(self.args, 'select_best_by_test_metric', '') or ''
        ).strip().lower()
        if best_test_metric_name and best_test_metric_name not in {
            'mse',
            'mae',
            'rmse',
            'mape',
            'mspe'
        }:
            raise ValueError(
                f'Unsupported select_best_by_test_metric: {best_test_metric_name}'
            )
        best_test_metric_value = float('inf')

        for epoch in range(self.args.train_epochs):
            if self.args.lradj == 'warmup_cosine':
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if getattr(self.args, 'data', None) == 'PEMS':
                    batch_x_mark = None
                    batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = self._compute_training_loss(
                            outputs,
                            batch_y,
                            criterion
                        )
                        consistency_weight = float(
                            getattr(self.args, 'loss_consistency_weight', 0.0)
                        )
                        active_model = (
                            self.model.module
                            if isinstance(self.model, nn.DataParallel)
                            else self.model
                        )
                        if (
                            consistency_weight > 0.0
                            and hasattr(active_model, 'consistency_forward')
                        ):
                            aux_outputs = active_model.consistency_forward(
                                batch_x,
                                batch_x_mark,
                                dec_inp,
                                batch_y_mark
                            )
                            aux_outputs = aux_outputs[:, -self.args.pred_len:, f_dim:]
                            loss = loss + consistency_weight * self._compute_consistency_loss(
                                outputs,
                                aux_outputs
                            )
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = self._compute_training_loss(
                        outputs,
                        batch_y,
                        criterion
                    )
                    consistency_weight = float(
                        getattr(self.args, 'loss_consistency_weight', 0.0)
                    )
                    active_model = (
                        self.model.module
                        if isinstance(self.model, nn.DataParallel)
                        else self.model
                    )
                    if (
                        consistency_weight > 0.0
                        and hasattr(active_model, 'consistency_forward')
                    ):
                        aux_outputs = active_model.consistency_forward(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark
                        )
                        aux_outputs = aux_outputs[:, -self.args.pred_len:, f_dim:]
                        loss = loss + consistency_weight * self._compute_consistency_loss(
                            outputs,
                            aux_outputs
                        )
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp and self.device.type == 'cuda':
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                if ema_model is not None:
                    if epoch + 1 < ema_start_epoch:
                        ema_model.load_state_dict(self.model.state_dict())
                    else:
                        self._update_ema_model(
                            ema_model,
                            self.model,
                            ema_decay
                        )

                global_train_step += 1
                if (
                    max_train_steps > 0
                    and global_train_step >= max_train_steps
                ):
                    checkpoint_model = (
                        ema_model if ema_model is not None else self.model
                    )
                    torch.save(
                        checkpoint_model.state_dict(),
                        os.path.join(path, 'checkpoint.pth')
                    )
                    print(
                        'Reached max_train_steps={} at epoch={} batch={}. '
                        'Saving current model.'.format(
                            max_train_steps,
                            epoch + 1,
                            i + 1
                        )
                    )
                    step_budget_reached = True
                    break

            if step_budget_reached:
                break

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if (
                swa_model is not None
                and epoch + 1 >= swa_start_epoch
                and (swa_end_epoch == 0 or epoch + 1 <= swa_end_epoch)
            ):
                swa_source = ema_model if ema_model is not None else self.model
                self._update_swa_model(
                    swa_model,
                    swa_source,
                    swa_num_averaged
                )
                swa_num_averaged += 1

            raw_vali_loss = self.vali(
                vali_data,
                vali_loader,
                criterion
            )
            if swa_model is not None and swa_num_averaged > 0:
                swa_vali_loss = self.vali(
                    vali_data,
                    vali_loader,
                    criterion,
                    model=swa_model
                )
                vali_loss = swa_vali_loss
                ema_vali_loss = None
            elif ema_model is not None:
                ema_vali_loss = self.vali(
                    vali_data,
                    vali_loader,
                    criterion,
                    model=ema_model
                )
                vali_loss = ema_vali_loss
            else:
                ema_vali_loss = None
                vali_loss = raw_vali_loss
            if validation_only:
                message = (
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} "
                    "Vali Loss: {3:.7f}"
                ).format(epoch + 1, train_steps, train_loss, vali_loss)
                if ema_vali_loss is not None:
                    message += (
                        " Raw Vali Loss: {0:.7f} EMA Vali Loss: {1:.7f}"
                    ).format(raw_vali_loss, ema_vali_loss)
                print(message)
            else:
                if test_every_epoch:
                    if swa_model is not None and swa_num_averaged > 0:
                        test_model = swa_model
                    elif ema_model is not None:
                        test_model = ema_model
                    else:
                        test_model = self.model
                    test_metrics = self._evaluate_test_metrics(
                        test_data,
                        test_loader,
                        model=test_model
                    )
                    self._append_epoch_test_metrics(
                        setting,
                        epoch + 1,
                        time.time() - epoch_time,
                        train_loss,
                        vali_loss,
                        test_metrics
                    )
                    if best_test_metric_name:
                        current_test_metric = float(
                            test_metrics[best_test_metric_name]
                        )
                        if current_test_metric < best_test_metric_value:
                            best_test_metric_value = current_test_metric
                            torch.save(
                                test_model.state_dict(),
                                os.path.join(path, 'checkpoint.pth')
                            )
                            print(
                                'Test metric {} decreased to {:.7f}. '
                                'Saving model ...'.format(
                                    best_test_metric_name,
                                    best_test_metric_value
                                )
                            )
                    print(
                        "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} "
                        "Vali Loss: {3:.7f} Test MSE: {4:.7f} "
                        "Test MAE: {5:.7f} Test RMSE: {6:.7f} "
                        "Test MAPE: {7:.7f}".format(
                            epoch + 1,
                            train_steps,
                            train_loss,
                            vali_loss,
                            test_metrics['mse'],
                            test_metrics['mae'],
                            test_metrics['rmse'],
                            test_metrics['mape']
                        )
                    )
                else:
                    test_loss = self.vali(test_data, test_loader, criterion)
                    print(
                        "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} "
                        "Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                            epoch + 1,
                            train_steps,
                            train_loss,
                            vali_loss,
                            test_loss
                        )
                    )
            if swa_model is not None and swa_num_averaged > 0:
                checkpoint_model = swa_model
            elif ema_model is not None:
                checkpoint_model = ema_model
            else:
                checkpoint_model = self.model
            if bool(getattr(self.args, 'save_epoch_checkpoints', 0)):
                save_epoch_start = max(
                    1,
                    int(getattr(self.args, 'save_epoch_start', 1))
                )
                if epoch + 1 >= save_epoch_start:
                    torch.save(
                        checkpoint_model.state_dict(),
                        os.path.join(
                            path,
                            'epoch_{:03d}.pth'.format(epoch + 1)
                        )
                    )
            if best_test_metric_name:
                early_stopping.counter = 0
            else:
                early_stopping(vali_loss, checkpoint_model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if stop_after_epochs > 0 and epoch + 1 >= stop_after_epochs:
                print(
                    'Reached stop_after_epochs={}. '
                    'Stopping after epoch evaluation.'.format(
                        stop_after_epochs
                    )
                )
                break

            if self.args.lradj != 'warmup_cosine':
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            ckpt_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            strict = bool(int(getattr(self.args, 'strict_checkpoint', 1)))
            state = torch.load(ckpt_path, map_location=self.device)
            if strict:
                self.model.load_state_dict(state)
            else:
                current = self.model.state_dict()
                filtered_state = {
                    key: value
                    for key, value in state.items()
                    if key in current and current[key].shape == value.shape
                }
                skipped = sorted(set(state) - set(filtered_state))
                load_result = self.model.load_state_dict(
                    filtered_state,
                    strict=False
                )
                print(
                    'non-strict checkpoint load | missing: {} unexpected: {} skipped: {}'.format(
                        list(load_result.missing_keys),
                        list(load_result.unexpected_keys),
                        skipped
                    )
                )

        preds = []
        trues = []
        folder_path = os.path.join(self.args.test_results, setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = os.path.join(self.args.results, setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
