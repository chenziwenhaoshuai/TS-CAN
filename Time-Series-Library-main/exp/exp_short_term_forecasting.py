from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary, mase, mape as m4_mape, smape_2
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import csv
import os
import time
import warnings
import numpy as np
import pandas

warnings.filterwarnings('ignore')


class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def _collect_test_predictions(self, train_loader, test_loader):
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            batch_size = 500
            preds = []
            for start in range(0, x.shape[0], batch_size):
                xb = x[start:start + batch_size]
                dec_inp = torch.zeros(
                    (xb.shape[0], self.args.pred_len, xb.shape[-1])
                ).float().to(self.device)
                dec_inp = torch.cat(
                    [xb[:, -self.args.label_len:, :], dec_inp],
                    dim=1
                ).float()
                outputs = self.model(xb, None, dec_inp, None)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                preds.append(outputs.detach().cpu())

            pred = torch.cat(preds, dim=0).numpy()
            true = np.array(y)

        if was_training:
            self.model.train()
        return pred, true, x.detach().cpu().numpy()

    def _evaluate_test_metrics(self, train_loader, test_loader, criterion):
        preds, trues, insample = self._collect_test_predictions(
            train_loader,
            test_loader
        )
        preds = preds[:, :, 0]
        insample = insample[:, :, 0]
        batch_y_mark = torch.ones(trues.shape)
        test_loss = criterion(
            torch.from_numpy(insample),
            self.args.frequency_map,
            torch.from_numpy(preds),
            torch.from_numpy(trues),
            batch_y_mark
        )

        if self.args.data == 'm4':
            frequency = self.args.frequency_map
            test_mase = np.mean([
                mase(
                    forecast=preds[i],
                    insample=train_loader.dataset.timeseries[i],
                    outsample=trues[i],
                    frequency=frequency
                )
                for i in range(len(preds))
            ])
            return {
                'loss': float(test_loss),
                'smape': float(np.mean(smape_2(preds, trues))),
                'mape': float(np.mean(m4_mape(preds, trues))),
                'mase': float(test_mase),
            }

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        return {
            'loss': float(test_loss),
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape) * 100.0,
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
        folder_path = os.path.join(
            getattr(self.args, 'results', './short_term_results'),
            'epoch_test_metrics'
        )
        os.makedirs(folder_path, exist_ok=True)
        metrics_path = os.path.join(folder_path, setting + '.csv')
        base_fields = [
            'epoch',
            'elapsed_sec',
            'train_loss',
            'vali_loss',
            'test_loss',
        ]
        metric_fields = [
            'test_smape',
            'test_mape',
            'test_mase',
            'test_mse',
            'test_mae',
            'test_rmse',
            'test_mspe',
        ]
        fieldnames = base_fields + metric_fields
        write_header = not os.path.exists(metrics_path)
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            row = {
                'epoch': epoch,
                'elapsed_sec': elapsed_sec,
                'train_loss': train_loss,
                'vali_loss': vali_loss,
                'test_loss': test_metrics['loss'],
            }
            for key in metric_fields:
                row[key] = test_metrics.get(key.replace('test_', ''), '')
            writer.writerow(row)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)
        mse = nn.MSELoss()
        test_every_epoch = bool(getattr(self.args, 'test_every_epoch', 1))

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, None, dec_inp, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                loss = loss_value  # + loss_sharpness * 1e-5
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            if test_every_epoch:
                test_metrics = self._evaluate_test_metrics(
                    train_loader,
                    test_loader,
                    criterion
                )
                self._append_epoch_test_metrics(
                    setting,
                    epoch + 1,
                    time.time() - epoch_time,
                    train_loss,
                    vali_loss,
                    test_metrics
                )
            else:
                test_metrics = {'loss': float(vali_loss)}
            message = (
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} "
                "Vali Loss: {3:.7f} Test Loss: {4:.7f}"
            ).format(
                epoch + 1,
                train_steps,
                train_loss,
                vali_loss,
                test_metrics['loss']
            )
            if test_every_epoch:
                if self.args.data == 'm4':
                    message += (
                        " Test SMAPE: {0:.7f} Test MAPE: {1:.7f} "
                        "Test MASE: {2:.7f}"
                    ).format(
                        test_metrics['smape'],
                        test_metrics['mape'],
                        test_metrics['mase']
                    )
                else:
                    message += (
                        " Test MSE: {0:.7f} Test MAE: {1:.7f} "
                        "Test RMSE: {2:.7f} Test MAPE: {3:.7f}"
                    ).format(
                        test_metrics['mse'],
                        test_metrics['mae'],
                        test_metrics['rmse'],
                        test_metrics['mape']
                    )
            print(message)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float()  # .to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None,
                                                                      dec_inp[id_list[i]:id_list[i + 1]],
                                                                      None).detach().cpu()
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            pred = outputs
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape)

            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train()
        return loss

    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None,
                                                                      dec_inp[id_list[i]:id_list[i + 1]], None)

                if id_list[i] % 1000 == 0:
                    print(id_list[i])

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x = x.detach().cpu().numpy()

            for i in range(0, preds.shape[0], preds.shape[0] // 10):
                gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        print('test shape:', preds.shape)

        # result save
        folder_path = './m4_results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        print(self.args.model)
        file_path = './m4_results/' + self.args.model + '/'
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            try:
                m4_summary = M4Summary(file_path, self.args.root_path)
                # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
                smape_results, owa_results, mape, mase = m4_summary.evaluate()
                print('smape:', smape_results)
                print('mape:', mape)
                print('mase:', mase)
                print('owa:', owa_results)
            except Exception as exc:
                print(f'M4 summary skipped: {exc}')
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
        return
