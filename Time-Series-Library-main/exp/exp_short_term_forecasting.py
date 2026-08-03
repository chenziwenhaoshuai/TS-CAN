from data_provider.data_factory import data_provider
from collections import OrderedDict

from data_provider.m4 import M4Dataset, M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas

warnings.filterwarnings('ignore')


def _m4_group_values(values, groups, group_name):
    indices = np.where(groups == group_name)[0]
    grouped = []
    for index in indices:
        value = np.asarray(values[index], dtype=np.float32)
        grouped.append(value[~np.isnan(value)])
    if not grouped:
        return np.array([])
    lengths = {len(v) for v in grouped}
    if len(lengths) == 1:
        return np.stack(grouped)
    return grouped


def _m4_mase(forecast, insample, outsample, frequency):
    return np.mean(np.abs(forecast - outsample)) / np.mean(np.abs(insample[:-frequency] - insample[frequency:]))


def _m4_smape_2(forecast, target):
    denom = np.abs(target) + np.abs(forecast)
    denom[denom == 0.0] = 1.0
    return 200 * np.abs(forecast - target) / denom


def _m4_mape(forecast, target):
    denom = np.abs(target)
    denom[denom == 0.0] = 1.0
    return 100 * np.abs(forecast - target) / denom


def _m4_summarize_groups(scores, groups):
    scores_summary = OrderedDict()

    def group_count(group_name):
        return len(np.where(groups == group_name)[0])

    if set(scores) != set(M4Meta.seasonal_patterns):
        return OrderedDict((key, scores[key]) for key in scores)

    weighted_score = {}
    for group_name in ['Yearly', 'Quarterly', 'Monthly']:
        weighted_score[group_name] = scores[group_name] * group_count(group_name)
        scores_summary[group_name] = scores[group_name]

    others_score = 0
    others_count = 0
    for group_name in ['Weekly', 'Daily', 'Hourly']:
        others_score += scores[group_name] * group_count(group_name)
        others_count += group_count(group_name)
    weighted_score['Others'] = others_score
    scores_summary['Others'] = others_score / others_count

    average = sum(weighted_score.values()) / len(groups)
    scores_summary['Average'] = average
    return scores_summary


def _evaluate_m4_summary(file_path, root_path):
    training_set = M4Dataset.load(training=True, dataset_file=root_path)
    test_set = M4Dataset.load(training=False, dataset_file=root_path)
    naive_path = os.path.join(root_path, 'submission-Naive2.csv')
    naive2_values = pandas.read_csv(naive_path).values[:, 1:].astype(np.float32)
    naive2_forecasts = [v[~np.isnan(v)] for v in naive2_values]

    model_mases = {}
    naive2_smapes = {}
    naive2_mases = {}
    grouped_smapes = {}
    grouped_mapes = {}
    for group_name in M4Meta.seasonal_patterns:
        forecast_path = file_path + group_name + '_forecast.csv'
        if not os.path.exists(forecast_path):
            continue
        model_forecast = pandas.read_csv(forecast_path).values[:, 1:].astype(np.float32)

        naive2_forecast = _m4_group_values(naive2_forecasts, test_set.groups, group_name)
        target = _m4_group_values(test_set.values, test_set.groups, group_name)
        frequency = training_set.frequencies[test_set.groups == group_name][0]
        insample = _m4_group_values(training_set.values, test_set.groups, group_name)

        model_mases[group_name] = np.mean([
            _m4_mase(forecast=model_forecast[i], insample=insample[i], outsample=target[i], frequency=frequency)
            for i in range(len(model_forecast))
        ])
        naive2_mases[group_name] = np.mean([
            _m4_mase(forecast=naive2_forecast[i], insample=insample[i], outsample=target[i], frequency=frequency)
            for i in range(len(model_forecast))
        ])
        naive2_smapes[group_name] = np.mean(_m4_smape_2(naive2_forecast, target))
        grouped_smapes[group_name] = np.mean(_m4_smape_2(forecast=model_forecast, target=target))
        grouped_mapes[group_name] = np.mean(_m4_mape(forecast=model_forecast, target=target))

    grouped_smapes = _m4_summarize_groups(grouped_smapes, test_set.groups)
    grouped_mapes = _m4_summarize_groups(grouped_mapes, test_set.groups)
    grouped_model_mases = _m4_summarize_groups(model_mases, test_set.groups)
    grouped_naive2_smapes = _m4_summarize_groups(naive2_smapes, test_set.groups)
    grouped_naive2_mases = _m4_summarize_groups(naive2_mases, test_set.groups)
    grouped_owa = OrderedDict()
    for key in grouped_model_mases.keys():
        grouped_owa[key] = (
            grouped_model_mases[key] / grouped_naive2_mases[key]
            + grouped_smapes[key] / grouped_naive2_smapes[key]
        ) / 2

    def round_all(scores):
        return dict((key, np.round(value, 3)) for key, value in scores.items())

    return round_all(grouped_smapes), round_all(grouped_owa), round_all(grouped_mapes), round_all(grouped_model_mases)


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
        elif loss_name == 'OWA':
            smape = smape_loss()
            mase = mase_loss()

            def criterion(insample, freq, forecast, target, mask):
                return (
                    0.045406786 * smape(insample, freq, forecast, target, mask)
                    + 0.364595642 * mase(insample, freq, forecast, target, mask)
                )

            return criterion

    def _parse_loss_schedule(self):
        schedule_text = str(getattr(self.args, 'loss_schedule', '') or '').strip()
        if not schedule_text:
            return []
        schedule = []
        for chunk in schedule_text.split(','):
            name, count = chunk.split(':', 1)
            schedule.append((name.strip().upper(), int(count)))
        return schedule

    @staticmethod
    def _loss_name_for_epoch(schedule, epoch_index):
        cursor = 0
        for name, count in schedule:
            cursor += count
            if epoch_index < cursor:
                return name
        return schedule[-1][0]

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        loss_schedule = self._parse_loss_schedule()
        if loss_schedule:
            criteria = {
                name: self._select_criterion(name)
                for name in sorted({name for name, _ in loss_schedule})
            }
            criterion = criteria[self._loss_name_for_epoch(loss_schedule, 0)]
        else:
            criteria = {}
            criterion = self._select_criterion(self.args.loss)
        mse = nn.MSELoss()

        for epoch in range(self.args.train_epochs):
            if loss_schedule:
                active_loss = self._loss_name_for_epoch(loss_schedule, epoch)
                criterion = criteria[active_loss]
            else:
                active_loss = self.args.loss
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
            test_loss = vali_loss
            print("Epoch: {0}, Steps: {1} | Loss: {2} Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, active_loss, train_loss, vali_loss, test_loss))
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
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        print(self.args.model)
        file_path = './m4_results/' + self.args.model + '/'
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            smape_results, owa_results, mape, mase = _evaluate_m4_summary(file_path, self.args.root_path)
            print('smape:', smape_results)
            print('mape:', mape)
            print('mase:', mase)
            print('owa:', owa_results)
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
        return
