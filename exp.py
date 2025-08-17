import os
import sys
import time

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from model import HRS
from utils.metrics import MSE, MAE, metric, scheduling_aware_loss
from utils.tools import EarlyStopping, adjust_learning_rate, visual


class Exp_Forecast(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(
                self.args.gpu if not self.args.use_multi_gpu else self.args.devices)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model_dict = {
            'HRS': HRS,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, val_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = optim.Adam(list(self.model.parameters()), lr=self.args.learning_rate)
        
        # scheduling aware loss or mse loss
        criterion = scheduling_aware_loss if self.args.scheduling_aware else nn.MSELoss()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim, steps_per_epoch=train_steps, epochs=self.args.train_epochs, max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            time_now = time.time()
            for i, (seq_x, seq_y, fig_x, seq_x_mark, seq_y_mark) in enumerate(train_loader):
                iter_count += 1
                seq_x = seq_x.float().to(self.device)
                seq_x_mark = seq_x_mark.float().to(self.device)
                seq_y = seq_y.float().to(self.device)
                model_optim.zero_grad()
                if self.args.model_type == 0:
                    fig_x = fig_x.to(self.device)
                    y_pred = self.model(seq_x, fig_x, seq_x_mark)
                else:
                    seq_x = seq_x.float().to(self.device)
                    seq_x_mark = seq_x_mark.float().to(self.device)
                    seq_y_mark = seq_y_mark.float().to(self.device)
                    dec_inp = torch.zeros_like(seq_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([seq_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    y_pred = self.model(seq_x, seq_x_mark, dec_inp, seq_y_mark)
 
                
                loss = criterion(y_pred, seq_y)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

                if self.args.lradj == 'optim':
                    adjust_learning_rate(model_optim, epoch, self.args, scheduler, printnot=False)
                    scheduler.step()

            cost_time = time.time() - epoch_time
            print("Epoch: {} cost time: {}  speed: {:.4f}s/iter".format(epoch + 1, cost_time, cost_time / train_steps))
            train_loss = np.mean(train_loss)
            vali_loss = self.vali(vali_data, val_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            sys.stdout.flush()
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'optim':
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, val_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (seq_x, seq_y, fig_x, seq_x_mark, seq_y_mark) in enumerate(val_loader):
                seq_x = seq_x.float().to(self.device)
                seq_x_mark = seq_x_mark.float().to(self.device)
                seq_y = seq_y.float().to(self.device)
                if self.args.model_type == 0:
                    fig_x = fig_x.to(self.device)
                    y_pred = self.model(seq_x, fig_x, seq_x_mark)
                else:
                    seq_x = seq_x.float().to(self.device)
                    seq_x_mark = seq_x_mark.float().to(self.device)
                    seq_y_mark = seq_y_mark.float().to(self.device)
                    dec_inp = torch.zeros_like(seq_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([seq_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    y_pred = self.model(seq_x, seq_x_mark, dec_inp, seq_y_mark)
                loss = criterion(y_pred, seq_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
        print(f"os.environ['CUDA_VISIBLE_DEVICES'] = {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
        if test:
            print('loading model')
            state_dict = torch.load(os.path.join(self.args.checkpoints, setting) + '/' + 'checkpoint.pth', map_location=self.device)
            self.model.load_state_dict(state_dict)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if self.args.draw_test and not os.path.exists(folder_path):
            os.makedirs(folder_path)

        times = []
        memory_usage = []
        self.model.eval()
        with torch.no_grad():
            for i, (seq_x, seq_y, fig_x, seq_x_mark, seq_y_mark) in enumerate(test_loader):                   
                start_time = time.time()                
                seq_x = seq_x.float().to(self.device)
                seq_x_mark = seq_x_mark.float().to(self.device)
                seq_y = seq_y.float().to(self.device)
                if self.args.model_type == 0:
                    fig_x = fig_x.to(self.device)
                    y_pred = self.model(seq_x, fig_x, seq_x_mark)
                else:
                    seq_x = seq_x.float().to(self.device)
                    seq_x_mark = seq_x_mark.float().to(self.device)
                    seq_y_mark = seq_y_mark.float().to(self.device)
                    dec_inp = torch.zeros_like(seq_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([seq_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    y_pred = self.model(seq_x, seq_x_mark, dec_inp, seq_y_mark)
                
                times.append(time.time() - start_time)
                
                outputs = y_pred.cpu().detach().numpy()
                batch_y = seq_y.cpu().detach().numpy()
                seq_x = seq_x.cpu().detach().numpy()

                if self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                    seq_x = test_data.inverse_transform(seq_x.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y
                preds.append(pred[0])
                trues.append(true[0])
    
        trues = np.array(trues)
        preds = np.array(preds)
        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        apl = metric(preds, trues)
            
        print('apl:{}'.format(apl))

        return
