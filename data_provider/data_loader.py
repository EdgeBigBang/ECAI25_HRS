import os
from collections import defaultdict
import concurrent.futures
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset


class Dataset_Basic(Dataset):
    def __init__(self, args, flag):
        self.args = args
        self.normalize = False  
        self.dir_path = self.args.dir_path
        self.data_path = self.args.data_path
        self.seq_len = self.args.seq_len
        self.label_len = self.args.label_len
        self.pred_len = self.args.pred_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = self.args.target
        self.features = self.args.features
        assert self.features in ['S', 'MS', 'M']

        self.h = self.args.h
        self.channel = self.args.channel
        assert self.channel in [1, 3]
        self.bc = self.args.bc
        self.lw = self.args.lw
        self.lc = self.args.lc
        self.expand = self.args.expand
        self.model_type = self.args.model_type
        self.divide = 12

        cache_path = os.path.join(
            self.dir_path, 
            f"{self.data_path}_cached_{self.set_type}_seq{self.seq_len}_pred{self.pred_len}.pkl"
        )
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
                self.data = cache['data']
                self.scaler = cache['scaler']
            print(f"Loaded cached data and scaler from {cache_path}")
            return

        if 'ECW' in self.data_path:
            self.__read_data_ECW__()
        else:
            self.__read_data__()

    def data2Pixel(self, dataXIn, draw_type='opencv'):
        assert draw_type in ['matplotlib', 'opencv', 'sampling']
        dataX = np.copy(dataXIn.T)
        feature = dataX.shape[0]
        lenX = dataX.shape[1]

        imgX = np.zeros([feature * self.channel, lenX * self.expand, self.h * self.expand], dtype=np.float32)
        for i in range(feature):
            if draw_type == 'matplotlib':
                canvas = FigureCanvasAgg(plt.figure(figsize=(lenX / 100 * self.expand, self.h / 100 * self.expand), facecolor=self.bc))
                plt.plot(dataX[i], linewidth=self.lw, color=self.lc)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['bottom'].set_visible(False)
                plt.gca().spines['left'].set_visible(False)
                plt.axis('off')
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                plt.margins(0, 0)
                canvas.draw()
                buf = canvas.buffer_rgba()
                if self.channel == 1:
                    img = np.dot(np.asarray(buf)[:, :, :3] / 255, [0.2989, 0.5870, 0.1140])
                    imgX[i, :img.shape[1], :] = img.T
                else:  # self.channel == 3:
                    img = np.asarray(buf)[:, :, :3] / 255
                    imgX[i * self.channel:(i + 1) * self.channel, :img.shape[1], :] = np.transpose(img, (2, 1, 0))
                plt.close()
            else:
                img = (np.ones((self.h * self.expand, lenX * self.expand, 3), dtype=np.uint8) * (int(self.bc[0] * 255), int(self.bc[1] * 255), int(self.bc[2] * 255))).astype(np.uint8)
                diff = np.max(dataX[i]) - np.min(dataX[i])
                if diff == 0:
                    data_line = np.ones_like(dataX[i])  
                else:
                    data_line = 1 - (dataX[i] - np.min(dataX[i])) / diff
                if draw_type == 'opencv':
                    for j in range(lenX - 1):
                        pt1 = (int(j * self.expand), round(data_line[j] * (self.h * self.expand - 1)))
                        pt2 = (int((j + 1) * self.expand), round(data_line[j + 1] * (self.h * self.expand - 1)))
                        cv2.line(img, pt1, pt2, (int(self.lc[0] * 255), int(self.lc[1] * 255), int(self.lc[2] * 255)), self.lw if type(self.lw) == int else 1)
                else:  # if draw_type == 'sampling'  self.lw is not used
                    data_line = np.round(data_line * (self.h - 1)).astype(int)
                    for j in range(self.expand):
                        img[np.repeat(data_line, self.expand) * self.expand + j,
                        np.arange(len(data_line) * self.expand), :] = (int(self.lc[0] * 255), int(self.lc[1] * 255), int(self.lc[2] * 255))
                if self.channel == 1:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    imgX[i, :gray_img.shape[1], :] = np.transpose(np.expand_dims(gray_img, axis=0) / 255, (0, 2, 1))
                else:  # self.channel == 3:
                    imgX[i * self.channel:(i + 1) * self.channel, :, :] = np.transpose(img / 255, (2, 1, 0))
        return np.transpose(imgX, (0, 2, 1))

    def __read_data__(self):
        df_raw = pd.read_csv(str(os.path.join(self.dir_path, self.data_path)))
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)  # long-term TSF
        num_test = int(len(df_raw) * 0.2) 
        num_vali = len(df_raw) - num_train - num_test
        print(num_train, num_test, num_vali)
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        else:  # self.features == 'S'
            df_data = df_raw[[self.target]]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler = StandardScaler()
        self.scaler.fit(train_data.values.reshape(-1, 1))
        shape = df_data.values.shape
        data = self.scaler.transform(df_data.values.reshape(-1, 1)).reshape(shape)
        ####
        
        data = data[border1:border2]
        self.data = defaultdict(list)
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp['month'] = df_stamp.date.astype(object).apply(lambda row: row.month)
        df_stamp['day'] = df_stamp.date.astype(object).apply(lambda row: row.day)
        df_stamp['weekday'] = df_stamp.date.astype(object).apply(lambda row: row.weekday())
        df_stamp['hour'] = df_stamp.date.astype(object).apply(lambda row: row.hour)
        df_stamp['minute'] = df_stamp.date.astype(object).apply(lambda row: row.minute)
        df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 5)
        data_stamp = df_stamp.drop(columns=['date']).values
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            x_data = []
            y_data = []
            for i in range(len(data) - self.seq_len - self.pred_len + 1):
                data_x = data[i:i + self.seq_len]
                self.data['x'].append(data_x)
                self.data['y'].append(data[i + self.seq_len:i + self.seq_len + self.pred_len])
                if self.model_type == 0:
                    self.data['x_mark'].append(data_stamp[i:i + self.seq_len])
                    self.data['y_mark'].append(data_stamp[i + self.seq_len - self.label_len:i + self.seq_len + self.pred_len])
                    futures.append(executor.submit(self.data2Pixel, data_x))
                else:
                    self.data['x_mark'].append(data_stamp[i:i + self.seq_len])
                    self.data['y_mark'].append(data_stamp[i + self.seq_len - self.label_len:i + self.seq_len + self.pred_len])
            for future in futures:
                self.data['fig'].append(future.result())

        save_path = os.path.join(self.dir_path, f"{self.data_path}_cached_{self.set_type}_seq{self.seq_len}_pred{self.pred_len}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({'data': self.data, 'scaler': self.scaler}, f)

    def __read_data_ECW__(self):
        df_raw = pd.read_csv(str(os.path.join(self.dir_path, self.data_path)))
        num_train = int(len(df_raw) * 0.6)  # short-term TSF
        num_test = int(len(df_raw) * 0.2) 
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler = MinMaxScaler()
        self.scaler.fit(train_data.values.reshape(-1, 1))
        shape = df_data.values.shape
        data = self.scaler.transform(df_data.values.reshape(-1, 1)).reshape(shape)

        data = data[border1:border2]
        self.data = defaultdict(list)
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp['month'] = df_stamp.date.astype(object).apply(lambda row: row.month)
        df_stamp['day'] = df_stamp.date.astype(object).apply(lambda row: row.day)
        df_stamp['weekday'] = df_stamp.date.astype(object).apply(lambda row: row.weekday())
        df_stamp['hour'] = df_stamp.date.astype(object).apply(lambda row: row.hour)
        data_stamp = df_stamp.drop(columns=['date']).values
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for device in range(data.shape[1]):
                for i in range(0, len(data) - self.seq_len - self.pred_len + 1, self.divide):
                    data_x = data[i:i + self.seq_len, device].reshape(-1, 1)
                    self.data['x'].append(data_x)
                    self.data['y'].append(data[i + self.seq_len:i + self.seq_len + self.pred_len, device].reshape(-1, 1))
                    if self.model_type == 0:
                        futures.append(executor.submit(self.data2Pixel, data_x))
                        self.data['x_mark'].append(data_stamp[i:i + self.seq_len])
                        self.data['y_mark'].append(data_stamp[i + self.seq_len - self.label_len:i + self.seq_len + self.pred_len])
                    else:
                        self.data['x_mark'].append(data_stamp[i:i + self.seq_len])
                        self.data['y_mark'].append(data_stamp[i + self.seq_len - self.label_len:i + self.seq_len + self.pred_len])
            for future in futures:
                self.data['fig'].append(future.result())
        
        save_path = os.path.join(self.dir_path, f"{self.data_path}_cached_{self.set_type}_seq{self.seq_len}_pred{self.pred_len}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({'data': self.data, 'scaler': self.scaler}, f)

    def __getitem__(self, index):
        if self.model_type == 0:
            return self.data['x'][index], self.data['y'][index], self.data['fig'][index], self.data['x_mark'][index], \
                   self.data['y_mark'][index]
        else:
            return self.data['x'][index], self.data['y'][index], 0, self.data['x_mark'][index], self.data['y_mark'][
                index]

    def __len__(self):
        return len(self.data['x'])

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
