# -----------------------------------------------------------------------------
# This file is adapted from PatchTST (https://github.com/PatchTST/PatchTST)
# Original project: “PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.”
# Original repository license: Apache-2.0
# Copyright (c) Nie et al.
#
# Modifications made by Yuan Feng et al., 2025:
# Simplify the code workflow.
# Remove the encoding of data stamp.
# -----------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class Dataset_ETTh(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                    data_path='ETTh1.csv', target='OT', scale=True):
        super(Dataset_ETTh, self).__init__()
        
        if size == None:
            self.seq_len = 336
            self.label_len = 0
            self.pred_len = 720
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        self.target = target
        
        self.__read_data__()
        
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                            self.data_path))
        
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
    def __getitem__(self, index):
        
        s_begin = index
        s_end = s_begin + self.seq_len
        
        r_begin = s_begin
        r_end = s_end + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.empty(1)
        seq_y_mark = torch.empty(1)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        
        return self.scaler.inverse_transform(data)
    
class Dataset_ETTm(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                    data_path='ETTm1.csv', target='OT', scale=True):
        super(Dataset_ETTm, self).__init__()
        
        if size == None:
            self.seq_len = 336
            self.label_len = 0
            self.pred_len = 720
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        self.target = target
        
        self.__read_data__()
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                            self.data_path))
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
    
    def __getitem__(self, index):
        
        s_begin = index
        s_end = s_begin + self.seq_len
        
        r_begin = s_begin
        r_end = s_end + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.empty(1)
        seq_y_mark = torch.empty(1)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        
        return self.scaler.inverse_transform(data)
    
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                    data_path='weather.csv', target='OT', scale=True):
        super(Dataset_Custom, self).__init__()
        
        if size == None:
            self.seq_len = 336
            self.label_len = 0
            self.pred_len = 720
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        self.target = target
        
        self.__read_data__()
        
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                            self.data_path))
        
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
    def __getitem__(self, index):
        
        s_begin = index
        s_end = s_begin + self.seq_len
        
        r_begin = s_begin
        r_end = s_end + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.empty(1)
        seq_y_mark = torch.empty(1)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        
        return self.scaler.inverse_transform(data)
