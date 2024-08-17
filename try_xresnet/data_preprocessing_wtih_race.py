import pandas as pd
import numpy as np
import wfdb
import ast
import os
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset
import random
import sys
sys.path.append(r"D:\pycharm_software\PyCharm 2024.1\codes\race")
from try_xresnet.helper_code import *



#%%
import re
data_folder = r"D:\pycharm_software\PyCharm 2024.1\codes\race\python-example-2024-main\ptb-xl/records100"
records = find_records(data_folder)
print(records)
num_records = len(records)
labels_all = []
data_all = []
for i in range(num_records):
    width = len(str(num_records))
    #print(f'- {i + 1:>{width}}/{num_records}: {records[i]}...')

    data_record = os.path.join(data_folder, records[i])
    data_header = load_header(data_record)
    comments = [l for l in data_header.split('\n') if l.startswith('#')]
    #print(comments)

    # 初始化标签变量
    labels = None

    # 遍历列表，查找包含标签的行
    for line in comments:
        label_match = re.match(r'# Labels: (.+)', line)
        if label_match:
            labels = label_match.group(1).split(', ')
            break

    if labels:
        #print("Labels:", labels)
        labels_all.append(labels)
        data = [wfdb.rdsamp(r"race\python-example-2024-main\ptb-xl\records100/"+records[i])]
        data = np.array([signal for signal, meta in data])
        data = np.transpose(data, (0, 2, 1))  # 转换为 (samples, channels, length)
        data_all.append(data)
    else:
        print(comments)
        print("Labels not found")
#print(labels_all)
print(len(labels_all))
print(len(data_all))
print(np.array(data_all).shape)
#%%
Y,classes = encode_labels(labels_all)
print(Y)

#%%
class PTBXLDataLoader:
    def __init__(self, path, sampling_rate):
        self.path = path
        self.sampling_rate = sampling_rate

    def load_data(self):
        Y = pd.read_csv(self.path + 'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        X = self.load_raw_data(Y)
        agg_df = self.load_agg_df()
        Y['diagnostic_superclass'] = Y.scp_codes.apply(lambda x: self.aggregate_diagnostic(x, agg_df))
        return X, Y

    def load_raw_data(self, df):
        if self.sampling_rate == 100:
            data = [wfdb.rdsamp(self.path + f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(self.path + f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        data = np.transpose(data, (0, 2, 1))  # 转换为 (samples, channels, length)
        return data

    def load_agg_df(self):
        agg_df = pd.read_csv(self.path + 'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        return agg_df

    def aggregate_diagnostic(self, y_dic, agg_df):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def standardize_signal_data(self, signal_data):
        """
        对形状为 (batch, channel, length) 的信号数据进行标准化。

        参数:
        signal_data (numpy.ndarray): 形状为 (batch, channel, length) 的信号数据。

        返回:
        numpy.ndarray: 标准化后的信号数据，形状与输入相同。
        list: 每个通道的 StandardScaler 实例。
        """
        # 获取数据的形状
        batch_size, num_channels, signal_length = signal_data.shape

        # 初始化标准化器列表
        scalers = [StandardScaler() for _ in range(num_channels)]

        # 创建一个新的数组存储标准化后的数据
        standardized_data = np.empty_like(signal_data)

        # 对每个通道独立进行标准化
        for i in range(num_channels):
            # 获取当前通道的数据，形状为 (batch, length)
            channel_data = signal_data[:, i, :]

            # 进行标准化，注意 reshape(-1, 1) 将数据转换为二维数组以适配 StandardScaler
            reshaped_data = channel_data.reshape(-1, 1)
            scaled_data = scalers[i].fit_transform(reshaped_data)

            # 将标准化后的数据还原为 (batch, length) 形状
            standardized_data[:, i, :] = scaled_data.reshape(channel_data.shape)

        return standardized_data
    def split_data(self, X, Y, test_fold=10):
        X_train = X[np.where(Y.strat_fold < 9)]
        y_train = Y[(Y.strat_fold < 9)].diagnostic_superclass
        X_val = X[np.where(Y.strat_fold == 9)]
        y_val = Y[(Y.strat_fold == 9)].diagnostic_superclass
        X_test = X[np.where(Y.strat_fold == test_fold)]
        y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
        return X_train, y_train, X_val, y_val, X_test, y_test

    def encode_labels(self, y_train, y_val, y_test):
        mlb = MultiLabelBinarizer()
        y_train_one_hot_labels = mlb.fit_transform(y_train)
        y_val_one_hot_labels = mlb.transform(y_val)
        y_test_one_hot_labels = mlb.transform(y_test)
        return y_train_one_hot_labels, y_val_one_hot_labels, y_test_one_hot_labels, mlb.classes_


class Dataset(Dataset):

    # Random Crop Signal to Segment with w_size seconds length
    def __init__(self, data, labels, window_size ):
        self.data = data
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the full record and its corresponding label
        record = self.data[index]
        label = self.labels[index]

        # Randomly select a segment of fixed length from the full record
        start = random.randint(0, record.shape[-1] - self.window_size)
        segment = record[:, start: start + self.window_size]

        return segment, label


class StepWiseTestset(Dataset):

    # Testset for Element-Wise Maximum
    def __init__(self, data, labels, window_size, stride):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the full record and its corresponding label
        record = self.data[index]
        label = self.labels[index]

        # Create segments with overlapping windows
        segments = []
        start = 0
        while start + self.window_size <= record.shape[-1]:
            segment = record[:, start: start + self.window_size]
            segments.append(segment)
            start += self.stride

        return segments, label

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, w_size):
    train_dataset = Dataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32),int(w_size * 100))
    val_dataset = Dataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32), int(w_size * 100))
    test_dataset = StepWiseTestset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32), int(w_size * 100), int(.5 * 100))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def encode_labels(Y):
    mlb = MultiLabelBinarizer()
    Y_one_hot_labels = mlb.fit_transform(Y)
    return  Y_one_hot_labels, mlb.classes_


