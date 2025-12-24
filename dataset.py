import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset
import os

class ShipMotionDataset(Dataset):
    def __init__(self, data, input_len, pred_len):
        self.data = data  # shape: [T, D]
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_len]
        y = self.data[idx + self.input_len : idx + self.input_len + self.pred_len, :18]  # ✅ 输出全部18维
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_data(csv_path='data/五级海况.csv', input_len=300, pred_len=100, normalize=True):
    # 读取原始 CSV
    df = pd.read_csv(csv_path)

    # 特征列选择（18维）
    feature_cols = [
        'x', 'y', 'z', 'rx', 'ry', 'rz',
        'xv', 'yv', 'zv', 'rxv', 'ryv', 'rzv',
        'xa', 'ya', 'za', 'rxa', 'rya', 'rza'
    ]

    # 检查列是否存在
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}")

    # 转为矩阵
    data_raw = df[feature_cols].values  # shape: [T, D]
    total_len = len(data_raw)

    # 标准化
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_raw) if normalize else data_raw

    # 明确测试区间为最后 20000 帧（约2000秒），训练用前面的
    test_len = 20000
    train_len = total_len - test_len
    test_start = train_len - input_len - pred_len  # 留出滑窗缓冲

    train_data = data_scaled[:train_len]
    test_data = data_scaled[test_start:]

    # 显式保存为 CSV
    os.makedirs('data', exist_ok=True)
    pd.DataFrame(train_data, columns=feature_cols).to_csv('data/train.csv', index=False)
    pd.DataFrame(test_data, columns=feature_cols).to_csv('data/test.csv', index=False)

    # 构造 PyTorch 数据集
    train_dataset = ShipMotionDataset(train_data, input_len, pred_len)
    test_dataset = ShipMotionDataset(test_data, input_len, pred_len)

    return train_dataset, test_dataset, scaler
