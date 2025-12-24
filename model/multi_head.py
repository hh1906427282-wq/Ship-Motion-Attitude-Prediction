# model/multi_head.py

import torch
import torch.nn as nn

class MultiTaskHead(nn.Module):
    def __init__(self, input_dim, pred_len):
        super(MultiTaskHead, self).__init__()
        self.pred_len = pred_len

        # 每个子任务的输出为 [B, T, pred_len * 6]
        self.head_x = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len * 6)
        )

        self.head_v = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len * 6)
        )

        self.head_a = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len * 6)
        )

    def forward(self, x):
        """
        x: [B, T, D]
        返回: [B, pred_len, 18]（每帧的 x/v/a 合并）
        """
        out_x = self.head_x(x)  # [B, T, pred_len * 6]
        out_v = self.head_v(x)
        out_a = self.head_a(x)

        # 取最后一帧输出 → [B, pred_len * 6]
        out_x = out_x[:, -1, :].view(-1, self.pred_len, 6)
        out_v = out_v[:, -1, :].view(-1, self.pred_len, 6)
        out_a = out_a[:, -1, :].view(-1, self.pred_len, 6)

        # 拼接为 [B, pred_len, 18]
        out = torch.cat([out_x, out_v, out_a], dim=-1)
        return out
