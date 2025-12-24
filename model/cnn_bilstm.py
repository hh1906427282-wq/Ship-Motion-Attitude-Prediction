import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModule(nn.Module):
    def __init__(self, input_dim=18, output_dim=64):
        super(CNNModule, self).__init__()

        self.conv_3 = nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(input_dim, output_dim, kernel_size=5, padding=2)
        self.conv_7 = nn.Conv1d(input_dim, output_dim, kernel_size=7, padding=3)

        self.shortcut = nn.Conv1d(input_dim, output_dim * 3, kernel_size=1)

        self.bn = nn.BatchNorm1d(output_dim * 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)  # 600 → 300
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        x: [B, T=600, D=18]
        return: [B, 300, output_dim*3]
        """
        x = x.permute(0, 2, 1)  # [B, D, T]
        out_3 = self.conv_3(x)
        out_5 = self.conv_5(x)
        out_7 = self.conv_7(x)

        out = torch.cat([out_3, out_5, out_7], dim=1)
        residual = self.shortcut(x)
        out = out + residual

        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)  # [B, D_out, 300]
        out = self.dropout(out)
        out = out.permute(0, 2, 1)  # [B, 300, D_out]

        return out


class MIBiLSTMModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, bidirectional=True, dropout=0.3):
        super(MIBiLSTMModule, self).__init__()

        self.input_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x):
        """
        x: [B, T=300, D]
        return: [B, T, hidden_dim * num_directions]
        """
        gate = self.input_attention(x)
        gated_x = x * gate
        out, _ = self.lstm(gated_x)
        return out
