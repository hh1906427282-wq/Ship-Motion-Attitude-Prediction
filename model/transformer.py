import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerBranch(nn.Module):
    def __init__(self, input_dim=18, d_model=128, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerBranch, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

        # ✅ 加池化层（对齐 CNN 分支）
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        """
        输入: x ∈ [B, T=300, D=18]
        输出: [B, T=150, d_model] ← 与 CNN/BiLSTM 分支一致
        """
        x = self.input_proj(x)           # → [B, T, d_model]
        x = self.pos_encoder(x)          # 加位置编码
        x = self.transformer_encoder(x)  # → [B, T, d_model]
        x = self.dropout(x)

        # ✅ 统一时间维度：从 T=300 → T=150
        x = x.permute(0, 2, 1)           # → [B, d_model, T]
        x = self.pool(x)                 # → [B, d_model, T/2]
        x = x.permute(0, 2, 1)           # → [B, T/2, d_model]

        return x
