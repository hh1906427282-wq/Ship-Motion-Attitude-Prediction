import torch
import torch.nn as nn
from model.cnn_bilstm import CNNModule, MIBiLSTMModule
from model.transformer import TransformerBranch
from model.fusion import FusionGate
from model.multi_head import MultiTaskHead  # 多任务输出头

class Predictor(nn.Module):
    def __init__(self,
                 input_dim=18,
                 cnn_output_dim=64,
                 bilstm_hidden=128,
                 transformer_dim=128,
                 pred_len=100):
        super(Predictor, self).__init__()

        # 分支1：CNN + MI-BiLSTM
        self.cnn = CNNModule(input_dim=input_dim, output_dim=cnn_output_dim)
        self.bilstm = MIBiLSTMModule(input_dim=cnn_output_dim * 3, hidden_dim=bilstm_hidden)

        # 分支2：Transformer
        self.transformer = TransformerBranch(input_dim=input_dim, d_model=transformer_dim)

        # 映射 BiLSTM 输出 → transformer_dim
        self.bilstm_proj = nn.Linear(bilstm_hidden * 2, transformer_dim)

        # 动态融合
        self.fusion = FusionGate(input_dim=transformer_dim)

        # 多任务输出模块
        self.head = MultiTaskHead(input_dim=transformer_dim, pred_len=pred_len)

        self.pred_len = pred_len

    def forward(self, x):
        """
        输入: x ∈ [B, T=300, 18]
        输出:
            - y_pred ∈ [B, 100, 18]
            - attn_weights ∈ [B, T=300, D=128]（融合时对 BiLSTM 的注意力）
        """

        # CNN → BiLSTM
        x_cnn = self.cnn(x)                         # [B, 300, 192]
        x_bilstm = self.bilstm(x_cnn)               # [B, 300, 256]
        x_bilstm_proj = self.bilstm_proj(x_bilstm)  # [B, 300, 128]

        # Transformer
        x_transformer = self.transformer(x)         # [B, 300, 128]

        # 融合 + 注意力权重
        x_fused, attn_weights = self.fusion(x_bilstm_proj, x_transformer, return_weights=True)

        # 多任务预测输出
        out = self.head(x_fused)  # [B, 100, 18]

        return out, attn_weights
