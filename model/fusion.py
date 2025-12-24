import torch
import torch.nn as nn

class FusionGate(nn.Module):
    def __init__(self, input_dim):
        super(FusionGate, self).__init__()
        self.linear = nn.Linear(input_dim * 2, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, return_weights=False):
        """
        x1, x2: [B, T, D]
        return:
            - out: [B, T, D] 融合结果
            - gate: [B, T, D] 注意力权重（若启用 return_weights）
        """
        fusion_input = torch.cat([x1, x2], dim=-1)        # [B, T, 2D]
        gate = self.sigmoid(self.linear(fusion_input))    # [B, T, D]
        out = gate * x1 + (1 - gate) * x2                 # 加权融合

        if return_weights:
            return out, gate  # 返回融合结果和注意力权重
        else:
            return out        # 仅返回融合结果
