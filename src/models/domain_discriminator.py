"""域判别器 (DANN)"""
import torch
import torch.nn as nn
from .layers import GradientReversalLayer


class DomainDiscriminator(nn.Module):
    """DANN 风格域判别器

    通过梯度反转层学习域不变特征:
    - 前向: 判断样本来自源域还是目标域
    - 反向: 梯度反转, 迫使 encoder 学习域不变表示
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 256, alpha: float = 1.0):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=alpha)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (batch, feat_dim) 特征向量
        Returns:
            domain logits (batch, 1)
        """
        feat = self.grl(feat)
        return self.net(feat)
