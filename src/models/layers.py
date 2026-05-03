"""自定义层: Adapter, 梯度反转层"""
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """梯度反转层 (DANN)"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class AdapterLayer(nn.Module):
    """轻量适配器层 (bottleneck 结构)

    用于参数高效微调, 仅训练适配器参数, 避免过拟合小目标域。
    """

    def __init__(self, d: int, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(d, bottleneck)
        self.up = nn.Linear(bottleneck, d)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # 初始化: 接近恒等映射
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up(x)
        return residual + x  # 残差连接
