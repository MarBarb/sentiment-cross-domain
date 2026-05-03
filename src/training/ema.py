"""EMA 特征统计 — 解决小目标域 batch 统计噪声"""
import torch


class EMAMeter:
    """指数移动平均特征统计

    N_t 太小导致 batch 统计噪声大, EMA 平滑 μ_t / σ_t 是稳定训练的关键。
    """

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.mu: torch.Tensor = None
        self.var: torch.Tensor = None

    def update(self, feat: torch.Tensor):
        """更新 EMA 统计

        Args:
            feat: (batch, d) 特征张量
        """
        with torch.no_grad():
            batch_mu = feat.mean(0)
            batch_var = feat.var(0) + 1e-8

            if self.mu is None:
                self.mu = batch_mu.clone()
                self.var = batch_var.clone()
            else:
                self.mu = self.decay * self.mu + (1 - self.decay) * batch_mu
                self.var = self.decay * self.var + (1 - self.decay) * batch_var

    def reset(self):
        """重置统计"""
        self.mu = None
        self.var = None
