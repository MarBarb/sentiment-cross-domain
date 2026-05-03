"""训练模块测试"""
import pytest
import torch
from src.training.kl_loss import gaussian_kl
from src.training.ema import EMAMeter


class TestGaussianKL:
    """闭式高斯 KL 散度测试"""

    def test_kl_zero_same_distribution(self):
        """相同分布 KL == 0"""
        mu = torch.randn(768)
        var = torch.rand(768) + 0.1
        kl = gaussian_kl(mu, var, mu, var)
        assert abs(kl.item()) < 1e-5

    def test_kl_positive_different(self):
        """不同分布 KL > 0"""
        mu_s = torch.zeros(768)
        var_s = torch.ones(768)
        mu_t = torch.ones(768)
        var_t = torch.ones(768) * 2
        kl = gaussian_kl(mu_s, var_s, mu_t, var_t)
        assert kl.item() > 0

    def test_kl_symmetry(self):
        """KL 不对称: KL(P‖Q) ≠ KL(Q‖P) 一般情况下"""
        mu_s = torch.zeros(10)
        var_s = torch.ones(10)
        mu_t = torch.ones(10) * 2
        var_t = torch.ones(10) * 0.5
        kl_forward = gaussian_kl(mu_s, var_s, mu_t, var_t)
        kl_backward = gaussian_kl(mu_t, var_t, mu_s, var_s)
        # 一般不相等
        assert abs(kl_forward.item() - kl_backward.item()) > 0.01

    def test_kl_numerical_stability(self):
        """方差接近零时不崩溃"""
        mu_s = torch.zeros(10)
        var_s = torch.ones(10) * 1e-10
        mu_t = torch.ones(10)
        var_t = torch.ones(10)
        kl = gaussian_kl(mu_s, var_s, mu_t, var_t)
        assert torch.isfinite(kl)


class TestEMAMeter:
    """EMA 特征统计测试"""

    def test_first_update(self):
        ema = EMAMeter(decay=0.99)
        feat = torch.randn(4, 768)
        ema.update(feat)
        assert ema.mu is not None
        assert ema.var is not None
        assert ema.mu.shape == (768,)

    def test_smoothing(self):
        ema = EMAMeter(decay=0.9)
        feat1 = torch.zeros(4, 10)
        feat2 = torch.ones(4, 10)
        ema.update(feat1)
        ema.update(feat2)
        # 应该在 0 和 1 之间
        assert (ema.mu > 0).all() and (ema.mu < 1).all()

    def test_reset(self):
        ema = EMAMeter(decay=0.99)
        ema.update(torch.randn(4, 10))
        ema.reset()
        assert ema.mu is None
        assert ema.var is None
