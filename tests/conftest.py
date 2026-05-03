"""Pytest 配置和公共 fixtures"""
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from omegaconf import OmegaConf
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False


@pytest.fixture
def cfg():
    """默认测试配置"""
    if HAS_OMEGACONF:
        return OmegaConf.create({
            "backbone": "bert-base-uncased",
            "max_length": 128,
            "batch_size_s": 4,
            "batch_size_t": 4,
            "seed": 42,
            "dropout": 0.1,
            "freeze_layers": 0,
            "num_workers": 0,
        })
    else:
        # 简单 dict fallback
        return {
            "backbone": "bert-base-uncased",
            "max_length": 128,
            "batch_size_s": 4,
            "batch_size_t": 4,
            "seed": 42,
            "dropout": 0.1,
            "freeze_layers": 0,
            "num_workers": 0,
        }


@pytest.fixture
def device():
    if HAS_TORCH:
        return torch.device("cpu")
    return "cpu"
