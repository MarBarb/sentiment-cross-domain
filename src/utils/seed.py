"""随机种子设置"""
import random
import numpy as np
import torch


def set_seed(seed: int):
    """设置全局随机种子, 确保可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
