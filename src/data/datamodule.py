"""跨域数据模块 — 管理源域和目标域数据加载"""
import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from transformers import AutoTokenizer

from .cleaner import TextCleaner
from .datasets import SourceDataset, TargetLabeledDataset, TargetUnlabeledDataset

logger = logging.getLogger(__name__)


class CrossDomainDataModule:
    """跨域情感分析数据模块

    管理:
    - 源域: SST-2 + IMDB (大规模标注影评)
    - 目标域: 社会事件评论 (少量标注 + 大量无标注)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
        self.max_length = cfg.max_length
        self.batch_size_s = cfg.batch_size_s
        self.batch_size_t = cfg.batch_size_t
        self.seed = cfg.seed
        self.cleaner = TextCleaner()

        self._pos_weight = None

    def setup(self):
        """加载数据集, 划分目标域"""
        logger.info("Setting up data module...")

        # 加载源域数据
        self.ds_source = self._load_source()
        logger.info(f"Source domain: {len(self.ds_source)} samples")

        # 加载目标域数据
        self.ds_target_labeled, self.ds_target_unlabeled, self.ds_target_test = \
            self._load_target()
        logger.info(
            f"Target domain: {len(self.ds_target_labeled)} labeled, "
            f"{len(self.ds_target_unlabeled)} unlabeled, "
            f"{len(self.ds_target_test)} test"
        )

        # 计算类别权重
        self._compute_pos_weight()

    def train_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """返回 (loader_s, loader_t)"""
        loader_s = DataLoader(
            self.ds_source,
            batch_size=self.batch_size_s,
            shuffle=True,
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )

        # 目标域可能很小, 需要过采样
        if len(self.ds_target_labeled) < self.batch_size_t:
            sampler = RandomSampler(
                self.ds_target_labeled,
                replacement=True,
                num_samples=self.batch_size_t * 100,
            )
            loader_t = DataLoader(
                self.ds_target_labeled,
                batch_size=self.batch_size_t,
                sampler=sampler,
                num_workers=self.cfg.get("num_workers", 4),
                pin_memory=True,
            )
        else:
            loader_t = DataLoader(
                self.ds_target_labeled,
                batch_size=self.batch_size_t,
                shuffle=True,
                num_workers=self.cfg.get("num_workers", 4),
                pin_memory=True,
                drop_last=True,
            )

        return loader_s, loader_t

    def val_loader(self) -> DataLoader:
        """目标域验证集"""
        # TODO: 实现验证集加载
        raise NotImplementedError

    def test_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """返回 (loader_s_test, loader_t_test)"""
        # TODO: 实现测试集加载
        raise NotImplementedError

    @property
    def pos_weight(self) -> Optional[torch.Tensor]:
        """目标域类别不平衡权重"""
        return self._pos_weight

    def _load_source(self):
        """加载源域数据 (SST-2 + IMDB)"""
        # TODO: 实现数据加载
        # 1. 从 HuggingFace / 本地加载 SST-2 和 IMDB
        # 2. 合并为统一的源域数据集
        # 3. 应用 TextCleaner
        raise NotImplementedError("实现源域数据加载")

    def _load_target(self):
        """加载目标域数据 (社会事件评论)"""
        # TODO: 实现数据加载
        # 1. 从本地 CSV / JSON 加载
        # 2. 分层划分 labeled / unlabeled / test (60/20/20)
        # 3. 应用 TextCleaner
        raise NotImplementedError("实现目标域数据加载")

    def _compute_pos_weight(self):
        """计算目标域正负样本比, 用于 BCEWithLogitsLoss"""
        # TODO: 从目标域标注数据计算
        # self._pos_weight = torch.tensor([neg_count / pos_count])
        pass
