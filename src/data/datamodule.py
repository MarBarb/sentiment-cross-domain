"""跨域数据模块.

完整 BERT/RoBERTa 训练仍依赖 torch 与 transformers；中期阶段先把本地 CSV
数据加载、目标域划分和类别权重补齐，便于后续直接接入深度模型。
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

try:
    import torch
    from torch.utils.data import DataLoader, RandomSampler
except ImportError:  # pragma: no cover - 轻量基线环境可不装 torch
    torch = None
    DataLoader = None
    RandomSampler = None

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None

from .cleaner import TextCleaner
from .datasets import SourceDataset, TargetLabeledDataset, TargetUnlabeledDataset
from .local_loader import load_cross_domain_splits

logger = logging.getLogger(__name__)


def _cfg_get(cfg, key, default=None):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class CrossDomainDataModule:
    """跨域情感分析数据模块.

    管理：
    - 源域：影评/书评样例或 SST-2/IMDB 扩展数据
    - 目标域：社会事件评论，按 train/val/test 划分
    """

    def __init__(self, cfg):
        if torch is None or AutoTokenizer is None:
            raise ImportError(
                "CrossDomainDataModule 需要 torch 和 transformers。"
                "轻量 TF-IDF 基线请使用 `python run.py model=tfidf_lr`。"
            )

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(_cfg_get(cfg, "backbone", "bert-base-uncased"))
        self.max_length = _cfg_get(cfg, "max_length", 128)
        self.batch_size_s = _cfg_get(cfg, "batch_size_s", 32)
        self.batch_size_t = _cfg_get(cfg, "batch_size_t", 16)
        self.seed = _cfg_get(cfg, "seed", 42)
        self.cleaner = TextCleaner()

        self._pos_weight: Optional[torch.Tensor] = None
        self.splits = None

    def setup(self):
        """加载数据集并构建 Dataset."""
        logger.info("Setting up data module...")
        self.splits = load_cross_domain_splits(
            source_path=_cfg_get(self.cfg, "source_data_path", "data/processed/source_sample.csv"),
            target_path=_cfg_get(self.cfg, "target_data_path", "data/processed/social_sample.csv"),
            cleaner=self.cleaner,
        )

        self.ds_source = self._make_source(self.splits.source_train)
        self.ds_source_test = self._make_source(self.splits.source_test)
        self.ds_target_labeled = self._make_target(self.splits.target_train)
        self.ds_target_val = self._make_target(self.splits.target_val)
        self.ds_target_test = self._make_target(self.splits.target_test)
        self.ds_target_unlabeled = TargetUnlabeledDataset(
            self.splits.target_train.texts + self.splits.target_val.texts,
            self.tokenizer,
            self.max_length,
        )

        self._compute_pos_weight()
        logger.info(
            "Data ready: source_train=%s, target_train=%s, target_val=%s, target_test=%s",
            len(self.ds_source),
            len(self.ds_target_labeled),
            len(self.ds_target_val),
            len(self.ds_target_test),
        )

    def train_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """返回 (loader_s, loader_t)."""
        loader_s = DataLoader(
            self.ds_source,
            batch_size=self.batch_size_s,
            shuffle=True,
            num_workers=_cfg_get(self.cfg, "num_workers", 0),
            pin_memory=True,
            drop_last=len(self.ds_source) >= self.batch_size_s,
        )

        if len(self.ds_target_labeled) < self.batch_size_t:
            sampler = RandomSampler(
                self.ds_target_labeled,
                replacement=True,
                num_samples=self.batch_size_t * 50,
            )
            loader_t = DataLoader(
                self.ds_target_labeled,
                batch_size=self.batch_size_t,
                sampler=sampler,
                num_workers=_cfg_get(self.cfg, "num_workers", 0),
                pin_memory=True,
            )
        else:
            loader_t = DataLoader(
                self.ds_target_labeled,
                batch_size=self.batch_size_t,
                shuffle=True,
                num_workers=_cfg_get(self.cfg, "num_workers", 0),
                pin_memory=True,
                drop_last=len(self.ds_target_labeled) >= self.batch_size_t,
            )
        return loader_s, loader_t

    def val_loader(self) -> DataLoader:
        """目标域验证集."""
        return DataLoader(
            self.ds_target_val,
            batch_size=self.batch_size_t,
            shuffle=False,
            num_workers=_cfg_get(self.cfg, "num_workers", 0),
            pin_memory=True,
        )

    def test_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """返回 (loader_s_test, loader_t_test)."""
        loader_s = DataLoader(
            self.ds_source_test,
            batch_size=self.batch_size_s,
            shuffle=False,
            num_workers=_cfg_get(self.cfg, "num_workers", 0),
            pin_memory=True,
        )
        loader_t = DataLoader(
            self.ds_target_test,
            batch_size=self.batch_size_t,
            shuffle=False,
            num_workers=_cfg_get(self.cfg, "num_workers", 0),
            pin_memory=True,
        )
        return loader_s, loader_t

    @property
    def pos_weight(self) -> Optional[torch.Tensor]:
        """目标域正类权重，供 BCEWithLogitsLoss 使用."""
        return self._pos_weight

    def _make_source(self, split):
        return SourceDataset(split.texts, split.labels, self.tokenizer, self.max_length)

    def _make_target(self, split):
        return TargetLabeledDataset(split.texts, split.labels, self.tokenizer, self.max_length)

    def _compute_pos_weight(self):
        labels = self.splits.target_train.labels
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        if pos_count == 0:
            self._pos_weight = torch.tensor([1.0])
        else:
            self._pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)

