"""训练器 — 跨域情感分析核心训练循环"""
import logging
from pathlib import Path
from itertools import cycle

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)


class Trainer:
    """跨域适应训练器

    训练循环: 源域预训练 → KL 对齐迁移 → 目标域微调
    """

    def __init__(self, model, datamodule, cfg):
        self.model = model
        self.datamodule = datamodule
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 优化器 (分层学习率)
        self.optimizer = self._build_optimizer()
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.epochs, eta_min=1e-7
        )

        # 损失
        from .losses import DomainAdaptLoss
        self.criterion = DomainAdaptLoss(
            lambda_kl=cfg.lambda_kl,
            lambda_adv=cfg.get("lambda_adv", 0.0),
            pos_weight=datamodule.pos_weight,
            ema_decay=cfg.get("ema_decay", 0.99),
        )

        # 域判别器 (可选)
        self.discriminator = None
        if cfg.get("use_adversarial", False):
            from ..models import DomainDiscriminator
            hidden_size = model.encoder.config.hidden_size
            self.discriminator = DomainDiscriminator(hidden_size).to(self.device)

        # W&B
        self.global_step = 0

    def _build_optimizer(self):
        """分层学习率: backbone lr, head lr_head, adapter lr_adapter"""
        cfg = self.cfg
        param_groups = [
            {"params": self.model.encoder.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": self.model.head.parameters(), "lr": cfg.get("lr_head", 1e-3), "weight_decay": 0.0},
        ]
        return AdamW(param_groups)

    def fit(self) -> dict:
        """完整训练流程

        Returns:
            checkpoint dict
        """
        cfg = self.cfg
        best_delta_f1 = float("inf")
        patience_counter = 0

        loader_s, loader_t = self.datamodule.train_loaders()

        for epoch in range(cfg.epochs):
            # 冻结策略切换
            if epoch == cfg.get("warmup_epochs", 2):
                self.model.unfreeze_all()
                logger.info(f"Epoch {epoch}: Unfreezing all layers")

            # 训练一个 epoch
            train_metrics = self._train_epoch(loader_s, loader_t, epoch)

            # 验证
            # val_metrics = self._validate()

            # Early stopping
            # if val_metrics["deltaF1"] < best_delta_f1:
            #     best_delta_f1 = val_metrics["deltaF1"]
            #     patience_counter = 0
            #     self._save_checkpoint(epoch, val_metrics)
            # else:
            #     patience_counter += 1
            #     if patience_counter >= cfg.get("patience", 5):
            #         logger.info(f"Early stopping at epoch {epoch}")
            #         break

            self.scheduler.step()

        return {"model_state": self.model.state_dict(), "epoch": epoch}

    def _train_epoch(self, loader_s, loader_t, epoch: int) -> dict:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch_s, batch_t in zip(loader_s, cycle(loader_t)):
            # Move to device
            batch_s = {k: v.to(self.device) for k, v in batch_s.items()}
            batch_t = {k: v.to(self.device) for k, v in batch_t.items()}

            # Forward
            feat_s, _ = self.model(batch_s["input_ids"], batch_s["attention_mask"])
            feat_t, logit_t = self.model(batch_t["input_ids"], batch_t["attention_mask"])

            # 域判别 (可选)
            domain_pred = None
            domain_labels = None
            if self.discriminator is not None:
                feat_all = torch.cat([feat_s, feat_t], dim=0)
                domain_pred = self.discriminator(feat_all)
                domain_labels = torch.cat([
                    torch.ones(feat_s.size(0), 1),
                    torch.zeros(feat_t.size(0), 1),
                ], dim=0).to(self.device)

            # Loss
            losses = self.criterion(
                feat_s, feat_t, logit_t, batch_t["labels"],
                domain_pred, domain_labels,
            )

            # Backward
            self.optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.get("grad_clip", 1.0)
            )
            self.optimizer.step()

            total_loss += losses["loss"].item()
            n_batches += 1
            self.global_step += 1

            # Logging
            if self.global_step % self.cfg.get("log_interval", 50) == 0:
                logger.info(
                    f"Epoch {epoch} Step {self.global_step}: "
                    f"loss={losses['loss'].item():.4f} "
                    f"L_sup={losses['L_sup'].item():.4f} "
                    f"L_kl={losses['L_kl'].item():.4f}"
                )

        return {"avg_loss": total_loss / max(n_batches, 1)}
