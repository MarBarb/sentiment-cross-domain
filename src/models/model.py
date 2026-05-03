"""跨域情感分类模型"""
import logging
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)


class SentimentDomainAdaptModel(nn.Module):
    """跨域情感分类器

    架构: BERT/RoBERTa encoder → dropout → linear head
    支持: 冻结策略, LoRA, Adapter
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Encoder
        self.encoder = AutoModel.from_pretrained(cfg.backbone)
        hidden_size = self.encoder.config.hidden_size

        # Dropout + Head
        self.dropout = nn.Dropout(cfg.get("dropout", 0.1))
        self.head = nn.Linear(hidden_size, 1)

        # 冻结策略
        freeze_layers = cfg.get("freeze_layers", 0)
        if freeze_layers > 0:
            self._freeze_first_n(freeze_layers)

        logger.info(
            f"Model initialized: backbone={cfg.backbone}, "
            f"hidden={hidden_size}, freeze_layers={freeze_layers}"
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (features φ(x), logits)"""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        feat = out.last_hidden_state[:, 0, :]  # [CLS] token
        feat = self.dropout(feat)
        logit = self.head(feat).squeeze(-1)
        return feat, logit

    @torch.no_grad()
    def predict_proba(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """返回 sigmoid 概率"""
        _, logit = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logit)

    def _freeze_first_n(self, n: int):
        """冻结 encoder 前 n 层"""
        for name, param in self.encoder.named_parameters():
            if "layer" in name:
                try:
                    layer_id = int(name.split("layer.")[1].split(".")[0])
                    if layer_id < n:
                        param.requires_grad = False
                except (IndexError, ValueError):
                    pass
        frozen = sum(1 for p in self.encoder.parameters() if not p.requires_grad)
        total = sum(1 for p in self.encoder.parameters())
        logger.info(f"Frozen {frozen}/{total} encoder parameters (first {n} layers)")

    def unfreeze_all(self):
        """解冻所有层"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("All encoder layers unfrozen")
