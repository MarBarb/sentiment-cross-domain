"""组合损失函数"""
import torch
import torch.nn as nn
from .kl_loss import gaussian_kl
from .ema import EMAMeter


class DomainAdaptLoss(nn.Module):
    """跨域适应损失 = L_sup + λ_kl * L_kl + λ_adv * L_adv"""

    def __init__(
        self,
        lambda_kl: float = 0.1,
        lambda_adv: float = 0.0,
        pos_weight: torch.Tensor = None,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_adv = lambda_adv

        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()

        self.ema_s = EMAMeter(decay=ema_decay)
        self.ema_t = EMAMeter(decay=ema_decay)

    def forward(
        self,
        feat_s: torch.Tensor,
        feat_t: torch.Tensor,
        logit_t: torch.Tensor,
        labels_t: torch.Tensor,
        domain_pred: torch.Tensor = None,
        domain_labels: torch.Tensor = None,
    ) -> dict:
        """
        Returns:
            dict with keys: loss, L_sup, L_kl, L_adv
        """
        # 更新 EMA
        self.ema_s.update(feat_s.detach())
        self.ema_t.update(feat_t.detach())

        # 监督损失
        L_sup = self.bce(logit_t, labels_t.float())

        # KL 对齐损失
        L_kl = gaussian_kl(
            self.ema_s.mu, self.ema_s.var, self.ema_t.mu, self.ema_t.var
        )

        # 对抗损失 (可选)
        L_adv = torch.tensor(0.0, device=logit_t.device)
        if domain_pred is not None and domain_labels is not None:
            L_adv = nn.functional.binary_cross_entropy_with_logits(
                domain_pred, domain_labels
            )

        loss = L_sup + self.lambda_kl * L_kl + self.lambda_adv * L_adv

        return {
            "loss": loss,
            "L_sup": L_sup.detach(),
            "L_kl": L_kl.detach(),
            "L_adv": L_adv.detach(),
        }
