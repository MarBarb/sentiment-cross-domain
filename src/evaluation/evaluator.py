"""完整评估流程"""
import logging
from typing import Dict

import numpy as np
import torch

from .metrics import compute_metrics, find_optimal_threshold

logger = logging.getLogger(__name__)


class Evaluator:
    """跨域评估器

    流程:
    1. 在目标域验证集上找 τ* = argmax_τ F1_t_val(τ)
    2. 用 τ* 计算 F1_s_test, F1_t_test
    3. 计算 ΔF1, AUC, KL/MMD
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def evaluate(
        self, loader_s_test, loader_t_val, loader_t_test
    ) -> Dict[str, float]:
        """完整评估

        Returns:
            dict with keys: F1_s, F1_t, deltaF1, tau, AUC_s, AUC_t, ...
        """
        # 1. 收集目标域验证集概率
        probs_t_val, labels_t_val = self._collect_probs(loader_t_val)

        # 2. 找最优阈值
        tau, _ = find_optimal_threshold(probs_t_val, labels_t_val)
        logger.info(f"Optimal threshold τ* = {tau:.2f}")

        # 3. 在测试集上评估
        probs_s, labels_s = self._collect_probs(loader_s_test)
        probs_t, labels_t = self._collect_probs(loader_t_test)

        preds_s = (probs_s >= tau).astype(int)
        preds_t = (probs_t >= tau).astype(int)

        metrics_s = compute_metrics(labels_s, preds_s, probs_s)
        metrics_t = compute_metrics(labels_t, preds_t, probs_t)

        result = {
            "F1_s": metrics_s["f1_macro"],
            "F1_t": metrics_t["f1_macro"],
            "deltaF1": metrics_s["f1_macro"] - metrics_t["f1_macro"],
            "tau": tau,
            "AUC_s": metrics_s.get("auc", 0.0),
            "AUC_t": metrics_t.get("auc", 0.0),
            "recall_neg_s": metrics_s["recall_negative"],
            "recall_neg_t": metrics_t["recall_negative"],
        }

        logger.info(
            f"F1(s)={result['F1_s']:.3f}, F1(t)={result['F1_t']:.3f}, "
            f"ΔF1={result['deltaF1']:.3f}, τ={result['tau']:.2f}"
        )

        return result

    @torch.no_grad()
    def _collect_probs(self, loader):
        """收集模型预测概率和标签"""
        self.model.eval()
        all_probs = []
        all_labels = []

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            probs = self.model.predict_proba(batch["input_ids"], batch["attention_mask"])
            all_probs.append(probs.cpu().numpy())
            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())

        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels) if all_labels else None
        return probs, labels
