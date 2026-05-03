"""评估指标计算"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_prob=None):
    """计算分类指标"""
    metrics = {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
    }
    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc"] = 0.0

    # 负面类别 recall (业务指标)
    metrics["recall_negative"] = recall_score(y_true, y_pred, pos_label=0)

    return metrics


def find_optimal_threshold(y_prob, y_true, metric="f1"):
    """扫描 [0.05, 0.95], step=0.01, 找最优阈值

    Args:
        y_prob: sigmoid 概率
        y_true: 真实标签
        metric: 优化目标 ('f1', 'recall_negative')

    Returns:
        (best_tau, best_score)
    """
    best_tau = 0.5
    best_score = 0.0

    for tau in np.arange(0.05, 0.95, 0.01):
        preds = (y_prob >= tau).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, average="macro")
        elif metric == "recall_negative":
            score = recall_score(y_true, preds, pos_label=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_tau = tau
            best_score = score

    return best_tau, best_score
