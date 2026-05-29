"""评估指标计算.

本模块默认使用轻量 numpy 实现，避免基础基线实验依赖 scikit-learn。
如果后续安装了 scikit-learn，也可以用这些函数得到一致的二分类指标。
"""
import numpy as np


def _as_np(x):
    return np.asarray(x).astype(int)


def _binary_counts(y_true, y_pred, label):
    y_true = _as_np(y_true)
    y_pred = _as_np(y_pred)
    tp = int(((y_true == label) & (y_pred == label)).sum())
    fp = int(((y_true != label) & (y_pred == label)).sum())
    fn = int(((y_true == label) & (y_pred != label)).sum())
    support = int((y_true == label).sum())
    return tp, fp, fn, support


def _prf(y_true, y_pred, label):
    tp, fp, fn, support = _binary_counts(y_true, y_pred, label)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, support


def _roc_auc_score(y_true, y_prob):
    y_true = _as_np(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    wins = 0.0
    for p in pos:
        wins += float((p > neg).sum())
        wins += 0.5 * float((p == neg).sum())
    return wins / (len(pos) * len(neg))


def compute_metrics(y_true, y_pred, y_prob=None):
    """计算分类指标"""
    y_true = _as_np(y_true)
    y_pred = _as_np(y_pred)
    p0, r0, f10, s0 = _prf(y_true, y_pred, 0)
    p1, r1, f11, s1 = _prf(y_true, y_pred, 1)
    total = max(s0 + s1, 1)

    metrics = {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "f1_macro": (f10 + f11) / 2,
        "f1_weighted": (s0 * f10 + s1 * f11) / total,
        "precision": (p0 + p1) / 2,
        "recall": (r0 + r1) / 2,
        "precision_negative": p0,
        "recall_negative": r0,
        "f1_negative": f10,
        "precision_positive": p1,
        "recall_positive": r1,
        "f1_positive": f11,
    }
    if y_prob is not None:
        metrics["auc"] = _roc_auc_score(y_true, y_prob)

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
            score = compute_metrics(y_true, preds)["f1_macro"]
        elif metric == "recall_negative":
            score = compute_metrics(y_true, preds)["recall_negative"]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_tau = tau
            best_score = score

    return best_tau, best_score
