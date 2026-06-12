"""Cohen's Kappa 一致性系数计算模块.

用于评估两位标注员对目标域社会事件评论人工标注的一致性。
"""
from __future__ import annotations
import numpy as np


def compute_cohens_kappa(y1: list[int] | np.ndarray, y2: list[int] | np.ndarray) -> float:
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    
    total = len(y1)
    if total == 0:
        return 0.0

    classes = np.unique(np.concatenate([y1, y2]))
    n_classes = len(classes)
    
    if n_classes <= 1:
        return 1.0 if np.array_equal(y1, y2) else 0.0

    class_to_idx = {c: i for i, c in enumerate(classes)}
    conf_matrix = np.zeros((n_classes, n_classes), dtype=np.float64)
    for val1, val2 in zip(y1, y2):
        conf_matrix[class_to_idx[val1], class_to_idx[val2]] += 1
        
    po = np.trace(conf_matrix) / total

    row_sums = conf_matrix.sum(axis=1)
    col_sums = conf_matrix.sum(axis=0)
    pe = np.sum(row_sums * col_sums) / (total ** 2)

    if pe >= 1.0:
        return 1.0
        
    kappa = (po - pe) / (1.0 - pe)
    return float(kappa)

