"""评估模块测试"""
import pytest
import numpy as np
from src.evaluation.metrics import compute_metrics, find_optimal_threshold


class TestMetrics:
    """指标计算测试"""

    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["f1_macro"] == 1.0

    def test_worst_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["f1_macro"] == 0.0

    def test_recall_negative(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        metrics = compute_metrics(y_true, y_pred)
        # 负面样本 3 个, 预测正确 2 个
        assert abs(metrics["recall_negative"] - 2 / 3) < 1e-5


class TestThresholdSearch:
    """阈值搜索测试"""

    def test_optimal_threshold(self):
        # 完美分离的情况
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        y_true = np.array([0, 0, 0, 1, 1, 1])
        tau, score = find_optimal_threshold(y_prob, y_true)
        assert score == 1.0
        assert 0.3 < tau < 0.7  # 最优阈值在间隙中

    def test_threshold_range(self):
        y_prob = np.random.rand(100)
        y_true = (y_prob > 0.5).astype(int)
        tau, _ = find_optimal_threshold(y_prob, y_true)
        assert 0.05 <= tau <= 0.95

    def test_delta_f1_sign(self):
        """ΔF1 符号: F1_s >= F1_t 时为正"""
        f1_s = 0.9
        f1_t = 0.7
        delta = f1_s - f1_t
        assert delta > 0  # 正常迁移后应为正
