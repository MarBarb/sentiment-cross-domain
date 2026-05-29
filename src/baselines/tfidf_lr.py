"""轻量 TF-IDF + Logistic Regression 基线.

这个实现只依赖 numpy，便于在课程中期阶段先跑通可复现实验闭环。
正式实验可替换为 scikit-learn 或 BERT/RoBERTa 版本。
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.data.local_loader import TextSplit, describe_splits, load_cross_domain_splits
from src.evaluation.metrics import compute_metrics, find_optimal_threshold


class CharTfidfVectorizer:
    """字符 n-gram TF-IDF，适合中文短文本 smoke benchmark."""

    def __init__(self, ngram_range=(2, 4), max_features=5000, min_df=1):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.vocab_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def _ngrams(self, text: str) -> list[str]:
        text = "".join(text.lower().split())
        grams: list[str] = []
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            if len(text) < n:
                continue
            grams.extend(text[i : i + n] for i in range(len(text) - n + 1))
        return grams

    def fit(self, texts: list[str]):
        df = Counter()
        tf = Counter()
        for text in texts:
            grams = self._ngrams(text)
            tf.update(grams)
            df.update(set(grams))

        candidates = [g for g, c in df.items() if c >= self.min_df]
        candidates.sort(key=lambda g: (tf[g], df[g], g), reverse=True)
        selected = candidates[: self.max_features]
        self.vocab_ = {g: i for i, g in enumerate(selected)}

        n_docs = max(len(texts), 1)
        self.idf_ = np.ones(len(self.vocab_), dtype=np.float64)
        for gram, idx in self.vocab_.items():
            self.idf_[idx] = math.log((1 + n_docs) / (1 + df[gram])) + 1.0
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.idf_ is None:
            raise RuntimeError("Vectorizer must be fitted before transform().")
        x = np.zeros((len(texts), len(self.vocab_)), dtype=np.float64)
        for row, text in enumerate(texts):
            counts = Counter(g for g in self._ngrams(text) if g in self.vocab_)
            if not counts:
                continue
            total = sum(counts.values())
            for gram, count in counts.items():
                x[row, self.vocab_[gram]] = (count / total) * self.idf_[self.vocab_[gram]]
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


class LogisticRegressionGD:
    """二分类逻辑回归，使用全批量梯度下降."""

    def __init__(self, lr=0.8, epochs=900, l2=1e-4, seed=42):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.w: np.ndarray | None = None
        self.b = 0.0
        self.loss_history_: list[float] = []

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -40, 40)
        return 1 / (1 + np.exp(-z))

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0, 0.01, size=x.shape[1])
        self.b = 0.0
        y = y.astype(np.float64)
        weight = np.ones_like(y) if sample_weight is None else sample_weight.astype(np.float64)
        weight = weight / weight.mean()

        for _ in range(self.epochs):
            prob = self._sigmoid(x @ self.w + self.b)
            err = (prob - y) * weight
            grad_w = (x.T @ err) / len(y) + self.l2 * self.w
            grad_b = float(err.mean())
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            loss = -np.mean(weight * (y * np.log(prob + 1e-9) + (1 - y) * np.log(1 - prob + 1e-9)))
            loss += 0.5 * self.l2 * float((self.w**2).sum())
            self.loss_history_.append(float(loss))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model must be fitted before predict_proba().")
        return self._sigmoid(x @ self.w + self.b)


def _labels(split: TextSplit) -> np.ndarray:
    return np.asarray(split.labels, dtype=int)


def _evaluate_method(name, model, vectorizer, splits, tune_on_target=True):
    p_val = model.predict_proba(vectorizer.transform(splits.target_val.texts))
    tau, val_macro_f1 = find_optimal_threshold(p_val, _labels(splits.target_val))
    if not tune_on_target:
        tau = 0.5

    rows = {}
    for split_name in ["source_test", "target_test"]:
        split = getattr(splits, split_name)
        prob = model.predict_proba(vectorizer.transform(split.texts))
        pred = (prob >= tau).astype(int)
        rows[split_name] = {
            "prob": prob,
            "pred": pred,
            "metrics": compute_metrics(_labels(split), pred, prob),
        }

    metrics_s = rows["source_test"]["metrics"]
    metrics_t = rows["target_test"]["metrics"]
    return {
        "method": name,
        "tau": float(tau),
        "target_val_macro_f1_at_tau": float(val_macro_f1),
        "F1_s": float(metrics_s["f1_macro"]),
        "F1_t": float(metrics_t["f1_macro"]),
        "deltaF1": float(metrics_s["f1_macro"] - metrics_t["f1_macro"]),
        "AUC_s": float(metrics_s.get("auc", 0.0)),
        "AUC_t": float(metrics_t.get("auc", 0.0)),
        "accuracy_t": float(metrics_t["accuracy"]),
        "recall_negative_t": float(metrics_t["recall_negative"]),
        "recall_positive_t": float(metrics_t["recall_positive"]),
        "rows": rows,
    }


def _prediction_rows(method_result, split: TextSplit, split_name: str):
    rows = method_result["rows"][split_name]
    for text, label, prob, pred in zip(split.texts, split.labels, rows["prob"], rows["pred"]):
        yield {
            "method": method_result["method"],
            "split": split_name,
            "text": text,
            "label": int(label),
            "prob_positive": round(float(prob), 6),
            "prediction": int(pred),
            "correct": int(pred == label),
        }


def run_tfidf_experiment(
    source_path="data/processed/source_sample.csv",
    target_path="data/processed/social_sample.csv",
    output_dir="results",
    seed=42,
    max_features=3000,
    epochs=900,
):
    """运行 E0-lite 和 target-calibrated 轻量实验."""
    splits = load_cross_domain_splits(source_path=source_path, target_path=target_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_vectorizer = CharTfidfVectorizer()
    source_ngrams = set()
    target_ngrams = set()
    for text in splits.source_train.texts + splits.source_test.texts:
        source_ngrams.update(audit_vectorizer._ngrams(text))
    for text in splits.target_train.texts + splits.target_val.texts + splits.target_test.texts:
        target_ngrams.update(audit_vectorizer._ngrams(text))
    ngram_union = source_ngrams | target_ngrams
    ngram_overlap = source_ngrams & target_ngrams

    # E0-lite: 仅源域训练，阈值在目标域验证集上扫描。
    vec_source = CharTfidfVectorizer(max_features=max_features)
    x_source = vec_source.fit_transform(splits.source_train.texts)
    y_source = _labels(splits.source_train)
    source_model = LogisticRegressionGD(seed=seed, epochs=epochs).fit(x_source, y_source)
    source_result = _evaluate_method("E0-lite: source TF-IDF+LR", source_model, vec_source, splits)

    # Midterm adapted: 加入少量目标域金标，并采用保守权重避免小样本过拟合。
    train_texts = splits.source_train.texts + splits.target_train.texts
    train_labels = np.asarray(splits.source_train.labels + splits.target_train.labels, dtype=int)
    sample_weight = np.asarray([1.0] * len(splits.source_train) + [0.5] * len(splits.target_train))
    vec_adapt = CharTfidfVectorizer(max_features=max_features)
    x_adapt = vec_adapt.fit_transform(train_texts)
    adapt_model = LogisticRegressionGD(seed=seed, epochs=epochs).fit(x_adapt, train_labels, sample_weight)
    adapt_result = _evaluate_method("E1-lite: source + target calibration", adapt_model, vec_adapt, splits)

    summary = {
        "seed": seed,
        "source_path": str(source_path),
        "target_path": str(target_path),
        "dataset": describe_splits(splits),
        "audit": {
            "source_char_ngrams": len(source_ngrams),
            "target_char_ngrams": len(target_ngrams),
            "shared_char_ngrams": len(ngram_overlap),
            "char_ngram_jaccard": len(ngram_overlap) / len(ngram_union) if ngram_union else 0.0,
        },
        "experiments": [
            {k: v for k, v in source_result.items() if k != "rows"},
            {k: v for k, v in adapt_result.items() if k != "rows"},
        ],
    }

    metrics_path = output_dir / "midterm_tfidf_metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    pred_path = output_dir / "midterm_tfidf_predictions.csv"
    import csv

    fieldnames = ["method", "split", "text", "label", "prob_positive", "prediction", "correct"]
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in (source_result, adapt_result):
            for split_name in ("source_test", "target_test"):
                split = getattr(splits, split_name)
                writer.writerows(_prediction_rows(result, split, split_name))

    return summary, metrics_path, pred_path
