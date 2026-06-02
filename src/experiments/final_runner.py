"""最终可交付实验矩阵.

该 runner 在 CPU 环境下运行，不依赖 PyTorch / Transformers / scikit-learn。
它实现 E0-E6 的可复现消融矩阵、三随机种子统计、目标域阈值调优、
KL/MMD 风格分布诊断、错误案例导出和图表生成。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.data.cleaner import TextCleaner
from src.evaluation.metrics import compute_metrics


@dataclass
class SplitData:
    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)


@dataclass
class DatasetBundle:
    source_train: SplitData
    source_val: SplitData
    source_test: SplitData
    target_train: SplitData
    target_val: SplitData
    target_test: SplitData
    target_unlabeled: SplitData


POSITIVE_HINTS = {
    "好", "喜欢", "满意", "推荐", "棒", "赞", "支持", "开心", "幸福", "感谢", "温暖", "顺利", "优秀",
    "舒服", "惊喜", "及时", "靠谱", "贴心", "值得", "放心", "有效", "不错", "满意", "成功",
}
NEGATIVE_HINTS = {
    "差", "慢", "坏", "失望", "投诉", "垃圾", "难吃", "糟糕", "生气", "愤怒", "离谱", "不满", "延迟",
    "延期", "混乱", "欺骗", "谣言", "危险", "不公平", "失败", "敷衍", "推诿", "恶心", "崩溃",
}


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


class HashingTextVectorizer:
    def __init__(
        self,
        n_features: int = 8192,
        ngram_range: tuple[int, int] = (2, 4),
        use_lexicon: bool = True,
        drop_tokens: set[str] | None = None,
    ):
        self.n_features = n_features
        self.ngram_range = ngram_range
        self.use_lexicon = use_lexicon
        self.drop_tokens = drop_tokens or set()

    def tokens(self, text: str) -> list[str]:
        text = "".join(text.lower().split())
        out: list[str] = []
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            if len(text) < n:
                continue
            for i in range(len(text) - n + 1):
                token = f"c{n}:{text[i : i + n]}"
                if token not in self.drop_tokens:
                    out.append(token)
        if self.use_lexicon:
            pos_count = sum(1 for w in POSITIVE_HINTS if w in text)
            neg_count = sum(1 for w in NEGATIVE_HINTS if w in text)
            if pos_count:
                out.extend(["lex:pos"] * min(pos_count, 5))
            if neg_count:
                out.extend(["lex:neg"] * min(neg_count, 5))
            if "不" in text or "没" in text or "无" in text:
                out.append("lex:negation")
        return out

    def transform_one(self, text: str) -> dict[int, float]:
        counts = Counter(stable_hash(tok) % self.n_features for tok in self.tokens(text))
        if not counts:
            return {}
        norm = math.sqrt(sum(v * v for v in counts.values()))
        return {idx: value / norm for idx, value in counts.items()}

    def transform(self, texts: list[str]) -> list[dict[int, float]]:
        return [self.transform_one(text) for text in texts]


class SparseLogisticRegression:
    def __init__(self, n_features: int, lr: float = 0.18, epochs: int = 14, l2: float = 1e-5, seed: int = 42):
        self.n_features = n_features
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def sigmoid(z: float) -> float:
        z = max(min(z, 40.0), -40.0)
        return 1.0 / (1.0 + math.exp(-z))

    def dot(self, features: dict[int, float]) -> float:
        return float(sum(self.w[idx] * value for idx, value in features.items()) + self.b)

    def fit(self, x: list[dict[int, float]], y: list[int], sample_weight: list[float] | None = None):
        rng = random.Random(self.seed)
        weights = sample_weight or [1.0] * len(y)
        order = list(range(len(y)))
        for epoch in range(self.epochs):
            rng.shuffle(order)
            eta = self.lr / (1.0 + 0.08 * epoch)
            epoch_loss = 0.0
            for i in order:
                prob = self.sigmoid(self.dot(x[i]))
                err = (prob - y[i]) * weights[i]
                for idx, value in x[i].items():
                    self.w[idx] -= eta * (err * value + self.l2 * self.w[idx])
                self.b -= eta * err
                epoch_loss += -weights[i] * (
                    y[i] * math.log(prob + 1e-9) + (1 - y[i]) * math.log(1 - prob + 1e-9)
                )
            self.loss_history.append(epoch_loss / max(len(y), 1))
        return self

    def predict_proba(self, x: list[dict[int, float]]) -> np.ndarray:
        return np.asarray([self.sigmoid(self.dot(row)) for row in x], dtype=np.float64)


def read_split_csv(path: str | Path) -> dict[str, SplitData]:
    cleaner = TextCleaner()
    buckets: dict[str, list[tuple[str, int]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"].strip().lower()
            text = cleaner.clean(row["text"])
            if cleaner.is_noise(text):
                continue
            buckets[split].append((text, int(row["label"])))
    return {
        split: SplitData([t for t, _ in rows], [y for _, y in rows])
        for split, rows in buckets.items()
    }


def load_bundle(source_path: str | Path, target_path: str | Path) -> DatasetBundle:
    source = read_split_csv(source_path)
    target = read_split_csv(target_path)
    required_source = {"train", "val", "test"}
    required_target = {"train", "val", "test", "unlabeled"}
    if not required_source <= source.keys():
        raise ValueError(f"source split 不完整，需要 {required_source}，实际 {source.keys()}")
    if not required_target <= target.keys():
        raise ValueError(f"target split 不完整，需要 {required_target}，实际 {target.keys()}")
    return DatasetBundle(
        source_train=source["train"],
        source_val=source["val"],
        source_test=source["test"],
        target_train=target["train"],
        target_val=target["val"],
        target_test=target["test"],
        target_unlabeled=target["unlabeled"],
    )


def choose_threshold(
    probs: np.ndarray,
    labels: list[int],
    min_negative_recall: float | None = None,
) -> tuple[float, float, float]:
    best_tau, best_f1, best_neg = 0.5, -1.0, 0.0
    fallback = (0.5, -1.0, 0.0)
    for tau in np.arange(0.05, 0.96, 0.01):
        pred = (probs >= tau).astype(int)
        metrics = compute_metrics(labels, pred, probs)
        score = metrics["f1_macro"]
        neg_recall = metrics["recall_negative"]
        if score > fallback[1]:
            fallback = (float(tau), float(score), float(neg_recall))
        if min_negative_recall is not None and neg_recall < min_negative_recall:
            continue
        if score > best_f1:
            best_tau, best_f1, best_neg = float(tau), float(score), float(neg_recall)
    if best_f1 < 0:
        return fallback
    return best_tau, best_f1, best_neg


def domain_specific_tokens(source_texts: list[str], target_texts: list[str], top_k: int = 450) -> set[str]:
    base = HashingTextVectorizer(ngram_range=(2, 4), use_lexicon=False)

    def doc_freq(texts):
        df = Counter()
        for text in texts:
            df.update(set(base.tokens(text)))
        return df

    s_df = doc_freq(source_texts)
    t_df = doc_freq(target_texts)
    n_s, n_t = len(source_texts), len(target_texts)
    scores = []
    for token in set(s_df) | set(t_df):
        ps = (s_df[token] + 1) / (n_s + 2)
        pt = (t_df[token] + 1) / (n_t + 2)
        scores.append((abs(math.log(ps / pt)), token))
    scores.sort(reverse=True)
    return {token for _, token in scores[:top_k]}


def dense_means(features: list[dict[int, float]], n_features: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.zeros((len(features), n_features), dtype=np.float64)
    for i, row in enumerate(features):
        for idx, value in row.items():
            arr[i, idx] = value
    return arr.mean(axis=0), arr.var(axis=0) + 1e-8


def feature_distances(vectorizer: HashingTextVectorizer, source: SplitData, target: SplitData) -> dict:
    # 使用较低维采样做诊断，避免报告指标受高维稀疏噪声主导。
    s_feat = vectorizer.transform(source.texts[: min(len(source), 1200)])
    t_feat = vectorizer.transform(target.texts[: min(len(target), 1200)])
    mu_s, var_s = dense_means(s_feat, vectorizer.n_features)
    mu_t, var_t = dense_means(t_feat, vectorizer.n_features)
    kl = 0.5 * np.sum(var_s / var_t + ((mu_t - mu_s) ** 2) / var_t - 1 + np.log(var_t / var_s))
    mmd_linear = float(np.sum((mu_s - mu_t) ** 2))
    return {"kl_diag_gaussian": float(kl), "mmd_linear": mmd_linear}


def train_and_eval(
    method_id: str,
    method_name: str,
    bundle: DatasetBundle,
    seed: int,
    vectorizer: HashingTextVectorizer,
    train_texts: list[str],
    train_labels: list[int],
    sample_weight: list[float],
    min_negative_recall: float | None = 0.85,
) -> dict:
    train_features = vectorizer.transform(train_texts)
    model = SparseLogisticRegression(
        n_features=vectorizer.n_features,
        lr=0.16,
        epochs=16,
        l2=1e-5,
        seed=seed,
    ).fit(train_features, train_labels, sample_weight)

    val_features = vectorizer.transform(bundle.target_val.texts)
    val_probs = model.predict_proba(val_features)
    tau, val_f1, val_neg_recall = choose_threshold(
        val_probs, bundle.target_val.labels, min_negative_recall=min_negative_recall
    )

    def eval_split(split: SplitData) -> tuple[np.ndarray, np.ndarray, dict]:
        features = vectorizer.transform(split.texts)
        probs = model.predict_proba(features)
        preds = (probs >= tau).astype(int)
        return probs, preds, compute_metrics(split.labels, preds, probs)

    p_s, pred_s, metrics_s = eval_split(bundle.source_test)
    p_t, pred_t, metrics_t = eval_split(bundle.target_test)
    distances = feature_distances(vectorizer, bundle.source_train, bundle.target_train)

    errors = []
    for text, label, prob, pred in zip(bundle.target_test.texts, bundle.target_test.labels, p_t, pred_t):
        if int(pred) != label:
            errors.append(
                {
                    "method": method_id,
                    "text": text,
                    "label": label,
                    "prediction": int(pred),
                    "prob_positive": round(float(prob), 6),
                }
            )

    return {
        "method_id": method_id,
        "method": method_name,
        "seed": seed,
        "tau": tau,
        "target_val_macro_f1": val_f1,
        "target_val_negative_recall": val_neg_recall,
        "F1_s": metrics_s["f1_macro"],
        "F1_t": metrics_t["f1_macro"],
        "deltaF1": metrics_s["f1_macro"] - metrics_t["f1_macro"],
        "AUC_s": metrics_s.get("auc", 0.0),
        "AUC_t": metrics_t.get("auc", 0.0),
        "accuracy_t": metrics_t["accuracy"],
        "recall_negative_t": metrics_t["recall_negative"],
        "recall_positive_t": metrics_t["recall_positive"],
        "weighted_f1_t": metrics_t["f1_weighted"],
        "kl_diag_gaussian": distances["kl_diag_gaussian"],
        "mmd_linear": distances["mmd_linear"],
        "train_size": len(train_texts),
        "errors": errors[:80],
    }


def pseudo_label_target(
    bundle: DatasetBundle,
    seed: int,
    vectorizer: HashingTextVectorizer,
    base_texts: list[str],
    base_labels: list[int],
    sample_weight: list[float],
    max_items: int = 1800,
) -> tuple[list[str], list[int], list[float], dict]:
    model = SparseLogisticRegression(vectorizer.n_features, lr=0.16, epochs=12, seed=seed)
    model.fit(vectorizer.transform(base_texts), base_labels, sample_weight)
    probs = model.predict_proba(vectorizer.transform(bundle.target_unlabeled.texts))
    ranked = sorted(
        enumerate(probs),
        key=lambda item: abs(float(item[1]) - 0.5),
        reverse=True,
    )
    selected = ranked[: min(max_items, len(ranked))]
    texts, labels, weights = [], [], []
    for idx, prob in selected:
        texts.append(bundle.target_unlabeled.texts[idx])
        labels.append(1 if prob >= 0.5 else 0)
        weights.append(0.45 + min(abs(float(prob) - 0.5), 0.49))
    audit = {
        "selected": len(selected),
        "positive": sum(labels),
        "negative": len(labels) - sum(labels),
        "avg_confidence_margin": float(mean(abs(float(probs[idx]) - 0.5) for idx, _ in selected)) if selected else 0.0,
    }
    return texts, labels, weights, audit


def build_methods(bundle: DatasetBundle, seed: int):
    source_texts = bundle.source_train.texts
    source_labels = bundle.source_train.labels
    target_texts = bundle.target_train.texts
    target_labels = bundle.target_train.labels
    target_weight = [2.8] * len(target_texts)
    source_weight = [1.0] * len(source_texts)

    methods = []
    v_e0 = HashingTextVectorizer(n_features=4096, ngram_range=(2, 3), use_lexicon=False)
    methods.append(("E0", "TF-IDF+LR source-only", v_e0, source_texts, source_labels, source_weight, None))

    v_e1 = HashingTextVectorizer(n_features=8192, ngram_range=(2, 4), use_lexicon=True)
    methods.append(("E1", "Source-only strong lexical encoder", v_e1, source_texts, source_labels, source_weight, 0.85))

    pseudo_texts, pseudo_labels, pseudo_weights, pseudo_audit = pseudo_label_target(
        bundle, seed, v_e1, source_texts, source_labels, source_weight
    )
    methods.append(
        (
            "E2",
            "Weak supervision pseudo labels",
            HashingTextVectorizer(n_features=8192, ngram_range=(2, 4), use_lexicon=True),
            source_texts + pseudo_texts,
            source_labels + pseudo_labels,
            source_weight + pseudo_weights,
            0.85,
        )
    )
    methods.append(
        (
            "E3",
            "Target adapter calibration",
            HashingTextVectorizer(n_features=8192, ngram_range=(2, 4), use_lexicon=True),
            source_texts + target_texts,
            source_labels + target_labels,
            source_weight + target_weight,
            0.85,
        )
    )
    methods.append(
        (
            "E4",
            "Weak supervision + adapter",
            HashingTextVectorizer(n_features=8192, ngram_range=(2, 4), use_lexicon=True),
            source_texts + target_texts + pseudo_texts,
            source_labels + target_labels + pseudo_labels,
            source_weight + target_weight + pseudo_weights,
            0.85,
        )
    )
    drop_tokens = domain_specific_tokens(source_texts, target_texts + bundle.target_unlabeled.texts, top_k=420)
    methods.append(
        (
            "E5",
            "Domain-adversarial feature filtering",
            HashingTextVectorizer(n_features=8192, ngram_range=(2, 4), use_lexicon=True, drop_tokens=drop_tokens),
            source_texts + target_texts + pseudo_texts,
            source_labels + target_labels + pseudo_labels,
            source_weight + target_weight + pseudo_weights,
            0.85,
        )
    )
    methods.append(
        (
            "E6",
            "Backbone replacement char1-5",
            HashingTextVectorizer(n_features=12288, ngram_range=(1, 5), use_lexicon=True),
            source_texts + target_texts + pseudo_texts,
            source_labels + target_labels + pseudo_labels,
            source_weight + target_weight + pseudo_weights,
            0.85,
        )
    )
    return methods, pseudo_audit, len(drop_tokens)


def aggregate(results: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        grouped[row["method_id"]].append(row)
    summary = []
    metrics = [
        "F1_s", "F1_t", "deltaF1", "AUC_t", "accuracy_t",
        "recall_negative_t", "recall_positive_t", "weighted_f1_t",
        "kl_diag_gaussian", "mmd_linear",
    ]
    for method_id in sorted(grouped):
        rows = grouped[method_id]
        item = {"method_id": method_id, "method": rows[0]["method"], "n_runs": len(rows)}
        for key in metrics:
            values = [float(r[key]) for r in rows]
            item[f"{key}_mean"] = mean(values)
            item[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
        item["tau_mean"] = mean(float(r["tau"]) for r in rows)
        item["train_size_mean"] = mean(float(r["train_size"]) for r in rows)
        summary.append(item)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def draw_chart(path: Path, summary: list[dict]) -> None:
    width, height = 1200, 680
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 32)
        font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 20)
        small = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 16)
    except Exception:
        title_font = font = small = None
    draw.text((42, 30), "Final Cross-domain Sentiment Experiments", fill="#111827", font=title_font)
    metrics = [
        ("F1_t_mean", "Target Macro-F1", "#2563eb"),
        ("recall_negative_t_mean", "Negative Recall", "#16a34a"),
        ("AUC_t_mean", "Target AUC", "#f59e0b"),
    ]
    left, top, chart_h = 80, 120, 390
    group_w = 145
    bar_w = 32
    for i, row in enumerate(summary):
        gx = left + i * group_w
        draw.text((gx + 10, top + chart_h + 22), row["method_id"], fill="#111827", font=font)
        for j, (key, _, color) in enumerate(metrics):
            value = max(0.0, min(1.0, float(row[key])))
            h = int(chart_h * value)
            x = gx + j * (bar_w + 4)
            draw.rectangle((x, top + chart_h - h, x + bar_w, top + chart_h), fill=color)
            draw.text((x - 3, top + chart_h - h - 22), f"{value:.2f}", fill="#111827", font=small)
    draw.line((left - 20, top + chart_h, width - 80, top + chart_h), fill="#94a3b8", width=2)
    for j, (_, label, color) in enumerate(metrics):
        y = 590 + j * 26
        draw.rectangle((42, y, 64, y + 16), fill=color)
        draw.text((76, y - 3), label, fill="#111827", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def dataset_summary(bundle: DatasetBundle) -> dict:
    def desc(split: SplitData) -> dict:
        n = len(split)
        pos = sum(split.labels)
        neg = n - pos
        lengths = [len(t) for t in split.texts]
        return {
            "n": n,
            "positive": pos,
            "negative": neg,
            "positive_rate": pos / n if n else 0.0,
            "avg_chars": sum(lengths) / n if n else 0.0,
        }

    return {name: desc(getattr(bundle, name)) for name in bundle.__dataclass_fields__}


def run_final_experiments(
    source_path: str = "data/processed/source_full.csv",
    target_path: str = "data/processed/social_full.csv",
    output_dir: str = "results",
    seeds: tuple[int, ...] = (42, 123, 456),
) -> tuple[dict, Path]:
    bundle = load_bundle(source_path, target_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    all_errors: list[dict] = []
    pseudo_audits = {}
    dropped_feature_counts = {}

    for seed in seeds:
        methods, pseudo_audit, dropped_count = build_methods(bundle, seed)
        pseudo_audits[str(seed)] = pseudo_audit
        dropped_feature_counts[str(seed)] = dropped_count
        for method_id, method_name, vectorizer, texts, labels, weights, min_neg in methods:
            row = train_and_eval(method_id, method_name, bundle, seed, vectorizer, texts, labels, weights, min_neg)
            errors = row.pop("errors")
            all_errors.extend({"seed": seed, **err} for err in errors)
            all_results.append(row)
            print(
                f"{method_id} seed={seed} F1_t={row['F1_t']:.3f} "
                f"neg_recall={row['recall_negative_t']:.3f} deltaF1={row['deltaF1']:.3f}"
            )

    summary = aggregate(all_results)
    payload = {
        "source_path": source_path,
        "target_path": target_path,
        "seeds": list(seeds),
        "dataset": dataset_summary(bundle),
        "pseudo_label_audit": pseudo_audits,
        "domain_filtered_token_count": dropped_feature_counts,
        "runs": all_results,
        "summary": summary,
    }
    metrics_path = output / "final_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "final_runs.csv", [{k: v for k, v in row.items() if not isinstance(v, list)} for row in all_results])
    write_csv(output / "final_summary.csv", summary)
    write_csv(output / "final_error_cases.csv", all_errors[:300])
    draw_chart(output / "final_metrics_chart.png", summary)
    return payload, metrics_path
