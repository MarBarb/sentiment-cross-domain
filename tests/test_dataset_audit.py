"""Tests for final dataset audit and unlabeled-split safeguards."""
from __future__ import annotations

import csv

from src.evaluation.dataset_audit import audit_cross_domain_dataset, read_processed_rows
from src.experiments.final_runner import HashingTextVectorizer, SplitData, pseudo_label_target


def _write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "split", "domain"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _source_rows():
    return [
        {"text": "服务好", "label": 1, "split": "train", "domain": "source"},
        {"text": "味道差", "label": 0, "split": "train", "domain": "source"},
        {"text": "配送快", "label": 1, "split": "val", "domain": "source"},
        {"text": "体验差", "label": 0, "split": "val", "domain": "source"},
        {"text": "很满意", "label": 1, "split": "test", "domain": "source"},
        {"text": "很失望", "label": 0, "split": "test", "domain": "source"},
    ]


def _target_rows(include_unlabeled=True, unlabeled_label=0):
    rows = [
        {"text": "微博支持", "label": 1, "split": "train", "domain": "target"},
        {"text": "微博投诉", "label": 0, "split": "train", "domain": "target"},
        {"text": "网友开心", "label": 1, "split": "val", "domain": "target"},
        {"text": "网友愤怒", "label": 0, "split": "val", "domain": "target"},
        {"text": "评论温暖", "label": 1, "split": "test", "domain": "target"},
        {"text": "评论糟糕", "label": 0, "split": "test", "domain": "target"},
    ]
    if include_unlabeled:
        rows.extend(
            [
                {"text": "未标注支持", "label": unlabeled_label, "split": "unlabeled", "domain": "target"},
                {"text": "未标注投诉", "label": unlabeled_label, "split": "unlabeled", "domain": "target"},
                {"text": "未标注开心", "label": unlabeled_label, "split": "unlabeled", "domain": "target"},
            ]
        )
    return rows


def test_audit_passes_expected_cross_domain_contract(tmp_path):
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    _write_rows(source, _source_rows())
    _write_rows(target, _target_rows(unlabeled_label=1))

    report = audit_cross_domain_dataset(
        source,
        target,
        min_source_rows=6,
        min_target_rows=9,
        min_target_unlabeled=3,
    )

    assert report["passed"] is True
    assert report["source"]["splits"]["train"]["rows"] == 2
    assert report["target"]["splits"]["unlabeled"]["rows"] == 3
    assert "training code must use unlabeled text only" in report["notes"][0]


def test_audit_fails_when_unlabeled_split_is_missing(tmp_path):
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    _write_rows(source, _source_rows())
    _write_rows(target, _target_rows(include_unlabeled=False))

    report = audit_cross_domain_dataset(
        source,
        target,
        min_source_rows=6,
        min_target_rows=6,
        min_target_unlabeled=1,
    )

    assert report["passed"] is False
    assert any(check["message"] == "target split present: unlabeled" and not check["ok"] for check in report["checks"])


def test_blank_unlabeled_labels_do_not_count_as_evaluated_label_errors(tmp_path):
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    target_rows = _target_rows(unlabeled_label="")
    _write_rows(source, _source_rows())
    _write_rows(target, target_rows)

    rows, issues = read_processed_rows(target)
    report = audit_cross_domain_dataset(
        source,
        target,
        min_source_rows=6,
        min_target_rows=9,
        min_target_unlabeled=3,
    )

    assert issues == []
    assert sum(1 for row in rows if row["split"] == "unlabeled" and row["label"] is None) == 3
    assert report["passed"] is True
    assert report["target"]["splits"]["unlabeled"]["missing_label"] == 3


class _Bundle:
    def __init__(self, unlabeled_labels):
        self.target_unlabeled = SplitData(
            texts=["微博支持开心", "微博投诉糟糕", "微博服务不错"],
            labels=list(unlabeled_labels),
        )


def test_pseudo_labeling_ignores_public_unlabeled_labels():
    base_texts = ["服务很好", "体验糟糕", "非常满意", "非常失望"]
    base_labels = [1, 0, 1, 0]
    weights = [1.0] * len(base_labels)
    vectorizer_a = HashingTextVectorizer(n_features=256, ngram_range=(1, 2), use_lexicon=True)
    vectorizer_b = HashingTextVectorizer(n_features=256, ngram_range=(1, 2), use_lexicon=True)

    result_a = pseudo_label_target(_Bundle([0, 0, 0]), 42, vectorizer_a, base_texts, base_labels, weights)
    result_b = pseudo_label_target(_Bundle([1, 1, 1]), 42, vectorizer_b, base_texts, base_labels, weights)

    assert result_a == result_b

