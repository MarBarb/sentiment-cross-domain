"""本地 CSV 数据加载工具.

中期阶段先保证仓库在无外部数据服务时也能复现实验。CSV 需要包含：
`text,label,split,domain` 四列，其中 label 取 0/1，split 取 train/val/test。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .cleaner import TextCleaner


@dataclass
class TextSplit:
    texts: list[str]
    labels: list[int]
    domains: list[str]

    def __len__(self) -> int:
        return len(self.texts)


@dataclass
class CrossDomainSplits:
    source_train: TextSplit
    source_test: TextSplit
    target_train: TextSplit
    target_val: TextSplit
    target_test: TextSplit


def _read_csv(path: str | Path, cleaner: TextCleaner) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"text", "label", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} 缺少列: {sorted(missing)}")

        for i, row in enumerate(reader, 2):
            text = cleaner.clean(row.get("text", ""))
            if cleaner.is_noise(text):
                continue
            try:
                label = int(row.get("label", ""))
            except ValueError as exc:
                raise ValueError(f"{path}:{i} label 不是整数: {row.get('label')}") from exc
            if label not in (0, 1):
                raise ValueError(f"{path}:{i} label 必须为 0/1，实际为 {label}")
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "split": row.get("split", "train").strip().lower(),
                    "domain": row.get("domain", path.stem).strip() or path.stem,
                }
            )
    return rows


def _select(rows: Sequence[dict], split: str) -> TextSplit:
    selected = [r for r in rows if r["split"] == split]
    return TextSplit(
        texts=[r["text"] for r in selected],
        labels=[r["label"] for r in selected],
        domains=[r["domain"] for r in selected],
    )


def load_cross_domain_splits(
    source_path: str | Path = "data/processed/source_sample.csv",
    target_path: str | Path = "data/processed/social_sample.csv",
    cleaner: TextCleaner | None = None,
) -> CrossDomainSplits:
    """加载源域和目标域的本地样例数据."""
    cleaner = cleaner or TextCleaner()
    source_rows = _read_csv(source_path, cleaner)
    target_rows = _read_csv(target_path, cleaner)

    splits = CrossDomainSplits(
        source_train=_select(source_rows, "train"),
        source_test=_select(source_rows, "test"),
        target_train=_select(target_rows, "train"),
        target_val=_select(target_rows, "val"),
        target_test=_select(target_rows, "test"),
    )
    for name, split in vars(splits).items():
        if len(split) == 0:
            raise ValueError(f"{name} 为空，请检查 CSV 的 split 列")
    return splits


def describe_split(split: TextSplit) -> dict:
    n = len(split)
    pos = sum(split.labels)
    neg = n - pos
    lengths = [len(t) for t in split.texts]
    return {
        "n": n,
        "positive": pos,
        "negative": neg,
        "pos_rate": pos / n if n else 0.0,
        "avg_chars": sum(lengths) / n if n else 0.0,
    }


def describe_splits(splits: CrossDomainSplits) -> dict:
    return {name: describe_split(split) for name, split in vars(splits).items()}

