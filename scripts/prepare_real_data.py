"""下载并生成最终实验用的真实公开数据 split.

源域：ChineseNlpCorpus `waimai_10k` 外卖评论。
目标域：HuggingFace `dirtycomputer/weibo_senti_100k` 微博情感数据。

输出：
- data/processed/source_full.csv
- data/processed/social_full.csv
- results/final_data_audit.json
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cleaner import TextCleaner


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

SOURCE_URL = (
    "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/"
    "datasets/waimai_10k/waimai_10k.csv"
)
TARGET_URL = (
    "https://huggingface.co/datasets/dirtycomputer/weibo_senti_100k/"
    "resolve/main/weibo_senti_100k.csv"
)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        return
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("需要 curl 下载公开数据。")
    subprocess.run([curl, "-L", "--fail", url, "-o", str(dest)], check=True)


def read_labeled_csv(path: Path) -> list[dict]:
    rows = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                fieldnames = {name.strip().lower(): name for name in reader.fieldnames}
                label_col = fieldnames.get("label")
                text_col = fieldnames.get("review") or fieldnames.get("text")
                if not label_col or not text_col:
                    raise ValueError(f"{path} 缺少 label/review 字段，实际字段为 {reader.fieldnames}")
                for row in reader:
                    rows.append({"label": int(row[label_col]), "text": row[text_col]})
            return rows
        except UnicodeDecodeError:
            rows.clear()
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"无法解码 {path}")


def clean_rows(rows: list[dict], domain: str, cleaner: TextCleaner) -> list[dict]:
    seen = set()
    cleaned = []
    for row in rows:
        text = cleaner.clean(row["text"])
        if cleaner.is_noise(text):
            continue
        label = int(row["label"])
        if label not in (0, 1):
            continue
        key = (text, label)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"text": text, "label": label, "domain": domain})
    return cleaned


def stratified_sample(rows: list[dict], total: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for row in rows:
        by_label[row["label"]].append(row)
    for values in by_label.values():
        rng.shuffle(values)
    if total >= len(rows):
        out = rows[:]
        rng.shuffle(out)
        return out
    n0 = min(len(by_label[0]), total // 2)
    n1 = min(len(by_label[1]), total - n0)
    # 如果某类不足，另一类补齐。
    if n0 + n1 < total:
        if len(by_label[0]) - n0 > len(by_label[1]) - n1:
            n0 = min(len(by_label[0]), total - n1)
        else:
            n1 = min(len(by_label[1]), total - n0)
    out = by_label[0][:n0] + by_label[1][:n1]
    rng.shuffle(out)
    return out


def stratified_sample_counts(rows: list[dict], counts: dict[int, int], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for row in rows:
        by_label[row["label"]].append(row)
    out = []
    for label, count in counts.items():
        values = by_label[label]
        rng.shuffle(values)
        out.extend(values[: min(count, len(values))])
    rng.shuffle(out)
    return out


def split_rows(
    rows: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    test_ratio: float | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for row in rows:
        by_label[row["label"]].append(row)
    output = []
    for label, values in by_label.items():
        rng.shuffle(values)
        n = len(values)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio) if test_ratio is not None else n - n_train - n_val
        for i, row in enumerate(values):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            elif i < n_train + n_val + n_test:
                split = "test"
            else:
                split = "unlabeled"
            item = row.copy()
            item["split"] = split
            output.append(item)
    rng.shuffle(output)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "split", "domain"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def describe(rows: list[dict]) -> dict:
    by_split = {}
    for split in sorted({r["split"] for r in rows}):
        values = [r for r in rows if r["split"] == split]
        labels = Counter(r["label"] for r in values)
        lengths = [len(r["text"]) for r in values]
        by_split[split] = {
            "n": len(values),
            "positive": labels[1],
            "negative": labels[0],
            "positive_rate": labels[1] / len(values) if values else 0.0,
            "avg_chars": sum(lengths) / len(lengths) if lengths else 0.0,
        }
    return by_split


def main() -> None:
    seed = 42
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    source_raw = RAW_DIR / "waimai_10k.csv"
    target_raw = RAW_DIR / "weibo_senti_100k.csv"
    download(SOURCE_URL, source_raw)
    download(TARGET_URL, target_raw)

    cleaner = TextCleaner()
    source_all = clean_rows(read_labeled_csv(source_raw), "waimai_review", cleaner)
    target_all = clean_rows(read_labeled_csv(target_raw), "weibo_social", cleaner)

    # 控制仓库和实验体量，同时满足目标域 5k+ 的最终要求。
    source = stratified_sample(source_all, total=8000, seed=seed)
    # 目标域保持负面偏重，贴近社会事件舆情场景（负:正 = 2:1）。
    target = stratified_sample_counts(target_all, counts={0: 4000, 1: 2000}, seed=seed + 1)

    source_split = split_rows(source, train_ratio=0.70, val_ratio=0.15, seed=seed)
    target_split = split_rows(
        target,
        train_ratio=0.10,
        val_ratio=0.10,
        test_ratio=0.20,
        seed=seed + 1,
    )

    source_path = PROCESSED_DIR / "source_full.csv"
    target_path = PROCESSED_DIR / "social_full.csv"
    write_csv(source_path, source_split)
    write_csv(target_path, target_split)

    audit = {
        "source": {
            "raw_path": str(source_raw.relative_to(ROOT)),
            "raw_rows": len(source_all),
            "processed_path": str(source_path.relative_to(ROOT)),
            "sampled_rows": len(source_split),
            "splits": describe(source_split),
        },
        "target": {
            "raw_path": str(target_raw.relative_to(ROOT)),
            "raw_rows": len(target_all),
            "processed_path": str(target_path.relative_to(ROOT)),
            "sampled_rows": len(target_split),
            "splits": describe(target_split),
        },
        "seed": seed,
        "sources": {
            "source_url": SOURCE_URL,
            "target_url": TARGET_URL,
        },
    }
    (RESULTS_DIR / "final_data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
