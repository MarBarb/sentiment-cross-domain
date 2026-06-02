"""最终交付验收脚本.

检查：
- 真实 processed 数据已生成，目标域总量 >= 5k 且包含 unlabeled split。
- final_metrics.json 包含 E0-E6，每个方法 3 个随机种子。
- 至少一个方法达到 DeltaF1 <= 15% 且目标域负面召回率 >= 85%。
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def count_splits(path: Path) -> Counter:
    counts = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            counts[row["split"]] += 1
    return counts


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAILED: {message}")
    print(f"OK: {message}")


def main() -> None:
    source = ROOT / "data/processed/source_full.csv"
    target = ROOT / "data/processed/social_full.csv"
    metrics_path = ROOT / "results/final_metrics.json"
    summary_path = ROOT / "results/final_summary.csv"
    report_md = ROOT / "report/final_report.md"

    require(source.exists(), "source_full.csv exists")
    require(target.exists(), "social_full.csv exists")
    require(metrics_path.exists(), "final_metrics.json exists")
    require(summary_path.exists(), "final_summary.csv exists")

    source_counts = count_splits(source)
    target_counts = count_splits(target)
    require(sum(source_counts.values()) >= 8000, f"source rows >= 8000 ({sum(source_counts.values())})")
    require(sum(target_counts.values()) >= 5000, f"target rows >= 5000 ({sum(target_counts.values())})")
    require(target_counts["unlabeled"] >= 3000, f"target unlabeled >= 3000 ({target_counts['unlabeled']})")
    require(target_counts["train"] > 0 and target_counts["val"] > 0 and target_counts["test"] > 0, "target train/val/test present")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    runs_by_method = defaultdict(list)
    for run in payload["runs"]:
        runs_by_method[run["method_id"]].append(run)
    require(set(runs_by_method) == {f"E{i}" for i in range(7)}, "E0-E6 all present")
    require(all(len(rows) == 3 for rows in runs_by_method.values()), "each method has 3 seeds")

    best = min(payload["summary"], key=lambda row: abs(row["deltaF1_mean"]))
    eligible = [
        row for row in payload["summary"]
        if abs(row["deltaF1_mean"]) <= 0.15 and row["recall_negative_t_mean"] >= 0.85
    ]
    require(bool(eligible), "at least one method meets |DeltaF1| <= 0.15 and negative recall >= 0.85")
    print(
        "BEST:",
        best["method_id"],
        f"F1_t={best['F1_t_mean']:.3f}",
        f"DeltaF1={best['deltaF1_mean']:.3f}",
        f"NegRecall={best['recall_negative_t_mean']:.3f}",
    )

    if report_md.exists():
        require("E0-E6" in report_md.read_text(encoding="utf-8"), "final report mentions E0-E6")


if __name__ == "__main__":
    main()
