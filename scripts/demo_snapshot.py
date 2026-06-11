"""打印系统演示用的精简状态摘要.

该脚本不重跑实验，只读取最终交付产物，用于 3 分钟视频演示时快速展示：
- 数据规模与 split
- E0-E6 是否完整
- 最佳目标域 F1 与最平衡 DeltaF1
- 核心报告/图表文件是否存在
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fmt(x: float) -> str:
    return f"{float(x):.3f}"


def exists(path: str) -> str:
    return "OK" if (ROOT / path).exists() else "MISSING"


def main() -> None:
    audit = json.loads((ROOT / "results/final_data_audit.json").read_text(encoding="utf-8"))
    metrics = json.loads((ROOT / "results/final_metrics.json").read_text(encoding="utf-8"))
    summary = metrics["summary"]
    best_target = max(summary, key=lambda row: row["F1_t_mean"])
    best_delta = min(summary, key=lambda row: abs(row["deltaF1_mean"]))
    methods = ", ".join(row["method_id"] for row in summary)

    target_splits = audit["target"]["splits"]
    source_splits = audit["source"]["splits"]

    print("=" * 72)
    print("Cross-domain Sentiment Final Demo Snapshot")
    print("=" * 72)
    print(
        "Source data:",
        audit["source"]["sampled_rows"],
        "rows | train/val/test =",
        f"{source_splits['train']['n']}/{source_splits['val']['n']}/{source_splits['test']['n']}",
    )
    print(
        "Target data:",
        audit["target"]["sampled_rows"],
        "rows | train/val/test/unlabeled =",
        f"{target_splits['train']['n']}/{target_splits['val']['n']}/"
        f"{target_splits['test']['n']}/{target_splits['unlabeled']['n']}",
    )
    print("Experiment matrix:", methods, "| seeds =", ", ".join(map(str, metrics["seeds"])))
    print(
        "Best target F1:",
        best_target["method_id"],
        best_target["method"],
        "| F1_t =",
        fmt(best_target["F1_t_mean"]),
        "| NegRecall =",
        fmt(best_target["recall_negative_t_mean"]),
    )
    print(
        "Most balanced DeltaF1:",
        best_delta["method_id"],
        best_delta["method"],
        "| DeltaF1 =",
        fmt(best_delta["deltaF1_mean"]),
        "| F1_t =",
        fmt(best_delta["F1_t_mean"]),
    )
    print("Artifacts:")
    for path in [
        "results/final_summary.csv",
        "results/final_metrics_chart.png",
        "results/final_error_cases.csv",
        "report/final_report.pdf",
        "presentation/final_defense_7min.pptx",
    ]:
        print(f"  [{exists(path)}] {path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
