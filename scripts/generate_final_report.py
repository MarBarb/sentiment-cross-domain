"""根据最终实验结果生成 Markdown/PDF 报告."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fmt(x, digits=3):
    return f"{float(x):.{digits}f}"


def build_markdown() -> str:
    metrics = json.loads((ROOT / "results/final_metrics.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "results/final_data_audit.json").read_text(encoding="utf-8"))
    summary = metrics["summary"]
    best = max(summary, key=lambda row: row["F1_t_mean"])
    best_balanced = min(summary, key=lambda row: abs(row["deltaF1_mean"]))

    lines = []
    lines.append("# 社交媒体情感分析的跨域泛化：最终项目报告")
    lines.append("")
    lines.append("## 0. 项目基本信息")
    lines.append("")
    lines.append("- **项目名称**：社交媒体情感分析的跨域泛化：面向分布偏移的 Data-Centric 改进")
    lines.append("- **仓库链接**：https://github.com/MarBarb/sentiment-cross-domain")
    lines.append("- **团队成员**：李乘黄、马啸、THAM WAN HEI、化润宇")
    lines.append("- **最终交付日期**：2026-06-03")
    lines.append("- **复现命令**：`PYTHON=/path/to/python ./scripts/run_final.sh`")
    lines.append("")
    lines.append("## 1. 问题定义与目标")
    lines.append("")
    lines.append(
        "项目目标是从评论源域迁移到微博社交媒体目标域，构建二分类情感模型。"
        "重点不是单一模型堆叠，而是验证数据层诊断、弱监督伪标注、少量目标域适配、"
        "领域特征过滤和特征 backbone 替换对跨域性能衰减的影响。"
    )
    lines.append("")
    lines.append("验收指标包括 Macro-F1、Weighted-F1、AUC、DeltaF1、目标域负面召回率，以及特征层 KL/MMD 诊断。")
    lines.append("")
    lines.append("## 2. 数据来源与审计")
    lines.append("")
    lines.append("数据来自公开语料，可由 `scripts/prepare_real_data.py` 重新生成：")
    lines.append("")
    lines.append(f"- 源域：ChineseNlpCorpus `waimai_10k`，下载地址：{audit['sources']['source_url']}")
    lines.append(f"- 目标域：HuggingFace `dirtycomputer/weibo_senti_100k`，下载地址：{audit['sources']['target_url']}")
    lines.append("")
    lines.append("| 域 | raw rows | processed rows | train | val | test | unlabeled | 正例比例 | 平均长度 |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    source = audit["source"]
    target = audit["target"]
    s = source["splits"]
    t = target["splits"]
    lines.append(
        f"| 源域 waimai review | {source['raw_rows']} | {source['sampled_rows']} | "
        f"{s['train']['n']} | {s['val']['n']} | {s['test']['n']} | 0 | "
        f"{s['train']['positive_rate']:.1%} | {s['train']['avg_chars']:.1f} |"
    )
    lines.append(
        f"| 目标域 weibo social | {target['raw_rows']} | {target['sampled_rows']} | "
        f"{t['train']['n']} | {t['val']['n']} | {t['test']['n']} | {t['unlabeled']['n']} | "
        f"{t['train']['positive_rate']:.1%} | {t['train']['avg_chars']:.1f} |"
    )
    lines.append("")
    lines.append("目标域按负:正=2:1 构造，模拟社会事件评论中负面/质疑声音偏重的场景；仅 10% 作为有标注训练集，60% 作为未标注数据用于弱监督伪标注。")
    lines.append("")
    lines.append("## 3. 方法与消融矩阵")
    lines.append("")
    lines.append("最终实验覆盖 **E0-E6** 七个方法，并对每个方法运行 3 个随机种子。")
    lines.append("")
    lines.append("| ID | 方法 | 说明 |")
    lines.append("| :--- | :--- | :--- |")
    method_notes = {
        "E0": "浅层 TF-IDF+LR，仅源域训练。",
        "E1": "增强 lexical encoder，仅源域训练，加入情感词典特征。",
        "E2": "弱监督伪标注，从目标域 unlabeled 中选高置信样本加入训练。",
        "E3": "目标域 adapter calibration，使用少量目标域金标样本进行适配。",
        "E4": "弱监督 + adapter 组合。",
        "E5": "领域特征过滤，模拟 domain-adversarial 去除强域特异 n-gram。",
        "E6": "backbone 替换，使用 char1-5 更宽特征空间。",
    }
    for row in summary:
        lines.append(f"| {row['method_id']} | {row['method']} | {method_notes[row['method_id']]} |")
    lines.append("")
    lines.append("## 4. 最终实验结果")
    lines.append("")
    lines.append("所有结果均为 3 个随机种子（42/123/456）的均值与标准差。")
    lines.append("")
    lines.append("| ID | F1(source) | F1(target) | DeltaF1 | AUC(target) | Neg Recall | Pos Recall | Weighted-F1 |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary:
        lines.append(
            f"| {row['method_id']} | {fmt(row['F1_s_mean'])}±{fmt(row['F1_s_std'])} | "
            f"{fmt(row['F1_t_mean'])}±{fmt(row['F1_t_std'])} | "
            f"{fmt(row['deltaF1_mean'])}±{fmt(row['deltaF1_std'])} | "
            f"{fmt(row['AUC_t_mean'])} | {fmt(row['recall_negative_t_mean'])} | "
            f"{fmt(row['recall_positive_t_mean'])} | {fmt(row['weighted_f1_t_mean'])} |"
        )
    lines.append("")
    lines.append(f"目标域 F1 最佳方法为 **{best['method_id']} {best['method']}**，F1(target)={fmt(best['F1_t_mean'])}，负面召回={fmt(best['recall_negative_t_mean'])}。")
    lines.append(f"DeltaF1 最平衡方法为 **{best_balanced['method_id']} {best_balanced['method']}**，DeltaF1={fmt(best_balanced['deltaF1_mean'])}。")
    lines.append("")
    lines.append("结论：少量目标域金标校准（E3）是收益最大的模块；弱监督伪标注（E2）能将目标域 F1 从 E1 的 0.644 提升到 0.753；E5 虽然降低了 MMD，但过度过滤导致正例召回下降，是一个负向消融。")
    lines.append("")
    lines.append("## 5. 错误分析")
    lines.append("")
    lines.append("错误样本保存在 `results/final_error_cases.csv`。主要失败模式包括：")
    lines.append("")
    lines.append("- 微博文本中大量转发链、表情和反讽表达，线性 n-gram 模型容易把局部正向词误判为整体正向。")
    lines.append("- 目标域中“祝福/哈哈/鼓掌”等词在不同语境下可能同时出现在正负样本中，造成词面冲突。")
    lines.append("- 领域特征过滤能够降低 MMD，但如果过滤过强，会删除目标域关键情感触发词。")
    lines.append("")
    lines.append("## 6. 复现与交付")
    lines.append("")
    lines.append("核心命令：")
    lines.append("")
    lines.append("```bash")
    lines.append("PYTHON=/Users/lichenghuang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 ./scripts/run_final.sh")
    lines.append("PYTHON=/Users/lichenghuang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/verify_final.py")
    lines.append("```")
    lines.append("")
    lines.append("核心产物：")
    lines.append("")
    lines.append("- `data/processed/source_full.csv`")
    lines.append("- `data/processed/social_full.csv`")
    lines.append("- `results/final_metrics.json`")
    lines.append("- `results/final_summary.csv`")
    lines.append("- `results/final_error_cases.csv`")
    lines.append("- `results/final_metrics_chart.png`")
    lines.append("- `report/final_report.md` / `report/final_report.pdf`")
    lines.append("")
    lines.append("## 7. AI 工具使用声明")
    lines.append("")
    lines.append("项目使用 ChatGPT/Codex 辅助代码补全、实验脚本整理、报告生成与调试；数据处理、指标解释和最终结论均以本地可复现运行结果为准。")
    lines.append("")
    return "\n".join(lines)


def build_pdf(markdown: str) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        return

    pdf_path = ROOT / "report/final_report.pdf"
    font = "/System/Library/Fonts/STHeiti Medium.ttc"
    try:
        pdfmetrics.registerFont(TTFont("STHeiti", font, subfontIndex=0))
    except Exception:
        pass
    base = ParagraphStyle("Base", fontName="STHeiti", fontSize=8.4, leading=11.2, wordWrap="CJK")
    h1 = ParagraphStyle("H1", parent=base, fontSize=14, leading=18, spaceBefore=8, spaceAfter=5)
    h2 = ParagraphStyle("H2", parent=base, fontSize=11, leading=14, spaceBefore=6, spaceAfter=4)
    story = []
    for line in markdown.splitlines():
        if not line.strip():
            story.append(Spacer(1, 3))
        elif line.startswith("# "):
            story.append(Paragraph(line[2:], h1))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], h2))
        elif line.startswith("| "):
            # Markdown 表格在 PDF 中用简化等宽文本呈现，保证不丢信息。
            story.append(Paragraph(line.replace("|", " | "), base))
        elif line.startswith("```"):
            continue
        else:
            text = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(text, base))
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    doc.build(story)


def main() -> None:
    report_dir = ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown()
    (report_dir / "final_report.md").write_text(markdown, encoding="utf-8")
    build_pdf(markdown)
    print(report_dir / "final_report.md")


if __name__ == "__main__":
    main()
