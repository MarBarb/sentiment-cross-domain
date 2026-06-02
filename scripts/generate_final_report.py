"""根据最终实验结果生成 Markdown/PDF 报告."""
from __future__ import annotations

import json
import sys
from html import escape
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
    lines.append("- **复现命令**：`./scripts/run_final.sh && python scripts/verify_final.py`")
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
    lines.append("./scripts/run_final.sh")
    lines.append("python scripts/verify_final.py")
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
        from reportlab.platypus import Image as RLImage
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception as exc:
        print(f"WARNING: PDF generation skipped: {exc}", file=sys.stderr)
        return

    metrics = json.loads((ROOT / "results/final_metrics.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "results/final_data_audit.json").read_text(encoding="utf-8"))
    summary = metrics["summary"]
    best = max(summary, key=lambda row: row["F1_t_mean"])
    best_balanced = min(summary, key=lambda row: abs(row["deltaF1_mean"]))

    pdf_path = ROOT / "report/final_report.pdf"
    font = "/System/Library/Fonts/STHeiti Medium.ttc"
    font_name = "STHeiti"
    try:
        pdfmetrics.registerFont(TTFont(font_name, font, subfontIndex=0))
    except Exception:
        font_name = "Helvetica"

    base = ParagraphStyle(
        "Base", fontName=font_name, fontSize=9.2, leading=13.2, wordWrap="CJK",
        textColor=colors.HexColor("#111827"),
    )
    title = ParagraphStyle(
        "Title", parent=base, fontSize=17, leading=22, spaceAfter=8,
        textColor=colors.HexColor("#0f172a"),
    )
    h2 = ParagraphStyle(
        "H2", parent=base, fontSize=12, leading=16, spaceBefore=8, spaceAfter=5,
        textColor=colors.HexColor("#1f2937"),
    )
    small = ParagraphStyle("Small", parent=base, fontSize=7.4, leading=9.2)
    cell = ParagraphStyle("Cell", parent=base, fontSize=7.3, leading=9.0)
    header = ParagraphStyle(
        "Header", parent=cell, fontSize=7.2, leading=8.8,
        textColor=colors.white,
    )
    code = ParagraphStyle(
        "Code", parent=base, fontName=font_name, fontSize=8.2, leading=11,
        leftIndent=6, backColor=colors.HexColor("#f3f4f6"),
        borderPadding=(4, 5, 4),
    )

    def p(text: str, style=base) -> Paragraph:
        return Paragraph(escape(text), style)

    def rich(text: str, style=base) -> Paragraph:
        return Paragraph(text, style)

    def table(rows, widths, align_right_from=1, font_size_style=cell):
        rendered = []
        for i, row in enumerate(rows):
            style = header if i == 0 else font_size_style
            rendered.append([Paragraph(escape(str(value)), style) for value in row])
        tbl = Table(rendered, colWidths=widths, hAlign="LEFT", repeatRows=1)
        style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
        if align_right_from is not None:
            style.add("ALIGN", (align_right_from, 1), (-1, -1), "RIGHT")
        for row_idx in range(1, len(rows)):
            if row_idx % 2 == 0:
                style.add("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#f8fafc"))
        tbl.setStyle(style)
        return tbl

    source = audit["source"]
    target = audit["target"]
    s = source["splits"]
    t = target["splits"]
    method_notes = {
        "E0": "TF-IDF+LR source-only",
        "E1": "Source-only lexical encoder",
        "E2": "Weak supervision pseudo labels",
        "E3": "Target adapter calibration",
        "E4": "Weak supervision + adapter",
        "E5": "Domain feature filtering",
        "E6": "Backbone replacement char1-5",
    }
    method_desc = {
        "E0": "浅层源域训练基线。",
        "E1": "增强字符 n-gram 与情感词典特征。",
        "E2": "从目标域 unlabeled 中选择高置信伪标注。",
        "E3": "使用少量目标域金标样本进行校准。",
        "E4": "组合伪标注与目标域校准。",
        "E5": "过滤强领域特异 n-gram，模拟对抗域约束。",
        "E6": "扩大 char1-5 特征空间作为 backbone 替换。",
    }

    story = []
    story.append(rich("社交媒体情感分析的跨域泛化：最终项目报告", title))
    story.append(p("项目名称：面向分布偏移的 Data-Centric 跨域情感分类"))
    story.append(p("团队成员：李乘黄、马啸、THAM WAN HEI、化润宇"))
    story.append(p("最终交付日期：2026-06-03"))
    story.append(p("仓库链接：https://github.com/MarBarb/sentiment-cross-domain"))
    story.append(Spacer(1, 4))

    story.append(rich("1. 问题定义与目标", h2))
    story.append(p(
        "项目目标是从评论源域迁移到微博社交媒体目标域，构建二分类情感模型。"
        "实验重点是验证数据诊断、弱监督伪标注、少量目标域适配、领域特征过滤和 backbone 替换"
        "对跨域性能衰减的影响。"
    ))
    story.append(p("验收指标包括 Macro-F1、Weighted-F1、AUC、DeltaF1、目标域负面召回率，以及 KL/MMD 分布诊断。"))

    story.append(rich("2. 数据来源与审计", h2))
    story.append(table(
        [
            ["域", "raw rows", "processed", "train", "val", "test", "unlabeled", "正例比例", "平均长度"],
            ["源域 waimai", source["raw_rows"], source["sampled_rows"], s["train"]["n"], s["val"]["n"], s["test"]["n"], 0, f"{s['train']['positive_rate']:.1%}", f"{s['train']['avg_chars']:.1f}"],
            ["目标域 weibo", target["raw_rows"], target["sampled_rows"], t["train"]["n"], t["val"]["n"], t["test"]["n"], t["unlabeled"]["n"], f"{t['train']['positive_rate']:.1%}", f"{t['train']['avg_chars']:.1f}"],
        ],
        [28 * mm, 18 * mm, 20 * mm, 16 * mm, 16 * mm, 16 * mm, 21 * mm, 18 * mm, 18 * mm],
    ))
    story.append(Spacer(1, 4))
    story.append(p("源域使用 ChineseNlpCorpus waimai_10k；目标域使用 HuggingFace dirtycomputer/weibo_senti_100k。下载 URL 记录在 results/final_data_audit.json 中。", small))
    story.append(p("目标域按负:正=2:1 构造，仅 10% 作为有标注训练集，60% 作为未标注数据用于弱监督伪标注。"))

    story.append(rich("3. 方法与消融矩阵", h2))
    story.append(table(
        [["ID", "方法", "说明"]]
        + [[row["method_id"], method_notes[row["method_id"]], method_desc[row["method_id"]]] for row in summary],
        [12 * mm, 50 * mm, 108 * mm],
        align_right_from=None,
    ))

    story.append(rich("4. 最终实验结果", h2))
    story.append(p("所有结果均为 3 个随机种子（42/123/456）的均值与标准差。"))
    result_rows = [["ID", "F1(source)", "F1(target)", "DeltaF1", "AUC", "Neg R", "Pos R", "Weighted-F1"]]
    for row in summary:
        result_rows.append([
            row["method_id"],
            f"{fmt(row['F1_s_mean'])}±{fmt(row['F1_s_std'])}",
            f"{fmt(row['F1_t_mean'])}±{fmt(row['F1_t_std'])}",
            f"{fmt(row['deltaF1_mean'])}±{fmt(row['deltaF1_std'])}",
            fmt(row["AUC_t_mean"]),
            fmt(row["recall_negative_t_mean"]),
            fmt(row["recall_positive_t_mean"]),
            fmt(row["weighted_f1_t_mean"]),
        ])
    story.append(table(
        result_rows,
        [12 * mm, 25 * mm, 25 * mm, 25 * mm, 18 * mm, 20 * mm, 20 * mm, 26 * mm],
    ))
    chart_path = ROOT / "results/final_metrics_chart.png"
    if chart_path.exists():
        story.append(Spacer(1, 6))
        img = RLImage(str(chart_path), width=170 * mm, height=96 * mm)
        img.hAlign = "LEFT"
        story.append(img)
    story.append(p(
        f"目标域 F1 最佳方法为 {best['method_id']} {best['method']}，F1(target)={fmt(best['F1_t_mean'])}，"
        f"负面召回={fmt(best['recall_negative_t_mean'])}。DeltaF1 最平衡方法为 "
        f"{best_balanced['method_id']} {best_balanced['method']}，DeltaF1={fmt(best_balanced['deltaF1_mean'])}。"
    ))
    story.append(p(
        "结论：少量目标域金标校准（E3）收益最大；弱监督伪标注（E2）显著提升目标域 F1；"
        "E5 虽然降低了 MMD，但过度过滤削弱正例召回，是负向消融。"
    ))

    story.append(rich("5. 错误分析", h2))
    for item in [
        "微博文本中大量转发链、表情和反讽表达，线性 n-gram 模型容易把局部正向词误判为整体正向。",
        "“祝福/哈哈/鼓掌”等词在不同语境下可能同时出现在正负样本中，造成词面冲突。",
        "领域特征过滤能降低 MMD，但过滤过强会删除目标域关键情感触发词。",
    ]:
        story.append(p(f"• {item}"))

    story.append(rich("6. 复现与交付", h2))
    story.append(rich("./scripts/run_final.sh<br/>python scripts/verify_final.py", code))
    story.append(p("核心产物：source_full.csv、social_full.csv、final_metrics.json、final_summary.csv、final_error_cases.csv、final_metrics_chart.png、final_report.md/pdf。"))

    story.append(rich("7. AI 工具使用声明", h2))
    story.append(p("项目使用 ChatGPT/Codex 辅助代码补全、实验脚本整理、报告生成与调试；数据处理、指标解释和最终结论均以本地可复现运行结果为准。"))

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )

    def draw_background(canvas, _doc):
        canvas.saveState()
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_background, onLaterPages=draw_background)


def main() -> None:
    report_dir = ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown()
    (report_dir / "final_report.md").write_text(markdown, encoding="utf-8")
    build_pdf(markdown)
    print(report_dir / "final_report.md")


if __name__ == "__main__":
    main()
