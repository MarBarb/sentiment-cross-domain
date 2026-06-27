"""Render the expanded speaker notes Markdown to 演讲稿.pdf."""
from __future__ import annotations

from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


ROOT = Path(__file__).resolve().parent
MD = ROOT / "演讲稿.md"
PDF = ROOT / "演讲稿.pdf"


def register_font() -> str:
    candidates = [
        ("/System/Library/Fonts/STHeiti Medium.ttc", 0),
        ("/System/Library/Fonts/PingFang.ttc", 0),
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
    ]
    for path, subfont in candidates:
        try:
            pdfmetrics.registerFont(TTFont("DeckCJK", path, subfontIndex=subfont))
            return "DeckCJK"
        except Exception:
            continue
    return "Helvetica"


def parse_markdown(text: str):
    blocks = []
    current = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if current:
                blocks.append(("p", " ".join(current)))
                current = []
            continue
        if line.startswith("# "):
            if current:
                blocks.append(("p", " ".join(current)))
                current = []
            blocks.append(("h1", line[2:].strip()))
        elif line.startswith("## "):
            if current:
                blocks.append(("p", " ".join(current)))
                current = []
            blocks.append(("h2", line[3:].strip()))
        else:
            current.append(line)
    if current:
        blocks.append(("p", " ".join(current)))
    return blocks


def main() -> None:
    font = register_font()
    base = ParagraphStyle(
        "Base",
        fontName=font,
        fontSize=10.2,
        leading=16,
        wordWrap="CJK",
        textColor=colors.HexColor("#111827"),
        spaceAfter=6,
    )
    title = ParagraphStyle(
        "Title",
        parent=base,
        fontSize=20,
        leading=26,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=base,
        fontSize=13.2,
        leading=18,
        textColor=colors.HexColor("#1E40AF"),
        spaceBefore=7,
        spaceAfter=4,
    )
    story = []
    first_h2 = True
    for kind, content in parse_markdown(MD.read_text(encoding="utf-8")):
        if kind == "h1":
            story.append(Paragraph(escape(content), title))
            story.append(Spacer(1, 2 * mm))
        elif kind == "h2":
            if not first_h2 and content.startswith("Slide "):
                story.append(Spacer(1, 1.5 * mm))
            first_h2 = False
            story.append(Paragraph(escape(content), h2))
        else:
            story.append(Paragraph(escape(content), base))

    doc = SimpleDocTemplate(
        str(PDF),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=16 * mm,
        title="14 分钟答辩讲稿",
        author="李乘黄、马啸、THAM WAN HEI、化润宇",
    )

    def draw_footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(18 * mm, 8 * mm, "社交媒体情感分析的跨域泛化 · 扩展讲稿")
        canvas.drawRightString(A4[0] - 18 * mm, 8 * mm, f"第 {doc_obj.page} 页")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    print(PDF)


if __name__ == "__main__":
    main()
