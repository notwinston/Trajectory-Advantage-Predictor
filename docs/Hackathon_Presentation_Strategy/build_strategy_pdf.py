#!/usr/bin/env python3
"""Build hackathon_presentation_strategy.pdf from the Markdown source.

PDF engine: reportlab (matches the repo's tap/report.py convention; no LaTeX
toolchain is installed on this host). Parses the Markdown into reportlab
Platypus flowables: title, section / subsection headings, paragraphs with
inline **bold** / *italic* / `code` / $math$, bullet lists, Markdown tables,
and $$ display math $$ rendered as monospace LaTeX-lite text.

Run with the venv interpreter that has reportlab:
    /workspace/.venv/bin/python build_strategy_pdf.py
"""

import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(HERE, "hackathon_presentation_strategy.md")
PDF_PATH = os.path.join(HERE, "hackathon_presentation_strategy.pdf")

USABLE_WIDTH = letter[0] - 2 * inch  # 1in margins each side

# --- Unicode -> ASCII (reportlab core fonts lack most math glyphs) ----------
_UNI = {
    "—": "--", "–": "-", "‑": "-", "−": "-",
    "“": '"', "”": '"', "‘": "'", "’": "'", "′": "'",
    "×": "x", "→": "->", "⇒": "=>", "≥": ">=", "≤": "<=",
    "≈": "~", "∝": "prop", "∈": " in ", "…": "...", "•": "-",
    "²": "^2", "³": "^3", "μ": "mu", "ε": "eps", "σ": "sigma",
    "π": "pi", "ρ": "rho", "θ": "theta", "λ": "lambda",
    "⚠️": "[!]", "⚠": "[!]", "✅": "[x]", "❌": "[no]", "✓": "[x]",
    " ": " ", "️": "", "​": "",
}


def _unicode_ascii(text):
    for k, v in _UNI.items():
        text = text.replace(k, v)
    return text


def _xml_escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --- LaTeX-lite: make math content readable without a TeX engine -----------
_TEX_TOKENS = {
    r"\times": "x", r"\cdot": ".", r"\propto": " prop ", r"\in": " in ",
    r"\max": "max", r"\min": "min", r"\sum": "sum ", r"\log": "log",
    r"\pi": "pi", r"\rho": "rho", r"\sigma": "sigma", r"\mu": "mu",
    r"\varepsilon": "eps", r"\theta": "theta", r"\lambda": "lambda",
    r"\geq": ">=", r"\leq": "<=", r"\approx": "~", r"\rightarrow": "->",
    r"\Rightarrow": "=>", r"\to": "->", r"\lVert": "||", r"\rVert": "||",
    r"\,": " ", r"\;": " ", r"\quad": "  ", r"\!": "",
    r"\left": "", r"\right": "", r"\big": "", r"\Big": "",
    r"\{": "{", r"\}": "}",
}


def tex_to_plain(s):
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbb\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\widehat\{([^}]*)\}", r"\1^", s)
    s = re.sub(r"\\overline\{([^}]*)\}", r"\1", s)
    for k, v in _TEX_TOKENS.items():
        s = s.replace(k, v)
    # Any remaining "\word" -> word.
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    s = s.replace("\\", "")
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def render_inline(text):
    """Markdown inline -> reportlab mini-markup (with XML escaping)."""
    code_spans, math_spans, money_spans = [], [], []

    def _sc(m):
        code_spans.append(m.group(1))
        return "\x00C%d\x00" % (len(code_spans) - 1)

    def _sd(m):
        money_spans.append(m.group(0)[1:])
        return "\x00D%d\x00" % (len(money_spans) - 1)

    def _sm(m):
        math_spans.append(m.group(1))
        return "\x00M%d\x00" % (len(math_spans) - 1)

    text = re.sub(r"`([^`]*)`", _sc, text)
    # Currency before math so unpaired '$' does not corrupt $...$ pairing.
    text = re.sub(r"\$\d[\d,]*K\+?", _sd, text)
    text = re.sub(r"\$([^$]+)\$", _sm, text)

    text = _xml_escape(text)
    text = _unicode_ascii(text)

    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)

    text = re.sub(r"\x00D(\d+)\x00",
                  lambda m: "$" + _xml_escape(money_spans[int(m.group(1))]), text)
    text = re.sub(
        r"\x00C(\d+)\x00",
        lambda m: '<font face="Courier" size="9">%s</font>'
        % _unicode_ascii(_xml_escape(code_spans[int(m.group(1))])),
        text,
    )
    text = re.sub(
        r"\x00M(\d+)\x00",
        lambda m: "<i>%s</i>"
        % _unicode_ascii(_xml_escape(tex_to_plain(math_spans[int(m.group(1))]))),
        text,
    )
    return text


def build_styles():
    ss = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "DocTitle", parent=ss["Title"], fontSize=20, leading=24,
            textColor=colors.HexColor("#003278"), spaceAfter=10,
        ),
        "intro": ParagraphStyle(
            "Intro", parent=ss["Normal"], fontSize=9.5, leading=13,
            textColor=colors.HexColor("#444444"), spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "H1", parent=ss["Heading1"], fontSize=14, leading=17,
            textColor=colors.HexColor("#003278"), spaceBefore=12, spaceAfter=5,
        ),
        "h2": ParagraphStyle(
            "H2", parent=ss["Heading2"], fontSize=11.5, leading=14,
            textColor=colors.HexColor("#1a1a1a"), spaceBefore=8, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "Body", parent=ss["Normal"], fontSize=9.7, leading=13.5,
            alignment=TA_LEFT, spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=ss["Normal"], fontSize=9.7, leading=13,
        ),
        "math": ParagraphStyle(
            "Math", parent=ss["Code"], fontSize=9.5, leading=13,
            textColor=colors.HexColor("#222222"), backColor=colors.HexColor("#f4f4f4"),
            borderPadding=4, spaceBefore=4, spaceAfter=6, leftIndent=6,
        ),
        "cell": ParagraphStyle(
            "Cell", parent=ss["Normal"], fontSize=8.6, leading=11,
        ),
        "cellh": ParagraphStyle(
            "CellH", parent=ss["Normal"], fontSize=8.8, leading=11,
            textColor=colors.white,
        ),
    }
    return styles


def make_table(rows, styles):
    parsed = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header = parsed[0]
    body = [r for r in parsed[1:]
            if not all(set(c) <= set("-: ") for c in r)]
    ncol = len(header)
    colw = [USABLE_WIDTH / ncol] * ncol

    data = [[Paragraph("<b>%s</b>" % render_inline(c), styles["cellh"])
             for c in header]]
    for row in body:
        row = (row + [""] * ncol)[:ncol]
        data.append([Paragraph(render_inline(c), styles["cell"]) for c in row])

    tbl = Table(data, colWidths=colw, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003278")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f3f6fb")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tbl


def build():
    with open(MD_PATH, "r", encoding="utf-8") as fh:
        lines = fh.read().split("\n")

    styles = build_styles()
    story = []
    i, n = 0, len(lines)
    title_done = False

    while i < n:
        line = lines[i]
        s = line.strip()

        if s == "$$":
            j = i + 1
            mlines = []
            while j < n and lines[j].strip() != "$$":
                mlines.append(tex_to_plain(lines[j]))
                j += 1
            txt = _xml_escape("\n".join(ml for ml in mlines if ml))
            txt = _unicode_ascii(txt).replace("\n", "<br/>")
            story.append(Paragraph(txt, styles["math"]))
            i = j + 1
            continue

        if s.startswith("|"):
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            story.append(make_table(tbl, styles))
            story.append(Spacer(1, 6))
            continue

        if s == "---":
            story.append(Spacer(1, 4))
            i += 1
            continue

        if s.startswith("### "):
            story.append(Paragraph(render_inline(s[4:]), styles["h2"]))
            i += 1
            continue
        if s.startswith("## "):
            story.append(Paragraph(render_inline(s[3:]), styles["h1"]))
            i += 1
            continue
        if s.startswith("# "):
            if not title_done:
                story.append(Paragraph(render_inline(s[2:]), styles["title"]))
                title_done = True
            else:
                story.append(Paragraph(render_inline(s[2:]), styles["h1"]))
            i += 1
            continue

        if s.startswith("- "):
            items = []
            while i < n and lines[i].strip().startswith("- "):
                items.append(ListItem(
                    Paragraph(render_inline(lines[i].strip()[2:]), styles["bullet"]),
                    leftIndent=14))
                i += 1
            story.append(ListFlowable(items, bulletType="bullet",
                                      start="circle", leftIndent=12))
            story.append(Spacer(1, 4))
            continue

        if s == "":
            i += 1
            continue

        # Paragraph (intro style for the leading italic blurb, else body).
        para_style = styles["intro"] if (s.startswith("*") and s.endswith("*")
                                         and i < 6) else styles["body"]
        story.append(Paragraph(render_inline(s), para_style))
        i += 1

    doc = SimpleDocTemplate(
        PDF_PATH, pagesize=letter,
        leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch,
        title="TAP v1 Hackathon Presentation Strategy",
    )
    doc.build(story)
    size = os.path.getsize(PDF_PATH)
    print("Wrote %s (%d bytes)" % (PDF_PATH, size))


if __name__ == "__main__":
    build()
