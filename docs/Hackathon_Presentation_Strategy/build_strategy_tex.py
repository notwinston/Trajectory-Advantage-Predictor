#!/usr/bin/env python3
"""Build hackathon_presentation_strategy.tex from the Markdown source.

Reads hackathon_presentation_strategy.md and emits a self-contained LaTeX
article. Handles: ## / ### headers, $$...$$ display math, $...$ inline math,
Markdown tables (booktabs), **bold**, *italic*, `code`, bullet lists, and a
comprehensive Unicode->TeX replacement for text-mode content. Math spans and
inline code are protected before escaping so their contents are not mangled.
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(HERE, "hackathon_presentation_strategy.md")
TEX_PATH = os.path.join(HERE, "hackathon_presentation_strategy.tex")

# --- LaTeX special-character escaping (text mode only) ---------------------
_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\^{}",
}

# --- Unicode -> TeX (applied to text-mode content after escaping) ----------
_UNICODE = {
    "—": "---", "–": "--", "‑": "-", "−": "$-$",
    "“": "``", "”": "''", "‘": "`", "’": "'", "′": "'",
    "×": r"$\times$", "→": r"$\rightarrow$", "⇒": r"$\Rightarrow$",
    "≥": r"$\geq$", "≤": r"$\leq$", "≈": r"$\approx$", "∝": r"$\propto$",
    "∈": r"$\in$", "…": "...", "•": "-",
    "²": r"\textsuperscript{2}", "³": r"\textsuperscript{3}",
    "μ": r"$\mu$", "ε": r"$\varepsilon$", "σ": r"$\sigma$",
    "π": r"$\pi$", "ρ": r"$\rho$", "θ": r"$\theta$", "λ": r"$\lambda$",
    "⚠️": "[!]", "⚠": "[!]", "✅": "[x]", "❌": "[no]", "✓": "[x]",
    " ": " ", "️": "", "​": "",
}


def _escape_text(text):
    out = []
    for ch in text:
        out.append(_ESCAPE.get(ch, ch))
    return "".join(out)


def _apply_unicode(text):
    for k, v in _UNICODE.items():
        text = text.replace(k, v)
    return text


def render_inline(text):
    """Render a line of Markdown inline syntax to LaTeX.

    Order: protect $math$ and `code`, escape, unicode-map, bold/italic,
    then restore protected spans.
    """
    math_spans = []
    code_spans = []
    money_spans = []

    def _stash_math(m):
        math_spans.append(m.group(1))
        return "\x00M%d\x00" % (len(math_spans) - 1)

    def _stash_code(m):
        code_spans.append(m.group(1))
        return "\x00C%d\x00" % (len(code_spans) - 1)

    def _stash_money(m):
        # m.group(0) like "$50K" / "$100K+"; store the part after the '$'.
        money_spans.append(m.group(0)[1:])
        return "\x00D%d\x00" % (len(money_spans) - 1)

    # Protect inline code first (so $ inside code is not treated as math).
    text = re.sub(r"`([^`]*)`", _stash_code, text)
    # Protect currency ($<digits>K[+]) BEFORE math: these are unpaired dollars
    # that would otherwise corrupt the $...$ math pairing. Math spans starting
    # with a digit never have digits immediately followed by 'K', so they are
    # not matched here.
    text = re.sub(r"\$\d[\d,]*K\+?", _stash_money, text)
    text = re.sub(r"\$([^$]+)\$", _stash_math, text)

    text = _escape_text(text)
    text = _apply_unicode(text)

    # Bold before italic (since ** contains *).
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"\*(.+?)\*", r"\\emph{\1}", text)

    # Restore code (escape its contents for \texttt).
    def _restore_code(m):
        raw = code_spans[int(m.group(1))]
        return r"\texttt{%s}" % _escape_text(raw)

    text = re.sub(r"\x00C(\d+)\x00", _restore_code, text)

    # Restore currency as an escaped dollar sign + the (digit/K/+) tail.
    def _restore_money(m):
        return r"\$" + _escape_text(money_spans[int(m.group(1))])

    text = re.sub(r"\x00D(\d+)\x00", _restore_money, text)

    # Restore math verbatim.
    def _restore_math(m):
        return "$%s$" % math_spans[int(m.group(1))]

    text = re.sub(r"\x00M(\d+)\x00", _restore_math, text)
    return text


def render_table(rows):
    """rows: list of raw '| a | b |' lines (incl. separator). -> LaTeX."""
    parsed = []
    for r in rows:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        parsed.append(cells)
    # Drop the separator row (the |---|---| line).
    header = parsed[0]
    body = [r for r in parsed[1:] if not all(set(c) <= set("-: ") for c in r)]
    ncol = len(header)
    colspec = "l" * ncol
    out = [r"\begin{center}", r"\begin{tabular}{%s}" % colspec, r"\toprule"]
    out.append(" & ".join(render_inline(c) for c in header) + r" \\")
    out.append(r"\midrule")
    for row in body:
        # Pad/truncate to ncol.
        row = (row + [""] * ncol)[:ncol]
        out.append(" & ".join(render_inline(c) for c in row) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{center}")
    return out


PREAMBLE = r"""\documentclass[11pt,letterpaper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{array}
\usepackage{parskip}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{hyperref}
\definecolor{darkblue}{RGB}{0,50,120}
\titleformat{\section}{\large\bfseries\color{darkblue}}{\thesection}{0.6em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.5em}{}
\setlength{\parskip}{6pt}
\begin{document}
"""

POSTAMBLE = r"""
\end{document}
"""


def build():
    with open(MD_PATH, "r", encoding="utf-8") as fh:
        lines = fh.read().split("\n")

    body = []
    i = 0
    n = len(lines)
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            body.append(r"\end{itemize}")
            in_list = False

    title_done = False

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Display math block: $$ ... $$
        if stripped == "$$":
            close_list()
            j = i + 1
            math_lines = []
            while j < n and lines[j].strip() != "$$":
                math_lines.append(lines[j])
                j += 1
            body.append(r"\[")
            body.extend(math_lines)
            body.append(r"\]")
            i = j + 1
            continue

        # Table block
        if stripped.startswith("|"):
            close_list()
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            body.extend(render_table(tbl))
            continue

        # Horizontal rule
        if stripped == "---":
            close_list()
            body.append(r"\medskip")
            i += 1
            continue

        # Headers
        if stripped.startswith("### "):
            close_list()
            body.append(r"\subsection*{%s}" % render_inline(stripped[4:]))
            i += 1
            continue
        if stripped.startswith("## "):
            close_list()
            body.append(r"\section*{%s}" % render_inline(stripped[3:]))
            i += 1
            continue
        if stripped.startswith("# "):
            close_list()
            if not title_done:
                body.append(r"\begin{center}")
                body.append(r"{\LARGE\bfseries %s}" % render_inline(stripped[2:]))
                body.append(r"\end{center}")
                title_done = True
            else:
                body.append(r"\section*{%s}" % render_inline(stripped[2:]))
            i += 1
            continue

        # Bullet list item
        if stripped.startswith("- "):
            if not in_list:
                body.append(r"\begin{itemize}")
                in_list = True
            body.append(r"\item %s" % render_inline(stripped[2:]))
            i += 1
            continue

        # Blank line
        if stripped == "":
            close_list()
            body.append("")
            i += 1
            continue

        # Plain paragraph
        close_list()
        body.append(render_inline(stripped))
        i += 1

    close_list()

    tex = PREAMBLE + "\n".join(body) + POSTAMBLE
    with open(TEX_PATH, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print("Wrote %s (%d lines)" % (TEX_PATH, tex.count("\n") + 1))


if __name__ == "__main__":
    build()
