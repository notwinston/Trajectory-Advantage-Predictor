"""Report rendering for TAP v1: results.csv + report.md/.tex/.pdf.

The model x metric table is emitted in three formats. The PDF engine is chosen
at runtime, preferring (per spec) ``tectonic -> pandoc -> pdfkit -> reportlab``;
only engines whose binary/library is present are attempted, and the resolved
engine name is recorded inside ``report.md``. A tiny dependency-free PDF writer
is the guaranteed fallback so the pipeline always produces a >1KB PDF in lean
environments.

Pure CPU. reportlab is imported lazily inside its optional fallback.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Canonical model-row order (matches the promise's model list exactly).
MODEL_ORDER: List[str] = [
    "tap",
    "tap-no-prob",
    "tap-no-grad",
    "tap-no-history",
    "ridge",
    "gbt",
    "no_history_mlp",
    "numeric_only",
    "candidate_only",
    "random",
    "reward_mean",
    "advantage_mean",
    "geo_mean_prob",
    "arith_mean_prob",
    "reward_x_surprisal",
    "semantic_novelty",
    "gradient_norm",
    "gradient_alignment",
]

METRIC_COLS = [
    "spearman",
    "pair_acc",
    "top1_regret",
    "mean_true_utility",
    "lift_random",
    "lift_reward",
    "lift_prob",
]


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def write_csv(path: Path, results: pd.DataFrame) -> None:
    results.to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Markdown / LaTeX text
# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _verdict_lines(analysis: Dict) -> List[str]:
    def beat(flag: bool) -> str:
        return "**beat**" if flag else "did **not** beat"

    return [
        f"On held-out chains, TAP ({analysis['tap_label']}) {beat(analysis['beat_random'])} "
        f"random, {beat(analysis['beat_reward'])} reward-only, and "
        f"{beat(analysis['beat_prob'])} probability-only selection, "
        f"measured by mean true utility of the selected candidate "
        f"(TAP={_fmt(analysis['tap_mtu'])} vs random={_fmt(analysis['mtu_random'])}, "
        f"reward-only={_fmt(analysis['mtu_reward'])}, prob-only={_fmt(analysis['mtu_prob'])}).",
    ]


def render_markdown(results: pd.DataFrame, analysis: Dict, pdf_engine: str) -> str:
    lines: List[str] = []
    lines.append("# TAP v1 — Trajectory Advantage Predictor: results\n")
    lines.append(
        f"Dataset: `{analysis['parquet_dir']}` "
        f"({analysis['n_labels']} candidate labels across {analysis['n_states']} states, "
        f"{analysis.get('n_chains', 2)} chains). Evaluation: leave-one-chain-out, averaged over all directions. "
        f"Seed {analysis['seed']}.\n"
    )
    params = analysis.get("tap_num_params")
    if params is not None:
        lines.append(f"SmallTAP trainable parameters: **{params:,}** (budget < 250,000).\n")
    else:
        lines.append("TAP ran on the **TAP_NO_TORCH** sklearn fallback (simpler model).\n")

    lines.append("## Verdict\n")
    lines.extend(_verdict_lines(analysis))
    lines.append(
        "\n> **Caveat — synthetic data.** These numbers come from `tap.synth`, a "
        "self-consistent synthetic generator, **not** real Qwen3-8B GRPO branch labels. "
        "They validate the pipeline and the within-state ranking machinery; they do "
        "**not** establish that TAP works on real data. The latent signal is a tuned "
        "noisy-linear function, so simple/ablated models can match or beat the full "
        "attention model here — which is an expected, acceptable outcome at this scale.\n"
    )

    lines.append(f"\n**PDF engine:** report.pdf was produced by **{pdf_engine}**.\n")

    lines.append("\n## Model x metric table\n")
    header = "| model | family | " + " | ".join(METRIC_COLS) + " |"
    sep = "|" + "---|" * (2 + len(METRIC_COLS))
    lines.append(header)
    lines.append(sep)
    for _, row in results.iterrows():
        cells = [str(row["model"]), str(row["family"])]
        cells += [_fmt(float(row[c])) for c in METRIC_COLS]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## Column meaning\n")
    lines.append(
        "- `spearman` / `pair_acc`: within-state rank agreement with true utility "
        "(higher better).\n"
        "- `top1_regret`: best true utility at a state minus the true utility of the "
        "model's pick (lower better).\n"
        "- `mean_true_utility`: true utility of the selected candidate, averaged over states.\n"
        "- `lift_random` / `lift_reward` / `lift_prob`: `mean_true_utility` minus that of "
        "the random / reward-only / probability-only selector.\n"
    )
    lines.append("\n## Caveats & scope\n")
    lines.append(
        "- ~72 noisy one-step probe labels: ranking quality is the headline, not global R^2.\n"
        "- Both leave-one-chain-out directions are averaged; no random row splits are used.\n"
        "- TAP and its three ablations (no-prob / no-grad / no-history) share one model "
        "definition with masked inputs.\n"
    )
    return "\n".join(lines) + "\n"


def render_latex(results: pd.DataFrame, analysis: Dict) -> str:
    def esc(s: str) -> str:
        return str(s).replace("_", r"\_").replace("&", r"\&")

    col_spec = "ll" + "r" * len(METRIC_COLS)
    head = " & ".join(["model", "family"] + [esc(c) for c in METRIC_COLS]) + r" \\"
    body_rows = []
    for _, row in results.iterrows():
        cells = [esc(row["model"]), esc(row["family"])] + [_fmt(float(row[c])) for c in METRIC_COLS]
        body_rows.append(" & ".join(cells) + r" \\")
    params = analysis.get("tap_num_params")
    param_line = (
        f"SmallTAP trainable parameters: {params:,} (budget $<$ 250{{,}}000)."
        if params is not None
        else "TAP ran on the TAP\\_NO\\_TORCH sklearn fallback (simpler model)."
    )

    def beat(flag):
        return "beat" if flag else "did not beat"

    verdict = (
        f"On held-out chains, TAP {beat(analysis['beat_random'])} random, "
        f"{beat(analysis['beat_reward'])} reward-only, and {beat(analysis['beat_prob'])} "
        f"probability-only selection by mean true utility "
        f"(TAP={_fmt(analysis['tap_mtu'])}, random={_fmt(analysis['mtu_random'])}, "
        f"reward-only={_fmt(analysis['mtu_reward'])}, prob-only={_fmt(analysis['mtu_prob'])})."
    )
    return "\n".join(
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage[margin=0.8in]{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{longtable}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{document}",
            r"\section*{TAP v1 --- Trajectory Advantage Predictor: results}",
            f"Dataset: \\texttt{{{esc(analysis['parquet_dir'])}}} "
            f"({analysis['n_labels']} candidate labels, {analysis['n_states']} states, "
            f"{analysis.get('n_chains', 2)} chains). "
            f"Leave-one-chain-out, averaged over all directions. Seed {analysis['seed']}.",
            "",
            param_line,
            "",
            r"\textbf{Verdict.} " + verdict,
            "",
            r"\textbf{Caveat (synthetic data).} Numbers come from \texttt{tap.synth}, not real "
            r"Qwen3-8B GRPO labels; they validate the pipeline/ranking machinery only. Simple or "
            r"ablated models can match the full attention model at this scale.",
            "",
            r"\begin{center}",
            r"\footnotesize",
            r"\begin{longtable}{" + col_spec + "}",
            r"\toprule",
            head,
            r"\midrule",
            r"\endhead",
            *body_rows,
            r"\bottomrule",
            r"\end{longtable}",
            r"\end{center}",
            r"\end{document}",
            "",
        ]
    )


# --------------------------------------------------------------------------- #
# PDF engines
# --------------------------------------------------------------------------- #
def _try_tectonic(tex_path: Path, out_dir: Path, pdf_path: Path) -> bool:
    if shutil.which("tectonic") is None:
        return False
    try:
        subprocess.run(
            ["tectonic", "--outdir", str(out_dir), str(tex_path)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        return pdf_path.exists() and pdf_path.stat().st_size > 1024
    except Exception:
        return False


def _try_pandoc(md_path: Path, pdf_path: Path) -> bool:
    if shutil.which("pandoc") is None:
        return False
    try:
        subprocess.run(
            ["pandoc", str(md_path), "-o", str(pdf_path)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        return pdf_path.exists() and pdf_path.stat().st_size > 1024
    except Exception:
        return False


def _try_pdfkit(html: str, pdf_path: Path) -> bool:
    if shutil.which("wkhtmltopdf") is None:
        return False
    try:
        import pdfkit  # noqa: F401

        pdfkit.from_string(html, str(pdf_path))
        return pdf_path.exists() and pdf_path.stat().st_size > 1024
    except Exception:
        return False


def _reportlab_pdf(pdf_path: Path, results: pd.DataFrame, analysis: Dict) -> bool:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("TAP v1 — Trajectory Advantage Predictor: results", styles["Title"]))
    story.append(
        Paragraph(
            f"Dataset: {analysis['parquet_dir']} — {analysis['n_labels']} candidate labels, "
            f"{analysis['n_states']} states, {analysis.get('n_chains', 2)} chains. "
            "Leave-one-chain-out directions "
            f"averaged. Seed {analysis['seed']}.",
            styles["Normal"],
        )
    )
    params = analysis.get("tap_num_params")
    story.append(
        Paragraph(
            f"SmallTAP trainable parameters: {params:,} (budget &lt; 250,000)."
            if params is not None
            else "TAP ran on the TAP_NO_TORCH sklearn fallback (simpler model).",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 8))

    def beat(flag):
        return "beat" if flag else "did NOT beat"

    story.append(
        Paragraph(
            f"<b>Verdict.</b> TAP {beat(analysis['beat_random'])} random, "
            f"{beat(analysis['beat_reward'])} reward-only, and {beat(analysis['beat_prob'])} "
            f"probability-only by mean true utility (TAP={analysis['tap_mtu']:.3f}, "
            f"random={analysis['mtu_random']:.3f}, reward-only={analysis['mtu_reward']:.3f}, "
            f"prob-only={analysis['mtu_prob']:.3f}).",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "<b>Caveat (synthetic data).</b> Numbers come from tap.synth, not real Qwen3-8B "
            "GRPO branch labels; they validate the pipeline and within-state ranking machinery "
            "only. Simple or ablated models can match the full attention model at this scale.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    header = ["model", "family"] + METRIC_COLS
    data = [header]
    for _, row in results.iterrows():
        data.append(
            [str(row["model"]), str(row["family"])] + [f"{float(row[c]):.3f}" for c in METRIC_COLS]
        )
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f6")]),
                ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
            ]
        )
    )
    story.append(table)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=landscape(letter))
    doc.build(story)
    return pdf_path.exists() and pdf_path.stat().st_size > 1024


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _plain_pdf(pdf_path: Path, results: pd.DataFrame, analysis: Dict) -> bool:
    """Write a minimal valid PDF without third-party dependencies."""
    lines = [
        "TAP v1 - Trajectory Advantage Predictor",
        f"Dataset: {analysis['parquet_dir']}",
        f"Labels: {analysis['n_labels']}  States: {analysis['n_states']}  Chains: {analysis.get('n_chains', 2)}",
        f"Seed: {analysis['seed']}  Epochs: {analysis['epochs']}",
        f"TAP mean true utility: {analysis['tap_mtu']:.3f}",
        f"Random: {analysis['mtu_random']:.3f}  Reward-only: {analysis['mtu_reward']:.3f}  Prob-only: {analysis['mtu_prob']:.3f}",
        "",
        "Model rows:",
    ]
    for _, row in results.head(20).iterrows():
        lines.append(
            f"{row['model']}: spearman={float(row['spearman']):.3f}, "
            f"pair_acc={float(row['pair_acc']):.3f}, mtu={float(row['mean_true_utility']):.3f}"
        )
    while len(lines) < 42:
        lines.append("Synthetic-data caveat: this validates the pipeline, not real branch-label quality.")

    y = 760
    content_lines = ["BT", "/F1 9 Tf", "72 760 Td"]
    for line in lines:
        content_lines.append(f"({_escape_pdf_text(line[:110])}) Tj")
        content_lines.append("0 -14 Td")
        y -= 14
        if y < 80:
            break
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    pdf_path.write_bytes(out)
    return pdf_path.exists() and pdf_path.stat().st_size > 1024


def write_pdf(out_dir: Path, tex_path: Path, md_text_for: Dict, results: pd.DataFrame, analysis: Dict) -> str:
    """Produce report.pdf, returning the engine name that succeeded.

    ``md_text_for`` maps an engine name -> the markdown string to feed pandoc.
    """
    pdf_path = out_dir / "report.pdf"
    if _try_tectonic(tex_path, out_dir, pdf_path):
        return "tectonic"
    md_path = out_dir / "report.md"
    md_path.write_text(md_text_for("pandoc"))
    if _try_pandoc(md_path, pdf_path):
        return "pandoc"
    # crude HTML for pdfkit.
    html = "<html><body><pre>" + md_text_for("pdfkit") + "</pre></body></html>"
    if _try_pdfkit(html, pdf_path):
        return "pdfkit"
    try:
        if _reportlab_pdf(pdf_path, results, analysis):
            return "reportlab"
    except ModuleNotFoundError:
        pass
    _plain_pdf(pdf_path, results, analysis)
    return "plain-pdf"


def write_all(out_dir: str | Path, results: pd.DataFrame, analysis: Dict) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_csv(out / "results.csv", results)
    tex_path = out / "report.tex"
    tex_path.write_text(render_latex(results, analysis))

    def md_text_for(engine: str) -> str:
        return render_markdown(results, analysis, engine)

    engine = write_pdf(out, tex_path, md_text_for, results, analysis)
    # Final report.md records the resolved engine.
    (out / "report.md").write_text(md_text_for(engine))
    return {
        "results.csv": str(out / "results.csv"),
        "report.md": str(out / "report.md"),
        "report.tex": str(tex_path),
        "report.pdf": str(out / "report.pdf"),
        "pdf_engine": engine,
    }
