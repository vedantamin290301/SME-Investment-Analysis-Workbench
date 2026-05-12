from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_pdf_report(company: str, analysis: dict, financials, memo: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []
    overall = analysis["overall"]

    story.append(Paragraph(f"{company} SME IPO Analysis", styles["Title"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph(f"Overall Score: {overall['score']} / 100", styles["Heading2"]))
    story.append(Paragraph(f"Rating: {overall['rating']}", styles["Heading2"]))
    story.append(Paragraph("Professional fund-manager screening report generated from 3-year financials.", styles["BodyText"]))
    story.append(PageBreak())

    rows = [["Module", "Metric", "Score", "Status", "Latest / Values"]]
    for module_name, module in analysis["modules"].items():
        for metric in module["metrics"].values():
            rows.append([
                module_name,
                metric["name"],
                metric["score"],
                metric["status"].upper(),
                _compact_values(metric.get("values", {})),
            ])
    table = Table(rows, repeatRows=1, colWidths=[90, 120, 45, 55, 220])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(Paragraph("Ratio Tables", styles["Heading1"]))
    story.append(table)
    story.append(PageBreak())

    chart_path = _save_trend_chart(financials)
    story.append(Paragraph("Trend Charts", styles["Heading1"]))
    story.append(Image(chart_path, width=7.0 * inch, height=4.2 * inch))
    story.append(PageBreak())

    story.append(Paragraph("AI Investment Memo", styles["Heading1"]))
    for para in memo.split("\n"):
        if para.strip():
            story.append(Paragraph(para.replace("**", ""), styles["BodyText"]))
            story.append(Spacer(1, 0.08 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def _compact_values(values: dict) -> str:
    chunks = []
    for key, value in values.items():
        if isinstance(value, float):
            value = round(value, 2)
        if isinstance(value, list):
            value = [round(v, 2) if isinstance(v, float) else v for v in value]
        chunks.append(f"{key}: {value}")
    return "; ".join(chunks)[:180]


def _save_trend_chart(financials) -> str:
    path = Path(tempfile.gettempdir()) / "sme_ipo_trends.png"
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.4))
    years = financials["year"].astype(str)
    axes[0, 0].plot(years, financials["revenue"], marker="o")
    axes[0, 0].set_title("Revenue")
    axes[0, 1].plot(years, financials["ebitda"] / financials["revenue"] * 100, marker="o")
    axes[0, 1].set_title("EBITDA Margin %")
    axes[1, 0].plot(years, financials["operating_cash_flow"], marker="o")
    axes[1, 0].set_title("OCF")
    axes[1, 1].plot(years, financials["operating_cash_flow"] - financials["capex"], marker="o")
    axes[1, 1].set_title("FCF")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)
