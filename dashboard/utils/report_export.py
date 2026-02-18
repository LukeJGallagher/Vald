"""
Team Saudi VALD Performance Dashboard - Report Export Module

Provides PDF and HTML report generation for group and individual athlete reports.
Uses ReportLab for PDF generation and f-string templates for self-contained HTML.
All outputs follow Team Saudi branding guidelines.

Usage:
    from dashboard.utils.report_export import (
        generate_group_pdf_report,
        generate_group_html_report,
        generate_individual_pdf_report,
        generate_individual_html_report,
    )
"""

import io
import base64
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Team Saudi Brand Colors
# ---------------------------------------------------------------------------
SAUDI_GREEN = '#235032'
GOLD_ACCENT = '#a08e66'
DARK_GREEN = '#1a3d25'
LIGHT_GREEN = '#3a7050'
STATUS_WARNING = '#FFB800'
STATUS_DANGER = '#dc3545'


# ===================================================================
# Helper Functions
# ===================================================================

def _fig_to_base64(fig: go.Figure, width: int = 800, height: int = 400) -> str:
    """
    Convert a Plotly figure to a base64-encoded PNG string.

    Args:
        fig: Plotly figure object.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Base64-encoded PNG string, or empty string on failure.
    """
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"[report_export] Error converting figure to base64: {e}")
        return ""


def _fig_to_reportlab_image(fig: go.Figure, width_cm: float = 24, height_cm: float = 12):
    """
    Convert a Plotly figure to a ReportLab Image element.

    Args:
        fig: Plotly figure object.
        width_cm: Desired width in centimetres.
        height_cm: Desired height in centimetres.

    Returns:
        ReportLab Image flowable, or None on failure.
    """
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as RLImage

    try:
        img_bytes = fig.to_image(format="png", width=900, height=450, scale=2)
        return RLImage(io.BytesIO(img_bytes), width=width_cm * cm, height=height_cm * cm)
    except Exception as e:
        print(f"[report_export] Error converting figure to ReportLab image: {e}")
        return None


def _build_html_header(title: str, subtitle: str = "") -> str:
    """
    Build a Team Saudi branded HTML header block.

    Args:
        title: Main heading text.
        subtitle: Optional sub-heading text.

    Returns:
        HTML string for the header section.
    """
    subtitle_html = ""
    if subtitle:
        subtitle_html = (
            f'<p style="color: {GOLD_ACCENT}; font-size: 1.1rem; '
            f'margin: 0.5rem 0 0 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">'
            f'{subtitle}</p>'
        )

    return f"""
    <div style="background: linear-gradient(135deg, {SAUDI_GREEN} 0%, {DARK_GREEN} 100%);
         padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
         border-bottom: 4px solid {GOLD_ACCENT}; text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.4);">{title}</h1>
        {subtitle_html}
    </div>
    """


def _build_html_table(df: pd.DataFrame, title: str = "") -> str:
    """
    Build an HTML table with Team Saudi styling.

    Numeric columns are formatted to one decimal place, date columns to YYYY-MM-DD.

    Args:
        df: DataFrame to render.
        title: Optional section title above the table.

    Returns:
        HTML string for the styled table.
    """
    if df is None or df.empty:
        return ""

    display_df = df.copy()

    # Format numeric columns
    for col in display_df.select_dtypes(include=["float64", "float32"]).columns:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else ""
        )

    # Format date columns
    for col in display_df.columns:
        if "date" in col.lower():
            try:
                display_df[col] = pd.to_datetime(display_df[col]).dt.strftime("%Y-%m-%d")
            except Exception:
                pass

    title_html = ""
    if title:
        title_html = (
            f'<div style="background: linear-gradient(135deg, {SAUDI_GREEN} 0%, {DARK_GREEN} 100%); '
            f'padding: 0.75rem 1rem; border-radius: 8px 8px 0 0; '
            f'border-left: 4px solid {GOLD_ACCENT};">'
            f'<h3 style="color: white; margin: 0; font-size: 1rem;">{title}</h3></div>'
        )

    # Build header row
    header_cells = "".join(
        f'<th style="background: {SAUDI_GREEN}; color: white; padding: 10px 12px; '
        f'text-align: center; font-weight: 600; font-size: 0.85rem; '
        f'border-bottom: 3px solid {GOLD_ACCENT};">{col}</th>'
        for col in display_df.columns
    )

    # Build body rows with alternating backgrounds
    body_rows = ""
    for idx, row in enumerate(display_df.itertuples(index=False)):
        bg = "#ffffff" if idx % 2 == 0 else "#f8f9fa"
        cells = "".join(
            f'<td style="padding: 8px 12px; text-align: center; font-size: 0.85rem; '
            f'border-bottom: 1px solid #e9ecef;">{val}</td>'
            for val in row
        )
        body_rows += f'<tr style="background: {bg};">{cells}</tr>\n'

    return f"""
    {title_html}
    <table style="width: 100%; border-collapse: collapse; margin-bottom: 1.5rem;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-radius: 0 0 8px 8px;
           overflow: hidden; font-family: Inter, sans-serif;">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{body_rows}</tbody>
    </table>
    """


def _build_html_footer() -> str:
    """
    Build the confidential footer for HTML reports.

    Returns:
        HTML string for the footer.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
    <div style="margin-top: 3rem; padding: 1rem; border-top: 2px solid {GOLD_ACCENT};
         text-align: center; color: #999; font-size: 0.8rem; font-family: Inter, sans-serif;">
        <p style="margin: 0;">Team Saudi Performance Analysis | Confidential</p>
        <p style="margin: 0.25rem 0 0 0;">Generated: {now_str}</p>
    </div>
    """


def _safe_metadata(metadata: Optional[dict]) -> dict:
    """
    Return metadata with sensible defaults for any missing keys.

    Args:
        metadata: Caller-supplied metadata dict (may be None or partial).

    Returns:
        Dict with guaranteed keys.
    """
    defaults = {
        "sport": "All Sports",
        "gender": "All",
        "date_from": "",
        "date_to": "",
        "athlete_count": 0,
        "test_count": 0,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    if metadata is None:
        return defaults
    for key, val in defaults.items():
        metadata.setdefault(key, val)
    return metadata


# ===================================================================
# PDF Helpers (ReportLab)
# ===================================================================

def _get_pdf_styles():
    """
    Build the standard set of ReportLab paragraph styles for reports.

    Returns:
        Tuple of (styles_dict, reportlab_colors_dict).
    """
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    RL_SAUDI_GREEN = rl_colors.HexColor(SAUDI_GREEN)
    RL_GOLD_ACCENT = rl_colors.HexColor(GOLD_ACCENT)
    RL_DARK_GREEN = rl_colors.HexColor(DARK_GREEN)

    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Heading1"],
            fontSize=22,
            textColor=RL_SAUDI_GREEN,
            spaceAfter=6,
            alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=12,
            textColor=RL_GOLD_ACCENT,
            spaceAfter=20,
            alignment=TA_CENTER,
        ),
        "section": ParagraphStyle(
            "ReportSection",
            parent=base["Heading2"],
            fontSize=14,
            textColor=RL_SAUDI_GREEN,
            spaceBefore=15,
            spaceAfter=10,
        ),
        "body": base["Normal"],
        "italic": base["Italic"],
        "footer": ParagraphStyle(
            "ReportFooter",
            parent=base["Normal"],
            fontSize=8,
            textColor=rl_colors.grey,
            alignment=TA_CENTER,
        ),
        "observation": ParagraphStyle(
            "Observation",
            parent=base["Normal"],
            fontSize=10,
            textColor=rl_colors.HexColor("#333333"),
            spaceBefore=4,
            spaceAfter=4,
            leftIndent=12,
            bulletIndent=0,
            bulletFontSize=10,
        ),
    }

    rl_brand = {
        "green": RL_SAUDI_GREEN,
        "gold": RL_GOLD_ACCENT,
        "dark": RL_DARK_GREEN,
    }

    return styles, rl_brand


def _build_pdf_cover_elements(title: str, meta: dict, styles: dict, rl_brand: dict) -> list:
    """
    Build ReportLab flowable elements for the cover / title section.

    Args:
        title: Report title text.
        meta: Metadata dict (already passed through _safe_metadata).
        styles: Style dict from _get_pdf_styles.
        rl_brand: ReportLab color dict from _get_pdf_styles.

    Returns:
        List of ReportLab flowables.
    """
    from reportlab.platypus import Paragraph, Spacer

    elements = [
        Paragraph(title, styles["title"]),
        Paragraph(meta["generated_at"], styles["subtitle"]),
        Spacer(1, 10),
    ]

    info_parts = []
    if meta.get("sport") and meta["sport"] != "All Sports":
        info_parts.append(f"Sport: {meta['sport']}")
    if meta.get("gender") and meta["gender"] != "All":
        info_parts.append(f"Gender: {meta['gender']}")
    if meta.get("date_from"):
        info_parts.append(f"From: {meta['date_from']}")
    if meta.get("date_to"):
        info_parts.append(f"To: {meta['date_to']}")
    if meta.get("athlete_count"):
        info_parts.append(f"Athletes: {meta['athlete_count']}")
    if meta.get("test_count"):
        info_parts.append(f"Tests: {meta['test_count']}")

    if info_parts:
        elements.append(Paragraph("  |  ".join(info_parts), styles["body"]))
        elements.append(Spacer(1, 20))

    return elements


def _build_pdf_table(df: pd.DataFrame, rl_brand: dict, max_rows: int = 30) -> list:
    """
    Build a ReportLab Table flowable from a DataFrame.

    Args:
        df: DataFrame to render.
        rl_brand: ReportLab color dict.
        max_rows: Maximum rows to include (table is truncated beyond this).

    Returns:
        List of ReportLab flowables (may include a truncation note).
    """
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet

    if df is None or df.empty:
        return []

    base_styles = getSampleStyleSheet()
    elements = []
    table_df = df.copy()

    # Format numeric columns
    for col in table_df.select_dtypes(include=["float64", "float32"]).columns:
        table_df[col] = table_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")

    # Format date columns
    for col in table_df.columns:
        if "date" in col.lower():
            try:
                table_df[col] = pd.to_datetime(table_df[col]).dt.strftime("%Y-%m-%d")
            except Exception:
                pass

    if len(table_df) > max_rows:
        table_df = table_df.head(max_rows)
        elements.append(Paragraph(f"(Showing top {max_rows} entries)", base_styles["Italic"]))

    table_data = [table_df.columns.tolist()] + table_df.values.tolist()
    num_cols = len(table_df.columns)
    col_width = 24 * cm / max(num_cols, 1)

    table = Table(table_data, colWidths=[col_width] * num_cols)
    table.setStyle(TableStyle([
        # Header
        ("BACKGROUND", (0, 0), (-1, 0), rl_brand["green"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        # Body
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ALIGN", (0, 1), (-1, -1), "CENTER"),
        # Alternating rows
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#f5f5f5")]),
        # Grid
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#cccccc")),
        ("BOX", (0, 0), (-1, -1), 1, rl_brand["green"]),
        # Padding
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)
    return elements


def _build_pdf_footer_elements(styles: dict) -> list:
    """
    Build the standard footer paragraph for PDF reports.

    Args:
        styles: Style dict from _get_pdf_styles.

    Returns:
        List of ReportLab flowables.
    """
    from reportlab.platypus import Spacer, Paragraph

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    return [
        Spacer(1, 30),
        Paragraph(
            f"Team Saudi Performance Analysis | Confidential | Generated: {now_str}",
            styles["footer"],
        ),
    ]


# ===================================================================
# Group PDF Report
# ===================================================================

def generate_group_pdf_report(
    sport: str,
    charts: Optional[Dict[str, go.Figure]] = None,
    data_tables: Optional[Dict[str, pd.DataFrame]] = None,
    metadata: Optional[dict] = None,
) -> bytes:
    """
    Generate a multi-page landscape A4 PDF report for a group of athletes.

    Page 1: Cover with Team Saudi branding, sport name, filters and summary.
    Pages 2+: One chart per page with section titles.
    Final page: Summary data tables.
    Every page includes a branded footer.

    Args:
        sport: Sport name (e.g. "Fencing", "Karate").
        charts: Map of chart_name -> Plotly figure. Can be None or empty.
        data_tables: Map of table_name -> DataFrame. Can be None or empty.
        metadata: Report metadata dict (see module docstring for keys).

    Returns:
        PDF file contents as bytes.
    """
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Spacer, Paragraph, PageBreak

    charts = charts or {}
    data_tables = data_tables or {}
    meta = _safe_metadata(metadata)

    styles, rl_brand = _get_pdf_styles()

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        leftMargin=1 * cm,
        rightMargin=1 * cm,
        topMargin=1 * cm,
        bottomMargin=1.5 * cm,
    )

    elements = []

    # --- Page 1: Cover ---
    title = f"S&C Group Report: {sport}"
    elements.extend(_build_pdf_cover_elements(title, meta, styles, rl_brand))

    # Summary stats on cover
    summary_lines = [
        f"Athletes: {meta.get('athlete_count', 'N/A')}",
        f"Tests: {meta.get('test_count', 'N/A')}",
    ]
    if meta.get("date_from") and meta.get("date_to"):
        summary_lines.append(f"Date Range: {meta['date_from']} to {meta['date_to']}")
    for line in summary_lines:
        elements.append(Paragraph(line, styles["body"]))
    elements.append(Spacer(1, 12))

    # --- Chart pages ---
    for chart_name, fig in charts.items():
        elements.append(PageBreak())
        elements.append(Paragraph(chart_name, styles["section"]))
        img = _fig_to_reportlab_image(fig, width_cm=26, height_cm=13)
        if img is not None:
            elements.append(img)
        else:
            elements.append(
                Paragraph(f"Chart '{chart_name}' could not be rendered.", styles["body"])
            )
        elements.append(Spacer(1, 10))

    # --- Data table pages ---
    for table_name, df in data_tables.items():
        if df is not None and not df.empty:
            elements.append(PageBreak())
            elements.append(Paragraph(table_name, styles["section"]))
            elements.extend(_build_pdf_table(df, rl_brand))

    # --- Footer ---
    elements.extend(_build_pdf_footer_elements(styles))

    doc.build(elements)
    output.seek(0)
    return output.getvalue()


# ===================================================================
# Group HTML Report
# ===================================================================

def generate_group_html_report(
    sport: str,
    charts: Optional[Dict[str, go.Figure]] = None,
    data_tables: Optional[Dict[str, pd.DataFrame]] = None,
    metadata: Optional[dict] = None,
) -> str:
    """
    Generate a self-contained HTML report for a group of athletes.

    The output uses inline CSS, base64-encoded chart images, and requires no
    external resources so it works offline when opened in any browser.

    Args:
        sport: Sport name.
        charts: Map of chart_name -> Plotly figure.
        data_tables: Map of table_name -> DataFrame.
        metadata: Report metadata dict.

    Returns:
        Complete HTML document as a string.
    """
    charts = charts or {}
    data_tables = data_tables or {}
    meta = _safe_metadata(metadata)

    # Build the metadata info bar
    info_items = []
    if meta.get("sport") and meta["sport"] != "All Sports":
        info_items.append(f"<strong>Sport:</strong> {meta['sport']}")
    if meta.get("gender") and meta["gender"] != "All":
        info_items.append(f"<strong>Gender:</strong> {meta['gender']}")
    if meta.get("athlete_count"):
        info_items.append(f"<strong>Athletes:</strong> {meta['athlete_count']}")
    if meta.get("test_count"):
        info_items.append(f"<strong>Tests:</strong> {meta['test_count']}")
    if meta.get("date_from") and meta.get("date_to"):
        info_items.append(f"<strong>Period:</strong> {meta['date_from']} to {meta['date_to']}")

    info_bar = ""
    if info_items:
        info_bar = (
            f'<div style="display: flex; flex-wrap: wrap; gap: 1.5rem; padding: 1rem; '
            f'background: #f8f9fa; border-radius: 8px; margin-bottom: 1.5rem; '
            f'border-left: 4px solid {GOLD_ACCENT}; font-family: Inter, sans-serif; '
            f'font-size: 0.9rem; color: #333;">'
            + "".join(f"<span>{item}</span>" for item in info_items)
            + "</div>"
        )

    # Build chart sections
    chart_sections = ""
    for chart_name, fig in charts.items():
        b64 = _fig_to_base64(fig, width=900, height=450)
        if b64:
            chart_sections += f"""
            <div style="margin-bottom: 2rem;">
                <div style="background: linear-gradient(135deg, {SAUDI_GREEN} 0%, {DARK_GREEN} 100%);
                     padding: 0.75rem 1rem; border-radius: 8px 8px 0 0;
                     border-left: 4px solid {GOLD_ACCENT};">
                    <h3 style="color: white; margin: 0; font-size: 1rem;">{chart_name}</h3>
                </div>
                <div style="text-align: center; background: white; padding: 1rem;
                     border: 1px solid #e9ecef; border-radius: 0 0 8px 8px;">
                    <img src="data:image/png;base64,{b64}"
                         style="max-width: 100%; height: auto;" alt="{chart_name}">
                </div>
            </div>
            """
        else:
            chart_sections += (
                f'<p style="color: #999; font-style: italic;">Chart "{chart_name}" '
                f"could not be rendered.</p>"
            )

    # Build data tables
    table_sections = ""
    for table_name, df in data_tables.items():
        table_sections += _build_html_table(df, title=table_name)

    # Assemble full HTML document
    header = _build_html_header(
        f"S&C Group Report: {sport}",
        subtitle=f"Generated: {meta['generated_at']}",
    )
    footer = _build_html_footer()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Group Report - {sport} | Team Saudi</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f0f2f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1100px;
            margin: 2rem auto;
            padding: 2rem;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}
        @media print {{
            body {{ background: white; }}
            .container {{ box-shadow: none; margin: 0; padding: 1rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {header}
        {info_bar}
        {chart_sections}
        {table_sections}
        {footer}
    </div>
</body>
</html>"""

    return html


# ===================================================================
# Individual PDF Report
# ===================================================================

def generate_individual_pdf_report(
    athlete_name: str,
    sport: str,
    charts: Optional[Dict[str, go.Figure]] = None,
    data_tables: Optional[Dict[str, pd.DataFrame]] = None,
    observations: Optional[List[str]] = None,
    metadata: Optional[dict] = None,
) -> bytes:
    """
    Generate a landscape A4 PDF report for an individual athlete.

    Includes a branded header, metric summary cards, trend charts,
    observations section, and data tables.

    Args:
        athlete_name: Full name of the athlete.
        sport: Athlete's sport.
        charts: Map of chart_name -> Plotly figure (trend lines, etc.).
        data_tables: Map of table_name -> DataFrame.
        observations: List of auto-generated text observations
            (e.g. "CMJ Height improved 5% since last test").
        metadata: Report metadata dict.

    Returns:
        PDF file contents as bytes.
    """
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Spacer, Paragraph, PageBreak, Table, TableStyle,
    )

    charts = charts or {}
    data_tables = data_tables or {}
    observations = observations or []
    meta = _safe_metadata(metadata)

    styles, rl_brand = _get_pdf_styles()

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        leftMargin=1 * cm,
        rightMargin=1 * cm,
        topMargin=1 * cm,
        bottomMargin=1.5 * cm,
    )

    elements = []

    # --- Cover / Header ---
    elements.append(
        Paragraph(f"Individual Report: {athlete_name}", styles["title"])
    )
    elements.append(
        Paragraph(f"{sport}  |  {meta['generated_at']}", styles["subtitle"])
    )

    # Metric summary cards (4-column layout) from metadata
    summary_keys = [
        ("athlete_count", "Athletes in Squad"),
        ("test_count", "Total Tests"),
    ]
    # Also accept arbitrary "summary_cards" list in metadata
    # Each entry: {"label": str, "value": str}
    summary_cards = meta.get("summary_cards", [])
    if summary_cards:
        card_row = []
        for card in summary_cards[:4]:
            card_row.append(
                f"<b>{card.get('label', '')}</b><br/>{card.get('value', '')}"
            )
        # Pad to 4 columns
        while len(card_row) < 4:
            card_row.append("")

        card_paras = [
            Paragraph(cell, styles["body"]) for cell in card_row
        ]
        card_table = Table([card_paras], colWidths=[6.5 * cm] * 4)
        card_table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 1, rl_brand["green"]),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#cccccc")),
            ("BACKGROUND", (0, 0), (-1, -1), rl_colors.HexColor("#f8f9fa")),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(card_table)
        elements.append(Spacer(1, 16))

    # Filter info
    filter_parts = []
    if meta.get("date_from") and meta.get("date_to"):
        filter_parts.append(f"Period: {meta['date_from']} - {meta['date_to']}")
    if meta.get("gender") and meta["gender"] != "All":
        filter_parts.append(f"Gender: {meta['gender']}")
    if filter_parts:
        elements.append(Paragraph("  |  ".join(filter_parts), styles["body"]))
        elements.append(Spacer(1, 12))

    # --- Observations ---
    if observations:
        elements.append(PageBreak())
        elements.append(Paragraph("Key Observations", styles["section"]))
        for obs in observations:
            # Use bullet character
            elements.append(
                Paragraph(f"\u2022  {obs}", styles["observation"])
            )
        elements.append(Spacer(1, 16))

    # --- Charts ---
    for chart_name, fig in charts.items():
        elements.append(PageBreak())
        elements.append(Paragraph(chart_name, styles["section"]))
        img = _fig_to_reportlab_image(fig, width_cm=26, height_cm=13)
        if img is not None:
            elements.append(img)
        else:
            elements.append(
                Paragraph(f"Chart '{chart_name}' could not be rendered.", styles["body"])
            )
        elements.append(Spacer(1, 10))

    # --- Data tables ---
    for table_name, df in data_tables.items():
        if df is not None and not df.empty:
            elements.append(PageBreak())
            elements.append(Paragraph(table_name, styles["section"]))
            elements.extend(_build_pdf_table(df, rl_brand))

    # --- Footer ---
    elements.extend(_build_pdf_footer_elements(styles))

    doc.build(elements)
    output.seek(0)
    return output.getvalue()


# ===================================================================
# Individual HTML Report
# ===================================================================

def generate_individual_html_report(
    athlete_name: str,
    sport: str,
    charts: Optional[Dict[str, go.Figure]] = None,
    data_tables: Optional[Dict[str, pd.DataFrame]] = None,
    observations: Optional[List[str]] = None,
    metadata: Optional[dict] = None,
) -> str:
    """
    Generate a self-contained HTML report for an individual athlete.

    Uses inline CSS and base64-encoded images. No external dependencies.

    Args:
        athlete_name: Full name of the athlete.
        sport: Athlete's sport.
        charts: Map of chart_name -> Plotly figure.
        data_tables: Map of table_name -> DataFrame.
        observations: List of text observations.
        metadata: Report metadata dict.

    Returns:
        Complete HTML document as a string.
    """
    charts = charts or {}
    data_tables = data_tables or {}
    observations = observations or []
    meta = _safe_metadata(metadata)

    # Build metadata strip
    info_items = []
    if sport:
        info_items.append(f"<strong>Sport:</strong> {sport}")
    if meta.get("gender") and meta["gender"] != "All":
        info_items.append(f"<strong>Gender:</strong> {meta['gender']}")
    if meta.get("date_from") and meta.get("date_to"):
        info_items.append(
            f"<strong>Period:</strong> {meta['date_from']} to {meta['date_to']}"
        )
    if meta.get("test_count"):
        info_items.append(f"<strong>Tests:</strong> {meta['test_count']}")

    info_bar = ""
    if info_items:
        info_bar = (
            f'<div style="display: flex; flex-wrap: wrap; gap: 1.5rem; padding: 1rem; '
            f'background: #f8f9fa; border-radius: 8px; margin-bottom: 1.5rem; '
            f'border-left: 4px solid {GOLD_ACCENT}; font-family: Inter, sans-serif; '
            f'font-size: 0.9rem; color: #333;">'
            + "".join(f"<span>{item}</span>" for item in info_items)
            + "</div>"
        )

    # Summary cards from metadata
    cards_html = ""
    summary_cards = meta.get("summary_cards", [])
    if summary_cards:
        card_items = ""
        for card in summary_cards[:6]:
            label = card.get("label", "")
            value = card.get("value", "")
            status = card.get("status", "neutral")
            color_map = {
                "excellent": SAUDI_GREEN,
                "good": LIGHT_GREEN,
                "warning": STATUS_WARNING,
                "danger": STATUS_DANGER,
                "gold": GOLD_ACCENT,
                "neutral": "#6c757d",
            }
            bg = color_map.get(status, "#6c757d")
            card_items += (
                f'<div style="background: {bg}; padding: 1rem; border-radius: 8px; '
                f'text-align: center; flex: 1; min-width: 120px;">'
                f'<p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.8rem;">'
                f'{label}</p>'
                f'<p style="color: white; margin: 0.25rem 0 0 0; font-size: 1.3rem; '
                f'font-weight: bold;">{value}</p></div>'
            )
        cards_html = (
            f'<div style="display: flex; gap: 1rem; flex-wrap: wrap; '
            f'margin-bottom: 1.5rem;">{card_items}</div>'
        )

    # Observations section
    observations_html = ""
    if observations:
        obs_items = "".join(
            f'<li style="margin-bottom: 0.5rem; color: #333;">{obs}</li>'
            for obs in observations
        )
        observations_html = f"""
        <div style="margin-bottom: 2rem;">
            <div style="background: linear-gradient(135deg, {SAUDI_GREEN} 0%, {DARK_GREEN} 100%);
                 padding: 0.75rem 1rem; border-radius: 8px 8px 0 0;
                 border-left: 4px solid {GOLD_ACCENT};">
                <h3 style="color: white; margin: 0; font-size: 1rem;">Key Observations</h3>
            </div>
            <div style="background: white; padding: 1rem 1.5rem;
                 border: 1px solid #e9ecef; border-radius: 0 0 8px 8px;">
                <ul style="padding-left: 1.2rem; font-size: 0.9rem; line-height: 1.8;">
                    {obs_items}
                </ul>
            </div>
        </div>
        """

    # Chart sections
    chart_sections = ""
    for chart_name, fig in charts.items():
        b64 = _fig_to_base64(fig, width=900, height=450)
        if b64:
            chart_sections += f"""
            <div style="margin-bottom: 2rem;">
                <div style="background: linear-gradient(135deg, {SAUDI_GREEN} 0%, {DARK_GREEN} 100%);
                     padding: 0.75rem 1rem; border-radius: 8px 8px 0 0;
                     border-left: 4px solid {GOLD_ACCENT};">
                    <h3 style="color: white; margin: 0; font-size: 1rem;">{chart_name}</h3>
                </div>
                <div style="text-align: center; background: white; padding: 1rem;
                     border: 1px solid #e9ecef; border-radius: 0 0 8px 8px;">
                    <img src="data:image/png;base64,{b64}"
                         style="max-width: 100%; height: auto;" alt="{chart_name}">
                </div>
            </div>
            """
        else:
            chart_sections += (
                f'<p style="color: #999; font-style: italic;">Chart "{chart_name}" '
                f"could not be rendered.</p>"
            )

    # Data tables
    table_sections = ""
    for table_name, df in data_tables.items():
        table_sections += _build_html_table(df, title=table_name)

    # Assemble
    header = _build_html_header(
        f"Individual Report: {athlete_name}",
        subtitle=f"{sport}  |  {meta['generated_at']}",
    )
    footer = _build_html_footer()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Individual Report - {athlete_name} | Team Saudi</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f0f2f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1100px;
            margin: 2rem auto;
            padding: 2rem;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}
        @media print {{
            body {{ background: white; }}
            .container {{ box-shadow: none; margin: 0; padding: 1rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {header}
        {info_bar}
        {cards_html}
        {observations_html}
        {chart_sections}
        {table_sections}
        {footer}
    </div>
</body>
</html>"""

    return html
