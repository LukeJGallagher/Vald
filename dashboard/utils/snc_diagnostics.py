"""
S&C Diagnostics Canvas Module

Provides comprehensive S&C testing visualizations:
- Group Ranked Bar Charts with squad average and benchmark lines
- Individual Line Charts with squad average and multi-athlete selection
- Unilateral Side-by-Side Charts with asymmetry flagging
- Multi-Line Strength RM Charts

Test Types:
- Tier 1: IMTP, CMJ, 6 Minute Aerobic
- Tier 2: SL ISO Squat, Strength RM, SL CMJ, Broad Jump, 10:5 Hop, Peak Power, Repeat Power, Glycolytic Power

Benchmarks are loaded from the benchmark_database module which provides:
- VALD normative data as defaults
- Editable benchmarks with audit trail
- Gender-specific values
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import os
import io

# Import benchmark database for VALD norms
try:
    from .benchmark_database import (
        load_benchmarks,
        get_benchmark_for_test,
        get_asymmetry_threshold,
        get_injury_threshold,
        get_benchmark_status,
        get_status_color,
        get_status_emoji,
        VALD_NORMS
    )
    BENCHMARK_DB_AVAILABLE = True
except ImportError:
    BENCHMARK_DB_AVAILABLE = False
    VALD_NORMS = {}

# Team Saudi Brand Colors
TEAL_PRIMARY = '#1D4D3B'
GOLD_ACCENT = '#a08e66'
TEAL_DARK = '#153829'
TEAL_LIGHT = '#4CAF50'
GRAY_BLUE = '#78909C'
CORAL_ACCENT = '#FF6B6B'

# Chart colors for multiple athletes/lines
MULTI_LINE_COLORS = [
    '#1D4D3B',  # Teal
    '#FF6B6B',  # Coral
    '#0077B6',  # Blue
    '#a08e66',  # Gold
    '#9C27B0',  # Purple
    '#FF9800',  # Orange
]

# Squad average styling
SQUAD_AVG_COLOR = '#005430'  # Saudi Green
BENCHMARK_COLOR = '#0077B6'  # Blue dashed


# =====================
# Export Helper Functions
# =====================
def export_to_excel(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    """Export DataFrame to Excel format and return as bytes."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.getvalue()


def get_persisted_athlete_selection(key: str, available_athletes: List[str]) -> List[str]:
    """
    Get athlete selection with session state persistence.

    Prevents page resets by:
    1. Checking for previous selection in session_state
    2. Filtering to only athletes still in current filtered list
    3. Only defaulting to first athlete if no valid previous selection

    Args:
        key: Unique session state key for this multiselect
        available_athletes: List of athletes available in current filtered data

    Returns:
        List of athlete names to use as default
    """
    if not available_athletes:
        return []

    # Check for previous selection
    prev_selection = st.session_state.get(key, [])

    if prev_selection:
        # Filter to only athletes still in the available list
        valid_selection = [a for a in prev_selection if a in available_athletes]
        if valid_selection:
            return valid_selection

    # No valid previous selection, default to first athlete
    return [available_athletes[0]]


def get_persisted_selectbox_index(key: str, options: List[str], default_index: int = 0) -> int:
    """
    Get selectbox index with persistence across filter changes.

    Prevents page resets by preserving the previously selected option
    if it still exists in the current options list.

    Args:
        key: Unique session state key for tracking selection
        options: List of available options
        default_index: Default index if no valid previous selection

    Returns:
        Index of the option to select
    """
    if not options:
        return default_index

    # Check for previous selection
    prev_value = st.session_state.get(key)

    if prev_value is not None and prev_value in options:
        return options.index(prev_value)

    return min(default_index, len(options) - 1)


def export_group_summary(df: pd.DataFrame, metric_col: str, name_col: str = 'Name',
                         test_name: str = "Test") -> bytes:
    """
    Export group summary (latest values for all athletes) to Excel.

    Args:
        df: DataFrame with athlete data
        metric_col: Column name for the metric to summarize
        name_col: Column name for athlete names
        test_name: Name of the test for the sheet name

    Returns:
        Excel file as bytes
    """
    if df.empty:
        return export_to_excel(pd.DataFrame(columns=['Athlete', 'Latest Value', 'Date']))

    # Get latest value for each athlete
    if 'date' in df.columns:
        summary_df = df.sort_values('date').groupby(name_col).last().reset_index()
    else:
        summary_df = df.groupby(name_col).last().reset_index()

    # Select and rename relevant columns
    export_cols = {name_col: 'Athlete'}
    if metric_col in summary_df.columns:
        export_cols[metric_col] = 'Latest Value'
    if 'date' in summary_df.columns:
        export_cols['date'] = 'Date'
    if 'athlete_sport' in summary_df.columns:
        export_cols['athlete_sport'] = 'Sport'
    if 'exercise' in summary_df.columns:
        export_cols['exercise'] = 'Exercise'

    # Filter and rename
    available_cols = [c for c in export_cols.keys() if c in summary_df.columns]
    export_df = summary_df[available_cols].rename(columns={k: v for k, v in export_cols.items() if k in available_cols})

    # Sort by value descending (best performers first)
    if 'Latest Value' in export_df.columns:
        export_df = export_df.sort_values('Latest Value', ascending=False)

    return export_to_excel(export_df, sheet_name=f"{test_name}_Group")


def export_individual_data(df: pd.DataFrame, athletes: List[str], metric_col: str,
                          name_col: str = 'Name', test_name: str = "Test") -> bytes:
    """
    Export individual athlete time-series data to Excel.

    Args:
        df: DataFrame with athlete data
        athletes: List of athlete names to include
        metric_col: Column name for the metric
        name_col: Column name for athlete names
        test_name: Name of the test for the sheet name

    Returns:
        Excel file as bytes
    """
    if df.empty or not athletes:
        return export_to_excel(pd.DataFrame(columns=['Athlete', 'Date', 'Value']))

    # Filter to selected athletes
    ind_df = df[df[name_col].isin(athletes)].copy()

    # Select and rename relevant columns
    export_cols = {name_col: 'Athlete'}
    if 'date' in ind_df.columns:
        export_cols['date'] = 'Date'
    if metric_col in ind_df.columns:
        export_cols[metric_col] = 'Value'
    if 'athlete_sport' in ind_df.columns:
        export_cols['athlete_sport'] = 'Sport'
    if 'exercise' in ind_df.columns:
        export_cols['exercise'] = 'Exercise'
    if 'reps' in ind_df.columns:
        export_cols['reps'] = 'Reps'
    if 'sets' in ind_df.columns:
        export_cols['sets'] = 'Sets'
    if 'weight_kg' in ind_df.columns:
        export_cols['weight_kg'] = 'Weight (kg)'
    if 'rpe' in ind_df.columns:
        export_cols['rpe'] = 'RPE'

    # Filter and rename
    available_cols = [c for c in export_cols.keys() if c in ind_df.columns]
    export_df = ind_df[available_cols].rename(columns={k: v for k, v in export_cols.items() if k in available_cols})

    # Sort by athlete then date
    sort_cols = []
    if 'Athlete' in export_df.columns:
        sort_cols.append('Athlete')
    if 'Date' in export_df.columns:
        sort_cols.append('Date')
    if sort_cols:
        export_df = export_df.sort_values(sort_cols)

    return export_to_excel(export_df, sheet_name=f"{test_name}_Individual")


def render_export_buttons(group_df: pd.DataFrame, individual_df: pd.DataFrame,
                         selected_athletes: List[str], metric_col: str,
                         name_col: str = 'Name', test_name: str = "Test",
                         key_prefix: str = "export"):
    """
    Render export buttons for group and individual data.

    Args:
        group_df: DataFrame for group summary (latest values)
        individual_df: DataFrame for individual time-series
        selected_athletes: List of selected athlete names for individual export
        metric_col: Column name for the metric
        name_col: Column name for athlete names
        test_name: Name of the test
        key_prefix: Unique prefix for widget keys
    """
    st.markdown("---")
    st.markdown("##### Export Data")

    col1, col2 = st.columns(2)

    with col1:
        # Group summary export
        if not group_df.empty:
            excel_data = export_group_summary(group_df, metric_col, name_col, test_name)
            st.download_button(
                label="📥 Export Group Summary",
                data=excel_data,
                file_name=f"{test_name.replace(' ', '_')}_group_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"{key_prefix}_group_export"
            )
            st.caption(f"All athletes - latest values")
        else:
            st.info("No group data to export")

    with col2:
        # Individual data export
        if not individual_df.empty and selected_athletes:
            excel_data = export_individual_data(individual_df, selected_athletes, metric_col, name_col, test_name)
            st.download_button(
                label="📥 Export Individual Data",
                data=excel_data,
                file_name=f"{test_name.replace(' ', '_')}_individual_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"{key_prefix}_individual_export"
            )
            st.caption(f"{len(selected_athletes)} athlete(s) - time-series")
        else:
            if not selected_athletes:
                st.info("Select athletes for individual export")
            else:
                st.info("No individual data to export")


# =====================
# PDF Export Functions
# =====================
def export_chart_to_image(fig: go.Figure, width: int = 800, height: int = 500) -> bytes:
    """Convert Plotly figure to PNG image bytes."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception as e:
        print(f"Error exporting chart to image: {e}")
        return None


def export_to_pdf_with_chart(fig: go.Figure, df: pd.DataFrame, test_name: str,
                             subtitle: str = "", metric_col: str = None,
                             name_col: str = 'Name') -> bytes:
    """
    Export chart and data table to PDF with Team Saudi branding.

    Args:
        fig: Plotly figure to include
        df: DataFrame with data to include as table
        test_name: Name of the test for the title
        subtitle: Additional subtitle text
        metric_col: Column name for the metric (for formatting)
        name_col: Column name for athlete names

    Returns:
        PDF file as bytes
    """
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    # Team Saudi colors
    SAUDI_GREEN = colors.HexColor('#005430')
    GOLD_ACCENT = colors.HexColor('#a08e66')
    DARK_GREEN = colors.HexColor('#003d1f')

    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=landscape(A4),
                           leftMargin=1*cm, rightMargin=1*cm,
                           topMargin=1*cm, bottomMargin=1*cm)

    styles = getSampleStyleSheet()
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=SAUDI_GREEN,
        spaceAfter=6,
        alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=GOLD_ACCENT,
        spaceAfter=20,
        alignment=TA_CENTER
    )
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=SAUDI_GREEN,
        spaceBefore=15,
        spaceAfter=10
    )

    elements = []

    # Title
    elements.append(Paragraph(f"S&C Diagnostics Report: {test_name}", title_style))
    if subtitle:
        elements.append(Paragraph(subtitle, subtitle_style))
    else:
        # Add date
        from datetime import datetime
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_style))

    # Chart section
    if fig is not None:
        elements.append(Paragraph("Performance Chart", section_style))
        try:
            chart_bytes = export_chart_to_image(fig, width=900, height=450)
            if chart_bytes:
                chart_image = Image(io.BytesIO(chart_bytes), width=24*cm, height=12*cm)
                elements.append(chart_image)
        except Exception as e:
            elements.append(Paragraph(f"Chart could not be rendered: {str(e)}", styles['Normal']))

    elements.append(Spacer(1, 20))

    # Data table section
    if df is not None and not df.empty:
        elements.append(Paragraph("Data Summary", section_style))

        # Prepare table data
        table_df = df.copy()

        # Format numeric columns
        for col in table_df.select_dtypes(include=['float64', 'float32']).columns:
            table_df[col] = table_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")

        # Format date columns
        for col in table_df.columns:
            if 'date' in col.lower():
                try:
                    table_df[col] = pd.to_datetime(table_df[col]).dt.strftime('%Y-%m-%d')
                except:
                    pass

        # Limit to reasonable number of rows for PDF
        if len(table_df) > 20:
            table_df = table_df.head(20)
            elements.append(Paragraph("(Showing top 20 entries)", styles['Italic']))

        # Create table
        table_data = [table_df.columns.tolist()] + table_df.values.tolist()

        # Calculate column widths
        num_cols = len(table_df.columns)
        col_width = 24*cm / num_cols if num_cols > 0 else 4*cm

        table = Table(table_data, colWidths=[col_width] * num_cols)
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), SAUDI_GREEN),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            # Body styling
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('BOX', (0, 0), (-1, -1), 1, SAUDI_GREEN),
            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(table)

    # Footer
    elements.append(Spacer(1, 30))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    elements.append(Paragraph("Team Saudi Performance Analysis | Confidential", footer_style))

    doc.build(elements)
    output.seek(0)
    return output.getvalue()


def render_export_buttons_with_pdf(group_df: pd.DataFrame, individual_df: pd.DataFrame,
                                   selected_athletes: List[str], metric_col: str,
                                   group_fig: go.Figure = None, individual_fig: go.Figure = None,
                                   name_col: str = 'Name', test_name: str = "Test",
                                   key_prefix: str = "export"):
    """
    Render export buttons with Excel and PDF options.

    Args:
        group_df: DataFrame for group summary
        individual_df: DataFrame for individual time-series
        selected_athletes: List of selected athlete names
        metric_col: Column name for the metric
        group_fig: Plotly figure for group chart (optional)
        individual_fig: Plotly figure for individual chart (optional)
        name_col: Column name for athlete names
        test_name: Name of the test
        key_prefix: Unique prefix for widget keys
    """
    st.markdown("---")
    st.markdown("##### Export Data")

    # Format selector
    export_format = st.radio(
        "Export Format:",
        ["Excel (.xlsx)", "PDF with Chart"],
        horizontal=True,
        key=f"{key_prefix}_format"
    )

    col1, col2 = st.columns(2)

    if export_format == "Excel (.xlsx)":
        with col1:
            if not group_df.empty:
                excel_data = export_group_summary(group_df, metric_col, name_col, test_name)
                st.download_button(
                    label="📥 Export Group Summary",
                    data=excel_data,
                    file_name=f"{test_name.replace(' ', '_')}_group_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"{key_prefix}_group_excel"
                )
                st.caption("All athletes - latest values")
            else:
                st.info("No group data to export")

        with col2:
            if not individual_df.empty and selected_athletes:
                excel_data = export_individual_data(individual_df, selected_athletes, metric_col, name_col, test_name)
                st.download_button(
                    label="📥 Export Individual Data",
                    data=excel_data,
                    file_name=f"{test_name.replace(' ', '_')}_individual_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"{key_prefix}_individual_excel"
                )
                st.caption(f"{len(selected_athletes)} athlete(s) - time-series")
            else:
                st.info("Select athletes for individual export")

    else:  # PDF with Chart
        with col1:
            if not group_df.empty:
                # Prepare summary data for PDF
                if 'date' in group_df.columns:
                    summary_df = group_df.sort_values('date').groupby(name_col).last().reset_index()
                else:
                    summary_df = group_df.groupby(name_col).last().reset_index()

                if metric_col in summary_df.columns:
                    summary_df = summary_df.sort_values(metric_col, ascending=False)

                # Select columns for PDF table
                pdf_cols = [name_col]
                if metric_col in summary_df.columns:
                    pdf_cols.append(metric_col)
                if 'date' in summary_df.columns:
                    pdf_cols.append('date')
                if 'athlete_sport' in summary_df.columns:
                    pdf_cols.append('athlete_sport')

                pdf_df = summary_df[[c for c in pdf_cols if c in summary_df.columns]].copy()
                pdf_df.columns = ['Athlete', 'Value', 'Date', 'Sport'][:len(pdf_df.columns)]

                pdf_data = export_to_pdf_with_chart(
                    group_fig, pdf_df, test_name,
                    subtitle="Group Comparison - Latest Values"
                )
                st.download_button(
                    label="📄 Export Group PDF",
                    data=pdf_data,
                    file_name=f"{test_name.replace(' ', '_')}_group_report.pdf",
                    mime="application/pdf",
                    key=f"{key_prefix}_group_pdf"
                )
                st.caption("Chart + data table")
            else:
                st.info("No group data to export")

        with col2:
            if not individual_df.empty and selected_athletes:
                # Filter individual data
                ind_pdf_df = individual_df[individual_df[name_col].isin(selected_athletes)].copy()

                # Select columns
                pdf_cols = [name_col, 'date', metric_col]
                ind_pdf_df = ind_pdf_df[[c for c in pdf_cols if c in ind_pdf_df.columns]]
                ind_pdf_df.columns = ['Athlete', 'Date', 'Value'][:len(ind_pdf_df.columns)]

                if 'Date' in ind_pdf_df.columns:
                    ind_pdf_df = ind_pdf_df.sort_values(['Athlete', 'Date'])

                pdf_data = export_to_pdf_with_chart(
                    individual_fig, ind_pdf_df, test_name,
                    subtitle=f"Individual Progression - {', '.join(selected_athletes[:3])}{'...' if len(selected_athletes) > 3 else ''}"
                )
                st.download_button(
                    label="📄 Export Individual PDF",
                    data=pdf_data,
                    file_name=f"{test_name.replace(' ', '_')}_individual_report.pdf",
                    mime="application/pdf",
                    key=f"{key_prefix}_individual_pdf"
                )
                st.caption(f"{len(selected_athletes)} athlete(s) - time-series")
            else:
                st.info("Select athletes for individual export")


# Test configurations
# Supports both legacy API format and local_sync format column names
TEST_CONFIG = {
    'IMTP': {
        'tier': 1,
        'group_chart': 'ranked_bar',
        'individual_chart': 'line_squad_avg',
        'metric1': ['ISO_BM_REL_FORCE_PEAK', 'Peak Force / BM_Trial'],  # local_sync, legacy
        'metric1_name': 'Relative Peak Force',
        'unit1': 'N/kg',
        'metric2': None,
        'source': 'VALD',
        'test_type_filter': ['IMTP', 'ISOT']
    },
    'CMJ': {
        'tier': 1,
        'group_chart': 'ranked_bar',
        'individual_chart': 'line_squad_avg',
        'metric1': ['BODYMASS_RELATIVE_TAKEOFF_POWER', 'Peak Power / BM_Trial'],
        'metric1_name': 'Relative Peak Power',
        'unit1': 'W/kg',
        'metric2': ['JUMP_HEIGHT_IMP_MOM', 'Jump Height (Imp-Mom)_Trial'],
        'metric2_name': 'Height (Impulse-Mom)',
        'unit2': 'cm',
        'metric2_multiplier': 1,  # Already in cm from local_sync
        'source': 'VALD',
        'test_type_filter': ['CMJ', 'ABCMJ']
    },
    '6_Min_Aerobic': {
        'tier': 1,
        'group_chart': 'ranked_bar',
        'individual_chart': 'line_squad_avg',
        'metric1': 'avg_relative_wattage',
        'metric1_name': 'Average Relative Wattage',
        'unit1': 'W/kg',
        'metric2': 'vo2max_estimated',
        'metric2_name': 'VO2max Estimated',
        'unit2': 'ml/kg/min',
        'source': 'Manual',
        'data_source': 'aerobic_tests'
    },
    'SL_ISO_Squat': {
        'tier': 2,
        'group_chart': 'ranked_side_by_side',
        'individual_chart': 'dual_line_diff',
        'metric1_left': ['PEAK_VERTICAL_FORCE_Left', 'Peak Force / BM_Left'],
        'metric1_right': ['PEAK_VERTICAL_FORCE_Right', 'Peak Force / BM_Right'],
        'metric1_name': 'Peak Force',
        'unit1': 'N',
        'metric2': 'asymmetry',
        'metric2_name': '% Difference',
        'unit2': '%',
        'source': 'VALD',
        'test_type_filter': ['SLISOSQT', 'ISOSQT']
    },
    'SL_IMTP': {
        'tier': 2,
        'group_chart': 'ranked_side_by_side',
        'individual_chart': 'dual_line_diff',
        'metric1_left': ['PEAK_VERTICAL_FORCE_Left', 'Peak Force / BM_Left'],
        'metric1_right': ['PEAK_VERTICAL_FORCE_Right', 'Peak Force / BM_Right'],
        'metric1_name': 'Peak Force',
        'unit1': 'N',
        'metric2': 'asymmetry',
        'metric2_name': '% Difference',
        'unit2': '%',
        'source': 'VALD',
        'test_type_filter': ['SLIMTP', 'SLISOT']
    },
    'Strength_RM': {
        'tier': 2,
        'group_chart': 'ranked_bar',
        'individual_chart': 'multi_line',
        'metric1': 'relative_1rm',  # estimated_1rm / body_mass
        'metric1_name': 'Relative Strength',
        'unit1': 'kg/BM',
        'metric2': 'estimated_1rm',
        'metric2_name': 'Absolute Strength',
        'unit2': 'kg',
        'source': 'Manual',
        'data_source': 'sc_lower_body'
    },
    'SL_CMJ': {
        'tier': 2,
        'group_chart': 'ranked_side_by_side',
        'individual_chart': 'dual_line_diff',
        'metric1_left': ['BODYMASS_RELATIVE_TAKEOFF_POWER_Left', 'Peak Power / BM_Left'],
        'metric1_right': ['BODYMASS_RELATIVE_TAKEOFF_POWER_Right', 'Peak Power / BM_Right'],
        'metric1_name': 'Relative Peak Power',
        'unit1': 'W/kg',
        'metric2': ['JUMP_HEIGHT_IMP_MOM', 'Jump Height (Imp-Mom)_Trial'],
        'metric2_name': 'Height (Impulse-Mom)',
        'unit2': 'cm',
        'source': 'VALD',
        'test_type_filter': ['SLJ', 'SLCMRJ']
    },
    'Broad_Jump': {
        'tier': 2,
        'group_chart': 'ranked_bar',
        'individual_chart': 'line_squad_avg',
        'metric1': 'distance_cm',
        'metric1_name': 'Distance',
        'unit1': 'cm',
        'metric2': None,
        'source': 'Manual',
        'data_source': 'broad_jump'
    },
    '10_5_Hop': {
        'tier': 2,
        'group_chart': 'ranked_bar',
        'individual_chart': 'line_squad_avg',
        'metric1': ['HOP_BEST_RSI', 'RSI-modified_Trial'],  # Use HOP_BEST_RSI (already correct units)
        'metric1_name': 'RSI',
        'unit1': '',
        'metric2': None,
        'source': 'VALD',
        'test_type_filter': ['RSHIP', 'RSKIP', 'RSAIP']
    },
    'Peak_Power_10s': {
        'tier': 2,
        'group_chart': 'ranked_bar',
        'individual_chart': 'line_squad_avg',
        'metric1': 'peak_relative_wattage',
        'metric1_name': 'Peak Relative Wattage',
        'unit1': 'W/kg',
        'metric2': None,
        'source': 'Manual',
        'data_source': 'power_tests'
    },
    'Repeat_Power': {
        'tier': 2,
        'group_chart': 'ranked_bar',
        'individual_chart': 'line_squad_avg',
        'metric1': 'peak_relative_wattage',
        'metric1_name': 'Peak Relative Wattage',
        'unit1': 'W/kg',
        'metric2': 'fade_percent',
        'metric2_name': '% Fade (Rep 1-6)',
        'unit2': '%',
        'source': 'Manual',
        'data_source': 'power_tests'
    },
    'Glycolytic_Power': {
        'tier': 2,
        'group_chart': 'ranked_bar',
        'individual_chart': 'line_squad_avg',
        'metric1': 'peak_relative_wattage',
        'metric1_name': 'Peak Relative Wattage',
        'unit1': 'W/kg',
        'metric2': None,
        'source': 'Manual',
        'data_source': 'power_tests'
    },
    'NordBord': {
        'tier': 2,
        'group_chart': 'ranked_side_by_side',
        'individual_chart': 'dual_line_diff',
        'metric1_left': 'leftMaxForce',
        'metric1_right': 'rightMaxForce',
        'metric1_name': 'Max Force',
        'unit1': 'N',
        'metric2': 'asymmetry',
        'metric2_name': '% Difference',
        'unit2': '%',
        'source': 'VALD',
        'data_source': 'nordbord'
    }
}

# Metric column resolver - finds actual column from list of possible names
def resolve_metric_column(df: pd.DataFrame, metric_spec, require_data: bool = True) -> Optional[str]:
    """Find actual column name from a metric specification (string or list of possibilities).

    Args:
        df: DataFrame to search
        metric_spec: Column name (str) or list of possible column names
        require_data: If True, only return columns that have at least one non-null value.
                      This prevents selecting a column that exists but has no data for the
                      filtered subset (e.g., JUMP_HEIGHT_IMP_MOM exists in header but is
                      NaN for ABCMJ tests).
    """
    if metric_spec is None:
        return None
    candidates = [metric_spec] if isinstance(metric_spec, str) else metric_spec if isinstance(metric_spec, list) else []
    for col in candidates:
        if col in df.columns:
            if require_data and df[col].notna().sum() == 0:
                continue  # Column exists but has no data - try next candidate
            return col
    return None

# Default benchmarks - These are fallbacks if benchmark database not available
# Actual benchmarks are loaded from benchmark_database.py which uses VALD norms
DEFAULT_BENCHMARKS = {
    'IMTP': {'benchmark': 35.0, 'unit': 'N/kg'},
    'CMJ': {'benchmark': 50.0, 'unit': 'W/kg'},
    '6_Min_Aerobic': {'benchmark': 3.0, 'unit': 'W/kg'},  # Typical recreational cyclist
    'SL_ISO_Squat': {'benchmark': 1200.0, 'unit': 'N'},  # Typical bilateral target per leg
    'SL_IMTP': {'benchmark': 1200.0, 'unit': 'N'},
    'Strength_RM': {'benchmark': 1.5, 'unit': 'kg/BM'},
    'SL_CMJ': {'benchmark': 25.0, 'unit': 'W/kg'},
    'Broad_Jump': {'benchmark': 250.0, 'unit': 'cm'},
    '10_5_Hop': {'benchmark': 2.0, 'unit': ''},  # RSI (absolute)
    'Peak_Power_10s': {'benchmark': 10.0, 'unit': 'W/kg'},
    'Repeat_Power': {'benchmark': 8.0, 'unit': 'W/kg'},
    'Glycolytic_Power': {'benchmark': 6.0, 'unit': 'W/kg'},
    'NordBord': {'benchmark': 337.0, 'unit': 'N'},  # Injury risk threshold
}


def get_benchmark_value(test_type: str, metric: str = None, gender: str = "male") -> float:
    """
    Get benchmark value from database or fallback to defaults.

    Args:
        test_type: Test type code (CMJ, IMTP, etc.)
        metric: Specific metric column name (optional)
        gender: 'male' or 'female'

    Returns:
        Benchmark value (uses 'good' level from database)
    """
    if BENCHMARK_DB_AVAILABLE:
        # Try to get from database with specific metric
        if metric:
            benchmark = get_benchmark_for_test(test_type, metric, gender, "good")
            if benchmark is not None:
                return benchmark

        # If no metric specified, try to get first available metric from test type
        benchmarks = load_benchmarks()
        if test_type in benchmarks:
            metrics = benchmarks[test_type].get("metrics", {})
            for metric_key, metric_config in metrics.items():
                gender_values = metric_config.get(gender.lower(), {})
                if "good" in gender_values:
                    return gender_values["good"]

    # Fallback to simple defaults
    return DEFAULT_BENCHMARKS.get(test_type, {}).get('benchmark', 0)


def render_filters(df: pd.DataFrame, key_prefix: str = "snc") -> Tuple[pd.DataFrame, str, str]:
    """
    Render gender and date filters for S&C diagnostics.
    Returns filtered dataframe, selected gender, and date range.
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        # Sport/Group filter with state persistence
        sports = ['All']
        if 'athlete_sport' in df.columns:
            sports += sorted([s for s in df['athlete_sport'].dropna().unique()])
        sport_idx = get_persisted_selectbox_index(f"{key_prefix}_sport", sports, 0)
        selected_sport = st.selectbox("Sport/Group:", sports, index=sport_idx, key=f"{key_prefix}_sport")

    with col2:
        # Gender filter with state persistence
        # Hardcode options since athlete_sex may not exist in data
        genders = ['All', 'Male', 'Female']
        gender_idx = get_persisted_selectbox_index(f"{key_prefix}_gender", genders, 0)
        selected_gender = st.selectbox("Gender:", genders, index=gender_idx, key=f"{key_prefix}_gender")

    with col3:
        # Date filter with state persistence
        date_options = ['Most Recent', 'Last 7 Days', 'Last 30 Days', 'Last 90 Days', 'All Time', 'Custom Range']
        date_idx = get_persisted_selectbox_index(f"{key_prefix}_date", date_options, 0)
        selected_date = st.selectbox("Date Range:", date_options, index=date_idx, key=f"{key_prefix}_date")

    # Apply filters
    filtered_df = df.copy()

    if selected_sport != 'All' and 'athlete_sport' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['athlete_sport'] == selected_sport]

    if selected_gender != 'All' and 'athlete_sex' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['athlete_sex'] == selected_gender]

    # Date filtering
    if 'recordedDateUtc' in filtered_df.columns:
        filtered_df['recordedDateUtc'] = pd.to_datetime(filtered_df['recordedDateUtc'], errors='coerce')

        # Remove timezone info for comparison (avoids timezone-aware vs naive errors)
        if filtered_df['recordedDateUtc'].dt.tz is not None:
            filtered_df['recordedDateUtc'] = filtered_df['recordedDateUtc'].dt.tz_localize(None)

        if selected_date == 'Most Recent':
            # Get most recent test per athlete
            if 'Name' in filtered_df.columns:
                idx = filtered_df.groupby('Name')['recordedDateUtc'].idxmax()
                filtered_df = filtered_df.loc[idx]
        elif selected_date == 'Last 7 Days':
            cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=7)
            filtered_df = filtered_df[filtered_df['recordedDateUtc'] >= cutoff]
        elif selected_date == 'Last 30 Days':
            cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=30)
            filtered_df = filtered_df[filtered_df['recordedDateUtc'] >= cutoff]
        elif selected_date == 'Last 90 Days':
            cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=90)
            filtered_df = filtered_df[filtered_df['recordedDateUtc'] >= cutoff]

    return filtered_df, selected_sport, selected_gender


def render_benchmark_input(test_key: str, key_prefix: str = "snc", gender: str = "male") -> float:
    """
    Render benchmark display with value from VALD norms database.

    Benchmarks are now managed through the Benchmark Settings tab.
    This shows the current benchmark value from the database.
    """
    default = DEFAULT_BENCHMARKS.get(test_key, {}).get('benchmark', 0)
    unit = DEFAULT_BENCHMARKS.get(test_key, {}).get('unit', '')

    # Try to get from benchmark database
    if BENCHMARK_DB_AVAILABLE:
        db_benchmark = get_benchmark_value(test_key, None, gender)
        if db_benchmark > 0:
            default = db_benchmark

    # Show as read-only with info about where to edit
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Benchmark:** {default:.1f} {unit}")
    with col2:
        if BENCHMARK_DB_AVAILABLE:
            st.caption("📊 VALD Norm")
        else:
            st.caption("Default")

    return default


def create_ranked_bar_chart(
    df: pd.DataFrame,
    metric_col: str,
    metric_name: str,
    unit: str,
    benchmark: float = None,
    title: str = None,
    horizontal: bool = True
) -> go.Figure:
    """
    Create a ranked bar chart with squad average and benchmark lines.

    Args:
        df: DataFrame with athlete data
        metric_col: Column name for the metric
        metric_name: Display name for the metric
        unit: Unit string for axis label
        benchmark: Optional benchmark value to show as reference line
        title: Chart title
        horizontal: If True, creates horizontal bars (athletes on y-axis). Default True.
    """
    if 'Name' not in df.columns or metric_col not in df.columns:
        return None

    # Get data and sort
    plot_df = df[['Name', metric_col]].dropna()
    if plot_df.empty:
        return None

    # Sort ascending for horizontal (so best performers appear at top)
    plot_df = plot_df.sort_values(metric_col, ascending=True)

    # Calculate squad average
    squad_avg = plot_df[metric_col].mean()

    # Create figure
    fig = go.Figure()

    if horizontal:
        # Horizontal bars (athletes on y-axis) - like NordBord
        fig.add_trace(go.Bar(
            y=plot_df['Name'],
            x=plot_df[metric_col],
            orientation='h',
            marker_color=TEAL_PRIMARY,
            text=[f"{v:.1f}" for v in plot_df[metric_col]],
            textposition='auto',
            textfont=dict(size=10),
            name='Athletes'
        ))

        # Add squad average line (vertical)
        fig.add_vline(
            x=squad_avg,
            line_dash="dash",
            line_color=SQUAD_AVG_COLOR,
            line_width=2,
            annotation_text=f"Squad Avg: {squad_avg:.1f}",
            annotation_position="top",
            annotation_font_color=SQUAD_AVG_COLOR
        )

        # Add benchmark line if provided
        if benchmark and benchmark > 0:
            fig.add_vline(
                x=benchmark,
                line_dash="dash",
                line_color=BENCHMARK_COLOR,
                line_width=2,
                annotation_text=f"Benchmark: {benchmark:.1f}",
                annotation_position="top",
                annotation_font_color=BENCHMARK_COLOR
            )

        # Update layout for horizontal
        fig.update_layout(
            title=title or f"{metric_name}",
            xaxis_title=f"{metric_name} ({unit})" if unit else metric_name,
            yaxis_title="",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter, sans-serif', color='#333'),
            height=max(350, len(plot_df) * 40),
            margin=dict(l=10, r=10, t=50, b=30),
            showlegend=False
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=False)

    else:
        # Vertical bars (original format)
        plot_df = plot_df.sort_values(metric_col, ascending=False)

        fig.add_trace(go.Bar(
            x=plot_df['Name'],
            y=plot_df[metric_col],
            marker_color=TEAL_PRIMARY,
            text=[f"{v:.1f}" for v in plot_df[metric_col]],
            textposition='outside',
            textfont=dict(size=10),
            name='Athletes'
        ))

        # Add squad average line (horizontal)
        fig.add_hline(
            y=squad_avg,
            line_dash="dash",
            line_color=SQUAD_AVG_COLOR,
            line_width=2,
            annotation_text=f"Squad Avg: {squad_avg:.2f}",
            annotation_position="right",
            annotation_font_color=SQUAD_AVG_COLOR
        )

        # Add benchmark line if provided
        if benchmark and benchmark > 0:
            fig.add_hline(
                y=benchmark,
                line_dash="dash",
                line_color=BENCHMARK_COLOR,
                line_width=2,
                annotation_text=f"Benchmark: {benchmark:.2f}",
                annotation_position="right",
                annotation_font_color=BENCHMARK_COLOR
            )

        # Update layout for vertical
        fig.update_layout(
            title=title or f"{metric_name}",
            xaxis_title="",
            yaxis_title=f"{metric_name} ({unit})" if unit else metric_name,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter, sans-serif', color='#333'),
            height=400,
            margin=dict(l=10, r=100, t=50, b=100),
            showlegend=False,
            xaxis=dict(tickangle=-45)
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_ranked_side_by_side_chart(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    metric_name: str,
    unit: str,
    benchmark: float = None,
    title: str = None
) -> go.Figure:
    """
    Create a horizontal side-by-side bar chart for unilateral tests (L vs R).
    Like the NordBord example with left/right bars and injury risk line.
    """
    if 'Name' not in df.columns:
        return None

    # Check columns exist
    if left_col not in df.columns or right_col not in df.columns:
        return None

    # Get data
    plot_df = df[['Name', left_col, right_col]].dropna()
    if plot_df.empty:
        return None

    # Calculate average for sorting
    plot_df['avg'] = (plot_df[left_col] + plot_df[right_col]) / 2
    plot_df = plot_df.sort_values('avg', ascending=True)

    # Create figure
    fig = go.Figure()

    # Add left bars
    fig.add_trace(go.Bar(
        y=plot_df['Name'],
        x=plot_df[left_col],
        orientation='h',
        marker_color=TEAL_PRIMARY,
        text=[f"{v:.0f}" for v in plot_df[left_col]],
        textposition='auto',
        name='Left'
    ))

    # Add right bars
    fig.add_trace(go.Bar(
        y=plot_df['Name'],
        x=plot_df[right_col],
        orientation='h',
        marker_color=TEAL_LIGHT,
        text=[f"{v:.0f}" for v in plot_df[right_col]],
        textposition='auto',
        name='Right'
    ))

    # Add benchmark line if provided
    if benchmark and benchmark > 0:
        fig.add_vline(
            x=benchmark,
            line_dash="dash",
            line_color=BENCHMARK_COLOR,
            line_width=2,
            annotation_text=f"{benchmark:.0f} {unit}",
            annotation_position="top"
        )

    # Update layout
    fig.update_layout(
        title=title or f"{metric_name} - Left & Right",
        xaxis_title=f"{metric_name} ({unit})" if unit else metric_name,
        yaxis_title="",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#333'),
        height=max(300, len(plot_df) * 50),
        margin=dict(l=10, r=10, t=50, b=30),
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=False)

    return fig


def create_asymmetry_table(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    unit: str = ''
) -> pd.DataFrame:
    """Create asymmetry status table for unilateral tests."""
    if 'Name' not in df.columns:
        return pd.DataFrame()

    if left_col not in df.columns or right_col not in df.columns:
        return pd.DataFrame()

    table_df = df[['Name', left_col, right_col]].dropna().copy()
    if table_df.empty:
        return pd.DataFrame()

    # Calculate asymmetry
    table_df['Asymmetry'] = abs(table_df[left_col] - table_df[right_col]) / ((table_df[left_col] + table_df[right_col]) / 2) * 100

    # Determine status
    def get_status(asym):
        if asym <= 5:
            return '🟢 Good'
        elif asym <= 10:
            return '🟡 Monitor'
        else:
            return '🔴 Flag'

    table_df['Status'] = table_df['Asymmetry'].apply(get_status)

    # Format columns
    table_df = table_df.rename(columns={
        'Name': 'Athlete',
        left_col: f'Left ({unit})',
        right_col: f'Right ({unit})'
    })

    table_df['Asymmetry'] = table_df['Asymmetry'].apply(lambda x: f"{x:.1f}%")

    return table_df[['Athlete', f'Left ({unit})', f'Right ({unit})', 'Asymmetry', 'Status']]


# Stacked bar colors for quadrant tests (4 directions)
QUADRANT_COLORS = {
    'Supine': '#1D4D3B',      # Teal
    'Prone': '#FF9800',       # Orange
    'Lateral_Left': '#0077B6', # Blue
    'Lateral_Right': '#a08e66', # Gold
    'Flexion': '#1D4D3B',
    'Extension': '#FF9800',
    'Left': '#0077B6',
    'Right': '#a08e66',
    'IR': '#1D4D3B',          # Internal Rotation
    'ER': '#FF9800',          # External Rotation
    'Adduction': '#0077B6',
    'Abduction': '#a08e66',
}


def create_stacked_quadrant_chart(
    df: pd.DataFrame,
    metric_cols: Dict[str, str],
    metric_name: str,
    unit: str,
    title: str = None,
    vertical: bool = True
) -> go.Figure:
    """
    Create a stacked multi-variable bar chart for quadrant tests.

    Args:
        df: DataFrame with athlete data
        metric_cols: Dict mapping display names to column names
                    e.g., {'Supine': 'supine_col', 'Prone': 'prone_col', ...}
        metric_name: Name of the metric being measured
        unit: Unit of measurement
        title: Chart title
        vertical: If True, athletes on X-axis (vertical bars). If False, horizontal bars.
    """
    if 'Name' not in df.columns:
        return None

    # Check that at least some metric columns exist
    available_cols = {k: v for k, v in metric_cols.items() if v in df.columns}
    if not available_cols:
        return None

    # Get data
    cols_to_use = ['Name'] + list(available_cols.values())
    plot_df = df[cols_to_use].dropna(subset=list(available_cols.values()), how='all')
    if plot_df.empty:
        return None

    # Calculate total for sorting
    plot_df['total'] = plot_df[list(available_cols.values())].sum(axis=1)
    plot_df = plot_df.sort_values('total', ascending=not vertical)

    fig = go.Figure()

    # Add stacked bars for each metric
    for display_name, col_name in available_cols.items():
        color = QUADRANT_COLORS.get(display_name, TEAL_PRIMARY)

        if vertical:
            fig.add_trace(go.Bar(
                x=plot_df['Name'],
                y=plot_df[col_name],
                name=display_name,
                marker_color=color,
                text=[f"{v:.0f}" for v in plot_df[col_name]],
                textposition='inside',
                textfont=dict(size=9, color='white')
            ))
        else:
            fig.add_trace(go.Bar(
                y=plot_df['Name'],
                x=plot_df[col_name],
                name=display_name,
                marker_color=color,
                orientation='h',
                text=[f"{v:.0f}" for v in plot_df[col_name]],
                textposition='inside',
                textfont=dict(size=9, color='white')
            ))

    # Update layout
    if vertical:
        fig.update_layout(
            barmode='stack',
            title=title or f"{metric_name} - Quadrant Profile",
            xaxis_title="",
            yaxis_title=f"{metric_name} ({unit})" if unit else metric_name,
            xaxis=dict(tickangle=-45),
            height=400,
        )
    else:
        fig.update_layout(
            barmode='stack',
            title=title or f"{metric_name} - Quadrant Profile",
            xaxis_title=f"{metric_name} ({unit})" if unit else metric_name,
            yaxis_title="",
            height=max(300, len(plot_df) * 40),
        )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#333'),
        margin=dict(l=10, r=10, t=50, b=100 if vertical else 30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    fig.update_xaxes(showgrid=False if vertical else True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True if vertical else False, gridwidth=1, gridcolor='lightgray')

    return fig


def create_quadrant_summary_table(
    df: pd.DataFrame,
    metric_cols: Dict[str, str],
    unit: str = ''
) -> pd.DataFrame:
    """
    Create summary table for quadrant tests with % differences and flagging.
    Compares opposite directions (e.g., Left vs Right, Flexion vs Extension).
    """
    if 'Name' not in df.columns:
        return pd.DataFrame()

    available_cols = {k: v for k, v in metric_cols.items() if v in df.columns}
    if len(available_cols) < 2:
        return pd.DataFrame()

    cols_to_use = ['Name'] + list(available_cols.values())
    table_df = df[cols_to_use].dropna(subset=list(available_cols.values()), how='all').copy()
    if table_df.empty:
        return pd.DataFrame()

    # Rename columns for display
    rename_map = {'Name': 'Athlete'}
    for display_name, col_name in available_cols.items():
        rename_map[col_name] = f"{display_name} ({unit})"

    table_df = table_df.rename(columns=rename_map)

    # Calculate asymmetries between opposing pairs
    # Common pairs: Left/Right, Flexion/Extension, IR/ER, Adduction/Abduction
    opposing_pairs = [
        ('Lateral_Left', 'Lateral_Right'),
        ('Left', 'Right'),
        ('Flexion', 'Extension'),
        ('Supine', 'Prone'),
        ('IR', 'ER'),
        ('Adduction', 'Abduction'),
    ]

    for pair in opposing_pairs:
        if pair[0] in available_cols and pair[1] in available_cols:
            col1 = f"{pair[0]} ({unit})"
            col2 = f"{pair[1]} ({unit})"

            if col1 in table_df.columns and col2 in table_df.columns:
                # Calculate % difference
                avg_val = (table_df[col1] + table_df[col2]) / 2
                diff = abs(table_df[col1] - table_df[col2]) / avg_val * 100
                table_df[f'{pair[0]}/{pair[1]} Diff'] = diff.apply(lambda x: f"{x:.1f}%")

                # Status flag
                def get_status(val):
                    if val <= 10:
                        return '🟢'
                    elif val <= 20:
                        return '🟡'
                    else:
                        return '🔴'

                table_df[f'{pair[0]}/{pair[1]} Status'] = diff.apply(get_status)

    # Format numeric columns
    for col in table_df.columns:
        if f'({unit})' in col and 'Diff' not in col and 'Status' not in col:
            table_df[col] = table_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")

    return table_df


def create_stacked_individual_trend_chart(
    df: pd.DataFrame,
    athlete_name: str,
    metric_cols: Dict[str, str],
    metric_name: str,
    unit: str,
    title: str = None
) -> go.Figure:
    """
    Create stacked bar chart showing quadrant test progression over time for an individual.
    X-axis: dates, Y-axis: stacked values for each direction.
    """
    if 'Name' not in df.columns or 'recordedDateUtc' not in df.columns:
        return None

    available_cols = {k: v for k, v in metric_cols.items() if v in df.columns}
    if not available_cols:
        return None

    # Filter for athlete
    athlete_df = df[df['Name'] == athlete_name].copy()
    if athlete_df.empty:
        return None

    # Sort by date
    athlete_df['recordedDateUtc'] = pd.to_datetime(athlete_df['recordedDateUtc'], errors='coerce')
    athlete_df = athlete_df.sort_values('recordedDateUtc')

    # Format dates for display
    athlete_df['date_label'] = athlete_df['recordedDateUtc'].dt.strftime('%d %b %Y')

    fig = go.Figure()

    # Add stacked bars for each metric
    for display_name, col_name in available_cols.items():
        if col_name not in athlete_df.columns:
            continue

        color = QUADRANT_COLORS.get(display_name, TEAL_PRIMARY)

        fig.add_trace(go.Bar(
            x=athlete_df['date_label'],
            y=athlete_df[col_name],
            name=display_name,
            marker_color=color,
            text=[f"{v:.0f}" for v in athlete_df[col_name]],
            textposition='inside',
            textfont=dict(size=9, color='white')
        ))

    fig.update_layout(
        barmode='stack',
        title=title or f"{athlete_name} - {metric_name} Progression",
        xaxis_title="Test Date",
        yaxis_title=f"{metric_name} ({unit})" if unit else metric_name,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#333'),
        height=400,
        margin=dict(l=10, r=10, t=50, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_individual_line_chart(
    df: pd.DataFrame,
    selected_athletes: List[str],
    metric_col: str,
    metric_name: str,
    unit: str,
    show_squad_avg: bool = False,
    title: str = None,
    show_value_labels: bool = True
) -> go.Figure:
    """
    Create individual line chart with multiple athlete selection.
    Styled to match Strength RM chart with value labels on data points.

    Args:
        df: DataFrame with athlete data
        selected_athletes: List of athlete names to display
        metric_col: Column name for the metric
        metric_name: Display name for the metric
        unit: Unit string for axis label
        show_squad_avg: If True, shows squad average line (default False)
        title: Chart title
        show_value_labels: If True, shows value labels on data points
    """
    if 'Name' not in df.columns:
        return None

    # Support both 'recordedDateUtc' (VALD) and 'date' (manual entry) columns
    date_col = 'recordedDateUtc'
    if date_col not in df.columns:
        if 'date' in df.columns:
            date_col = 'date'
        else:
            return None

    if metric_col not in df.columns:
        return None

    fig = go.Figure()

    # Add line for each selected athlete
    for i, athlete in enumerate(selected_athletes):
        athlete_df = df[df['Name'] == athlete].sort_values(date_col)
        if athlete_df.empty:
            continue

        color = MULTI_LINE_COLORS[i % len(MULTI_LINE_COLORS)]

        # Create value labels
        if show_value_labels:
            text_labels = [f"{v:.1f}" for v in athlete_df[metric_col]]
            mode = 'markers+lines+text'
        else:
            text_labels = None
            mode = 'markers+lines'

        fig.add_trace(go.Scatter(
            x=athlete_df[date_col],
            y=athlete_df[metric_col],
            mode=mode,
            marker=dict(size=10, color=color),
            line=dict(color=color, width=2),
            name=athlete,
            text=text_labels,
            textposition='top center',
            textfont=dict(size=9, color=color),
            hovertemplate=f"<b>{athlete}</b><br>Date: %{{x|%d %b %Y}}<br>{metric_name}: %{{y:.1f}} {unit}<extra></extra>"
        ))

    # Add squad average line if requested
    if show_squad_avg:
        # Calculate squad average per date
        squad_avg = df.groupby(date_col)[metric_col].mean().reset_index()
        squad_avg = squad_avg.sort_values(date_col)

        fig.add_trace(go.Scatter(
            x=squad_avg[date_col],
            y=squad_avg[metric_col],
            mode='lines',
            line=dict(color=SQUAD_AVG_COLOR, width=2, dash='dash'),
            name='Squad Average'
        ))

    # Update layout - matching Strength RM style
    fig.update_layout(
        title=dict(
            text=title or f"{metric_name} - Individual Trends",
            font=dict(size=14)
        ),
        xaxis_title="Test Date",
        yaxis_title=f"{metric_name} ({unit})" if unit else metric_name,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#333'),
        height=450,
        margin=dict(l=60, r=60, t=80, b=50),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0
        ),
        hovermode='x unified'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_dual_line_chart(
    df: pd.DataFrame,
    selected_athletes: List[str],
    left_col: str,
    right_col: str,
    metric_name: str,
    unit: str,
    show_squad_avg: bool = True,
    title: str = None
) -> go.Figure:
    """
    Create dual line chart for unilateral tests (Left vs Right over time).
    """
    if 'Name' not in df.columns or 'recordedDateUtc' not in df.columns:
        return None

    fig = go.Figure()

    # For single athlete, show L/R lines
    if len(selected_athletes) == 1:
        athlete = selected_athletes[0]
        athlete_df = df[df['Name'] == athlete].sort_values('recordedDateUtc')

        if not athlete_df.empty and left_col in athlete_df.columns and right_col in athlete_df.columns:
            # Left line
            fig.add_trace(go.Scatter(
                x=athlete_df['recordedDateUtc'],
                y=athlete_df[left_col],
                mode='markers+lines',
                marker=dict(size=8, color=TEAL_PRIMARY),
                line=dict(color=TEAL_PRIMARY, width=2),
                name='Left'
            ))

            # Right line
            fig.add_trace(go.Scatter(
                x=athlete_df['recordedDateUtc'],
                y=athlete_df[right_col],
                mode='markers+lines',
                marker=dict(size=8, color=CORAL_ACCENT),
                line=dict(color=CORAL_ACCENT, width=2),
                name='Right'
            ))

        # Add squad average if requested
        if show_squad_avg and left_col in df.columns and right_col in df.columns:
            df['avg_lr'] = (df[left_col] + df[right_col]) / 2
            squad_avg = df.groupby('recordedDateUtc')['avg_lr'].mean().reset_index()
            squad_avg = squad_avg.sort_values('recordedDateUtc')

            fig.add_trace(go.Scatter(
                x=squad_avg['recordedDateUtc'],
                y=squad_avg['avg_lr'],
                mode='lines',
                line=dict(color=SQUAD_AVG_COLOR, width=2, dash='dash'),
                name='Squad Average'
            ))
    else:
        # Multiple athletes - show average of L/R for each
        for i, athlete in enumerate(selected_athletes):
            athlete_df = df[df['Name'] == athlete].sort_values('recordedDateUtc')
            if athlete_df.empty:
                continue

            if left_col in athlete_df.columns and right_col in athlete_df.columns:
                athlete_df['avg_lr'] = (athlete_df[left_col] + athlete_df[right_col]) / 2
                color = MULTI_LINE_COLORS[i % len(MULTI_LINE_COLORS)]

                fig.add_trace(go.Scatter(
                    x=athlete_df['recordedDateUtc'],
                    y=athlete_df['avg_lr'],
                    mode='markers+lines',
                    marker=dict(size=8, color=color),
                    line=dict(color=color, width=2),
                    name=athlete
                ))

    # Update layout
    fig.update_layout(
        title=title or f"{metric_name} - Left vs Right Trends",
        xaxis_title="Date",
        yaxis_title=f"{metric_name} ({unit})" if unit else metric_name,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#333'),
        height=400,
        margin=dict(l=10, r=10, t=50, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_multi_line_strength_chart(
    df: pd.DataFrame,
    athlete_name: str,
    exercises: List[str],
    bodyweight: float = 80.0,
    bodyweight_exercises: List[str] = None,
    title: str = None
) -> go.Figure:
    """
    Create multi-line chart for Strength RM progression.
    Shows multiple lifts with relative strength labels (kg/BM).

    Args:
        df: DataFrame with columns: date, athlete, exercise, estimated_1rm (or weight_kg)
        athlete_name: Name of athlete to display
        exercises: List of exercise names to plot
        bodyweight: Athlete's bodyweight for relative strength calculation
        bodyweight_exercises: Exercises that use secondary Y-axis (e.g., Chin-Up, Pull Up)
        title: Chart title

    Data structure expected (from manual S&C entry):
        date, athlete, exercise, weight_kg, reps, sets, estimated_1rm, ...
    """
    if df.empty or athlete_name is None:
        return None

    if bodyweight_exercises is None:
        bodyweight_exercises = ['Chin-Up', 'Pull Up', 'Weighted Pull Up', 'Chin Up']

    # Filter for athlete
    athlete_df = df[df['athlete'] == athlete_name].copy()
    if athlete_df.empty:
        return None

    # Ensure date column is datetime
    if 'date' in athlete_df.columns:
        athlete_df['date'] = pd.to_datetime(athlete_df['date'], errors='coerce')

    # Determine value column (prefer estimated_1rm, fallback to weight_kg)
    value_col = 'estimated_1rm' if 'estimated_1rm' in athlete_df.columns else 'weight_kg'

    # Create figure with secondary y-axis for bodyweight exercises
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Color palette for exercises
    exercise_colors = {
        'Back Squat': '#1D4D3B',      # Teal
        'Bench Press': '#FF9800',      # Orange
        'Deadlift': '#0077B6',         # Blue
        'Chin-Up': '#a08e66',          # Gold (dashed)
        'Pull Up': '#a08e66',
        'Weighted Pull Up': '#a08e66',
        'Overhead Press': '#9C27B0',   # Purple
        'Push Press': '#E91E63',       # Pink
        'Front Squat': '#00BCD4',      # Cyan
    }

    for i, exercise in enumerate(exercises):
        exercise_df = athlete_df[athlete_df['exercise'].str.contains(exercise, case=False, na=False)]
        if exercise_df.empty:
            continue

        # Sort by date
        exercise_df = exercise_df.sort_values('date')

        # Get color
        color = exercise_colors.get(exercise, MULTI_LINE_COLORS[i % len(MULTI_LINE_COLORS)])

        # Check if this is a bodyweight exercise (secondary axis)
        is_bw_exercise = any(bw in exercise for bw in bodyweight_exercises)

        # Calculate relative strength labels
        labels = []
        for val in exercise_df[value_col]:
            rel_strength = val / bodyweight if bodyweight > 0 else 0
            labels.append(f"{rel_strength:.2f} kg/BM")

        # Line style
        line_style = dict(color=color, width=2)
        if is_bw_exercise:
            line_style['dash'] = 'dash'

        fig.add_trace(
            go.Scatter(
                x=exercise_df['date'],
                y=exercise_df[value_col],
                mode='markers+lines+text',
                marker=dict(size=10, color=color),
                line=line_style,
                name=exercise,
                text=labels,
                textposition='top center',
                textfont=dict(size=9, color=color),
                hovertemplate=f"<b>{exercise}</b><br>Date: %{{x|%d %b %Y}}<br>{value_col}: %{{y:.1f}} kg<br>Relative: %{{text}}<extra></extra>"
            ),
            secondary_y=is_bw_exercise
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=title or f"Repetition Max Strength Progression<br><sub>Relative Strength Labels (kg/BM), BM={bodyweight:.0f} kg</sub>",
            font=dict(size=14)
        ),
        xaxis_title="Test Date",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#333'),
        height=450,
        margin=dict(l=60, r=60, t=80, b=50),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0
        ),
        hovermode='x unified'
    )

    # Update y-axes
    fig.update_yaxes(
        title_text="Repetition Max Load (kg)",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Chin-Up (kg)" if any(bw in exercises for bw in bodyweight_exercises) else "",
        showgrid=False,
        secondary_y=True
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


# =====================
# Hip Diagnostics Helpers
# =====================

def prepare_hip_diagnostics_data(
    forceframe_df: pd.DataFrame,
    forcedecks_df: pd.DataFrame = None
) -> Dict[str, pd.DataFrame]:
    """
    Prepare hip diagnostics data from ForceFrame and ForceDecks.

    Returns dict with keys: 'adab', 'flexion', 'extension'
    Each DataFrame has standardized Name, L/R metric columns, and recordedDateUtc.
    """
    result = {'adab': pd.DataFrame(), 'flexion': pd.DataFrame(), 'extension': pd.DataFrame()}

    if forceframe_df is None or forceframe_df.empty:
        return result

    ff = forceframe_df.copy()

    # Standardize Name column
    if 'Name' not in ff.columns and 'full_name' in ff.columns:
        ff['Name'] = ff['full_name']
    # Add recordedDateUtc from testDateUtc for filter compatibility
    if 'recordedDateUtc' not in ff.columns and 'testDateUtc' in ff.columns:
        ff['recordedDateUtc'] = ff['testDateUtc']

    # Build body mass lookup from ForceDecks
    bm_map = {}
    if forcedecks_df is not None and not forcedecks_df.empty and 'weight' in forcedecks_df.columns:
        fd = forcedecks_df.copy()
        if 'Name' not in fd.columns and 'full_name' in fd.columns:
            fd['Name'] = fd['full_name']
        fd_valid = fd[fd['weight'] > 0].copy()
        if not fd_valid.empty and 'recordedDateUtc' in fd_valid.columns:
            fd_valid['recordedDateUtc'] = pd.to_datetime(fd_valid['recordedDateUtc'], errors='coerce')
            bm_map = fd_valid.sort_values('recordedDateUtc').groupby('Name')['weight'].last().to_dict()

    def _normalize_nkg(df, cols):
        """Add N/kg columns using body mass lookup."""
        df['body_mass'] = df['Name'].map(bm_map)
        for col in cols:
            nkg_col = f'{col}_nkg'
            df[nkg_col] = np.where(
                df['body_mass'] > 0,
                df[col] / df['body_mass'],
                np.nan
            )
        return df

    # --- Hip AD/AB (ForceFrame) ---
    if 'testTypeName' in ff.columns:
        adab_mask = ff['testTypeName'].str.contains('Hip AD/AB', case=False, na=False)
        if adab_mask.any():
            adab = ff[adab_mask].copy()
            # Extract position (e.g., "60" from "Hip AD/AB - 60")
            adab['position'] = adab['testTypeName'].str.extract(r'(\d+)').fillna('')
            # Map inner->Adduction, outer->Abduction
            adab['Add_Left'] = pd.to_numeric(adab.get('innerLeftMaxForce'), errors='coerce')
            adab['Add_Right'] = pd.to_numeric(adab.get('innerRightMaxForce'), errors='coerce')
            adab['Abd_Left'] = pd.to_numeric(adab.get('outerLeftMaxForce'), errors='coerce')
            adab['Abd_Right'] = pd.to_numeric(adab.get('outerRightMaxForce'), errors='coerce')
            adab = _normalize_nkg(adab, ['Add_Left', 'Add_Right', 'Abd_Left', 'Abd_Right'])
            result['adab'] = adab

    # --- Hip Flexion (ForceFrame) ---
    if 'testTypeName' in ff.columns:
        flex_mask = ff['testTypeName'].str.contains('Hip Flexion', case=False, na=False)
        if flex_mask.any():
            flex = ff[flex_mask].copy()
            # Flexion uses outer columns (inner is ~0)
            flex['Flx_Left'] = pd.to_numeric(flex.get('outerLeftMaxForce'), errors='coerce')
            flex['Flx_Right'] = pd.to_numeric(flex.get('outerRightMaxForce'), errors='coerce')
            flex = _normalize_nkg(flex, ['Flx_Left', 'Flx_Right'])
            result['flexion'] = flex

    # --- Hip Extension (ForceDecks SLISOT "Hip Thrust") ---
    if forcedecks_df is not None and not forcedecks_df.empty and 'testType' in forcedecks_df.columns:
        fd = forcedecks_df.copy()
        if 'Name' not in fd.columns and 'full_name' in fd.columns:
            fd['Name'] = fd['full_name']

        # Filter for SLISOT tests tagged as hip thrust
        slisot_mask = fd['testType'] == 'SLISOT'
        hip_thrust_mask = pd.Series(False, index=fd.index)
        for col_name in ['notes', 'testTypeName']:
            if col_name in fd.columns:
                hip_thrust_mask = hip_thrust_mask | fd[col_name].str.contains('hip thrust', case=False, na=False)

        ext_mask = slisot_mask & hip_thrust_mask
        if ext_mask.any():
            ext = fd[ext_mask].copy()
            ext['recordedDateUtc'] = pd.to_datetime(ext['recordedDateUtc'], errors='coerce')

            # Prefer NET_PEAK_VERTICAL_FORCE (net of BW), fallback to PEAK_VERTICAL_FORCE
            left_col = None
            right_col = None
            for prefix in ['NET_PEAK_VERTICAL_FORCE', 'PEAK_VERTICAL_FORCE']:
                if f'{prefix}_Left' in ext.columns and f'{prefix}_Right' in ext.columns:
                    left_col = f'{prefix}_Left'
                    right_col = f'{prefix}_Right'
                    break

            if left_col and right_col:
                ext[left_col] = pd.to_numeric(ext[left_col], errors='coerce')
                ext[right_col] = pd.to_numeric(ext[right_col], errors='coerce')

                # Determine which side each row tests
                ext['side'] = np.where(
                    ext[left_col].notna() & (ext[left_col] != 0), 'Left',
                    np.where(ext[right_col].notna() & (ext[right_col] != 0), 'Right', None)
                )
                ext['force_value'] = ext[left_col].fillna(ext[right_col])

                # Also get body mass for N/kg
                if 'weight' in ext.columns:
                    ext['body_mass'] = pd.to_numeric(ext['weight'], errors='coerce')
                    ext.loc[ext['body_mass'] <= 0, 'body_mass'] = np.nan
                    ext['force_value_nkg'] = ext['force_value'] / ext['body_mass']

                # Carry athlete_sport and athlete_sex
                result['extension'] = ext

    return result


def pivot_extension_lr(ext_df: pd.DataFrame, use_nkg: bool = False) -> pd.DataFrame:
    """
    Pivot single-limb ForceDecks extension tests into one row per athlete with Left & Right columns.
    """
    if ext_df is None or ext_df.empty or 'side' not in ext_df.columns:
        return pd.DataFrame()

    val_col = 'force_value_nkg' if use_nkg and 'force_value_nkg' in ext_df.columns else 'force_value'

    # Get latest test per athlete per side
    valid = ext_df.dropna(subset=['Name', 'side', val_col]).copy()
    if valid.empty:
        return pd.DataFrame()

    valid = valid.sort_values('recordedDateUtc')
    latest = valid.groupby(['Name', 'side']).agg({
        val_col: 'last',
        'recordedDateUtc': 'last'
    }).reset_index()

    # Pivot to get Left and Right columns
    pivot = latest.pivot(index='Name', columns='side', values=val_col).reset_index()
    pivot.columns.name = None

    # Rename to standard format
    rename = {}
    if 'Left' in pivot.columns:
        rename['Left'] = 'Ext_Left'
    if 'Right' in pivot.columns:
        rename['Right'] = 'Ext_Right'
    pivot = pivot.rename(columns=rename)

    # Carry over athlete_sport and athlete_sex from ext_df
    meta_cols = {}
    for meta in ['athlete_sport', 'athlete_sex']:
        if meta in ext_df.columns:
            meta_map = ext_df.dropna(subset=['Name', meta]).drop_duplicates('Name').set_index('Name')[meta].to_dict()
            meta_cols[meta] = pivot['Name'].map(meta_map)
    for k, v in meta_cols.items():
        pivot[k] = v

    return pivot


def create_hip_summary_table(
    adab_df: pd.DataFrame,
    flex_df: pd.DataFrame,
    ext_pivot_df: pd.DataFrame,
    unit: str = 'N/kg'
) -> pd.DataFrame:
    """
    Create comprehensive hip diagnostics summary table.

    Merges latest values from all hip metrics per athlete,
    calculates L/R asymmetry and Add/Abd & Flx/Ext ratios.
    """
    dfs_to_merge = []

    # --- Adduction & Abduction (from same ForceFrame Hip AD/AB tests) ---
    if adab_df is not None and not adab_df.empty and 'Name' in adab_df.columns:
        adab = adab_df.copy()
        if 'testDateUtc' in adab.columns:
            adab['testDateUtc'] = pd.to_datetime(adab['testDateUtc'], errors='coerce')
            adab = adab.sort_values('testDateUtc').groupby('Name').last().reset_index()
        else:
            adab = adab.groupby('Name').last().reset_index()

        suffix = '_nkg' if 'N/kg' in unit else ''
        cols = {f'Add_Left{suffix}': 'Add L', f'Add_Right{suffix}': 'Add R',
                f'Abd_Left{suffix}': 'Abd L', f'Abd_Right{suffix}': 'Abd R'}
        keep = ['Name'] + [c for c in cols.keys() if c in adab.columns]
        if len(keep) > 1:
            adab_summary = adab[keep].rename(columns=cols)
            dfs_to_merge.append(adab_summary)

    # --- Flexion ---
    if flex_df is not None and not flex_df.empty and 'Name' in flex_df.columns:
        flex = flex_df.copy()
        if 'testDateUtc' in flex.columns:
            flex['testDateUtc'] = pd.to_datetime(flex['testDateUtc'], errors='coerce')
            flex = flex.sort_values('testDateUtc').groupby('Name').last().reset_index()
        else:
            flex = flex.groupby('Name').last().reset_index()

        suffix = '_nkg' if 'N/kg' in unit else ''
        cols = {f'Flx_Left{suffix}': 'Flx L', f'Flx_Right{suffix}': 'Flx R'}
        keep = ['Name'] + [c for c in cols.keys() if c in flex.columns]
        if len(keep) > 1:
            flex_summary = flex[keep].rename(columns=cols)
            dfs_to_merge.append(flex_summary)

    # --- Extension (already pivoted) ---
    if ext_pivot_df is not None and not ext_pivot_df.empty and 'Name' in ext_pivot_df.columns:
        ext = ext_pivot_df.rename(columns={'Ext_Left': 'Ext L', 'Ext_Right': 'Ext R'}).copy()
        keep = ['Name'] + [c for c in ['Ext L', 'Ext R'] if c in ext.columns]
        if len(keep) > 1:
            dfs_to_merge.append(ext[keep])

    if not dfs_to_merge:
        return pd.DataFrame()

    # Merge all by athlete Name
    summary = dfs_to_merge[0]
    for df in dfs_to_merge[1:]:
        summary = summary.merge(df, on='Name', how='outer')

    # --- Calculate asymmetry percentages ---
    def calc_asym(row, left_col, right_col):
        l, r = row.get(left_col), row.get(right_col)
        if pd.isna(l) or pd.isna(r) or (l + r) == 0:
            return np.nan
        return abs(l - r) / ((l + r) / 2) * 100

    for metric in ['Add', 'Abd', 'Flx', 'Ext']:
        l_col, r_col = f'{metric} L', f'{metric} R'
        if l_col in summary.columns and r_col in summary.columns:
            summary[f'{metric} Asym%'] = summary.apply(lambda row: calc_asym(row, l_col, r_col), axis=1)

    # --- Calculate ratios ---
    # Add/Abd ratio per limb
    for side in ['L', 'R']:
        add_col, abd_col = f'Add {side}', f'Abd {side}'
        if add_col in summary.columns and abd_col in summary.columns:
            summary[f'Add/Abd {side}'] = np.where(
                summary[abd_col] > 0,
                summary[add_col] / summary[abd_col],
                np.nan
            )
    # Flx/Ext ratio per limb
    for side in ['L', 'R']:
        flx_col, ext_col = f'Flx {side}', f'Ext {side}'
        if flx_col in summary.columns and ext_col in summary.columns:
            summary[f'Flx/Ext {side}'] = np.where(
                summary[ext_col] > 0,
                summary[flx_col] / summary[ext_col],
                np.nan
            )

    # --- Format status flags ---
    def asym_flag(val):
        if pd.isna(val):
            return ''
        if val <= 5:
            return '🟢'
        elif val <= 10:
            return '🟡'
        return '🔴'

    for metric in ['Add', 'Abd', 'Flx', 'Ext']:
        asym_col = f'{metric} Asym%'
        if asym_col in summary.columns:
            summary[f'{metric} Flag'] = summary[asym_col].apply(asym_flag)
            summary[asym_col] = summary[asym_col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else '--')

    # Round metric values
    for col in summary.columns:
        if summary[col].dtype in ['float64', 'float32']:
            summary[col] = summary[col].round(2)

    # Reorder columns
    ordered = ['Name']
    for metric in ['Add', 'Abd']:
        for c in [f'{metric} L', f'{metric} R', f'{metric} Asym%', f'{metric} Flag']:
            if c in summary.columns:
                ordered.append(c)
    for c in ['Add/Abd L', 'Add/Abd R']:
        if c in summary.columns:
            ordered.append(c)
    for metric in ['Flx', 'Ext']:
        for c in [f'{metric} L', f'{metric} R', f'{metric} Asym%', f'{metric} Flag']:
            if c in summary.columns:
                ordered.append(c)
    for c in ['Flx/Ext L', 'Flx/Ext R']:
        if c in summary.columns:
            ordered.append(c)

    remaining = [c for c in summary.columns if c not in ordered]
    ordered.extend(remaining)

    return summary[[c for c in ordered if c in summary.columns]]


def render_snc_diagnostics_tab(forcedecks_df: pd.DataFrame, nordbord_df: pd.DataFrame = None, forceframe_df: pd.DataFrame = None, dynamo_df: pd.DataFrame = None):
    """
    Main function to render the S&C Diagnostics Canvas tab.
    """
    # Ensure Name column exists (map from full_name if needed)
    if 'Name' not in forcedecks_df.columns and 'full_name' in forcedecks_df.columns:
        forcedecks_df = forcedecks_df.copy()
        forcedecks_df['Name'] = forcedecks_df['full_name']

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1D4D3B 0%, #153829 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">S&C Diagnostics Canvas</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Comprehensive strength & conditioning testing analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Test type selector with persistence (st.tabs doesn't persist on rerun)
    test_tab_options = [
        "📊 IMTP",           # Ranked Bar
        "🦘 CMJ",            # Ranked Bar
        "🦵 SL Tests",       # Side-by-Side (includes Ash Test)
        "💪 NordBord",       # Side-by-Side (Nordic)
        "🏃 10:5 Hop",       # Ranked Bar
        "🔄 Quadrant Tests", # Stacked Multi-Variable
        "🦴 Hip Diagnostics", # ForceFrame + ForceDecks hip assessment
        "🏋️ Strength RM",    # Ranked Bar (Manual Entry)
        "🦘 Broad Jump",     # Ranked Bar (Manual Entry)
        "🏃 Fitness Tests",  # Ranked Bar (6 Min Aerobic, etc.)
        "💥 Plyo Pushup",    # Ranked Bar (Upper Body Power)
        "✊ DynaMo",          # Ranked Bar
        "⚖️ Balance"         # Ranked Bar (Shooting)
    ]

    # Initialize session state for active test tab
    if 'active_snc_tab' not in st.session_state:
        st.session_state.active_snc_tab = test_tab_options[0]

    # Ensure active tab is valid
    if st.session_state.active_snc_tab not in test_tab_options:
        st.session_state.active_snc_tab = test_tab_options[0]

    # Get current index for the selectbox
    current_tab_idx = test_tab_options.index(st.session_state.active_snc_tab)

    # Use selectbox for tab navigation (persists across reruns)
    selected_test_tab = st.selectbox(
        "Select Test:",
        test_tab_options,
        index=current_tab_idx,
        key="snc_test_tab_selector"
    )

    # Update session state
    st.session_state.active_snc_tab = selected_test_tab

    st.markdown("---")

    # =====================
    # IMTP Tab
    # =====================
    if selected_test_tab == "📊 IMTP":
        st.markdown("### Isometric Mid-Thigh Pull (IMTP)")

        # Filter for IMTP tests
        imtp_df = forcedecks_df[forcedecks_df['testType'] == 'IMTP'].copy() if 'testType' in forcedecks_df.columns else pd.DataFrame()

        if imtp_df.empty:
            st.warning("No IMTP test data available.")
        else:
            # Filters
            filtered_df, sport, gender = render_filters(imtp_df, "imtp")

            # Benchmark input
            col1, col2 = st.columns([3, 1])
            with col2:
                benchmark = render_benchmark_input('IMTP', 'imtp')

            # Sub-tabs for Group vs Individual
            selected_view = st.radio("View:", ["👥 Group View", "🏃 Individual View"], horizontal=True, key="imtp_view")

            if selected_view == "👥 Group View":
                # Group ranked bar chart - use config for column resolution
                metric_col = resolve_metric_column(filtered_df, TEST_CONFIG['IMTP']['metric1'])

                if metric_col:
                    fig = create_ranked_bar_chart(
                        filtered_df,
                        metric_col,
                        'Relative Peak Force',
                        'N/Kg',
                        benchmark,
                        'IMTP - Relative Peak Force'
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="imtp_group_bar")
                else:
                    st.warning("Peak Force metric not found in data.")

            if selected_view == "🏃 Individual View":
                # Individual line chart with multi-select
                athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []

                if athletes:
                    selected_athletes = st.multiselect(
                        "Select Athletes:",
                        options=athletes,
                        default=get_persisted_athlete_selection("imtp_athlete_select", athletes),
                        key="imtp_athlete_select"
                    )

                    if selected_athletes:
                        # Get all data (not just most recent) for trends
                        all_imtp = forcedecks_df[forcedecks_df['testType'] == 'IMTP'].copy()

                        if sport != 'All' and 'athlete_sport' in all_imtp.columns:
                            all_imtp = all_imtp[all_imtp['athlete_sport'] == sport]
                        if gender != 'All' and 'athlete_sex' in all_imtp.columns:
                            all_imtp = all_imtp[all_imtp['athlete_sex'] == gender]

                        fig = create_individual_line_chart(
                            all_imtp,
                            selected_athletes,
                            metric_col,
                            'Relative Peak Force',
                            'N/Kg',
                            False,  # No squad average
                            'IMTP - Individual Trends'
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key="imtp_ind_line")
                else:
                    st.info("No athletes found in filtered data.")

    # =====================
    # CMJ Tab
    # =====================
    elif selected_test_tab == "🦘 CMJ":
        st.markdown("### Counter Movement Jump (CMJ)")

        cmj_df = forcedecks_df[forcedecks_df['testType'].isin(['CMJ', 'ABCMJ'])].copy() if 'testType' in forcedecks_df.columns else pd.DataFrame()

        if cmj_df.empty:
            st.warning("No CMJ test data available.")
        else:
            filtered_df, sport, gender = render_filters(cmj_df, "cmj")

            col1, col2 = st.columns([3, 1])
            with col2:
                benchmark = render_benchmark_input('CMJ', 'cmj')

            selected_view = st.radio("View:", ["👥 Group View", "🏃 Individual View"], horizontal=True, key="cmj_view")

            if selected_view == "👥 Group View":
                # Use config for column resolution
                metric_col = resolve_metric_column(filtered_df, TEST_CONFIG['CMJ']['metric1'])

                if metric_col:
                    fig = create_ranked_bar_chart(
                        filtered_df,
                        metric_col,
                        'Relative Peak Power',
                        'W/Kg',
                        benchmark,
                        'CMJ - Relative Peak Power'
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="cmj_group_bar")

                    # Also show jump height if available
                    height_col = resolve_metric_column(filtered_df, TEST_CONFIG['CMJ']['metric2'])
                    if height_col:
                        st.markdown("---")
                        height_df = filtered_df.copy()
                        # Convert to cm if values look like meters (< 1)
                        if height_df[height_col].median() < 1:
                            height_df[height_col] = height_df[height_col] * 100  # m to cm

                        fig2 = create_ranked_bar_chart(
                            height_df,
                            height_col,
                            'Jump Height (Imp-Mom)',
                            'cm',
                            None,
                            'CMJ - Jump Height'
                        )
                        if fig2:
                            st.plotly_chart(fig2, use_container_width=True, key="cmj_height_bar")
                else:
                    st.warning("CMJ Power metric not found in data.")

            if selected_view == "🏃 Individual View":
                athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []

                if athletes:
                    selected_athletes = st.multiselect(
                        "Select Athletes:",
                        options=athletes,
                        default=get_persisted_athlete_selection("cmj_athlete_select", athletes),
                        key="cmj_athlete_select"
                    )

                    if selected_athletes:
                        all_cmj = forcedecks_df[forcedecks_df['testType'].isin(['CMJ', 'ABCMJ'])].copy()

                        if sport != 'All' and 'athlete_sport' in all_cmj.columns:
                            all_cmj = all_cmj[all_cmj['athlete_sport'] == sport]
                        if gender != 'All' and 'athlete_sex' in all_cmj.columns:
                            all_cmj = all_cmj[all_cmj['athlete_sex'] == gender]

                        fig = create_individual_line_chart(
                            all_cmj,
                            selected_athletes,
                            metric_col,
                            'Relative Peak Power',
                            'W/Kg',
                            False,  # No squad average
                            'CMJ - Individual Trends'
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key="cmj_ind_line")

    # =====================
    # SL Tests Tab
    # =====================
    elif selected_test_tab == "🦵 SL Tests":
        st.markdown("### Single Leg Tests")

        # Map display names to actual VALD test type codes
        sl_test_mapping = {
            'SL ISO Squat': 'SLISOSQT',
            'SL IMTP': 'SLIMTP',
            'SL CMJ': 'SLCMRJ',
            'SL Drop Jump': 'SLDJ',
            'SL Jump': 'SLJ',
            'SL Hop Jump': 'SLHJ',
            'Ash Test (ForceFrame)': 'ASH'  # ForceFrame Shoulder Assessment
        }

        sl_test_options = list(sl_test_mapping.keys())
        selected_sl_test = st.selectbox("Select Test:", sl_test_options, key="sl_test_select")

        # Get the actual test type code
        test_type_code = sl_test_mapping.get(selected_sl_test, selected_sl_test)

        # Handle Ash Test separately (uses ForceFrame data)
        if 'Ash Test' in selected_sl_test:
            # Ash Test uses ForceFrame data for shoulder assessment
            if forceframe_df is not None and not forceframe_df.empty:
                # Look for shoulder assessment tests
                sl_df = forceframe_df.copy()
            else:
                sl_df = pd.DataFrame()
                st.info("Ash Test requires ForceFrame data. No ForceFrame data available.")
        else:
            # Filter for selected SL test using exact match (ForceDecks)
            if 'testType' in forcedecks_df.columns:
                sl_df = forcedecks_df[forcedecks_df['testType'] == test_type_code].copy()
            else:
                sl_df = pd.DataFrame()

        if sl_df.empty:
            st.warning(f"No {selected_sl_test} test data available.")
        else:
            filtered_df, sport, gender = render_filters(sl_df, "sl")

            col1, col2 = st.columns([3, 1])
            with col2:
                test_key = selected_sl_test.replace(' ', '_')
                benchmark = render_benchmark_input(test_key, 'sl')

            selected_view = st.radio("View:", ["👥 Group View", "🏃 Individual View"], horizontal=True, key="sl_view")

            if selected_view == "👥 Group View":
                # Determine left/right columns based on test type
                if 'ISO' in selected_sl_test or 'IMTP' in selected_sl_test:
                    left_col = 'Peak Force / BM_Left'
                    right_col = 'Peak Force / BM_Right'
                    metric_name = 'Relative Peak Force'
                    unit = 'N/Kg'
                else:  # CMJ
                    left_col = 'Peak Power / BM_Left'
                    right_col = 'Peak Power / BM_Right'
                    metric_name = 'Relative Peak Power'
                    unit = 'W/Kg'

                # Try alternative column names
                if left_col not in filtered_df.columns:
                    alt_left = ['Peak Vertical Force / BM_Left', 'Takeoff Peak Force / BM_Left']
                    for alt in alt_left:
                        if alt in filtered_df.columns:
                            left_col = alt
                            break

                if right_col not in filtered_df.columns:
                    alt_right = ['Peak Vertical Force / BM_Right', 'Takeoff Peak Force / BM_Right']
                    for alt in alt_right:
                        if alt in filtered_df.columns:
                            right_col = alt
                            break

                if left_col in filtered_df.columns and right_col in filtered_df.columns:
                    fig = create_ranked_side_by_side_chart(
                        filtered_df,
                        left_col,
                        right_col,
                        metric_name,
                        unit,
                        benchmark,
                        f'{selected_sl_test} - {metric_name} (Left & Right)'
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="sl_group_bar")

                    # Asymmetry table
                    st.markdown("### Asymmetry Status")
                    asym_table = create_asymmetry_table(filtered_df, left_col, right_col, unit)
                    if not asym_table.empty:
                        st.dataframe(asym_table, use_container_width=True, hide_index=True)
                else:
                    st.warning(f"Left/Right metrics not found for {selected_sl_test}.")

            if selected_view == "🏃 Individual View":
                athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []

                if athletes:
                    selected_athletes = st.multiselect(
                        "Select Athletes:",
                        options=athletes,
                        default=get_persisted_athlete_selection("sl_athlete_select", athletes),
                        key="sl_athlete_select"
                    )

                    show_squad = st.checkbox("Show Squad Average", value=True, key="sl_show_squad")

                    if selected_athletes:
                        all_sl = forcedecks_df[forcedecks_df['testType'].str.contains(selected_sl_test, case=False, na=False)].copy()

                        if sport != 'All' and 'athlete_sport' in all_sl.columns:
                            all_sl = all_sl[all_sl['athlete_sport'] == sport]
                        if gender != 'All' and 'athlete_sex' in all_sl.columns:
                            all_sl = all_sl[all_sl['athlete_sex'] == gender]

                        fig = create_dual_line_chart(
                            all_sl,
                            selected_athletes,
                            left_col,
                            right_col,
                            metric_name,
                            unit,
                            show_squad,
                            f'{selected_sl_test} - Left vs Right Trends'
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key="sl_ind_line")

    # =====================
    # NordBord Tab
    # =====================
    elif selected_test_tab == "💪 NordBord":
        st.markdown("### NordBord - Hamstring Strength")

        if nordbord_df is None or nordbord_df.empty:
            st.warning("No NordBord data available.")
        else:
            filtered_df, sport, gender = render_filters(nordbord_df, "nordbord")

            col1, col2 = st.columns([3, 1])
            with col2:
                benchmark = render_benchmark_input('NordBord', 'nordbord')

            selected_view = st.radio("View:", ["👥 Group View", "🏃 Individual View"], horizontal=True, key="nordbord_view")

            if selected_view == "👥 Group View":
                left_col = 'leftMaxForce'
                right_col = 'rightMaxForce'

                # Try alternative column names
                if left_col not in filtered_df.columns:
                    for alt in ['maxForceLeftN', 'leftMax']:
                        if alt in filtered_df.columns:
                            left_col = alt
                            break

                if right_col not in filtered_df.columns:
                    for alt in ['maxForceRightN', 'rightMax']:
                        if alt in filtered_df.columns:
                            right_col = alt
                            break

                if left_col in filtered_df.columns and right_col in filtered_df.columns:
                    fig = create_ranked_side_by_side_chart(
                        filtered_df,
                        left_col,
                        right_col,
                        'Max Force',
                        'N',
                        benchmark,
                        'NordBord - Left vs Right Hamstring Force'
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="nordbord_group_bar")

                    # Asymmetry table
                    st.markdown("### Asymmetry Status")
                    asym_table = create_asymmetry_table(filtered_df, left_col, right_col, 'N')
                    if not asym_table.empty:
                        st.dataframe(asym_table, use_container_width=True, hide_index=True)
                else:
                    st.warning("NordBord force columns not found.")

            if selected_view == "🏃 Individual View":
                athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []

                if athletes:
                    selected_athletes = st.multiselect(
                        "Select Athletes:",
                        options=athletes,
                        default=get_persisted_athlete_selection("nordbord_athlete_select", athletes),
                        key="nordbord_athlete_select"
                    )

                    show_squad = st.checkbox("Show Squad Average", value=True, key="nordbord_show_squad")

                    if selected_athletes and left_col in nordbord_df.columns and right_col in nordbord_df.columns:
                        fig = create_dual_line_chart(
                            nordbord_df,
                            selected_athletes,
                            left_col,
                            right_col,
                            'Max Force',
                            'N',
                            show_squad,
                            'NordBord - Left vs Right Trends'
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key="nordbord_ind_line")

    # =====================
    # 10:5 Hop Test Tab
    # =====================
    elif selected_test_tab == "🏃 10:5 Hop":
        st.markdown("### 10:5 Hop Test")

        # Filter for hop/reactive strength tests: HJ, SLHJ, RSHIP, RSKIP, RSAIP
        hop_test_types = ['HJ', 'SLHJ', 'RSHIP', 'RSKIP', 'RSAIP']
        hop_df = forcedecks_df[forcedecks_df['testType'].isin(hop_test_types)].copy() if 'testType' in forcedecks_df.columns else pd.DataFrame()

        if hop_df.empty:
            st.warning("No Hop Test data available.")
        else:
            filtered_df, sport, gender = render_filters(hop_df, "hop")

            col1, col2 = st.columns([3, 1])
            with col2:
                benchmark = render_benchmark_input('10_5_Hop', 'hop')

            selected_view = st.radio("View:", ["👥 Group View", "🏃 Individual View"], horizontal=True, key="hop_view")

            if selected_view == "👥 Group View":
                metric_col = None
                needs_conversion = False

                # Check for RSI columns - hop-specific columns first, then general
                rsi_columns = [
                    ('HOP_BEST_RSI', False),      # Primary hop RSI metric
                    ('HOP_RSI', False),           # Alternative hop RSI
                    ('HOP_MEAN_RSI', False),      # Mean RSI from hop tests
                    ('RSI_MODIFIED', True),       # local_sync format, needs /100 conversion
                    ('RSI_MODIFIED_IMP_MOM', True),
                    ('RSI', True),
                    ('RSI-modified_Trial', False),  # legacy format, already correct
                    ('RSI (Flight/Contact Time)_Trial', False),
                    ('Best RSI (Flight/Contact Time)_Trial', False),
                ]

                for col, convert in rsi_columns:
                    if col in filtered_df.columns and filtered_df[col].notna().sum() > 0:
                        metric_col = col
                        needs_conversion = convert
                        break

                if metric_col and metric_col in filtered_df.columns:
                    # Apply conversion if needed (RSI should be ~0.3-0.6, not 30-60)
                    display_df = filtered_df.copy()
                    if needs_conversion and display_df[metric_col].median() > 10:
                        display_df[metric_col] = display_df[metric_col] * 0.01

                    fig = create_ranked_bar_chart(
                        display_df,
                        metric_col,
                        'RSI',
                        '',
                        benchmark,
                        '10:5 Hop Test - Reactive Strength Index'
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="hop_group_bar")
                else:
                    st.warning("RSI metric not found in data.")

            if selected_view == "🏃 Individual View":
                athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []

                if athletes:
                    selected_athletes = st.multiselect(
                        "Select Athletes:",
                        options=athletes,
                        default=get_persisted_athlete_selection("hop_athlete_select", athletes),
                        key="hop_athlete_select"
                    )

                    if selected_athletes and metric_col and metric_col in hop_df.columns:
                        # Apply conversion if needed
                        ind_hop_df = hop_df.copy()
                        if needs_conversion and ind_hop_df[metric_col].median() > 10:
                            ind_hop_df[metric_col] = ind_hop_df[metric_col] * 0.01

                        fig = create_individual_line_chart(
                            ind_hop_df,
                            selected_athletes,
                            metric_col,
                            'RSI',
                            '',
                            False,  # No squad average
                            '10:5 Hop Test - Individual Trends'
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key="hop_ind_line")

    # =====================
    # Quadrant Tests Tab (ForceFrame)
    # =====================
    elif selected_test_tab == "🔄 Quadrant Tests":
        st.markdown("### Quadrant Tests (ForceFrame)")

        # Use ForceFrame data if available
        if forceframe_df is not None and not forceframe_df.empty:
            # Get available test types from actual data
            available_tests = forceframe_df['testTypeName'].dropna().unique().tolist() if 'testTypeName' in forceframe_df.columns else []

            if available_tests:
                selected_test_type = st.selectbox(
                    "Select Test Type:",
                    options=sorted(available_tests),
                    key="quadrant_test_select"
                )

                # Filter to selected test type
                test_df = forceframe_df[forceframe_df['testTypeName'] == selected_test_type].copy()

                # Add Name column if missing (use full_name)
                if 'Name' not in test_df.columns and 'full_name' in test_df.columns:
                    test_df['Name'] = test_df['full_name']

                # Apply filters
                filtered_df, sport, gender = render_filters(test_df, "quadrant")

                if filtered_df.empty:
                    st.warning(f"No data found for {selected_test_type} with current filters.")
                else:
                    # ForceFrame uses inner/outer for bilateral measurements
                    # Inner = Adduction/Internal Rotation, Outer = Abduction/External Rotation
                    metric_cols = {
                        'Inner Left': 'innerLeftMaxForce',
                        'Inner Right': 'innerRightMaxForce',
                        'Outer Left': 'outerLeftMaxForce',
                        'Outer Right': 'outerRightMaxForce'
                    }

                    # Check which columns are available
                    available_cols = {k: v for k, v in metric_cols.items() if v in filtered_df.columns}

                    if not available_cols:
                        st.info("No force metrics found in data.")
                    else:
                        selected_view = st.radio("View:", ["👥 Group View", "🏃 Individual View"], horizontal=True, key="quadrant_view")

                        if selected_view == "👥 Group View":
                            st.markdown(f"**{selected_test_type}** - Latest values per athlete")

                            # Get latest test for each athlete
                            if 'testDateUtc' in filtered_df.columns:
                                filtered_df['testDateUtc'] = pd.to_datetime(filtered_df['testDateUtc'], errors='coerce')
                                latest_df = filtered_df.sort_values('testDateUtc').groupby('Name').last().reset_index()
                            else:
                                latest_df = filtered_df.groupby('Name').last().reset_index()

                            if not latest_df.empty and 'Name' in latest_df.columns:
                                # Create grouped bar chart for Inner vs Outer comparison
                                fig = go.Figure()

                                athletes_list = latest_df['Name'].tolist()

                                # Add bars for each metric
                                colors = {'Inner Left': TEAL_PRIMARY, 'Inner Right': TEAL_LIGHT,
                                         'Outer Left': GOLD_ACCENT, 'Outer Right': '#FFB800'}

                                for label, col in available_cols.items():
                                    if col in latest_df.columns:
                                        values = latest_df[col].fillna(0).tolist()
                                        fig.add_trace(go.Bar(
                                            name=label,
                                            x=athletes_list,
                                            y=values,
                                            marker_color=colors.get(label, GRAY_BLUE)
                                        ))

                                fig.update_layout(
                                    barmode='group',
                                    title=f'{selected_test_type} - Group Comparison',
                                    xaxis_title='Athlete',
                                    yaxis_title='Max Force (N)',
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    font=dict(family='Inter, sans-serif', color='#333'),
                                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                                )
                                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', tickangle=45)
                                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

                                st.plotly_chart(fig, use_container_width=True, key="quadrant_group_chart")

                                # Summary table
                                st.markdown("### Summary Table")
                                summary_cols = ['Name'] + [v for v in available_cols.values() if v in latest_df.columns]
                                if 'testDateUtc' in latest_df.columns:
                                    summary_cols.append('testDateUtc')
                                summary_df = latest_df[summary_cols].copy()

                                # Rename columns for display
                                rename_map = {v: k for k, v in available_cols.items()}
                                if 'testDateUtc' in summary_df.columns:
                                    rename_map['testDateUtc'] = 'Date'
                                summary_df = summary_df.rename(columns=rename_map)

                                # Round numeric columns
                                for col in summary_df.columns:
                                    if summary_df[col].dtype in ['float64', 'float32']:
                                        summary_df[col] = summary_df[col].round(1)

                                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No athlete data available for display.")

                        if selected_view == "🏃 Individual View":
                            athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []

                            if athletes:
                                selected_athlete = st.selectbox(
                                    "Select Athlete:",
                                    options=athletes,
                                    key="quadrant_athlete_select"
                                )

                                if selected_athlete:
                                    # Get all data for this athlete over time
                                    athlete_df = test_df[test_df['Name'] == selected_athlete].copy()

                                    if not athlete_df.empty and 'testDateUtc' in athlete_df.columns:
                                        athlete_df['testDateUtc'] = pd.to_datetime(athlete_df['testDateUtc'], errors='coerce')
                                        athlete_df = athlete_df.sort_values('testDateUtc')

                                        fig = go.Figure()

                                        colors = {'Inner Left': TEAL_PRIMARY, 'Inner Right': TEAL_LIGHT,
                                                 'Outer Left': GOLD_ACCENT, 'Outer Right': '#FFB800'}

                                        for label, col in available_cols.items():
                                            if col in athlete_df.columns:
                                                fig.add_trace(go.Scatter(
                                                    x=athlete_df['testDateUtc'],
                                                    y=athlete_df[col],
                                                    mode='lines+markers',
                                                    name=label,
                                                    line=dict(color=colors.get(label, GRAY_BLUE))
                                                ))

                                        fig.update_layout(
                                            title=f'{selected_athlete} - {selected_test_type} Progression',
                                            xaxis_title='Date',
                                            yaxis_title='Max Force (N)',
                                            plot_bgcolor='white',
                                            paper_bgcolor='white',
                                            font=dict(family='Inter, sans-serif', color='#333'),
                                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                                        )
                                        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                                        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

                                        st.plotly_chart(fig, use_container_width=True, key="quadrant_ind_chart")
                                    else:
                                        st.info("Not enough data points to show progression.")
                            else:
                                st.info("No athletes found in filtered data.")
            else:
                st.warning("No test types found in ForceFrame data.")
        else:
            st.warning("No ForceFrame data available for quadrant tests.")

    # =====================
    # Hip Diagnostics Tab (ForceFrame + ForceDecks)
    # =====================
    elif selected_test_tab == "🦴 Hip Diagnostics":
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1D4D3B 0%, #153829 100%);
             padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid #a08e66;">
            <h2 style="color: white; margin: 0;">Hip Diagnostics</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
                Adduction &amp; Abduction (ForceFrame) | Flexion (ForceFrame) &amp; Extension (ForceDecks)
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Prepare hip data from both devices
        hip_data = prepare_hip_diagnostics_data(forceframe_df, forcedecks_df)
        adab_df = hip_data['adab']
        flex_df = hip_data['flexion']
        ext_df = hip_data['extension']

        has_ff_data = not adab_df.empty or not flex_df.empty
        has_ext_data = not ext_df.empty

        if not has_ff_data and not has_ext_data:
            st.warning("No hip diagnostics data available. Ensure ForceFrame hip tests have been synced.")
        else:
            # --- Controls row ---
            ctrl_col1, ctrl_col2 = st.columns(2)

            with ctrl_col1:
                # Position selector for AD/AB angle
                positions_available = []
                if not adab_df.empty and 'position' in adab_df.columns:
                    positions_available = sorted(adab_df['position'].dropna().unique().tolist())
                if positions_available:
                    pos_idx = get_persisted_selectbox_index("hip_position_select", positions_available)
                    selected_position = st.selectbox(
                        "AD/AB Position (angle):",
                        positions_available,
                        index=pos_idx,
                        key="hip_position_select"
                    )
                    adab_df = adab_df[adab_df['position'] == selected_position].copy()
                elif not adab_df.empty:
                    st.info("No position variants found - showing all Hip AD/AB data.")

            with ctrl_col2:
                unit_option = st.radio(
                    "Display Units:",
                    ["N/kg (body mass relative)", "Newtons (N)"],
                    horizontal=True,
                    key="hip_unit_toggle"
                )
                use_nkg = "N/kg" in unit_option
                unit_label = 'N/kg' if use_nkg else 'N'

            # --- Apply filters using combined ForceFrame hip data ---
            # Build a combined df for filtering (sport/gender/date)
            combined_parts = []
            if not adab_df.empty:
                combined_parts.append(adab_df)
            if not flex_df.empty:
                combined_parts.append(flex_df)
            if combined_parts:
                combined_ff = pd.concat(combined_parts, ignore_index=True)
                filtered_combined, sport, gender = render_filters(combined_ff, "hip")

                # Apply the same sport/gender/date filters back to individual dfs
                filter_names = filtered_combined['Name'].unique() if 'Name' in filtered_combined.columns else []

                if not adab_df.empty and 'Name' in adab_df.columns:
                    adab_df = adab_df[adab_df['Name'].isin(filter_names)]
                if not flex_df.empty and 'Name' in flex_df.columns:
                    flex_df = flex_df[flex_df['Name'].isin(filter_names)]

                # Also filter extension data by the same athletes
                if has_ext_data and 'Name' in ext_df.columns:
                    # Apply sport/gender filters to extension data too
                    if sport != 'All' and 'athlete_sport' in ext_df.columns:
                        ext_df = ext_df[ext_df['athlete_sport'] == sport]
                    if gender != 'All' and 'athlete_sex' in ext_df.columns:
                        ext_df = ext_df[ext_df['athlete_sex'] == gender]
            else:
                sport, gender = 'All', 'All'

            # Pivot extension to L/R per athlete
            ext_pivot = pivot_extension_lr(ext_df, use_nkg=use_nkg)

            # Get suffix for column selection
            suffix = '_nkg' if use_nkg else ''

            # --- View toggle ---
            selected_view = st.radio("View:", ["👥 Group View", "🏃 Individual View"], horizontal=True, key="hip_view")

            if selected_view == "👥 Group View":
                # ---- ROW 1: Adduction | Abduction ----
                if not adab_df.empty:
                    # Get latest test per athlete
                    if 'testDateUtc' in adab_df.columns:
                        adab_df['testDateUtc'] = pd.to_datetime(adab_df['testDateUtc'], errors='coerce')
                        latest_adab = adab_df.sort_values('testDateUtc').groupby('Name').last().reset_index()
                    else:
                        latest_adab = adab_df.groupby('Name').last().reset_index()

                    col1, col2 = st.columns(2)

                    with col1:
                        add_l = f'Add_Left{suffix}'
                        add_r = f'Add_Right{suffix}'
                        if add_l in latest_adab.columns and add_r in latest_adab.columns:
                            fig = create_ranked_side_by_side_chart(
                                latest_adab, add_l, add_r,
                                'Hip Adduction', unit_label,
                                title='Hip Adduction - Left vs Right'
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key="hip_add_group")
                            else:
                                st.info("No Adduction data to display.")
                        else:
                            st.info("Adduction columns not available.")

                    with col2:
                        abd_l = f'Abd_Left{suffix}'
                        abd_r = f'Abd_Right{suffix}'
                        if abd_l in latest_adab.columns and abd_r in latest_adab.columns:
                            fig = create_ranked_side_by_side_chart(
                                latest_adab, abd_l, abd_r,
                                'Hip Abduction', unit_label,
                                title='Hip Abduction - Left vs Right'
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key="hip_abd_group")
                            else:
                                st.info("No Abduction data to display.")
                        else:
                            st.info("Abduction columns not available.")
                else:
                    st.info("No Hip AD/AB data available from ForceFrame.")

                st.markdown("---")

                # ---- ROW 2: Flexion | Extension ----
                col3, col4 = st.columns(2)

                with col3:
                    if not flex_df.empty:
                        if 'testDateUtc' in flex_df.columns:
                            flex_df['testDateUtc'] = pd.to_datetime(flex_df['testDateUtc'], errors='coerce')
                            latest_flex = flex_df.sort_values('testDateUtc').groupby('Name').last().reset_index()
                        else:
                            latest_flex = flex_df.groupby('Name').last().reset_index()

                        flx_l = f'Flx_Left{suffix}'
                        flx_r = f'Flx_Right{suffix}'
                        if flx_l in latest_flex.columns and flx_r in latest_flex.columns:
                            fig = create_ranked_side_by_side_chart(
                                latest_flex, flx_l, flx_r,
                                'Hip Flexion', unit_label,
                                title='Hip Flexion - Left vs Right'
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key="hip_flx_group")
                            else:
                                st.info("No Flexion data to display.")
                        else:
                            st.info("Flexion columns not available.")
                    else:
                        st.info("No Hip Flexion data available from ForceFrame.")

                with col4:
                    if not ext_pivot.empty and 'Ext_Left' in ext_pivot.columns and 'Ext_Right' in ext_pivot.columns:
                        fig = create_ranked_side_by_side_chart(
                            ext_pivot, 'Ext_Left', 'Ext_Right',
                            'Hip Extension', unit_label,
                            title='Hip Extension - Left vs Right (ForceDecks SL Hip Thrust)'
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key="hip_ext_group")
                        else:
                            st.info("Not enough Hip Extension data to chart.")
                    else:
                        st.info("Hip Extension: Limited data from ForceDecks SLISOT hip thrust tests.")

                st.markdown("---")

                # ---- ROW 3: Summary Table ----
                st.markdown("### Comprehensive Hip Summary")

                # Build summary table using latest values
                summary_adab = latest_adab if not adab_df.empty else pd.DataFrame()
                summary_flex = latest_flex if not flex_df.empty else pd.DataFrame()

                summary = create_hip_summary_table(summary_adab, summary_flex, ext_pivot, unit=unit_label)

                if not summary.empty:
                    st.dataframe(summary, use_container_width=True, hide_index=True)
                else:
                    st.info("Not enough data across metrics to generate summary table.")

            if selected_view == "🏃 Individual View":
                # ---- Individual View ----
                # Collect all athletes across hip metrics
                all_athletes = set()
                if not adab_df.empty and 'Name' in adab_df.columns:
                    all_athletes.update(adab_df['Name'].dropna().unique())
                if not flex_df.empty and 'Name' in flex_df.columns:
                    all_athletes.update(flex_df['Name'].dropna().unique())
                if not ext_df.empty and 'Name' in ext_df.columns:
                    all_athletes.update(ext_df['Name'].dropna().unique())
                all_athletes = sorted(all_athletes)

                if not all_athletes:
                    st.info("No athletes found in hip diagnostics data.")
                else:
                    default_athletes = get_persisted_athlete_selection("hip_athlete_select", all_athletes)
                    selected_athletes = st.multiselect(
                        "Select Athletes:",
                        all_athletes,
                        default=default_athletes,
                        key="hip_athlete_select"
                    )

                    metric_options = ['Hip Adduction', 'Hip Abduction', 'Hip Flexion', 'Hip Extension']
                    metric_idx = get_persisted_selectbox_index("hip_ind_metric", metric_options)
                    selected_metric = st.selectbox(
                        "Select Metric:",
                        metric_options,
                        index=metric_idx,
                        key="hip_ind_metric"
                    )

                    if selected_athletes and selected_metric:
                        # Map metric to data source and columns
                        metric_map = {
                            'Hip Adduction': {
                                'df': adab_df, 'left': f'Add_Left{suffix}', 'right': f'Add_Right{suffix}',
                                'date_col': 'testDateUtc'
                            },
                            'Hip Abduction': {
                                'df': adab_df, 'left': f'Abd_Left{suffix}', 'right': f'Abd_Right{suffix}',
                                'date_col': 'testDateUtc'
                            },
                            'Hip Flexion': {
                                'df': flex_df, 'left': f'Flx_Left{suffix}', 'right': f'Flx_Right{suffix}',
                                'date_col': 'testDateUtc'
                            },
                            'Hip Extension': {
                                'df': ext_df, 'left': None, 'right': None,
                                'date_col': 'recordedDateUtc'
                            }
                        }

                        config = metric_map[selected_metric]
                        source_df = config['df']
                        date_col = config['date_col']

                        if source_df is None or source_df.empty:
                            st.info(f"No {selected_metric} data available.")
                        elif selected_metric == 'Hip Extension':
                            # Special handling: single-limb data plotted as separate lines
                            val_col = 'force_value_nkg' if use_nkg and 'force_value_nkg' in source_df.columns else 'force_value'
                            athlete_ext = source_df[source_df['Name'].isin(selected_athletes)].copy()
                            if athlete_ext.empty:
                                st.info("No Hip Extension data for selected athletes.")
                            else:
                                athlete_ext[date_col] = pd.to_datetime(athlete_ext[date_col], errors='coerce')
                                athlete_ext = athlete_ext.sort_values(date_col)

                                fig = go.Figure()
                                for athlete in selected_athletes:
                                    adf = athlete_ext[athlete_ext['Name'] == athlete]
                                    for side_label, side_val, color in [('Left', 'Left', TEAL_PRIMARY), ('Right', 'Right', TEAL_LIGHT)]:
                                        side_data = adf[adf['side'] == side_val]
                                        if not side_data.empty:
                                            fig.add_trace(go.Scatter(
                                                x=side_data[date_col],
                                                y=side_data[val_col],
                                                mode='lines+markers',
                                                name=f'{athlete} - {side_label}',
                                                line=dict(color=color)
                                            ))

                                fig.update_layout(
                                    title=f'Hip Extension Progression ({unit_label})',
                                    xaxis_title='Date', yaxis_title=f'Net Peak Force ({unit_label})',
                                    plot_bgcolor='white', paper_bgcolor='white',
                                    font=dict(family='Inter, sans-serif', color='#333'),
                                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                                )
                                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                                st.plotly_chart(fig, use_container_width=True, key="hip_ext_ind")
                        else:
                            # Standard ForceFrame bilateral metric
                            left_col = config['left']
                            right_col = config['right']

                            if 'Name' not in source_df.columns or left_col not in source_df.columns:
                                st.info(f"Required columns missing for {selected_metric}.")
                            else:
                                athlete_data = source_df[source_df['Name'].isin(selected_athletes)].copy()
                                if athlete_data.empty:
                                    st.info(f"No {selected_metric} data for selected athletes.")
                                else:
                                    athlete_data[date_col] = pd.to_datetime(athlete_data[date_col], errors='coerce')
                                    athlete_data = athlete_data.sort_values(date_col)

                                    fig = go.Figure()
                                    for athlete in selected_athletes:
                                        adf = athlete_data[athlete_data['Name'] == athlete]
                                        if not adf.empty:
                                            fig.add_trace(go.Scatter(
                                                x=adf[date_col], y=adf[left_col],
                                                mode='lines+markers',
                                                name=f'{athlete} - Left',
                                                line=dict(color=TEAL_PRIMARY)
                                            ))
                                            fig.add_trace(go.Scatter(
                                                x=adf[date_col], y=adf[right_col],
                                                mode='lines+markers',
                                                name=f'{athlete} - Right',
                                                line=dict(color=TEAL_LIGHT)
                                            ))

                                    fig.update_layout(
                                        title=f'{selected_metric} Progression ({unit_label})',
                                        xaxis_title='Date', yaxis_title=f'{selected_metric} ({unit_label})',
                                        plot_bgcolor='white', paper_bgcolor='white',
                                        font=dict(family='Inter, sans-serif', color='#333'),
                                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                                    )
                                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                                    st.plotly_chart(fig, use_container_width=True, key="hip_ind_chart")

                                    # Asymmetry table for latest values
                                    if date_col in athlete_data.columns:
                                        latest = athlete_data.sort_values(date_col).groupby('Name').last().reset_index()
                                    else:
                                        latest = athlete_data.groupby('Name').last().reset_index()

                                    asym_table = create_asymmetry_table(latest, left_col, right_col, unit=unit_label)
                                    if not asym_table.empty:
                                        st.markdown(f"**{selected_metric} - Asymmetry Summary**")
                                        st.dataframe(asym_table, use_container_width=True, hide_index=True)

    # =====================
    # Strength RM Tab (Manual Entry)
    # =====================
    elif selected_test_tab == "🏋️ Strength RM":
        st.markdown("### Strength RM (Manual Entry)")

        # Load strength data - check session state first, then files
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        lower_body_path = os.path.join(data_dir, 'sc_lower_body.csv')
        upper_body_path = os.path.join(data_dir, 'sc_upper_body.csv')

        strength_dfs = []

        # Try session state first for lower body (captures recent entries)
        if hasattr(st, 'session_state') and 'sc_lower_body' in st.session_state and not st.session_state.sc_lower_body.empty:
            lb_df = st.session_state.sc_lower_body.copy()
            lb_df['body_region'] = 'Lower Body'
            strength_dfs.append(lb_df)
        elif os.path.exists(lower_body_path):
            try:
                lb_df = pd.read_csv(lower_body_path)
                lb_df['body_region'] = 'Lower Body'
                strength_dfs.append(lb_df)
            except Exception as e:
                st.warning(f"Could not load lower body data: {e}")

        # Try session state first for upper body (captures recent entries)
        if hasattr(st, 'session_state') and 'sc_upper_body' in st.session_state and not st.session_state.sc_upper_body.empty:
            ub_df = st.session_state.sc_upper_body.copy()
            ub_df['body_region'] = 'Upper Body'
            strength_dfs.append(ub_df)
        elif os.path.exists(upper_body_path):
            try:
                ub_df = pd.read_csv(upper_body_path)
                ub_df['body_region'] = 'Upper Body'
                strength_dfs.append(ub_df)
            except Exception as e:
                st.warning(f"Could not load upper body data: {e}")

        if strength_dfs:
            strength_df = pd.concat(strength_dfs, ignore_index=True)

            # Rename columns to match expected format
            if 'athlete' in strength_df.columns and 'Name' not in strength_df.columns:
                strength_df['Name'] = strength_df['athlete']
            if 'date' in strength_df.columns:
                strength_df['date'] = pd.to_datetime(strength_df['date'], errors='coerce')

            # Add sport filter for manual entry data
            # Try to match athletes to sports from ForceDecks data if available
            if 'athlete_sport' not in strength_df.columns and 'full_name' in forcedecks_df.columns:
                # Create athlete-to-sport mapping from ForceDecks
                athlete_sport_map = forcedecks_df.drop_duplicates('full_name').set_index('full_name')['athlete_sport'].to_dict() if 'athlete_sport' in forcedecks_df.columns else {}
                strength_df['athlete_sport'] = strength_df['Name'].map(athlete_sport_map).fillna('Unknown')

            # Sport filter with state persistence
            if 'athlete_sport' in strength_df.columns:
                sports = ['All Sports'] + sorted(strength_df['athlete_sport'].dropna().unique().tolist())
                sport_idx = get_persisted_selectbox_index("strength_sport_filter", sports, 0)
                selected_sport_rm = st.selectbox("Filter by Sport:", sports, index=sport_idx, key="strength_sport_filter")
                if selected_sport_rm != "All Sports":
                    strength_df = strength_df[strength_df['athlete_sport'] == selected_sport_rm]

            if not strength_df.empty:
                # Exercise filter with state persistence
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    body_region_options = ["All", "Lower Body", "Upper Body"]
                    body_region_idx = get_persisted_selectbox_index("strength_body_region", body_region_options, 0)
                    body_region = st.selectbox("Body Region:", body_region_options, index=body_region_idx, key="strength_body_region")

                # Get exercises based on body region filter
                if body_region != "All":
                    exercises = sorted(strength_df[strength_df['body_region'] == body_region]['exercise'].unique())
                else:
                    exercises = sorted(strength_df['exercise'].unique()) if 'exercise' in strength_df.columns else []

                with col2:
                    if exercises:
                        exercise_idx = get_persisted_selectbox_index("strength_exercise_select", exercises, 0)
                        selected_exercise = st.selectbox("Exercise:", exercises, index=exercise_idx, key="strength_exercise_select")
                    else:
                        selected_exercise = None
                        st.info("No exercises available for selected filter.")
                with col3:
                    benchmark = render_benchmark_input('Strength_RM', 'strength_rm')

                if selected_exercise:
                    exercise_df = strength_df[strength_df['exercise'] == selected_exercise].copy()

                    # Initialize variables for export
                    athletes_rm = sorted(exercise_df['Name'].dropna().unique())
                    selected_athletes_rm = []
                    group_fig_rm = None
                    individual_fig_rm = None

                    selected_view_rm = st.radio("View:", ["👥 Group View", "🏃 Individual Progression", "📥 Export"], horizontal=True, key="strength_view")

                    if selected_view_rm == "👥 Group View":
                        # Get latest 1RM for each athlete
                        if 'estimated_1rm' in exercise_df.columns:
                            latest_df = exercise_df.sort_values('date').groupby('Name').last().reset_index()
                            latest_df['Name'] = latest_df['Name'].astype(str)

                            group_fig_rm = create_ranked_bar_chart(
                                latest_df,
                                'estimated_1rm',
                                f'{selected_exercise} Est. 1RM',
                                'kg',
                                benchmark,
                                f'{selected_exercise} - Estimated 1RM Comparison'
                            )
                            if group_fig_rm:
                                st.plotly_chart(group_fig_rm, use_container_width=True, key="strength_group_bar")
                        else:
                            st.info("No estimated 1RM data available.")

                    if selected_view_rm == "🏃 Individual Progression":
                        if athletes_rm:
                            selected_athletes_rm = st.multiselect(
                                "Select Athletes:",
                                options=athletes_rm,
                                default=get_persisted_athlete_selection("strength_athlete_select", athletes_rm),
                                key="strength_athlete_select"
                            )

                            show_squad = st.checkbox("Show Squad Average", value=True, key="strength_show_squad")

                            if selected_athletes_rm and 'estimated_1rm' in exercise_df.columns:
                                individual_fig_rm = create_individual_line_chart(
                                    exercise_df,
                                    selected_athletes_rm,
                                    'estimated_1rm',
                                    f'{selected_exercise} Est. 1RM',
                                    'kg',
                                    show_squad,
                                    f'{selected_exercise} - 1RM Progression'
                                )
                                if individual_fig_rm:
                                    st.plotly_chart(individual_fig_rm, use_container_width=True, key="strength_ind_line")

                    if selected_view_rm == "📥 Export":
                        # Export tab
                        st.markdown(f"#### Export {selected_exercise} Data")
                        st.markdown("Download group summary or individual athlete data as Excel or PDF with charts.")

                        # Get latest data for group export
                        if 'estimated_1rm' in exercise_df.columns:
                            latest_df = exercise_df.sort_values('date').groupby('Name').last().reset_index()
                        else:
                            latest_df = pd.DataFrame()

                        # Athletes for individual export
                        export_athletes = st.multiselect(
                            "Select Athletes for Individual Export:",
                            options=athletes_rm,
                            default=get_persisted_athlete_selection("strength_export_athletes", athletes_rm),
                            key="strength_export_athletes"
                        )

                        # Recreate individual figure for export if athletes selected
                        export_individual_fig = None
                        if export_athletes and 'estimated_1rm' in exercise_df.columns:
                            export_individual_fig = create_individual_line_chart(
                                exercise_df,
                                export_athletes,
                                'estimated_1rm',
                                f'{selected_exercise} Est. 1RM',
                                'kg',
                                True,
                                f'{selected_exercise} - 1RM Progression'
                            )

                        render_export_buttons_with_pdf(
                            group_df=latest_df,
                            individual_df=exercise_df,
                            selected_athletes=export_athletes,
                            metric_col='estimated_1rm',
                            group_fig=group_fig_rm,
                            individual_fig=export_individual_fig,
                            name_col='Name',
                            test_name=f"Strength_RM_{selected_exercise}",
                            key_prefix="strength_rm_export"
                        )
            else:
                st.info("No strength data available for the selected filters.")
        else:
            st.info("Strength RM data comes from manual entry. Enter data through the Data Entry tab.")
            st.markdown("""
            **Available Exercises:**
            - Back Squat, Front Squat, Deadlift (Lower Body)
            - Bench Press, Pull Up, Overhead Press (Upper Body)

            **Metrics:**
            - Estimated 1RM (kg)
            - Weight × Reps × Sets
            """)

    # =====================
    # Broad Jump Tab (Manual Entry)
    # =====================
    elif selected_test_tab == "🦘 Broad Jump":
        st.markdown("### Broad Jump (Manual Entry)")
        st.info("Broad Jump data comes from manual entry. Enter data through the Data Entry tab.")

        # Try to load manual entry data
        broad_jump_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'broad_jump.csv')

        if os.path.exists(broad_jump_path):
            try:
                bj_df = pd.read_csv(broad_jump_path)
                if 'date' in bj_df.columns:
                    bj_df['date'] = pd.to_datetime(bj_df['date'], errors='coerce')

                if not bj_df.empty:
                    filtered_df, sport, gender = render_filters(bj_df, "broad_jump")

                    col1, col2 = st.columns([3, 1])
                    with col2:
                        benchmark = render_benchmark_input('Broad_Jump', 'broad_jump')

                    selected_view = st.radio("View:", ["👥 Group View", "🏃 Individual View"], horizontal=True, key="broadjump_view")

                    if selected_view == "👥 Group View":
                        metric_col = 'distance_m' if 'distance_m' in filtered_df.columns else 'distance'
                        if metric_col in filtered_df.columns:
                            fig = create_ranked_bar_chart(
                                filtered_df,
                                metric_col,
                                'Broad Jump Distance',
                                'm',
                                benchmark,
                                'Broad Jump - Distance'
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key="broad_jump_group_bar")

                    if selected_view == "🏃 Individual View":
                        athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []
                        if athletes:
                            selected_athletes = st.multiselect(
                                "Select Athletes:",
                                options=athletes,
                                default=get_persisted_athlete_selection("broad_jump_athlete_select", athletes),
                                key="broad_jump_athlete_select"
                            )

                            show_squad = st.checkbox("Show Squad Average", value=True, key="broad_jump_show_squad")

                            if selected_athletes and metric_col in filtered_df.columns:
                                fig = create_individual_line_chart(
                                    filtered_df,
                                    selected_athletes,
                                    metric_col,
                                    'Broad Jump Distance',
                                    'm',
                                    show_squad,
                                    'Broad Jump - Individual Trends'
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True, key="broad_jump_ind_line")
            except Exception as e:
                st.warning(f"Could not load Broad Jump data: {e}")
        else:
            st.markdown("""
            **Broad Jump measures:**
            - Standing broad jump distance (m)
            - Can be used as power indicator

            **Data Entry:**
            Enter broad jump results through the Data Entry tab.
            """)

    # =====================
    # Fitness Tests Tab (6 Min Aerobic, etc.)
    # =====================
    elif selected_test_tab == "🏃 Fitness Tests":
        st.markdown("### Fitness Testing")
        st.markdown("*Aerobic capacity and endurance tests*")

        fitness_test_options = [
            '6 Minute Aerobic',
            'VO2 Max Estimate',
            'Yo-Yo Test',
            '30-15 IFT'
        ]
        selected_fitness_test = st.selectbox("Select Test:", fitness_test_options, key="fitness_test_select")

        # Try to load fitness test data
        fitness_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fitness_tests.csv')

        if os.path.exists(fitness_path):
            try:
                fitness_df = pd.read_csv(fitness_path)
                if 'date' in fitness_df.columns:
                    fitness_df['date'] = pd.to_datetime(fitness_df['date'], errors='coerce')

                if not fitness_df.empty and 'test_type' in fitness_df.columns:
                    test_df = fitness_df[fitness_df['test_type'] == selected_fitness_test].copy()

                    if not test_df.empty:
                        filtered_df, sport, gender = render_filters(test_df, "fitness")

                        col1, col2 = st.columns([3, 1])
                        with col2:
                            benchmark = render_benchmark_input(selected_fitness_test.replace(' ', '_'), 'fitness')

                        selected_view = st.radio("View:", ["👥 Group View", "🏃 Individual View"], horizontal=True, key="fitness_view")

                        if selected_view == "👥 Group View":
                            # Common fitness metrics
                            metric_cols = ['distance', 'vo2_max', 'result', 'score']
                            metric_col = None
                            for col in metric_cols:
                                if col in filtered_df.columns:
                                    metric_col = col
                                    break

                            if metric_col:
                                unit = 'm' if metric_col == 'distance' else 'ml/kg/min' if metric_col == 'vo2_max' else ''
                                fig = create_ranked_bar_chart(
                                    filtered_df,
                                    metric_col,
                                    selected_fitness_test,
                                    unit,
                                    benchmark,
                                    f'{selected_fitness_test} Results'
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True, key="fitness_group_bar")

                        if selected_view == "🏃 Individual View":
                            athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []
                            if athletes:
                                selected_athletes = st.multiselect(
                                    "Select Athletes:",
                                    options=athletes,
                                    default=get_persisted_athlete_selection("fitness_athlete_select", athletes),
                                    key="fitness_athlete_select"
                                )

                                show_squad = st.checkbox("Show Squad Average", value=True, key="fitness_show_squad")

                                if selected_athletes and metric_col:
                                    fig = create_individual_line_chart(
                                        filtered_df,
                                        selected_athletes,
                                        metric_col,
                                        selected_fitness_test,
                                        unit,
                                        show_squad,
                                        f'{selected_fitness_test} - Individual Trends'
                                    )
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True, key="fitness_ind_line")
                    else:
                        st.info(f"No {selected_fitness_test} data available.")
            except Exception as e:
                st.warning(f"Could not load fitness test data: {e}")
        else:
            st.info("Fitness test data comes from manual entry. Enter data through the Data Entry tab.")
            st.markdown("""
            **Available Tests:**
            - **6 Minute Aerobic** - Distance covered in 6 minutes
            - **VO2 Max Estimate** - Aerobic capacity
            - **Yo-Yo Test** - Intermittent recovery test
            - **30-15 IFT** - Intermittent fitness test

            **Metrics:**
            - Distance (m)
            - VO2 Max (ml/kg/min)
            - Final stage/level
            """)

    # =====================
    # Plyo Pushup Tab (Upper Body Power)
    # =====================
    elif selected_test_tab == "💥 Plyo Pushup":
        st.markdown("### Plyo Pushup (Upper Body Power)")

        # Filter for Plyo Pushup tests - VALD test type is PPU
        pp_df = forcedecks_df[forcedecks_df['testType'] == 'PPU'].copy() if 'testType' in forcedecks_df.columns else pd.DataFrame()

        # Ensure Name column exists (map from full_name if needed)
        if not pp_df.empty and 'Name' not in pp_df.columns and 'full_name' in pp_df.columns:
            pp_df['Name'] = pp_df['full_name']

        if pp_df.empty:
            st.warning("No Plyo Pushup test data available.")
            st.markdown("""
            **Plyo Pushup measures:**
            - Pushup Height (cm) - primary metric
            - Flight Time (s)
            - Upper Body Mass (kg)

            *Used to assess upper body explosive power*
            """)
        else:
            filtered_df, sport, gender = render_filters(pp_df, "plyo_pushup")

            col1, col2 = st.columns([3, 1])
            with col2:
                benchmark = render_benchmark_input('Plyo_Pushup', 'plyo_pushup')

            selected_view = st.radio("View:", ["👥 Group View", "🏃 Individual View"], horizontal=True, key="ppu_view")

            if selected_view == "👥 Group View":
                # Look for pushup metrics - PUSHUP_HEIGHT is the primary metric
                metric_col = None
                metric_name = 'Pushup Height'
                metric_unit = 'cm'

                # Check columns in order of preference
                # Include both UPPERCASE (local_sync) and mixed-case (legacy API) formats
                metric_options = [
                    ('PUSHUP_HEIGHT', 'Pushup Height', 'cm'),
                    ('Pushup Height_Trial', 'Pushup Height', 'cm'),
                    ('PUSHUP_HEIGHT_INCHES', 'Pushup Height', 'in'),
                    ('FLIGHT_TIME', 'Flight Time', 's'),
                    ('Flight Time_Trial', 'Flight Time', 's'),
                    ('BODYMASS_RELATIVE_TAKEOFF_POWER', 'Relative Peak Power', 'W/kg'),
                    ('Peak Power / BM_Trial', 'Relative Peak Power', 'W/kg'),
                    ('PEAK_TAKEOFF_FORCE', 'Peak Takeoff Force', 'N'),
                    ('Peak Takeoff Force_Trial', 'Peak Takeoff Force', 'N'),
                ]

                for col, name, unit in metric_options:
                    if col in filtered_df.columns and filtered_df[col].notna().sum() > 0:
                        metric_col = col
                        metric_name = name
                        metric_unit = unit
                        break

                # Fallback: search for any column with 'PUSHUP' or 'pushup' in name
                if metric_col is None:
                    for col in filtered_df.columns:
                        if 'PUSHUP' in col.upper() and filtered_df[col].notna().sum() > 0:
                            metric_col = col
                            metric_name = col.replace('_', ' ').title()
                            metric_unit = ''
                            break

                if metric_col:
                    # Convert m to cm if values look like meters
                    if metric_unit == 'cm' and filtered_df[metric_col].median() < 1:
                        filtered_df[metric_col] = filtered_df[metric_col] * 100

                    # Get latest test per athlete for group view
                    if 'recordedDateUtc' in filtered_df.columns:
                        latest_df = filtered_df.sort_values('recordedDateUtc').groupby('Name').last().reset_index()
                    else:
                        latest_df = filtered_df.groupby('Name').last().reset_index()

                    fig = create_ranked_bar_chart(
                        latest_df,
                        metric_col,
                        metric_name,
                        metric_unit,
                        benchmark,
                        f'Plyo Pushup - {metric_name}'
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="plyo_pushup_group_bar")
                else:
                    st.warning("Plyo Pushup metrics not found in data.")
                    # Show available columns for debugging
                    ppu_cols = [c for c in filtered_df.columns if any(k in c.upper() for k in ['PUSH', 'FLIGHT', 'TAKEOFF', 'POWER'])]
                    if ppu_cols:
                        st.info(f"Available PPU columns: {', '.join(ppu_cols[:10])}")

            if selected_view == "🏃 Individual View":
                athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []

                if athletes:
                    selected_athletes = st.multiselect(
                        "Select Athletes:",
                        options=athletes,
                        default=get_persisted_athlete_selection("plyo_pushup_athlete_select", athletes),
                        key="plyo_pushup_athlete_select"
                    )

                    if selected_athletes and metric_col:
                        # Get all PPU data for trends (still apply sport/gender filters)
                        all_ppu = pp_df.copy()
                        if sport != 'All' and 'athlete_sport' in all_ppu.columns:
                            all_ppu = all_ppu[all_ppu['athlete_sport'] == sport]
                        if gender != 'All' and 'athlete_sex' in all_ppu.columns:
                            all_ppu = all_ppu[all_ppu['athlete_sex'] == gender]

                        fig = create_individual_line_chart(
                            all_ppu,
                            selected_athletes,
                            metric_col,
                            metric_name,
                            metric_unit,
                            False,  # No squad average
                            'Plyo Pushup - Individual Trends'
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key="plyo_pushup_ind_line")
                else:
                    st.info("No athletes found in filtered data.")

    # =====================
    # DynaMo Tab (Grip Strength)
    # =====================
    elif selected_test_tab == "✊ DynaMo":
        st.markdown("### DynaMo (Grip Strength)")

        # Use passed dynamo_df if available, otherwise load from file
        if dynamo_df is None or dynamo_df.empty:
            try:
                from .data_loader import load_vald_data
                dynamo_df = load_vald_data('dynamo')
            except Exception:
                dynamo_df = pd.DataFrame()

        # Standardize column names
        if dynamo_df is not None and not dynamo_df.empty:
            if 'full_name' in dynamo_df.columns and 'Name' not in dynamo_df.columns:
                dynamo_df['Name'] = dynamo_df['full_name']
            if 'startTimeUTC' in dynamo_df.columns and 'recordedDateUtc' not in dynamo_df.columns:
                dynamo_df['recordedDateUtc'] = pd.to_datetime(dynamo_df['startTimeUTC'], errors='coerce')
            elif 'analysedDateUTC' in dynamo_df.columns and 'recordedDateUtc' not in dynamo_df.columns:
                dynamo_df['recordedDateUtc'] = pd.to_datetime(dynamo_df['analysedDateUTC'], errors='coerce')
        else:
            dynamo_df = pd.DataFrame()

        if dynamo_df.empty:
            st.warning("No DynaMo (grip strength) data available. Run local_sync.py to fetch data.")
            st.info("""
            **DynaMo measures:**
            - Peak grip force (N)
            - Average grip force (N)
            - Left/Right comparison
            - Grip strength asymmetry
            """)
        else:
            # Filter for grip squeeze tests only
            if 'movement' in dynamo_df.columns:
                grip_df = dynamo_df[dynamo_df['movement'] == 'GripSqueeze'].copy()
            else:
                grip_df = dynamo_df.copy()

            if grip_df.empty:
                st.warning("No grip squeeze tests found in DynaMo data.")
            else:
                # Filters
                filtered_df, sport, gender = render_filters(grip_df, "dynamo")

                if filtered_df.empty:
                    st.warning("No DynaMo data for selected filters.")
                else:
                    # Show available metrics
                    st.markdown(f"**{len(filtered_df)} Grip Strength tests found**")

                    col1, col2 = st.columns([3, 1])
                    with col2:
                        benchmark = st.number_input(
                            "Benchmark (N):",
                            min_value=0.0,
                            value=400.0,
                            step=10.0,
                            key="dynamo_benchmark"
                        )

                    # Use maxForceNewtons as the primary metric
                    metric_col = 'maxForceNewtons' if 'maxForceNewtons' in filtered_df.columns else None

                    if metric_col:
                        selected_view = st.radio("View:", ["👥 Group View", "🏃 Individual View"], horizontal=True, key="dynamo_view")

                        if selected_view == "👥 Group View":
                            # Get latest test per athlete for group view
                            if 'recordedDateUtc' in filtered_df.columns:
                                latest_df = filtered_df.sort_values('recordedDateUtc').groupby('Name').last().reset_index()
                            else:
                                latest_df = filtered_df.groupby('Name').last().reset_index()

                            fig = create_ranked_bar_chart(
                                latest_df,
                                metric_col,
                                'Max Grip Force',
                                'N',
                                benchmark,
                                'DynaMo - Grip Strength'
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key="dynamo_group_bar")

                        if selected_view == "🏃 Individual View":
                            athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []
                            if athletes:
                                selected_athletes = st.multiselect(
                                    "Select Athletes:",
                                    options=athletes,
                                    default=get_persisted_athlete_selection("dynamo_athlete_select", athletes),
                                    key="dynamo_athlete_select"
                                )

                                if selected_athletes:
                                    # Get all grip data for trends (still apply sport/gender filters)
                                    all_grip = grip_df.copy()
                                    if sport != 'All' and 'athlete_sport' in all_grip.columns:
                                        all_grip = all_grip[all_grip['athlete_sport'] == sport]
                                    if gender != 'All' and 'athlete_sex' in all_grip.columns:
                                        all_grip = all_grip[all_grip['athlete_sex'] == gender]

                                    fig = create_individual_line_chart(
                                        all_grip,
                                        selected_athletes,
                                        metric_col,
                                        'Max Grip Force',
                                        'N',
                                        False,  # No squad average
                                        'DynaMo - Individual Trends'
                                    )
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True, key="dynamo_ind_line")
                            else:
                                st.info("No athletes found in filtered data.")
                    else:
                        st.warning("Grip force metric (maxForceNewtons) not found in data.")

    # =====================
    # Balance Tab (QSB/SLSB for Shooting)
    # =====================
    elif selected_test_tab == "⚖️ Balance":
        st.markdown("### Balance Testing (QSB / SLSB)")
        st.markdown("*Quiet Static Balance & Single Leg Static Balance - primarily used by Shooting athletes*")

        # Filter for balance tests
        balance_test_types = ['QSB', 'SLSB']
        balance_df = forcedecks_df[forcedecks_df['testType'].isin(balance_test_types)].copy() if 'testType' in forcedecks_df.columns else pd.DataFrame()

        if balance_df.empty:
            st.warning("No Balance test data available.")
            st.markdown("""
            **Balance test types:**
            - **QSB** - Quiet Static Balance (bilateral)
            - **SLSB** - Single Leg Static Balance

            **Key metrics:**
            - CoP Total Excursion (mm) - lower is better
            - CoP Mean Velocity (mm/s) - lower is better
            - CoP Ellipse Area (mm²) - smaller is better
            """)
        else:
            filtered_df, sport, gender = render_filters(balance_df, "balance")

            # Metric selection with unit conversion factors
            # VALD stores values in m/m², display in mm/mm²
            balance_metrics = [
                ('BAL_COP_MEAN_VELOCITY', 'CoP Mean Velocity', 'mm/s', 1000),      # m/s -> mm/s
                ('BAL_COP_TOTAL_EXCURSION', 'CoP Total Excursion', 'mm', 1000),    # m -> mm
                ('BAL_COP_ELLIPSE_AREA', 'CoP Ellipse Area', 'mm²', 1000000),      # m² -> mm²
                ('BAL_COP_RANGE_MEDLAT', 'CoP Range Med-Lat', 'mm', 1000),         # m -> mm
                ('BAL_COP_RANGE_ANTPOST', 'CoP Range Ant-Post', 'mm', 1000),       # m -> mm
            ]

            available_metrics = [(col, name, unit, conv) for col, name, unit, conv in balance_metrics if col in filtered_df.columns]

            if not available_metrics:
                st.warning("No balance metrics found in data.")
            else:
                col1, col2 = st.columns([3, 1])
                with col2:
                    metric_options = [f"{name} ({unit})" for _, name, unit, _ in available_metrics]
                    selected_metric_display = st.selectbox("Metric:", metric_options, key="balance_metric")
                    selected_idx = metric_options.index(selected_metric_display)
                    metric_col, metric_name, metric_unit, conversion = available_metrics[selected_idx]

                    # For balance, lower is better - no benchmark needed
                    st.info("Lower values = better stability")

                # Apply unit conversion to display data
                display_df = filtered_df.copy()
                if metric_col in display_df.columns and conversion != 1:
                    display_df[metric_col] = display_df[metric_col] * conversion

                selected_view = st.radio("View:", ["👥 Group View", "🏃 Individual View"], horizontal=True, key="balance_view")

                if selected_view == "👥 Group View":
                    if metric_col in display_df.columns:
                        # For balance, we want to rank by lowest (best)
                        fig = create_ranked_bar_chart(
                            display_df,
                            metric_col,
                            metric_name,
                            metric_unit,
                            None,  # No benchmark
                            f'Balance Test - {metric_name}'
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key="balance_group_bar")
                    else:
                        st.warning(f"Metric {metric_col} not found in data.")

                if selected_view == "🏃 Individual View":
                    athletes = sorted(display_df['Name'].dropna().unique()) if 'Name' in display_df.columns else []

                    if athletes:
                        selected_athletes = st.multiselect(
                            "Select Athletes:",
                            options=athletes,
                            default=get_persisted_athlete_selection("balance_athlete_select", athletes),
                            key="balance_athlete_select"
                        )

                        show_squad = st.checkbox("Show Squad Average", value=True, key="balance_show_squad")

                        if selected_athletes and metric_col in display_df.columns:
                            # Use converted display_df for individual chart too
                            fig = create_individual_line_chart(
                                display_df,
                                selected_athletes,
                                metric_col,
                                metric_name,
                                metric_unit,
                                show_squad,
                                f'Balance Test - {metric_name} Trends'
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key="balance_ind_line")
                    else:
                        st.warning("No athletes found in filtered data.")
