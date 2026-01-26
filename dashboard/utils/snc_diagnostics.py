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
from datetime import datetime, timedelta

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
TEAL_LIGHT = '#2A6A50'
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
        'test_type_filter': ['CMJ']
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
def resolve_metric_column(df: pd.DataFrame, metric_spec) -> Optional[str]:
    """Find actual column name from a metric specification (string or list of possibilities)."""
    if metric_spec is None:
        return None
    if isinstance(metric_spec, str):
        return metric_spec if metric_spec in df.columns else None
    if isinstance(metric_spec, list):
        for col in metric_spec:
            if col in df.columns:
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
        # Sport/Group filter
        sports = ['All']
        if 'athlete_sport' in df.columns:
            sports += sorted([s for s in df['athlete_sport'].dropna().unique()])
        selected_sport = st.selectbox("Sport/Group:", sports, key=f"{key_prefix}_sport")

    with col2:
        # Gender filter
        genders = ['All']
        if 'athlete_sex' in df.columns:
            genders += sorted([g for g in df['athlete_sex'].dropna().unique() if g])
        selected_gender = st.selectbox("Gender:", genders, key=f"{key_prefix}_gender")

    with col3:
        # Date filter
        date_options = ['Most Recent', 'Last 7 Days', 'Last 30 Days', 'Last 90 Days', 'All Time', 'Custom Range']
        selected_date = st.selectbox("Date Range:", date_options, key=f"{key_prefix}_date")

    # Apply filters
    filtered_df = df.copy()

    if selected_sport != 'All' and 'athlete_sport' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['athlete_sport'] == selected_sport]

    if selected_gender != 'All' and 'athlete_sex' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['athlete_sex'] == selected_gender]

    # Date filtering
    if 'recordedDateUtc' in filtered_df.columns:
        filtered_df['recordedDateUtc'] = pd.to_datetime(filtered_df['recordedDateUtc'], errors='coerce')

        if selected_date == 'Most Recent':
            # Get most recent test per athlete
            if 'Name' in filtered_df.columns:
                idx = filtered_df.groupby('Name')['recordedDateUtc'].idxmax()
                filtered_df = filtered_df.loc[idx]
        elif selected_date == 'Last 7 Days':
            cutoff = datetime.now() - timedelta(days=7)
            filtered_df = filtered_df[filtered_df['recordedDateUtc'] >= cutoff]
        elif selected_date == 'Last 30 Days':
            cutoff = datetime.now() - timedelta(days=30)
            filtered_df = filtered_df[filtered_df['recordedDateUtc'] >= cutoff]
        elif selected_date == 'Last 90 Days':
            cutoff = datetime.now() - timedelta(days=90)
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
    title: str = None
) -> go.Figure:
    """
    Create a vertical ranked bar chart with squad average and benchmark lines.

    Like the PowerBI example: bars sorted by value, with horizontal reference lines.
    """
    if 'Name' not in df.columns or metric_col not in df.columns:
        return None

    # Get data and sort
    plot_df = df[['Name', metric_col]].dropna()
    if plot_df.empty:
        return None

    plot_df = plot_df.sort_values(metric_col, ascending=False)

    # Calculate squad average
    squad_avg = plot_df[metric_col].mean()

    # Create figure
    fig = go.Figure()

    # Add bars
    fig.add_trace(go.Bar(
        x=plot_df['Name'],
        y=plot_df[metric_col],
        marker_color=TEAL_PRIMARY,
        text=[f"{v:.1f}" for v in plot_df[metric_col]],
        textposition='outside',
        textfont=dict(size=10),
        name='Athletes'
    ))

    # Add squad average line (green dashed)
    fig.add_hline(
        y=squad_avg,
        line_dash="dash",
        line_color=SQUAD_AVG_COLOR,
        line_width=2,
        annotation_text=f"Squad Avg: {squad_avg:.2f}",
        annotation_position="right",
        annotation_font_color=SQUAD_AVG_COLOR
    )

    # Add benchmark line (blue dashed) if provided
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

    # Update layout
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
        marker_color=CORAL_ACCENT,
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
    athlete_df['recordedDateUtc'] = pd.to_datetime(athlete_df['recordedDateUtc'])
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
    show_squad_avg: bool = True,
    title: str = None
) -> go.Figure:
    """
    Create individual line chart with multiple athlete selection and squad average.
    """
    if 'Name' not in df.columns or 'recordedDateUtc' not in df.columns:
        return None

    if metric_col not in df.columns:
        return None

    fig = go.Figure()

    # Add line for each selected athlete
    for i, athlete in enumerate(selected_athletes):
        athlete_df = df[df['Name'] == athlete].sort_values('recordedDateUtc')
        if athlete_df.empty:
            continue

        color = MULTI_LINE_COLORS[i % len(MULTI_LINE_COLORS)]

        fig.add_trace(go.Scatter(
            x=athlete_df['recordedDateUtc'],
            y=athlete_df[metric_col],
            mode='markers+lines',
            marker=dict(size=8, color=color),
            line=dict(color=color, width=2),
            name=athlete
        ))

    # Add squad average line if requested
    if show_squad_avg:
        # Calculate squad average per date
        squad_avg = df.groupby('recordedDateUtc')[metric_col].mean().reset_index()
        squad_avg = squad_avg.sort_values('recordedDateUtc')

        fig.add_trace(go.Scatter(
            x=squad_avg['recordedDateUtc'],
            y=squad_avg[metric_col],
            mode='lines',
            line=dict(color=SQUAD_AVG_COLOR, width=2, dash='dash'),
            name='Squad Average'
        ))

    # Update layout
    fig.update_layout(
        title=title or f"{metric_name} - Individual Trends",
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
        athlete_df['date'] = pd.to_datetime(athlete_df['date'])

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


def render_snc_diagnostics_tab(forcedecks_df: pd.DataFrame, nordbord_df: pd.DataFrame = None, forceframe_df: pd.DataFrame = None):
    """
    Main function to render the S&C Diagnostics Canvas tab.
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1D4D3B 0%, #153829 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">S&C Diagnostics Canvas</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Comprehensive strength & conditioning testing analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Test type selector
    test_tabs = st.tabs([
        "📊 IMTP",
        "🦘 CMJ",
        "🦵 SL Tests",
        "💪 NordBord",
        "🏃 10:5 Hop",
        "🔄 Quadrant Tests",
        "🏋️ Strength RM",
        "✊ DynaMo",
        "⚖️ Balance"
    ])

    # =====================
    # IMTP Tab
    # =====================
    with test_tabs[0]:
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
            view_tabs = st.tabs(["👥 Group View", "🏃 Individual View"])

            with view_tabs[0]:
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

            with view_tabs[1]:
                # Individual line chart with multi-select
                athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []

                if athletes:
                    selected_athletes = st.multiselect(
                        "Select Athletes:",
                        options=athletes,
                        default=[athletes[0]] if athletes else [],
                        key="imtp_athlete_select"
                    )

                    show_squad = st.checkbox("Show Squad Average", value=True, key="imtp_show_squad")

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
                            show_squad,
                            'IMTP - Individual Trends'
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key="imtp_ind_line")
                else:
                    st.info("No athletes found in filtered data.")

    # =====================
    # CMJ Tab
    # =====================
    with test_tabs[1]:
        st.markdown("### Counter Movement Jump (CMJ)")

        cmj_df = forcedecks_df[forcedecks_df['testType'] == 'CMJ'].copy() if 'testType' in forcedecks_df.columns else pd.DataFrame()

        if cmj_df.empty:
            st.warning("No CMJ test data available.")
        else:
            filtered_df, sport, gender = render_filters(cmj_df, "cmj")

            col1, col2 = st.columns([3, 1])
            with col2:
                benchmark = render_benchmark_input('CMJ', 'cmj')

            view_tabs = st.tabs(["👥 Group View", "🏃 Individual View"])

            with view_tabs[0]:
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

            with view_tabs[1]:
                athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []

                if athletes:
                    selected_athletes = st.multiselect(
                        "Select Athletes:",
                        options=athletes,
                        default=[athletes[0]] if athletes else [],
                        key="cmj_athlete_select"
                    )

                    show_squad = st.checkbox("Show Squad Average", value=True, key="cmj_show_squad")

                    if selected_athletes:
                        all_cmj = forcedecks_df[forcedecks_df['testType'] == 'CMJ'].copy()

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
                            show_squad,
                            'CMJ - Individual Trends'
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key="cmj_ind_line")

    # =====================
    # SL Tests Tab
    # =====================
    with test_tabs[2]:
        st.markdown("### Single Leg Tests")

        # Map display names to actual VALD test type codes
        sl_test_mapping = {
            'SL ISO Squat': 'SLISOSQT',
            'SL IMTP': 'SLIMTP',
            'SL CMJ': 'SLCMRJ',
            'SL Drop Jump': 'SLDJ',
            'SL Jump': 'SLJ',
            'SL Hop Jump': 'SLHJ'
        }

        sl_test_options = list(sl_test_mapping.keys())
        selected_sl_test = st.selectbox("Select Test:", sl_test_options, key="sl_test_select")

        # Get the actual test type code
        test_type_code = sl_test_mapping.get(selected_sl_test, selected_sl_test)

        # Filter for selected SL test using exact match
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

            view_tabs = st.tabs(["👥 Group View", "🏃 Individual View"])

            with view_tabs[0]:
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

            with view_tabs[1]:
                athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []

                if athletes:
                    selected_athletes = st.multiselect(
                        "Select Athletes:",
                        options=athletes,
                        default=[athletes[0]] if athletes else [],
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
    with test_tabs[3]:
        st.markdown("### NordBord - Hamstring Strength")

        if nordbord_df is None or nordbord_df.empty:
            st.warning("No NordBord data available.")
        else:
            filtered_df, sport, gender = render_filters(nordbord_df, "nordbord")

            col1, col2 = st.columns([3, 1])
            with col2:
                benchmark = render_benchmark_input('NordBord', 'nordbord')

            view_tabs = st.tabs(["👥 Group View", "🏃 Individual View"])

            with view_tabs[0]:
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

            with view_tabs[1]:
                athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []

                if athletes:
                    selected_athletes = st.multiselect(
                        "Select Athletes:",
                        options=athletes,
                        default=[athletes[0]] if athletes else [],
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
    with test_tabs[4]:
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

            view_tabs = st.tabs(["👥 Group View", "🏃 Individual View"])

            with view_tabs[0]:
                metric_col = None
                needs_conversion = False

                # Check for RSI columns - local_sync format first, then legacy
                rsi_columns = [
                    ('RSI_MODIFIED', True),       # local_sync format, needs /100 conversion
                    ('RSI_MODIFIED_IMP_MOM', True),
                    ('RSI', True),
                    ('RSI-modified_Trial', False),  # legacy format, already correct
                    ('RSI (Flight/Contact Time)_Trial', False),
                    ('Best RSI (Flight/Contact Time)_Trial', False),
                ]

                for col, convert in rsi_columns:
                    if col in filtered_df.columns:
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

            with view_tabs[1]:
                athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []

                if athletes:
                    selected_athletes = st.multiselect(
                        "Select Athletes:",
                        options=athletes,
                        default=[athletes[0]] if athletes else [],
                        key="hop_athlete_select"
                    )

                    show_squad = st.checkbox("Show Squad Average", value=True, key="hop_show_squad")

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
                            show_squad,
                            '10:5 Hop Test - Individual Trends'
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key="hop_ind_line")

    # =====================
    # Quadrant Tests Tab (ForceFrame)
    # =====================
    with test_tabs[5]:
        st.markdown("### Quadrant Tests (ForceFrame)")

        quadrant_test_options = [
            'Trunk Profile (Supine/Prone/Lateral)',
            '4-Way Neck Profile',
            'Shoulder IR/ER',
            'Hip Adduction/Abduction'
        ]
        selected_quadrant_test = st.selectbox("Select Test:", quadrant_test_options, key="quadrant_test_select")

        # Use ForceFrame data if available
        if forceframe_df is not None and not forceframe_df.empty:
            filtered_df, sport, gender = render_filters(forceframe_df, "quadrant")

            # Define metric columns based on selected test
            # These column names may need adjustment based on actual ForceFrame data structure
            if 'Trunk' in selected_quadrant_test:
                metric_cols = {
                    'Supine': 'supine_force',
                    'Prone': 'prone_force',
                    'Lateral_Left': 'lateral_left_force',
                    'Lateral_Right': 'lateral_right_force'
                }
                metric_name = 'Force'
                unit = 'N'
            elif 'Neck' in selected_quadrant_test:
                metric_cols = {
                    'Flexion': 'neck_flexion_force',
                    'Extension': 'neck_extension_force',
                    'Left': 'neck_left_force',
                    'Right': 'neck_right_force'
                }
                metric_name = 'Force'
                unit = 'N'
            elif 'Shoulder' in selected_quadrant_test:
                metric_cols = {
                    'IR': 'shoulder_ir_force',
                    'ER': 'shoulder_er_force'
                }
                metric_name = 'Force'
                unit = 'N'
            else:  # Hip
                metric_cols = {
                    'Adduction': 'hip_adduction_force',
                    'Abduction': 'hip_abduction_force'
                }
                metric_name = 'Force'
                unit = 'N'

            view_tabs = st.tabs(["👥 Group View", "🏃 Individual View"])

            with view_tabs[0]:
                # Check if we have the required columns
                available_cols = {k: v for k, v in metric_cols.items() if v in filtered_df.columns}

                if available_cols:
                    # Vertical orientation toggle
                    vertical = st.checkbox("Vertical Layout", value=True, key="quadrant_vertical")

                    fig = create_stacked_quadrant_chart(
                        filtered_df,
                        metric_cols,
                        metric_name,
                        unit,
                        f'{selected_quadrant_test} - Group Profile',
                        vertical
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="quadrant_group_chart")

                    # Summary table with flagging
                    st.markdown("### Summary Table")
                    summary_table = create_quadrant_summary_table(filtered_df, metric_cols, unit)
                    if not summary_table.empty:
                        st.dataframe(summary_table, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No {selected_quadrant_test} data found. ForceFrame quadrant test columns not detected in the data.")
                    st.markdown("""
                    **Expected data structure for quadrant tests:**
                    - Trunk: supine_force, prone_force, lateral_left_force, lateral_right_force
                    - Neck: neck_flexion_force, neck_extension_force, neck_left_force, neck_right_force
                    - Shoulder: shoulder_ir_force, shoulder_er_force
                    - Hip: hip_adduction_force, hip_abduction_force

                    *Note: Column names may vary based on ForceFrame export settings.*
                    """)

            with view_tabs[1]:
                athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []

                if athletes:
                    selected_athlete = st.selectbox(
                        "Select Athlete:",
                        options=athletes,
                        key="quadrant_athlete_select"
                    )

                    if selected_athlete:
                        # Get all data for this athlete over time
                        fig = create_stacked_individual_trend_chart(
                            forceframe_df,
                            selected_athlete,
                            metric_cols,
                            metric_name,
                            unit,
                            f'{selected_athlete} - {selected_quadrant_test} Progression'
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key="quadrant_ind_chart")
                        else:
                            st.info("Not enough data points to show progression.")
                else:
                    st.info("No athletes found in filtered data.")
        else:
            st.warning("No ForceFrame data available for quadrant tests.")

    # =====================
    # Strength RM Tab (Manual Entry)
    # =====================
    with test_tabs[6]:
        st.markdown("### Strength RM (Manual Entry)")
        st.info("Strength RM data comes from manual entry. This section will display data once entered through the Data Entry tab.")

        # Placeholder for manual entry integration
        st.markdown("""
        **Available Exercises:**
        - Back Squat
        - Bench Press
        - Deadlift
        - Chin-Up
        - And more...

        **Metrics:**
        - Absolute Strength (Kg)
        - Relative Strength (Kg/BM)
        """)

    # =====================
    # DynaMo Tab (Grip Strength)
    # =====================
    with test_tabs[7]:
        st.markdown("### DynaMo (Grip Strength)")

        # Load DynaMo data
        dynamo_df = pd.DataFrame()

        # Try to load from various sources
        dynamo_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dynamo_allsports_with_athletes.csv'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'vald-data', 'data', 'dynamo_allsports_with_athletes.csv'),
        ]

        for path in dynamo_paths:
            if os.path.exists(path):
                try:
                    dynamo_df = pd.read_csv(path)
                    if 'recordedDateUtc' in dynamo_df.columns:
                        dynamo_df['recordedDateUtc'] = pd.to_datetime(dynamo_df['recordedDateUtc'])
                    break
                except Exception as e:
                    pass

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
            # Filters
            filtered_df, sport, gender = render_filters(dynamo_df, "dynamo")

            if filtered_df.empty:
                st.warning("No DynaMo data for selected filters.")
            else:
                # Find grip force columns
                grip_cols = [c for c in filtered_df.columns if 'GRIP' in c.upper() or 'FORCE' in c.upper() or 'PEAK' in c.upper()]

                # Show available metrics
                st.markdown(f"**{len(filtered_df)} DynaMo tests found**")

                if grip_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        metric_col = st.selectbox(
                            "Select Metric:",
                            options=grip_cols[:10],
                            key="dynamo_metric"
                        )

                    with col2:
                        benchmark = st.number_input(
                            "Benchmark (N):",
                            min_value=0.0,
                            value=400.0,
                            step=10.0,
                            key="dynamo_benchmark"
                        )

                    if metric_col and metric_col in filtered_df.columns:
                        view_tabs = st.tabs(["👥 Group View", "🏃 Individual View"])

                        with view_tabs[0]:
                            fig = create_ranked_bar_chart(
                                filtered_df,
                                metric_col,
                                metric_col.replace('_', ' ').title(),
                                'N',
                                benchmark,
                                f'DynaMo - {metric_col}'
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key="dynamo_group_bar")

                        with view_tabs[1]:
                            athletes = sorted(filtered_df['Name'].dropna().unique()) if 'Name' in filtered_df.columns else []
                            if athletes:
                                selected_athletes = st.multiselect(
                                    "Select Athletes:",
                                    options=athletes,
                                    default=[athletes[0]] if athletes else [],
                                    key="dynamo_athlete_select"
                                )

                                show_squad = st.checkbox("Show Squad Average", value=True, key="dynamo_show_squad")

                                if selected_athletes:
                                    fig = create_individual_line_chart(
                                        filtered_df,
                                        selected_athletes,
                                        metric_col,
                                        metric_col.replace('_', ' ').title(),
                                        'N',
                                        show_squad,
                                        f'DynaMo - Individual Trends'
                                    )
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True, key="dynamo_ind_line")
                else:
                    st.info("DynaMo data loaded but no grip force metrics found. Check column names.")

    # =====================
    # Balance Tab (QSB/SLSB for Shooting)
    # =====================
    with test_tabs[8]:
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

                view_tabs = st.tabs(["👥 Group View", "🏃 Individual View"])

                with view_tabs[0]:
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

                with view_tabs[1]:
                    athletes = sorted(display_df['Name'].dropna().unique()) if 'Name' in display_df.columns else []

                    if athletes:
                        selected_athletes = st.multiselect(
                            "Select Athletes:",
                            options=athletes,
                            default=[athletes[0]] if athletes else [],
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
