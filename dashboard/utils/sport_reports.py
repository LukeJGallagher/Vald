"""
Sport-Specific Group & Individual Reports Module

Provides generic visualization templates for sport reports with:
- Group reports showing team performance with benchmark zones
- Individual athlete reports with trend analysis
- Shaded benchmark zones on all graphs (green/yellow/red)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Team Saudi colors
TEAL_PRIMARY = '#007167'
GOLD_ACCENT = '#a08e66'
TEAL_DARK = '#005a51'

# Benchmark zone colors (with transparency)
ZONE_COLORS = {
    'excellent': 'rgba(34, 139, 34, 0.15)',   # Forest green
    'good': 'rgba(255, 193, 7, 0.15)',        # Amber/yellow
    'average': 'rgba(220, 53, 69, 0.15)',     # Red
    'border_excellent': 'rgba(34, 139, 34, 0.5)',
    'border_good': 'rgba(255, 193, 7, 0.5)',
    'border_average': 'rgba(220, 53, 69, 0.5)',
}

# Default benchmarks if sport-specific not available
DEFAULT_BENCHMARKS = {
    'cmj_height': {'excellent': 40, 'good': 35, 'average': 30},
    'peak_power': {'excellent': 55, 'good': 48, 'average': 40},
    'rsi': {'excellent': 2.0, 'good': 1.5, 'average': 1.2},
    'peak_force': {'excellent': 30, 'good': 25, 'average': 20},
    'asymmetry': {'excellent': 5, 'good': 10, 'average': 15},
}

# Metric column mappings
METRIC_COLUMNS = {
    'cmj_height': ['Jump Height (Flight Time)_Trial', 'Jump Height (Imp-Mom)_Trial'],
    'peak_power': ['Peak Power / BM_Trial', 'Peak Power_Trial'],
    'relative_power': ['Peak Power / BM_Trial'],
    'peak_force': ['Peak Force / BM_Trial', 'Peak Vertical Force / BM_Trial'],
    'rsi': ['RSI-modified_Trial', 'RSI (Flight/Contact Time)_Trial', 'RSI-modified (Imp-Mom)_Trial'],
    'contraction_time': ['Contraction Time_Trial'],
    'countermovement_depth': ['Countermovement Depth_Trial'],
}


def get_metric_column(df: pd.DataFrame, metric_key: str) -> Optional[str]:
    """Find the actual column name for a metric in the dataframe."""
    possible_cols = METRIC_COLUMNS.get(metric_key, [metric_key])
    for col in possible_cols:
        if col in df.columns:
            return col
    return None


def get_sport_benchmarks(sport: str, config: Dict = None) -> Dict:
    """
    Get benchmarks for a specific sport from config.
    Falls back to defaults if sport not found.
    """
    if config and sport in config:
        return config[sport]

    # Sport-specific defaults
    sport_defaults = {
        'Swimming': {
            'cmj_height': {'excellent': 38, 'good': 34, 'average': 30},
            'peak_power': {'excellent': 55, 'good': 48, 'average': 42},
            'rsi': {'excellent': 1.8, 'good': 1.5, 'average': 1.2},
            'peak_force': {'excellent': 26, 'good': 23, 'average': 20},
            'asymmetry': {'excellent': 5, 'good': 8, 'average': 12},
        },
        'Athletics - Throws': {
            'cmj_height': {'excellent': 50, 'good': 45, 'average': 40},
            'peak_power': {'excellent': 65, 'good': 55, 'average': 48},
            'rsi': {'excellent': 2.2, 'good': 1.8, 'average': 1.5},
            'peak_force': {'excellent': 35, 'good': 30, 'average': 25},
            'asymmetry': {'excellent': 8, 'good': 12, 'average': 15},
        },
        'Rowing': {
            'cmj_height': {'excellent': 45, 'good': 40, 'average': 35},
            'peak_power': {'excellent': 60, 'good': 52, 'average': 45},
            'rsi': {'excellent': 1.6, 'good': 1.3, 'average': 1.0},
            'peak_force': {'excellent': 32, 'good': 28, 'average': 24},
            'asymmetry': {'excellent': 4, 'good': 6, 'average': 10},
        },
    }

    return sport_defaults.get(sport, DEFAULT_BENCHMARKS)


def add_benchmark_zones(fig: go.Figure, benchmarks: Dict,
                        orientation: str = 'h',
                        metric_type: str = 'cmj_height',
                        max_val: float = None) -> go.Figure:
    """
    Add shaded benchmark zones to a plotly figure.

    Args:
        fig: Plotly figure to modify
        benchmarks: Dict with 'excellent', 'good', 'average' thresholds
        orientation: 'h' for horizontal bars, 'v' for vertical
        metric_type: Key to look up in benchmarks dict
        max_val: Maximum value for the zone (auto-calculated if None)
    """
    bench = benchmarks.get(metric_type, DEFAULT_BENCHMARKS.get(metric_type, {}))

    if not bench:
        return fig

    excellent = bench.get('excellent', 0)
    good = bench.get('good', 0)
    average = bench.get('average', 0)

    # For asymmetry, lower is better (reverse the zones)
    if metric_type == 'asymmetry':
        if max_val is None:
            max_val = 25
        # Excellent zone: 0 to excellent threshold
        fig.add_vrect(x0=0, x1=excellent,
                      fillcolor=ZONE_COLORS['excellent'],
                      layer="below", line_width=0)
        # Good zone: excellent to good threshold
        fig.add_vrect(x0=excellent, x1=good,
                      fillcolor=ZONE_COLORS['good'],
                      layer="below", line_width=0)
        # Average/risk zone: good to max
        fig.add_vrect(x0=good, x1=max_val,
                      fillcolor=ZONE_COLORS['average'],
                      layer="below", line_width=0)
    else:
        # Normal metrics (higher is better)
        if max_val is None:
            max_val = excellent * 1.3

        if orientation == 'h':
            # Horizontal bar chart - zones on x-axis
            # Average/risk zone: 0 to average
            fig.add_vrect(x0=0, x1=average,
                          fillcolor=ZONE_COLORS['average'],
                          layer="below", line_width=0)
            # Good zone: average to good
            fig.add_vrect(x0=average, x1=good,
                          fillcolor=ZONE_COLORS['good'],
                          layer="below", line_width=0)
            # Excellent zone: good to max
            fig.add_vrect(x0=good, x1=max_val,
                          fillcolor=ZONE_COLORS['excellent'],
                          layer="below", line_width=0)
        else:
            # Vertical orientation - zones on y-axis
            fig.add_hrect(y0=0, y1=average,
                          fillcolor=ZONE_COLORS['average'],
                          layer="below", line_width=0)
            fig.add_hrect(y0=average, y1=good,
                          fillcolor=ZONE_COLORS['good'],
                          layer="below", line_width=0)
            fig.add_hrect(y0=good, y1=max_val,
                          fillcolor=ZONE_COLORS['excellent'],
                          layer="below", line_width=0)

    return fig


def create_benchmark_bar_chart(df: pd.DataFrame,
                                metric_col: str,
                                name_col: str,
                                benchmarks: Dict,
                                title: str,
                                metric_type: str = 'cmj_height',
                                unit: str = '') -> go.Figure:
    """
    Create a horizontal bar chart with benchmark zones.

    Args:
        df: DataFrame with athlete data
        metric_col: Column name for the metric values
        name_col: Column name for athlete names
        benchmarks: Benchmark thresholds dict
        title: Chart title
        metric_type: Key for benchmark lookup
        unit: Unit label for values
    """
    if metric_col not in df.columns:
        return None

    # Get latest value per athlete
    plot_data = df.groupby(name_col)[metric_col].last().reset_index()
    plot_data = plot_data.dropna(subset=[metric_col])
    plot_data = plot_data.sort_values(metric_col, ascending=True)

    if plot_data.empty:
        return None

    # Create horizontal bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=plot_data[metric_col],
        y=plot_data[name_col],
        orientation='h',
        marker_color=TEAL_PRIMARY,
        text=[f"{v:.1f}{unit}" for v in plot_data[metric_col]],
        textposition='outside',
        hovertemplate="%{y}: %{x:.1f}" + unit + "<extra></extra>"
    ))

    # Add benchmark zones
    max_val = plot_data[metric_col].max() * 1.2
    fig = add_benchmark_zones(fig, benchmarks, 'h', metric_type, max_val)

    # Styling
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=f"{title} ({unit})" if unit else title,
        yaxis_title="",
        height=max(300, len(plot_data) * 30 + 100),
        margin=dict(l=150, r=50, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )

    fig.update_xaxes(range=[0, max_val], gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)')

    return fig


def create_trend_chart(df: pd.DataFrame,
                       metric_col: str,
                       date_col: str,
                       benchmarks: Dict,
                       title: str,
                       metric_type: str = 'cmj_height',
                       unit: str = '',
                       athlete_name: str = None) -> go.Figure:
    """
    Create a line chart showing metric trend over time with benchmark zones.

    Args:
        df: DataFrame with test data
        metric_col: Column name for the metric values
        date_col: Column name for dates
        benchmarks: Benchmark thresholds dict
        title: Chart title
        metric_type: Key for benchmark lookup
        unit: Unit label for values
        athlete_name: Optional athlete name for title
    """
    if metric_col not in df.columns or date_col not in df.columns:
        return None

    plot_data = df[[date_col, metric_col]].dropna()
    plot_data = plot_data.sort_values(date_col)

    if plot_data.empty:
        return None

    # Create line chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_data[date_col],
        y=plot_data[metric_col],
        mode='lines+markers',
        line=dict(color=TEAL_PRIMARY, width=2),
        marker=dict(size=8, color=TEAL_PRIMARY),
        name=title,
        hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}" + unit + "<extra></extra>"
    ))

    # Add benchmark zones (vertical orientation for line charts)
    min_val = plot_data[metric_col].min() * 0.9
    max_val = plot_data[metric_col].max() * 1.1
    fig = add_benchmark_zones(fig, benchmarks, 'v', metric_type, max_val)

    # Styling
    chart_title = f"{title} - {athlete_name}" if athlete_name else title
    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=14)),
        xaxis_title="Date",
        yaxis_title=f"{title} ({unit})" if unit else title,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )

    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(range=[min_val, max_val], gridcolor='rgba(128,128,128,0.2)')

    return fig


def create_group_report(df: pd.DataFrame,
                        sport: str,
                        forceframe_df: pd.DataFrame = None,
                        nordbord_df: pd.DataFrame = None,
                        config: Dict = None):
    """
    Create a group report for a sport showing team performance.

    Layout:
    - Lower Body Strength & Power (IMTP, CMJ, Repeat Hop)
    - Upper Body Strength & Power (if data available)
    - Shoulder Health (ForceFrame)
    - Hip Health (ForceFrame)
    """
    benchmarks = get_sport_benchmarks(sport, config)

    st.markdown(f"## {sport} Group Report")
    st.markdown("---")

    # Filter data for the sport
    sport_df = df.copy()
    if 'athlete_sport' in sport_df.columns:
        # Fuzzy match for sport name
        sport_mask = sport_df['athlete_sport'].str.contains(sport.split()[0], case=False, na=False)
        if sport_mask.any():
            sport_df = sport_df[sport_mask]

    if sport_df.empty:
        st.warning(f"No data available for {sport}")
        return

    # =========================================================================
    # SECTION 1: Lower Body Strength & Power
    # =========================================================================
    st.markdown("### Lower Body Strength & Power")

    col1, col2, col3 = st.columns(3)

    # IMTP - Relative Peak Force
    with col1:
        metric_col = get_metric_column(sport_df, 'peak_force')
        if metric_col:
            # Filter for IMTP/ISOT tests
            imtp_df = sport_df[sport_df['testType'].str.contains('IMTP|ISOT|Isometric', case=False, na=False)]
            if not imtp_df.empty:
                fig = create_benchmark_bar_chart(
                    imtp_df, metric_col, 'Name', benchmarks,
                    "IMTP - Relative Peak Force", 'peak_force', 'N/kg'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No IMTP data available")
        else:
            st.info("IMTP metric not found")

    # CMJ - Jump Height
    with col2:
        metric_col = get_metric_column(sport_df, 'cmj_height')
        if metric_col:
            cmj_df = sport_df[sport_df['testType'].str.contains('CMJ|Counter', case=False, na=False)]
            if not cmj_df.empty:
                fig = create_benchmark_bar_chart(
                    cmj_df, metric_col, 'Name', benchmarks,
                    "CMJ - Jump Height", 'cmj_height', 'cm'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No CMJ data available")
        else:
            st.info("CMJ metric not found")

    # CMJ - Relative Peak Power
    with col3:
        metric_col = get_metric_column(sport_df, 'relative_power')
        if metric_col:
            cmj_df = sport_df[sport_df['testType'].str.contains('CMJ|Counter', case=False, na=False)]
            if not cmj_df.empty:
                fig = create_benchmark_bar_chart(
                    cmj_df, metric_col, 'Name', benchmarks,
                    "CMJ - Relative Peak Power", 'peak_power', 'W/kg'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No CMJ power data available")
        else:
            st.info("Power metric not found")

    # Second row - Repeat Hop
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_col = get_metric_column(sport_df, 'rsi')
        if metric_col:
            hop_df = sport_df[sport_df['testType'].str.contains('Hop|Repeat|DJ|Drop', case=False, na=False)]
            if not hop_df.empty:
                fig = create_benchmark_bar_chart(
                    hop_df, metric_col, 'Name', benchmarks,
                    "Repeat Hop - RSI", 'rsi', ''
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Repeat Hop data available")

    st.markdown("---")

    # =========================================================================
    # SECTION 2: Upper Body Strength & Power (if data available)
    # =========================================================================
    st.markdown("### Upper Body Strength & Power")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Bench Press
        bench_df = sport_df[sport_df['testType'].str.contains('Bench|Press', case=False, na=False)]
        if not bench_df.empty:
            metric_col = get_metric_column(bench_df, 'peak_force')
            if metric_col:
                fig = create_benchmark_bar_chart(
                    bench_df, metric_col, 'Name', benchmarks,
                    "Bench Press - Peak Force", 'peak_force', 'N/kg'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Bench Press data not available")

    with col2:
        # Pull Up
        pullup_df = sport_df[sport_df['testType'].str.contains('Pull|Chin', case=False, na=False)]
        if not pullup_df.empty:
            metric_col = get_metric_column(pullup_df, 'peak_force')
            if metric_col:
                fig = create_benchmark_bar_chart(
                    pullup_df, metric_col, 'Name', benchmarks,
                    "Pull Up - Peak Force", 'peak_force', 'N/kg'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pull Up data not available")

    with col3:
        # Plyo Push Up
        plyo_df = sport_df[sport_df['testType'].str.contains('Plyo|Plyometric', case=False, na=False)]
        if not plyo_df.empty:
            metric_col = get_metric_column(plyo_df, 'peak_power')
            if metric_col:
                fig = create_benchmark_bar_chart(
                    plyo_df, metric_col, 'Name', benchmarks,
                    "Plyo Push Up - Peak Power", 'peak_power', 'W/kg'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Plyo Push Up data not available")

    st.markdown("---")

    # =========================================================================
    # SECTION 3: Shoulder Health (ForceFrame)
    # =========================================================================
    st.markdown("### Shoulder Health")

    col1, col2 = st.columns(2)

    with col1:
        if forceframe_df is not None and not forceframe_df.empty:
            # Filter for shoulder tests
            shoulder_df = forceframe_df[
                forceframe_df['testTypeName'].str.contains('Shoulder|IR|ER', case=False, na=False)
            ] if 'testTypeName' in forceframe_df.columns else pd.DataFrame()

            if not shoulder_df.empty and 'Name' in shoulder_df.columns:
                # Calculate IR/ER ratio if possible
                st.markdown("**Shoulder IR/ER**")
                st.info("ForceFrame shoulder IR/ER data - visualization pending")
            else:
                st.info("Shoulder IR/ER data not available")
        else:
            st.info("ForceFrame data not available")

    with col2:
        if forceframe_df is not None and not forceframe_df.empty:
            st.markdown("**Shoulder ASH**")
            st.info("ForceFrame shoulder ASH data - visualization pending")
        else:
            st.info("ForceFrame data not available")

    st.markdown("---")

    # =========================================================================
    # SECTION 4: Hip Health (ForceFrame)
    # =========================================================================
    st.markdown("### Hip Health")

    if forceframe_df is not None and not forceframe_df.empty:
        # Filter for hip tests
        hip_df = forceframe_df[
            forceframe_df['testTypeName'].str.contains('Hip|ABD|ADD|Adduct|Abduct', case=False, na=False)
        ] if 'testTypeName' in forceframe_df.columns else pd.DataFrame()

        if not hip_df.empty:
            st.markdown("**Hip IR/ER - ABD/ADD**")
            st.info("ForceFrame hip ABD/ADD data - visualization pending")
        else:
            st.info("Hip ABD/ADD data not available")
    else:
        st.info("ForceFrame data not available")


def create_individual_report(df: pd.DataFrame,
                             athlete_name: str,
                             sport: str,
                             forceframe_df: pd.DataFrame = None,
                             nordbord_df: pd.DataFrame = None,
                             config: Dict = None):
    """
    Create an individual athlete report with trends over time.

    Layout:
    - Latest Performance Summary Cards
    - Trend Charts for each test type
    """
    benchmarks = get_sport_benchmarks(sport, config)

    st.markdown(f"## {athlete_name} - Individual Report")
    st.markdown(f"**Sport:** {sport}")
    st.markdown("---")

    # Filter data for athlete
    athlete_df = df[df['Name'] == athlete_name].copy()

    if athlete_df.empty:
        st.warning(f"No data available for {athlete_name}")
        return

    # Determine date column
    date_col = None
    for col in ['recordedDateUtc', 'testDateUtc', 'modifiedDateUtc']:
        if col in athlete_df.columns:
            date_col = col
            break

    if date_col:
        athlete_df[date_col] = pd.to_datetime(athlete_df[date_col], errors='coerce')
        athlete_df = athlete_df.sort_values(date_col)

    # =========================================================================
    # SUMMARY CARDS - Latest Performance
    # =========================================================================
    st.markdown("### Latest Performance Summary")

    latest = athlete_df.iloc[-1] if not athlete_df.empty else None

    if latest is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_col = get_metric_column(athlete_df, 'cmj_height')
            if metric_col and pd.notna(latest.get(metric_col)):
                st.metric("Jump Height", f"{latest[metric_col]:.1f} cm")

        with col2:
            metric_col = get_metric_column(athlete_df, 'peak_power')
            if metric_col and pd.notna(latest.get(metric_col)):
                st.metric("Relative Power", f"{latest[metric_col]:.1f} W/kg")

        with col3:
            metric_col = get_metric_column(athlete_df, 'peak_force')
            if metric_col and pd.notna(latest.get(metric_col)):
                st.metric("Relative Force", f"{latest[metric_col]:.1f} N/kg")

        with col4:
            metric_col = get_metric_column(athlete_df, 'rsi')
            if metric_col and pd.notna(latest.get(metric_col)):
                st.metric("RSI", f"{latest[metric_col]:.2f}")

    st.markdown("---")

    # =========================================================================
    # TREND CHARTS
    # =========================================================================
    st.markdown("### Trends Over Time")

    if date_col is None:
        st.warning("No date column found for trend analysis")
        return

    # CMJ Jump Height Trend
    col1, col2 = st.columns(2)

    with col1:
        metric_col = get_metric_column(athlete_df, 'cmj_height')
        if metric_col:
            cmj_df = athlete_df[athlete_df['testType'].str.contains('CMJ|Counter', case=False, na=False)]
            if not cmj_df.empty:
                fig = create_trend_chart(
                    cmj_df, metric_col, date_col, benchmarks,
                    "Jump Height", 'cmj_height', 'cm', athlete_name
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    with col2:
        metric_col = get_metric_column(athlete_df, 'peak_power')
        if metric_col:
            cmj_df = athlete_df[athlete_df['testType'].str.contains('CMJ|Counter', case=False, na=False)]
            if not cmj_df.empty:
                fig = create_trend_chart(
                    cmj_df, metric_col, date_col, benchmarks,
                    "Relative Peak Power", 'peak_power', 'W/kg', athlete_name
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    # RSI and Force trends
    col1, col2 = st.columns(2)

    with col1:
        metric_col = get_metric_column(athlete_df, 'rsi')
        if metric_col:
            hop_df = athlete_df[athlete_df['testType'].str.contains('Hop|DJ|Drop|CMJ', case=False, na=False)]
            if not hop_df.empty:
                fig = create_trend_chart(
                    hop_df, metric_col, date_col, benchmarks,
                    "RSI Modified", 'rsi', '', athlete_name
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    with col2:
        metric_col = get_metric_column(athlete_df, 'peak_force')
        if metric_col:
            imtp_df = athlete_df[athlete_df['testType'].str.contains('IMTP|ISOT|Isometric', case=False, na=False)]
            if not imtp_df.empty:
                fig = create_trend_chart(
                    imtp_df, metric_col, date_col, benchmarks,
                    "IMTP Relative Peak Force", 'peak_force', 'N/kg', athlete_name
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)


def render_benchmark_legend():
    """Render a legend explaining the benchmark zones."""
    st.markdown("""
    <div style="display: flex; gap: 20px; padding: 10px; background: #f8f9fa; border-radius: 5px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 5px;">
            <div style="width: 20px; height: 20px; background: rgba(34, 139, 34, 0.3); border-radius: 3px;"></div>
            <span style="font-size: 12px;">Excellent</span>
        </div>
        <div style="display: flex; align-items: center; gap: 5px;">
            <div style="width: 20px; height: 20px; background: rgba(255, 193, 7, 0.3); border-radius: 3px;"></div>
            <span style="font-size: 12px;">Good</span>
        </div>
        <div style="display: flex; align-items: center; gap: 5px;">
            <div style="width: 20px; height: 20px; background: rgba(220, 53, 69, 0.3); border-radius: 3px;"></div>
            <span style="font-size: 12px;">Needs Improvement</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
