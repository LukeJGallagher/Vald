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

# Benchmark zone colors (with transparency) - teal palette
ZONE_COLORS = {
    'excellent': 'rgba(0, 113, 103, 0.20)',      # Teal (excellent)
    'good': 'rgba(0, 150, 136, 0.15)',           # Light teal (good)
    'average': 'rgba(120, 144, 156, 0.15)',      # Gray-blue (needs work)
    'border_excellent': 'rgba(0, 113, 103, 0.5)',
    'border_good': 'rgba(0, 150, 136, 0.5)',
    'border_average': 'rgba(120, 144, 156, 0.5)',
}

# Secondary color for bilateral comparisons (coral instead of gold)
CORAL_ACCENT = '#FF6B6B'

# Default benchmarks if sport-specific not available
DEFAULT_BENCHMARKS = {
    'cmj_height': {'excellent': 40, 'good': 35, 'average': 30},
    'peak_power': {'excellent': 55, 'good': 48, 'average': 40},
    'rsi': {'excellent': 2.0, 'good': 1.5, 'average': 1.2},
    'peak_force': {'excellent': 30, 'good': 25, 'average': 20},
    'asymmetry': {'excellent': 5, 'good': 10, 'average': 15},
    'nordbord_force': {'excellent': 400, 'good': 337, 'average': 280},  # N - 337N is injury risk threshold
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
    # NordBord columns - actual CSV column names: leftMaxForce, rightMaxForce
    'nordbord_left': ['leftMaxForce', 'maxForceLeftN', 'leftMax', 'left_max_force'],
    'nordbord_right': ['rightMaxForce', 'maxForceRightN', 'rightMax', 'right_max_force'],
    'nordbord_avg': ['avgMaxForce', 'average_max_force'],
}


def get_nordbord_force_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Get the left and right force column names for NordBord data."""
    left_options = ['leftMaxForce', 'maxForceLeftN', 'leftMax', 'left_max_force']
    right_options = ['rightMaxForce', 'maxForceRightN', 'rightMax', 'right_max_force']

    left_col = None
    right_col = None

    for col in left_options:
        if col in df.columns:
            left_col = col
            break

    for col in right_options:
        if col in df.columns:
            right_col = col
            break

    return left_col, right_col


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
            'nordbord_force': {'excellent': 380, 'good': 337, 'average': 280},
        },
        'Athletics - Throws': {
            'cmj_height': {'excellent': 50, 'good': 45, 'average': 40},
            'peak_power': {'excellent': 65, 'good': 55, 'average': 48},
            'rsi': {'excellent': 2.2, 'good': 1.8, 'average': 1.5},
            'peak_force': {'excellent': 35, 'good': 30, 'average': 25},
            'asymmetry': {'excellent': 8, 'good': 12, 'average': 15},
            'nordbord_force': {'excellent': 450, 'good': 400, 'average': 337},  # Higher for power athletes
        },
        'Rowing': {
            'cmj_height': {'excellent': 45, 'good': 40, 'average': 35},
            'peak_power': {'excellent': 60, 'good': 52, 'average': 45},
            'rsi': {'excellent': 1.6, 'good': 1.3, 'average': 1.0},
            'peak_force': {'excellent': 32, 'good': 28, 'average': 24},
            'asymmetry': {'excellent': 4, 'good': 6, 'average': 10},
            'nordbord_force': {'excellent': 400, 'good': 350, 'average': 300},
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

    # Second row - Repeat Hop and NordBord
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

    # NordBord - Hamstring Strength (in Lower Body section)
    with col2:
        if nordbord_df is not None and not nordbord_df.empty:
            # Filter NordBord data for sport if possible
            nb_df = nordbord_df.copy()
            if 'athlete_sport' in nb_df.columns:
                sport_mask = nb_df['athlete_sport'].str.contains(sport.split()[0], case=False, na=False)
                if sport_mask.any():
                    nb_df = nb_df[sport_mask]

            # Get the force columns using helper function
            left_col, right_col = get_nordbord_force_columns(nb_df)

            if left_col and right_col and 'Name' in nb_df.columns:
                # Calculate average of left/right
                nb_df['avg_hamstring_force'] = (nb_df[left_col] + nb_df[right_col]) / 2

                fig = create_benchmark_bar_chart(
                    nb_df, 'avg_hamstring_force', 'Name', benchmarks,
                    "NordBord - Hamstring Strength", 'nordbord_force', 'N'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("NordBord data not available")
        else:
            st.info("NordBord data not available")

    # NordBord Asymmetry
    with col3:
        if nordbord_df is not None and not nordbord_df.empty:
            nb_df = nordbord_df.copy()
            if 'athlete_sport' in nb_df.columns:
                sport_mask = nb_df['athlete_sport'].str.contains(sport.split()[0], case=False, na=False)
                if sport_mask.any():
                    nb_df = nb_df[sport_mask]

            # Calculate asymmetry if both sides available
            left_col, right_col = get_nordbord_force_columns(nb_df)
            if left_col and right_col:
                nb_df['hamstring_asymmetry'] = abs(
                    (nb_df[left_col] - nb_df[right_col]) /
                    ((nb_df[left_col] + nb_df[right_col]) / 2) * 100
                )
                if 'Name' in nb_df.columns:
                    fig = create_benchmark_bar_chart(
                        nb_df, 'hamstring_asymmetry', 'Name', benchmarks,
                        "NordBord - Asymmetry", 'asymmetry', '%'
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("NordBord asymmetry data not available")
            else:
                st.info("NordBord asymmetry data not available")
        else:
            st.info("NordBord asymmetry data not available")

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
            # Filter for shoulder tests (tests starting with 'Shoulder')
            shoulder_df = forceframe_df[
                forceframe_df['testTypeName'].str.contains('^Shoulder', case=False, na=False, regex=True)
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
        # Filter for hip tests (tests starting with 'Hip')
        hip_df = forceframe_df[
            forceframe_df['testTypeName'].str.contains('^Hip', case=False, na=False, regex=True)
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

    # NordBord trend (if data available)
    if nordbord_df is not None and not nordbord_df.empty:
        # Filter NordBord for this athlete
        athlete_nb = nordbord_df[nordbord_df['Name'] == athlete_name].copy() if 'Name' in nordbord_df.columns else pd.DataFrame()

        if not athlete_nb.empty:
            st.markdown("### Hamstring Strength (NordBord)")

            # Determine date column for NordBord
            nb_date_col = None
            for col in ['recordedDateUtc', 'testDateUtc', 'modifiedDateUtc']:
                if col in athlete_nb.columns:
                    nb_date_col = col
                    break

            if nb_date_col:
                athlete_nb[nb_date_col] = pd.to_datetime(athlete_nb[nb_date_col], errors='coerce')
                athlete_nb = athlete_nb.sort_values(nb_date_col)

                col1, col2 = st.columns(2)

                left_col, right_col = get_nordbord_force_columns(athlete_nb)

                # Left hamstring trend
                with col1:
                    if left_col:
                        fig = create_trend_chart(
                            athlete_nb, left_col, nb_date_col, benchmarks,
                            "Left Hamstring", 'nordbord_force', 'N', athlete_name
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                # Right hamstring trend
                with col2:
                    if right_col:
                        fig = create_trend_chart(
                            athlete_nb, right_col, nb_date_col, benchmarks,
                            "Right Hamstring", 'nordbord_force', 'N', athlete_name
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)


def create_group_report_v2(df: pd.DataFrame,
                           sport: str,
                           forceframe_df: pd.DataFrame = None,
                           nordbord_df: pd.DataFrame = None,
                           config: Dict = None):
    """
    Alternative Group Report Layout (V2) - Summary table with RAG status.

    This version focuses on:
    - Summary table showing all athletes with status indicators
    - Compact metric cards per athlete
    - RAG (Red/Amber/Green) status indicators
    - Sortable/filterable data view
    """
    benchmarks = get_sport_benchmarks(sport, config)

    st.markdown(f"## {sport} Group Report - Summary View")
    st.markdown("*Alternative layout for staff feedback*")
    st.markdown("---")

    # Filter data for the sport
    sport_df = df.copy()
    if 'athlete_sport' in sport_df.columns:
        sport_mask = sport_df['athlete_sport'].str.contains(sport.split()[0], case=False, na=False)
        if sport_mask.any():
            sport_df = sport_df[sport_mask]

    if sport_df.empty:
        st.warning(f"No data available for {sport}")
        return

    # =========================================================================
    # SECTION 1: Summary Table with RAG Status
    # =========================================================================
    st.markdown("### 📊 Team Summary Table")

    # Get latest values per athlete
    athletes = sport_df['Name'].dropna().unique() if 'Name' in sport_df.columns else []

    if len(athletes) == 0:
        st.warning("No athletes found in data")
        return

    summary_data = []

    for athlete in athletes:
        athlete_data = sport_df[sport_df['Name'] == athlete]
        row = {'Athlete': athlete}

        # CMJ Jump Height
        cmj_col = get_metric_column(athlete_data, 'cmj_height')
        if cmj_col:
            cmj_df = athlete_data[athlete_data['testType'].str.contains('CMJ|Counter', case=False, na=False)]
            if not cmj_df.empty:
                val = cmj_df[cmj_col].dropna().iloc[-1] if not cmj_df[cmj_col].dropna().empty else None
                if val is not None:
                    row['CMJ (cm)'] = round(val, 1)
                    row['CMJ_status'] = _get_rag_status(val, benchmarks.get('cmj_height', {}))

        # Peak Power
        power_col = get_metric_column(athlete_data, 'relative_power')
        if power_col:
            cmj_df = athlete_data[athlete_data['testType'].str.contains('CMJ|Counter', case=False, na=False)]
            if not cmj_df.empty:
                val = cmj_df[power_col].dropna().iloc[-1] if not cmj_df[power_col].dropna().empty else None
                if val is not None:
                    row['Power (W/kg)'] = round(val, 1)
                    row['Power_status'] = _get_rag_status(val, benchmarks.get('peak_power', {}))

        # IMTP Peak Force
        force_col = get_metric_column(athlete_data, 'peak_force')
        if force_col:
            imtp_df = athlete_data[athlete_data['testType'].str.contains('IMTP|ISOT|Isometric', case=False, na=False)]
            if not imtp_df.empty:
                val = imtp_df[force_col].dropna().iloc[-1] if not imtp_df[force_col].dropna().empty else None
                if val is not None:
                    row['IMTP (N/kg)'] = round(val, 1)
                    row['IMTP_status'] = _get_rag_status(val, benchmarks.get('peak_force', {}))

        # RSI
        rsi_col = get_metric_column(athlete_data, 'rsi')
        if rsi_col:
            hop_df = athlete_data[athlete_data['testType'].str.contains('Hop|DJ|Drop|CMJ', case=False, na=False)]
            if not hop_df.empty:
                val = hop_df[rsi_col].dropna().iloc[-1] if not hop_df[rsi_col].dropna().empty else None
                if val is not None:
                    row['RSI'] = round(val, 2)
                    row['RSI_status'] = _get_rag_status(val, benchmarks.get('rsi', {}))

        # NordBord (if available)
        if nordbord_df is not None and not nordbord_df.empty and 'Name' in nordbord_df.columns:
            nb_athlete = nordbord_df[nordbord_df['Name'] == athlete]
            if not nb_athlete.empty:
                left_col, right_col = get_nordbord_force_columns(nb_athlete)
                if left_col and right_col:
                    left = nb_athlete[left_col].dropna().iloc[-1] if not nb_athlete[left_col].dropna().empty else None
                    right = nb_athlete[right_col].dropna().iloc[-1] if not nb_athlete[right_col].dropna().empty else None
                    if left is not None and right is not None:
                        avg = (left + right) / 2
                        asym = abs((left - right) / avg * 100)
                        row['Hamstring (N)'] = round(avg, 0)
                        row['Ham_status'] = _get_rag_status(avg, benchmarks.get('nordbord_force', {}))
                        row['Ham Asym (%)'] = round(asym, 1)
                        row['HamAsym_status'] = _get_rag_status_inverse(asym, benchmarks.get('asymmetry', {}))

        summary_data.append(row)

    # Create DataFrame for display
    summary_df = pd.DataFrame(summary_data)

    # Display with colored status indicators
    if not summary_df.empty:
        # Create styled HTML table
        _display_rag_table(summary_df)

    st.markdown("---")

    # =========================================================================
    # SECTION 2: Metric Distribution Charts
    # =========================================================================
    st.markdown("### 📈 Metric Distributions")

    col1, col2 = st.columns(2)

    with col1:
        # CMJ Distribution
        cmj_col = get_metric_column(sport_df, 'cmj_height')
        if cmj_col:
            cmj_df = sport_df[sport_df['testType'].str.contains('CMJ|Counter', case=False, na=False)]
            if not cmj_df.empty:
                # Get latest per athlete
                latest_cmj = cmj_df.groupby('Name')[cmj_col].last().reset_index()
                latest_cmj = latest_cmj.dropna()

                if not latest_cmj.empty:
                    fig = go.Figure()

                    # Add histogram
                    fig.add_trace(go.Histogram(
                        x=latest_cmj[cmj_col],
                        nbinsx=10,
                        marker_color=TEAL_PRIMARY,
                        opacity=0.7,
                        name='Athletes'
                    ))

                    # Add benchmark lines
                    bench = benchmarks.get('cmj_height', {})
                    if bench:
                        for level, color in [('excellent', 'green'), ('good', 'orange'), ('average', 'red')]:
                            if level in bench:
                                fig.add_vline(x=bench[level], line_dash="dash",
                                             line_color=color, annotation_text=level.title())

                    fig.update_layout(
                        title="CMJ Jump Height Distribution",
                        xaxis_title="Jump Height (cm)",
                        yaxis_title="Number of Athletes",
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with col2:
        # NordBord Distribution
        if nordbord_df is not None and not nordbord_df.empty:
            left_col, right_col = get_nordbord_force_columns(nordbord_df)
            if left_col and right_col:
                # Filter for sport
                nb_sport = nordbord_df.copy()
                if 'athlete_sport' in nb_sport.columns:
                    sport_mask = nb_sport['athlete_sport'].str.contains(sport.split()[0], case=False, na=False)
                    if sport_mask.any():
                        nb_sport = nb_sport[sport_mask]

                if not nb_sport.empty and 'Name' in nb_sport.columns:
                    nb_sport['avg_force'] = (nb_sport[left_col] + nb_sport[right_col]) / 2
                    latest_nb = nb_sport.groupby('Name')['avg_force'].last().reset_index()
                    latest_nb = latest_nb.dropna()

                    if not latest_nb.empty:
                        fig = go.Figure()

                        fig.add_trace(go.Histogram(
                            x=latest_nb['avg_force'],
                            nbinsx=10,
                            marker_color=TEAL_PRIMARY,
                            opacity=0.7,
                            name='Athletes'
                        ))

                        # Add injury threshold line
                        fig.add_vline(x=337, line_dash="dash", line_color="red",
                                     annotation_text="Injury Risk (337N)")

                        bench = benchmarks.get('nordbord_force', {})
                        if bench and 'excellent' in bench:
                            fig.add_vline(x=bench['excellent'], line_dash="dash",
                                         line_color="green", annotation_text="Excellent")

                        fig.update_layout(
                            title="NordBord Hamstring Force Distribution",
                            xaxis_title="Average Force (N)",
                            yaxis_title="Number of Athletes",
                            height=300,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # =========================================================================
    # SECTION 3: Risk Flags
    # =========================================================================
    st.markdown("### ⚠️ Athletes Requiring Attention")

    risk_athletes = []

    for athlete in athletes:
        risks = []

        # Check NordBord < 337N
        if nordbord_df is not None and not nordbord_df.empty and 'Name' in nordbord_df.columns:
            nb_athlete = nordbord_df[nordbord_df['Name'] == athlete]
            if not nb_athlete.empty:
                left_col, right_col = get_nordbord_force_columns(nb_athlete)
                if left_col and right_col:
                    left = nb_athlete[left_col].dropna().iloc[-1] if not nb_athlete[left_col].dropna().empty else None
                    right = nb_athlete[right_col].dropna().iloc[-1] if not nb_athlete[right_col].dropna().empty else None
                    if left is not None and right is not None:
                        avg = (left + right) / 2
                        if avg < 337:
                            risks.append(f"🔴 Hamstring strength below injury threshold ({avg:.0f}N < 337N)")
                        asym = abs((left - right) / avg * 100)
                        if asym > 15:
                            risks.append(f"🟠 High hamstring asymmetry ({asym:.1f}%)")

        # Check poor CMJ
        athlete_data = sport_df[sport_df['Name'] == athlete]
        cmj_col = get_metric_column(athlete_data, 'cmj_height')
        if cmj_col:
            cmj_df = athlete_data[athlete_data['testType'].str.contains('CMJ|Counter', case=False, na=False)]
            if not cmj_df.empty:
                val = cmj_df[cmj_col].dropna().iloc[-1] if not cmj_df[cmj_col].dropna().empty else None
                avg_bench = benchmarks.get('cmj_height', {}).get('average', 30)
                if val is not None and val < avg_bench:
                    risks.append(f"🟠 CMJ below average ({val:.1f}cm < {avg_bench}cm)")

        if risks:
            risk_athletes.append({'Athlete': athlete, 'Risks': risks})

    if risk_athletes:
        for item in risk_athletes:
            with st.expander(f"**{item['Athlete']}** - {len(item['Risks'])} flag(s)"):
                for risk in item['Risks']:
                    st.markdown(f"- {risk}")
    else:
        st.success("✅ No athletes currently flagged for attention")


def _get_rag_status(value: float, benchmark: Dict) -> str:
    """Get RAG status (Red/Amber/Green) for a value based on benchmarks."""
    if not benchmark or value is None:
        return 'grey'

    excellent = benchmark.get('excellent', float('inf'))
    good = benchmark.get('good', float('inf'))
    average = benchmark.get('average', 0)

    if value >= good:
        return 'green'
    elif value >= average:
        return 'amber'
    else:
        return 'red'


def _get_rag_status_inverse(value: float, benchmark: Dict) -> str:
    """Get RAG status for metrics where lower is better (e.g., asymmetry)."""
    if not benchmark or value is None:
        return 'grey'

    excellent = benchmark.get('excellent', 0)
    good = benchmark.get('good', 10)

    if value <= excellent:
        return 'green'
    elif value <= good:
        return 'amber'
    else:
        return 'red'


def _display_rag_table(df: pd.DataFrame):
    """Display a DataFrame with RAG status indicators."""
    # Define status colors
    status_colors = {
        'green': '🟢',
        'amber': '🟠',
        'red': '🔴',
        'grey': '⚪'
    }

    # Build display columns
    display_cols = ['Athlete']
    metric_cols = ['CMJ (cm)', 'Power (W/kg)', 'IMTP (N/kg)', 'RSI', 'Hamstring (N)', 'Ham Asym (%)']

    # Create display data
    display_data = []
    for _, row in df.iterrows():
        display_row = {'Athlete': row.get('Athlete', '')}

        for col in metric_cols:
            if col in row:
                status_col = col.replace(' (cm)', '_status').replace(' (W/kg)', '_status').replace(' (N/kg)', '_status').replace(' (N)', '_status').replace(' (%)', '_status')
                status_col = status_col.replace('Hamstring', 'Ham').replace('Ham Asym', 'HamAsym')
                status = row.get(status_col, 'grey')
                indicator = status_colors.get(status, '⚪')
                display_row[col] = f"{indicator} {row[col]}"
            else:
                display_row[col] = "—"

        display_data.append(display_row)

    display_df = pd.DataFrame(display_data)

    # Display as dataframe
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # Legend
    st.markdown("""
    <div style="font-size: 12px; color: #666; margin-top: 5px;">
        🟢 Excellent/Good | 🟠 Average | 🔴 Below Average | ⚪ No Data
    </div>
    """, unsafe_allow_html=True)


def _create_radar_chart(athlete_data: Dict[str, float], benchmarks: Dict, athlete_name: str) -> go.Figure:
    """Create a radar/spider chart showing athlete's performance across metrics."""
    categories = list(athlete_data.keys())
    values = list(athlete_data.values())

    # Normalize values to 0-100 scale based on benchmarks
    normalized = []
    for cat, val in zip(categories, values):
        bench_key = cat.lower().replace(' ', '_').replace('/', '_')
        bench = benchmarks.get(bench_key, {})
        excellent = bench.get('excellent', 100)
        avg = bench.get('average', 0)
        if excellent != avg:
            norm = ((val - avg) / (excellent - avg)) * 100
            norm = max(0, min(100, norm))  # Clamp to 0-100
        else:
            norm = 50
        normalized.append(norm)

    # Close the polygon
    categories = categories + [categories[0]]
    normalized = normalized + [normalized[0]]

    fig = go.Figure()

    # Add benchmark zone (excellent)
    fig.add_trace(go.Scatterpolar(
        r=[100] * len(categories),
        theta=categories,
        fill='toself',
        fillcolor='rgba(34, 139, 34, 0.1)',
        line=dict(color='green', width=1, dash='dash'),
        name='Excellent Zone'
    ))

    # Add athlete data
    fig.add_trace(go.Scatterpolar(
        r=normalized,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 113, 103, 0.3)',
        line=dict(color=TEAL_PRIMARY, width=2),
        name=athlete_name
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)
        ),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig


def _create_lollipop_chart(df: pd.DataFrame, metric_col: str, name_col: str,
                           benchmarks: Dict, title: str, bench_key: str) -> go.Figure:
    """Create a lollipop chart (dot + line) for cleaner visualization."""
    if df.empty or metric_col not in df.columns or name_col not in df.columns:
        return None

    # Get latest value per athlete
    latest = df.groupby(name_col)[metric_col].last().reset_index()
    latest = latest.dropna().sort_values(metric_col, ascending=True)

    if latest.empty:
        return None

    bench = benchmarks.get(bench_key, {})
    excellent = bench.get('excellent', latest[metric_col].max())
    good = bench.get('good', excellent * 0.85)
    average = bench.get('average', excellent * 0.7)

    # Color based on performance - teal palette
    colors = []
    for val in latest[metric_col]:
        if val >= good:
            colors.append(TEAL_PRIMARY)  # Teal for good/excellent
        elif val >= average:
            colors.append('#4DB6AC')  # Medium teal for average
        else:
            colors.append('#78909C')  # Gray-blue for below average

    fig = go.Figure()

    # Add horizontal lines (stems)
    for i, (_, row) in enumerate(latest.iterrows()):
        fig.add_trace(go.Scatter(
            x=[0, row[metric_col]],
            y=[row[name_col], row[name_col]],
            mode='lines',
            line=dict(color=colors[i], width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add dots
    fig.add_trace(go.Scatter(
        x=latest[metric_col],
        y=latest[name_col],
        mode='markers',
        marker=dict(size=14, color=colors, line=dict(color='white', width=2)),
        text=[f"{v:.1f}" for v in latest[metric_col]],
        textposition='middle right',
        name='Athletes'
    ))

    # Add benchmark lines - teal palette
    max_val = latest[metric_col].max() * 1.1
    fig.add_vline(x=excellent, line_dash="dash", line_color=TEAL_PRIMARY,
                  annotation_text="Excellent", annotation_position="top")
    fig.add_vline(x=good, line_dash="dot", line_color="#4DB6AC",
                  annotation_text="Good", annotation_position="top")

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="",
        yaxis_title="",
        height=250,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig


def _create_bullet_chart(value: float, benchmarks: Dict, title: str, unit: str) -> go.Figure:
    """Create a bullet chart showing value against benchmarks."""
    bench = benchmarks
    excellent = bench.get('excellent', 100)
    good = bench.get('good', 80)
    average = bench.get('average', 60)
    max_val = excellent * 1.2

    fig = go.Figure()

    # Background ranges - using teal-based palette (no gold, no red)
    # Gray-blue for below average, light teal for average, teal for good, dark teal for excellent
    fig.add_trace(go.Bar(
        x=[max_val], y=[''], orientation='h',
        marker=dict(color='rgba(120, 144, 156, 0.25)'),  # Gray-blue for needs work
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Bar(
        x=[good], y=[''], orientation='h',
        marker=dict(color='rgba(0, 150, 136, 0.25)'),  # Light teal for average
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Bar(
        x=[excellent], y=[''], orientation='h',
        marker=dict(color='rgba(0, 113, 103, 0.35)'),  # Teal for excellent
        showlegend=False, hoverinfo='skip'
    ))

    # Actual value bar - teal shades based on performance
    if value >= good:
        color = TEAL_PRIMARY  # Strong teal for good/excellent
    elif value >= average:
        color = '#4DB6AC'  # Medium teal for average
    else:
        color = '#78909C'  # Gray-blue for below average

    fig.add_trace(go.Bar(
        x=[value], y=[''], orientation='h',
        marker=dict(color=color),
        width=0.3,
        name=f'{value:.1f} {unit}'
    ))

    fig.update_layout(
        title=dict(text=f"{title}: {value:.1f} {unit}", font=dict(size=12)),
        barmode='overlay',
        height=80,
        showlegend=False,
        xaxis=dict(range=[0, max_val], showticklabels=True),
        yaxis=dict(showticklabels=False),
        margin=dict(l=10, r=10, t=30, b=10)
    )

    return fig


def _create_scatter_comparison(df: pd.DataFrame, x_metric: str, y_metric: str,
                               name_col: str, title: str, x_label: str, y_label: str) -> go.Figure:
    """Create a scatter plot comparing two metrics."""
    if df.empty or x_metric not in df.columns or y_metric not in df.columns:
        return None

    # Get latest values per athlete
    latest = df.groupby(name_col).agg({x_metric: 'last', y_metric: 'last'}).reset_index()
    latest = latest.dropna()

    if latest.empty:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=latest[x_metric],
        y=latest[y_metric],
        mode='markers+text',
        marker=dict(size=12, color=TEAL_PRIMARY, line=dict(color='white', width=1)),
        text=latest[name_col].apply(lambda x: x.split()[0] if ' ' in str(x) else str(x)[:8]),
        textposition='top center',
        textfont=dict(size=9),
        hovertemplate='<b>%{text}</b><br>' + x_label + ': %{x:.1f}<br>' + y_label + ': %{y:.1f}<extra></extra>'
    ))

    # Add quadrant lines at median
    x_med = latest[x_metric].median()
    y_med = latest[y_metric].median()
    fig.add_hline(y=y_med, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=x_med, line_dash="dot", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig


def _create_box_plot(df: pd.DataFrame, metric_col: str, name_col: str,
                     benchmarks: Dict, title: str, bench_key: str) -> go.Figure:
    """Create a box plot with individual athlete dots overlaid."""
    if df.empty or metric_col not in df.columns:
        return None

    # Get latest value per athlete
    if name_col in df.columns:
        latest = df.groupby(name_col)[metric_col].last().reset_index()
        latest = latest.dropna(subset=[metric_col])
    else:
        latest = df[[metric_col]].dropna()

    if latest.empty:
        return None

    bench = benchmarks.get(bench_key, {})

    fig = go.Figure()

    # Add benchmark zones as shapes
    max_val = latest[metric_col].max()
    excellent = bench.get('excellent', max_val)
    good = bench.get('good', excellent * 0.85)

    fig.add_hrect(y0=excellent, y1=max_val * 1.15,
                  fillcolor="rgba(0, 113, 103, 0.20)", line_width=0)  # Teal for excellent
    fig.add_hrect(y0=good, y1=excellent,
                  fillcolor="rgba(0, 150, 136, 0.15)", line_width=0)  # Light teal for good
    fig.add_hrect(y0=0, y1=good,
                  fillcolor="rgba(120, 144, 156, 0.12)", line_width=0)  # Gray-blue for needs work

    # Box plot without points (we add them separately)
    fig.add_trace(go.Box(
        y=latest[metric_col],
        name='Team',
        marker_color=TEAL_PRIMARY,
        boxpoints=False,  # No built-in points
        fillcolor='rgba(0, 113, 103, 0.3)',
        line=dict(color=TEAL_PRIMARY)
    ))

    # Overlay individual athlete dots with names
    if name_col in latest.columns:
        # Use jitter to spread points
        import numpy as np
        np.random.seed(42)
        jitter = np.random.uniform(-0.15, 0.15, len(latest))

        fig.add_trace(go.Scatter(
            x=jitter,
            y=latest[metric_col],
            mode='markers+text',
            marker=dict(
                color=TEAL_PRIMARY,
                size=12,
                line=dict(color='white', width=1)
            ),
            text=[n.split()[0] if ' ' in str(n) else str(n)[:8] for n in latest[name_col]],
            textposition='middle right',
            textfont=dict(size=8, color='#333'),
            hovertemplate='<b>%{text}</b><br>Value: %{y:.1f}<extra></extra>',
            showlegend=False
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis_title="",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        height=320,
        showlegend=False,
        margin=dict(l=10, r=80, t=40, b=10)
    )

    return fig


def create_group_report_v3(df: pd.DataFrame,
                           sport: str,
                           forceframe_df: pd.DataFrame = None,
                           nordbord_df: pd.DataFrame = None,
                           config: Dict = None):
    """
    Group Report V3 - Alternative Visualization Styles.

    Uses different chart types for a fresh perspective:
    - Radar charts for multi-metric athlete profiles
    - Lollipop charts instead of bar charts
    - Scatter plots for metric comparisons
    - Box plots for team distributions
    - Bullet charts for quick status
    """
    benchmarks = get_sport_benchmarks(sport, config)

    st.markdown(f"## {sport} Group Report - Alternative View")
    st.markdown("*Different visualization styles for comparison*")
    st.markdown("---")

    # Filter data for the sport
    sport_df = df.copy()
    if 'athlete_sport' in sport_df.columns:
        sport_mask = sport_df['athlete_sport'].str.contains(sport.split()[0], case=False, na=False)
        if sport_mask.any():
            sport_df = sport_df[sport_mask]

    if sport_df.empty:
        st.warning(f"No data available for {sport}")
        return

    athletes = sport_df['Name'].dropna().unique() if 'Name' in sport_df.columns else []

    # =========================================================================
    # SECTION 1: Team Overview - Radar Charts
    # =========================================================================
    st.markdown("### 🎯 Athlete Performance Profiles")
    st.caption("Radar charts showing normalized performance (0-100) across metrics")

    # Create radar charts for top athletes
    if len(athletes) > 0:
        # Select athletes to display (max 4)
        display_athletes = athletes[:4]
        cols = st.columns(len(display_athletes))

        for i, athlete in enumerate(display_athletes):
            athlete_data = sport_df[sport_df['Name'] == athlete]

            # Gather metrics for radar
            metrics = {}

            # CMJ Height
            cmj_col = get_metric_column(athlete_data, 'cmj_height')
            if cmj_col:
                cmj_df = athlete_data[athlete_data['testType'].str.contains('CMJ|Counter', case=False, na=False)]
                if not cmj_df.empty and not cmj_df[cmj_col].dropna().empty:
                    metrics['CMJ Height'] = cmj_df[cmj_col].dropna().iloc[-1]

            # Power
            power_col = get_metric_column(athlete_data, 'relative_power')
            if power_col:
                cmj_df = athlete_data[athlete_data['testType'].str.contains('CMJ|Counter', case=False, na=False)]
                if not cmj_df.empty and not cmj_df[power_col].dropna().empty:
                    metrics['Power'] = cmj_df[power_col].dropna().iloc[-1]

            # IMTP
            force_col = get_metric_column(athlete_data, 'peak_force')
            if force_col:
                imtp_df = athlete_data[athlete_data['testType'].str.contains('IMTP|ISOT|Isometric', case=False, na=False)]
                if not imtp_df.empty and not imtp_df[force_col].dropna().empty:
                    metrics['IMTP'] = imtp_df[force_col].dropna().iloc[-1]

            # RSI
            rsi_col = get_metric_column(athlete_data, 'rsi')
            if rsi_col:
                hop_df = athlete_data[athlete_data['testType'].str.contains('Hop|DJ|Drop|CMJ', case=False, na=False)]
                if not hop_df.empty and not hop_df[rsi_col].dropna().empty:
                    metrics['RSI'] = hop_df[rsi_col].dropna().iloc[-1]

            with cols[i]:
                if len(metrics) >= 3:
                    # Map benchmarks correctly
                    radar_benchmarks = {
                        'cmj_height': benchmarks.get('cmj_height', {}),
                        'power': benchmarks.get('peak_power', {}),
                        'imtp': benchmarks.get('peak_force', {}),
                        'rsi': benchmarks.get('rsi', {})
                    }
                    fig = _create_radar_chart(metrics, radar_benchmarks, athlete)
                    st.plotly_chart(fig, use_container_width=True, key=f"v3_radar_{i}")
                    st.caption(f"**{athlete.split()[0] if ' ' in str(athlete) else athlete}**")
                else:
                    st.info(f"Limited data for {athlete}")

    st.markdown("---")

    # =========================================================================
    # SECTION 2: Lower Body - Lollipop & Box Charts
    # =========================================================================
    st.markdown("### 🦵 Lower Body Strength & Power")

    # Row 1: IMTP and CMJ Height
    col1, col2 = st.columns(2)

    with col1:
        # IMTP - Lollipop Chart
        metric_col = get_metric_column(sport_df, 'peak_force')
        if metric_col:
            imtp_df = sport_df[sport_df['testType'].str.contains('IMTP|ISOT|Isometric', case=False, na=False)]
            if not imtp_df.empty:
                fig = _create_lollipop_chart(
                    imtp_df, metric_col, 'Name', benchmarks,
                    "IMTP - Relative Peak Force (N/kg)", 'peak_force'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="v3_imtp_lollipop")
            else:
                st.info("No IMTP data available")
        else:
            st.info("IMTP metric not found")

    with col2:
        # CMJ Box Plot - Team Distribution with dots
        metric_col = get_metric_column(sport_df, 'cmj_height')
        if metric_col:
            cmj_df = sport_df[sport_df['testType'].str.contains('CMJ|Counter', case=False, na=False)]
            if not cmj_df.empty:
                fig = _create_box_plot(
                    cmj_df, metric_col, 'Name', benchmarks,
                    "CMJ Jump Height - Team Distribution", 'cmj_height'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="v3_cmj_box")
            else:
                st.info("No CMJ data available")
        else:
            st.info("CMJ metric not found")

    # Row 2: CMJ Power and Repeat Hop RSI
    col1, col2 = st.columns(2)

    with col1:
        # CMJ Power - Box plot with dots
        power_col = get_metric_column(sport_df, 'relative_power')
        if power_col:
            cmj_df = sport_df[sport_df['testType'].str.contains('CMJ|Counter', case=False, na=False)]
            if not cmj_df.empty:
                fig = _create_box_plot(
                    cmj_df, power_col, 'Name', benchmarks,
                    "CMJ Relative Power - Team Distribution", 'peak_power'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="v3_cmj_power_box")
            else:
                st.info("No CMJ power data available")
        else:
            st.info("Power metric not found")

    with col2:
        # Repeat Hop RSI - Lollipop Chart
        rsi_col = get_metric_column(sport_df, 'rsi')
        if rsi_col:
            hop_df = sport_df[sport_df['testType'].str.contains('Hop|Repeat|DJ|Drop', case=False, na=False)]
            if not hop_df.empty:
                fig = _create_lollipop_chart(
                    hop_df, rsi_col, 'Name', benchmarks,
                    "Repeat Hop - RSI", 'rsi'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="v3_rsi_lollipop")
            else:
                st.info("No Repeat Hop data available")
        else:
            st.info("RSI metric not found")

    # Row 3: Performance Quadrant - Power vs Height
    st.markdown("#### Performance Quadrant: Height vs Power")
    height_col = get_metric_column(sport_df, 'cmj_height')
    power_col = get_metric_column(sport_df, 'relative_power')
    if height_col and power_col:
        cmj_df = sport_df[sport_df['testType'].str.contains('CMJ|Counter', case=False, na=False)]
        if not cmj_df.empty and 'Name' in cmj_df.columns:
            # Create quadrant scatter with distinct styling
            latest = cmj_df.groupby('Name').agg({
                height_col: 'last',
                power_col: 'last'
            }).reset_index().dropna()

            if not latest.empty:
                fig = go.Figure()

                # Quadrant zones
                x_med = latest[height_col].median()
                y_med = latest[power_col].median()

                # Add quadrant labels
                fig.add_annotation(x=latest[height_col].max(), y=latest[power_col].max(),
                                   text="Elite", showarrow=False, font=dict(size=10, color="green"))
                fig.add_annotation(x=latest[height_col].min(), y=latest[power_col].max(),
                                   text="Powerful", showarrow=False, font=dict(size=10, color="orange"))
                fig.add_annotation(x=latest[height_col].max(), y=latest[power_col].min(),
                                   text="Springy", showarrow=False, font=dict(size=10, color="blue"))
                fig.add_annotation(x=latest[height_col].min(), y=latest[power_col].min(),
                                   text="Developing", showarrow=False, font=dict(size=10, color="gray"))

                # Quadrant lines
                fig.add_hline(y=y_med, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_vline(x=x_med, line_dash="dash", line_color="gray", opacity=0.5)

                # Scatter points with athlete names
                fig.add_trace(go.Scatter(
                    x=latest[height_col],
                    y=latest[power_col],
                    mode='markers+text',
                    marker=dict(
                        size=14,
                        color=TEAL_PRIMARY,
                        line=dict(color='white', width=2)
                    ),
                    text=[n.split()[0] if ' ' in str(n) else str(n)[:8] for n in latest['Name']],
                    textposition='top center',
                    textfont=dict(size=9),
                    hovertemplate='<b>%{text}</b><br>Height: %{x:.1f}cm<br>Power: %{y:.1f}W/kg<extra></extra>'
                ))

                fig.update_layout(
                    title=dict(text="Jump Height vs Power Quadrant", font=dict(size=14)),
                    xaxis_title="Jump Height (cm)",
                    yaxis_title="Relative Power (W/kg)",
                    height=350,
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                st.plotly_chart(fig, use_container_width=True, key="v3_quadrant_cmj")
            else:
                st.info("Insufficient CMJ data for quadrant plot")
        else:
            st.info("Insufficient CMJ data for quadrant plot")
    else:
        st.info("CMJ metrics not found for comparison")

    st.markdown("---")

    # =========================================================================
    # SECTION 3: NordBord - Hamstring Health (Left & Right)
    # =========================================================================
    st.markdown("### 🦵 Hamstring Strength (NordBord) - Left & Right")

    if nordbord_df is not None and not nordbord_df.empty:
        nb_df = nordbord_df.copy()
        if 'athlete_sport' in nb_df.columns:
            sport_mask = nb_df['athlete_sport'].str.contains(sport.split()[0], case=False, na=False)
            if sport_mask.any():
                nb_df = nb_df[sport_mask]

        left_col, right_col = get_nordbord_force_columns(nb_df)

        if left_col and right_col and 'Name' in nb_df.columns:
            # Get latest per athlete
            latest = nb_df.groupby('Name').agg({left_col: 'last', right_col: 'last'}).reset_index()
            latest = latest.dropna()

            if not latest.empty:
                # Calculate asymmetry for coloring
                latest['asymmetry'] = abs(
                    (latest[left_col] - latest[right_col]) /
                    ((latest[left_col] + latest[right_col]) / 2) * 100
                )

                # Sort by average force
                latest['avg'] = (latest[left_col] + latest[right_col]) / 2
                latest = latest.sort_values('avg', ascending=True)

                # Grouped bar chart - Left vs Right
                fig = go.Figure()

                # Left bars
                fig.add_trace(go.Bar(
                    name='Left',
                    y=latest['Name'],
                    x=latest[left_col],
                    orientation='h',
                    marker_color=TEAL_PRIMARY,
                    text=[f"{v:.0f}N" for v in latest[left_col]],
                    textposition='auto'
                ))

                # Right bars - coral for bilateral comparison
                fig.add_trace(go.Bar(
                    name='Right',
                    y=latest['Name'],
                    x=latest[right_col],
                    orientation='h',
                    marker_color=CORAL_ACCENT,
                    text=[f"{v:.0f}N" for v in latest[right_col]],
                    textposition='auto'
                ))

                # Add injury threshold line
                fig.add_vline(x=337, line_dash="dash", line_color="red",
                              annotation_text="337N Injury Risk", annotation_position="top")

                fig.update_layout(
                    title="NordBord - Left vs Right Hamstring Force",
                    barmode='group',
                    xaxis_title="Force (N)",
                    yaxis_title="",
                    height=max(300, len(latest) * 50),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=10, t=60, b=10)
                )
                st.plotly_chart(fig, use_container_width=True, key="v3_nb_left_right")

                # Asymmetry indicator table
                st.markdown("#### Asymmetry Status")
                asym_data = []
                for _, row in latest.iterrows():
                    asym = row['asymmetry']
                    if asym < 8:
                        status = "🟢 Good"
                    elif asym < 15:
                        status = "🟠 Monitor"
                    else:
                        status = "🔴 High Risk"
                    asym_data.append({
                        'Athlete': row['Name'],
                        'Left (N)': f"{row[left_col]:.0f}",
                        'Right (N)': f"{row[right_col]:.0f}",
                        'Asymmetry': f"{asym:.1f}%",
                        'Status': status
                    })
                st.dataframe(pd.DataFrame(asym_data), use_container_width=True, hide_index=True)
            else:
                st.info("No NordBord data after filtering")
        else:
            st.info("NordBord data not available")
    else:
        st.info("NordBord data not available")

    st.markdown("---")

    # =========================================================================
    # SECTION 4: Upper Body Strength & Power
    # =========================================================================
    st.markdown("### 💪 Upper Body Strength & Power")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Bench Press - Lollipop Chart
        bench_df = sport_df[sport_df['testType'].str.contains('Bench|Press', case=False, na=False)]
        if not bench_df.empty:
            metric_col = get_metric_column(bench_df, 'peak_force')
            if metric_col:
                fig = _create_lollipop_chart(
                    bench_df, metric_col, 'Name', benchmarks,
                    "Bench Press - Peak Force", 'peak_force'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="v3_bench_lollipop")
            else:
                st.info("Bench Press data not available")
        else:
            st.info("Bench Press data not available")

    with col2:
        # Pull Up - Lollipop Chart
        pullup_df = sport_df[sport_df['testType'].str.contains('Pull|Chin', case=False, na=False)]
        if not pullup_df.empty:
            metric_col = get_metric_column(pullup_df, 'peak_force')
            if metric_col:
                fig = _create_lollipop_chart(
                    pullup_df, metric_col, 'Name', benchmarks,
                    "Pull Up - Peak Force", 'peak_force'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="v3_pullup_lollipop")
            else:
                st.info("Pull Up data not available")
        else:
            st.info("Pull Up data not available")

    with col3:
        # Plyo Push Up - Box plot with dots
        plyo_df = sport_df[sport_df['testType'].str.contains('Plyo|Plyometric', case=False, na=False)]
        if not plyo_df.empty:
            metric_col = get_metric_column(plyo_df, 'peak_power')
            if metric_col:
                fig = _create_box_plot(
                    plyo_df, metric_col, 'Name', benchmarks,
                    "Plyo Push Up - Power", 'peak_power'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="v3_plyo_box")
            else:
                st.info("Plyo Push Up data not available")
        else:
            st.info("Plyo Push Up data not available")

    st.markdown("---")

    # =========================================================================
    # SECTION 5: Team Bullet Summary
    # =========================================================================
    st.markdown("### 📊 Team Average Performance")
    st.caption("Bullet charts showing team average vs benchmarks")

    # Calculate team averages
    metrics_summary = []

    # CMJ Height average
    cmj_col = get_metric_column(sport_df, 'cmj_height')
    if cmj_col:
        cmj_df = sport_df[sport_df['testType'].str.contains('CMJ|Counter', case=False, na=False)]
        if not cmj_df.empty:
            avg_val = cmj_df.groupby('Name')[cmj_col].last().mean()
            if pd.notna(avg_val):
                metrics_summary.append(('CMJ Height', avg_val, benchmarks.get('cmj_height', {}), 'cm'))

    # Power average
    power_col = get_metric_column(sport_df, 'relative_power')
    if power_col:
        cmj_df = sport_df[sport_df['testType'].str.contains('CMJ|Counter', case=False, na=False)]
        if not cmj_df.empty:
            avg_val = cmj_df.groupby('Name')[power_col].last().mean()
            if pd.notna(avg_val):
                metrics_summary.append(('Relative Power', avg_val, benchmarks.get('peak_power', {}), 'W/kg'))

    # IMTP average
    force_col = get_metric_column(sport_df, 'peak_force')
    if force_col:
        imtp_df = sport_df[sport_df['testType'].str.contains('IMTP|ISOT|Isometric', case=False, na=False)]
        if not imtp_df.empty:
            avg_val = imtp_df.groupby('Name')[force_col].last().mean()
            if pd.notna(avg_val):
                metrics_summary.append(('IMTP Peak Force', avg_val, benchmarks.get('peak_force', {}), 'N/kg'))

    # NordBord average
    if nordbord_df is not None and not nordbord_df.empty:
        nb_df = nordbord_df.copy()
        left_col, right_col = get_nordbord_force_columns(nb_df)
        if left_col and right_col and 'Name' in nb_df.columns:
            nb_df['avg_force'] = (nb_df[left_col] + nb_df[right_col]) / 2
            avg_val = nb_df.groupby('Name')['avg_force'].last().mean()
            if pd.notna(avg_val):
                metrics_summary.append(('Hamstring Force', avg_val, benchmarks.get('nordbord_force', {}), 'N'))

    # Display bullet charts in grid
    if metrics_summary:
        cols = st.columns(2)
        for i, (name, value, bench, unit) in enumerate(metrics_summary):
            with cols[i % 2]:
                fig = _create_bullet_chart(value, bench, name, unit)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"v3_bullet_{i}")
    else:
        st.info("No data available for team summary")

    st.markdown("---")

    # =========================================================================
    # SECTION 6: ForceFrame - Left & Right (if available)
    # =========================================================================
    if forceframe_df is not None and not forceframe_df.empty:
        st.markdown("### 💪 ForceFrame - Isometric Strength (Left & Right)")

        ff_df = forceframe_df.copy()
        if 'athlete_sport' in ff_df.columns:
            sport_mask = ff_df['athlete_sport'].str.contains(sport.split()[0], case=False, na=False)
            if sport_mask.any():
                ff_df = ff_df[sport_mask]

        if 'testTypeName' in ff_df.columns and 'Name' in ff_df.columns:
            # Find left/right force columns for ForceFrame
            # ForceFrame has: innerLeftMaxForce, innerRightMaxForce, outerLeftMaxForce, outerRightMaxForce
            inner_left = 'innerLeftMaxForce' if 'innerLeftMaxForce' in ff_df.columns else None
            inner_right = 'innerRightMaxForce' if 'innerRightMaxForce' in ff_df.columns else None
            outer_left = 'outerLeftMaxForce' if 'outerLeftMaxForce' in ff_df.columns else None
            outer_right = 'outerRightMaxForce' if 'outerRightMaxForce' in ff_df.columns else None

            # Shoulder tests - Left vs Right grouped bar
            st.markdown("#### Shoulder Tests")
            shoulder_df = ff_df[ff_df['testTypeName'].str.contains('^Shoulder', case=False, na=False, regex=True)]
            if not shoulder_df.empty and inner_left and inner_right:
                # Get latest per athlete for shoulder
                latest_shoulder = shoulder_df.groupby('Name').agg({
                    inner_left: 'last',
                    inner_right: 'last',
                    'testTypeName': 'last'
                }).reset_index()
                latest_shoulder = latest_shoulder.dropna(subset=[inner_left, inner_right])

                if not latest_shoulder.empty:
                    fig = go.Figure()

                    # Left (Inner) bars
                    fig.add_trace(go.Bar(
                        name='Left (Inner)',
                        y=latest_shoulder['Name'],
                        x=latest_shoulder[inner_left],
                        orientation='h',
                        marker_color=TEAL_PRIMARY,
                        text=[f"{v:.0f}N" for v in latest_shoulder[inner_left]],
                        textposition='auto'
                    ))

                    # Right (Inner) bars - coral for bilateral comparison
                    fig.add_trace(go.Bar(
                        name='Right (Inner)',
                        y=latest_shoulder['Name'],
                        x=latest_shoulder[inner_right],
                        orientation='h',
                        marker_color=CORAL_ACCENT,
                        text=[f"{v:.0f}N" for v in latest_shoulder[inner_right]],
                        textposition='auto'
                    ))

                    fig.update_layout(
                        title="Shoulder - Left vs Right Force",
                        barmode='group',
                        xaxis_title="Force (N)",
                        yaxis_title="",
                        height=max(250, len(latest_shoulder) * 50),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=10, r=10, t=60, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True, key="v3_ff_shoulder_lr")

                    # Asymmetry table for shoulder
                    asym_data = []
                    for _, row in latest_shoulder.iterrows():
                        left_val = row[inner_left]
                        right_val = row[inner_right]
                        avg = (left_val + right_val) / 2
                        asym = abs((left_val - right_val) / avg * 100) if avg > 0 else 0
                        if asym < 8:
                            status = "🟢 Good"
                        elif asym < 15:
                            status = "🟠 Monitor"
                        else:
                            status = "🔴 High"
                        asym_data.append({
                            'Athlete': row['Name'],
                            'Test': row['testTypeName'],
                            'Left (N)': f"{left_val:.0f}",
                            'Right (N)': f"{right_val:.0f}",
                            'Asymmetry': f"{asym:.1f}%",
                            'Status': status
                        })
                    st.dataframe(pd.DataFrame(asym_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No shoulder data with left/right values")
            else:
                st.info("No shoulder tests found")

            st.markdown("---")

            # Hip tests - Left vs Right grouped bar
            st.markdown("#### Hip Tests")
            hip_df = ff_df[ff_df['testTypeName'].str.contains('^Hip', case=False, na=False, regex=True)]
            if not hip_df.empty and inner_left and inner_right:
                # Get latest per athlete for hip
                latest_hip = hip_df.groupby('Name').agg({
                    inner_left: 'last',
                    inner_right: 'last',
                    'testTypeName': 'last'
                }).reset_index()
                latest_hip = latest_hip.dropna(subset=[inner_left, inner_right])

                if not latest_hip.empty:
                    fig = go.Figure()

                    # Left (Inner) bars
                    fig.add_trace(go.Bar(
                        name='Left (Inner)',
                        y=latest_hip['Name'],
                        x=latest_hip[inner_left],
                        orientation='h',
                        marker_color=TEAL_PRIMARY,
                        text=[f"{v:.0f}N" for v in latest_hip[inner_left]],
                        textposition='auto'
                    ))

                    # Right (Inner) bars - coral for bilateral comparison
                    fig.add_trace(go.Bar(
                        name='Right (Inner)',
                        y=latest_hip['Name'],
                        x=latest_hip[inner_right],
                        orientation='h',
                        marker_color=CORAL_ACCENT,
                        text=[f"{v:.0f}N" for v in latest_hip[inner_right]],
                        textposition='auto'
                    ))

                    fig.update_layout(
                        title="Hip - Left vs Right Force",
                        barmode='group',
                        xaxis_title="Force (N)",
                        yaxis_title="",
                        height=max(250, len(latest_hip) * 50),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=10, r=10, t=60, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True, key="v3_ff_hip_lr")

                    # Asymmetry table for hip
                    asym_data = []
                    for _, row in latest_hip.iterrows():
                        left_val = row[inner_left]
                        right_val = row[inner_right]
                        avg = (left_val + right_val) / 2
                        asym = abs((left_val - right_val) / avg * 100) if avg > 0 else 0
                        if asym < 8:
                            status = "🟢 Good"
                        elif asym < 15:
                            status = "🟠 Monitor"
                        else:
                            status = "🔴 High"
                        asym_data.append({
                            'Athlete': row['Name'],
                            'Test': row['testTypeName'],
                            'Left (N)': f"{left_val:.0f}",
                            'Right (N)': f"{right_val:.0f}",
                            'Asymmetry': f"{asym:.1f}%",
                            'Status': status
                        })
                    st.dataframe(pd.DataFrame(asym_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No hip data with left/right values")
            else:
                st.info("No hip tests found")
        else:
            st.info("ForceFrame data format not compatible")


def _render_placeholder_chart(title: str, description: str, height: int = 200):
    """Render a placeholder for charts that don't have data yet."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #5B9BD5 0%, #4A8AC7 100%);
        border-radius: 8px;
        padding: 20px;
        min-height: {height}px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: white;
        margin-bottom: 10px;
    ">
        <h4 style="margin: 0 0 10px 0; font-size: 14px;">{title}</h4>
        <p style="margin: 0; font-size: 12px; opacity: 0.9;">{description}</p>
    </div>
    """, unsafe_allow_html=True)


def render_benchmark_legend():
    """Render a legend explaining the benchmark zones - teal palette."""
    st.markdown("""
    <div style="display: flex; gap: 20px; padding: 10px; background: #f8f9fa; border-radius: 5px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 5px;">
            <div style="width: 20px; height: 20px; background: rgba(0, 113, 103, 0.35); border-radius: 3px;"></div>
            <span style="font-size: 12px;">Excellent</span>
        </div>
        <div style="display: flex; align-items: center; gap: 5px;">
            <div style="width: 20px; height: 20px; background: rgba(0, 150, 136, 0.3); border-radius: 3px;"></div>
            <span style="font-size: 12px;">Good</span>
        </div>
        <div style="display: flex; align-items: center; gap: 5px;">
            <div style="width: 20px; height: 20px; background: rgba(120, 144, 156, 0.3); border-radius: 3px;"></div>
            <span style="font-size: 12px;">Needs Improvement</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
