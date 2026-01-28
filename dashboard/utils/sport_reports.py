"""
Sport-Specific Group & Individual Reports Module

Provides generic visualization templates for sport reports with:
- Group reports showing team performance with benchmark zones
- Individual athlete reports with trend analysis
- Shaded benchmark zones on all graphs (green/yellow/red)
- Dynamic benchmarks from VALD norms database (sport & gender specific)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import io
import base64

# Import benchmark database for dynamic VALD norms and Saudi population norms
try:
    from .benchmark_database import (
        load_benchmarks,
        load_saudi_norms,
        get_benchmark_for_test,
        get_asymmetry_threshold,
        VALD_NORMS
    )
    BENCHMARK_DB_AVAILABLE = True
except ImportError:
    BENCHMARK_DB_AVAILABLE = False
    VALD_NORMS = {}
    def load_saudi_norms():
        return {}

# Import S&C Diagnostics chart functions
try:
    from .snc_diagnostics import (
        create_ranked_bar_chart,
        create_ranked_side_by_side_chart,
        create_individual_line_chart,
        create_stacked_quadrant_chart,
        create_multi_line_strength_chart,
        render_snc_diagnostics_tab,
        TEST_CONFIG,
        QUADRANT_COLORS
    )
    SNC_CHARTS_AVAILABLE = True
except ImportError:
    SNC_CHARTS_AVAILABLE = False
    TEST_CONFIG = {}
    QUADRANT_COLORS = {}
    render_snc_diagnostics_tab = None

# Team Saudi colors (from THEME_GUIDE.md)
TEAL_PRIMARY = '#255035'      # Saudi Green
GOLD_ACCENT = '#a08e66'       # Gold accent
TEAL_DARK = '#1C3D28'         # Dark green
TEAL_LIGHT = '#2E6040'        # Light green
GRAY_BLUE = '#78909C'         # Neutral gray
INFO_BLUE = '#0077B6'         # Info/testing blue

# Benchmark zone colors (with transparency) - Team Saudi teal palette
ZONE_COLORS = {
    'excellent': 'rgba(37, 80, 53, 0.20)',        # Saudi Green (excellent)
    'good': 'rgba(46, 96, 64, 0.15)',             # Light green (good)
    'average': 'rgba(120, 144, 156, 0.15)',       # Gray-blue (needs work)
    'border_excellent': 'rgba(37, 80, 53, 0.5)',
    'border_good': 'rgba(46, 96, 64, 0.5)',
    'border_average': 'rgba(120, 144, 156, 0.5)',
}

# Secondary color for bilateral comparisons (blue instead of coral/red)
SECONDARY_ACCENT = '#0077B6'  # Info blue for Right side in L/R comparisons

# Default benchmarks if sport-specific not available (fallback values)
DEFAULT_BENCHMARKS = {
    'cmj_height': {'excellent': 40, 'good': 35, 'average': 30},
    'peak_power': {'excellent': 55, 'good': 48, 'average': 40},
    'rsi': {'excellent': 2.0, 'good': 1.5, 'average': 1.2},
    'peak_force': {'excellent': 30, 'good': 25, 'average': 20},
    'asymmetry': {'excellent': 5, 'good': 10, 'average': 15},
    'nordbord_force': {'excellent': 400, 'good': 337, 'average': 280},  # N - 337N is injury risk threshold
}


# =============================================================================
# EXPORT HELPER FUNCTIONS
# =============================================================================

def export_chart_to_png(fig: go.Figure, filename: str = "chart") -> bytes:
    """
    Export a Plotly figure to PNG bytes for download.

    Args:
        fig: Plotly figure object
        filename: Base filename (without extension)

    Returns:
        PNG image as bytes
    """
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
        return img_bytes
    except Exception as e:
        st.warning(f"PNG export requires kaleido package: {e}")
        return None


def export_chart_to_pdf(fig: go.Figure, filename: str = "chart") -> bytes:
    """
    Export a Plotly figure to PDF bytes for download.

    Args:
        fig: Plotly figure object
        filename: Base filename (without extension)

    Returns:
        PDF image as bytes
    """
    try:
        pdf_bytes = fig.to_image(format="pdf", width=1200, height=800)
        return pdf_bytes
    except Exception as e:
        st.warning(f"PDF export requires kaleido package: {e}")
        return None


def export_dataframe_to_csv(df: pd.DataFrame) -> str:
    """
    Export a DataFrame to CSV string for download.

    Args:
        df: DataFrame to export

    Returns:
        CSV string
    """
    return df.to_csv(index=False)


def create_download_button_for_chart(fig: go.Figure, chart_name: str, key_prefix: str):
    """
    Create download buttons for a chart (PNG and PDF).

    Args:
        fig: Plotly figure object
        chart_name: Name of the chart for filename
        key_prefix: Unique key prefix for Streamlit buttons
    """
    col1, col2 = st.columns(2)

    with col1:
        try:
            png_bytes = export_chart_to_png(fig, chart_name)
            if png_bytes:
                st.download_button(
                    label="📥 Download PNG",
                    data=png_bytes,
                    file_name=f"{chart_name.replace(' ', '_').lower()}.png",
                    mime="image/png",
                    key=f"{key_prefix}_png"
                )
        except Exception:
            pass

    with col2:
        try:
            pdf_bytes = export_chart_to_pdf(fig, chart_name)
            if pdf_bytes:
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_bytes,
                    file_name=f"{chart_name.replace(' ', '_').lower()}.pdf",
                    mime="application/pdf",
                    key=f"{key_prefix}_pdf"
                )
        except Exception:
            pass


def create_download_button_for_table(df: pd.DataFrame, table_name: str, key_prefix: str):
    """
    Create download button for a table (CSV).

    Args:
        df: DataFrame to export
        table_name: Name of the table for filename
        key_prefix: Unique key prefix for Streamlit button
    """
    csv_data = export_dataframe_to_csv(df)
    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name=f"{table_name.replace(' ', '_').lower()}.csv",
        mime="text/csv",
        key=f"{key_prefix}_csv"
    )


def get_dynamic_benchmarks(gender: str = "male", source: str = "VALD") -> Dict:
    """
    Get benchmarks from the VALD norms database or Saudi population norms, adjusted for gender.

    Args:
        gender: 'male' or 'female'
        source: 'VALD' for international norms, 'Saudi' for Saudi population norms

    Returns:
        Dictionary of benchmarks for use in charts
    """
    if not BENCHMARK_DB_AVAILABLE:
        return DEFAULT_BENCHMARKS.copy()

    result = DEFAULT_BENCHMARKS.copy()
    gender_key = gender.lower() if gender else "male"

    # Check if using Saudi norms
    if source == "Saudi":
        saudi_norms = load_saudi_norms()
        if saudi_norms:
            # Map Saudi norm keys to our metric keys
            saudi_mapping = {
                'cmj_height': ('CMJ', 'jump_height', 1),  # Already in cm in Saudi norms
                'peak_power': ('CMJ', 'peak_power', 1),
                'peak_force': ('IMTP', 'peak_force', 1),
                'rsi': ('DJ', 'rsi', 1),
                'nordbord_force': ('NordBord', 'max_force', 1),
            }

            for metric_key, (test_type, saudi_metric, multiplier) in saudi_mapping.items():
                if test_type in saudi_norms:
                    metric_data = saudi_norms[test_type].get(saudi_metric, {})
                    gender_vals = metric_data.get(gender_key, {})
                    if gender_vals and gender_vals.get('elite', 0) > 0:  # Only use if values are set
                        result[metric_key] = {
                            'excellent': gender_vals.get('elite', result[metric_key]['excellent']) * multiplier,
                            'good': gender_vals.get('good', result[metric_key]['good']) * multiplier,
                            'average': gender_vals.get('average', result[metric_key]['average']) * multiplier,
                        }
            return result

    # Default: Use VALD international norms
    benchmarks = load_benchmarks()

    # Map test types to metric keys
    metric_mapping = {
        'cmj_height': ('CMJ', 'Jump Height (Imp-Mom)_Trial', 100),  # multiply by 100 for cm
        'peak_power': ('CMJ', 'Peak Power / BM_Trial', 1),
        'peak_force': ('IMTP', 'Peak Force / BM_Trial', 1),
        'rsi': ('DJ', 'RSI_Trial', 1),
        'nordbord_force': ('NordBord', 'leftMaxForce', 1),
    }

    for metric_key, (test_type, metric_col, multiplier) in metric_mapping.items():
        if test_type in benchmarks:
            metrics = benchmarks[test_type].get('metrics', {})
            if metric_col in metrics:
                gender_vals = metrics[metric_col].get(gender_key, {})
                if gender_vals:
                    result[metric_key] = {
                        'excellent': gender_vals.get('elite', result[metric_key]['excellent']) * multiplier,
                        'good': gender_vals.get('good', result[metric_key]['good']) * multiplier,
                        'average': gender_vals.get('average', result[metric_key]['average']) * multiplier,
                    }

    return result

# Metric column mappings
# Note: Jump Height (Imp-Mom) is preferred over Flight Time method for accuracy
# Supports both legacy API format (with _Trial suffix) and local_sync format (UPPERCASE)
METRIC_COLUMNS = {
    'cmj_height': [
        'JUMP_HEIGHT_IMP_MOM',  # local_sync format (meters)
        'JUMP_HEIGHT',  # local_sync format (meters)
        'Jump Height (Imp-Mom)_Trial',  # legacy API format
        'Jump Height (Flight Time)_Trial',
    ],
    'peak_power': [
        'BODYMASS_RELATIVE_TAKEOFF_POWER',  # local_sync format (W/kg)
        'PEAK_TAKEOFF_POWER',  # local_sync format (W) - needs BM division
        'Peak Power / BM_Trial',  # legacy API format
        'Peak Power_Trial',
    ],
    'relative_power': [
        'BODYMASS_RELATIVE_TAKEOFF_POWER',  # local_sync format
        'BODYMASS_RELATIVE_MEAN_CONCENTRIC_POWER',
        'Peak Power / BM_Trial',  # legacy API format
    ],
    'peak_force': [
        'ISO_BM_REL_FORCE_PEAK',  # IMTP body-mass relative (local_sync)
        'RELATIVE_PEAK_TAKEOFF_FORCE',  # CMJ relative force (local_sync)
        'PEAK_VERTICAL_FORCE',  # IMTP absolute force (local_sync)
        'Peak Force / BM_Trial',  # legacy API format
        'Peak Vertical Force / BM_Trial',
    ],
    'rsi': [
        'RSI_MODIFIED',  # local_sync format
        'RSI_MODIFIED_IMP_MOM',
        'RSI',  # local_sync format
        'RSI-modified_Trial',  # legacy API format
        'RSI (Flight/Contact Time)_Trial',
        'RSI-modified (Imp-Mom)_Trial',
    ],
    'contraction_time': [
        'CONTRACTION_TIME',  # local_sync format
        'Contraction Time_Trial',  # legacy API format
    ],
    'countermovement_depth': [
        'COUNTERMOVEMENT_DEPTH',  # local_sync format
        'Countermovement Depth_Trial',  # legacy API format
    ],
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


# Metric conversion factors - applied when values need unit adjustment
# RSI values from VALD API are often reported as percentages (e.g., 45.0 instead of 0.45)
METRIC_CONVERSIONS = {
    'rsi': {
        # Columns that need /100 conversion
        'RSI_MODIFIED': 0.01,
        'RSI_MODIFIED_IMP_MOM': 0.01,
        'RSI': 0.01,
        # Legacy columns already in correct units
        'RSI-modified_Trial': 1.0,
        'RSI (Flight/Contact Time)_Trial': 1.0,
        'RSI-modified (Imp-Mom)_Trial': 1.0,
    }
}


def apply_metric_conversion(df: pd.DataFrame, metric_key: str, col_name: str) -> pd.DataFrame:
    """
    Apply unit conversion to a metric column if needed.

    Args:
        df: DataFrame to modify (makes a copy)
        metric_key: The metric key (e.g., 'rsi')
        col_name: The actual column name in the dataframe

    Returns:
        DataFrame with converted values
    """
    if metric_key not in METRIC_CONVERSIONS:
        return df

    conversions = METRIC_CONVERSIONS[metric_key]
    if col_name not in conversions:
        return df

    factor = conversions[col_name]
    if factor == 1.0:
        return df

    # Make a copy to avoid modifying original
    df = df.copy()

    # Check if values need conversion (RSI should be < 3.0 typically)
    if col_name in df.columns and df[col_name].notna().any():
        median_val = df[col_name].median()
        # If median is > 10, values likely need conversion
        if median_val > 10:
            df[col_name] = df[col_name] * factor

    return df


def enrich_athlete_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich DataFrame with athlete names from VALD Profiles API if missing.

    This ensures athlete names appear when data comes from API
    without pre-enriched names.
    """
    # Check if we already have names
    if 'Name' in df.columns and df['Name'].notna().any():
        # Check if names are not just placeholders
        sample_names = df['Name'].dropna().head(5).tolist()
        if sample_names and not all(str(n).startswith('Athlete_') or str(n) == 'Unknown' for n in sample_names):
            return df

    # No valid names - try to enrich from profiles
    if 'profileId' not in df.columns:
        return df

    try:
        # Import data_loader function to get profiles
        from dashboard.utils.data_loader import _get_vald_credentials, _fetch_athlete_profiles, _get_oauth_token
    except ImportError:
        try:
            from utils.data_loader import _get_vald_credentials, _fetch_athlete_profiles, _get_oauth_token
        except ImportError:
            return df

    try:
        credentials = _get_vald_credentials()
        if not credentials:
            return df

        token = credentials.get('manual_token')
        if not token and credentials.get('client_id') and credentials.get('client_secret'):
            token = _get_oauth_token(credentials['client_id'], credentials['client_secret'])

        if not token:
            return df

        region = credentials.get('region', 'euw')
        tenant_id = credentials.get('tenant_id')

        if not tenant_id:
            return df

        # Fetch profiles
        profile_map = _fetch_athlete_profiles(token, region, tenant_id)

        if profile_map:
            df = df.copy()
            df['Name'] = df['profileId'].map(lambda pid: profile_map.get(pid, {}).get('Name', f"Athlete_{str(pid)[:8]}"))
            df['full_name'] = df['Name']  # Sync both columns
    except Exception as e:
        st.warning(f"Could not enrich athlete names: {e}")

    return df


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
        height=max(250, min(400, len(plot_data) * 25 + 80)),  # Compact height, max 400px
        margin=dict(l=120, r=40, t=40, b=40),
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

    Features:
    - Gender filter for gender-specific VALD norms
    - Dynamic benchmarks from benchmark database
    - Layout: Lower Body, Upper Body, Shoulder Health, Hip Health

    Benchmarks are loaded from VALD norms and can be adjusted in Benchmark Settings.
    """
    # Ensure Name column exists (map from full_name if needed)
    if 'Name' not in df.columns and 'full_name' in df.columns:
        df = df.copy()
        df['Name'] = df['full_name']

    st.markdown(f"## {sport} Group Report")

    # =========================================================================
    # FILTERS - Gender, Benchmark Source, and Athlete selection
    # =========================================================================
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])

    with filter_col1:
        # Gender filter - affects benchmark values
        genders = ['All', 'Male', 'Female']
        selected_gender = st.selectbox(
            "Gender (for benchmarks):",
            genders,
            key="group_v1_gender_filter",
            help="Benchmarks adjust based on gender selection"
        )

    with filter_col2:
        # Benchmark source selection
        benchmark_sources = ['VALD International', 'Saudi Population']
        selected_source = st.selectbox(
            "Benchmark Source:",
            benchmark_sources,
            key="group_v1_benchmark_source",
            help="VALD = International norms, Saudi = Custom population benchmarks"
        )

    # Get dynamic benchmarks based on gender and source
    gender_for_benchmarks = selected_gender.lower() if selected_gender != 'All' else 'male'
    source_key = 'Saudi' if selected_source == 'Saudi Population' else 'VALD'
    benchmarks = get_dynamic_benchmarks(gender_for_benchmarks, source_key)

    # Also merge with sport-specific benchmarks if available
    sport_benchmarks = get_sport_benchmarks(sport, config)
    for key, value in sport_benchmarks.items():
        if key not in benchmarks:
            benchmarks[key] = value

    # Show benchmark source indicator
    if BENCHMARK_DB_AVAILABLE:
        source_label = "Saudi Population" if source_key == 'Saudi' else "VALD International"
        st.caption(f"📊 Using {source_label} norms ({selected_gender}) - Edit in Benchmark Settings tab")
    else:
        st.caption("Using default benchmarks")

    # Filter data for the sport - use EXACT match for full sport name
    sport_df = df.copy()
    # Handle "All Sports" or None - don't filter by sport
    if sport and sport != "All Sports" and 'athlete_sport' in sport_df.columns:
        # Try exact match first
        exact_mask = sport_df['athlete_sport'] == sport
        if exact_mask.any():
            sport_df = sport_df[exact_mask]
        else:
            # Fall back to fuzzy match for parent sport category (e.g., "Athletics")
            sport_mask = sport_df['athlete_sport'].str.contains(sport.split()[0], case=False, na=False)
            if sport_mask.any():
                sport_df = sport_df[sport_mask]

    # Optionally filter by gender if not "All"
    if selected_gender != 'All' and 'athlete_sex' in sport_df.columns:
        sport_df = sport_df[sport_df['athlete_sex'].str.lower() == selected_gender.lower()]

    if sport_df.empty:
        st.warning(f"No data available for {sport} with current filters")
        return

    # =========================================================================
    # ATHLETE FILTER - Allow selecting specific athletes
    # =========================================================================
    if 'Name' in sport_df.columns:
        available_athletes = sorted(sport_df['Name'].dropna().unique().tolist())
        if available_athletes:
            with filter_col3:
                selected_athletes = st.multiselect(
                    "Filter Athletes:",
                    options=available_athletes,
                    default=[],  # Empty = show all
                    key="group_v1_athlete_filter",
                    help="Leave empty to show all athletes, or select specific athletes"
                )
            # Apply athlete filter if selections made
            if selected_athletes:
                sport_df = sport_df[sport_df['Name'].isin(selected_athletes)]
                # Also filter ForceFrame and NordBord if passed
                if forceframe_df is not None and not forceframe_df.empty and 'Name' in forceframe_df.columns:
                    forceframe_df = forceframe_df[forceframe_df['Name'].isin(selected_athletes)]
                if nordbord_df is not None and not nordbord_df.empty and 'Name' in nordbord_df.columns:
                    nordbord_df = nordbord_df[nordbord_df['Name'].isin(selected_athletes)]

    st.markdown("---")

    # =========================================================================
    # SECTION 1: Lower Body Strength & Power (S&C Diagnostics Style)
    # =========================================================================
    st.markdown("### Lower Body Strength & Power")

    # Get benchmark values for reference lines
    imtp_benchmark = benchmarks.get('peak_force', {}).get('good', 30)
    cmj_power_benchmark = benchmarks.get('peak_power', {}).get('good', 50)
    cmj_height_benchmark = benchmarks.get('cmj_height', {}).get('good', 35)
    rsi_benchmark = benchmarks.get('rsi', {}).get('good', 1.5)
    nordbord_benchmark = benchmarks.get('nordbord_force', {}).get('good', 337)

    # Row 1: IMTP and CMJ Power (Tier 1 tests - Ranked Bars)
    col1, col2 = st.columns(2)

    # IMTP - Relative Peak Force (Ranked Bar with Squad Avg + Benchmark)
    with col1:
        metric_col = get_metric_column(sport_df, 'peak_force')
        if metric_col:
            imtp_df = sport_df[sport_df['testType'].str.contains('IMTP|ISOT|Isometric', case=False, na=False)]
            if not imtp_df.empty and SNC_CHARTS_AVAILABLE:
                # Get latest per athlete
                latest_imtp = imtp_df.groupby('Name')[metric_col].last().reset_index()
                fig = create_ranked_bar_chart(
                    latest_imtp, metric_col,
                    "Relative Peak Force", "N/kg",
                    benchmark=imtp_benchmark,
                    title="IMTP - Relative Peak Force"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            elif not imtp_df.empty:
                # Fallback to old style
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

    # CMJ - Relative Peak Power (Ranked Bar)
    with col2:
        metric_col = get_metric_column(sport_df, 'relative_power')
        if metric_col:
            cmj_df = sport_df[sport_df['testType'].str.contains('CMJ|Counter', case=False, na=False)]
            if not cmj_df.empty and SNC_CHARTS_AVAILABLE:
                latest_cmj = cmj_df.groupby('Name')[metric_col].last().reset_index()
                fig = create_ranked_bar_chart(
                    latest_cmj, metric_col,
                    "Relative Peak Power", "W/kg",
                    benchmark=cmj_power_benchmark,
                    title="CMJ - Relative Peak Power"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            elif not cmj_df.empty:
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

    # Row 2: CMJ Height and 10:5 Hop RSI (Tier 1/2 - Ranked Bars)
    col1, col2 = st.columns(2)

    # CMJ - Jump Height
    with col1:
        metric_col = get_metric_column(sport_df, 'cmj_height')
        if metric_col:
            cmj_df = sport_df[sport_df['testType'].str.contains('CMJ|Counter', case=False, na=False)]
            if not cmj_df.empty and SNC_CHARTS_AVAILABLE:
                latest_cmj = cmj_df.groupby('Name')[metric_col].last().reset_index()
                # Convert m to cm if needed
                if latest_cmj[metric_col].max() < 1:
                    latest_cmj[metric_col] = latest_cmj[metric_col] * 100
                fig = create_ranked_bar_chart(
                    latest_cmj, metric_col,
                    "Jump Height", "cm",
                    benchmark=cmj_height_benchmark,
                    title="CMJ - Jump Height (Impulse-Mom)"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            elif not cmj_df.empty:
                fig = create_benchmark_bar_chart(
                    cmj_df, metric_col, 'Name', benchmarks,
                    "CMJ - Jump Height", 'cmj_height', 'cm'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No CMJ height data available")
        else:
            st.info("CMJ height metric not found")

    # 10:5 Hop Test - RSI (Ranked Bar)
    with col2:
        metric_col = get_metric_column(sport_df, 'rsi')
        if metric_col:
            # RSHIP = 10:5 Repeat Single Hop In Place
            hop_df = sport_df[sport_df['testType'].str.contains('RSHIP|DJ|SLDJ|Hop|Drop', case=False, na=False)]
            # Apply RSI unit conversion if needed
            hop_df = apply_metric_conversion(hop_df, 'rsi', metric_col)
            if not hop_df.empty and SNC_CHARTS_AVAILABLE:
                latest_hop = hop_df.groupby('Name')[metric_col].last().reset_index()
                fig = create_ranked_bar_chart(
                    latest_hop, metric_col,
                    "RSI", "",
                    benchmark=rsi_benchmark,
                    title="10:5 Hop Test - RSI"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            elif not hop_df.empty:
                fig = create_benchmark_bar_chart(
                    hop_df, metric_col, 'Name', benchmarks,
                    "10:5 Hop Test - RSI", 'rsi', ''
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No 10:5 Hop / reactive strength data available")

    # Row 3: NordBord (Unilateral Side-by-Side)
    col1, col2 = st.columns(2)

    # NordBord - Left vs Right (Side-by-Side Bar)
    with col1:
        if nordbord_df is not None and not nordbord_df.empty:
            nb_df = nordbord_df.copy()
            if sport and sport != "All Sports" and 'athlete_sport' in nb_df.columns:
                sport_mask = nb_df['athlete_sport'].str.contains(sport.split()[0], case=False, na=False)
                if sport_mask.any():
                    nb_df = nb_df[sport_mask]

            left_col, right_col = get_nordbord_force_columns(nb_df)

            if left_col and right_col and 'Name' in nb_df.columns and SNC_CHARTS_AVAILABLE:
                # Get latest per athlete
                latest_nb = nb_df.groupby('Name').agg({left_col: 'last', right_col: 'last'}).reset_index()
                fig = create_ranked_side_by_side_chart(
                    latest_nb, left_col, right_col,
                    "Hamstring Force", "N",
                    benchmark=nordbord_benchmark,
                    title="NordBord - L/R Hamstring Strength"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            elif left_col and right_col and 'Name' in nb_df.columns:
                # Fallback: Calculate average of left/right
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

    # SL ISO Squat - Unilateral (Side-by-Side Bar)
    with col2:
        # Look for Single Leg Isometric Squat tests
        sliso_df = sport_df[sport_df['testType'].str.contains('SLISOSQT|SLISO|SL.*Squat', case=False, na=False)]
        if not sliso_df.empty:
            # Try to find left/right force columns
            left_col = None
            right_col = None
            for col in sliso_df.columns:
                if 'Left' in col and ('Force' in col or 'Peak' in col):
                    left_col = col
                elif 'Right' in col and ('Force' in col or 'Peak' in col):
                    right_col = col

            if left_col and right_col and 'Name' in sliso_df.columns and SNC_CHARTS_AVAILABLE:
                latest_sliso = sliso_df.groupby('Name').agg({left_col: 'last', right_col: 'last'}).reset_index()
                fig = create_ranked_side_by_side_chart(
                    latest_sliso, left_col, right_col,
                    "Relative Peak Force", "N/kg",
                    title="SL ISO Squat - L/R"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("SL ISO Squat data not available")
        else:
            st.info("SL ISO Squat data not available")

    # Row 4: SL IMTP and SL CMJ (Tier 2 - Unilateral Side-by-Side)
    col1, col2 = st.columns(2)

    # SL IMTP - Unilateral
    with col1:
        slimtp_df = sport_df[sport_df['testType'].str.contains('SLIMTP', case=False, na=False)]
        if not slimtp_df.empty:
            left_col = None
            right_col = None
            for col in slimtp_df.columns:
                if 'Left' in col and ('Force' in col or 'Peak' in col):
                    left_col = col
                elif 'Right' in col and ('Force' in col or 'Peak' in col):
                    right_col = col

            if left_col and right_col and 'Name' in slimtp_df.columns and SNC_CHARTS_AVAILABLE:
                latest_slimtp = slimtp_df.groupby('Name').agg({left_col: 'last', right_col: 'last'}).reset_index()
                fig = create_ranked_side_by_side_chart(
                    latest_slimtp, left_col, right_col,
                    "Relative Peak Force", "N/kg",
                    title="SL IMTP - L/R"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("SL IMTP data not available")
        else:
            st.info("SL IMTP data not available")

    # SL CMJ - Unilateral
    with col2:
        slcmj_df = sport_df[sport_df['testType'].str.contains('SLCMJ|SLCMRJ|SL.*CMJ', case=False, na=False)]
        if not slcmj_df.empty:
            # Try to find left/right power columns
            left_col = None
            right_col = None
            for col in slcmj_df.columns:
                if 'Left' in col and ('Power' in col or 'Peak' in col):
                    left_col = col
                elif 'Right' in col and ('Power' in col or 'Peak' in col):
                    right_col = col

            if left_col and right_col and 'Name' in slcmj_df.columns and SNC_CHARTS_AVAILABLE:
                latest_slcmj = slcmj_df.groupby('Name').agg({left_col: 'last', right_col: 'last'}).reset_index()
                fig = create_ranked_side_by_side_chart(
                    latest_slcmj, left_col, right_col,
                    "Relative Peak Power", "W/kg",
                    title="SL CMJ - L/R Peak Power"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("SL CMJ data not available")
        else:
            st.info("SL CMJ data not available")

    st.markdown("---")

    # =========================================================================
    # SECTION 2: Upper Body Strength & Power (if data available)
    # =========================================================================
    st.markdown("### Upper Body Strength & Power")

    # Try to load S&C data from session state or CSV
    sc_upper_body_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sc_upper_body.csv')
    sc_df = pd.DataFrame()

    if hasattr(st, 'session_state') and 'sc_upper_body' in st.session_state:
        sc_df = st.session_state.sc_upper_body
    elif os.path.exists(sc_upper_body_path):
        try:
            sc_df = pd.read_csv(sc_upper_body_path)
            if 'date' in sc_df.columns:
                sc_df['date'] = pd.to_datetime(sc_df['date'], errors='coerce')
        except Exception:
            pass

    # Filter S&C data by sport athletes (use Name or athlete column)
    if not sc_df.empty and 'Name' in sport_df.columns:
        sport_athletes = sport_df['Name'].dropna().unique().tolist()
        # Try to match by athlete column first, then Name
        if 'athlete' in sc_df.columns:
            sc_df = sc_df[sc_df['athlete'].isin(sport_athletes)]
        elif 'Name' in sc_df.columns:
            sc_df = sc_df[sc_df['Name'].isin(sport_athletes)]

    # Row 1: Bench Press and Pull Up
    col1, col2 = st.columns(2)

    with col1:
        # Bench Press - check VALD data first, then S&C data
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
        elif not sc_df.empty and 'exercise' in sc_df.columns:
            # Show S&C manual entry data as bar chart
            bench_sc = sc_df[sc_df['exercise'].str.contains('Bench', case=False, na=False)]
            if not bench_sc.empty:
                # Get best 1RM per athlete
                best_bench = bench_sc.groupby('athlete').agg({
                    'estimated_1rm': 'max',
                    'weight_kg': 'max',
                    'date': 'max'
                }).reset_index().sort_values('estimated_1rm', ascending=True)

                # Create horizontal bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=best_bench['athlete'],
                    x=best_bench['estimated_1rm'],
                    orientation='h',
                    marker_color=TEAL_PRIMARY,
                    text=[f"{v:.0f}kg" for v in best_bench['estimated_1rm']],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Bench Press - Est. 1RM (Manual Entry)",
                    xaxis_title="Estimated 1RM (kg)",
                    yaxis_title="",
                    height=max(250, min(350, len(best_bench) * 35)),
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Bench Press data not available - Use ✏️ Data Entry tab")
        else:
            st.info("Bench Press data not available - Use ✏️ Data Entry tab")

    with col2:
        # Pull Up - check VALD data first, then S&C data
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
        elif not sc_df.empty and 'exercise' in sc_df.columns:
            # Show S&C manual entry data as bar chart
            pullup_sc = sc_df[sc_df['exercise'].str.contains('Pull Up|Pullup|Pull-up|Chin', case=False, na=False)]
            if not pullup_sc.empty:
                # Get best reps per athlete
                best_pullup = pullup_sc.groupby('athlete').agg({
                    'reps': 'max',
                    'weight_kg': 'max',
                    'date': 'max'
                }).reset_index().sort_values('reps', ascending=True)

                # Create horizontal bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=best_pullup['athlete'],
                    x=best_pullup['reps'],
                    orientation='h',
                    marker_color=TEAL_PRIMARY,
                    text=[f"{int(v)} reps" for v in best_pullup['reps']],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Pull Up - Max Reps (Manual Entry)",
                    xaxis_title="Reps",
                    yaxis_title="",
                    height=max(250, min(350, len(best_pullup) * 35)),
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Pull Up data not available - Use ✏️ Data Entry tab")
        else:
            st.info("Pull Up data not available - Use ✏️ Data Entry tab")

    # Row 2: Plyo Push Up (PPU test type) - Same chart as Canvas Group View
    col1, col2 = st.columns(2)
    with col1:
        # Plyo Push Up - PPU is the exact test type code
        plyo_df = sport_df[sport_df['testType'] == 'PPU'].copy() if 'testType' in sport_df.columns else pd.DataFrame()

        if not plyo_df.empty and 'Name' in plyo_df.columns:
            # Use same metric priority as Canvas view
            metric_col = None
            metric_name = 'Pushup Height'
            metric_unit = 'cm'

            metric_options = [
                ('PUSHUP_HEIGHT', 'Pushup Height', 'cm'),
                ('FLIGHT_TIME', 'Flight Time', 's'),
                ('BODYMASS_RELATIVE_TAKEOFF_POWER', 'Relative Peak Power', 'W/kg'),
                ('PEAK_TAKEOFF_FORCE', 'Peak Takeoff Force', 'N'),
            ]

            for col, name, unit in metric_options:
                if col in plyo_df.columns and plyo_df[col].notna().sum() > 0:
                    metric_col = col
                    metric_name = name
                    metric_unit = unit
                    break

            if metric_col:
                # Get latest test per athlete
                if 'recordedDateUtc' in plyo_df.columns:
                    latest_ppu = plyo_df.sort_values('recordedDateUtc').groupby('Name').last().reset_index()
                else:
                    latest_ppu = plyo_df.groupby('Name').last().reset_index()

                # Use same create_ranked_bar_chart as Canvas
                if SNC_CHARTS_AVAILABLE:
                    fig = create_ranked_bar_chart(
                        latest_ppu,
                        metric_col,
                        metric_name,
                        metric_unit,
                        None,  # No benchmark
                        f'Plyo Pushup - {metric_name}'
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback to simple bar chart
                    latest_ppu = latest_ppu.sort_values(metric_col, ascending=True)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=latest_ppu['Name'],
                        x=latest_ppu[metric_col],
                        orientation='h',
                        marker_color=TEAL_PRIMARY,
                        text=[f"{v:.1f} {metric_unit}" for v in latest_ppu[metric_col]],
                        textposition='outside'
                    ))
                    squad_avg = latest_ppu[metric_col].mean()
                    fig.add_vline(x=squad_avg, line_dash="dash", line_color=GOLD_ACCENT,
                                 annotation_text=f"Avg: {squad_avg:.1f}")
                    fig.update_layout(
                        title=f"Plyo Pushup - {metric_name}",
                        xaxis_title=metric_unit, yaxis_title="",
                        plot_bgcolor='white',
                        height=max(250, min(400, len(latest_ppu) * 35)),
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Plyo Pushup metrics not found in data")
        else:
            st.info("Plyo Pushup (PPU) data not available")

    # Row 3: Grip Strength (DynaMo) - Side by Side L/R
    col1, col2 = st.columns(2)

    # Load DynaMo data
    dynamo_df = pd.DataFrame()
    dynamo_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dynamo_allsports_with_athletes.csv'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'vald-data', 'data', 'dynamo_allsports_with_athletes.csv'),
    ]
    for path in dynamo_paths:
        if os.path.exists(path):
            try:
                dynamo_df = pd.read_csv(path)
                break
            except Exception:
                pass

    # Filter DynaMo for grip tests and sport
    if not dynamo_df.empty:
        # Filter for GripSqueeze movement
        if 'movement' in dynamo_df.columns:
            dynamo_df = dynamo_df[dynamo_df['movement'] == 'GripSqueeze']

        # Filter by sport (flexible matching, not exact)
        if sport and sport != "All Sports" and 'athlete_sport' in dynamo_df.columns:
            sport_pattern = sport.split()[0]  # First word of sport
            sport_mask = dynamo_df['athlete_sport'].str.contains(sport_pattern, case=False, na=False)
            if sport_mask.any():
                dynamo_df = dynamo_df[sport_mask]

    with col1:
        if not dynamo_df.empty and 'maxForceNewtons' in dynamo_df.columns and 'full_name' in dynamo_df.columns:
            # Filter for Left hand using laterality column - match 'LEFT', 'LeftSide', 'LeftThenRight', etc.
            left_df = dynamo_df[dynamo_df['laterality'].str.contains('Left', case=False, na=False)] if 'laterality' in dynamo_df.columns else pd.DataFrame()

            if not left_df.empty:
                latest = left_df.groupby('full_name')['maxForceNewtons'].last().reset_index()
                latest = latest.sort_values('maxForceNewtons', ascending=True)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=latest['full_name'],
                    x=latest['maxForceNewtons'],
                    orientation='h',
                    marker_color=TEAL_PRIMARY,
                    text=[f"{v:.0f} N" for v in latest['maxForceNewtons']],
                    textposition='outside'
                ))
                squad_avg = latest['maxForceNewtons'].mean()
                fig.add_vline(x=squad_avg, line_dash="dash", line_color=GOLD_ACCENT,
                             annotation_text=f"Avg: {squad_avg:.0f}N")
                fig.update_layout(
                    title="DynaMo Grip - Left Hand",
                    xaxis_title="Peak Force (N)",
                    yaxis_title="",
                    plot_bgcolor='white',
                    height=max(250, min(400, len(latest) * 35)),
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Left grip data in DynaMo")
        else:
            st.info("DynaMo grip data not available")

    with col2:
        if not dynamo_df.empty and 'maxForceNewtons' in dynamo_df.columns and 'full_name' in dynamo_df.columns:
            # Filter for Right hand using laterality column - match 'RIGHT', 'RightSide', 'RightThenLeft', etc.
            right_df = dynamo_df[dynamo_df['laterality'].str.contains('Right', case=False, na=False)] if 'laterality' in dynamo_df.columns else pd.DataFrame()

            if not right_df.empty:
                latest = right_df.groupby('full_name')['maxForceNewtons'].last().reset_index()
                latest = latest.sort_values('maxForceNewtons', ascending=True)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=latest['full_name'],
                    x=latest['maxForceNewtons'],
                    orientation='h',
                    marker_color=TEAL_PRIMARY,
                    text=[f"{v:.0f} N" for v in latest['maxForceNewtons']],
                    textposition='outside'
                ))
                squad_avg = latest['maxForceNewtons'].mean()
                fig.add_vline(x=squad_avg, line_dash="dash", line_color=GOLD_ACCENT,
                             annotation_text=f"Avg: {squad_avg:.0f}N")
                fig.update_layout(
                    title="DynaMo Grip - Right Hand",
                    xaxis_title="Peak Force (N)",
                    yaxis_title="",
                    plot_bgcolor='white',
                    height=max(250, min(400, len(latest) * 35)),
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Right grip data in DynaMo")
        else:
            st.info("DynaMo grip data not available")

    st.markdown("---")

    # =========================================================================
    # SECTION 3: Shoulder Health (ForceFrame) - Stacked Quadrant Charts
    # =========================================================================
    st.markdown("### Shoulder Health")

    col1, col2 = st.columns(2)

    with col1:
        if forceframe_df is not None and not forceframe_df.empty:
            # Filter for Shoulder IR/ER tests specifically
            shoulder_irer_df = forceframe_df[
                forceframe_df['testTypeName'].str.contains('Shoulder IR/ER', case=False, na=False)
            ] if 'testTypeName' in forceframe_df.columns else pd.DataFrame()

            if not shoulder_irer_df.empty and 'Name' in shoulder_irer_df.columns and SNC_CHARTS_AVAILABLE:
                # ForceFrame uses inner/outer columns for IR/ER
                # Inner = Internal Rotation, Outer = External Rotation (typically)
                inner_col = 'innerLeftMaxForce' if 'innerLeftMaxForce' in shoulder_irer_df.columns else None
                outer_col = 'outerLeftMaxForce' if 'outerLeftMaxForce' in shoulder_irer_df.columns else None

                if inner_col and outer_col:
                    # Get latest per athlete, averaging left and right
                    shoulder_irer_df['IR_avg'] = (shoulder_irer_df['innerLeftMaxForce'].fillna(0) + shoulder_irer_df.get('innerRightMaxForce', shoulder_irer_df['innerLeftMaxForce']).fillna(0)) / 2
                    shoulder_irer_df['ER_avg'] = (shoulder_irer_df['outerLeftMaxForce'].fillna(0) + shoulder_irer_df.get('outerRightMaxForce', shoulder_irer_df['outerLeftMaxForce']).fillna(0)) / 2

                    latest_shoulder = shoulder_irer_df.groupby('Name').agg({'IR_avg': 'last', 'ER_avg': 'last'}).reset_index()
                    metric_cols = {'IR': 'IR_avg', 'ER': 'ER_avg'}
                    fig = create_stacked_quadrant_chart(
                        latest_shoulder, metric_cols,
                        "Shoulder Rotation", "N",
                        title="Shoulder IR/ER Profile",
                        vertical=True
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Shoulder IR/ER force columns not found")
            elif not shoulder_irer_df.empty:
                st.info("Shoulder IR/ER data available - enable S&C charts for visualization")
            else:
                st.info("No Shoulder IR/ER tests found in ForceFrame data")
        else:
            st.info("ForceFrame data not available")

    with col2:
        if forceframe_df is not None and not forceframe_df.empty:
            # Look for ASH test or abduction/adduction
            ash_df = forceframe_df[
                forceframe_df['testTypeName'].str.contains('ASH|Abduct', case=False, na=False, regex=True)
            ] if 'testTypeName' in forceframe_df.columns else pd.DataFrame()

            if not ash_df.empty and 'Name' in ash_df.columns:
                st.markdown("**Shoulder ASH**")
                st.info("ForceFrame shoulder ASH data - visualization pending")
            else:
                st.info("Shoulder ASH data not available")
        else:
            st.info("ForceFrame data not available")

    st.markdown("---")

    # =========================================================================
    # SECTION 4: Hip Health (ForceFrame) - Stacked Quadrant Charts
    # =========================================================================
    st.markdown("### Hip Health")

    col1, col2 = st.columns(2)

    with col1:
        if forceframe_df is not None and not forceframe_df.empty:
            # Filter for Hip AD/AB tests (Adduction/Abduction)
            hip_adab_df = forceframe_df[
                forceframe_df['testTypeName'].str.contains('Hip AD/AB', case=False, na=False)
            ] if 'testTypeName' in forceframe_df.columns else pd.DataFrame()

            if not hip_adab_df.empty and 'Name' in hip_adab_df.columns and SNC_CHARTS_AVAILABLE:
                # ForceFrame uses inner/outer columns
                # Inner = Adduction, Outer = Abduction (typically)
                inner_col = 'innerLeftMaxForce' if 'innerLeftMaxForce' in hip_adab_df.columns else None
                outer_col = 'outerLeftMaxForce' if 'outerLeftMaxForce' in hip_adab_df.columns else None

                if inner_col and outer_col:
                    # Get latest per athlete, averaging left and right
                    hip_adab_df['ADD_avg'] = (hip_adab_df['innerLeftMaxForce'].fillna(0) + hip_adab_df.get('innerRightMaxForce', hip_adab_df['innerLeftMaxForce']).fillna(0)) / 2
                    hip_adab_df['ABD_avg'] = (hip_adab_df['outerLeftMaxForce'].fillna(0) + hip_adab_df.get('outerRightMaxForce', hip_adab_df['outerLeftMaxForce']).fillna(0)) / 2

                    latest_hip = hip_adab_df.groupby('Name').agg({'ADD_avg': 'last', 'ABD_avg': 'last'}).reset_index()
                    metric_cols = {'ADD': 'ADD_avg', 'ABD': 'ABD_avg'}
                    fig = create_stacked_quadrant_chart(
                        latest_hip, metric_cols,
                        "Hip Strength", "N",
                        title="Hip ADD/ABD Profile",
                        vertical=True
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Hip AD/AB force columns not found")
            elif not hip_adab_df.empty:
                st.info("Hip AD/AB data available - enable S&C charts for visualization")
            else:
                st.info("No Hip AD/AB tests found in ForceFrame data")
        else:
            st.info("ForceFrame data not available")

    with col2:
        # Hip IR/ER if available
        if forceframe_df is not None and not forceframe_df.empty:
            hip_irer_df = forceframe_df[
                forceframe_df['testTypeName'].str.contains('Hip IR/ER', case=False, na=False)
            ] if 'testTypeName' in forceframe_df.columns else pd.DataFrame()

            if not hip_irer_df.empty and 'Name' in hip_irer_df.columns and SNC_CHARTS_AVAILABLE:
                # Use inner/outer for IR/ER
                inner_col = 'innerLeftMaxForce' if 'innerLeftMaxForce' in hip_irer_df.columns else None
                outer_col = 'outerLeftMaxForce' if 'outerLeftMaxForce' in hip_irer_df.columns else None

                if inner_col and outer_col:
                    hip_irer_df['IR_avg'] = (hip_irer_df['innerLeftMaxForce'].fillna(0) + hip_irer_df.get('innerRightMaxForce', hip_irer_df['innerLeftMaxForce']).fillna(0)) / 2
                    hip_irer_df['ER_avg'] = (hip_irer_df['outerLeftMaxForce'].fillna(0) + hip_irer_df.get('outerRightMaxForce', hip_irer_df['outerLeftMaxForce']).fillna(0)) / 2

                    latest_hip_irer = hip_irer_df.groupby('Name').agg({'IR_avg': 'last', 'ER_avg': 'last'}).reset_index()
                    metric_cols = {'IR': 'IR_avg', 'ER': 'ER_avg'}
                    fig = create_stacked_quadrant_chart(
                        latest_hip_irer, metric_cols,
                        "Hip Rotation", "N",
                        title="Hip IR/ER Profile",
                        vertical=True
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Hip IR/ER force columns not found")
            elif not hip_irer_df.empty:
                st.info("Hip IR/ER data available - enable S&C charts")
            else:
                st.info("No Hip IR/ER tests found in ForceFrame data")
        else:
            st.info("ForceFrame data not available")

    st.markdown("---")

    # =========================================================================
    # SECTION 5: Strength RM (Manual Entry Data) - Tabbed View
    # =========================================================================
    st.markdown("### Strength RM")
    st.caption("Manual entry S&C data (Back Squat, Bench Press, Deadlift, etc.)")

    # Load S&C manual entry data
    sc_upper_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sc_upper_body.csv')
    sc_lower_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sc_lower_body.csv')
    sc_df = pd.DataFrame()

    # Try to load from session state first, then CSV
    if hasattr(st, 'session_state') and 'sc_upper_body' in st.session_state and not st.session_state.sc_upper_body.empty:
        sc_df = st.session_state.sc_upper_body.copy()
    elif os.path.exists(sc_upper_path):
        try:
            sc_df = pd.read_csv(sc_upper_path)
            if 'date' in sc_df.columns:
                sc_df['date'] = pd.to_datetime(sc_df['date'], errors='coerce')
        except Exception:
            pass

    # Also try to load lower body data
    if hasattr(st, 'session_state') and 'sc_lower_body' in st.session_state and not st.session_state.sc_lower_body.empty:
        sc_lower = st.session_state.sc_lower_body.copy()
        if not sc_df.empty:
            sc_df = pd.concat([sc_df, sc_lower], ignore_index=True)
        else:
            sc_df = sc_lower
    elif os.path.exists(sc_lower_path):
        try:
            sc_lower = pd.read_csv(sc_lower_path)
            if 'date' in sc_lower.columns:
                sc_lower['date'] = pd.to_datetime(sc_lower['date'], errors='coerce')
            if not sc_df.empty:
                sc_df = pd.concat([sc_df, sc_lower], ignore_index=True)
            else:
                sc_df = sc_lower
        except Exception:
            pass

    if not sc_df.empty and 'athlete' in sc_df.columns:
        # Filter S&C data to athletes in the current sport group (skip for "All Sports")
        if sport != "All Sports":
            sport_athletes = sport_df['Name'].dropna().unique().tolist() if 'Name' in sport_df.columns else []
            if sport_athletes:
                sc_df_filtered = sc_df[sc_df['athlete'].isin(sport_athletes)]
                if not sc_df_filtered.empty:
                    sc_df = sc_df_filtered

        # Tab view like Canvas
        # Use selectbox + session state for persistence
        strength_tab_options = ["👥 Group View", "🏃 Individual View"]

        if 'active_strength_view_tab' not in st.session_state:
            st.session_state.active_strength_view_tab = strength_tab_options[0]

        current_strength_idx = 0
        if st.session_state.active_strength_view_tab in strength_tab_options:
            current_strength_idx = strength_tab_options.index(st.session_state.active_strength_view_tab)

        selected_strength_view = st.selectbox(
            "View:",
            strength_tab_options,
            index=current_strength_idx,
            key="strength_view_selector_classic"
        )
        st.session_state.active_strength_view_tab = selected_strength_view

        if selected_strength_view == "👥 Group View":
            # Group View - Ranked bar chart for each exercise
            if 'exercise' in sc_df.columns and 'estimated_1rm' in sc_df.columns:
                exercises = sorted(sc_df['exercise'].dropna().unique().tolist())
                if exercises:
                    exercise_select = st.selectbox("Select Exercise:", exercises, key="strength_group_exercise")

                    exercise_df = sc_df[sc_df['exercise'] == exercise_select].copy()
                    if not exercise_df.empty:
                        # Get latest per athlete
                        latest = exercise_df.sort_values('date').groupby('athlete')['estimated_1rm'].last().reset_index()
                        latest = latest.sort_values('estimated_1rm', ascending=True)

                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            y=latest['athlete'],
                            x=latest['estimated_1rm'],
                            orientation='h',
                            marker_color=TEAL_PRIMARY,
                            text=[f"{v:.0f} kg" for v in latest['estimated_1rm']],
                            textposition='outside'
                        ))
                        squad_avg = latest['estimated_1rm'].mean()
                        fig.add_vline(x=squad_avg, line_dash="dash", line_color=GOLD_ACCENT,
                                     annotation_text=f"Avg: {squad_avg:.0f}kg")
                        fig.update_layout(
                            title=f"{exercise_select} - Estimated 1RM",
                            xaxis_title="Estimated 1RM (kg)",
                            yaxis_title="",
                            plot_bgcolor='white',
                            height=max(250, min(450, len(latest) * 35)),
                            margin=dict(l=10, r=10, t=40, b=10)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No exercises found in data.")
            else:
                st.info("Required columns not found in S&C data.")

        elif selected_strength_view == "🏃 Individual View":
            # Individual View - Multi-line progression
            sc_athletes = sorted(sc_df['athlete'].dropna().unique().tolist())

            if sc_athletes:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    selected_sc_athlete = st.selectbox(
                        "Select Athlete:",
                        sc_athletes,
                        key="group_v1_strength_athlete"
                    )

                with col2:
                    if 'exercise' in sc_df.columns:
                        available_exercises = sorted(sc_df['exercise'].dropna().unique().tolist())
                        default_exercises = [e for e in ['Back Squat', 'Bench Press', 'Deadlift'] if e in available_exercises]
                        if not default_exercises and available_exercises:
                            default_exercises = available_exercises[:3]

                        selected_exercises = st.multiselect(
                            "Select Exercises:",
                            available_exercises,
                            default=default_exercises,
                            key="group_v1_strength_exercises"
                        )
                    else:
                        selected_exercises = []

                with col3:
                    bodyweight = st.number_input(
                        "Bodyweight (kg):",
                        min_value=40.0, max_value=200.0, value=80.0, step=1.0,
                        key="group_v1_bodyweight"
                    )

                if selected_sc_athlete and selected_exercises and SNC_CHARTS_AVAILABLE:
                    fig = create_multi_line_strength_chart(
                        sc_df,
                        selected_sc_athlete,
                        selected_exercises,
                        bodyweight=bodyweight,
                        title=f"{selected_sc_athlete} - Strength RM Progression"
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No strength data found for {selected_sc_athlete}")
                elif selected_sc_athlete and selected_exercises:
                    st.info("Select athlete and exercises to view progression")
            else:
                st.info(f"No S&C data available for {sport} athletes")
    else:
        st.info("No S&C manual entry data available. Use ✏️ Data Entry tab to add strength records.")

    st.markdown("---")

    # =========================================================================
    # SECTION 6: Individual Athlete Trends (Line charts with Squad Avg)
    # =========================================================================
    st.markdown("### Individual Athlete Trends")
    st.caption("Select athletes to view their performance over time compared to squad average")

    # Get all athletes in the sport group
    all_athletes = sorted(sport_df['Name'].dropna().unique().tolist()) if 'Name' in sport_df.columns else []

    if not all_athletes:
        st.info("No athlete data available for individual trends")
    else:
        # Athlete multiselect - allow up to 4 athletes for comparison
        selected_athletes = st.multiselect(
            "Select Athletes (max 4):",
            all_athletes,
            default=all_athletes[:min(2, len(all_athletes))],  # Default to first 2
            max_selections=4,
            key="group_v1_individual_athletes"
        )

        if selected_athletes and SNC_CHARTS_AVAILABLE:
            # Row 1: IMTP Trend and CMJ Power Trend
            col1, col2 = st.columns(2)

            with col1:
                # IMTP Trend Line Chart
                metric_col = get_metric_column(sport_df, 'peak_force')
                if metric_col:
                    imtp_df = sport_df[sport_df['testType'].str.contains('IMTP|ISOT|Isometric', case=False, na=False)]
                    if not imtp_df.empty:
                        fig = create_individual_line_chart(
                            imtp_df, selected_athletes, metric_col,
                            "Relative Peak Force", "N/kg",
                            show_squad_avg=True,
                            title="IMTP - Individual Trends"
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No IMTP trend data")
                else:
                    st.info("IMTP metric not found")

            with col2:
                # CMJ Power Trend Line Chart
                metric_col = get_metric_column(sport_df, 'relative_power')
                if metric_col:
                    cmj_df = sport_df[sport_df['testType'].str.contains('CMJ|Counter', case=False, na=False)]
                    if not cmj_df.empty:
                        fig = create_individual_line_chart(
                            cmj_df, selected_athletes, metric_col,
                            "Relative Peak Power", "W/kg",
                            show_squad_avg=True,
                            title="CMJ - Individual Trends"
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No CMJ trend data")
                else:
                    st.info("CMJ metric not found")

            # Row 2: CMJ Height Trend and 10:5 Hop Trend
            col1, col2 = st.columns(2)

            with col1:
                # CMJ Height Trend
                metric_col = get_metric_column(sport_df, 'cmj_height')
                if metric_col:
                    cmj_df = sport_df[sport_df['testType'].str.contains('CMJ|Counter', case=False, na=False)]
                    if not cmj_df.empty:
                        # Create a copy and convert if needed
                        cmj_plot = cmj_df.copy()
                        if cmj_plot[metric_col].max() < 1:
                            cmj_plot[metric_col] = cmj_plot[metric_col] * 100
                        fig = create_individual_line_chart(
                            cmj_plot, selected_athletes, metric_col,
                            "Jump Height", "cm",
                            show_squad_avg=True,
                            title="CMJ Height - Individual Trends"
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No CMJ height trend data")
                else:
                    st.info("CMJ height metric not found")

            with col2:
                # 10:5 Hop RSI Trend
                metric_col = get_metric_column(sport_df, 'rsi')
                if metric_col:
                    hop_df = sport_df[sport_df['testType'].str.contains('RSHIP|DJ|SLDJ|Hop|Drop', case=False, na=False)]
                    # Apply RSI unit conversion if needed
                    hop_df = apply_metric_conversion(hop_df, 'rsi', metric_col)
                    if not hop_df.empty:
                        fig = create_individual_line_chart(
                            hop_df, selected_athletes, metric_col,
                            "RSI", "",
                            show_squad_avg=True,
                            title="10:5 Hop - Individual Trends"
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No hop test trend data")

            # Row 3: NordBord Trend (if available)
            if nordbord_df is not None and not nordbord_df.empty:
                col1, col2 = st.columns(2)

                with col1:
                    nb_df = nordbord_df.copy()
                    if sport and sport != "All Sports" and 'athlete_sport' in nb_df.columns:
                        sport_mask = nb_df['athlete_sport'].str.contains(sport.split()[0], case=False, na=False)
                        if sport_mask.any():
                            nb_df = nb_df[sport_mask]

                    left_col, right_col = get_nordbord_force_columns(nb_df)
                    if left_col and right_col and 'Name' in nb_df.columns:
                        # Calculate average force for trend
                        nb_df['avg_hamstring_force'] = (nb_df[left_col] + nb_df[right_col]) / 2
                        fig = create_individual_line_chart(
                            nb_df, selected_athletes, 'avg_hamstring_force',
                            "Avg Hamstring Force", "N",
                            show_squad_avg=True,
                            title="NordBord - Individual Trends"
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

        elif selected_athletes:
            st.warning("Individual line charts require S&C Diagnostics module")
        else:
            st.info("Select athletes above to view individual trends")


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
                rsi_val = latest[metric_col]
                # Apply RSI conversion if value is too high (likely *100)
                if rsi_val > 10:
                    rsi_val = rsi_val * 0.01
                st.metric("RSI", f"{rsi_val:.2f}")

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
            # Apply RSI unit conversion if needed
            hop_df = apply_metric_conversion(hop_df, 'rsi', metric_col)
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

    # Plyo Pushup trend (if PPU data available)
    ppu_df = athlete_df[athlete_df['testType'] == 'PPU'].copy() if 'testType' in athlete_df.columns else pd.DataFrame()
    if not ppu_df.empty and date_col:
        st.markdown("### Plyo Pushup (Upper Body Power)")
        metric_col = None
        metric_name = 'Pushup Height'
        metric_unit = 'cm'
        for col, name, unit in [('PUSHUP_HEIGHT', 'Pushup Height', 'cm'), ('FLIGHT_TIME', 'Flight Time', 's')]:
            if col in ppu_df.columns and ppu_df[col].notna().sum() > 0:
                metric_col = col
                metric_name = name
                metric_unit = unit
                break
        if metric_col:
            fig = create_trend_chart(ppu_df, metric_col, date_col, benchmarks, metric_name, 'ppu', metric_unit, athlete_name)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # DynaMo Grip trends (if data available)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    dynamo_paths = [
        os.path.join(data_dir, 'dynamo_allsports_with_athletes.csv'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'vald-data', 'data', 'dynamo_allsports_with_athletes.csv'),
    ]
    for path in dynamo_paths:
        if os.path.exists(path):
            try:
                dynamo_df = pd.read_csv(path)
                # Filter for GripSqueeze and this athlete
                if 'movement' in dynamo_df.columns:
                    dynamo_df = dynamo_df[dynamo_df['movement'] == 'GripSqueeze']
                name_col = 'full_name' if 'full_name' in dynamo_df.columns else 'Name'
                athlete_grip = dynamo_df[dynamo_df[name_col] == athlete_name].copy() if name_col in dynamo_df.columns else pd.DataFrame()

                if not athlete_grip.empty and 'maxForceNewtons' in athlete_grip.columns and 'laterality' in athlete_grip.columns:
                    st.markdown("### DynaMo Grip Strength")
                    # Parse date
                    grip_date_col = None
                    for col in ['testDateUtc', 'recordedDateUtc', 'modifiedDateUtc']:
                        if col in athlete_grip.columns:
                            grip_date_col = col
                            athlete_grip[col] = pd.to_datetime(athlete_grip[col], errors='coerce')
                            break

                    if grip_date_col:
                        col1, col2 = st.columns(2)
                        with col1:
                            # Match 'LEFT', 'LeftSide', 'LeftThenRight', etc.
                            left_grip = athlete_grip[athlete_grip['laterality'].str.contains('Left', case=False, na=False)].sort_values(grip_date_col)
                            if not left_grip.empty:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=left_grip[grip_date_col], y=left_grip['maxForceNewtons'],
                                                        mode='lines+markers', name='Left Grip', line=dict(color=TEAL_PRIMARY)))
                                fig.update_layout(title='Left Grip Strength', xaxis_title='Date', yaxis_title='Force (N)',
                                                 plot_bgcolor='white', paper_bgcolor='white')
                                st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            # Match 'RIGHT', 'RightSide', 'RightThenLeft', etc.
                            right_grip = athlete_grip[athlete_grip['laterality'].str.contains('Right', case=False, na=False)].sort_values(grip_date_col)
                            if not right_grip.empty:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=right_grip[grip_date_col], y=right_grip['maxForceNewtons'],
                                                        mode='lines+markers', name='Right Grip', line=dict(color=GOLD_ACCENT)))
                                fig.update_layout(title='Right Grip Strength', xaxis_title='Date', yaxis_title='Force (N)',
                                                 plot_bgcolor='white', paper_bgcolor='white')
                                st.plotly_chart(fig, use_container_width=True)
                break
            except Exception:
                pass

    # Strength RM trends (Manual Entry)
    lower_body_path = os.path.join(data_dir, 'sc_lower_body.csv')
    upper_body_path = os.path.join(data_dir, 'sc_upper_body.csv')
    strength_dfs = []
    if os.path.exists(lower_body_path):
        try:
            lb_df = pd.read_csv(lower_body_path)
            strength_dfs.append(lb_df)
        except Exception:
            pass
    if os.path.exists(upper_body_path):
        try:
            ub_df = pd.read_csv(upper_body_path)
            strength_dfs.append(ub_df)
        except Exception:
            pass

    if strength_dfs:
        strength_df = pd.concat(strength_dfs, ignore_index=True)
        if 'athlete' in strength_df.columns:
            athlete_str = strength_df[strength_df['athlete'] == athlete_name].copy()
            if not athlete_str.empty and 'date' in athlete_str.columns and 'estimated_1rm' in athlete_str.columns:
                st.markdown("### Strength RM (Manual Entry)")
                athlete_str['date'] = pd.to_datetime(athlete_str['date'], errors='coerce')

                exercises = athlete_str['exercise'].unique() if 'exercise' in athlete_str.columns else []
                if len(exercises) > 0:
                    fig = go.Figure()
                    colors = [TEAL_PRIMARY, GOLD_ACCENT, '#0077B6', '#2A8F5C', '#dc3545']
                    for i, ex in enumerate(exercises[:5]):
                        ex_data = athlete_str[athlete_str['exercise'] == ex].sort_values('date')
                        fig.add_trace(go.Scatter(x=ex_data['date'], y=ex_data['estimated_1rm'],
                                                mode='lines+markers', name=ex, line=dict(color=colors[i % len(colors)])))
                    fig.update_layout(title='Strength RM Progression', xaxis_title='Date', yaxis_title='Estimated 1RM (kg)',
                                     plot_bgcolor='white', paper_bgcolor='white')
                    st.plotly_chart(fig, use_container_width=True)

    # Broad Jump trend (Manual Entry)
    broad_jump_path = os.path.join(data_dir, 'broad_jump.csv')
    if os.path.exists(broad_jump_path):
        try:
            bj_df = pd.read_csv(broad_jump_path)
            if 'athlete' in bj_df.columns:
                athlete_bj = bj_df[bj_df['athlete'] == athlete_name].copy()
                if not athlete_bj.empty and 'date' in athlete_bj.columns and 'distance_cm' in athlete_bj.columns:
                    st.markdown("### Broad Jump (Manual Entry)")
                    athlete_bj['date'] = pd.to_datetime(athlete_bj['date'], errors='coerce')
                    athlete_bj = athlete_bj.sort_values('date')
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=athlete_bj['date'], y=athlete_bj['distance_cm'],
                                            mode='lines+markers', name='Distance', line=dict(color=TEAL_PRIMARY)))
                    # Add PB line
                    pb = athlete_bj['distance_cm'].max()
                    fig.add_hline(y=pb, line_dash="dash", line_color=GOLD_ACCENT, annotation_text=f"PB: {pb:.0f}cm")
                    fig.update_layout(title='Broad Jump Progression', xaxis_title='Date', yaxis_title='Distance (cm)',
                                     plot_bgcolor='white', paper_bgcolor='white')
                    st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    # ForceFrame trends (if data available)
    if forceframe_df is not None and not forceframe_df.empty and 'Name' in forceframe_df.columns:
        athlete_ff = forceframe_df[forceframe_df['Name'] == athlete_name].copy()
        if not athlete_ff.empty:
            st.markdown("### ForceFrame - Isometric Strength")
            test_col = 'testTypeName' if 'testTypeName' in athlete_ff.columns else 'testType'
            ff_date_col = None
            for col in ['testDateUtc', 'recordedDateUtc', 'modifiedDateUtc']:
                if col in athlete_ff.columns:
                    ff_date_col = col
                    athlete_ff[col] = pd.to_datetime(athlete_ff[col], errors='coerce')
                    break

            if ff_date_col and test_col in athlete_ff.columns:
                # Show Shoulder tests
                shoulder_ff = athlete_ff[athlete_ff[test_col].str.contains('Shoulder', case=False, na=False)]
                if not shoulder_ff.empty:
                    # Find force column
                    force_cols = [c for c in shoulder_ff.columns if 'MaxForce' in c or 'maxForce' in c]
                    if force_cols:
                        force_col = force_cols[0]
                        shoulder_ff = shoulder_ff.sort_values(ff_date_col)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=shoulder_ff[ff_date_col], y=shoulder_ff[force_col],
                                                mode='lines+markers', name='Shoulder Force', line=dict(color=TEAL_PRIMARY)))
                        fig.update_layout(title='Shoulder Strength Progression', xaxis_title='Date', yaxis_title='Force (N)',
                                         plot_bgcolor='white', paper_bgcolor='white')
                        st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # EXPORT SECTION - Download athlete data
    # =========================================================================
    st.markdown("---")
    with st.expander("📥 Export Athlete Data", expanded=False):
        st.markdown(f"### Download Data for {athlete_name}")

        # Create summary dataframe from athlete data
        summary_data = []

        # Add VALD test data
        if not athlete_df.empty:
            for _, row in athlete_df.iterrows():
                test_date = row.get(date_col, '')
                test_type = row.get('testType', 'Unknown')

                summary_row = {
                    'Date': str(test_date)[:10] if pd.notna(test_date) else '',
                    'Test Type': test_type,
                    'Source': 'VALD ForceDecks'
                }

                # Add available metrics
                metric_mappings = [
                    ('Jump Height (cm)', ['JUMP_HEIGHT_IMP_MOM', 'JUMP_HEIGHT_FLIGHT', 'jumpHeight']),
                    ('Peak Power (W/kg)', ['BODYMASS_RELATIVE_PEAK_POWER', 'BODYMASS_RELATIVE_CONCENTRIC_PEAK_POWER']),
                    ('Peak Force (N/kg)', ['BODYMASS_RELATIVE_PEAK_FORCE', 'BODYMASS_RELATIVE_CONCENTRIC_PEAK_FORCE']),
                    ('RSI', ['RSI_MODIFIED', 'RSI_MODIFIED_PP', 'RSI'])
                ]

                for metric_name, cols in metric_mappings:
                    for col in cols:
                        if col in row and pd.notna(row[col]):
                            summary_row[metric_name] = round(float(row[col]), 2)
                            break

                summary_data.append(summary_row)

        # Add NordBord data if available
        if nordbord_df is not None and not nordbord_df.empty and 'Name' in nordbord_df.columns:
            athlete_nb = nordbord_df[nordbord_df['Name'] == athlete_name]
            for _, row in athlete_nb.iterrows():
                nb_date_col = None
                for col in ['recordedDateUtc', 'testDateUtc', 'modifiedDateUtc']:
                    if col in row and pd.notna(row[col]):
                        nb_date_col = col
                        break

                summary_row = {
                    'Date': str(row.get(nb_date_col, ''))[:10] if nb_date_col else '',
                    'Test Type': 'NordBord',
                    'Source': 'VALD NordBord'
                }

                # Add left/right force if available
                left_col, right_col = get_nordbord_force_columns(nordbord_df)
                if left_col and left_col in row and pd.notna(row[left_col]):
                    summary_row['Left Hamstring (N)'] = round(float(row[left_col]), 0)
                if right_col and right_col in row and pd.notna(row[right_col]):
                    summary_row['Right Hamstring (N)'] = round(float(row[right_col]), 0)

                summary_data.append(summary_row)

        if summary_data:
            export_df = pd.DataFrame(summary_data)

            # Sort by date
            if 'Date' in export_df.columns:
                export_df['Date'] = pd.to_datetime(export_df['Date'], errors='coerce')
                export_df = export_df.sort_values('Date', ascending=False)
                export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d')

            # Display preview
            st.markdown("**Data Preview:**")
            st.dataframe(export_df.head(10), use_container_width=True, hide_index=True)

            # Download button
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Athlete Report (CSV)",
                data=csv_data,
                file_name=f"{athlete_name.replace(' ', '_').lower()}_performance_report.csv",
                mime="text/csv",
                key=f"individual_export_{athlete_name.replace(' ', '_')}"
            )
        else:
            st.info("No data available to export")


def create_group_report_v2(df: pd.DataFrame,
                           sport: str,
                           forceframe_df: pd.DataFrame = None,
                           nordbord_df: pd.DataFrame = None,
                           config: Dict = None):
    """
    Group Report V2 - Table format matching Strength Diagnostics layout.

    Same sections as Strength Diagnostics but with tables:
    - Lower Body Strength & Power
    - Upper Body Strength (ForceFrame)
    - Shoulder Health
    - Hip Health
    """
    # Ensure Name column exists (map from full_name if needed)
    if 'Name' not in df.columns and 'full_name' in df.columns:
        df = df.copy()
        df['Name'] = df['full_name']

    st.markdown(f"## {sport} - Table View")

    # =========================================================================
    # FILTERS - Gender, Benchmark Source, and Athlete selection
    # =========================================================================
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])

    with filter_col1:
        genders = ['All', 'Male', 'Female']
        selected_gender = st.selectbox(
            "Gender (for benchmarks):",
            genders,
            key="group_v2_gender_filter",
            help="Benchmarks adjust based on gender selection"
        )

    with filter_col2:
        benchmark_sources = ['VALD International', 'Saudi Population']
        selected_source = st.selectbox(
            "Benchmark Source:",
            benchmark_sources,
            key="group_v2_benchmark_source",
            help="VALD = International norms, Saudi = Custom population benchmarks"
        )

    # Get dynamic benchmarks
    gender_for_benchmarks = selected_gender.lower() if selected_gender != 'All' else 'male'
    source_key = 'Saudi' if selected_source == 'Saudi Population' else 'VALD'
    benchmarks = get_dynamic_benchmarks(gender_for_benchmarks, source_key)

    sport_benchmarks = get_sport_benchmarks(sport, config)
    for key, value in sport_benchmarks.items():
        if key not in benchmarks:
            benchmarks[key] = value

    # Filter data for the sport
    sport_df = df.copy()
    if sport and sport != "All Sports" and 'athlete_sport' in sport_df.columns:
        exact_mask = sport_df['athlete_sport'] == sport
        if exact_mask.any():
            sport_df = sport_df[exact_mask]
        else:
            sport_mask = sport_df['athlete_sport'].str.contains(sport.split()[0], case=False, na=False)
            if sport_mask.any():
                sport_df = sport_df[sport_mask]

    if selected_gender != 'All' and 'athlete_sex' in sport_df.columns:
        sport_df = sport_df[sport_df['athlete_sex'].str.lower() == selected_gender.lower()]

    if sport_df.empty:
        st.warning(f"No data available for {sport if sport else 'selected filters'} with current filters")
        return

    # Athlete filter
    if 'Name' in sport_df.columns:
        available_athletes = sorted(sport_df['Name'].dropna().unique().tolist())
        if available_athletes:
            with filter_col3:
                selected_athletes = st.multiselect(
                    "Filter Athletes:",
                    options=available_athletes,
                    default=[],
                    key="group_v2_athlete_filter",
                    help="Leave empty to show all athletes"
                )
            if selected_athletes:
                sport_df = sport_df[sport_df['Name'].isin(selected_athletes)]
                if forceframe_df is not None and not forceframe_df.empty and 'Name' in forceframe_df.columns:
                    forceframe_df = forceframe_df[forceframe_df['Name'].isin(selected_athletes)]
                if nordbord_df is not None and not nordbord_df.empty and 'Name' in nordbord_df.columns:
                    nordbord_df = nordbord_df[nordbord_df['Name'].isin(selected_athletes)]

    st.markdown("---")

    athletes = sport_df['Name'].dropna().unique() if 'Name' in sport_df.columns else []
    if len(athletes) == 0:
        st.warning("No athletes found in data")
        return

    # =========================================================================
    # SECTION 1: Lower Body Strength & Power (Table)
    # =========================================================================
    st.markdown("### Lower Body Strength & Power")

    lower_body_data = []
    for athlete in athletes:
        athlete_data = sport_df[sport_df['Name'] == athlete]
        row = {'Athlete': athlete}

        # IMTP - Relative Peak Force
        force_col = get_metric_column(athlete_data, 'peak_force')
        if force_col:
            imtp_df = athlete_data[athlete_data['testType'].str.contains('IMTP|ISOT|Isometric', case=False, na=False)]
            if not imtp_df.empty:
                val = imtp_df[force_col].dropna().iloc[-1] if not imtp_df[force_col].dropna().empty else None
                if val is not None:
                    row['IMTP (N/kg)'] = round(val, 1)
                    row['IMTP_status'] = _get_rag_status(val, benchmarks.get('peak_force', {}))

        # CMJ - Relative Peak Power
        power_col = get_metric_column(athlete_data, 'relative_power')
        if power_col:
            cmj_df = athlete_data[athlete_data['testType'].str.contains('CMJ|Counter', case=False, na=False)]
            if not cmj_df.empty:
                val = cmj_df[power_col].dropna().iloc[-1] if not cmj_df[power_col].dropna().empty else None
                if val is not None:
                    row['CMJ Power (W/kg)'] = round(val, 1)
                    row['CMJPower_status'] = _get_rag_status(val, benchmarks.get('peak_power', {}))

        # CMJ - Jump Height
        height_col = get_metric_column(athlete_data, 'cmj_height')
        if height_col:
            cmj_df = athlete_data[athlete_data['testType'].str.contains('CMJ|Counter', case=False, na=False)]
            if not cmj_df.empty:
                val = cmj_df[height_col].dropna().iloc[-1] if not cmj_df[height_col].dropna().empty else None
                if val is not None:
                    if val < 1:  # Convert m to cm
                        val = val * 100
                    row['CMJ Height (cm)'] = round(val, 1)
                    row['CMJHeight_status'] = _get_rag_status(val, benchmarks.get('cmj_height', {}))

        # 10:5 Hop - RSI
        rsi_col = get_metric_column(athlete_data, 'rsi')
        if rsi_col:
            hop_df = athlete_data[athlete_data['testType'].str.contains('RSHIP|DJ|SLDJ|Hop|Drop', case=False, na=False)]
            if not hop_df.empty:
                val = hop_df[rsi_col].dropna().iloc[-1] if not hop_df[rsi_col].dropna().empty else None
                if val is not None:
                    # Apply RSI conversion if value is too high (likely *100)
                    if val > 10:
                        val = val * 0.01
                    row['RSI'] = round(val, 2)
                    row['RSI_status'] = _get_rag_status(val, benchmarks.get('rsi', {}))

        # NordBord - L/R Hamstring
        if nordbord_df is not None and not nordbord_df.empty and 'Name' in nordbord_df.columns:
            nb_athlete = nordbord_df[nordbord_df['Name'] == athlete]
            if not nb_athlete.empty:
                left_col, right_col = get_nordbord_force_columns(nb_athlete)
                if left_col and right_col:
                    left = nb_athlete[left_col].dropna().iloc[-1] if not nb_athlete[left_col].dropna().empty else None
                    right = nb_athlete[right_col].dropna().iloc[-1] if not nb_athlete[right_col].dropna().empty else None
                    if left is not None and right is not None:
                        row['Ham L (N)'] = round(left, 0)
                        row['Ham R (N)'] = round(right, 0)
                        avg = (left + right) / 2
                        asym = abs((left - right) / avg * 100)
                        row['Ham Asym (%)'] = round(asym, 1)
                        row['Ham_status'] = _get_rag_status(avg, benchmarks.get('nordbord_force', {}))
                        row['HamAsym_status'] = _get_rag_status_inverse(asym, {'excellent': 5, 'good': 10})

        lower_body_data.append(row)

    if lower_body_data:
        _display_section_table(lower_body_data, 'Lower Body')

    st.markdown("---")

    # =========================================================================
    # SECTION 2: Upper Body Strength (ForceFrame)
    # =========================================================================
    st.markdown("### Upper Body Strength")

    if forceframe_df is not None and not forceframe_df.empty:
        ff_df = forceframe_df.copy()
        if sport and sport != "All Sports" and 'athlete_sport' in ff_df.columns:
            sport_mask = ff_df['athlete_sport'].str.contains(sport.split()[0], case=False, na=False)
            if sport_mask.any():
                ff_df = ff_df[sport_mask]

        if not ff_df.empty and 'Name' in ff_df.columns:
            upper_body_data = []
            ff_athletes = ff_df['Name'].dropna().unique()

            for athlete in ff_athletes:
                athlete_ff = ff_df[ff_df['Name'] == athlete]
                row = {'Athlete': athlete}

                # Get latest test by body region
                for region in ['Shoulder', 'Hip', 'Trunk', 'Knee']:
                    region_df = athlete_ff[athlete_ff['testType'].str.contains(region, case=False, na=False)]
                    if not region_df.empty:
                        # Find force columns
                        for col in region_df.columns:
                            if 'Peak' in col and 'Force' in col and 'Left' not in col and 'Right' not in col:
                                val = region_df[col].dropna().iloc[-1] if not region_df[col].dropna().empty else None
                                if val is not None:
                                    row[f'{region} (N)'] = round(val, 0)
                                break

                upper_body_data.append(row)

            if upper_body_data:
                upper_df = pd.DataFrame(upper_body_data)
                st.dataframe(upper_df, use_container_width=True, hide_index=True)
        else:
            st.info("No ForceFrame data for this sport")
    else:
        st.info("ForceFrame data not available")

    st.markdown("---")

    # =========================================================================
    # SECTION 3: Shoulder Health (ForceFrame data - testTypeName column)
    # =========================================================================
    st.markdown("### Shoulder Health")

    shoulder_data = []

    # Shoulder data comes from ForceFrame (testTypeName: 'Shoulder IR/ER', 'ASH')
    if forceframe_df is not None and not forceframe_df.empty and 'Name' in forceframe_df.columns:
        ff_df = forceframe_df.copy()
        # Filter for sport if possible
        if sport and sport != "All Sports" and 'athlete_sport' in ff_df.columns:
            sport_mask = ff_df['athlete_sport'].str.contains(sport.split()[0], case=False, na=False)
            if sport_mask.any():
                ff_df = ff_df[sport_mask]

        # Find shoulder tests using testTypeName (ForceFrame column)
        test_col = 'testTypeName' if 'testTypeName' in ff_df.columns else 'testType'
        if test_col in ff_df.columns:
            shoulder_ff = ff_df[ff_df[test_col].str.contains('Shoulder|IR.*ER|ASH|Abduct', case=False, na=False)]

            if not shoulder_ff.empty:
                for athlete in shoulder_ff['Name'].dropna().unique():
                    athlete_shoulder = shoulder_ff[shoulder_ff['Name'] == athlete]
                    row = {'Athlete': athlete}

                    # Look for Left/Right peak force columns
                    for col in athlete_shoulder.columns:
                        col_lower = col.lower()
                        if ('left' in col_lower or 'l ' in col_lower) and ('peak' in col_lower or 'force' in col_lower):
                            val = athlete_shoulder[col].dropna().iloc[-1] if not athlete_shoulder[col].dropna().empty else None
                            if val is not None and pd.notna(val):
                                row['L Shoulder (N)'] = round(float(val), 0)
                        elif ('right' in col_lower or 'r ' in col_lower) and ('peak' in col_lower or 'force' in col_lower):
                            val = athlete_shoulder[col].dropna().iloc[-1] if not athlete_shoulder[col].dropna().empty else None
                            if val is not None and pd.notna(val):
                                row['R Shoulder (N)'] = round(float(val), 0)

                    # Calculate asymmetry if both sides available
                    if 'L Shoulder (N)' in row and 'R Shoulder (N)' in row:
                        avg = (row['L Shoulder (N)'] + row['R Shoulder (N)']) / 2
                        if avg > 0:
                            asym = abs((row['L Shoulder (N)'] - row['R Shoulder (N)']) / avg * 100)
                            row['Asym (%)'] = round(asym, 1)

                    if len(row) > 1:
                        shoulder_data.append(row)

    if shoulder_data:
        shoulder_df = pd.DataFrame(shoulder_data)
        st.dataframe(shoulder_df, use_container_width=True, hide_index=True)
    else:
        st.info("No shoulder health data available (ForceFrame Shoulder IR/ER tests)")

    st.markdown("---")

    # =========================================================================
    # SECTION 4: Hip Health (ForceFrame data - testTypeName column)
    # =========================================================================
    st.markdown("### Hip Health")

    hip_data = []

    # Hip data comes from ForceFrame (testTypeName: 'Hip AD/AB', 'Hip IR/ER')
    if forceframe_df is not None and not forceframe_df.empty and 'Name' in forceframe_df.columns:
        ff_df = forceframe_df.copy()
        # Filter for sport if possible
        if sport and sport != "All Sports" and 'athlete_sport' in ff_df.columns:
            sport_mask = ff_df['athlete_sport'].str.contains(sport.split()[0], case=False, na=False)
            if sport_mask.any():
                ff_df = ff_df[sport_mask]

        # Find hip tests using testTypeName (ForceFrame column)
        test_col = 'testTypeName' if 'testTypeName' in ff_df.columns else 'testType'
        if test_col in ff_df.columns:
            hip_ff = ff_df[ff_df[test_col].str.contains('Hip|AD.*AB|Adduct|Groin', case=False, na=False)]

            if not hip_ff.empty:
                for athlete in hip_ff['Name'].dropna().unique():
                    athlete_hip = hip_ff[hip_ff['Name'] == athlete]
                    row = {'Athlete': athlete}

                    # Look for Left/Right peak force columns
                    for col in athlete_hip.columns:
                        col_lower = col.lower()
                        if ('left' in col_lower or 'l ' in col_lower) and ('peak' in col_lower or 'force' in col_lower):
                            val = athlete_hip[col].dropna().iloc[-1] if not athlete_hip[col].dropna().empty else None
                            if val is not None and pd.notna(val):
                                row['L Hip (N)'] = round(float(val), 0)
                        elif ('right' in col_lower or 'r ' in col_lower) and ('peak' in col_lower or 'force' in col_lower):
                            val = athlete_hip[col].dropna().iloc[-1] if not athlete_hip[col].dropna().empty else None
                            if val is not None and pd.notna(val):
                                row['R Hip (N)'] = round(float(val), 0)

                    # Calculate asymmetry if both sides available
                    if 'L Hip (N)' in row and 'R Hip (N)' in row:
                        avg = (row['L Hip (N)'] + row['R Hip (N)']) / 2
                        if avg > 0:
                            asym = abs((row['L Hip (N)'] - row['R Hip (N)']) / avg * 100)
                            row['Asym (%)'] = round(asym, 1)

                    if len(row) > 1:
                        hip_data.append(row)

    if hip_data:
        hip_df = pd.DataFrame(hip_data)
        st.dataframe(hip_df, use_container_width=True, hide_index=True)
    else:
        st.info("No hip health data available (ForceFrame Hip AD/AB tests)")

    st.markdown("---")

    # =========================================================================
    # SECTION 5: Plyo Pushup (PPU from ForceDecks)
    # =========================================================================
    st.markdown("### Plyo Pushup (Upper Body Power)")

    ppu_data = []
    ppu_df = sport_df[sport_df['testType'] == 'PPU'].copy() if 'testType' in sport_df.columns else pd.DataFrame()

    if not ppu_df.empty and 'Name' in ppu_df.columns:
        # Find best metric column
        metric_col = None
        metric_name = 'Pushup Height'
        for col in ['PUSHUP_HEIGHT', 'FLIGHT_TIME', 'BODYMASS_RELATIVE_TAKEOFF_POWER']:
            if col in ppu_df.columns and ppu_df[col].notna().sum() > 0:
                metric_col = col
                if col == 'PUSHUP_HEIGHT':
                    metric_name = 'Height (cm)'
                elif col == 'FLIGHT_TIME':
                    metric_name = 'Flight Time (s)'
                else:
                    metric_name = 'Rel Power (W/kg)'
                break

        if metric_col:
            for athlete in ppu_df['Name'].dropna().unique():
                athlete_ppu = ppu_df[ppu_df['Name'] == athlete]
                val = athlete_ppu[metric_col].dropna().iloc[-1] if not athlete_ppu[metric_col].dropna().empty else None
                if val is not None:
                    ppu_data.append({
                        'Athlete': athlete,
                        metric_name: round(float(val), 2)
                    })

    if ppu_data:
        ppu_table = pd.DataFrame(ppu_data).sort_values(list(ppu_data[0].keys())[1], ascending=False)
        st.dataframe(ppu_table, use_container_width=True, hide_index=True)
    else:
        st.info("No Plyo Pushup (PPU) data available")

    st.markdown("---")

    # =========================================================================
    # SECTION 6: DynaMo Grip Strength
    # =========================================================================
    st.markdown("### DynaMo Grip Strength")

    dynamo_df = pd.DataFrame()
    dynamo_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dynamo_allsports_with_athletes.csv'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'vald-data', 'data', 'dynamo_allsports_with_athletes.csv'),
    ]
    for path in dynamo_paths:
        if os.path.exists(path):
            try:
                dynamo_df = pd.read_csv(path)
                break
            except Exception:
                pass

    grip_data = []
    if not dynamo_df.empty and 'maxForceNewtons' in dynamo_df.columns:
        # Filter for GripSqueeze
        if 'movement' in dynamo_df.columns:
            dynamo_df = dynamo_df[dynamo_df['movement'] == 'GripSqueeze']

        # Filter by sport
        if sport and sport != "All Sports" and 'athlete_sport' in dynamo_df.columns:
            sport_pattern = sport.split()[0]
            sport_mask = dynamo_df['athlete_sport'].str.contains(sport_pattern, case=False, na=False)
            if sport_mask.any():
                dynamo_df = dynamo_df[sport_mask]

        name_col = 'full_name' if 'full_name' in dynamo_df.columns else 'Name'
        if name_col in dynamo_df.columns and 'laterality' in dynamo_df.columns:
            for athlete in dynamo_df[name_col].dropna().unique():
                athlete_grip = dynamo_df[dynamo_df[name_col] == athlete]
                row = {'Athlete': athlete}

                # Left hand - match 'LEFT', 'LeftSide', 'LeftThenRight', etc.
                left_grip = athlete_grip[athlete_grip['laterality'].str.contains('Left', case=False, na=False)]
                if not left_grip.empty:
                    row['Left (N)'] = round(left_grip['maxForceNewtons'].iloc[-1], 0)

                # Right hand - match 'RIGHT', 'RightSide', 'RightThenLeft', etc.
                right_grip = athlete_grip[athlete_grip['laterality'].str.contains('Right', case=False, na=False)]
                if not right_grip.empty:
                    row['Right (N)'] = round(right_grip['maxForceNewtons'].iloc[-1], 0)

                # Asymmetry
                if 'Left (N)' in row and 'Right (N)' in row:
                    avg = (row['Left (N)'] + row['Right (N)']) / 2
                    if avg > 0:
                        row['Asym (%)'] = round(abs((row['Left (N)'] - row['Right (N)']) / avg * 100), 1)

                if len(row) > 1:
                    grip_data.append(row)

    if grip_data:
        grip_table = pd.DataFrame(grip_data)
        st.dataframe(grip_table, use_container_width=True, hide_index=True)
    else:
        st.info("No DynaMo grip strength data available")

    st.markdown("---")

    # =========================================================================
    # SECTION 7: Strength RM (Manual Entry)
    # =========================================================================
    st.markdown("### Strength RM (Manual Entry)")

    # Get list of athletes from this sport for filtering manual data
    sport_athletes = sport_df['Name'].dropna().unique().tolist() if 'Name' in sport_df.columns else []

    strength_data = []
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    lower_body_path = os.path.join(data_dir, 'sc_lower_body.csv')
    upper_body_path = os.path.join(data_dir, 'sc_upper_body.csv')

    strength_dfs = []
    if os.path.exists(lower_body_path):
        try:
            lb_df = pd.read_csv(lower_body_path)
            lb_df['body_region'] = 'Lower'
            strength_dfs.append(lb_df)
        except Exception:
            pass
    if os.path.exists(upper_body_path):
        try:
            ub_df = pd.read_csv(upper_body_path)
            ub_df['body_region'] = 'Upper'
            strength_dfs.append(ub_df)
        except Exception:
            pass

    if strength_dfs:
        strength_df = pd.concat(strength_dfs, ignore_index=True)
        if 'athlete' in strength_df.columns and 'Name' not in strength_df.columns:
            strength_df['Name'] = strength_df['athlete']

        # Filter by sport athletes (only show athletes from the selected sport)
        if sport_athletes and 'Name' in strength_df.columns:
            strength_df = strength_df[strength_df['Name'].isin(sport_athletes)]

        # Get latest 1RM per athlete per exercise
        if 'estimated_1rm' in strength_df.columns and 'exercise' in strength_df.columns and not strength_df.empty:
            exercises = strength_df['exercise'].unique()
            athletes_str = strength_df['Name'].dropna().unique()

            for athlete in athletes_str:
                row = {'Athlete': athlete}
                athlete_str = strength_df[strength_df['Name'] == athlete]

                for exercise in ['Back Squat', 'Front Squat', 'Deadlift', 'Bench Press', 'Pull Up']:
                    ex_df = athlete_str[athlete_str['exercise'].str.contains(exercise, case=False, na=False)]
                    if not ex_df.empty and 'estimated_1rm' in ex_df.columns:
                        val = ex_df['estimated_1rm'].max()
                        if pd.notna(val):
                            short_name = exercise.replace(' ', '')[:6]
                            row[f'{short_name} (kg)'] = round(val, 0)

                if len(row) > 1:
                    strength_data.append(row)

    if strength_data:
        strength_table = pd.DataFrame(strength_data)
        st.dataframe(strength_table, use_container_width=True, hide_index=True)
    else:
        st.info(f"No Strength RM data available for {sport} athletes. Use ✏️ Data Entry tab to add.")

    st.markdown("---")

    # =========================================================================
    # SECTION 8: Broad Jump (Manual Entry)
    # =========================================================================
    st.markdown("### Broad Jump (Manual Entry)")

    broad_jump_path = os.path.join(data_dir, 'broad_jump.csv')
    broad_data = []

    if os.path.exists(broad_jump_path):
        try:
            bj_df = pd.read_csv(broad_jump_path)
            if 'athlete' in bj_df.columns and 'distance_cm' in bj_df.columns:
                # Filter by sport athletes
                if sport_athletes:
                    bj_df = bj_df[bj_df['athlete'].isin(sport_athletes)]

                if not bj_df.empty:
                    # Get best jump per athlete
                    best_jumps = bj_df.groupby('athlete')['distance_cm'].max().reset_index()
                    for _, row in best_jumps.iterrows():
                        broad_data.append({
                            'Athlete': row['athlete'],
                            'Best Distance (cm)': round(row['distance_cm'], 0)
                        })
        except Exception:
            pass

    if broad_data:
        broad_table = pd.DataFrame(broad_data).sort_values('Best Distance (cm)', ascending=False)
        st.dataframe(broad_table, use_container_width=True, hide_index=True)
    else:
        st.info(f"No Broad Jump data available for {sport} athletes. Use ✏️ Data Entry tab to add.")

    st.markdown("---")

    # =========================================================================
    # SECTION 9: Fitness Tests (Manual Entry)
    # =========================================================================
    st.markdown("### Fitness Tests (Manual Entry)")

    fitness_data = []
    aerobic_path = os.path.join(data_dir, 'aerobic_tests.csv')
    power_path = os.path.join(data_dir, 'power_tests.csv')

    if os.path.exists(aerobic_path):
        try:
            aero_df = pd.read_csv(aerobic_path)
            if 'athlete' in aero_df.columns and 'avg_relative_wattage' in aero_df.columns:
                # Filter by sport athletes
                if sport_athletes:
                    aero_df = aero_df[aero_df['athlete'].isin(sport_athletes)]

                if not aero_df.empty:
                    latest_aero = aero_df.groupby('athlete')['avg_relative_wattage'].last().reset_index()
                    for _, row in latest_aero.iterrows():
                        fitness_data.append({
                            'Athlete': row['athlete'],
                            '6 Min Aerobic (W/kg)': round(row['avg_relative_wattage'], 2)
                        })
        except Exception:
            pass

    if fitness_data:
        fitness_table = pd.DataFrame(fitness_data).sort_values('6 Min Aerobic (W/kg)', ascending=False)
        st.dataframe(fitness_table, use_container_width=True, hide_index=True)
    else:
        st.info(f"No Fitness test data available for {sport} athletes. Use ✏️ Data Entry tab to add.")


def _display_section_table(data: list, section_name: str):
    """Display a section table with RAG status indicators."""
    if not data:
        return

    df = pd.DataFrame(data)

    # Define status colors
    status_colors = {
        'green': '🟢',
        'amber': '🟠',
        'red': '🔴',
        'grey': '⚪'
    }

    # Create display data with status indicators
    display_data = []
    for _, row in df.iterrows():
        display_row = {'Athlete': row.get('Athlete', '')}

        for col in row.index:
            if col == 'Athlete' or col.endswith('_status'):
                continue

            val = row[col]
            if pd.notna(val):
                # Find corresponding status column
                status_key = col.replace(' (N/kg)', '').replace(' (W/kg)', '').replace(' (cm)', '').replace(' (N)', '').replace(' (%)', '').replace(' ', '')
                status_col = f"{status_key}_status"

                # Try different status column names
                status = row.get(status_col, 'grey')
                if status == 'grey':
                    # Try without spaces
                    for key in row.index:
                        if key.endswith('_status') and status_key.lower() in key.lower():
                            status = row[key]
                            break

                indicator = status_colors.get(status, '⚪')
                display_row[col] = f"{indicator} {val}"
            else:
                display_row[col] = "—"

        display_data.append(display_row)

    display_df = pd.DataFrame(display_data)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="font-size: 12px; color: #666; margin-top: 5px;">
        🟢 Excellent/Good | 🟠 Average | 🔴 Below Average | ⚪ No Data
    </div>
    """, unsafe_allow_html=True)


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
    # Ensure Name column exists (map from full_name if needed)
    if 'Name' not in df.columns and 'full_name' in df.columns:
        df = df.copy()
        df['Name'] = df['full_name']

    benchmarks = get_sport_benchmarks(sport, config)

    st.markdown(f"## {sport} Group Report - Alternative View")
    st.markdown("*Different visualization styles for comparison*")
    st.markdown("---")

    # Filter data for the sport
    sport_df = df.copy()
    if sport and sport != "All Sports" and 'athlete_sport' in sport_df.columns:
        sport_mask = sport_df['athlete_sport'].str.contains(sport.split()[0], case=False, na=False)
        if sport_mask.any():
            sport_df = sport_df[sport_mask]

    if sport_df.empty:
        st.warning(f"No data available for {sport if sport else 'selected filters'}")
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
                    rsi_val = hop_df[rsi_col].dropna().iloc[-1]
                    # Apply RSI conversion if value is too high (likely *100)
                    if rsi_val > 10:
                        rsi_val = rsi_val * 0.01
                    metrics['RSI'] = rsi_val

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
        # Reactive Strength RSI (DJ/RSHIP) - Lollipop Chart
        rsi_col = get_metric_column(sport_df, 'rsi')
        if rsi_col:
            # RSHIP = Repeat Single Hop In Place (10-5), DJ = Drop Jump
            hop_df = sport_df[sport_df['testType'].str.contains('RSHIP|DJ|SLDJ|Hop|Drop', case=False, na=False)]
            # Apply RSI unit conversion if needed
            hop_df = apply_metric_conversion(hop_df, 'rsi', rsi_col)
            if not hop_df.empty:
                fig = _create_lollipop_chart(
                    hop_df, rsi_col, 'Name', benchmarks,
                    "Reactive Strength - RSI", 'rsi'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="v3_rsi_lollipop")
            else:
                st.info("No reactive strength data (DJ/RSHIP) available")
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
        if sport and sport != "All Sports" and 'athlete_sport' in nb_df.columns:
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
                    marker_color=SECONDARY_ACCENT,
                    text=[f"{v:.0f}N" for v in latest[right_col]],
                    textposition='auto'
                ))

                # Add injury threshold line (using gray-blue for warning)
                fig.add_vline(x=337, line_dash="dash", line_color=GRAY_BLUE,
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

    # Try to load S&C data from session state or CSV
    sc_upper_body_path_v3 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sc_upper_body.csv')
    sc_df_v3 = pd.DataFrame()

    if hasattr(st, 'session_state') and 'sc_upper_body' in st.session_state:
        sc_df_v3 = st.session_state.sc_upper_body
    elif os.path.exists(sc_upper_body_path_v3):
        try:
            sc_df_v3 = pd.read_csv(sc_upper_body_path_v3)
            if 'date' in sc_df_v3.columns:
                sc_df_v3['date'] = pd.to_datetime(sc_df_v3['date'], errors='coerce')
        except Exception:
            pass

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
                st.info("Bench Press data not available - Use ✏️ Data Entry tab")
        elif not sc_df_v3.empty and 'exercise' in sc_df_v3.columns:
            # Show S&C manual entry data as bar chart
            bench_sc_v3 = sc_df_v3[sc_df_v3['exercise'].str.contains('Bench', case=False, na=False)]
            if not bench_sc_v3.empty:
                best_bench_v3 = bench_sc_v3.groupby('athlete').agg({
                    'estimated_1rm': 'max',
                    'weight_kg': 'max',
                    'date': 'max'
                }).reset_index().sort_values('estimated_1rm', ascending=True)

                # Create horizontal bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=best_bench_v3['athlete'],
                    x=best_bench_v3['estimated_1rm'],
                    orientation='h',
                    marker_color=TEAL_PRIMARY,
                    text=[f"{v:.0f}kg" for v in best_bench_v3['estimated_1rm']],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Bench Press - Est. 1RM (Manual Entry)",
                    xaxis_title="Estimated 1RM (kg)",
                    yaxis_title="",
                    height=max(250, len(best_bench_v3) * 40),
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True, key="v3_bench_manual")
            else:
                st.info("Bench Press data not available - Use ✏️ Data Entry tab")
        else:
            st.info("Bench Press data not available - Use ✏️ Data Entry tab")

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
                st.info("Pull Up data not available - Use ✏️ Data Entry tab")
        elif not sc_df_v3.empty and 'exercise' in sc_df_v3.columns:
            pullup_sc_v3 = sc_df_v3[sc_df_v3['exercise'].str.contains('Pull Up|Pullup|Pull-up|Chin', case=False, na=False)]
            if not pullup_sc_v3.empty:
                best_pullup_v3 = pullup_sc_v3.groupby('athlete').agg({
                    'reps': 'max',
                    'weight_kg': 'max',
                    'date': 'max'
                }).reset_index().sort_values('reps', ascending=True)

                # Create horizontal bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=best_pullup_v3['athlete'],
                    x=best_pullup_v3['reps'],
                    orientation='h',
                    marker_color=TEAL_PRIMARY,
                    text=[f"{int(v)} reps" for v in best_pullup_v3['reps']],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Pull Up - Max Reps (Manual Entry)",
                    xaxis_title="Reps",
                    yaxis_title="",
                    height=max(250, len(best_pullup_v3) * 40),
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True, key="v3_pullup_manual")
            else:
                st.info("Pull Up data not available - Use ✏️ Data Entry tab")
        else:
            st.info("Pull Up data not available - Use ✏️ Data Entry tab")

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
        if sport and sport != "All Sports" and 'athlete_sport' in ff_df.columns:
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
                        marker_color=SECONDARY_ACCENT,
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
                        marker_color=SECONDARY_ACCENT,
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


# =============================================================================
# SHOOTING / 10M PISTOL BALANCE REPORTS
# =============================================================================

# Shooting balance benchmarks (lower is better for stability metrics)
# Based on quiet standing bilateral tests - 30 seconds, eyes open
SHOOTING_BALANCE_BENCHMARKS = {
    'total_excursion': {
        'excellent': 100,   # mm - less movement is better
        'good': 150,
        'average': 200
    },
    'mean_velocity': {
        'excellent': 5,     # mm/s - slower is more stable
        'good': 10,
        'average': 15
    },
    'cop_ellipse_area': {
        'excellent': 50,    # mm² - smaller area is more stable
        'good': 100,
        'average': 200
    }
}


def create_shooting_group_report(df: pd.DataFrame, sport: str = "Shooting"):
    """
    Create a group report for Shooting/10m Pistol athletes.
    Focuses on Quiet Standing Balance (QSB) metrics:
    - Total Excursion (mm)
    - Mean Velocity (mm/s)
    - Area of CoP Ellipse (mm²)

    Lower values indicate better stability for shooters.
    """
    st.markdown("### 🎯 10m Pistol - Quiet Standing Balance")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1D4D3B 0%, #153829 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <p style="color: white; margin: 0; font-size: 0.95rem;">
            <strong>Balance Assessment</strong> • 30-second quiet standing • Eyes open • Lower values = better stability
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Filter for QSB (Quiet Standing Balance) tests - flexible matching
    qsb_df = pd.DataFrame()
    if 'testType' in df.columns:
        qsb_mask = df['testType'].str.contains('QSB|Quiet|Standing|Balance', case=False, na=False)
        qsb_df = df[qsb_mask].copy()

    if qsb_df.empty:
        st.warning("No Quiet Standing Balance (QSB) test data available.")

        # Show what test types exist to help with debugging
        if 'testType' in df.columns:
            available_tests = df['testType'].dropna().unique().tolist()
            with st.expander("📋 Available test types in data"):
                st.write("The following test types are available in your data:")
                for t in sorted(available_tests)[:20]:
                    st.write(f"• {t}")
                if len(available_tests) > 20:
                    st.write(f"... and {len(available_tests) - 20} more")
                st.info("QSB tests should include 'QSB', 'Quiet', 'Standing', or 'Balance' in the test type name.")
        return

    # Enrich athlete names from API if needed (handles API data without pre-enriched names)
    qsb_df = enrich_athlete_names(qsb_df)

    # Get Name column
    if 'Name' not in qsb_df.columns:
        if 'full_name' in qsb_df.columns:
            qsb_df['Name'] = qsb_df['full_name']
        elif 'profileId' in qsb_df.columns:
            qsb_df['Name'] = qsb_df['profileId'].apply(lambda x: f"Athlete_{str(x)[:8]}")
        else:
            st.error("No athlete name column found")
            return

    # Convert units from meters to millimeters
    # VALD stores: Total Excursion in m, Mean Velocity in m/s, Area in m²
    # Check both local_sync format (BAL_COP_*) and legacy format (*_Trial)
    metric_conversions = {
        # local_sync format columns
        'BAL_COP_TOTAL_EXCURSION': ('total_excursion_mm', 1000),      # m to mm
        'BAL_COP_MEAN_VELOCITY': ('mean_velocity_mm_s', 1000),        # m/s to mm/s
        'BAL_COP_ELLIPSE_AREA': ('cop_ellipse_area_mm2', 1000000),    # m² to mm²
        # legacy API format columns
        'Total Excursion_Trial': ('total_excursion_mm', 1000),
        'Mean Velocity_Trial': ('mean_velocity_mm_s', 1000),
        'Area of CoP Ellipse_Trial': ('cop_ellipse_area_mm2', 1000000),
    }

    for orig_col, (new_col, factor) in metric_conversions.items():
        if orig_col in qsb_df.columns and new_col not in qsb_df.columns:
            qsb_df[new_col] = qsb_df[orig_col] * factor

    # Get latest test per athlete - filter out NaN names first
    qsb_df = qsb_df[qsb_df['Name'].notna()].copy()

    if qsb_df.empty:
        st.warning("No valid athlete data available.")
        return

    if 'recordedDateUtc' in qsb_df.columns:
        qsb_df['recordedDateUtc'] = pd.to_datetime(qsb_df['recordedDateUtc'], errors='coerce')
        # Filter out NaN dates before groupby
        valid_dates = qsb_df[qsb_df['recordedDateUtc'].notna()]
        if not valid_dates.empty:
            latest_idx = valid_dates.groupby('Name')['recordedDateUtc'].idxmax()
            latest_df = qsb_df.loc[latest_idx].copy()
        else:
            latest_df = qsb_df.drop_duplicates(subset='Name', keep='last')
    else:
        latest_df = qsb_df.drop_duplicates(subset='Name', keep='last')

    # Summary metrics
    st.markdown("#### 📊 Group Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Athletes Tested", len(latest_df))

    with col2:
        if 'total_excursion_mm' in latest_df.columns:
            avg_excursion = latest_df['total_excursion_mm'].mean()
            st.metric("Avg Total Excursion", f"{avg_excursion:.1f} mm")

    with col3:
        if 'mean_velocity_mm_s' in latest_df.columns:
            avg_velocity = latest_df['mean_velocity_mm_s'].mean()
            st.metric("Avg Mean Velocity", f"{avg_velocity:.1f} mm/s")

    with col4:
        if 'cop_ellipse_area_mm2' in latest_df.columns:
            avg_area = latest_df['cop_ellipse_area_mm2'].mean()
            st.metric("Avg CoP Ellipse Area", f"{avg_area:.1f} mm²")

    st.markdown("---")

    # Create visualizations in columns
    col1, col2 = st.columns(2)

    with col1:
        # Total Excursion bar chart (lower is better)
        if 'total_excursion_mm' in latest_df.columns:
            st.markdown("#### Total Excursion (mm)")
            st.caption("Lower values indicate better stability")

            sorted_df = latest_df.sort_values('total_excursion_mm', ascending=True)
            benchmarks = SHOOTING_BALANCE_BENCHMARKS['total_excursion']

            fig = go.Figure()

            # Add benchmark zones (horizontal for bar chart)
            fig.add_vrect(x0=0, x1=benchmarks['excellent'],
                          fillcolor=ZONE_COLORS['excellent'], layer="below", line_width=0)
            fig.add_vrect(x0=benchmarks['excellent'], x1=benchmarks['good'],
                          fillcolor=ZONE_COLORS['good'], layer="below", line_width=0)
            fig.add_vrect(x0=benchmarks['good'], x1=benchmarks['average'],
                          fillcolor=ZONE_COLORS['average'], layer="below", line_width=0)

            # Color bars based on benchmark
            colors = []
            for val in sorted_df['total_excursion_mm']:
                if val <= benchmarks['excellent']:
                    colors.append(TEAL_PRIMARY)
                elif val <= benchmarks['good']:
                    colors.append('#2A6A50')
                else:
                    colors.append('#78909C')

            fig.add_trace(go.Bar(
                y=sorted_df['Name'],
                x=sorted_df['total_excursion_mm'],
                orientation='h',
                marker_color=colors,
                text=[f"{v:.1f}" for v in sorted_df['total_excursion_mm']],
                textposition='auto'
            ))

            fig.update_layout(
                xaxis_title="Total Excursion (mm)",
                yaxis_title="",
                height=max(300, len(sorted_df) * 40),
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=30)
            )

            st.plotly_chart(fig, use_container_width=True, key="shooting_excursion")

    with col2:
        # Mean Velocity bar chart (lower is better)
        if 'mean_velocity_mm_s' in latest_df.columns:
            st.markdown("#### Mean Velocity (mm/s)")
            st.caption("Lower values indicate better stability")

            sorted_df = latest_df.sort_values('mean_velocity_mm_s', ascending=True)
            benchmarks = SHOOTING_BALANCE_BENCHMARKS['mean_velocity']

            fig = go.Figure()

            # Add benchmark zones
            fig.add_vrect(x0=0, x1=benchmarks['excellent'],
                          fillcolor=ZONE_COLORS['excellent'], layer="below", line_width=0)
            fig.add_vrect(x0=benchmarks['excellent'], x1=benchmarks['good'],
                          fillcolor=ZONE_COLORS['good'], layer="below", line_width=0)
            fig.add_vrect(x0=benchmarks['good'], x1=benchmarks['average'],
                          fillcolor=ZONE_COLORS['average'], layer="below", line_width=0)

            # Color bars based on benchmark
            colors = []
            for val in sorted_df['mean_velocity_mm_s']:
                if val <= benchmarks['excellent']:
                    colors.append(TEAL_PRIMARY)
                elif val <= benchmarks['good']:
                    colors.append('#2A6A50')
                else:
                    colors.append('#78909C')

            fig.add_trace(go.Bar(
                y=sorted_df['Name'],
                x=sorted_df['mean_velocity_mm_s'],
                orientation='h',
                marker_color=colors,
                text=[f"{v:.1f}" for v in sorted_df['mean_velocity_mm_s']],
                textposition='auto'
            ))

            fig.update_layout(
                xaxis_title="Mean Velocity (mm/s)",
                yaxis_title="",
                height=max(300, len(sorted_df) * 40),
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=30)
            )

            st.plotly_chart(fig, use_container_width=True, key="shooting_velocity")

    # Area of CoP Ellipse
    if 'cop_ellipse_area_mm2' in latest_df.columns:
        st.markdown("#### Area of CoP Ellipse (mm²)")
        st.caption("Smaller area indicates more stable platform - critical for precision shooting")

        sorted_df = latest_df.sort_values('cop_ellipse_area_mm2', ascending=True)
        benchmarks = SHOOTING_BALANCE_BENCHMARKS['cop_ellipse_area']

        fig = go.Figure()

        # Add benchmark zones
        fig.add_vrect(x0=0, x1=benchmarks['excellent'],
                      fillcolor=ZONE_COLORS['excellent'], layer="below", line_width=0)
        fig.add_vrect(x0=benchmarks['excellent'], x1=benchmarks['good'],
                      fillcolor=ZONE_COLORS['good'], layer="below", line_width=0)
        fig.add_vrect(x0=benchmarks['good'], x1=benchmarks['average'],
                      fillcolor=ZONE_COLORS['average'], layer="below", line_width=0)

        # Color bars based on benchmark
        colors = []
        for val in sorted_df['cop_ellipse_area_mm2']:
            if val <= benchmarks['excellent']:
                colors.append(TEAL_PRIMARY)
            elif val <= benchmarks['good']:
                colors.append('#2A6A50')
            else:
                colors.append('#78909C')

        fig.add_trace(go.Bar(
            y=sorted_df['Name'],
            x=sorted_df['cop_ellipse_area_mm2'],
            orientation='h',
            marker_color=colors,
            text=[f"{v:.1f}" for v in sorted_df['cop_ellipse_area_mm2']],
            textposition='auto'
        ))

        fig.update_layout(
            xaxis_title="CoP Ellipse Area (mm²)",
            yaxis_title="",
            height=max(300, len(sorted_df) * 40),
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=30)
        )

        st.plotly_chart(fig, use_container_width=True, key="shooting_ellipse")

    # Summary table
    st.markdown("---")
    st.markdown("#### 📋 Detailed Results")

    display_cols = ['Name']
    rename_map = {'Name': 'Athlete'}

    if 'total_excursion_mm' in latest_df.columns:
        display_cols.append('total_excursion_mm')
        rename_map['total_excursion_mm'] = 'Total Excursion (mm)'

    if 'mean_velocity_mm_s' in latest_df.columns:
        display_cols.append('mean_velocity_mm_s')
        rename_map['mean_velocity_mm_s'] = 'Mean Velocity (mm/s)'

    if 'cop_ellipse_area_mm2' in latest_df.columns:
        display_cols.append('cop_ellipse_area_mm2')
        rename_map['cop_ellipse_area_mm2'] = 'CoP Ellipse Area (mm²)'

    if 'recordedDateUtc' in latest_df.columns:
        display_cols.append('recordedDateUtc')
        rename_map['recordedDateUtc'] = 'Test Date'

    result_df = latest_df[display_cols].rename(columns=rename_map)

    # Format numeric columns
    for col in result_df.columns:
        if col not in ['Athlete', 'Test Date']:
            result_df[col] = result_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")

    st.dataframe(result_df, use_container_width=True, hide_index=True)


def create_shooting_individual_report(df: pd.DataFrame, athlete_name: str, sport: str = "Shooting"):
    """
    Create an individual report for a Shooting/10m Pistol athlete.
    Shows balance metrics over time with trend analysis.
    """
    st.markdown(f"### 🎯 {athlete_name} - Balance Analysis")

    # Enrich athlete names from API if needed
    df = enrich_athlete_names(df)

    # Filter for athlete and QSB tests
    if 'Name' not in df.columns:
        if 'full_name' in df.columns:
            df['Name'] = df['full_name']
        else:
            st.error("No athlete name column found")
            return

    # Filter for athlete and QSB tests - flexible matching
    qsb_mask = df['testType'].str.contains('QSB|Quiet|Standing|Balance', case=False, na=False) if 'testType' in df.columns else pd.Series([False] * len(df))
    athlete_df = df[(df['Name'] == athlete_name) & qsb_mask].copy()

    # If no QSB tests, show all athlete data
    if athlete_df.empty:
        athlete_df = df[df['Name'] == athlete_name].copy()

    if athlete_df.empty:
        st.warning(f"No data found for {athlete_name}")
        return

    # Convert units - check both local_sync format (BAL_COP_*) and legacy format (*_Trial)
    metric_conversions = {
        # local_sync format columns
        'BAL_COP_TOTAL_EXCURSION': ('total_excursion_mm', 1000),      # m to mm
        'BAL_COP_MEAN_VELOCITY': ('mean_velocity_mm_s', 1000),        # m/s to mm/s
        'BAL_COP_ELLIPSE_AREA': ('cop_ellipse_area_mm2', 1000000),    # m² to mm²
        # legacy API format columns
        'Total Excursion_Trial': ('total_excursion_mm', 1000),
        'Mean Velocity_Trial': ('mean_velocity_mm_s', 1000),
        'Area of CoP Ellipse_Trial': ('cop_ellipse_area_mm2', 1000000),
    }

    for orig_col, (new_col, factor) in metric_conversions.items():
        if orig_col in athlete_df.columns and new_col not in athlete_df.columns:
            athlete_df[new_col] = athlete_df[orig_col] * factor

    # Sort by date
    if 'recordedDateUtc' in athlete_df.columns:
        athlete_df['recordedDateUtc'] = pd.to_datetime(athlete_df['recordedDateUtc'], errors='coerce')
        athlete_df = athlete_df.sort_values('recordedDateUtc')

    # Latest metrics
    latest = athlete_df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Tests Completed", len(athlete_df))

    with col2:
        if 'total_excursion_mm' in latest:
            val = latest['total_excursion_mm']
            bench = SHOOTING_BALANCE_BENCHMARKS['total_excursion']
            status = "🟢" if val <= bench['excellent'] else "🟡" if val <= bench['good'] else "��"
            st.metric(f"{status} Total Excursion", f"{val:.1f} mm")

    with col3:
        if 'mean_velocity_mm_s' in latest:
            val = latest['mean_velocity_mm_s']
            bench = SHOOTING_BALANCE_BENCHMARKS['mean_velocity']
            status = "🟢" if val <= bench['excellent'] else "🟡" if val <= bench['good'] else "🔴"
            st.metric(f"{status} Mean Velocity", f"{val:.1f} mm/s")

    with col4:
        if 'cop_ellipse_area_mm2' in latest:
            val = latest['cop_ellipse_area_mm2']
            bench = SHOOTING_BALANCE_BENCHMARKS['cop_ellipse_area']
            status = "🟢" if val <= bench['excellent'] else "🟡" if val <= bench['good'] else "🔴"
            st.metric(f"{status} CoP Ellipse Area", f"{val:.1f} mm²")

    st.markdown("---")

    # Trend charts if multiple tests
    if len(athlete_df) > 1 and 'recordedDateUtc' in athlete_df.columns:
        st.markdown("#### 📈 Progress Over Time")
        st.caption("Decreasing values indicate improving stability")

        col1, col2 = st.columns(2)

        with col1:
            if 'total_excursion_mm' in athlete_df.columns:
                fig = go.Figure()

                # Add benchmark zones
                bench = SHOOTING_BALANCE_BENCHMARKS['total_excursion']
                fig.add_hrect(y0=0, y1=bench['excellent'],
                              fillcolor=ZONE_COLORS['excellent'], layer="below", line_width=0)
                fig.add_hrect(y0=bench['excellent'], y1=bench['good'],
                              fillcolor=ZONE_COLORS['good'], layer="below", line_width=0)

                fig.add_trace(go.Scatter(
                    x=athlete_df['recordedDateUtc'],
                    y=athlete_df['total_excursion_mm'],
                    mode='markers+lines',
                    marker=dict(size=10, color=TEAL_PRIMARY),
                    line=dict(color=TEAL_PRIMARY, width=2),
                    name='Total Excursion'
                ))

                fig.update_layout(
                    title="Total Excursion Trend",
                    xaxis_title="Date",
                    yaxis_title="Total Excursion (mm)",
                    height=300,
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True, key="ind_excursion_trend")

        with col2:
            if 'cop_ellipse_area_mm2' in athlete_df.columns:
                fig = go.Figure()

                # Add benchmark zones
                bench = SHOOTING_BALANCE_BENCHMARKS['cop_ellipse_area']
                fig.add_hrect(y0=0, y1=bench['excellent'],
                              fillcolor=ZONE_COLORS['excellent'], layer="below", line_width=0)
                fig.add_hrect(y0=bench['excellent'], y1=bench['good'],
                              fillcolor=ZONE_COLORS['good'], layer="below", line_width=0)

                fig.add_trace(go.Scatter(
                    x=athlete_df['recordedDateUtc'],
                    y=athlete_df['cop_ellipse_area_mm2'],
                    mode='markers+lines',
                    marker=dict(size=10, color=GOLD_ACCENT),
                    line=dict(color=GOLD_ACCENT, width=2),
                    name='CoP Ellipse Area'
                ))

                fig.update_layout(
                    title="CoP Ellipse Area Trend",
                    xaxis_title="Date",
                    yaxis_title="Area (mm²)",
                    height=300,
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True, key="ind_ellipse_trend")

    # Test history table
    st.markdown("#### 📋 Test History")

    display_cols = []
    rename_map = {}

    if 'recordedDateUtc' in athlete_df.columns:
        display_cols.append('recordedDateUtc')
        rename_map['recordedDateUtc'] = 'Date'

    if 'total_excursion_mm' in athlete_df.columns:
        display_cols.append('total_excursion_mm')
        rename_map['total_excursion_mm'] = 'Total Excursion (mm)'

    if 'mean_velocity_mm_s' in athlete_df.columns:
        display_cols.append('mean_velocity_mm_s')
        rename_map['mean_velocity_mm_s'] = 'Mean Velocity (mm/s)'

    if 'cop_ellipse_area_mm2' in athlete_df.columns:
        display_cols.append('cop_ellipse_area_mm2')
        rename_map['cop_ellipse_area_mm2'] = 'CoP Ellipse Area (mm²)'

    if display_cols:
        history_df = athlete_df[display_cols].rename(columns=rename_map)

        # Format numeric columns
        for col in history_df.columns:
            if col != 'Date':
                history_df[col] = history_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")

        st.dataframe(history_df.sort_values('Date', ascending=False), use_container_width=True, hide_index=True)
