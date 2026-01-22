"""
Advanced Analysis Utilities for VALD Performance Dashboard
Based on research from Patrick Ward, mattsams89, and elite sports science practices
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import signal, stats
from typing import Dict, List, Optional, Tuple
import streamlit as st

# Team Saudi colors
TEAL_PRIMARY = '#255035'      # Saudi Green
GOLD_ACCENT = '#a08e66'       # Gold accent
TEAL_DARK = '#1C3D28'         # Dark green
TEAL_LIGHT = '#2E6040'        # Light green
GRAY_BLUE = '#78909C'         # Neutral gray/blue
INFO_BLUE = '#0077B6'         # Info blue


# ============================================================================
# ASYMMETRY ANALYSIS
# ============================================================================

def calculate_asymmetry(left_value: float, right_value: float) -> Dict:
    """
    Calculate bilateral asymmetry using symmetry index
    Formula: ((Left - Right) / ((Left + Right) / 2)) * 100

    Threshold: >15% requires intervention (VALD/NordBord standard)
    """
    if pd.isna(left_value) or pd.isna(right_value):
        return {
            'symmetry_index': None,
            'asymmetry_percent': None,
            'flag': False,
            'magnitude': 'Unknown',
            'interpretation': 'Insufficient data'
        }

    if left_value == 0 and right_value == 0:
        return {
            'symmetry_index': 0.0,
            'asymmetry_percent': 0.0,
            'flag': False,
            'magnitude': 'Acceptable',
            'interpretation': 'Balanced'
        }

    average = (left_value + right_value) / 2

    if average == 0:
        return {
            'symmetry_index': None,
            'asymmetry_percent': None,
            'flag': False,
            'magnitude': 'Unknown',
            'interpretation': 'Cannot calculate'
        }

    symmetry_index = ((left_value - right_value) / average) * 100
    asymmetry_percent = abs(symmetry_index)

    # Thresholds based on research
    if asymmetry_percent > 15:
        magnitude = 'High - Intervention Needed'
        flag = True
    elif asymmetry_percent > 10:
        magnitude = 'Moderate - Monitor'
        flag = True
    else:
        magnitude = 'Acceptable'
        flag = False

    # Determine dominance
    if symmetry_index > 0:
        dominant_side = 'Left'
    elif symmetry_index < 0:
        dominant_side = 'Right'
    else:
        dominant_side = 'Balanced'

    interpretation = f"{dominant_side} dominant by {asymmetry_percent:.1f}%"

    return {
        'symmetry_index': round(symmetry_index, 2),
        'asymmetry_percent': round(asymmetry_percent, 2),
        'flag': flag,
        'magnitude': magnitude,
        'interpretation': interpretation,
        'dominant_side': dominant_side
    }


def create_asymmetry_circle_plot(asymmetry_data: pd.DataFrame, athlete_name: str) -> go.Figure:
    """
    Create asymmetry visualization using circle size to represent magnitude
    Innovation from Hawkin Dynamics
    """
    fig = go.Figure()

    # Add circles for each test (Team Saudi colors)
    for _, row in asymmetry_data.iterrows():
        if pd.notna(row.get('asymmetry_percent')):
            size = max(10, row['asymmetry_percent'] * 2)  # Scale circle size
            color = GRAY_BLUE if row['asymmetry_percent'] > 15 else GOLD_ACCENT if row['asymmetry_percent'] > 10 else TEAL_PRIMARY

            fig.add_trace(go.Scatter(
                x=[row['recordedDateUtc']],
                y=[row['asymmetry_percent']],
                mode='markers',
                marker=dict(size=size, color=color, opacity=0.6),
                name=row.get('testType', 'Test'),
                hovertemplate=f"<b>{row.get('testType', 'Test')}</b><br>" +
                             f"Date: {row['recordedDateUtc']}<br>" +
                             f"Asymmetry: {row['asymmetry_percent']:.1f}%<br>" +
                             f"{row.get('interpretation', '')}<extra></extra>"
            ))

    # Add threshold lines (Team Saudi colors)
    fig.add_hline(y=15, line_dash="dash", line_color=GRAY_BLUE,
                  annotation_text="Intervention Threshold (15%)")
    fig.add_hline(y=10, line_dash="dash", line_color=GOLD_ACCENT,
                  annotation_text="Monitor Threshold (10%)")

    fig.update_layout(
        title=f"Bilateral Asymmetry Timeline - {athlete_name}",
        xaxis_title="Date",
        yaxis_title="Asymmetry (%)",
        showlegend=False,
        height=400
    )

    return fig


# ============================================================================
# MEANINGFUL CHANGE DETECTION
# ============================================================================

def calculate_meaningful_change(current: float, previous: float,
                               group_sd: float, threshold: float = 0.6) -> Dict:
    """
    Detect meaningful change using statistical and practical significance
    Based on Patrick Ward's approach: 0.6 × SD (vs conventional 0.2)

    Args:
        current: Current value
        previous: Previous value
        group_sd: Standard deviation of the group/sport
        threshold: Multiplier for SD (default 0.6)
    """
    if pd.isna(current) or pd.isna(previous) or pd.isna(group_sd):
        return {
            'is_meaningful': False,
            'change_value': 0,
            'change_percent': 0,
            'direction': 'Unknown',
            'direction_symbol': '→',
            'magnitude': 'Unknown',
            'threshold': 0
        }

    change_value = current - previous
    change_percent = ((current - previous) / previous) * 100 if previous != 0 else 0

    meaningful_threshold = threshold * group_sd
    is_meaningful = abs(change_value) > meaningful_threshold

    if change_value > 0:
        direction = 'Increase'
        direction_symbol = '↑'
    elif change_value < 0:
        direction = 'Decrease'
        direction_symbol = '↓'
    else:
        direction = 'No Change'
        direction_symbol = '→'

    # Magnitude categories
    if abs(change_value) > (threshold * 2 * group_sd):
        magnitude = 'Very Large'
    elif abs(change_value) > meaningful_threshold:
        magnitude = 'Moderate'
    else:
        magnitude = 'Small'

    return {
        'is_meaningful': is_meaningful,
        'change_value': round(change_value, 2),
        'change_percent': round(change_percent, 1),
        'direction': direction,
        'direction_symbol': direction_symbol,
        'magnitude': magnitude,
        'threshold': round(meaningful_threshold, 2)
    }


def create_meaningful_change_plot(df: pd.DataFrame, metric: str,
                                 sport: str = None) -> go.Figure:
    """
    Create time series with meaningful change indicators
    Patrick Ward style with color-coded meaningful changes
    """
    # Calculate group SD
    if sport:
        group_df = df[df['athlete_sport'] == sport]
    else:
        group_df = df

    group_sd = group_df[metric].std()
    threshold = 0.6 * group_sd

    fig = go.Figure()

    # Add mean line
    mean_val = group_df[metric].mean()
    fig.add_hline(y=mean_val, line_dash="solid", line_color="gray",
                  annotation_text=f"Group Mean: {mean_val:.1f}")

    # Add ±1 SD bands
    fig.add_hrect(y0=mean_val - group_sd, y1=mean_val + group_sd,
                  fillcolor="lightgray", opacity=0.2,
                  annotation_text="±1 SD", annotation_position="left")

    # Plot data points with meaningful change colors (Team Saudi)
    colors = []
    for i in range(1, len(df)):
        change = calculate_meaningful_change(
            df.iloc[i][metric],
            df.iloc[i-1][metric],
            group_sd
        )
        colors.append(GOLD_ACCENT if change['is_meaningful'] else TEAL_PRIMARY)

    colors.insert(0, TEAL_PRIMARY)  # First point

    fig.add_trace(go.Scatter(
        x=df['recordedDateUtc'],
        y=df[metric],
        mode='lines+markers',
        marker=dict(color=colors, size=8),
        line=dict(color=TEAL_LIGHT),
        name=metric,
        hovertemplate="<b>%{x}</b><br>" +
                     f"{metric}: %{{y:.2f}}<br>" +
                     "<extra></extra>"
    ))

    fig.update_layout(
        title=f"{metric} - Meaningful Change Detection",
        xaxis_title="Date",
        yaxis_title=metric,
        height=400
    )

    return fig


# ============================================================================
# TYPICAL ERROR MEASUREMENT (TEM)
# ============================================================================

def calculate_tem(values: List[float]) -> float:
    """
    Calculate Typical Error Measurement
    Formula: SD(differences) / √2

    Used for assessing test reliability
    """
    if len(values) < 2:
        return None

    differences = np.diff(values)
    sd_diff = np.std(differences, ddof=1)
    tem = sd_diff / np.sqrt(2)

    return round(tem, 2)


def calculate_tem_with_ci(df: pd.DataFrame, metric: str,
                         confidence: float = 0.95) -> Dict:
    """
    Calculate TEM with confidence intervals
    """
    values = df[metric].dropna().values

    if len(values) < 2:
        return {
            'tem': None,
            'tem_percent': None,
            'ci_lower': None,
            'ci_upper': None
        }

    tem = calculate_tem(values)
    mean_val = np.mean(values)
    tem_percent = (tem / mean_val) * 100 if mean_val != 0 else 0

    # Bootstrap CI
    n_bootstrap = 1000
    tem_bootstrap = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        tem_bootstrap.append(calculate_tem(sample))

    ci_lower, ci_upper = np.percentile(tem_bootstrap,
                                      [(1-confidence)*100/2,
                                       100-(1-confidence)*100/2])

    return {
        'tem': tem,
        'tem_percent': round(tem_percent, 1),
        'ci_lower': round(ci_lower, 2),
        'ci_upper': round(ci_upper, 2),
        'reliability': 'Excellent' if tem_percent < 5 else
                      'Good' if tem_percent < 10 else
                      'Moderate' if tem_percent < 15 else 'Poor'
    }


# ============================================================================
# NORMATIVE BENCHMARKING
# ============================================================================

def calculate_percentile_rank(value: float, reference_values: List[float]) -> Dict:
    """
    Calculate percentile rank with color zone
    """
    if pd.isna(value) or not reference_values:
        return {
            'percentile': None,
            'zone': 'Unknown',
            'color': 'gray',
            'interpretation': 'Insufficient data'
        }

    percentile = stats.percentileofscore(reference_values, value)

    # Color zones based on research (Team Saudi colors)
    if percentile >= 75:
        zone = 'Excellent'
        color = TEAL_PRIMARY
    elif percentile >= 50:
        zone = 'Good'
        color = TEAL_LIGHT
    elif percentile >= 25:
        zone = 'Average'
        color = GOLD_ACCENT
    else:
        zone = 'Below Average'
        color = GRAY_BLUE

    return {
        'percentile': round(percentile, 1),
        'zone': zone,
        'color': color,
        'interpretation': f"{zone} - {percentile:.0f}th percentile"
    }


def create_normative_benchmark_plot(athlete_value: float,
                                   reference_df: pd.DataFrame,
                                   metric: str, sport: str) -> go.Figure:
    """
    Create benchmarking visualization with percentile zones
    """
    reference_values = reference_df[metric].dropna().values

    if len(reference_values) == 0:
        return go.Figure()

    # Calculate percentiles
    p25 = np.percentile(reference_values, 25)
    p50 = np.percentile(reference_values, 50)
    p75 = np.percentile(reference_values, 75)

    fig = go.Figure()

    # Add distribution (Team Saudi colors)
    fig.add_trace(go.Histogram(
        x=reference_values,
        name='Sport Distribution',
        marker_color=TEAL_LIGHT,
        opacity=0.6,
        nbinsx=30
    ))

    # Add percentile lines (Team Saudi colors)
    fig.add_vline(x=p25, line_dash="dash", line_color=GOLD_ACCENT,
                  annotation_text="25th %ile")
    fig.add_vline(x=p50, line_dash="dash", line_color=INFO_BLUE,
                  annotation_text="50th %ile (Median)")
    fig.add_vline(x=p75, line_dash="dash", line_color=TEAL_PRIMARY,
                  annotation_text="75th %ile (Target)")

    # Add athlete value (use gold to highlight)
    if pd.notna(athlete_value):
        fig.add_vline(x=athlete_value, line_color=GOLD_ACCENT, line_width=3,
                      annotation_text="Athlete")

    fig.update_layout(
        title=f"{metric} - {sport} Normative Comparison",
        xaxis_title=metric,
        yaxis_title="Frequency",
        showlegend=True,
        height=400
    )

    return fig


# ============================================================================
# PHASE ANALYSIS
# ============================================================================

def detect_jump_phases(force_data: np.ndarray, time_data: np.ndarray,
                      bodyweight: float) -> Dict:
    """
    Detect jump phases: unweighting, braking, propulsion
    Based on mattsams89/shiny-vertical-jump approach
    """
    if len(force_data) == 0 or len(time_data) == 0:
        return {}

    # Find quiet standing period (lowest variance 1-second window)
    window_size = int(1.0 / (time_data[1] - time_data[0]))  # 1 second

    variances = []
    for i in range(len(force_data) - window_size):
        window = force_data[i:i+window_size]
        variances.append(np.var(window))

    if not variances:
        baseline_force = bodyweight
    else:
        baseline_idx = np.argmin(variances)
        baseline_force = np.mean(force_data[baseline_idx:baseline_idx+window_size])

    # Detect start of movement (5SD - BW method)
    threshold = baseline_force - (5 * np.std(force_data[:window_size]))

    start_idx = None
    for i in range(len(force_data)):
        if force_data[i] < threshold:
            start_idx = i
            break

    if start_idx is None:
        return {'error': 'Could not detect movement start'}

    # Find peak force in propulsion phase
    peak_idx = start_idx + np.argmax(force_data[start_idx:])

    # Find takeoff (force returns to ~0)
    takeoff_idx = None
    for i in range(peak_idx, len(force_data)):
        if force_data[i] < (0.1 * bodyweight):
            takeoff_idx = i
            break

    if takeoff_idx is None:
        takeoff_idx = len(force_data) - 1

    # Find transition from braking to propulsion (minimum velocity point)
    # This would require velocity data or integration
    braking_end_idx = int((start_idx + peak_idx) / 2)  # Simplified

    return {
        'baseline_force': baseline_force,
        'start_movement_idx': start_idx,
        'start_movement_time': time_data[start_idx],
        'braking_end_idx': braking_end_idx,
        'braking_end_time': time_data[braking_end_idx],
        'peak_force_idx': peak_idx,
        'peak_force_time': time_data[peak_idx],
        'peak_force': force_data[peak_idx],
        'takeoff_idx': takeoff_idx,
        'takeoff_time': time_data[takeoff_idx] if takeoff_idx < len(time_data) else None,
        'unweighting_duration': time_data[braking_end_idx] - time_data[start_idx],
        'braking_duration': time_data[braking_end_idx] - time_data[start_idx],
        'propulsion_duration': time_data[takeoff_idx] - time_data[braking_end_idx] if takeoff_idx < len(time_data) else None
    }


# ============================================================================
# Z-SCORE ANALYSIS
# ============================================================================

def calculate_zscore_with_context(df: pd.DataFrame, metric: str,
                                  filter_col: str = None,
                                  filter_val: str = None) -> pd.DataFrame:
    """
    Calculate z-scores on FULL dataset before filtering
    Critical insight from Patrick Ward: maintains proper context
    """
    # Calculate on full dataset first
    mean_val = df[metric].mean()
    std_val = df[metric].std()

    df['z_score'] = (df[metric] - mean_val) / std_val
    df['z_flag'] = df['z_score'].abs() > 1
    df['z_interpretation'] = df['z_score'].apply(
        lambda z: 'Very High' if z > 2 else
                 'High' if z > 1 else
                 'Very Low' if z < -2 else
                 'Low' if z < -1 else
                 'Normal'
    )

    # Now filter if needed
    if filter_col and filter_val:
        df = df[df[filter_col] == filter_val].copy()

    return df


def create_zscore_distribution_plot(df: pd.DataFrame, metric: str) -> go.Figure:
    """
    Create z-score distribution with flagged outliers
    """
    fig = go.Figure()

    # Add histogram (Team Saudi colors)
    fig.add_trace(go.Histogram(
        x=df['z_score'],
        name='Distribution',
        marker_color=TEAL_LIGHT,
        nbinsx=30
    ))

    # Add threshold lines (Team Saudi colors)
    for z in [-2, -1, 1, 2]:
        color = GRAY_BLUE if abs(z) == 2 else GOLD_ACCENT
        fig.add_vline(x=z, line_dash="dash", line_color=color,
                      annotation_text=f"Z={z}")

    fig.update_layout(
        title=f"{metric} - Z-Score Distribution",
        xaxis_title="Z-Score",
        yaxis_title="Frequency",
        height=400
    )

    return fig


# ============================================================================
# BILATERAL FORCE-TIME CURVES
# ============================================================================

def create_bilateral_force_curve(time: np.ndarray,
                                 force_left: np.ndarray,
                                 force_right: np.ndarray,
                                 phases: Dict = None) -> go.Figure:
    """
    Create bilateral force-time curve visualization
    Innovation from multiple elite sources
    """
    force_total = force_left + force_right

    fig = go.Figure()

    # Add force traces (Team Saudi colors)
    fig.add_trace(go.Scatter(
        x=time, y=force_left,
        name='Left',
        line=dict(color=TEAL_PRIMARY, width=2),
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=time, y=force_right,
        name='Right',
        line=dict(color=INFO_BLUE, width=2),
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=time, y=force_total,
        name='Total',
        line=dict(color='gray', width=3),
        mode='lines'
    ))

    # Add phase regions if provided
    if phases:
        if 'start_movement_time' in phases and 'braking_end_time' in phases:
            fig.add_vrect(
                x0=phases['start_movement_time'],
                x1=phases['braking_end_time'],
                fillcolor="lightcoral", opacity=0.2,
                annotation_text="Braking", annotation_position="top left"
            )

        if 'braking_end_time' in phases and 'takeoff_time' in phases:
            fig.add_vrect(
                x0=phases['braking_end_time'],
                x1=phases['takeoff_time'],
                fillcolor="lightgreen", opacity=0.2,
                annotation_text="Propulsion", annotation_position="top left"
            )

    fig.update_layout(
        title="Bilateral Force-Time Curve",
        xaxis_title="Time (s)",
        yaxis_title="Force (N)",
        hovermode='x unified',
        height=500
    )

    return fig
