"""
Force Trace Visualization Module
VALD Performance Dashboard - Saudi National Team

This module handles downloading and visualizing force-time curves from VALD API.

Features:
- Download raw force traces
- Overlay multiple trials
- Phase analysis (eccentric, concentric, landing)
- Compare athletes
- Export for biomechanics analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
import requests

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# ============================================================================
# FORCE TRACE API FUNCTIONS
# ============================================================================

def _get_force_trace_impl(test_id: str, trial_id: str, token: str, tenant_id: str, region: str = 'euw') -> pd.DataFrame:
    """
    Download raw force-time trace from VALD API

    API Endpoint:
    GET /v2019q3/teams/{tenant}/tests/{testId}/trials/{trialId}/trace

    Parameters:
    -----------
    test_id : str
        VALD test ID
    trial_id : str
        Trial ID within the test
    token : str
        OAuth bearer token
    tenant_id : str
        Your tenant/team ID
    region : str
        Data region (euw, use, aue)

    Returns:
    --------
    pd.DataFrame with columns:
        - time_ms: Time in milliseconds
        - force_n: Force in Newtons
        - force_left_n: Left limb force (if bilateral)
        - force_right_n: Right limb force (if bilateral)
    """

    url = f"https://prd-{region}-api-extforcedecks.valdperformance.com/v2019q3/teams/{tenant_id}/tests/{test_id}/trials/{trial_id}/trace"

    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            trace_data = response.json()

            # Convert to DataFrame
            # VALD typically returns: {time: [...], force: [...]}
            df = pd.DataFrame(trace_data)

            return df

        else:
            print(f"Error fetching trace: {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        print(f"Exception: {e}")
        return pd.DataFrame()


def get_force_trace(test_id: str, trial_id: str, token: str, tenant_id: str, region: str = 'euw') -> pd.DataFrame:
    """
    Download raw force-time trace from VALD API with caching.
    Wrapper that adds Streamlit caching when available.
    """
    if STREAMLIT_AVAILABLE:
        @st.cache_data(ttl=3600, show_spinner=False)
        def _cached_fetch(test_id, trial_id, token, tenant_id, region):
            return _get_force_trace_impl(test_id, trial_id, token, tenant_id, region)
        return _cached_fetch(test_id, trial_id, token, tenant_id, region)
    else:
        return _get_force_trace_impl(test_id, trial_id, token, tenant_id, region)


def detect_phases(force_trace: pd.DataFrame, test_type: str = 'CMJ') -> Dict[str, Tuple[int, int]]:
    """
    Detect movement phases in force trace

    For CMJ:
    - Unweighting: Force drops below bodyweight
    - Eccentric: Downward movement (negative velocity)
    - Amortization: Transition between eccentric and concentric
    - Concentric: Upward movement (positive velocity, force above BW)
    - Flight: Force = 0
    - Landing: Force spike upon contact

    Returns:
    --------
    Dict with phase names and (start_index, end_index) tuples
    """

    phases = {}

    if test_type == 'CMJ':
        # Simplified phase detection
        force = force_trace['force_n'].values

        # Estimate bodyweight (average of first 500ms)
        bodyweight = np.mean(force[:500])

        # Find takeoff (force = 0)
        takeoff_idx = np.where(force < 10)[0]
        if len(takeoff_idx) > 0:
            takeoff = takeoff_idx[0]
        else:
            takeoff = len(force)

        # Find unweighting start (force drops below 90% BW)
        unweight_threshold = bodyweight * 0.9
        unweight_idx = np.where(force < unweight_threshold)[0]
        if len(unweight_idx) > 0:
            unweight_start = unweight_idx[0]
        else:
            unweight_start = 0

        # Find lowest point (minimum force before takeoff)
        lowest_point = np.argmin(force[:takeoff])

        # Phases
        phases['quiet_stance'] = (0, unweight_start)
        phases['unweighting'] = (unweight_start, lowest_point)
        phases['eccentric'] = (unweight_start, lowest_point)
        phases['concentric'] = (lowest_point, takeoff)

        if takeoff < len(force):
            # Find landing (force spike after flight)
            flight_end = np.where(force[takeoff:] > bodyweight * 0.5)[0]
            if len(flight_end) > 0:
                landing_start = takeoff + flight_end[0]
                phases['flight'] = (takeoff, landing_start)
                phases['landing'] = (landing_start, len(force))

    return phases


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_force_trace(
    force_trace: pd.DataFrame,
    title: str = "Force-Time Curve",
    show_phases: bool = True,
    test_type: str = 'CMJ'
) -> go.Figure:
    """
    Plot a single force-time curve

    Parameters:
    -----------
    force_trace : pd.DataFrame
        Force trace data with time and force columns
    title : str
        Chart title
    show_phases : bool
        Whether to show phase annotations
    test_type : str
        Test type (CMJ, SJ, IMTP, etc.)

    Returns:
    --------
    Plotly Figure object
    """

    fig = go.Figure()

    # Main force trace
    fig.add_trace(go.Scatter(
        x=force_trace['time_ms'],
        y=force_trace['force_n'],
        mode='lines',
        name='Total Force',
        line=dict(color='#0d4f3c', width=2)
    ))

    # Add bilateral traces if available
    if 'force_left_n' in force_trace.columns:
        fig.add_trace(go.Scatter(
            x=force_trace['time_ms'],
            y=force_trace['force_left_n'],
            mode='lines',
            name='Left',
            line=dict(color='#3498db', width=1, dash='dash')
        ))

    if 'force_right_n' in force_trace.columns:
        fig.add_trace(go.Scatter(
            x=force_trace['time_ms'],
            y=force_trace['force_right_n'],
            mode='lines',
            name='Right',
            line=dict(color='#e74c3c', width=1, dash='dash')
        ))

    # Add phase annotations
    if show_phases:
        phases = detect_phases(force_trace, test_type)

        phase_colors = {
            'quiet_stance': 'rgba(128, 128, 128, 0.2)',
            'unweighting': 'rgba(255, 165, 0, 0.2)',
            'eccentric': 'rgba(255, 0, 0, 0.2)',
            'concentric': 'rgba(0, 255, 0, 0.2)',
            'flight': 'rgba(135, 206, 250, 0.2)',
            'landing': 'rgba(255, 105, 180, 0.2)'
        }

        for phase_name, (start_idx, end_idx) in phases.items():
            if start_idx < len(force_trace) and end_idx < len(force_trace):
                fig.add_vrect(
                    x0=force_trace['time_ms'].iloc[start_idx],
                    x1=force_trace['time_ms'].iloc[end_idx],
                    fillcolor=phase_colors.get(phase_name, 'rgba(128, 128, 128, 0.1)'),
                    layer="below",
                    line_width=0,
                    annotation_text=phase_name.capitalize(),
                    annotation_position="top left"
                )

    fig.update_layout(
        title=title,
        xaxis_title="Time (ms)",
        yaxis_title="Force (N)",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif'),
        height=600
    )

    return fig


def plot_multi_trial_overlay(
    traces: List[pd.DataFrame],
    trial_labels: List[str],
    title: str = "Multi-Trial Overlay"
) -> go.Figure:
    """
    Overlay multiple force traces for comparison

    Use Cases:
    - Compare trial-to-trial consistency
    - See how force profile changes over session
    - Identify optimal vs. sub-optimal patterns
    """

    fig = go.Figure()

    colors = ['#0d4f3c', '#155744', '#1e6349', '#2c7a5b', '#3a926d']

    for i, (trace, label) in enumerate(zip(traces, trial_labels)):
        fig.add_trace(go.Scatter(
            x=trace['time_ms'],
            y=trace['force_n'],
            mode='lines',
            name=label,
            line=dict(color=colors[i % len(colors)], width=2),
            opacity=0.7
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (ms)",
        yaxis_title="Force (N)",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif'),
        height=600
    )

    return fig


def plot_athlete_comparison(
    athlete_traces: Dict[str, pd.DataFrame],
    test_type: str = 'CMJ',
    title: str = "Athlete Comparison"
) -> go.Figure:
    """
    Compare force traces between different athletes

    Parameters:
    -----------
    athlete_traces : Dict[str, pd.DataFrame]
        Dictionary mapping athlete names to their force traces
    test_type : str
        Test type
    title : str
        Chart title

    Returns:
    --------
    Plotly Figure with overlaid athlete traces
    """

    fig = go.Figure()

    colors = ['#0d4f3c', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']

    for i, (athlete_name, trace) in enumerate(athlete_traces.items()):
        fig.add_trace(go.Scatter(
            x=trace['time_ms'],
            y=trace['force_n'],
            mode='lines',
            name=athlete_name,
            line=dict(color=colors[i % len(colors)], width=2)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (ms)",
        yaxis_title="Force (N)",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif'),
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def calculate_trace_metrics(force_trace: pd.DataFrame, test_type: str = 'CMJ') -> Dict:
    """
    Calculate derived metrics from force trace

    Returns:
    --------
    Dict with metrics like:
    - peak_force: Maximum force
    - average_force: Mean force
    - rfd_100ms: Rate of force development (first 100ms)
    - eccentric_duration: Time of eccentric phase
    - concentric_duration: Time of concentric phase
    - impulse: Area under curve
    """

    metrics = {}

    force = force_trace['force_n'].values
    time_ms = force_trace['time_ms'].values

    # Peak force
    metrics['peak_force'] = np.max(force)

    # Average force
    metrics['average_force'] = np.mean(force)

    # RFD (first 100ms)
    time_100ms = np.where(time_ms <= 100)[0]
    if len(time_100ms) > 1:
        force_at_100ms = force[time_100ms[-1]]
        force_at_start = force[time_100ms[0]]
        metrics['rfd_100ms'] = (force_at_100ms - force_at_start) / 0.1  # N/s

    # Impulse (area under curve)
    # Convert time to seconds
    time_s = time_ms / 1000
    # Use trapezoid (trapz was removed in NumPy 2.0)
    try:
        metrics['impulse'] = np.trapezoid(force, time_s)
    except AttributeError:
        # Fallback for older NumPy versions
        metrics['impulse'] = np.trapz(force, time_s)

    # Phase durations
    phases = detect_phases(force_trace, test_type)

    for phase_name, (start_idx, end_idx) in phases.items():
        if start_idx < len(time_ms) and end_idx < len(time_ms):
            duration_ms = time_ms[end_idx] - time_ms[start_idx]
            metrics[f'{phase_name}_duration_ms'] = duration_ms

    return metrics


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_force_trace(force_trace: pd.DataFrame, filename: str):
    """Export force trace to CSV for external analysis"""
    force_trace.to_csv(filename, index=False)
    print(f"Force trace exported to {filename}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Force Trace Visualization Module")
    print("="*50)
    print()
    print("Example Usage:")
    print()
    print("# 1. Download force trace")
    print("trace = get_force_trace(")
    print("    test_id='test-uuid',")
    print("    trial_id='trial-uuid',")
    print("    token='your-token',")
    print("    tenant_id='your-tenant'")
    print(")")
    print()
    print("# 2. Plot single trace")
    print("fig = plot_force_trace(trace, title='CMJ Force-Time Curve')")
    print("fig.show()")
    print()
    print("# 3. Overlay multiple trials")
    print("traces = [trial1, trial2, trial3]")
    print("labels = ['Trial 1', 'Trial 2', 'Trial 3']")
    print("fig = plot_multi_trial_overlay(traces, labels)")
    print("fig.show()")
    print()
    print("# 4. Compare athletes")
    print("athlete_traces = {")
    print("    'Athlete A': trace_a,")
    print("    'Athlete B': trace_b")
    print("}")
    print("fig = plot_athlete_comparison(athlete_traces)")
    print("fig.show()")
    print()
    print("# 5. Calculate metrics")
    print("metrics = calculate_trace_metrics(trace)")
    print("print(metrics)")
# Force redeploy Mon, Dec 22, 2025  7:47:29 AM
