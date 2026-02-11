"""
Test-Type Specific Dashboard Modules
=====================================

Focused, story-telling dashboard sections for specific test types:
- CMJ (Countermovement Jump) Analysis
- IMTP Single Leg Analysis
- IMTP Double Leg Analysis
- Throws Training Dashboard

Based on VALD example dashboards with research-backed insights.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime


# Helper function to get date column name
def get_date_column(df: pd.DataFrame) -> str:
    """Get the correct date column name from the dataframe."""
    if 'testDateTime' in df.columns:
        return 'testDateTime'
    elif 'recordedDateUtc' in df.columns:
        return 'recordedDateUtc'
    else:
        # Fallback - find any column with 'date' in name
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            return date_cols[0]
        return None


# ============================================================================
# CMJ (COUNTERMOVEMENT JUMP) ANALYSIS MODULE
# ============================================================================

class CMJAnalysisModule:
    """
    Comprehensive CMJ analysis matching VALD example dashboards.

    Based on:
    - CMJ Dashboard.jpg: Bilateral comparison across metrics
    - CMJ power.jpg: Power-focused progression with height overlay

    Research backing:
    - Moir (2008) - CMJ calculation methods
    - Claudino et al. (2017) - CMJ reliability (CV% <5%)
    - Gathercole et al. (2015) - CMJ for fatigue monitoring
    """

    @staticmethod
    def display_cmj_dashboard(athlete_df: pd.DataFrame, athlete_name: str):
        """
        Main CMJ dashboard display matching example layout.

        Layout:
        Row 1: Body Weight | Peak Power | Relative Peak Power
        Row 2: Jump Height | Contraction Time | Countermovement Depth

        All with bilateral (L/R) comparison where applicable.
        """
        st.markdown(f"## Counter Movement Jump Individual Analysis")
        st.markdown(f"### {athlete_name}")

        # Filter for CMJ tests only
        cmj_df = athlete_df[athlete_df['testType'].str.contains('CMJ', case=False, na=False)].copy()

        if cmj_df.empty:
            st.warning("No CMJ test data available for this athlete.")
            return

        # Sort by date
        date_col = get_date_column(cmj_df)
        if date_col:
            cmj_df[date_col] = pd.to_datetime(cmj_df[date_col])
            cmj_df = cmj_df.sort_values(date_col)
        else:
            st.warning("No date column found in data")
            return

        # Row 1: Body Weight, Peak Power, Relative Peak Power
        col1, col2, col3 = st.columns(3)

        with col1:
            CMJAnalysisModule._display_body_weight_card(cmj_df, date_col)

        with col2:
            CMJAnalysisModule._display_peak_power_card(cmj_df, date_col)

        with col3:
            CMJAnalysisModule._display_relative_power_card(cmj_df, date_col)

        st.markdown("---")

        # Row 2: Jump Height, Contraction Time, Countermovement Depth
        col1, col2, col3 = st.columns(3)

        with col1:
            CMJAnalysisModule._display_jump_height_card(cmj_df, date_col)

        with col2:
            CMJAnalysisModule._display_contraction_time_card(cmj_df, date_col)

        with col3:
            CMJAnalysisModule._display_cmj_depth_card(cmj_df, date_col)

    @staticmethod
    def _display_body_weight_card(cmj_df: pd.DataFrame, date_col: str):
        """Body weight progression over time."""
        st.markdown("#### Body Weight [kg]")

        # Get body weight column
        bw_col = None
        for col in cmj_df.columns:
            if 'body' in col.lower() and 'weight' in col.lower():
                bw_col = col
                break

        if bw_col and not cmj_df[bw_col].isna().all():
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=cmj_df[date_col],
                y=cmj_df[bw_col],
                mode='lines+markers',
                line=dict(color='#00B4D8', width=3),
                marker=dict(size=10, color='#00B4D8'),
                text=cmj_df[bw_col].round(1),
                textposition='top center',
                textfont=dict(size=12, color='#333')
            ))

            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=50),
                xaxis_title="Test Date",
                yaxis_title="Body Weight [kg]",
                showlegend=False,
                plot_bgcolor='white',
                xaxis=dict(tickformat='%d %b %Y')
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Body weight data not available")

    @staticmethod
    def _display_peak_power_card(cmj_df: pd.DataFrame, date_col: str):
        """Peak Power (W) with bilateral comparison."""
        st.markdown("#### Peak Power [W]")

        # Find peak power columns for left and right
        left_col = None
        right_col = None

        for col in cmj_df.columns:
            col_lower = col.lower()
            if 'peak' in col_lower and 'power' in col_lower:
                if 'left' in col_lower or '_l_' in col_lower or col_lower.endswith('_l'):
                    left_col = col
                elif 'right' in col_lower or '_r_' in col_lower or col_lower.endswith('_r'):
                    right_col = col

        if left_col and right_col:
            fig = go.Figure()

            # Left leg
            fig.add_trace(go.Scatter(
                x=cmj_df[date_col],
                y=cmj_df[left_col],
                mode='lines+markers',
                name='Left',
                line=dict(color='#00B4D8', width=3),
                marker=dict(size=10)
            ))

            # Right leg
            fig.add_trace(go.Scatter(
                x=cmj_df[date_col],
                y=cmj_df[right_col],
                mode='lines+markers',
                name='Right',
                line=dict(color='#0077B6', width=3),
                marker=dict(size=10)
            ))

            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=50),
                xaxis_title="Test Date",
                yaxis_title="Peak Power [W]",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor='white',
                xaxis=dict(tickformat='%d %b %Y')
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Peak power bilateral data not available")

    @staticmethod
    def _display_relative_power_card(cmj_df: pd.DataFrame, date_col: str):
        """Relative Peak Power (W/kg) with bilateral comparison or progression."""
        st.markdown("#### Relative Peak Power (W/kg)")

        # Find relative power columns - check both local_sync and legacy formats
        left_col = None
        right_col = None
        trial_col = None

        # Priority order: local_sync format first, then legacy API format
        rel_power_options = [
            'BODYMASS_RELATIVE_TAKEOFF_POWER',  # local_sync format
            'Peak Power / BM_Trial',             # legacy API format
        ]

        for col_option in rel_power_options:
            if col_option in cmj_df.columns:
                trial_col = col_option
                break

        # Fall back to fuzzy search if exact match not found
        if not trial_col:
            for col in cmj_df.columns:
                col_lower = col.lower()
                # Check for relative power (includes Peak Power / BM which is relative power)
                if 'power' in col_lower and ('/ bm' in col_lower or '/bm' in col_lower or 'bodymass_relative' in col_lower):
                    if 'left' in col_lower or '_l_' in col_lower:
                        left_col = col
                    elif 'right' in col_lower or '_r_' in col_lower:
                        right_col = col
                    elif not trial_col:
                        trial_col = col

        if left_col and right_col:
            # Display bilateral comparison
            recent_df = cmj_df.tail(3)

            fig = go.Figure()

            x_labels = [d.strftime('%d %b %Y') for d in recent_df[date_col]]

            fig.add_trace(go.Bar(
                x=x_labels,
                y=recent_df[left_col],
                name='Left',
                marker_color='#00B4D8',
                text=recent_df[left_col].round(1),
                textposition='outside'
            ))

            fig.add_trace(go.Bar(
                x=x_labels,
                y=recent_df[right_col],
                name='Right',
                marker_color='#0077B6',
                text=recent_df[right_col].round(1),
                textposition='outside'
            ))

            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=50),
                yaxis_title="Relative Peak Power [W/kg]",
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor='white'
            )

            st.plotly_chart(fig, use_container_width=True)

        elif trial_col:
            # Display progression over time using Trial-level data
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=cmj_df[date_col],
                y=cmj_df[trial_col],
                mode='lines+markers',
                line=dict(color='#E63946', width=3),
                marker=dict(size=10, color='#E63946'),
                name='Relative Peak Power',
                text=cmj_df[trial_col].round(1),
                hovertemplate='<b>%{x|%d %b %Y}</b><br>Power: %{y:.1f} W/kg<extra></extra>'
            ))

            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=50),
                yaxis_title="Relative Peak Power [W/kg]",
                xaxis_title="",
                plot_bgcolor='white',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Relative power data not available")

    @staticmethod
    def _display_jump_height_card(cmj_df: pd.DataFrame, date_col: str):
        """Jump Height with bilateral comparison or progression."""
        st.markdown("#### Jump Height (Imp-Mom) [cm]")

        # Find jump height columns - prioritize Impulse-Momentum method
        left_col = None
        right_col = None
        trial_col = None

        # Try Impulse-Momentum first (preferred), then Impulse-Displacement, then Flight Time
        if 'Jump Height (Imp-Mom)_Trial' in cmj_df.columns:
            trial_col = 'Jump Height (Imp-Mom)_Trial'
        elif 'Jump Height (Imp-Dis)_Trial' in cmj_df.columns:
            trial_col = 'Jump Height (Imp-Dis)_Trial'
        elif 'Jump Height (Flight Time)_Trial' in cmj_df.columns:
            trial_col = 'Jump Height (Flight Time)_Trial'

        # Fall back to search if exact match not found
        if not trial_col:
            for col in cmj_df.columns:
                col_lower = col.lower()
                if 'jump' in col_lower and 'height' in col_lower:
                    if 'left' in col_lower or '_l_' in col_lower:
                        left_col = col
                    elif 'right' in col_lower or '_r_' in col_lower:
                        right_col = col
                    elif 'trial' in col_lower and 'imp-mom' in col_lower:
                        trial_col = col
                    elif 'trial' in col_lower and 'imp' in col_lower and not trial_col:
                        trial_col = col

        if left_col and right_col:
            # Display bilateral comparison
            recent_df = cmj_df.tail(3)

            fig = go.Figure()

            x_labels = [d.strftime('%d %b %Y') for d in recent_df[date_col]]

            fig.add_trace(go.Bar(
                x=x_labels,
                y=recent_df[left_col],
                name='Left',
                marker_color='#00B4D8',
                text=recent_df[left_col].round(0),
                textposition='outside'
            ))

            fig.add_trace(go.Bar(
                x=x_labels,
                y=recent_df[right_col],
                name='Right',
                marker_color='#0077B6',
                text=recent_df[right_col].round(0),
                textposition='outside'
            ))

            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=80),
                yaxis_title="Jump Height [cm]",
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                plot_bgcolor='white'
            )

            st.plotly_chart(fig, use_container_width=True)

        elif trial_col:
            # Display progression over time using Trial-level data
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=cmj_df[date_col],
                y=cmj_df[trial_col],
                mode='lines+markers',
                line=dict(color='#00B4D8', width=3),
                marker=dict(size=10, color='#00B4D8'),
                name='Jump Height',
                text=cmj_df[trial_col].round(1),
                hovertemplate='<b>%{x|%d %b %Y}</b><br>Height: %{y:.1f} cm<extra></extra>'
            ))

            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=80),
                yaxis_title="Jump Height [cm]",
                xaxis_title="",
                plot_bgcolor='white',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Jump height data not available")

    @staticmethod
    def _display_contraction_time_card(cmj_df: pd.DataFrame, date_col: str):
        """Contraction Time (ms) with bilateral comparison or progression."""
        st.markdown("#### Contraction Time [ms]")

        # Find contraction time columns - use exact match first for reliability
        left_col = None
        right_col = None
        trial_col = None

        # Try exact match first
        if 'Contraction Time_Trial' in cmj_df.columns:
            trial_col = 'Contraction Time_Trial'

        # Fall back to search if exact match not found
        if not trial_col:
            for col in cmj_df.columns:
                col_lower = col.lower()
                if 'contraction' in col_lower and 'time' in col_lower:
                    if 'left' in col_lower or '_l_' in col_lower:
                        left_col = col
                    elif 'right' in col_lower or '_r_' in col_lower:
                        right_col = col
                    elif 'trial' in col_lower:
                        trial_col = col

        if left_col and right_col:
            # Display bilateral comparison
            recent_df = cmj_df.tail(3)

            fig = go.Figure()

            x_labels = [d.strftime('%d %b %Y') for d in recent_df[date_col]]

            fig.add_trace(go.Bar(
                x=x_labels,
                y=recent_df[left_col],
                name='Left',
                marker_color='#00B4D8',
                text=recent_df[left_col].round(0),
                textposition='outside'
            ))

            fig.add_trace(go.Bar(
                x=x_labels,
                y=recent_df[right_col],
                name='Right',
                marker_color='#0077B6',
                text=recent_df[right_col].round(0),
                textposition='outside'
            ))

            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=80),
                yaxis_title="Contraction Time [ms]",
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                plot_bgcolor='white'
            )

            st.plotly_chart(fig, use_container_width=True)

        elif trial_col:
            # Display progression over time using Trial-level data
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=cmj_df[date_col],
                y=cmj_df[trial_col],
                mode='lines+markers',
                line=dict(color='#0077B6', width=3),
                marker=dict(size=10, color='#0077B6'),
                name='Contraction Time',
                text=cmj_df[trial_col].round(0),
                hovertemplate='<b>%{x|%d %b %Y}</b><br>Time: %{y:.0f} ms<extra></extra>'
            ))

            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=80),
                yaxis_title="Contraction Time [ms]",
                xaxis_title="",
                plot_bgcolor='white',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Contraction time data not available")

    @staticmethod
    def _display_cmj_depth_card(cmj_df: pd.DataFrame, date_col: str):
        """Countermovement Depth (cm) with bilateral comparison or progression."""
        st.markdown("#### Countermovement Depth [cm]")

        # Find CMJ depth columns - check both local_sync and legacy formats
        left_col = None
        right_col = None
        trial_col = None

        # Priority order: local_sync format first, then legacy API format
        depth_options = [
            'COUNTERMOVEMENT_DEPTH',           # local_sync format
            'Countermovement Depth_Trial',     # legacy API format
        ]

        for col_option in depth_options:
            if col_option in cmj_df.columns:
                trial_col = col_option
                break

        # Fall back to fuzzy search if exact match not found
        if not trial_col:
            for col in cmj_df.columns:
                col_lower = col.lower()
                if ('countermovement' in col_lower or 'cmj' in col_lower) and 'depth' in col_lower:
                    if 'left' in col_lower or '_l_' in col_lower:
                        left_col = col
                    elif 'right' in col_lower or '_r_' in col_lower:
                        right_col = col
                    elif not trial_col:
                        trial_col = col

        if left_col and right_col:
            # Display bilateral comparison
            recent_df = cmj_df.tail(3)

            fig = go.Figure()

            x_labels = [d.strftime('%d %b %Y') for d in recent_df[date_col]]

            # Depth is typically negative, so display accordingly
            fig.add_trace(go.Bar(
                x=x_labels,
                y=recent_df[left_col],
                name='Left',
                marker_color='#00B4D8',
                text=recent_df[left_col].round(0),
                textposition='outside'
            ))

            fig.add_trace(go.Bar(
                x=x_labels,
                y=recent_df[right_col],
                name='Right',
                marker_color='#0077B6',
                text=recent_df[right_col].round(0),
                textposition='outside'
            ))

            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=80),
                yaxis_title="Countermovement Depth [cm]",
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                plot_bgcolor='white'
            )

            st.plotly_chart(fig, use_container_width=True)

        elif trial_col:
            # Display progression over time using Trial-level data
            fig = go.Figure()

            # Use absolute values since countermovement depth is stored as negative
            depth_values = cmj_df[trial_col].abs()

            fig.add_trace(go.Scatter(
                x=cmj_df[date_col],
                y=depth_values,
                mode='lines+markers',
                line=dict(color='#90BE6D', width=3),
                marker=dict(size=10, color='#90BE6D'),
                name='Countermovement Depth',
                text=depth_values.round(1),
                hovertemplate='<b>%{x|%d %b %Y}</b><br>Depth: %{y:.1f} cm<extra></extra>'
            ))

            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=80),
                yaxis_title="Countermovement Depth [cm]",
                xaxis_title="",
                plot_bgcolor='white',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("CMJ depth data not available")

    @staticmethod
    def display_cmj_power_focus(athlete_df: pd.DataFrame, athlete_name: str):
        """
        Power-focused CMJ dashboard matching CMJ power.jpg example.

        Shows:
        - Current performance card (Relative Power, Body Mass, CMJ Height)
        - Combined chart: Height (bars) + Relative Power (line overlay)
        """
        st.markdown("## POWER: CMJ")

        # Filter for CMJ tests
        cmj_df = athlete_df[athlete_df['testType'].str.contains('CMJ', case=False, na=False)].copy()

        if cmj_df.empty:
            st.warning("No CMJ test data available.")
            return

        # Get correct date column
        date_col = get_date_column(cmj_df)
        if not date_col:
            st.warning("No date column found in data")
            return

        cmj_df[date_col] = pd.to_datetime(cmj_df[date_col])
        cmj_df = cmj_df.sort_values(date_col)

        # Current Performance Card
        st.markdown("### Current CMJ Performance")

        # Get latest test values
        latest = cmj_df.iloc[-1]

        # Find relevant columns - check both local_sync and legacy formats
        rel_power_col = None
        body_mass_col = None
        height_col = None

        # Priority order: local_sync format first, then legacy API format
        rel_power_options = ['BODYMASS_RELATIVE_TAKEOFF_POWER', 'Peak Power / BM_Trial']
        height_options = ['JUMP_HEIGHT_IMP_MOM', 'JUMP_HEIGHT', 'Jump Height (Imp-Mom)_Trial', 'Jump Height (Imp-Dis)_Trial']
        body_mass_options = ['BODY_WEIGHT', 'Weight_Trial', 'Body Mass_Trial']

        for col_option in rel_power_options:
            if col_option in cmj_df.columns:
                rel_power_col = col_option
                break

        for col_option in height_options:
            if col_option in cmj_df.columns:
                height_col = col_option
                break

        for col_option in body_mass_options:
            if col_option in cmj_df.columns:
                body_mass_col = col_option
                break

        # Fall back to fuzzy search if exact matches not found
        if not rel_power_col or not height_col or not body_mass_col:
            for col in cmj_df.columns:
                col_lower = col.lower()
                if not rel_power_col and ('/ bm' in col_lower or '/bm' in col_lower or 'bodymass_relative' in col_lower) and 'power' in col_lower:
                    rel_power_col = col
                elif not body_mass_col and ('body' in col_lower or 'weight' in col_lower) and col_lower not in ['weight_relative', 'body_relative']:
                    body_mass_col = col
                elif not height_col and 'jump' in col_lower and 'height' in col_lower:
                    height_col = col

        # Display current values
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Relative Power**")
            if rel_power_col:
                st.markdown(f"## {latest[rel_power_col]:.1f} W/kg")
            else:
                st.markdown("## N/A")

        with col2:
            st.markdown("**Body Mass**")
            if body_mass_col:
                st.markdown(f"## {latest[body_mass_col]:.1f} kg")
            else:
                st.markdown("## N/A")

        with col3:
            st.markdown("**CMJ Height**")
            if height_col:
                # Convert m to cm if local_sync format (values < 1 are likely in meters)
                height_val = latest[height_col]
                if height_val < 1:
                    height_val = height_val * 100
                st.markdown(f"## {height_val:.1f} cm")
            else:
                st.markdown("## N/A")

        st.markdown("---")

        # Combined chart: Height bars + Power line
        if height_col and rel_power_col:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Convert height to cm if stored in meters (local_sync format)
            height_values = cmj_df[height_col].copy()
            if height_values.max() < 1:
                height_values = height_values * 100

            # Height bars (primary y-axis)
            fig.add_trace(
                go.Bar(
                    x=cmj_df[date_col],
                    y=height_values,
                    name='CMJ Height',
                    marker_color='#0000FF',
                    text=height_values.round(1),
                    textposition='inside',
                    textfont=dict(color='white', size=12)
                ),
                secondary_y=False
            )

            # Relative power line (secondary y-axis)
            fig.add_trace(
                go.Scatter(
                    x=cmj_df[date_col],
                    y=cmj_df[rel_power_col],
                    name='Relative Power',
                    mode='lines+markers',
                    line=dict(color='#333', width=3),
                    marker=dict(size=10, color='#666')
                ),
                secondary_y=True
            )

            fig.update_xaxes(title_text="Date", tickformat='%d %b %Y')
            fig.update_yaxes(title_text="Height [cm]", secondary_y=False)
            fig.update_yaxes(title_text="Relative Power [W/kg]", secondary_y=True)

            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=40, b=80),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor='white'
            )

            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# ISOMETRIC (IMTP) ANALYSIS MODULES
# ============================================================================

class IsometricSingleLegModule:
    """
    Single Leg Isometric Peak Force analysis.

    Based on: Isometrics Single Leg.jpg example

    Shows:
    - Max Force display with bodyweight
    - Bilateral comparison chart (Left vs Right)
    """

    @staticmethod
    def display_single_leg_analysis(athlete_df: pd.DataFrame, athlete_name: str):
        """Display single leg isometric analysis."""
        st.markdown("## SL ISO PEAK FORCE")

        # Filter for single leg isometric tests
        iso_df = athlete_df[
            (athlete_df['testType'].str.contains('ISO', case=False, na=False)) |
            (athlete_df['testType'].str.contains('IMTP', case=False, na=False))
        ].copy()

        # Further filter for single leg if possible
        single_leg_df = iso_df[
            iso_df['testType'].str.contains('Single', case=False, na=False) |
            iso_df['testType'].str.contains('SL', case=False, na=False)
        ].copy()

        if single_leg_df.empty:
            single_leg_df = iso_df  # Fallback to all isometric

        if single_leg_df.empty:
            st.warning("No single leg isometric test data available.")
            return

        # Get correct date column
        date_col = get_date_column(single_leg_df)
        if not date_col:
            st.warning("No date column found in data")
            return

        single_leg_df[date_col] = pd.to_datetime(single_leg_df[date_col])
        single_leg_df = single_leg_df.sort_values(date_col)

        # Get latest test
        latest = single_leg_df.iloc[-1]

        # Find peak force columns
        left_force_col = None
        right_force_col = None
        bw_col = None

        for col in single_leg_df.columns:
            col_lower = col.lower()
            if 'peak' in col_lower and 'force' in col_lower:
                if 'left' in col_lower or '_l_' in col_lower:
                    left_force_col = col
                elif 'right' in col_lower or '_r_' in col_lower:
                    right_force_col = col
            elif 'body' in col_lower and 'weight' in col_lower:
                bw_col = col

        # Display current values
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Max Force. N.1**")
            if left_force_col:
                max_force = latest[left_force_col] if not pd.isna(latest[left_force_col]) else 0
                st.markdown(f"## {max_force:.0f} N")
            else:
                st.markdown("## N/A")

        with col2:
            st.markdown("**BW.N**")
            if bw_col:
                bw = latest[bw_col] if not pd.isna(latest[bw_col]) else 0
                st.markdown(f"## {bw:.0f} N")
            else:
                st.markdown("## N/A")

        st.markdown("---")

        # Bilateral comparison chart
        if left_force_col and right_force_col:
            st.markdown("#### Absolute Peak Force (Newtons)")

            # Use only most recent test for side-by-side comparison
            recent_df = single_leg_df.tail(1)

            fig = go.Figure()

            date_label = recent_df[date_col].iloc[0].strftime('%d %b %Y')

            fig.add_trace(go.Bar(
                x=['Left'],
                y=[recent_df[left_force_col].iloc[0]],
                name='Left',
                marker_color='#00CED1',
                text=[f"{recent_df[left_force_col].iloc[0]:.0f}"],
                textposition='inside',
                textfont=dict(size=16, color='white'),
                width=0.4
            ))

            fig.add_trace(go.Bar(
                x=['Right'],
                y=[recent_df[right_force_col].iloc[0]],
                name='Right',
                marker_color='#2F4F4F',
                text=[f"{recent_df[right_force_col].iloc[0]:.0f}"],
                textposition='inside',
                textfont=dict(size=16, color='white'),
                width=0.4
            ))

            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=60),
                yaxis_title="Force (Newtons)",
                xaxis=dict(title="Leg Type"),
                showlegend=False,
                plot_bgcolor='white',
                annotations=[
                    dict(
                        text=f"Test Date: {date_label}",
                        xref="paper", yref="paper",
                        x=0.5, y=-0.15,
                        showarrow=False,
                        font=dict(size=12)
                    )
                ]
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Bilateral force data not available")


class IsometricDoubleLegModule:
    """
    Double Leg Isometric (IMTP) analysis.

    Based on: Isometrics Double leg.jpg example

    Shows:
    - Peak Force over time
    - RFD analysis
    - Force @ specific time points (100ms, 200ms, etc.)
    """

    @staticmethod
    def display_double_leg_analysis(athlete_df: pd.DataFrame, athlete_name: str):
        """Display double leg isometric (IMTP) analysis."""
        st.markdown("## IMTP - Double Leg Analysis")

        # Filter for double leg/bilateral isometric tests
        iso_df = athlete_df[
            (athlete_df['testType'].str.contains('ISO', case=False, na=False)) |
            (athlete_df['testType'].str.contains('IMTP', case=False, na=False))
        ].copy()

        # Filter for double leg
        double_leg_df = iso_df[
            ~(iso_df['testType'].str.contains('Single', case=False, na=False)) &
            ~(iso_df['testType'].str.contains('SL', case=False, na=False))
        ].copy()

        if double_leg_df.empty:
            double_leg_df = iso_df  # Fallback

        if double_leg_df.empty:
            st.warning("No double leg isometric test data available.")
            return

        # Get correct date column
        date_col = get_date_column(double_leg_df)
        if not date_col:
            st.warning("No date column found in data")
            return

        double_leg_df[date_col] = pd.to_datetime(double_leg_df[date_col])
        double_leg_df = double_leg_df.sort_values(date_col)

        # Find relevant columns
        peak_force_col = None
        rfd_col = None
        force_100ms_col = None
        force_200ms_col = None

        for col in double_leg_df.columns:
            col_lower = col.lower()
            if 'peak' in col_lower and 'force' in col_lower and peak_force_col is None:
                peak_force_col = col
            elif 'rfd' in col_lower and rfd_col is None:
                rfd_col = col
            elif '100' in col and 'force' in col_lower:
                force_100ms_col = col
            elif '200' in col and 'force' in col_lower:
                force_200ms_col = col

        # Display latest values
        latest = double_leg_df.iloc[-1]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Peak Force [N]**")
            if peak_force_col:
                st.markdown(f"## {latest[peak_force_col]:.0f}")
            else:
                st.markdown("## N/A")

        with col2:
            st.markdown("**RFD [N/s]**")
            if rfd_col:
                st.markdown(f"## {latest[rfd_col]:.0f}")
            else:
                st.markdown("## N/A")

        with col3:
            st.markdown("**Force @ 200ms [N]**")
            if force_200ms_col:
                st.markdown(f"## {latest[force_200ms_col]:.0f}")
            else:
                st.markdown("## N/A")

        st.markdown("---")

        # Peak Force progression
        if peak_force_col:
            st.markdown("#### Peak Force Progression")

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=double_leg_df[date_col],
                y=double_leg_df[peak_force_col],
                mode='lines+markers',
                line=dict(color='#0077B6', width=3),
                marker=dict(size=10, color='#00B4D8'),
                text=double_leg_df[peak_force_col].round(0),
                textposition='top center'
            ))

            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=20, b=60),
                xaxis_title="Test Date",
                yaxis_title="Peak Force [N]",
                showlegend=False,
                plot_bgcolor='white',
                xaxis=dict(tickformat='%d %b %Y')
            )

            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# THROWS TRAINING DASHBOARD MODULE
# ============================================================================

class ThrowsTrainingModule:
    """
    Athletics Throws Training Dashboard.

    Based on: THROWS.jpeg example

    Layout:
    - Training Distances (future: external data)
    - Mid Thigh Pull (IMTP) - Peak Force with 3-test trend
    - Peak Power - Absolute & Relative
    - CMJ Depth - Technical consistency
    """

    @staticmethod
    def display_throws_dashboard(athlete_df: pd.DataFrame, athlete_name: str, sport: str):
        """Display throws training dashboard - matches THROWS.jpeg example."""
        st.markdown("## 🥏 Athletics - Throws Training Dashboard")
        st.markdown(f"### {athlete_name} - {sport}")
        st.markdown("---")

        # Layout: 2x2 grid matching THROWS.jpeg
        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            ThrowsTrainingModule._display_training_distances_section(athlete_df, athlete_name)

        with row1_col2:
            ThrowsTrainingModule._display_imtp_section(athlete_df)

        st.markdown("<br>", unsafe_allow_html=True)

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            ThrowsTrainingModule._display_peak_power_section(athlete_df)

        with row2_col2:
            ThrowsTrainingModule._display_cmj_depth_section(athlete_df)

    @staticmethod
    def _display_training_distances_section(athlete_df: pd.DataFrame, athlete_name: str = None):
        """Training Distances - Display recorded training data or placeholder."""
        st.markdown("#### 📊 Training Distances")

        # Try to load training distances from session state or CSV
        import os
        training_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training_distances.csv')
        training_df = pd.DataFrame()

        # Check session state first
        if hasattr(st, 'session_state') and 'training_distances' in st.session_state:
            training_df = st.session_state.training_distances

        # Fall back to CSV file
        if training_df.empty and os.path.exists(training_data_path):
            try:
                training_df = pd.read_csv(training_data_path)
                if 'date' in training_df.columns:
                    training_df['date'] = pd.to_datetime(training_df['date'])
            except Exception:
                pass

        # Color mapping for session types
        session_colors = {
            'Training': ('#1D4D3B', '#153829'),      # Teal gradient
            'Competition': ('#FFB800', '#E5A600'),   # Gold gradient
            'Testing': ('#0077B6', '#005a8c'),       # Blue gradient
            'Warm-up': ('#6c757d', '#495057')        # Gray gradient
        }

        session_icons = {
            'Training': '🟢',
            'Competition': '🏆',
            'Testing': '🔵',
            'Warm-up': '⚪'
        }

        # Filter for this athlete if name provided
        if not training_df.empty and athlete_name:
            athlete_training = training_df[training_df['athlete'] == athlete_name].copy()

            if not athlete_training.empty:
                # Sort by date
                athlete_training = athlete_training.sort_values('date', ascending=False)

                # Get latest throws by event
                latest_throws = athlete_training.groupby('event').first().reset_index()

                # Display metrics
                if len(latest_throws) > 0:
                    for _, row in latest_throws.iterrows():
                        event = row['event']
                        distance = row['distance_m']
                        date = pd.to_datetime(row['date']).strftime('%d %b')
                        implement = row.get('implement_kg', 'N/A')
                        session_type = row.get('session_type', 'Training')

                        # Get colors based on session type
                        color1, color2 = session_colors.get(session_type, ('#1D4D3B', '#153829'))
                        icon = session_icons.get(session_type, '🟢')

                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, {color1} 0%, {color2} 100%);
                            border-radius: 8px;
                            padding: 0.75rem 1rem;
                            margin-bottom: 0.5rem;
                            color: white;
                        ">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-weight: 600;">{icon} {event}</span>
                                <span style="font-size: 1.2rem; font-weight: bold;">{distance:.2f}m</span>
                            </div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 0.25rem;">
                                {implement}kg • {date} • {session_type}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Show trend chart for most recent event with color by session type
                    most_recent_event = athlete_training['event'].iloc[0]
                    event_data = athlete_training[athlete_training['event'] == most_recent_event].sort_values('date').tail(8)

                    if len(event_data) > 1:
                        fig = go.Figure()

                        # Color mapping for chart
                        chart_colors = {
                            'Training': '#1D4D3B',
                            'Competition': '#FFB800',
                            'Testing': '#0077B6',
                            'Warm-up': '#6c757d'
                        }

                        # Add points colored by session type
                        if 'session_type' in event_data.columns:
                            for session_type in event_data['session_type'].unique():
                                session_data = event_data[event_data['session_type'] == session_type]
                                color = chart_colors.get(session_type, '#1D4D3B')

                                fig.add_trace(go.Scatter(
                                    x=session_data['date'],
                                    y=session_data['distance_m'],
                                    mode='markers',
                                    name=session_type,
                                    marker=dict(
                                        size=10 if session_type == 'Competition' else 8,
                                        color=color,
                                        symbol='star' if session_type == 'Competition' else 'circle'
                                    ),
                                    hovertemplate=f'{session_type}: %{{y:.2f}}m<extra></extra>'
                                ))

                            # Connecting line
                            fig.add_trace(go.Scatter(
                                x=event_data['date'],
                                y=event_data['distance_m'],
                                mode='lines',
                                line=dict(color='rgba(0,113,103,0.3)', width=1),
                                hoverinfo='skip',
                                showlegend=False
                            ))
                        else:
                            fig.add_trace(go.Scatter(
                                x=event_data['date'],
                                y=event_data['distance_m'],
                                mode='markers+lines',
                                marker=dict(size=8, color='#1D4D3B'),
                                line=dict(color='#1D4D3B', width=2),
                                hovertemplate='%{y:.2f}m<extra></extra>'
                            ))

                        fig.update_layout(
                            height=180,
                            margin=dict(l=10, r=10, t=10, b=30),
                            showlegend=True,
                            legend=dict(orientation='h', yanchor='bottom', y=1.0, font=dict(size=9)),
                            plot_bgcolor='white',
                            xaxis=dict(showgrid=False),
                            yaxis=dict(title="m", showgrid=True, gridcolor='lightgray')
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Summary stats
                    total_throws = len(athlete_training)
                    comp_throws = len(athlete_training[athlete_training.get('session_type', '') == 'Competition']) if 'session_type' in athlete_training.columns else 0

                    st.markdown(f"*{total_throws} throws recorded ({comp_throws} competitions)*")
                    return

        # Show placeholder if no data
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px;
            padding: 1.5rem;
            border-left: 5px solid #0077B6;
            margin-bottom: 1rem;
        ">
            <p style="margin: 0; color: #495057; font-size: 0.95rem;">
                <strong>No throws recorded</strong> - Use the <strong>✏️ Data Entry</strong> tab to record training distances.
            </p>
            <p style="margin: 0.5rem 0 0 0; color: #6c757d; font-size: 0.85rem;">
                Supported: Shot Put, Discus, Javelin, Hammer (various implement weights)
            </p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def _display_imtp_section(athlete_df: pd.DataFrame):
        """Mid Thigh Pull (IMTP) Peak Force section - enhanced to match THROWS.jpeg."""
        st.markdown("#### 💪 Mid Thigh Pull")

        # Filter IMTP tests
        imtp_df = athlete_df[
            (athlete_df['testType'].str.contains('IMTP', case=False, na=False)) |
            (athlete_df['testType'].str.contains('ISO', case=False, na=False))
        ].copy()

        if imtp_df.empty:
            st.warning("No IMTP data available")
            return

        # Get correct date column
        date_col = get_date_column(imtp_df)
        if not date_col:
            st.warning("No date column found in data")
            return

        imtp_df[date_col] = pd.to_datetime(imtp_df[date_col])
        imtp_df = imtp_df.sort_values(date_col).tail(5)  # Last 5 tests

        # Find peak force and body mass columns
        peak_force_col = None
        body_mass_col = None

        for col in imtp_df.columns:
            col_lower = col.lower()
            if 'peak' in col_lower and 'force' in col_lower and not peak_force_col:
                peak_force_col = col
            elif ('body' in col_lower or 'mass' in col_lower or 'weight' in col_lower) and not body_mass_col:
                body_mass_col = col

        if peak_force_col:
            # Latest values
            latest_force = imtp_df[peak_force_col].iloc[-1]

            # Calculate relative force if body mass available
            relative_force = None
            if body_mass_col and pd.notna(imtp_df[body_mass_col].iloc[-1]):
                body_mass = imtp_df[body_mass_col].iloc[-1]
                relative_force = latest_force / body_mass

            # Display metrics in columns
            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.metric("Peak Force", f"{latest_force:.0f} N")

            with metric_col2:
                if relative_force:
                    st.metric("Relative Force", f"{relative_force:.1f} N/kg")
                else:
                    st.metric("Relative Force", "N/A")

            # 3-test trend bars - filter out NaT dates
            recent_3 = imtp_df.tail(3).copy()
            recent_3 = recent_3[recent_3[date_col].notna()]

            if not recent_3.empty:
                fig = go.Figure()

                # Safe date formatting - handle NaT values
                date_labels = []
                for d in recent_3[date_col]:
                    if pd.notna(d):
                        date_labels.append(d.strftime('%d-%b'))
                    else:
                        date_labels.append('Unknown')

                fig.add_trace(go.Bar(
                    x=date_labels,
                    y=recent_3[peak_force_col],
                    marker_color='#0077B6',
                    text=recent_3[peak_force_col].round(0),
                    textposition='outside',
                    textfont=dict(size=11)
                ))

                fig.update_layout(
                    height=250,
                    margin=dict(l=10, r=10, t=10, b=40),
                    yaxis_title="Force [N]",
                    showlegend=False,
                    plot_bgcolor='white',
                    font=dict(family='Inter, sans-serif')
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid date data for trend chart")
        else:
            st.info("Peak force data not found")

    @staticmethod
    def _display_peak_power_section(athlete_df: pd.DataFrame):
        """Peak Power (Absolute & Relative) section - enhanced to match THROWS.jpeg."""
        st.markdown("#### ⚡ Peak Power")

        # Filter CMJ tests for power
        cmj_df = athlete_df[athlete_df['testType'].str.contains('CMJ', case=False, na=False)].copy()

        if cmj_df.empty:
            st.warning("No CMJ power data available")
            return

        # Get correct date column
        date_col = get_date_column(cmj_df)
        if not date_col:
            st.warning("No date column found in data")
            return

        cmj_df[date_col] = pd.to_datetime(cmj_df[date_col])
        cmj_df_sorted = cmj_df.sort_values(date_col)

        # Find power columns - check both local_sync and legacy formats
        abs_power_col = None
        rel_power_col = None

        # Priority order: local_sync format first, then legacy API format
        abs_power_options = [
            'PEAK_TAKEOFF_POWER',       # local_sync format
            'Peak Power_Trial',          # legacy API format
        ]
        rel_power_options = [
            'BODYMASS_RELATIVE_TAKEOFF_POWER',  # local_sync format
            'Peak Power / BM_Trial',             # legacy API format
        ]

        for col_option in abs_power_options:
            if col_option in cmj_df.columns:
                abs_power_col = col_option
                break

        for col_option in rel_power_options:
            if col_option in cmj_df.columns:
                rel_power_col = col_option
                break

        # Fall back to fuzzy search if exact match not found
        if not abs_power_col or not rel_power_col:
            for col in cmj_df.columns:
                col_lower = col.lower()
                if 'peak' in col_lower and 'power' in col_lower:
                    if ('/ bm' in col_lower or '/bm' in col_lower or 'relative' in col_lower or 'bodymass' in col_lower) and not rel_power_col:
                        rel_power_col = col
                    elif not abs_power_col and '/ bm' not in col_lower and '/bm' not in col_lower and 'relative' not in col_lower and 'bodymass' not in col_lower:
                        abs_power_col = col

        if abs_power_col:
            # Latest values
            latest_abs = cmj_df_sorted[abs_power_col].iloc[-1]
            all_time_best = cmj_df_sorted[abs_power_col].max()
            avg_5 = cmj_df_sorted[abs_power_col].tail(5).mean()

            # Absolute Power Metrics
            st.markdown("**Peak Power (Absolute)**")

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Latest", f"{latest_abs:.0f} W")
            with metric_col2:
                st.metric("All-Time Best", f"{all_time_best:.0f} W")
            with metric_col3:
                st.metric("Last 5 Avg", f"{avg_5:.0f} W")

            # Absolute Power Progression Chart
            fig_abs = go.Figure()

            fig_abs.add_trace(go.Scatter(
                x=cmj_df_sorted[date_col],
                y=cmj_df_sorted[abs_power_col],
                mode='lines+markers',
                line=dict(color='#0077B6', width=2),
                marker=dict(size=8, color='#0077B6'),
                name='Peak Power'
            ))

            fig_abs.update_layout(
                height=200,
                margin=dict(l=10, r=10, t=10, b=40),
                yaxis_title="Power [W]",
                showlegend=False,
                plot_bgcolor='white',
                font=dict(family='Inter, sans-serif'),
                xaxis=dict(tickformat='%d-%b')
            )

            st.plotly_chart(fig_abs, use_container_width=True)

        if rel_power_col:
            # Relative Power
            st.markdown("**Peak Power / BM (Relative)**")

            latest_rel = cmj_df_sorted[rel_power_col].iloc[-1]

            st.metric("Latest", f"{latest_rel:.1f} W/kg")

            # Relative Power Progression Chart
            fig_rel = go.Figure()

            fig_rel.add_trace(go.Scatter(
                x=cmj_df_sorted[date_col],
                y=cmj_df_sorted[rel_power_col],
                mode='lines+markers',
                line=dict(color='#00B4D8', width=2),
                marker=dict(size=8, color='#00B4D8'),
                name='Relative Power'
            ))

            fig_rel.update_layout(
                height=200,
                margin=dict(l=10, r=10, t=10, b=40),
                yaxis_title="Power [W/kg]",
                showlegend=False,
                plot_bgcolor='white',
                font=dict(family='Inter, sans-serif'),
                xaxis=dict(tickformat='%d-%b')
            )

            st.plotly_chart(fig_rel, use_container_width=True)

        if not abs_power_col and not rel_power_col:
            st.info("Power data not found")

    @staticmethod
    def _display_cmj_depth_section(athlete_df: pd.DataFrame):
        """CMJ Depth - Technical consistency - enhanced to match THROWS.jpeg."""
        st.markdown("#### 📏 Countermovement Jump Depth")

        cmj_df = athlete_df[athlete_df['testType'].str.contains('CMJ', case=False, na=False)].copy()

        if cmj_df.empty:
            st.warning("No CMJ depth data available")
            return

        # Get correct date column
        date_col = get_date_column(cmj_df)
        if not date_col:
            st.warning("No date column found in data")
            return

        cmj_df[date_col] = pd.to_datetime(cmj_df[date_col])
        cmj_df_sorted = cmj_df.sort_values(date_col)

        # Find depth column - check both local_sync and legacy formats
        depth_col = None

        # Priority order: local_sync format, then legacy API format
        depth_column_options = [
            'COUNTERMOVEMENT_DEPTH',           # local_sync format
            'Countermovement Depth_Trial',     # legacy API format
        ]

        for col_option in depth_column_options:
            if col_option in cmj_df.columns:
                depth_col = col_option
                break

        # Fallback fuzzy search if exact match not found
        if not depth_col:
            for col in cmj_df.columns:
                col_lower = col.lower()
                if ('countermovement' in col_lower or 'cmj' in col_lower) and 'depth' in col_lower:
                    depth_col = col
                    break

        if depth_col:
            # Take absolute values since countermovement depth is stored as negative
            latest_depth = abs(cmj_df_sorted[depth_col].iloc[-1])
            avg_depth = abs(cmj_df_sorted[depth_col].mean())

            # Consistency = low standard deviation (lower is better)
            std_dev = cmj_df_sorted[depth_col].tail(5).std()

            # Display metrics
            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.metric("Latest Depth", f"{latest_depth:.1f} cm")
            with metric_col2:
                st.metric("Avg Depth", f"{avg_depth:.1f} cm")

            st.caption(f"**Consistency (SD):** {std_dev:.1f} cm  •  Lower SD = More Consistent")

            # Progression Chart
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=cmj_df_sorted[date_col],
                y=cmj_df_sorted[depth_col].abs(),  # Use absolute values for display
                mode='lines+markers',
                line=dict(color='#0077B6', width=2),
                marker=dict(size=8, color='#0077B6'),
                name='CMJ Depth'
            ))

            # Add average line
            fig.add_hline(
                y=avg_depth,
                line_dash="dash",
                line_color="rgba(0, 119, 182, 0.3)",
                annotation_text=f"Avg: {avg_depth:.1f}",
                annotation_position="right"
            )

            fig.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=10, b=40),
                yaxis_title="Depth [cm]",
                showlegend=False,
                plot_bgcolor='white',
                font=dict(family='Inter, sans-serif'),
                xaxis=dict(tickformat='%d-%b')
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Depth data not found")


# ============================================================================
# MODULE ROUTER
# ============================================================================

# ============================================================================
# WEIGHTLIFTING PHYSICAL DIAGNOSTICS MODULE
# ============================================================================

class WeightliftingDiagnosticsModule:
    """
    Weightlifting Physical Diagnostics Summary.

    3-column layout:
    - Column 1: Performance Asymmetry (L/R bars for IMTP, CMJ impulse, CMJ peak force)
    - Column 2: Hip Profile (ForceFrame) + Upper Limb Profile
    - Column 3: Athlete Summary Card with auto-generated observations
    + Trunk Profile section below (DynaMo data)
    """

    # Team Saudi brand colors
    TEAL_PRIMARY = '#007167'
    TEAL_LIGHT = '#009688'
    TEAL_DARK = '#005a51'
    GOLD_ACCENT = '#a08e66'
    BLUE = '#0077B6'
    RED_FLAG = '#dc3545'

    @staticmethod
    def display_dashboard(forcedecks_df: pd.DataFrame, athlete_name: str,
                          forceframe_df: pd.DataFrame = None,
                          dynamo_df: pd.DataFrame = None):
        """Display the Weightlifting Physical Diagnostics Summary."""

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #007167 0%, #005a51 100%);
             padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid #a08e66;">
            <h2 style="color: white; margin: 0;">Physical Diagnostics Summary</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">{athlete_name} - Weightlifting</p>
        </div>
        """, unsafe_allow_html=True)

        # Get latest test data per type for this athlete
        date_col = get_date_column(forcedecks_df)
        if date_col:
            forcedecks_df[date_col] = pd.to_datetime(forcedecks_df[date_col], errors='coerce')
            forcedecks_df = forcedecks_df.sort_values(date_col, ascending=False)

        imtp_df = forcedecks_df[forcedecks_df['testType'] == 'IMTP'] if 'testType' in forcedecks_df.columns else pd.DataFrame()
        cmj_df = forcedecks_df[forcedecks_df['testType'].isin(['CMJ', 'ABCMJ'])] if 'testType' in forcedecks_df.columns else pd.DataFrame()
        isot_df = forcedecks_df[forcedecks_df['testType'] == 'ISOT'] if 'testType' in forcedecks_df.columns else pd.DataFrame()
        ppu_df = forcedecks_df[forcedecks_df['testType'] == 'PPU'] if 'testType' in forcedecks_df.columns else pd.DataFrame()

        # 3-column layout
        col1, col2, col3 = st.columns([1.2, 1.2, 1])

        with col1:
            WeightliftingDiagnosticsModule._display_asymmetry_profile(imtp_df, cmj_df, isot_df)

        with col2:
            WeightliftingDiagnosticsModule._display_body_profile(forceframe_df, ppu_df, cmj_df)

        with col3:
            WeightliftingDiagnosticsModule._display_athlete_card(
                athlete_name, forcedecks_df, imtp_df, cmj_df, forceframe_df
            )

        # Trunk Profile (DynaMo) below the 3-column layout
        if dynamo_df is not None and not dynamo_df.empty:
            st.markdown("---")
            WeightliftingDiagnosticsModule._display_trunk_profile(dynamo_df)

    @staticmethod
    def _get_latest_row(df: pd.DataFrame) -> pd.Series:
        """Get the most recent test row."""
        if df.empty:
            return pd.Series(dtype='float64')
        return df.iloc[0]  # Already sorted desc by date

    @staticmethod
    def _calc_asymmetry(left, right):
        """Calculate asymmetry percentage. Positive = right dominant."""
        if pd.isna(left) or pd.isna(right) or (left + right) == 0:
            return np.nan
        avg = (left + right) / 2
        return ((right - left) / avg) * 100

    @staticmethod
    def _display_asymmetry_profile(imtp_df, cmj_df, isot_df):
        """Column 1: Performance Asymmetry L/R bars."""
        st.markdown("#### Performance Asymmetry")

        metrics = []

        # IMTP Peak Force L/R
        if not imtp_df.empty:
            latest = WeightliftingDiagnosticsModule._get_latest_row(imtp_df)
            l_val = latest.get('PEAK_VERTICAL_FORCE_Left', np.nan)
            r_val = latest.get('PEAK_VERTICAL_FORCE_Right', np.nan)
            if pd.notna(l_val) and pd.notna(r_val):
                asym = WeightliftingDiagnosticsModule._calc_asymmetry(l_val, r_val)
                metrics.append({'name': 'IMTP Peak Force', 'left': l_val, 'right': r_val, 'asym': asym, 'unit': 'N'})

        # ISOT (Start Position) L/R
        if not isot_df.empty:
            latest = WeightliftingDiagnosticsModule._get_latest_row(isot_df)
            l_val = latest.get('PEAK_VERTICAL_FORCE_Left', np.nan)
            r_val = latest.get('PEAK_VERTICAL_FORCE_Right', np.nan)
            if pd.notna(l_val) and pd.notna(r_val):
                asym = WeightliftingDiagnosticsModule._calc_asymmetry(l_val, r_val)
                metrics.append({'name': 'ISO Start Position', 'left': l_val, 'right': r_val, 'asym': asym, 'unit': 'N'})

        # CMJ Concentric Impulse L/R
        if not cmj_df.empty:
            latest = WeightliftingDiagnosticsModule._get_latest_row(cmj_df)
            l_val = latest.get('CONCENTRIC_IMPULSE_Left', np.nan)
            r_val = latest.get('CONCENTRIC_IMPULSE_Right', np.nan)
            if pd.notna(l_val) and pd.notna(r_val):
                asym = WeightliftingDiagnosticsModule._calc_asymmetry(l_val, r_val)
                metrics.append({'name': 'CMJ Impulse', 'left': l_val, 'right': r_val, 'asym': asym, 'unit': 'Ns'})

            # CMJ Peak Concentric Force L/R
            l_val = latest.get('PEAK_CONCENTRIC_FORCE_Left', np.nan)
            r_val = latest.get('PEAK_CONCENTRIC_FORCE_Right', np.nan)
            if pd.notna(l_val) and pd.notna(r_val):
                asym = WeightliftingDiagnosticsModule._calc_asymmetry(l_val, r_val)
                metrics.append({'name': 'CMJ Peak Force', 'left': l_val, 'right': r_val, 'asym': asym, 'unit': 'N'})

        if not metrics:
            st.info("No asymmetry data available")
            return

        # Build diverging bar chart
        names = [m['name'] for m in metrics]
        left_vals = [-m['left'] for m in metrics]  # Negative for left side
        right_vals = [m['right'] for m in metrics]
        asym_vals = [m['asym'] for m in metrics]

        fig = go.Figure()

        # Left bars (negative direction)
        left_colors = []
        for m in metrics:
            a = abs(m['asym'])
            if a > 10:
                left_colors.append(WeightliftingDiagnosticsModule.RED_FLAG)
            elif a > 5:
                left_colors.append(WeightliftingDiagnosticsModule.GOLD_ACCENT)
            else:
                left_colors.append(WeightliftingDiagnosticsModule.TEAL_PRIMARY)

        right_colors = left_colors.copy()

        fig.add_trace(go.Bar(
            y=names, x=left_vals, orientation='h',
            marker_color=left_colors, name='Left',
            text=[f"{abs(v):.0f}" for v in left_vals],
            textposition='inside', textfont=dict(color='white')
        ))

        fig.add_trace(go.Bar(
            y=names, x=right_vals, orientation='h',
            marker_color=right_colors, name='Right',
            text=[f"{v:.0f}" for v in right_vals],
            textposition='inside', textfont=dict(color='white'),
            opacity=0.75
        ))

        # Add asymmetry annotations
        for i, m in enumerate(metrics):
            a = m['asym']
            color = WeightliftingDiagnosticsModule.RED_FLAG if abs(a) > 10 else '#333'
            fig.add_annotation(
                x=max(right_vals) * 1.1, y=names[i],
                text=f"<b>{abs(a):.1f}%</b>",
                font=dict(color=color, size=11),
                showarrow=False, xanchor='left'
            )

        fig.update_layout(
            barmode='overlay',
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='Inter, sans-serif', color='#333'),
            height=max(200, len(metrics) * 70),
            margin=dict(l=10, r=60, t=10, b=10),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='#333', zerolinewidth=2),
            yaxis=dict(showgrid=False)
        )

        st.plotly_chart(fig, use_container_width=True, key="wl_asymmetry_chart")

        # Asymmetry legend
        st.caption("Color: Green (<5%) | Gold (5-10%) | Red (>10% - flag)")

    @staticmethod
    def _display_body_profile(forceframe_df, ppu_df, cmj_df):
        """Column 2: Hip Profile + Upper Limb Profile."""
        st.markdown("#### Hip Profile")

        hip_metrics = []

        if forceframe_df is not None and not forceframe_df.empty:
            # Hip Flexion from ForceFrame
            hip_flex = forceframe_df[forceframe_df['testTypeName'].str.contains('Flex', case=False, na=False)] if 'testTypeName' in forceframe_df.columns else pd.DataFrame()
            if not hip_flex.empty:
                row = hip_flex.iloc[0]
                l_val = row.get('outerLeftMaxForce', np.nan)
                r_val = row.get('outerRightMaxForce', np.nan)
                if pd.notna(l_val) and pd.notna(r_val):
                    hip_metrics.append({'name': 'Hip Flexion', 'left': l_val, 'right': r_val})

            # Hip Adduction/Abduction if available
            hip_adab = forceframe_df[forceframe_df['testTypeName'].str.contains('Adduct|Abduct', case=False, na=False)] if 'testTypeName' in forceframe_df.columns else pd.DataFrame()
            if not hip_adab.empty:
                row = hip_adab.iloc[0]
                add_l = row.get('innerLeftMaxForce', np.nan)
                add_r = row.get('innerRightMaxForce', np.nan)
                abd_l = row.get('outerLeftMaxForce', np.nan)
                abd_r = row.get('outerRightMaxForce', np.nan)
                if pd.notna(add_l) and pd.notna(add_r):
                    hip_metrics.append({'name': 'Adduction', 'left': add_l, 'right': add_r})
                if pd.notna(abd_l) and pd.notna(abd_r):
                    hip_metrics.append({'name': 'Abduction', 'left': abd_l, 'right': abd_r})

        if hip_metrics:
            fig = go.Figure()
            names = [m['name'] for m in hip_metrics]
            for m in hip_metrics:
                fig.add_trace(go.Bar(
                    y=[m['name']], x=[m['left']], orientation='h',
                    marker_color=WeightliftingDiagnosticsModule.TEAL_PRIMARY,
                    text=[f"L: {m['left']:.0f}N"], textposition='outside',
                    showlegend=(m == hip_metrics[0]), name='Left',
                    legendgroup='left'
                ))
                fig.add_trace(go.Bar(
                    y=[m['name']], x=[m['right']], orientation='h',
                    marker_color=WeightliftingDiagnosticsModule.TEAL_LIGHT,
                    text=[f"R: {m['right']:.0f}N"], textposition='outside',
                    showlegend=(m == hip_metrics[0]), name='Right',
                    legendgroup='right'
                ))

            fig.update_layout(
                barmode='group',
                plot_bgcolor='white', paper_bgcolor='white',
                font=dict(family='Inter, sans-serif', color='#333'),
                height=max(150, len(hip_metrics) * 60),
                margin=dict(l=10, r=80, t=10, b=10),
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                xaxis=dict(showgrid=True, gridcolor='lightgray', title='Force (N)'),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True, key="wl_hip_profile")
        else:
            st.info("No ForceFrame hip data available")

        # Upper Limb Profile
        st.markdown("#### Upper Limb Profile")
        upper_metrics = []

        # Plyo Pushup height
        if not ppu_df.empty:
            latest = WeightliftingDiagnosticsModule._get_latest_row(ppu_df)
            ppu_height = latest.get('PUSHUP_HEIGHT', np.nan)
            if pd.notna(ppu_height):
                upper_metrics.append(('Plyo Pushup Height', f"{ppu_height:.1f} cm"))
            ppu_power = latest.get('BODYMASS_RELATIVE_TAKEOFF_POWER', np.nan)
            if pd.notna(ppu_power):
                upper_metrics.append(('PPU Rel. Power', f"{ppu_power:.2f} W/kg"))

        # CMJ concentric RFD
        if not cmj_df.empty:
            latest = WeightliftingDiagnosticsModule._get_latest_row(cmj_df)
            rfd = latest.get('CONCENTRIC_RFD', np.nan)
            if pd.notna(rfd):
                upper_metrics.append(('CMJ Concentric RFD', f"{rfd:.0f} N/s"))

        if upper_metrics:
            for label, value in upper_metrics:
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 0.5rem 1rem; border-radius: 6px;
                     margin-bottom: 0.5rem; border-left: 3px solid #007167;">
                    <span style="color: #666; font-size: 0.85rem;">{label}</span><br>
                    <span style="color: #333; font-size: 1.1rem; font-weight: bold;">{value}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No upper limb data available")

    @staticmethod
    def _display_athlete_card(athlete_name, forcedecks_df, imtp_df, cmj_df, forceframe_df):
        """Column 3: Athlete Summary Card with auto-generated observations."""
        st.markdown("#### Athlete Summary")

        # Get body mass
        body_mass = None
        if not forcedecks_df.empty and 'bodyMassKg' in forcedecks_df.columns:
            bm = forcedecks_df['bodyMassKg'].dropna()
            if not bm.empty:
                body_mass = bm.iloc[0]

        # Build card HTML
        bm_text = f"{body_mass:.1f} kg" if body_mass else "N/A"
        test_count = len(forcedecks_df)
        date_col = get_date_column(forcedecks_df)
        latest_date = "N/A"
        if date_col and not forcedecks_df.empty:
            dates = pd.to_datetime(forcedecks_df[date_col], errors='coerce').dropna()
            if not dates.empty:
                latest_date = dates.max().strftime('%d %b %Y')

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #007167 0%, #005a51 100%);
             padding: 1.2rem; border-radius: 10px; color: white;">
            <h3 style="margin: 0; color: white;">{athlete_name}</h3>
            <p style="color: #a08e66; margin: 0.3rem 0;">Weightlifting</p>
            <hr style="border-color: rgba(255,255,255,0.2); margin: 0.8rem 0;">
            <p style="margin: 0.3rem 0; font-size: 0.9rem;">Body Mass: <b>{bm_text}</b></p>
            <p style="margin: 0.3rem 0; font-size: 0.9rem;">Total Tests: <b>{test_count}</b></p>
            <p style="margin: 0.3rem 0; font-size: 0.9rem;">Latest Test: <b>{latest_date}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # Key metrics
        st.markdown("##### Key Metrics")
        if not imtp_df.empty:
            latest = WeightliftingDiagnosticsModule._get_latest_row(imtp_df)
            npf = latest.get('NET_PEAK_VERTICAL_FORCE', np.nan)
            rel = latest.get('ISO_BM_REL_FORCE_PEAK', np.nan)
            if pd.notna(npf):
                st.metric("IMTP Net Peak Force", f"{npf:.0f} N")
            if pd.notna(rel):
                st.metric("IMTP Relative Force", f"{rel:.2f} N/kg")

        if not cmj_df.empty:
            latest = WeightliftingDiagnosticsModule._get_latest_row(cmj_df)
            jh = latest.get('JUMP_HEIGHT', np.nan)
            rp = latest.get('BODYMASS_RELATIVE_TAKEOFF_POWER', np.nan)
            if pd.notna(jh):
                # VALD stores jump height in cm (typically 20-60cm range)
                st.metric("CMJ Height", f"{jh:.1f} cm")
            if pd.notna(rp):
                st.metric("CMJ Rel. Power", f"{rp:.2f} W/kg")

        # Auto-generated observations
        st.markdown("##### Observations")
        observations = WeightliftingDiagnosticsModule._generate_observations(imtp_df, cmj_df, forceframe_df)
        if observations:
            for obs in observations:
                st.markdown(f"- {obs}")
        else:
            st.caption("Insufficient data for automated observations")

    @staticmethod
    def _generate_observations(imtp_df, cmj_df, forceframe_df):
        """Auto-generate observations based on asymmetry and trends."""
        observations = []

        # Check IMTP asymmetry
        if not imtp_df.empty:
            latest = imtp_df.iloc[0]
            l = latest.get('PEAK_VERTICAL_FORCE_Left', np.nan)
            r = latest.get('PEAK_VERTICAL_FORCE_Right', np.nan)
            if pd.notna(l) and pd.notna(r) and (l + r) > 0:
                asym = abs(r - l) / ((l + r) / 2) * 100
                if asym > 10:
                    dom = "Right" if r > l else "Left"
                    observations.append(f"IMTP asymmetry {asym:.1f}% ({dom} dominant) - monitor")
                elif asym < 5:
                    observations.append(f"IMTP symmetry good ({asym:.1f}%)")

        # Check CMJ impulse asymmetry
        if not cmj_df.empty:
            latest = cmj_df.iloc[0]
            l = latest.get('CONCENTRIC_IMPULSE_Left', np.nan)
            r = latest.get('CONCENTRIC_IMPULSE_Right', np.nan)
            if pd.notna(l) and pd.notna(r) and (l + r) > 0:
                asym = abs(r - l) / ((l + r) / 2) * 100
                if asym > 10:
                    dom = "Right" if r > l else "Left"
                    observations.append(f"CMJ impulse asymmetry {asym:.1f}% ({dom} dominant)")

        # Check hip flexion asymmetry
        if forceframe_df is not None and not forceframe_df.empty:
            hip_flex = forceframe_df[forceframe_df['testTypeName'].str.contains('Flex', case=False, na=False)] if 'testTypeName' in forceframe_df.columns else pd.DataFrame()
            if not hip_flex.empty:
                row = hip_flex.iloc[0]
                l = row.get('outerLeftMaxForce', np.nan)
                r = row.get('outerRightMaxForce', np.nan)
                if pd.notna(l) and pd.notna(r) and (l + r) > 0:
                    asym = abs(r - l) / ((l + r) / 2) * 100
                    if asym > 10:
                        dom = "Right" if r > l else "Left"
                        observations.append(f"Hip flexion asymmetry {asym:.1f}% ({dom} dominant)")

        # Check IMTP trend (if 3+ tests)
        if len(imtp_df) >= 3:
            forces = imtp_df['NET_PEAK_VERTICAL_FORCE'].dropna() if 'NET_PEAK_VERTICAL_FORCE' in imtp_df.columns else pd.Series(dtype='float64')
            if len(forces) >= 3:
                recent = forces.iloc[0]
                oldest = forces.iloc[-1]
                change_pct = ((recent - oldest) / oldest) * 100 if oldest > 0 else 0
                if change_pct > 5:
                    observations.append(f"IMTP trending up (+{change_pct:.0f}% over {len(forces)} tests)")
                elif change_pct < -5:
                    observations.append(f"IMTP trending down ({change_pct:.0f}% over {len(forces)} tests)")

        return observations

    @staticmethod
    def _display_trunk_profile(dynamo_df):
        """Trunk Profile section from DynaMo data."""
        st.markdown("#### Trunk Strength Profile (DynaMo)")

        if dynamo_df.empty or 'movement' not in dynamo_df.columns:
            st.info("No DynaMo trunk data available")
            return

        # Get latest test per movement (DynaMo uses camelCase: 'Flexion', 'Extension', 'LateralFlexionLeft', etc.)
        movements = {}
        movement_patterns = {
            'Flexion': r'^Flexion$',
            'Extension': r'^Extension$',
            'Lateral Flexion Left': r'LateralFlexionLeft|Lateral.*Flexion.*Left',
            'Lateral Flexion Right': r'LateralFlexionRight|Lateral.*Flexion.*Right',
        }
        for label, pattern in movement_patterns.items():
            m_df = dynamo_df[dynamo_df['movement'].str.contains(pattern, case=False, na=False, regex=True)]
            if not m_df.empty and 'maxForceNewtons' in m_df.columns:
                movements[label] = m_df['maxForceNewtons'].iloc[0]

        if not movements:
            st.info("No trunk strength data found")
            return

        col1, col2 = st.columns(2)

        with col1:
            # Flexion vs Extension
            flex = movements.get('Flexion', 0)
            ext = movements.get('Extension', 0)
            if flex > 0 or ext > 0:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Flexion', 'Extension'], y=[flex, ext],
                    marker_color=[WeightliftingDiagnosticsModule.TEAL_PRIMARY,
                                  WeightliftingDiagnosticsModule.TEAL_LIGHT],
                    text=[f"{flex:.0f}N", f"{ext:.0f}N"],
                    textposition='outside'
                ))
                ratio = flex / ext if ext > 0 else 0
                fig.update_layout(
                    title=f"Trunk Flex/Ext (Ratio: {ratio:.2f})",
                    plot_bgcolor='white', paper_bgcolor='white',
                    font=dict(family='Inter, sans-serif', color='#333'),
                    height=250, margin=dict(l=10, r=10, t=40, b=10),
                    yaxis=dict(title='Force (N)', showgrid=True, gridcolor='lightgray'),
                    xaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig, use_container_width=True, key="wl_trunk_flex_ext")

        with col2:
            # Lateral Flexion L vs R
            lat_l = movements.get('Lateral Flexion Left', 0)
            lat_r = movements.get('Lateral Flexion Right', 0)
            if lat_l > 0 or lat_r > 0:
                asym = abs(lat_r - lat_l) / ((lat_l + lat_r) / 2) * 100 if (lat_l + lat_r) > 0 else 0
                asym_color = WeightliftingDiagnosticsModule.RED_FLAG if asym > 10 else '#333'
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Left', 'Right'], y=[lat_l, lat_r],
                    marker_color=[WeightliftingDiagnosticsModule.TEAL_PRIMARY,
                                  WeightliftingDiagnosticsModule.TEAL_LIGHT],
                    text=[f"{lat_l:.0f}N", f"{lat_r:.0f}N"],
                    textposition='outside'
                ))
                fig.update_layout(
                    title=f"Lateral Flexion (Asym: {asym:.1f}%)",
                    plot_bgcolor='white', paper_bgcolor='white',
                    font=dict(family='Inter, sans-serif', color='#333'),
                    height=250, margin=dict(l=10, r=10, t=40, b=10),
                    yaxis=dict(title='Force (N)', showgrid=True, gridcolor='lightgray'),
                    xaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig, use_container_width=True, key="wl_trunk_lateral")


def display_test_type_module(test_type: str, athlete_df: pd.DataFrame, athlete_name: str, sport: str = None):
    """
    Route to appropriate test-type specific module.

    Args:
        test_type: Type of analysis ('CMJ', 'ISO_SL', 'ISO_DL', 'THROWS')
        athlete_df: Athlete's test data
        athlete_name: Athlete's name
        sport: Sport (optional, for throws)
    """
    if test_type == 'CMJ':
        CMJAnalysisModule.display_cmj_dashboard(athlete_df, athlete_name)
        st.markdown("---")
        CMJAnalysisModule.display_cmj_power_focus(athlete_df, athlete_name)

    elif test_type == 'ISO_SL':
        IsometricSingleLegModule.display_single_leg_analysis(athlete_df, athlete_name)

    elif test_type == 'ISO_DL':
        IsometricDoubleLegModule.display_double_leg_analysis(athlete_df, athlete_name)

    elif test_type == 'THROWS':
        if sport:
            ThrowsTrainingModule.display_throws_dashboard(athlete_df, athlete_name, sport)
        else:
            st.warning("Sport information required for Throws dashboard")

    elif test_type == 'WEIGHTLIFTING':
        WeightliftingDiagnosticsModule.display_dashboard(athlete_df, athlete_name)

    else:
        st.error(f"Unknown test type module: {test_type}")
