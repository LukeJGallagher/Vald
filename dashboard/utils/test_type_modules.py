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

        # Find relative power columns - use exact match first for reliability
        left_col = None
        right_col = None
        trial_col = None

        # Try exact match first
        if 'Peak Power / BM_Trial' in cmj_df.columns:
            trial_col = 'Peak Power / BM_Trial'

        # Fall back to search if exact match not found
        if not trial_col:
            for col in cmj_df.columns:
                col_lower = col.lower()
                # Check for relative power (includes Peak Power / BM which is relative power)
                if 'power' in col_lower and ('/ bm' in col_lower or '/bm' in col_lower):
                    if 'left' in col_lower or '_l_' in col_lower:
                        left_col = col
                    elif 'right' in col_lower or '_r_' in col_lower:
                        right_col = col
                    elif 'trial' in col_lower and 'peak' in col_lower:
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
        st.markdown("#### Jump Height (Imp-Dis) [cm]")

        # Find jump height columns - use exact match first
        left_col = None
        right_col = None
        trial_col = None

        # Try exact match first
        if 'Jump Height (Imp-Dis)_Trial' in cmj_df.columns:
            trial_col = 'Jump Height (Imp-Dis)_Trial'

        # Fall back to search if exact match not found
        if not trial_col:
            for col in cmj_df.columns:
                col_lower = col.lower()
                if 'jump' in col_lower and 'height' in col_lower:
                    if 'left' in col_lower or '_l_' in col_lower:
                        left_col = col
                    elif 'right' in col_lower or '_r_' in col_lower:
                        right_col = col
                    elif 'trial' in col_lower and 'imp' in col_lower:
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

        # Find CMJ depth columns - use exact match first for reliability
        left_col = None
        right_col = None
        trial_col = None

        # Try exact match first
        if 'Countermovement Depth_Trial' in cmj_df.columns:
            trial_col = 'Countermovement Depth_Trial'

        # Fall back to search if exact match not found
        if not trial_col:
            for col in cmj_df.columns:
                col_lower = col.lower()
                if ('countermovement' in col_lower or 'cmj') and 'depth' in col_lower:
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

            fig.add_trace(go.Scatter(
                x=cmj_df[date_col],
                y=cmj_df[trial_col],
                mode='lines+markers',
                line=dict(color='#90BE6D', width=3),
                marker=dict(size=10, color='#90BE6D'),
                name='Countermovement Depth',
                text=cmj_df[trial_col].round(1),
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

        # Find relevant columns - use exact matches first for reliability
        rel_power_col = None
        body_mass_col = None
        height_col = None

        # Try exact matches first
        if 'Peak Power / BM_Trial' in cmj_df.columns:
            rel_power_col = 'Peak Power / BM_Trial'
        if 'Jump Height (Imp-Dis)_Trial' in cmj_df.columns:
            height_col = 'Jump Height (Imp-Dis)_Trial'

        # Fall back to search if exact matches not found
        if not rel_power_col or not height_col or not body_mass_col:
            for col in cmj_df.columns:
                col_lower = col.lower()
                if not rel_power_col and ('/ bm' in col_lower or '/bm' in col_lower) and 'power' in col_lower:
                    rel_power_col = col
                elif not body_mass_col and ('body' in col_lower or 'mass' in col_lower) and ('weight' in col_lower or 'bm' in col_lower):
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
                st.markdown(f"## {latest[height_col]:.2f} cm")
            else:
                st.markdown("## N/A")

        st.markdown("---")

        # Combined chart: Height bars + Power line
        if height_col and rel_power_col:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Height bars (primary y-axis)
            fig.add_trace(
                go.Bar(
                    x=cmj_df[date_col],
                    y=cmj_df[height_col],
                    name='CMJ Height',
                    marker_color='#0000FF',
                    text=cmj_df[height_col].round(2),
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
            'Training': ('#007167', '#005a51'),      # Teal gradient
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
                        color1, color2 = session_colors.get(session_type, ('#007167', '#005a51'))
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
                            'Training': '#007167',
                            'Competition': '#FFB800',
                            'Testing': '#0077B6',
                            'Warm-up': '#6c757d'
                        }

                        # Add points colored by session type
                        if 'session_type' in event_data.columns:
                            for session_type in event_data['session_type'].unique():
                                session_data = event_data[event_data['session_type'] == session_type]
                                color = chart_colors.get(session_type, '#007167')

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
                                marker=dict(size=8, color='#007167'),
                                line=dict(color='#007167', width=2),
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

            # 3-test trend bars
            recent_3 = imtp_df.tail(3)

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=[d.strftime('%d-%b') for d in recent_3[date_col]],
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

        # Find power columns - use exact matches first
        abs_power_col = None
        rel_power_col = None

        # Try exact matches first
        if 'Peak Power_Trial' in cmj_df.columns:
            abs_power_col = 'Peak Power_Trial'
        if 'Peak Power / BM_Trial' in cmj_df.columns:
            rel_power_col = 'Peak Power / BM_Trial'

        # Fall back to search
        if not abs_power_col or not rel_power_col:
            for col in cmj_df.columns:
                col_lower = col.lower()
                if 'peak' in col_lower and 'power' in col_lower:
                    if ('/ bm' in col_lower or '/bm' in col_lower or 'relative' in col_lower) and not rel_power_col:
                        rel_power_col = col
                    elif not abs_power_col and '/ bm' not in col_lower and '/bm' not in col_lower and 'relative' not in col_lower:
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

        # Find depth column - use exact match first
        depth_col = None

        if 'Countermovement Depth_Trial' in cmj_df.columns:
            depth_col = 'Countermovement Depth_Trial'
        else:
            for col in cmj_df.columns:
                col_lower = col.lower()
                if ('countermovement' in col_lower or 'cmj' in col_lower) and 'depth' in col_lower:
                    depth_col = col
                    break

        if depth_col:
            latest_depth = cmj_df_sorted[depth_col].iloc[-1]
            avg_depth = cmj_df_sorted[depth_col].mean()

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
                y=cmj_df_sorted[depth_col],
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

    else:
        st.error(f"Unknown test type module: {test_type}")
