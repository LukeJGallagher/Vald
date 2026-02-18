"""
Generic Sport Diagnostics Module
Team Saudi VALD Performance Dashboard

Renders sport-specific physical diagnostics for athletes, driven by config
dicts from dashboard/config/sport_diagnostic_configs.py.

Architecture mirrors WeightliftingDiagnosticsModule (test_type_modules.py)
but is GENERIC - driven by config dicts instead of hardcoded metrics.

3-column layout:
  Column 1: Performance Asymmetry (L/R diverging bars from ForceDecks)
  Column 2: Body Profile (ForceFrame / DynaMo / NordBord)
  Column 3: Athlete Summary Card with auto-generated observations
+ Extra sections below (hop tests, balance, plyo pushup, strength RM)

Usage:
    from dashboard.utils.sport_diagnostics import SportDiagnosticsModule

    SportDiagnosticsModule.display_dashboard(
        sport_key='fencing',
        forcedecks_df=athlete_fd,
        athlete_name='Ahmed Al-Mutairi',
        forceframe_df=athlete_ff,
        dynamo_df=athlete_dyn,
        nordbord_df=athlete_nb,
    )
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config import (works both as package and direct run)
# ---------------------------------------------------------------------------
try:
    from dashboard.config.sport_diagnostic_configs import SPORT_DIAGNOSTIC_CONFIGS
except ImportError:
    try:
        from config.sport_diagnostic_configs import SPORT_DIAGNOSTIC_CONFIGS
    except ImportError:
        SPORT_DIAGNOSTIC_CONFIGS = {}

# ---------------------------------------------------------------------------
# Report export import (optional - only needed for export buttons)
# ---------------------------------------------------------------------------
try:
    from dashboard.utils.report_export import (
        generate_individual_pdf_report,
        generate_individual_html_report,
    )
except ImportError:
    try:
        from utils.report_export import (
            generate_individual_pdf_report,
            generate_individual_html_report,
        )
    except ImportError:
        generate_individual_pdf_report = None
        generate_individual_html_report = None


# ============================================================================
# HELPER: get_date_column
# ============================================================================

def get_date_column(df: pd.DataFrame) -> Optional[str]:
    """Return the best date column name present in *df*, or None."""
    if df is None or df.empty:
        return None
    for col in ['recordedDateUtc', 'testDateUtc', 'testDateTime', 'date']:
        if col in df.columns:
            return col
    # Fallback: any column containing 'date'
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    return date_cols[0] if date_cols else None


# ============================================================================
# SPORT DIAGNOSTICS MODULE
# ============================================================================

class SportDiagnosticsModule:
    """Generic sport diagnostics module driven by config dicts.

    Every public method is ``@staticmethod`` so callers do not need to
    instantiate the class - just call ``SportDiagnosticsModule.display_dashboard(...)``.
    """

    # Team Saudi brand colors (v2 - from banner/logo)
    TEAL_PRIMARY = '#235032'
    TEAL_LIGHT = '#3a7050'
    TEAL_DARK = '#1a3d25'
    GOLD_ACCENT = '#a08e66'
    BLUE = '#0077B6'
    RED_FLAG = '#dc3545'

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @staticmethod
    def display_dashboard(
        sport_key: str,
        forcedecks_df: pd.DataFrame,
        athlete_name: str,
        forceframe_df: Optional[pd.DataFrame] = None,
        dynamo_df: Optional[pd.DataFrame] = None,
        nordbord_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Render the full 3-column diagnostic layout + extras.

        Args:
            sport_key: Lowercase config key (e.g. ``'fencing'``, ``'karate'``).
            forcedecks_df: ForceDecks test data for the athlete (already filtered).
            athlete_name: Display name.
            forceframe_df: ForceFrame data (optional).
            dynamo_df: DynaMo data (optional).
            nordbord_df: NordBord data (optional).
        """
        config = SportDiagnosticsModule._get_config(sport_key)
        if config is None:
            st.warning(f"No diagnostic config found for sport key '{sport_key}'.")
            return

        display_name = config.get('display_name', sport_key.title())
        icon = config.get('icon', '')

        # Branded header
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #235032 0%, #1a3d25 100%);
             padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;
             border-left: 4px solid #a08e66;">
            <h2 style="color: white; margin: 0;">{icon} Physical Diagnostics Summary</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
                {athlete_name} &mdash; {display_name}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Ensure forcedecks_df is a usable DataFrame
        if forcedecks_df is None or forcedecks_df.empty:
            st.info("No ForceDecks data available for this athlete.")
            SportDiagnosticsModule.export_buttons(sport_key, athlete_name, pd.DataFrame())
            return

        # Parse and sort by date
        date_col = get_date_column(forcedecks_df)
        if date_col:
            forcedecks_df = forcedecks_df.copy()
            forcedecks_df[date_col] = pd.to_datetime(forcedecks_df[date_col], errors='coerce')
            forcedecks_df = forcedecks_df.sort_values(date_col, ascending=False)

        # 3-column layout
        col1, col2, col3 = st.columns([1.2, 1.2, 1])

        with col1:
            SportDiagnosticsModule._display_asymmetry_profile(forcedecks_df, config, sport_key)

        with col2:
            SportDiagnosticsModule._display_body_profile(
                forceframe_df, dynamo_df, nordbord_df, forcedecks_df, config, sport_key,
            )

        with col3:
            SportDiagnosticsModule._display_athlete_card(
                athlete_name, forcedecks_df, forceframe_df, config, sport_key,
            )

        # Extra sections below the 3-column layout
        extra_sections = config.get('extra_sections', [])
        if extra_sections:
            st.markdown("---")
            SportDiagnosticsModule._display_extra_sections(
                forcedecks_df, config, sport_key, forceframe_df=forceframe_df,
            )

        # Benchmarks section (if config has benchmarks)
        if config.get('benchmarks'):
            st.markdown("---")
            SportDiagnosticsModule._display_benchmarks_section(config, forcedecks_df, athlete_name)

        # Performance ratios (if config has performance_ratios)
        if config.get('performance_ratios'):
            st.markdown("---")
            SportDiagnosticsModule._display_performance_ratios(
                config, forcedecks_df, athlete_name, forceframe_df=forceframe_df,
            )

        # Injury risk panel (if config has injury_risk_indicators)
        if config.get('injury_risk_indicators'):
            st.markdown("---")
            SportDiagnosticsModule._display_injury_risk_panel(
                config, forcedecks_df, athlete_name,
                forceframe_df=forceframe_df, nordbord_df=nordbord_df,
            )

        # Export buttons
        st.markdown("---")
        SportDiagnosticsModule.export_buttons(
            sport_key, athlete_name, forcedecks_df, forceframe_df, dynamo_df,
        )

    # ------------------------------------------------------------------
    # Config lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _get_config(sport_key: str) -> Optional[Dict[str, Any]]:
        """Look up config from SPORT_DIAGNOSTIC_CONFIGS."""
        return SPORT_DIAGNOSTIC_CONFIGS.get(sport_key)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_latest_row(df: pd.DataFrame) -> pd.Series:
        """Get the most recent test row (df assumed sorted desc by date)."""
        if df is None or df.empty:
            return pd.Series(dtype='float64')
        return df.iloc[0]

    @staticmethod
    def _calc_asymmetry(left: float, right: float) -> float:
        """Calculate asymmetry %. Positive = right dominant."""
        if pd.isna(left) or pd.isna(right):
            return np.nan
        avg = (left + right) / 2
        if avg == 0:
            return np.nan
        return ((right - left) / avg) * 100

    @staticmethod
    def _filter_by_test_types(df: pd.DataFrame, test_types: List[str]) -> pd.DataFrame:
        """Filter a DataFrame to rows whose ``testType`` is in *test_types*."""
        if df is None or df.empty or not test_types:
            return pd.DataFrame()
        if 'testType' not in df.columns:
            return pd.DataFrame()
        return df[df['testType'].isin(test_types)]

    # ------------------------------------------------------------------
    # Column 1: Asymmetry profile
    # ------------------------------------------------------------------

    @staticmethod
    def _display_asymmetry_profile(
        forcedecks_df: pd.DataFrame,
        config: Dict[str, Any],
        sport_key: str,
    ) -> None:
        """Column 1: Performance Asymmetry diverging L/R bars."""
        st.markdown("#### Performance Asymmetry")

        asym_metrics_cfg = config.get('asymmetry_metrics', [])
        thresholds = config.get('asymmetry_thresholds', {'caution': 10.0, 'risk': 15.0})
        caution_pct = thresholds.get('caution')
        risk_pct = thresholds.get('risk')

        metrics: List[Dict[str, Any]] = []

        for m_cfg in asym_metrics_cfg:
            test_types = m_cfg.get('test_types', [])
            left_col = m_cfg.get('left_col', '')
            right_col = m_cfg.get('right_col', '')
            name = m_cfg.get('name', 'Unknown')
            unit = m_cfg.get('unit', '')

            subset = SportDiagnosticsModule._filter_by_test_types(forcedecks_df, test_types)
            if subset.empty:
                continue

            latest = SportDiagnosticsModule._get_latest_row(subset)

            l_val = pd.to_numeric(latest.get(left_col, np.nan), errors='coerce')
            r_val = pd.to_numeric(latest.get(right_col, np.nan), errors='coerce')
            if pd.isna(l_val) or pd.isna(r_val):
                continue

            asym = SportDiagnosticsModule._calc_asymmetry(l_val, r_val)
            metrics.append({
                'name': name,
                'left': float(l_val),
                'right': float(r_val),
                'asym': asym,
                'unit': unit,
            })

        if not metrics:
            st.info("No asymmetry data available")
            return

        # Build diverging bar chart
        names = [m['name'] for m in metrics]
        left_vals = [-m['left'] for m in metrics]
        right_vals = [m['right'] for m in metrics]

        # Color bars by threshold
        def _bar_color(abs_asym: float) -> str:
            if risk_pct is not None and abs_asym > risk_pct:
                return SportDiagnosticsModule.RED_FLAG
            if caution_pct is not None and abs_asym > caution_pct:
                return SportDiagnosticsModule.GOLD_ACCENT
            return SportDiagnosticsModule.TEAL_PRIMARY

        bar_colors = [_bar_color(abs(m['asym'])) for m in metrics]

        fig = go.Figure()

        # Left bars (negative direction)
        fig.add_trace(go.Bar(
            y=names, x=left_vals, orientation='h',
            marker_color=bar_colors, name='Left',
            text=[f"{abs(v):.0f}" for v in left_vals],
            textposition='inside', textfont=dict(color='white'),
        ))

        # Right bars (positive direction)
        fig.add_trace(go.Bar(
            y=names, x=right_vals, orientation='h',
            marker_color=bar_colors, name='Right',
            text=[f"{v:.0f}" for v in right_vals],
            textposition='inside', textfont=dict(color='white'),
            opacity=0.75,
        ))

        # Asymmetry annotations on the right
        max_right = max(right_vals) if right_vals else 100
        for i, m in enumerate(metrics):
            a = m['asym']
            ann_color = SportDiagnosticsModule.RED_FLAG if (risk_pct is not None and abs(a) > risk_pct) else '#333'
            fig.add_annotation(
                x=max_right * 1.1, y=names[i],
                text=f"<b>{abs(a):.1f}%</b>",
                font=dict(color=ann_color, size=11),
                showarrow=False, xanchor='left',
            )

        fig.update_layout(
            barmode='overlay',
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='Inter, sans-serif', color='#333'),
            height=max(200, len(metrics) * 70),
            margin=dict(l=10, r=60, t=10, b=10),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            xaxis=dict(
                showgrid=True, gridcolor='lightgray',
                zeroline=True, zerolinecolor='#333', zerolinewidth=2,
            ),
            yaxis=dict(showgrid=False),
        )

        st.plotly_chart(fig, width='stretch', key=f"sd_{sport_key}_asymmetry")

        # Threshold legend
        if caution_pct is not None and risk_pct is not None:
            st.caption(
                f"Color: Green (<{caution_pct:.0f}%) | Gold ({caution_pct:.0f}-{risk_pct:.0f}%) | Red (>{risk_pct:.0f}% - flag)"
            )
        elif caution_pct is None and risk_pct is None:
            st.caption("Asymmetry thresholds not applicable - track individual trends instead.")

        # Sport-specific asymmetry note
        note = config.get('asymmetry_note')
        if note:
            st.caption(note)

    # ------------------------------------------------------------------
    # Column 2: Body profile (ForceFrame / DynaMo / NordBord)
    # ------------------------------------------------------------------

    @staticmethod
    def _display_body_profile(
        forceframe_df: Optional[pd.DataFrame],
        dynamo_df: Optional[pd.DataFrame],
        nordbord_df: Optional[pd.DataFrame],
        forcedecks_df: pd.DataFrame,
        config: Dict[str, Any],
        sport_key: str,
    ) -> None:
        """Column 2: ForceFrame / DynaMo / NordBord data."""

        has_any_data = False

        # ----- ForceFrame sections -----
        ff_sections = config.get('forceframe_sections', [])
        if ff_sections and forceframe_df is not None and not forceframe_df.empty:
            for idx, section in enumerate(ff_sections):
                section_name = section.get('name', f'Section {idx}')
                test_pattern = section.get('test_pattern', '')
                inner_label = section.get('inner_label', 'Inner')
                outer_label = section.get('outer_label', 'Outer')

                st.markdown(f"#### {section_name}")

                if 'testTypeName' not in forceframe_df.columns:
                    st.info(f"No ForceFrame testTypeName column for {section_name}")
                    continue

                matched = forceframe_df[
                    forceframe_df['testTypeName'].str.contains(test_pattern, case=False, na=False, regex=True)
                ]
                if matched.empty:
                    st.info(f"No {section_name} data available")
                    continue

                has_any_data = True

                # Sort by date and take latest
                ff_date_col = get_date_column(matched)
                if ff_date_col:
                    matched = matched.copy()
                    matched[ff_date_col] = pd.to_datetime(matched[ff_date_col], errors='coerce')
                    matched = matched.sort_values(ff_date_col, ascending=False)
                row = matched.iloc[0]

                # Extract inner/outer L/R
                inner_l = pd.to_numeric(row.get('innerLeftMaxForce', np.nan), errors='coerce')
                inner_r = pd.to_numeric(row.get('innerRightMaxForce', np.nan), errors='coerce')
                outer_l = pd.to_numeric(row.get('outerLeftMaxForce', np.nan), errors='coerce')
                outer_r = pd.to_numeric(row.get('outerRightMaxForce', np.nan), errors='coerce')

                bar_data = []
                if pd.notna(inner_l) and pd.notna(inner_r):
                    bar_data.append({
                        'name': inner_label, 'left': float(inner_l), 'right': float(inner_r),
                    })
                if pd.notna(outer_l) and pd.notna(outer_r):
                    bar_data.append({
                        'name': outer_label, 'left': float(outer_l), 'right': float(outer_r),
                    })

                if bar_data:
                    fig = go.Figure()
                    for bd in bar_data:
                        fig.add_trace(go.Bar(
                            y=[bd['name']], x=[bd['left']], orientation='h',
                            marker_color=SportDiagnosticsModule.TEAL_PRIMARY,
                            text=[f"L: {bd['left']:.0f}N"], textposition='outside',
                            showlegend=(bd == bar_data[0]), name='Left',
                            legendgroup='left',
                        ))
                        fig.add_trace(go.Bar(
                            y=[bd['name']], x=[bd['right']], orientation='h',
                            marker_color=SportDiagnosticsModule.TEAL_LIGHT,
                            text=[f"R: {bd['right']:.0f}N"], textposition='outside',
                            showlegend=(bd == bar_data[0]), name='Right',
                            legendgroup='right',
                        ))

                    # Show ratio if both inner and outer available
                    if len(bar_data) == 2:
                        inner_avg = (bar_data[0]['left'] + bar_data[0]['right']) / 2
                        outer_avg = (bar_data[1]['left'] + bar_data[1]['right']) / 2
                        if outer_avg > 0:
                            ratio = inner_avg / outer_avg
                            fig.add_annotation(
                                x=0, y=-0.3, xref='paper', yref='paper',
                                text=f"{inner_label}/{outer_label} Ratio: <b>{ratio:.2f}</b>",
                                font=dict(size=11, color='#333'),
                                showarrow=False, xanchor='left',
                            )

                    fig.update_layout(
                        barmode='group',
                        plot_bgcolor='white', paper_bgcolor='white',
                        font=dict(family='Inter, sans-serif', color='#333'),
                        height=max(150, len(bar_data) * 65 + 30),
                        margin=dict(l=10, r=80, t=10, b=30),
                        showlegend=True,
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                        xaxis=dict(showgrid=True, gridcolor='lightgray', title='Force (N)'),
                        yaxis=dict(showgrid=False),
                    )
                    st.plotly_chart(fig, width='stretch', key=f"sd_{sport_key}_ff_{idx}")
                else:
                    st.info(f"No usable force values for {section_name}")

        # ----- DynaMo section -----
        dynamo_cfg = config.get('dynamo_section')
        if dynamo_cfg and dynamo_cfg.get('enabled') and dynamo_df is not None and not dynamo_df.empty:
            title = dynamo_cfg.get('title', 'Grip Strength (DynaMo)')
            movement_filter = dynamo_cfg.get('movement_filter', 'GripSqueeze')
            st.markdown(f"#### {title}")

            if 'movement' in dynamo_df.columns and 'maxForceNewtons' in dynamo_df.columns:
                grip = dynamo_df[dynamo_df['movement'].str.contains(movement_filter, case=False, na=False)]
                if not grip.empty:
                    has_any_data = True

                    # Sort by date and take latest per hand
                    grip = grip.copy()
                    grip['maxForceNewtons'] = pd.to_numeric(grip['maxForceNewtons'], errors='coerce')
                    dyn_date_col = get_date_column(grip)
                    if dyn_date_col:
                        grip[dyn_date_col] = pd.to_datetime(grip[dyn_date_col], errors='coerce')
                        grip = grip.sort_values(dyn_date_col, ascending=False)

                    # Try to extract L/R from 'hand' or 'limb' column
                    left_force = np.nan
                    right_force = np.nan
                    for hand_col in ['hand', 'limb', 'side']:
                        if hand_col in grip.columns:
                            left_rows = grip[grip[hand_col].str.contains('Left|L', case=False, na=False)]
                            right_rows = grip[grip[hand_col].str.contains('Right|R', case=False, na=False)]
                            if not left_rows.empty:
                                left_force = left_rows['maxForceNewtons'].iloc[0]
                            if not right_rows.empty:
                                right_force = right_rows['maxForceNewtons'].iloc[0]
                            break

                    # Fallback: if no hand column, just show top force
                    if pd.isna(left_force) and pd.isna(right_force):
                        top_force = grip['maxForceNewtons'].dropna()
                        if not top_force.empty:
                            st.markdown(f"""
                            <div style="background: #f8f9fa; padding: 0.5rem 1rem; border-radius: 6px;
                                 margin-bottom: 0.5rem; border-left: 3px solid #235032;">
                                <span style="color: #666; font-size: 0.85rem;">Max Grip Force</span><br>
                                <span style="color: #333; font-size: 1.1rem; font-weight: bold;">{top_force.iloc[0]:.0f} N</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Show L/R bars
                        grip_data = []
                        if pd.notna(left_force):
                            grip_data.append(('Left', float(left_force)))
                        if pd.notna(right_force):
                            grip_data.append(('Right', float(right_force)))

                        if grip_data:
                            fig = go.Figure()
                            labels = [g[0] for g in grip_data]
                            values = [g[1] for g in grip_data]
                            colors = [SportDiagnosticsModule.TEAL_PRIMARY, SportDiagnosticsModule.TEAL_LIGHT][:len(grip_data)]
                            fig.add_trace(go.Bar(
                                x=labels, y=values,
                                marker_color=colors,
                                text=[f"{v:.0f}N" for v in values],
                                textposition='outside',
                            ))

                            # Check asymmetry
                            if len(grip_data) == 2:
                                asym = SportDiagnosticsModule._calc_asymmetry(grip_data[0][1], grip_data[1][1])
                                asym_color = SportDiagnosticsModule.RED_FLAG if (not pd.isna(asym) and abs(asym) > 10) else '#333'
                                fig.update_layout(title=f"Grip Force (Asym: {abs(asym):.1f}%)")

                            fig.update_layout(
                                plot_bgcolor='white', paper_bgcolor='white',
                                font=dict(family='Inter, sans-serif', color='#333'),
                                height=220, margin=dict(l=10, r=10, t=40, b=10),
                                yaxis=dict(title='Force (N)', showgrid=True, gridcolor='lightgray'),
                                xaxis=dict(showgrid=False),
                            )
                            st.plotly_chart(fig, width='stretch', key=f"sd_{sport_key}_dynamo")
                else:
                    st.info("No grip data available")
            else:
                st.info("DynaMo data missing required columns")

        # ----- NordBord section -----
        nb_cfg = config.get('nordbord_section')
        if nb_cfg and nb_cfg.get('enabled') and nordbord_df is not None and not nordbord_df.empty:
            title = nb_cfg.get('title', 'Nordic Hamstring Strength')
            st.markdown(f"#### {title}")

            has_nb_data = False
            left_force = np.nan
            right_force = np.nan

            for l_col, r_col in [('leftMaxForce', 'rightMaxForce'), ('Left', 'Right')]:
                if l_col in nordbord_df.columns and r_col in nordbord_df.columns:
                    nb_sorted = nordbord_df.copy()
                    nb_date_col = get_date_column(nb_sorted)
                    if nb_date_col:
                        nb_sorted[nb_date_col] = pd.to_datetime(nb_sorted[nb_date_col], errors='coerce')
                        nb_sorted = nb_sorted.sort_values(nb_date_col, ascending=False)
                    latest = nb_sorted.iloc[0]
                    left_force = pd.to_numeric(latest.get(l_col, np.nan), errors='coerce')
                    right_force = pd.to_numeric(latest.get(r_col, np.nan), errors='coerce')
                    if pd.notna(left_force) or pd.notna(right_force):
                        has_nb_data = True
                        break

            if has_nb_data:
                has_any_data = True
                fig = go.Figure()
                labels = []
                values = []
                colors = []

                if pd.notna(left_force):
                    labels.append('Left')
                    values.append(float(left_force))
                    colors.append(SportDiagnosticsModule.TEAL_PRIMARY)
                if pd.notna(right_force):
                    labels.append('Right')
                    values.append(float(right_force))
                    colors.append(SportDiagnosticsModule.TEAL_LIGHT)

                fig.add_trace(go.Bar(
                    x=labels, y=values,
                    marker_color=colors,
                    text=[f"{v:.0f}N" for v in values],
                    textposition='outside',
                ))

                title_text = 'Nordic Hamstring L/R'
                if pd.notna(left_force) and pd.notna(right_force):
                    asym = SportDiagnosticsModule._calc_asymmetry(float(left_force), float(right_force))
                    if not pd.isna(asym):
                        asym_color = SportDiagnosticsModule.RED_FLAG if abs(asym) > 10 else '#333'
                        title_text = f"Nordic Hamstring (Asym: {abs(asym):.1f}%)"

                fig.update_layout(
                    title=title_text,
                    plot_bgcolor='white', paper_bgcolor='white',
                    font=dict(family='Inter, sans-serif', color='#333'),
                    height=220, margin=dict(l=10, r=10, t=40, b=10),
                    yaxis=dict(title='Force (N)', showgrid=True, gridcolor='lightgray'),
                    xaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig, width='stretch', key=f"sd_{sport_key}_nordbord")
            else:
                st.info("No NordBord data available")

        # If nothing rendered at all, show fallback
        if not has_any_data and not ff_sections and not (dynamo_cfg and dynamo_cfg.get('enabled')) and not (nb_cfg and nb_cfg.get('enabled')):
            st.info("No ForceFrame / DynaMo / NordBord sections configured for this sport.")

    # ------------------------------------------------------------------
    # Column 3: Athlete summary card
    # ------------------------------------------------------------------

    @staticmethod
    def _display_athlete_card(
        athlete_name: str,
        forcedecks_df: pd.DataFrame,
        forceframe_df: Optional[pd.DataFrame],
        config: Dict[str, Any],
        sport_key: str,
    ) -> None:
        """Column 3: Summary card with key metrics and auto observations."""
        st.markdown("#### Athlete Summary")

        display_name = config.get('display_name', sport_key.title())

        # ---- Body mass ----
        body_mass = None
        for bm_col in ['weight', 'BODY_MASS', 'bodyMassKg']:
            if bm_col in forcedecks_df.columns:
                bm_values = pd.to_numeric(forcedecks_df[bm_col], errors='coerce').dropna()
                if not bm_values.empty:
                    body_mass = float(bm_values.iloc[0])
                    break

        bm_text = f"{body_mass:.1f} kg" if body_mass else "N/A"
        test_count = len(forcedecks_df)

        # ---- Latest test date ----
        date_col = get_date_column(forcedecks_df)
        latest_date = "N/A"
        if date_col and not forcedecks_df.empty:
            dates = pd.to_datetime(forcedecks_df[date_col], errors='coerce').dropna()
            if not dates.empty:
                latest_date = dates.max().strftime('%d %b %Y')

        # ---- Card HTML ----
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #235032 0%, #1a3d25 100%);
             padding: 1.2rem; border-radius: 10px; color: white;">
            <h3 style="margin: 0; color: white;">{athlete_name}</h3>
            <p style="color: #a08e66; margin: 0.3rem 0;">{display_name}</p>
            <hr style="border-color: rgba(255,255,255,0.2); margin: 0.8rem 0;">
            <p style="margin: 0.3rem 0; font-size: 0.9rem;">Body Mass: <b>{bm_text}</b></p>
            <p style="margin: 0.3rem 0; font-size: 0.9rem;">Total Tests: <b>{test_count}</b></p>
            <p style="margin: 0.3rem 0; font-size: 0.9rem;">Latest Test: <b>{latest_date}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # ---- Key metrics ----
        key_metrics_cfg = config.get('key_metrics', [])
        if key_metrics_cfg:
            st.markdown("##### Key Metrics")
            for km in key_metrics_cfg:
                label = km.get('label', '')
                test_types = km.get('test_types', [])
                col = km.get('col', '')
                unit = km.get('unit', '')
                fmt = km.get('format', '.1f')

                # For metrics from DynaMo (empty test_types) skip here
                if not test_types:
                    continue

                subset = SportDiagnosticsModule._filter_by_test_types(forcedecks_df, test_types)
                if subset.empty:
                    continue

                latest = SportDiagnosticsModule._get_latest_row(subset)
                val = pd.to_numeric(latest.get(col, np.nan), errors='coerce')
                if pd.isna(val):
                    continue

                formatted_val = f"{val:{fmt}} {unit}".strip()
                st.metric(label, formatted_val)

        # ---- Auto-generated observations ----
        st.markdown("##### Observations")
        observations = SportDiagnosticsModule._generate_observations(forcedecks_df, forceframe_df, config)
        if observations:
            for obs in observations:
                st.markdown(f"- {obs}")
        else:
            st.caption("Insufficient data for automated observations")

        # ---- Context notes ----
        context_notes = config.get('context_notes', [])
        if context_notes:
            with st.expander("Sport Context Notes"):
                for note in context_notes:
                    st.markdown(f"- {note}")

    # ------------------------------------------------------------------
    # Extra sections (hop test, balance, plyo pushup, etc.)
    # ------------------------------------------------------------------

    @staticmethod
    def _display_extra_sections(
        forcedecks_df: pd.DataFrame,
        config: Dict[str, Any],
        sport_key: str,
        forceframe_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Render extra sections below the 3-column layout."""
        for sec_idx, section in enumerate(config.get('extra_sections', [])):
            section_type = section.get('type', '')
            title = section.get('title', '')
            test_types = section.get('test_types', [])

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #235032 0%, #1a3d25 100%);
                 padding: 1rem; border-radius: 8px; margin: 1.5rem 0 1rem 0;
                 border-left: 4px solid #a08e66;">
                <h3 style="color: white; margin: 0;">{title}</h3>
            </div>
            """, unsafe_allow_html=True)

            if section_type == 'hop_test':
                SportDiagnosticsModule._render_hop_test_section(forcedecks_df, test_types, sport_key, sec_idx)

            elif section_type == 'balance':
                SportDiagnosticsModule._render_balance_section(forcedecks_df, test_types, sport_key, sec_idx)

            elif section_type == 'ppu':
                SportDiagnosticsModule._render_ppu_section(forcedecks_df, test_types, sport_key, sec_idx)

            elif section_type == 'strength_rm':
                st.info("Strength RM data is loaded from manual entry CSV files and displayed in the S&C Diagnostics tab.")

            elif section_type == 'drop_jump':
                SportDiagnosticsModule._render_drop_jump_section(forcedecks_df, test_types, sport_key, sec_idx)

            elif section_type == 'shoulder_isometric':
                SportDiagnosticsModule._render_shoulder_isometric_section(forcedecks_df, test_types, sport_key, sec_idx)

            elif section_type == 'eur':
                SportDiagnosticsModule._render_eur_section(forcedecks_df, test_types, sport_key, sec_idx)

            elif section_type == 'neck_strength':
                SportDiagnosticsModule._render_neck_strength_section(forceframe_df, sport_key, sec_idx)

            elif section_type == 'landing_absorption':
                SportDiagnosticsModule._render_landing_absorption_section(forcedecks_df, test_types, sport_key, sec_idx)

            else:
                st.info(f"Unknown extra section type: '{section_type}'")

    @staticmethod
    def _render_hop_test_section(
        forcedecks_df: pd.DataFrame,
        test_types: List[str],
        sport_key: str,
        sec_idx: int,
    ) -> None:
        """Render 10:5 Hop / Reactive Strength section."""
        subset = SportDiagnosticsModule._filter_by_test_types(forcedecks_df, test_types)
        if subset.empty:
            st.info("No hop test data available")
            return

        # Try RSI columns
        rsi_col = None
        for candidate in ['HOP_BEST_RSI', 'RSI_MODIFIED', 'RSI']:
            if candidate in subset.columns:
                rsi_col = candidate
                break

        if rsi_col is None:
            st.info("No RSI metric column found in hop test data")
            return

        subset = subset.copy()
        subset[rsi_col] = pd.to_numeric(subset[rsi_col], errors='coerce')
        valid = subset.dropna(subset=[rsi_col])

        if valid.empty:
            st.info("No valid RSI values found")
            return

        # RSI scale correction: if median > 10, divide by 100
        rsi_median = valid[rsi_col].median()
        if rsi_median > 10:
            valid[rsi_col] = valid[rsi_col] / 100.0

        latest = valid.iloc[0]
        rsi_val = latest[rsi_col]

        # Show latest RSI
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latest RSI", f"{rsi_val:.2f}")

        with col2:
            # Count of hop tests
            st.metric("Hop Tests Available", f"{len(valid)}")

        # Bar chart if multiple tests
        if len(valid) >= 2:
            date_col = get_date_column(valid)
            if date_col:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=pd.to_datetime(valid[date_col], errors='coerce'),
                    y=valid[rsi_col],
                    marker_color=SportDiagnosticsModule.TEAL_PRIMARY,
                    text=[f"{v:.2f}" for v in valid[rsi_col]],
                    textposition='outside',
                ))
                fig.update_layout(
                    title='RSI Over Time',
                    plot_bgcolor='white', paper_bgcolor='white',
                    font=dict(family='Inter, sans-serif', color='#333'),
                    height=280, margin=dict(l=10, r=10, t=40, b=30),
                    yaxis=dict(title='RSI', showgrid=True, gridcolor='lightgray'),
                    xaxis=dict(showgrid=False, title='Test Date'),
                )
                st.plotly_chart(fig, width='stretch', key=f"sd_{sport_key}_hop_{sec_idx}")

    @staticmethod
    def _render_balance_section(
        forcedecks_df: pd.DataFrame,
        test_types: List[str],
        sport_key: str,
        sec_idx: int,
    ) -> None:
        """Render Balance (QSB/SLSB) section for Shooting athletes."""
        subset = SportDiagnosticsModule._filter_by_test_types(forcedecks_df, test_types)
        if subset.empty:
            st.info("No balance test data available")
            return

        latest = SportDiagnosticsModule._get_latest_row(subset)

        # Balance metrics - VALD stores in meters, display in mm
        balance_metrics = [
            ('BAL_COP_TOTAL_EXCURSION', 'Total CoP Excursion', 'mm', 1000.0),
            ('BAL_COP_MEAN_VELOCITY', 'Mean CoP Velocity', 'mm/s', 1000.0),
            ('BAL_COP_ELLIPSE_AREA', 'CoP 95% Ellipse Area', 'mm\u00b2', 1_000_000.0),
        ]

        found_any = False
        cols = st.columns(len(balance_metrics))
        for i, (col_name, label, unit, multiplier) in enumerate(balance_metrics):
            val = pd.to_numeric(latest.get(col_name, np.nan), errors='coerce')
            if pd.notna(val):
                found_any = True
                display_val = val * multiplier
                with cols[i]:
                    st.metric(label, f"{display_val:.1f} {unit}")

        if not found_any:
            st.info("No balance metric values found (columns may not be populated)")

        st.caption("Balance: lower values indicate better stability. Units converted from m to mm.")

    @staticmethod
    def _render_ppu_section(
        forcedecks_df: pd.DataFrame,
        test_types: List[str],
        sport_key: str,
        sec_idx: int,
    ) -> None:
        """Render Plyo Pushup (Upper Body Power) section."""
        subset = SportDiagnosticsModule._filter_by_test_types(forcedecks_df, test_types)
        if subset.empty:
            st.info("No Plyo Pushup data available")
            return

        latest = SportDiagnosticsModule._get_latest_row(subset)

        ppu_metrics = []
        ppu_height = pd.to_numeric(latest.get('PUSHUP_HEIGHT', np.nan), errors='coerce')
        if pd.notna(ppu_height):
            ppu_metrics.append(('Plyo Pushup Height', f"{ppu_height:.1f} cm"))

        ppu_power = pd.to_numeric(latest.get('BODYMASS_RELATIVE_TAKEOFF_POWER', np.nan), errors='coerce')
        if pd.notna(ppu_power):
            ppu_metrics.append(('PPU Rel. Power', f"{ppu_power:.2f} W/kg"))

        flight_time = pd.to_numeric(latest.get('FLIGHT_TIME', np.nan), errors='coerce')
        if pd.notna(flight_time):
            ppu_metrics.append(('Flight Time', f"{flight_time:.3f} s"))

        if ppu_metrics:
            cols = st.columns(len(ppu_metrics))
            for i, (label, value) in enumerate(ppu_metrics):
                with cols[i]:
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 0.5rem 1rem; border-radius: 6px;
                         margin-bottom: 0.5rem; border-left: 3px solid #235032;">
                        <span style="color: #666; font-size: 0.85rem;">{label}</span><br>
                        <span style="color: #333; font-size: 1.1rem; font-weight: bold;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No Plyo Pushup metric values found")

    # ------------------------------------------------------------------
    # Auto-generated observations
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_observations(
        forcedecks_df: pd.DataFrame,
        forceframe_df: Optional[pd.DataFrame],
        config: Dict[str, Any],
    ) -> List[str]:
        """Auto-detect flags based on asymmetry, trends, etc.

        Returns a list of observation strings.
        """
        observations: List[str] = []
        thresholds = config.get('asymmetry_thresholds', {'caution': 10.0, 'risk': 15.0})
        risk_pct = thresholds.get('risk')
        caution_pct = thresholds.get('caution')

        # ---- Asymmetry checks ----
        for m_cfg in config.get('asymmetry_metrics', []):
            test_types = m_cfg.get('test_types', [])
            left_col = m_cfg.get('left_col', '')
            right_col = m_cfg.get('right_col', '')
            name = m_cfg.get('name', 'Unknown')

            subset = SportDiagnosticsModule._filter_by_test_types(forcedecks_df, test_types)
            if subset.empty:
                continue

            latest = SportDiagnosticsModule._get_latest_row(subset)
            l_val = pd.to_numeric(latest.get(left_col, np.nan), errors='coerce')
            r_val = pd.to_numeric(latest.get(right_col, np.nan), errors='coerce')

            if pd.isna(l_val) or pd.isna(r_val):
                continue

            asym = SportDiagnosticsModule._calc_asymmetry(float(l_val), float(r_val))
            if pd.isna(asym):
                continue

            abs_asym = abs(asym)
            dom = "Right" if asym > 0 else "Left"

            if risk_pct is not None and abs_asym > risk_pct:
                observations.append(f"{name} asymmetry {abs_asym:.1f}% ({dom} dominant) - monitor closely")
            elif caution_pct is not None and abs_asym < caution_pct / 2:
                observations.append(f"{name} symmetry good ({abs_asym:.1f}%)")

        # ---- Trend checks (3+ tests of same type) ----
        for m_cfg in config.get('asymmetry_metrics', []):
            test_types = m_cfg.get('test_types', [])
            right_col = m_cfg.get('right_col', '')
            name = m_cfg.get('name', 'Unknown')

            subset = SportDiagnosticsModule._filter_by_test_types(forcedecks_df, test_types)
            if len(subset) < 3:
                continue

            # Use right_col as proxy for overall force trend
            vals = pd.to_numeric(subset[right_col], errors='coerce').dropna() if right_col in subset.columns else pd.Series(dtype='float64')
            if len(vals) < 3:
                continue

            recent = vals.iloc[0]
            oldest = vals.iloc[-1]
            if oldest > 0:
                change_pct = ((recent - oldest) / oldest) * 100
                base_name = name.split(' L/R')[0].split(' Left')[0].split(' Right')[0]
                if change_pct > 5:
                    observations.append(f"{base_name} trending up (+{change_pct:.0f}% over {len(vals)} tests)")
                elif change_pct < -5:
                    observations.append(f"{base_name} trending down ({change_pct:.0f}% over {len(vals)} tests)")

        # ---- ForceFrame asymmetry check ----
        if forceframe_df is not None and not forceframe_df.empty:
            for section in config.get('forceframe_sections', []):
                test_pattern = section.get('test_pattern', '')
                section_name = section.get('name', '')

                if 'testTypeName' not in forceframe_df.columns:
                    continue

                matched = forceframe_df[
                    forceframe_df['testTypeName'].str.contains(test_pattern, case=False, na=False, regex=True)
                ]
                if matched.empty:
                    continue

                row = matched.iloc[0]
                for force_prefix, label in [('inner', section.get('inner_label', '')), ('outer', section.get('outer_label', ''))]:
                    l = pd.to_numeric(row.get(f'{force_prefix}LeftMaxForce', np.nan), errors='coerce')
                    r = pd.to_numeric(row.get(f'{force_prefix}RightMaxForce', np.nan), errors='coerce')
                    if pd.notna(l) and pd.notna(r) and (l + r) > 0:
                        asym = abs(r - l) / ((l + r) / 2) * 100
                        if asym > 10:
                            dom = "Right" if r > l else "Left"
                            observations.append(
                                f"{section_name} {label} asymmetry {asym:.1f}% ({dom} dominant)"
                            )

        return observations

    # Mapping from benchmark config keys to ForceDecks columns + test types
    BENCHMARK_METRIC_MAP: Dict[str, Dict[str, Any]] = {
        'cmj_height': {'col': 'JUMP_HEIGHT_IMP_MOM', 'test_types': ['CMJ', 'ABCMJ'], 'multiplier': 100.0, 'higher_is_better': True},
        'imtp_relative': {'col': 'NET_PEAK_VERTICAL_FORCE', 'test_types': ['IMTP'], 'multiplier': 1.0, 'higher_is_better': True, 'relative': True},
        'imtp_absolute': {'col': 'NET_PEAK_VERTICAL_FORCE', 'test_types': ['IMTP'], 'multiplier': 1.0, 'higher_is_better': True},
        'rsi': {'col': 'RSI_MODIFIED', 'test_types': ['HJ', 'SLHJ', 'RSHIP'], 'multiplier': 1.0, 'higher_is_better': True},
        'concentric_rfd': {'col': 'CONCENTRIC_RFD', 'test_types': ['CMJ', 'ABCMJ'], 'multiplier': 1.0, 'higher_is_better': True},
        'sj_height': {'col': 'JUMP_HEIGHT_IMP_MOM', 'test_types': ['SJ'], 'multiplier': 100.0, 'higher_is_better': True},
        'ppu_height': {'col': 'PUSHUP_HEIGHT', 'test_types': ['PPU'], 'multiplier': 100.0, 'higher_is_better': True},
        'dj_rsi': {'col': 'RSI', 'test_types': ['DJ', 'SLDJ'], 'multiplier': 1.0, 'higher_is_better': True},
        'eccentric_decel_rfd': {'col': 'ECCENTRIC_DECELERATION_RFD', 'test_types': ['CMJ', 'ABCMJ'], 'multiplier': 1.0, 'higher_is_better': True},
        'peak_relative_power': {'col': 'PEAK_RELATIVE_PROPULSIVE_POWER', 'test_types': ['CMJ', 'ABCMJ'], 'multiplier': 1.0, 'higher_is_better': True},
        'peak_landing_force': {'col': 'RELATIVE_PEAK_LANDING_FORCE', 'test_types': ['CMJ', 'DJ'], 'multiplier': 1.0, 'higher_is_better': False},
        'contraction_time': {'col': 'CONTRACTION_TIME', 'test_types': ['CMJ', 'ABCMJ'], 'multiplier': 1000.0, 'higher_is_better': False},
        'flight_time_contraction_time': {'col': 'FLIGHT_TIME_CONTRACTION_TIME_RATIO', 'test_types': ['CMJ', 'ABCMJ'], 'multiplier': 1.0, 'higher_is_better': True},
        'eccentric_peak_force': {'col': 'ECCENTRIC_PEAK_FORCE', 'test_types': ['CMJ', 'ABCMJ'], 'multiplier': 1.0, 'higher_is_better': True},
        'concentric_peak_force': {'col': 'CONCENTRIC_PEAK_FORCE', 'test_types': ['CMJ', 'ABCMJ'], 'multiplier': 1.0, 'higher_is_better': True},
        # Sport-specific discipline variants (same metric, different test types or context)
        'cmj_height_sprinters': {'col': 'JUMP_HEIGHT_IMP_MOM', 'test_types': ['CMJ'], 'multiplier': 100.0, 'higher_is_better': True},
        'cmj_height_jumpers': {'col': 'JUMP_HEIGHT_IMP_MOM', 'test_types': ['CMJ'], 'multiplier': 100.0, 'higher_is_better': True},
        'cmj_height_throwers': {'col': 'JUMP_HEIGHT_IMP_MOM', 'test_types': ['CMJ'], 'multiplier': 100.0, 'higher_is_better': True},
        'cmj_height_mid_distance': {'col': 'JUMP_HEIGHT_IMP_MOM', 'test_types': ['CMJ'], 'multiplier': 100.0, 'higher_is_better': True},
        'rsi_sprinters': {'col': 'RSI_MODIFIED', 'test_types': ['HJ', 'SLHJ', 'RSHIP'], 'multiplier': 1.0, 'higher_is_better': True},
        'rsi_moguls': {'col': 'RSI_MODIFIED', 'test_types': ['HJ', 'SLHJ', 'RSHIP'], 'multiplier': 1.0, 'higher_is_better': True},
        'ppu_throwers': {'col': 'PUSHUP_HEIGHT', 'test_types': ['PPU'], 'multiplier': 100.0, 'higher_is_better': True},
        'cmj_relative_power': {'col': 'PEAK_RELATIVE_PROPULSIVE_POWER', 'test_types': ['CMJ', 'ABCMJ'], 'multiplier': 1.0, 'higher_is_better': True},
        # Balance metrics (Shooting)
        'qsb_cop_excursion': {'col': 'BAL_COP_TOTAL_EXCURSION', 'test_types': ['QSB'], 'multiplier': 1000.0, 'higher_is_better': False},
        'qsb_cop_velocity': {'col': 'BAL_COP_MEAN_VELOCITY', 'test_types': ['QSB'], 'multiplier': 1000.0, 'higher_is_better': False},
        'qsb_ellipse_area': {'col': 'BAL_COP_ELLIPSE_AREA', 'test_types': ['QSB'], 'multiplier': 1000000.0, 'higher_is_better': False},
        # NordBord / Grip (device-specific - these are approximate since they come from different devices)
        'nordbord_per_leg': {'col': 'NET_PEAK_VERTICAL_FORCE', 'test_types': ['IMTP'], 'multiplier': 1.0, 'higher_is_better': True},  # placeholder - NordBord data separate
        'nordbord_per_leg_sprinters': {'col': 'NET_PEAK_VERTICAL_FORCE', 'test_types': ['IMTP'], 'multiplier': 1.0, 'higher_is_better': True},
        'grip_relative': {'col': 'NET_PEAK_VERTICAL_FORCE', 'test_types': ['IMTP'], 'multiplier': 1.0, 'higher_is_better': True},  # placeholder - DynaMo data separate
        'shoulder_er_ir_ratio': {'col': 'PEAK_VERTICAL_FORCE', 'test_types': ['IMTP'], 'multiplier': 1.0, 'higher_is_better': True},  # placeholder - ForceFrame
    }

    # ------------------------------------------------------------------
    # Benchmarks section (horizontal bars with zone coloring)
    # ------------------------------------------------------------------

    @staticmethod
    def _display_benchmarks_section(
        config: Dict[str, Any],
        athlete_data: pd.DataFrame,
        athlete_name: str,
    ) -> None:
        """Display athlete metrics against sport-specific benchmark zones.

        Reads config['benchmarks'] which is a dict-of-dicts keyed by metric name:
        {'cmj_height': {'elite_m': (40, 48), 'national_m': (34, 40), 'unit': 'cm', ...}}
        Maps each key to ForceDecks column/test types via BENCHMARK_METRIC_MAP.
        """
        benchmarks = config.get('benchmarks')
        if not benchmarks:
            return

        # Determine athlete gender for gender-specific benchmarks
        gender = 'm'
        if athlete_data is not None and not athlete_data.empty:
            for c in ['athlete_sex', 'sex', 'gender']:
                if c in athlete_data.columns:
                    val = athlete_data.iloc[0].get(c, '')
                    if str(val).lower() in ('f', 'female'):
                        gender = 'f'
                    break

        st.markdown("""
        <div style="background: linear-gradient(135deg, #235032 0%, #1a3d25 100%);
             padding: 1rem; border-radius: 8px; margin: 1.5rem 0 1rem 0;
             border-left: 4px solid #a08e66;">
            <h3 style="color: white; margin: 0;">Benchmark Comparison</h3>
            <p style="color: rgba(255,255,255,0.8); margin: 0.25rem 0 0 0; font-size: 0.9rem;">
                Athlete values vs sport-specific benchmark zones
            </p>
        </div>
        """, unsafe_allow_html=True)

        sport_key = config.get('display_name', 'sport').lower().replace(' ', '_')
        rendered_count = 0

        for b_key, b_cfg in benchmarks.items():
            metric_map = SportDiagnosticsModule.BENCHMARK_METRIC_MAP.get(b_key)
            if not metric_map:
                continue

            col_name = metric_map['col']
            test_types = metric_map['test_types']
            multiplier = metric_map.get('multiplier', 1.0)
            higher_is_better = metric_map.get('higher_is_better', True)
            is_relative = metric_map.get('relative', False)
            unit = b_cfg.get('unit', '')

            elite_range = b_cfg.get(f'elite_{gender}', b_cfg.get('elite_m', (0, 0)))
            national_range = b_cfg.get(f'national_{gender}', b_cfg.get('national_m', (0, 0)))
            elite_min = elite_range[0] if isinstance(elite_range, tuple) else elite_range
            national_min = national_range[0] if isinstance(national_range, tuple) else national_range
            source = b_cfg.get('source', '')

            if elite_min == 0 and national_min == 0:
                continue

            athlete_val = None
            if athlete_data is not None and not athlete_data.empty:
                subset = SportDiagnosticsModule._filter_by_test_types(athlete_data, test_types)
                if not subset.empty:
                    latest = SportDiagnosticsModule._get_latest_row(subset)
                    raw = pd.to_numeric(latest.get(col_name, np.nan), errors='coerce')
                    if pd.notna(raw):
                        val = float(raw) * multiplier
                        if is_relative:
                            bm = None
                            for bm_col in ['BODY_MASS', 'CORRECTED_BODY_MASS', 'weight', 'bodyMassKg']:
                                bm_raw = pd.to_numeric(latest.get(bm_col, np.nan), errors='coerce')
                                if pd.notna(bm_raw) and bm_raw > 0:
                                    bm = float(bm_raw)
                                    break
                            if bm:
                                athlete_val = val / bm
                        else:
                            athlete_val = val

            if athlete_val is None:
                continue

            metric_name = b_key.replace('_', ' ').title()
            display_max = max(elite_min * 1.3, athlete_val * 1.2, national_min * 1.4)
            display_min = 0

            fig = go.Figure()
            fig.add_shape(type='rect', xref='x', yref='paper',
                          x0=display_min, x1=national_min, y0=0, y1=1,
                          fillcolor='rgba(120, 144, 156, 0.15)', line=dict(width=0))
            fig.add_shape(type='rect', xref='x', yref='paper',
                          x0=national_min, x1=elite_min, y0=0, y1=1,
                          fillcolor='rgba(58, 112, 80, 0.20)', line=dict(width=0))
            fig.add_shape(type='rect', xref='x', yref='paper',
                          x0=elite_min, x1=display_max, y0=0, y1=1,
                          fillcolor='rgba(35, 80, 50, 0.25)', line=dict(width=0))

            fig.add_vline(x=national_min, line_dash='dot', line_color='#78909C', line_width=1,
                          annotation_text='National', annotation_position='top',
                          annotation_font=dict(size=9, color='#78909C'))
            fig.add_vline(x=elite_min, line_dash='dot', line_color='#235032', line_width=1,
                          annotation_text='Elite', annotation_position='top',
                          annotation_font=dict(size=9, color='#235032'))

            if higher_is_better:
                if athlete_val >= elite_min:
                    marker_color = SportDiagnosticsModule.TEAL_PRIMARY
                elif athlete_val >= national_min:
                    marker_color = SportDiagnosticsModule.TEAL_LIGHT
                else:
                    marker_color = SportDiagnosticsModule.GOLD_ACCENT
            else:
                if athlete_val <= elite_min:
                    marker_color = SportDiagnosticsModule.TEAL_PRIMARY
                elif athlete_val <= national_min:
                    marker_color = SportDiagnosticsModule.TEAL_LIGHT
                else:
                    marker_color = SportDiagnosticsModule.GOLD_ACCENT

            fig.add_trace(go.Scatter(
                x=[athlete_val], y=[metric_name],
                mode='markers+text',
                marker=dict(size=16, color=marker_color, symbol='diamond',
                            line=dict(width=2, color='white')),
                text=[f"{athlete_val:.1f} {unit}"],
                textposition='top center',
                textfont=dict(size=11, color='#333'),
                showlegend=False,
            ))

            fig.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                font=dict(family='Inter, sans-serif', color='#333'),
                height=100, margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(range=[display_min, display_max], showgrid=False,
                           title=f'{unit}' if unit else None),
                yaxis=dict(showgrid=False, showticklabels=True),
            )
            st.plotly_chart(fig, width='stretch',
                            key=f"sd_{sport_key}_bench_{rendered_count}")
            if source:
                st.caption(f"Source: {source}")
            rendered_count += 1

        if rendered_count == 0:
            st.info("Insufficient data for benchmark comparison.")
            return

        st.markdown("""
        <div style="display: flex; gap: 1.5rem; align-items: center; font-size: 0.8rem; color: #666; margin-top: 0.25rem;">
            <span><span style="display:inline-block;width:12px;height:12px;background:rgba(120,144,156,0.35);border-radius:2px;margin-right:4px;vertical-align:middle;"></span>Below National</span>
            <span><span style="display:inline-block;width:12px;height:12px;background:rgba(58,112,80,0.40);border-radius:2px;margin-right:4px;vertical-align:middle;"></span>National Level</span>
            <span><span style="display:inline-block;width:12px;height:12px;background:rgba(35,80,50,0.50);border-radius:2px;margin-right:4px;vertical-align:middle;"></span>Elite Level</span>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Performance ratios
    # ------------------------------------------------------------------

    @staticmethod
    def _display_performance_ratios(
        config: Dict[str, Any],
        athlete_data: pd.DataFrame,
        athlete_name: str,
        forceframe_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Calculate and display performance ratios (EUR, IR:ER, Add:Abd, etc.).

        Reads ``config['performance_ratios']`` - each entry specifies:
          - name, formula, optimal_range, concern_threshold, source (data origin)
          - For ForceDecks: test_types + numerator_col / denominator_col
          - For ForceFrame: ff_test_pattern + numerator_prefix / denominator_prefix
        """
        ratios_cfg = config.get('performance_ratios')
        if not ratios_cfg:
            return

        st.markdown("""
        <div style="background: linear-gradient(135deg, #235032 0%, #1a3d25 100%);
             padding: 1rem; border-radius: 8px; margin: 1.5rem 0 1rem 0;
             border-left: 4px solid #a08e66;">
            <h3 style="color: white; margin: 0;">Performance Ratios</h3>
            <p style="color: rgba(255,255,255,0.8); margin: 0.25rem 0 0 0; font-size: 0.9rem;">
                Key ratios and their optimal ranges
            </p>
        </div>
        """, unsafe_allow_html=True)

        ratio_cards: List[Dict[str, Any]] = []

        for r_cfg in ratios_cfg:
            name = r_cfg.get('name', 'Ratio')
            formula_text = r_cfg.get('formula', '')
            _optimal = r_cfg.get('optimal', None)
            optimal_low = _optimal[0] if isinstance(_optimal, (tuple, list)) and len(_optimal) >= 1 else r_cfg.get('optimal_low', None)
            optimal_high = _optimal[1] if isinstance(_optimal, (tuple, list)) and len(_optimal) >= 2 else r_cfg.get('optimal_high', None)
            concern_low = r_cfg.get('concern_low', None)
            concern_high = r_cfg.get('concern_high', None)
            interpretation = r_cfg.get('description', r_cfg.get('interpretation', ''))
            source = r_cfg.get('source', 'forcedecks')  # 'forcedecks' or 'forceframe'

            ratio_val = None

            if source == 'forcedecks':
                # ForceDecks-based ratio
                num_test_types = r_cfg.get('test_types_num', r_cfg.get('numerator_test_types', []))
                den_test_types = r_cfg.get('test_types_den', r_cfg.get('denominator_test_types', []))
                num_col = r_cfg.get('col_num', r_cfg.get('numerator_col', ''))
                den_col = r_cfg.get('col_den', r_cfg.get('denominator_col', ''))

                if athlete_data is not None and not athlete_data.empty:
                    # Numerator
                    num_subset = SportDiagnosticsModule._filter_by_test_types(athlete_data, num_test_types)
                    den_subset = SportDiagnosticsModule._filter_by_test_types(athlete_data, den_test_types)

                    num_val = np.nan
                    den_val = np.nan

                    if not num_subset.empty and num_col:
                        latest_num = SportDiagnosticsModule._get_latest_row(num_subset)
                        num_val = pd.to_numeric(latest_num.get(num_col, np.nan), errors='coerce')

                    if not den_subset.empty and den_col:
                        latest_den = SportDiagnosticsModule._get_latest_row(den_subset)
                        den_val = pd.to_numeric(latest_den.get(den_col, np.nan), errors='coerce')

                    if pd.notna(num_val) and pd.notna(den_val) and den_val != 0:
                        ratio_val = float(num_val) / float(den_val)

            elif source == 'forceframe':
                # ForceFrame-based ratio (e.g., IR:ER, Add:Abd)
                ff_test_pattern = r_cfg.get('test_pattern', r_cfg.get('ff_test_pattern', ''))
                num_prefix = r_cfg.get('numerator_prefix', 'inner')
                den_prefix = r_cfg.get('denominator_prefix', 'outer')

                if forceframe_df is not None and not forceframe_df.empty and 'testTypeName' in forceframe_df.columns:
                    matched = forceframe_df[
                        forceframe_df['testTypeName'].str.contains(
                            ff_test_pattern, case=False, na=False, regex=True
                        )
                    ]
                    if not matched.empty:
                        ff_date_col = get_date_column(matched)
                        if ff_date_col:
                            matched = matched.copy()
                            matched[ff_date_col] = pd.to_datetime(matched[ff_date_col], errors='coerce')
                            matched = matched.sort_values(ff_date_col, ascending=False)
                        row = matched.iloc[0]

                        # Average L+R for numerator and denominator
                        num_l = pd.to_numeric(row.get(f'{num_prefix}LeftMaxForce', np.nan), errors='coerce')
                        num_r = pd.to_numeric(row.get(f'{num_prefix}RightMaxForce', np.nan), errors='coerce')
                        den_l = pd.to_numeric(row.get(f'{den_prefix}LeftMaxForce', np.nan), errors='coerce')
                        den_r = pd.to_numeric(row.get(f'{den_prefix}RightMaxForce', np.nan), errors='coerce')

                        num_vals = [v for v in [num_l, num_r] if pd.notna(v)]
                        den_vals = [v for v in [den_l, den_r] if pd.notna(v)]

                        if num_vals and den_vals:
                            num_avg = sum(num_vals) / len(num_vals)
                            den_avg = sum(den_vals) / len(den_vals)
                            if den_avg != 0:
                                ratio_val = num_avg / den_avg

            if ratio_val is None:
                continue

            # Determine status color
            status = 'neutral'
            if optimal_low is not None and optimal_high is not None:
                if optimal_low <= ratio_val <= optimal_high:
                    status = 'good'
                elif concern_low is not None and ratio_val < concern_low:
                    status = 'danger'
                elif concern_high is not None and ratio_val > concern_high:
                    status = 'danger'
                else:
                    status = 'warning'
            elif optimal_low is not None:
                status = 'good' if ratio_val >= optimal_low else 'warning'
            elif optimal_high is not None:
                status = 'good' if ratio_val <= optimal_high else 'warning'

            ratio_cards.append({
                'name': name,
                'value': ratio_val,
                'status': status,
                'formula': formula_text,
                'interpretation': interpretation,
                'optimal_low': optimal_low,
                'optimal_high': optimal_high,
            })

        if not ratio_cards:
            st.info("Insufficient data to calculate performance ratios.")
            return

        # Render ratio cards in columns (max 3 per row)
        cols_per_row = min(len(ratio_cards), 3)
        for row_start in range(0, len(ratio_cards), cols_per_row):
            row_cards = ratio_cards[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, card in zip(cols, row_cards):
                with col:
                    status_colors = {
                        'good': '#235032', 'warning': '#a08e66',
                        'danger': '#dc3545', 'neutral': '#6c757d',
                    }
                    bg_color = status_colors.get(card['status'], '#6c757d')
                    opt_text = ''
                    if card['optimal_low'] is not None and card['optimal_high'] is not None:
                        opt_text = f"Optimal: {card['optimal_low']:.2f} - {card['optimal_high']:.2f}"
                    elif card['optimal_low'] is not None:
                        opt_text = f"Optimal: >= {card['optimal_low']:.2f}"
                    elif card['optimal_high'] is not None:
                        opt_text = f"Optimal: <= {card['optimal_high']:.2f}"

                    st.markdown(f"""
                    <div style="background: {bg_color}; padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 0.5rem;">
                        <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.8rem;">{card['name']}</p>
                        <p style="color: white; margin: 0.25rem 0 0 0; font-size: 1.6rem; font-weight: bold;">{card['value']:.2f}</p>
                        <p style="color: rgba(255,255,255,0.7); margin: 0.25rem 0 0 0; font-size: 0.75rem;">{opt_text}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if card['formula']:
                        st.caption(f"Formula: {card['formula']}")
                    if card['interpretation']:
                        st.caption(card['interpretation'])

    # ------------------------------------------------------------------
    # Injury risk traffic-light panel
    # ------------------------------------------------------------------

    @staticmethod
    def _display_injury_risk_panel(
        config: Dict[str, Any],
        athlete_data: pd.DataFrame,
        athlete_name: str,
        forceframe_df: Optional[pd.DataFrame] = None,
        nordbord_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Display injury risk indicators as a traffic-light panel.

        Reads ``config['injury_risk_indicators']`` where each entry has:
          - name, metric_source, threshold_safe, threshold_concern
          - col / test_types (for ForceDecks) or ff_test_pattern (for ForceFrame)
          - higher_is_safer: bool (True = above threshold is safe)
          - source_citation: str (research reference)
        """
        indicators_cfg = config.get('injury_risk_indicators')
        if not indicators_cfg:
            return

        st.markdown("""
        <div style="background: linear-gradient(135deg, #235032 0%, #1a3d25 100%);
             padding: 1rem; border-radius: 8px; margin: 1.5rem 0 1rem 0;
             border-left: 4px solid #a08e66;">
            <h3 style="color: white; margin: 0;">Injury Risk Indicators</h3>
            <p style="color: rgba(255,255,255,0.8); margin: 0.25rem 0 0 0; font-size: 0.9rem;">
                Traffic-light status based on evidence-based thresholds
            </p>
        </div>
        """, unsafe_allow_html=True)

        rendered_indicators: List[Dict[str, Any]] = []

        for ind in indicators_cfg:
            ind_name = ind.get('name', 'Indicator')
            metric_source = ind.get('test', ind.get('metric_source', 'forcedecks')).lower().replace('forcedecks', 'forcedecks').replace('nordbord', 'nordbord').replace('forceframe', 'forceframe')
            # Determine gender-specific threshold
            _raw_thresh = ind.get('threshold', ind.get('threshold_safe'))
            # Check for gender-specific thresholds
            _gender_check = 'm'
            if athlete_data is not None and not athlete_data.empty:
                for _gc in ['athlete_sex', 'sex', 'gender']:
                    if _gc in athlete_data.columns:
                        _gval = str(athlete_data.iloc[0].get(_gc, '')).lower()
                        if _gval in ('f', 'female'):
                            _gender_check = 'f'
                        break
            if _gender_check == 'f' and 'threshold_female' in ind:
                _raw_thresh = ind['threshold_female']
            elif _gender_check == 'm' and 'threshold_male' in ind:
                _raw_thresh = ind['threshold_male']
            # direction: 'below' means risk if below threshold -> higher_is_safer=True
            _direction = ind.get('direction', 'below')
            higher_is_safer = _direction == 'below'  # below threshold = dangerous -> higher is safer
            threshold_safe = _raw_thresh
            threshold_concern = ind.get('threshold_concern')  # optional separate concern level
            source_citation = ind.get('source', ind.get('source_citation', ''))
            col = ind.get('col', '')
            test_types = ind.get('test_types', [])
            unit = ind.get('unit', '')
            multiplier = ind.get('multiplier', 1.0)

            athlete_val = None

            if metric_source == 'forcedecks':
                if athlete_data is not None and not athlete_data.empty and test_types and col:
                    subset = SportDiagnosticsModule._filter_by_test_types(athlete_data, test_types)
                    if not subset.empty:
                        latest = SportDiagnosticsModule._get_latest_row(subset)
                        raw = pd.to_numeric(latest.get(col, np.nan), errors='coerce')
                        if pd.notna(raw):
                            athlete_val = float(raw) * multiplier

            elif metric_source == 'forceframe':
                ff_test_pattern = ind.get('pattern', ind.get('ff_test_pattern', ''))
                ff_col = ind.get('ff_col', '')  # e.g. 'innerLeftMaxForce'
                if forceframe_df is not None and not forceframe_df.empty and 'testTypeName' in forceframe_df.columns:
                    matched = forceframe_df[
                        forceframe_df['testTypeName'].str.contains(
                            ff_test_pattern, case=False, na=False, regex=True
                        )
                    ]
                    if not matched.empty:
                        ff_date_col = get_date_column(matched)
                        if ff_date_col:
                            matched = matched.copy()
                            matched[ff_date_col] = pd.to_datetime(matched[ff_date_col], errors='coerce')
                            matched = matched.sort_values(ff_date_col, ascending=False)
                        row = matched.iloc[0]
                        if ff_col:
                            raw = pd.to_numeric(row.get(ff_col, np.nan), errors='coerce')
                        else:
                            # Average all four quadrants
                            vals = []
                            for prefix in ['inner', 'outer']:
                                for side in ['Left', 'Right']:
                                    v = pd.to_numeric(row.get(f'{prefix}{side}MaxForce', np.nan), errors='coerce')
                                    if pd.notna(v):
                                        vals.append(v)
                            raw = np.mean(vals) if vals else np.nan
                        if pd.notna(raw):
                            athlete_val = float(raw) * multiplier

            elif metric_source == 'nordbord':
                if nordbord_df is not None and not nordbord_df.empty:
                    nb_sorted = nordbord_df.copy()
                    nb_date_col = get_date_column(nb_sorted)
                    if nb_date_col:
                        nb_sorted[nb_date_col] = pd.to_datetime(nb_sorted[nb_date_col], errors='coerce')
                        nb_sorted = nb_sorted.sort_values(nb_date_col, ascending=False)
                    row = nb_sorted.iloc[0]
                    if col:
                        raw = pd.to_numeric(row.get(col, np.nan), errors='coerce')
                    else:
                        # Average L+R
                        l_val = pd.to_numeric(row.get('leftMaxForce', np.nan), errors='coerce')
                        r_val = pd.to_numeric(row.get('rightMaxForce', np.nan), errors='coerce')
                        vals = [v for v in [l_val, r_val] if pd.notna(v)]
                        raw = np.mean(vals) if vals else np.nan
                    if pd.notna(raw):
                        athlete_val = float(raw) * multiplier

            elif metric_source == 'asymmetry':
                # Special: compute asymmetry % from specified L/R cols
                left_col = ind.get('left_col', '')
                right_col = ind.get('right_col', '')
                if athlete_data is not None and not athlete_data.empty and test_types:
                    subset = SportDiagnosticsModule._filter_by_test_types(athlete_data, test_types)
                    if not subset.empty:
                        latest = SportDiagnosticsModule._get_latest_row(subset)
                        l_val = pd.to_numeric(latest.get(left_col, np.nan), errors='coerce')
                        r_val = pd.to_numeric(latest.get(right_col, np.nan), errors='coerce')
                        if pd.notna(l_val) and pd.notna(r_val):
                            asym = abs(SportDiagnosticsModule._calc_asymmetry(float(l_val), float(r_val)))
                            if not pd.isna(asym):
                                athlete_val = asym
                                unit = '%'

            if athlete_val is None:
                continue

            # Determine traffic light status
            if higher_is_safer:
                if threshold_safe is not None and athlete_val >= threshold_safe:
                    traffic = 'green'
                    status_text = 'Safe'
                elif threshold_concern is not None and athlete_val < threshold_concern:
                    traffic = 'red'
                    status_text = 'Intervene'
                else:
                    traffic = 'amber'
                    status_text = 'Monitor'
            else:
                # Lower is safer (e.g., asymmetry %)
                if threshold_safe is not None and athlete_val <= threshold_safe:
                    traffic = 'green'
                    status_text = 'Safe'
                elif threshold_concern is not None and athlete_val > threshold_concern:
                    traffic = 'red'
                    status_text = 'Intervene'
                else:
                    traffic = 'amber'
                    status_text = 'Monitor'

            rendered_indicators.append({
                'name': ind_name,
                'value': athlete_val,
                'unit': unit,
                'traffic': traffic,
                'status_text': status_text,
                'safe_threshold': threshold_safe,
                'concern_threshold': threshold_concern,
                'citation': source_citation,
                'higher_is_safer': higher_is_safer,
            })

        if not rendered_indicators:
            st.info("Insufficient data for injury risk assessment.")
            return

        # Render traffic light rows
        traffic_colors = {
            'green': '#235032',
            'amber': '#a08e66',
            'red': '#dc3545',
        }

        html_rows = []
        for ri in rendered_indicators:
            dot_color = traffic_colors[ri['traffic']]
            threshold_info = ''
            if ri['higher_is_safer']:
                if ri['safe_threshold'] is not None:
                    threshold_info = f"Safe >= {ri['safe_threshold']:.0f}{ri['unit']}"
            else:
                if ri['safe_threshold'] is not None:
                    threshold_info = f"Safe <= {ri['safe_threshold']:.0f}{ri['unit']}"

            citation_html = ''
            if ri['citation']:
                citation_html = f'<span style="color:#999; font-size:0.7rem; margin-left:0.5rem;">({ri["citation"]})</span>'

            html_rows.append(f"""
            <div style="display: flex; gap: 0.75rem; align-items: center; padding: 0.6rem 0;
                 border-bottom: 1px solid #eee;">
                <div style="width: 14px; height: 14px; border-radius: 50%; background: {dot_color};
                     flex-shrink: 0; box-shadow: 0 0 4px {dot_color}40;"></div>
                <div style="flex: 1;">
                    <span style="font-weight: 600; color: #333;">{ri['name']}</span>
                    <span style="color: #666; font-size: 0.85rem; margin-left: 0.5rem;">
                        {ri['value']:.1f} {ri['unit']}
                    </span>
                    <span style="color: {dot_color}; font-size: 0.8rem; font-weight: 500; margin-left: 0.5rem;">
                        ({ri['status_text']}{' - ' + threshold_info if threshold_info else ''})
                    </span>
                    {citation_html}
                </div>
            </div>
            """)

        st.markdown(f"""
        <div style="background: white; border-radius: 8px; padding: 1rem;
             border: 1px solid #eee; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            {''.join(html_rows)}
        </div>
        """, unsafe_allow_html=True)

        # Traffic light legend
        st.markdown("""
        <div style="display: flex; gap: 1.5rem; align-items: center; font-size: 0.8rem;
             color: #666; margin-top: 0.5rem;">
            <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#235032;margin-right:4px;vertical-align:middle;"></span>Safe</span>
            <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#a08e66;margin-right:4px;vertical-align:middle;"></span>Monitor</span>
            <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#dc3545;margin-right:4px;vertical-align:middle;"></span>Intervene</span>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Sport context notes (enhanced visual display)
    # ------------------------------------------------------------------

    @staticmethod
    def _display_sport_context(config: Dict[str, Any]) -> None:
        """Display sport-specific context notes in a styled, categorized panel.

        Reads ``config['context_notes']`` (simple list) and optionally
        ``config['context_categories']`` (dict mapping category name to list
        of notes).  Falls back to a flat list if categories are not provided.
        """
        categories = config.get('context_categories')
        flat_notes = config.get('context_notes', [])
        display_name = config.get('display_name', 'Sport')

        if not categories and not flat_notes:
            return

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #235032 0%, #1a3d25 100%);
             padding: 1rem; border-radius: 8px; margin: 1.5rem 0 1rem 0;
             border-left: 4px solid #a08e66;">
            <h3 style="color: white; margin: 0;">Sport Context: {display_name}</h3>
            <p style="color: rgba(255,255,255,0.8); margin: 0.25rem 0 0 0; font-size: 0.9rem;">
                Key considerations for physical assessment
            </p>
        </div>
        """, unsafe_allow_html=True)

        if categories:
            # Categorized display
            cat_icons = {
                'Performance': '&#x1F3AF;',
                'Injury Prevention': '&#x1F6E1;',
                'Training Implications': '&#x1F4CB;',
                'Assessment Notes': '&#x1F4CA;',
            }

            cols = st.columns(min(len(categories), 3))
            for idx, (cat_name, notes) in enumerate(categories.items()):
                col_idx = idx % min(len(categories), 3)
                with cols[col_idx]:
                    icon = cat_icons.get(cat_name, '&#x2022;')
                    notes_html = ''.join([
                        f'<li style="margin-bottom: 0.3rem; color: #444; font-size: 0.85rem;">{n}</li>'
                        for n in notes
                    ])
                    st.markdown(f"""
                    <div style="background: #f8f9fa; border-radius: 8px; padding: 1rem;
                         border-left: 3px solid #235032; margin-bottom: 0.5rem;">
                        <p style="font-weight: 600; color: #235032; margin: 0 0 0.5rem 0; font-size: 0.9rem;">
                            {icon} {cat_name}
                        </p>
                        <ul style="margin: 0; padding-left: 1.2rem;">
                            {notes_html}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Flat list in a styled panel
            notes_html = ''.join([
                f'<li style="margin-bottom: 0.4rem; color: #444; font-size: 0.85rem;">{n}</li>'
                for n in flat_notes
            ])
            st.markdown(f"""
            <div style="background: #f8f9fa; border-radius: 8px; padding: 1rem;
                 border-left: 3px solid #235032;">
                <ul style="margin: 0; padding-left: 1.2rem;">
                    {notes_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Extra section renderers: drop_jump, shoulder_isometric, eur,
    # neck_strength, landing_absorption
    # ------------------------------------------------------------------

    @staticmethod
    def _render_drop_jump_section(
        forcedecks_df: pd.DataFrame,
        test_types: List[str],
        sport_key: str,
        sec_idx: int,
    ) -> None:
        """Render Drop Jump / Single-Leg Drop Jump metrics (height, RSI, contact time, landing force)."""
        subset = SportDiagnosticsModule._filter_by_test_types(forcedecks_df, test_types)
        if subset.empty:
            st.info("No Drop Jump data available")
            return

        latest = SportDiagnosticsModule._get_latest_row(subset)

        # Key DJ metrics
        dj_metrics = []
        for col_name, label, unit, fmt, mult in [
            ('JUMP_HEIGHT_IMP_MOM', 'Jump Height', 'cm', '.1f', 1.0),
            ('RSI', 'RSI', '', '.2f', 1.0),
            ('CONTACT_TIME', 'Contact Time', 'ms', '.0f', 1000.0),  # s -> ms
            ('PEAK_LANDING_FORCE', 'Peak Landing Force', 'N', '.0f', 1.0),
            ('PEAK_LANDING_FORCE_Left', 'Landing Force L', 'N', '.0f', 1.0),
            ('PEAK_LANDING_FORCE_Right', 'Landing Force R', 'N', '.0f', 1.0),
            ('FLIGHT_TIME', 'Flight Time', 'ms', '.0f', 1000.0),
        ]:
            raw = pd.to_numeric(latest.get(col_name, np.nan), errors='coerce')
            if pd.notna(raw):
                val = float(raw) * mult
                dj_metrics.append((label, f"{val:{fmt}} {unit}"))

        if not dj_metrics:
            # Try SL DJ columns
            for side in ['Left', 'Right']:
                height_col = f'JUMP_HEIGHT_IMP_MOM_{side}'
                raw = pd.to_numeric(latest.get(height_col, np.nan), errors='coerce')
                if pd.notna(raw):
                    dj_metrics.append((f'DJ Height ({side})', f"{float(raw):.1f} cm"))

        if dj_metrics:
            cols_per_row = min(len(dj_metrics), 4)
            cols = st.columns(cols_per_row)
            for i, (label, value) in enumerate(dj_metrics):
                with cols[i % cols_per_row]:
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 0.5rem 1rem; border-radius: 6px;
                         margin-bottom: 0.5rem; border-left: 3px solid #235032;">
                        <span style="color: #666; font-size: 0.85rem;">{label}</span><br>
                        <span style="color: #333; font-size: 1.1rem; font-weight: bold;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # L/R landing asymmetry check
            l_landing = pd.to_numeric(latest.get('PEAK_LANDING_FORCE_Left', np.nan), errors='coerce')
            r_landing = pd.to_numeric(latest.get('PEAK_LANDING_FORCE_Right', np.nan), errors='coerce')
            if pd.notna(l_landing) and pd.notna(r_landing):
                asym = SportDiagnosticsModule._calc_asymmetry(float(l_landing), float(r_landing))
                if not pd.isna(asym):
                    color = SportDiagnosticsModule.RED_FLAG if abs(asym) > 15 else (
                        SportDiagnosticsModule.GOLD_ACCENT if abs(asym) > 10 else '#333'
                    )
                    dom = "Right" if asym > 0 else "Left"
                    st.caption(f"Landing Force Asymmetry: {abs(asym):.1f}% ({dom} dominant)")
        else:
            st.info("No Drop Jump metric values found in data")

    @staticmethod
    def _render_shoulder_isometric_section(
        forcedecks_df: pd.DataFrame,
        test_types: List[str],
        sport_key: str,
        sec_idx: int,
    ) -> None:
        """Render shoulder isometric data (SHLDISOI / SHLDISOY / SHLDISOT) for swimming etc."""
        subset = SportDiagnosticsModule._filter_by_test_types(forcedecks_df, test_types)
        if subset.empty:
            st.info("No Shoulder Isometric data available")
            return

        # Group by test type and show latest for each
        for tt in test_types:
            tt_data = subset[subset['testType'] == tt] if 'testType' in subset.columns else pd.DataFrame()
            if tt_data.empty:
                continue

            latest = SportDiagnosticsModule._get_latest_row(tt_data)
            test_label = {
                'SHLDISOI': 'Shoulder Iso - Internal Rotation',
                'SHLDISOY': 'Shoulder Iso - External Rotation',
                'SHLDISOT': 'Shoulder Iso - Total',
            }.get(tt, tt)

            metrics = []
            for col_name, label, unit, fmt in [
                ('PEAK_VERTICAL_FORCE', 'Peak Force', 'N', '.0f'),
                ('PEAK_VERTICAL_FORCE_Left', 'Peak Force L', 'N', '.0f'),
                ('PEAK_VERTICAL_FORCE_Right', 'Peak Force R', 'N', '.0f'),
                ('NET_PEAK_VERTICAL_FORCE', 'Net Peak Force', 'N', '.0f'),
                ('ISO_BM_REL_FORCE_PEAK', 'Rel. Peak Force', 'N/kg', '.1f'),
            ]:
                val = pd.to_numeric(latest.get(col_name, np.nan), errors='coerce')
                if pd.notna(val):
                    metrics.append((label, f"{float(val):{fmt}} {unit}"))

            if metrics:
                st.markdown(f"**{test_label}**")
                cols = st.columns(min(len(metrics), 4))
                for i, (label, value) in enumerate(metrics):
                    with cols[i % min(len(metrics), 4)]:
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 0.5rem 1rem; border-radius: 6px;
                             margin-bottom: 0.5rem; border-left: 3px solid #235032;">
                            <span style="color: #666; font-size: 0.85rem;">{label}</span><br>
                            <span style="color: #333; font-size: 1.1rem; font-weight: bold;">{value}</span>
                        </div>
                        """, unsafe_allow_html=True)

                # L/R asymmetry
                l_val = pd.to_numeric(latest.get('PEAK_VERTICAL_FORCE_Left', np.nan), errors='coerce')
                r_val = pd.to_numeric(latest.get('PEAK_VERTICAL_FORCE_Right', np.nan), errors='coerce')
                if pd.notna(l_val) and pd.notna(r_val):
                    asym = SportDiagnosticsModule._calc_asymmetry(float(l_val), float(r_val))
                    if not pd.isna(asym):
                        dom = "Right" if asym > 0 else "Left"
                        st.caption(f"{test_label} Asymmetry: {abs(asym):.1f}% ({dom} dominant)")

    @staticmethod
    def _render_eur_section(
        forcedecks_df: pd.DataFrame,
        test_types: List[str],
        sport_key: str,
        sec_idx: int,
    ) -> None:
        """Calculate and display Eccentric Utilization Ratio (CMJ height / SJ height).

        test_types should contain the CMJ code(s) first, then SJ code(s).
        Config can also provide 'cmj_types' and 'sj_types' explicitly.
        """
        if forcedecks_df is None or forcedecks_df.empty:
            st.info("No data available for EUR calculation")
            return

        # Default: infer CMJ and SJ types
        # Read test types from section config if available
        cmj_types = ['CMJ', 'ABCMJ']
        sj_types = ['SJ']
        if section_config:
            _num = section_config.get('test_types_num')
            _den = section_config.get('test_types_den')
            if _num:
                cmj_types = _num
            if _den:
                sj_types = _den

        # Allow config override via the section dict (test_types can be split)
        # Convention: test_types = ['CMJ', 'ABCMJ', 'SJ'] - but we separate them
        if test_types:
            cmj_candidates = [t for t in test_types if 'SJ' not in t or 'CMJ' in t.upper()]
            sj_candidates = [t for t in test_types if t == 'SJ']
            if cmj_candidates:
                cmj_types = cmj_candidates
            if sj_candidates:
                sj_types = sj_candidates

        height_col = 'JUMP_HEIGHT_IMP_MOM'

        cmj_subset = SportDiagnosticsModule._filter_by_test_types(forcedecks_df, cmj_types)
        sj_subset = SportDiagnosticsModule._filter_by_test_types(forcedecks_df, sj_types)

        if cmj_subset.empty or sj_subset.empty:
            st.info("Need both CMJ and SJ data to calculate Eccentric Utilization Ratio")
            return

        cmj_latest = SportDiagnosticsModule._get_latest_row(cmj_subset)
        sj_latest = SportDiagnosticsModule._get_latest_row(sj_subset)

        cmj_height = pd.to_numeric(cmj_latest.get(height_col, np.nan), errors='coerce')
        sj_height = pd.to_numeric(sj_latest.get(height_col, np.nan), errors='coerce')

        if pd.isna(cmj_height) or pd.isna(sj_height) or sj_height == 0:
            st.info("Missing jump height values for EUR calculation")
            return

        eur = float(cmj_height) / float(sj_height)

        # Determine color
        if 1.05 <= eur <= 1.15:
            status_color = '#235032'  # Optimal
            status_text = 'Optimal SSC utilization'
        elif eur < 1.0:
            status_color = '#dc3545'  # Red - poor SSC utilization
            status_text = 'Poor SSC utilization (CMJ < SJ)'
        elif eur > 1.25:
            status_color = '#a08e66'  # Gold - over-reliance on SSC
            status_text = 'High SSC reliance - may indicate strength deficit'
        else:
            status_color = '#3a7050'  # Acceptable
            status_text = 'Acceptable SSC utilization'

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style="background: {status_color}; padding: 1rem; border-radius: 8px; text-align: center;">
                <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.8rem;">EUR (CMJ:SJ)</p>
                <p style="color: white; margin: 0.25rem 0 0 0; font-size: 1.8rem; font-weight: bold;">{eur:.2f}</p>
                <p style="color: rgba(255,255,255,0.7); margin: 0.25rem 0 0 0; font-size: 0.75rem;">Optimal: 1.05-1.15</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.metric("CMJ Height", f"{float(cmj_height):.1f} cm")
        with col3:
            st.metric("SJ Height", f"{float(sj_height):.1f} cm")

        st.caption(f"{status_text}. EUR = CMJ Height / SJ Height. Values >1.0 indicate effective stretch-shortening cycle (SSC) utilization.")

    @staticmethod
    def _render_neck_strength_section(
        forceframe_df: Optional[pd.DataFrame],
        sport_key: str,
        sec_idx: int,
    ) -> None:
        """Render 4-Way Neck ForceFrame data for wrestling, combat sports, etc."""
        if forceframe_df is None or forceframe_df.empty:
            st.info("No ForceFrame data available for neck strength")
            return

        if 'testTypeName' not in forceframe_df.columns:
            st.info("No testTypeName column in ForceFrame data")
            return

        neck_data = forceframe_df[
            forceframe_df['testTypeName'].str.contains(r'Neck|4.Way', case=False, na=False, regex=True)
        ]
        if neck_data.empty:
            st.info("No 4-Way Neck test data available")
            return

        # Sort by date and take latest
        ff_date_col = get_date_column(neck_data)
        if ff_date_col:
            neck_data = neck_data.copy()
            neck_data[ff_date_col] = pd.to_datetime(neck_data[ff_date_col], errors='coerce')
            neck_data = neck_data.sort_values(ff_date_col, ascending=False)
        row = neck_data.iloc[0]

        # Extract quadrant forces
        inner_l = pd.to_numeric(row.get('innerLeftMaxForce', np.nan), errors='coerce')
        inner_r = pd.to_numeric(row.get('innerRightMaxForce', np.nan), errors='coerce')
        outer_l = pd.to_numeric(row.get('outerLeftMaxForce', np.nan), errors='coerce')
        outer_r = pd.to_numeric(row.get('outerRightMaxForce', np.nan), errors='coerce')

        # For 4-Way Neck: Inner = Flexion/Extension, Outer = Lateral Flexion (varies by setup)
        # Display all available values
        neck_metrics = []
        labels_map = [
            (inner_l, 'Flexion L (Inner L)'),
            (inner_r, 'Extension R (Inner R)'),
            (outer_l, 'Lat Flex L (Outer L)'),
            (outer_r, 'Lat Flex R (Outer R)'),
        ]

        for val, label in labels_map:
            if pd.notna(val):
                neck_metrics.append((label, float(val)))

        if not neck_metrics:
            st.info("No valid neck force values found")
            return

        # Bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[nm[0] for nm in neck_metrics],
            y=[nm[1] for nm in neck_metrics],
            marker_color=[SportDiagnosticsModule.TEAL_PRIMARY, SportDiagnosticsModule.TEAL_LIGHT,
                          SportDiagnosticsModule.GOLD_ACCENT, SportDiagnosticsModule.BLUE][:len(neck_metrics)],
            text=[f"{nm[1]:.0f}N" for nm in neck_metrics],
            textposition='outside',
        ))

        fig.update_layout(
            title='4-Way Neck Strength',
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='Inter, sans-serif', color='#333'),
            height=280, margin=dict(l=10, r=10, t=40, b=10),
            yaxis=dict(title='Force (N)', showgrid=True, gridcolor='lightgray'),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, width='stretch', key=f"sd_{sport_key}_neck_{sec_idx}")

        # Asymmetry checks
        if pd.notna(inner_l) and pd.notna(inner_r):
            flex_ext_asym = SportDiagnosticsModule._calc_asymmetry(float(inner_l), float(inner_r))
            if not pd.isna(flex_ext_asym):
                st.caption(f"Flexion/Extension asymmetry: {abs(flex_ext_asym):.1f}%")
        if pd.notna(outer_l) and pd.notna(outer_r):
            lat_asym = SportDiagnosticsModule._calc_asymmetry(float(outer_l), float(outer_r))
            if not pd.isna(lat_asym):
                st.caption(f"Lateral Flexion asymmetry: {abs(lat_asym):.1f}%")

    @staticmethod
    def _render_landing_absorption_section(
        forcedecks_df: pd.DataFrame,
        test_types: List[str],
        sport_key: str,
        sec_idx: int,
    ) -> None:
        """Render landing force / absorption metrics from DJ tests.

        Focuses on peak landing force, landing RFD, and time to stabilization.
        """
        subset = SportDiagnosticsModule._filter_by_test_types(forcedecks_df, test_types)
        if subset.empty:
            st.info("No landing data available (need DJ or similar test)")
            return

        latest = SportDiagnosticsModule._get_latest_row(subset)

        # Landing-specific metrics
        landing_metrics = []
        for col_name, label, unit, fmt, mult in [
            ('PEAK_LANDING_FORCE', 'Peak Landing Force', 'N', '.0f', 1.0),
            ('PEAK_LANDING_FORCE_Left', 'Landing Force L', 'N', '.0f', 1.0),
            ('PEAK_LANDING_FORCE_Right', 'Landing Force R', 'N', '.0f', 1.0),
            ('LANDING_RFD', 'Landing RFD', 'N/s', '.0f', 1.0),
            ('TIME_TO_STABILISATION', 'Time to Stabilisation', 's', '.2f', 1.0),
            ('RELATIVE_PEAK_LANDING_FORCE', 'Rel. Landing Force', 'BW', '.1f', 1.0),
        ]:
            raw = pd.to_numeric(latest.get(col_name, np.nan), errors='coerce')
            if pd.notna(raw):
                val = float(raw) * mult
                landing_metrics.append((label, f"{val:{fmt}} {unit}"))

        if not landing_metrics:
            st.info("No landing absorption metric values found")
            return

        cols_per_row = min(len(landing_metrics), 3)
        cols = st.columns(cols_per_row)
        for i, (label, value) in enumerate(landing_metrics):
            with cols[i % cols_per_row]:
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 0.5rem 1rem; border-radius: 6px;
                     margin-bottom: 0.5rem; border-left: 3px solid #235032;">
                    <span style="color: #666; font-size: 0.85rem;">{label}</span><br>
                    <span style="color: #333; font-size: 1.1rem; font-weight: bold;">{value}</span>
                </div>
                """, unsafe_allow_html=True)

        # L/R landing asymmetry
        l_landing = pd.to_numeric(latest.get('PEAK_LANDING_FORCE_Left', np.nan), errors='coerce')
        r_landing = pd.to_numeric(latest.get('PEAK_LANDING_FORCE_Right', np.nan), errors='coerce')
        if pd.notna(l_landing) and pd.notna(r_landing):
            asym = SportDiagnosticsModule._calc_asymmetry(float(l_landing), float(r_landing))
            if not pd.isna(asym):
                flag = " - FLAG" if abs(asym) > 15 else ""
                dom = "Right" if asym > 0 else "Left"
                st.caption(f"Landing Force Asymmetry: {abs(asym):.1f}% ({dom} dominant){flag}")

        st.caption("Lower landing force and faster stabilisation indicate better absorption capacity and reduced injury risk.")

    # ------------------------------------------------------------------
    # Export buttons
    # ------------------------------------------------------------------

    @staticmethod
    def export_buttons(
        sport_key: str,
        athlete_name: str,
        forcedecks_df: pd.DataFrame,
        forceframe_df: Optional[pd.DataFrame] = None,
        dynamo_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """PDF / HTML download buttons for the diagnostic report."""
        config = SportDiagnosticsModule._get_config(sport_key)
        display_name = config.get('display_name', sport_key.title()) if config else sport_key.title()

        if generate_individual_pdf_report is None and generate_individual_html_report is None:
            st.caption("Report export not available (report_export module not found)")
            return

        # Collect observations for reports
        observations = []
        if config:
            observations = SportDiagnosticsModule._generate_observations(
                forcedecks_df if forcedecks_df is not None else pd.DataFrame(),
                forceframe_df,
                config,
            )

        metadata = {
            'sport': display_name,
            'sport_key': sport_key,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'report_type': 'Sport Diagnostics',
        }

        col1, col2 = st.columns(2)

        with col1:
            if generate_individual_pdf_report is not None:
                try:
                    pdf_bytes = generate_individual_pdf_report(
                        athlete_name=athlete_name,
                        sport=display_name,
                        charts=None,
                        data_tables=None,
                        observations=observations,
                        metadata=metadata,
                    )
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"{athlete_name}_{sport_key}_diagnostics.pdf",
                        mime="application/pdf",
                        key=f"sd_{sport_key}_pdf_export",
                    )
                except Exception as e:
                    st.caption(f"PDF export unavailable: {e}")

        with col2:
            if generate_individual_html_report is not None:
                try:
                    html_content = generate_individual_html_report(
                        athlete_name=athlete_name,
                        sport=display_name,
                        charts=None,
                        data_tables=None,
                        observations=observations,
                        metadata=metadata,
                    )
                    st.download_button(
                        label="Download HTML Report",
                        data=html_content,
                        file_name=f"{athlete_name}_{sport_key}_diagnostics.html",
                        mime="text/html",
                        key=f"sd_{sport_key}_html_export",
                    )
                except Exception as e:
                    st.caption(f"HTML export unavailable: {e}")


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == '__main__':
    print("SportDiagnosticsModule loaded successfully.")
    print(f"Configured sports: {list(SPORT_DIAGNOSTIC_CONFIGS.keys())}")
    print(f"Report export available: PDF={'Yes' if generate_individual_pdf_report else 'No'}, "
          f"HTML={'Yes' if generate_individual_html_report else 'No'}")
