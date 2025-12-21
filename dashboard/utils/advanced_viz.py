"""
Advanced Visualization Module
VALD Performance Dashboard - Saudi National Team

Inspired by professional VALD Shiny apps
Features: Quadrant analysis, Parallel coordinates, Violin plots, Best-of-day analysis

Reference: https://www.linkedin.com/posts/đorđe-sekulić-7912a0248_rstudio-shiny-vald-activity-7376214096016404480
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
import streamlit as st

# ============================================================================
# QUADRANT ANALYSIS
# ============================================================================

def create_quadrant_plot(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    athlete_column: str = 'Name',
    sport_column: str = 'athlete_sport',
    title: Optional[str] = None,
    show_benchmarks: bool = True,
    color_by_sport: bool = True
) -> go.Figure:
    """
    Create quadrant analysis plot (2D scatter with reference lines)

    Quadrants:
    - Q1 (Top-Right): High X, High Y - ELITE
    - Q2 (Top-Left): Low X, High Y
    - Q3 (Bottom-Left): Low X, Low Y - NEEDS ATTENTION
    - Q4 (Bottom-Right): High X, Low Y

    Parameters:
    -----------
    df : pd.DataFrame
        Data with metrics
    x_metric : str
        Column name for X-axis metric
    y_metric : str
        Column name for Y-axis metric
    athlete_column : str
        Column with athlete names
    sport_column : str
        Column with sport names
    title : str
        Custom title (auto-generated if None)
    show_benchmarks : bool
        Show median reference lines
    color_by_sport : bool
        Color points by sport
    """

    # Filter valid data - handle missing sport column
    required_cols = [athlete_column, x_metric, y_metric]
    has_sport = sport_column in df.columns
    if has_sport:
        required_cols.insert(1, sport_column)

    plot_df = df[required_cols].dropna()

    if plot_df.empty:
        st.warning(f"No valid data for {x_metric} vs {y_metric}")
        return None

    # Calculate medians for reference lines
    x_median = plot_df[x_metric].median()
    y_median = plot_df[y_metric].median()

    # Create figure
    if color_by_sport and has_sport:
        fig = px.scatter(
            plot_df,
            x=x_metric,
            y=y_metric,
            color=sport_column,
            hover_name=athlete_column,
            title=title or f"{y_metric} vs {x_metric}",
            labels={
                x_metric: x_metric.split('[')[0].strip(),
                y_metric: y_metric.split('[')[0].strip()
            }
        )
    else:
        fig = px.scatter(
            plot_df,
            x=x_metric,
            y=y_metric,
            hover_name=athlete_column,
            title=title or f"{y_metric} vs {x_metric}",
            labels={
                x_metric: x_metric.split('[')[0].strip(),
                y_metric: y_metric.split('[')[0].strip()
            }
        )

    # Add reference lines (median splits)
    if show_benchmarks:
        # Vertical line (X median)
        fig.add_vline(
            x=x_median,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Median: {x_median:.1f}",
            annotation_position="top"
        )

        # Horizontal line (Y median)
        fig.add_hline(
            y=y_median,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Median: {y_median:.1f}",
            annotation_position="right"
        )

        # Add quadrant labels
        x_range = plot_df[x_metric].max() - plot_df[x_metric].min()
        y_range = plot_df[y_metric].max() - plot_df[y_metric].min()

        fig.add_annotation(
            x=x_median + x_range * 0.3,
            y=y_median + y_range * 0.3,
            text="<b>ELITE</b><br>High/High",
            showarrow=False,
            font=dict(size=14, color="green"),
            bgcolor="rgba(0,255,0,0.1)",
            borderpad=10
        )

        fig.add_annotation(
            x=x_median - x_range * 0.3,
            y=y_median - y_range * 0.3,
            text="<b>NEEDS FOCUS</b><br>Low/Low",
            showarrow=False,
            font=dict(size=14, color="red"),
            bgcolor="rgba(255,0,0,0.1)",
            borderpad=10
        )

    # Styling
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter",
        hovermode='closest',
        height=600
    )

    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color='white')),
        textposition='top center'
    )

    return fig


def create_multi_quadrant_grid(
    df: pd.DataFrame,
    metric_pairs: List[Tuple[str, str]],
    athlete_column: str = 'Name',
    sport_column: str = 'athlete_sport'
) -> go.Figure:
    """
    Create multiple quadrant plots in a grid layout

    Example:
    metric_pairs = [
        ('CMJ Height', 'Peak Power'),
        ('CMJ Height', 'RSI Modified'),
        ('Peak Force', 'RFD')
    ]
    """

    n_plots = len(metric_pairs)
    n_cols = 2
    n_rows = (n_plots + 1) // 2

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{y} vs {x}" for x, y in metric_pairs],
        horizontal_spacing=0.12,
        vertical_spacing=0.15
    )

    has_sport = sport_column in df.columns

    for i, (x_metric, y_metric) in enumerate(metric_pairs):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1

        required_cols = [athlete_column, x_metric, y_metric]
        if has_sport:
            required_cols.insert(1, sport_column)
        plot_df = df[required_cols].dropna()

        if not plot_df.empty:
            # Add scatter trace
            fig.add_trace(
                go.Scatter(
                    x=plot_df[x_metric],
                    y=plot_df[y_metric],
                    mode='markers',
                    marker=dict(size=8, color='#0d4f3c'),
                    text=plot_df[athlete_column],
                    hovertemplate='<b>%{text}</b><br>%{x:.2f}, %{y:.2f}<extra></extra>',
                    showlegend=False
                ),
                row=row,
                col=col
            )

            # Add median lines
            x_median = plot_df[x_metric].median()
            y_median = plot_df[y_metric].median()

            fig.add_vline(
                x=x_median,
                line_dash="dash",
                line_color="gray",
                line_width=1,
                row=row,
                col=col
            )

            fig.add_hline(
                y=y_median,
                line_dash="dash",
                line_color="gray",
                line_width=1,
                row=row,
                col=col
            )

    fig.update_layout(
        height=400 * n_rows,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig


# ============================================================================
# PARALLEL COORDINATES
# ============================================================================

def create_parallel_coordinates(
    df: pd.DataFrame,
    metrics: List[str],
    athlete_column: str = 'Name',
    sport_column: str = 'athlete_sport',
    color_by: str = 'sport',
    title: str = "Multi-Metric Comparison"
) -> go.Figure:
    """
    Create parallel coordinates plot for multi-dimensional comparison

    Perfect for:
    - Comparing multiple metrics simultaneously
    - Identifying athlete profiles
    - Spotting patterns across metrics

    Parameters:
    -----------
    df : pd.DataFrame
        Data
    metrics : List[str]
        List of metric columns to include
    athlete_column : str
        Column with athlete names
    sport_column : str
        Column with sport names
    color_by : str
        'sport' or 'athlete' or metric name
    """

    # Filter to valid data - handle missing sport column
    has_sport = sport_column in df.columns
    required_cols = [athlete_column] + metrics
    if has_sport:
        required_cols.insert(1, sport_column)
    plot_df = df[required_cols].dropna()

    if plot_df.empty:
        st.warning("No valid data for parallel coordinates")
        return None

    # Normalize metrics to 0-1 scale for better visualization
    normalized_df = plot_df.copy()
    for metric in metrics:
        min_val = plot_df[metric].min()
        max_val = plot_df[metric].max()
        if max_val > min_val:
            normalized_df[f'{metric}_norm'] = (plot_df[metric] - min_val) / (max_val - min_val)
        else:
            normalized_df[f'{metric}_norm'] = 0.5

    # Create dimensions for parallel coordinates
    dimensions = []

    for metric in metrics:
        dimensions.append(
            dict(
                label=metric.split('[')[0].strip(),
                values=plot_df[metric],
                range=[plot_df[metric].min(), plot_df[metric].max()]
            )
        )

    # Color mapping
    if color_by == 'sport' and has_sport:
        # Encode sports as numbers for coloring
        sport_codes = pd.Categorical(plot_df[sport_column]).codes
        color_values = sport_codes
        colorscale = 'Viridis'
    else:
        # Use first metric for coloring
        color_values = plot_df[metrics[0]]
        colorscale = 'RdYlGn'

    # Create parallel coordinates
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=color_values,
                colorscale=colorscale,
                showscale=True,
                cmin=min(color_values),
                cmax=max(color_values)
            ),
            dimensions=dimensions
        )
    )

    fig.update_layout(
        title=title,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12),
        height=600
    )

    return fig


# ============================================================================
# VIOLIN PLOTS
# ============================================================================

def create_violin_plot(
    df: pd.DataFrame,
    metric: str,
    group_by: str = 'athlete_sport',
    title: Optional[str] = None,
    show_box: bool = True,
    show_points: bool = True
) -> go.Figure:
    """
    Create violin plot for distribution visualization

    Better than box plots for:
    - Showing full distribution shape
    - Identifying bimodal distributions
    - Visualizing density
    """

    # Handle missing group_by column
    if group_by not in df.columns:
        st.warning(f"Column '{group_by}' not found in data. Cannot create violin plot.")
        return None

    plot_df = df[[group_by, metric]].dropna()

    if plot_df.empty:
        st.warning(f"No data for {metric}")
        return None

    fig = px.violin(
        plot_df,
        y=metric,
        x=group_by,
        box=show_box,
        points='all' if show_points else False,
        title=title or f"{metric} Distribution by {group_by}",
        color=group_by
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter",
        showlegend=False,
        height=600
    )

    return fig


# ============================================================================
# BEST-OF-DAY ANALYSIS
# ============================================================================

def get_best_of_day_per_athlete(
    df: pd.DataFrame,
    date_column: str = 'recordedDateUtc',
    athlete_column: str = 'Name',
    metric: str = 'Jump Height (Flight Time) [cm]',
    higher_is_better: bool = True
) -> pd.DataFrame:
    """
    Get best performance per athlete per day

    Use for:
    - Filtering out warm-up trials
    - Focusing on peak performance
    - Trend analysis of best efforts
    """

    df = df.copy()

    # Extract date (not datetime)
    df['test_date'] = pd.to_datetime(df[date_column]).dt.date

    # Group by athlete and date, get best
    if higher_is_better:
        best_df = df.loc[df.groupby([athlete_column, 'test_date'])[metric].idxmax()]
    else:
        best_df = df.loc[df.groupby([athlete_column, 'test_date'])[metric].idxmin()]

    return best_df


def create_best_of_day_trend(
    df: pd.DataFrame,
    athletes: List[str],
    metric: str,
    date_column: str = 'recordedDateUtc',
    athlete_column: str = 'Name',
    higher_is_better: bool = True
) -> go.Figure:
    """
    Create trend line showing best-of-day performance over time
    """

    # Get best of day
    best_df = get_best_of_day_per_athlete(
        df,
        date_column=date_column,
        athlete_column=athlete_column,
        metric=metric,
        higher_is_better=higher_is_better
    )

    # Filter to selected athletes
    best_df = best_df[best_df[athlete_column].isin(athletes)]

    if best_df.empty:
        st.warning("No data for selected athletes")
        return None

    fig = px.line(
        best_df,
        x='test_date',
        y=metric,
        color=athlete_column,
        markers=True,
        title=f"{metric} - Best of Day Trends"
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter",
        hovermode='x unified',
        height=600
    )

    return fig


# ============================================================================
# RELIABILITY ANALYSIS
# ============================================================================

def calculate_reliability_metrics(
    df: pd.DataFrame,
    athlete_column: str = 'Name',
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Calculate reliability metrics for test-retest consistency

    Metrics:
    - CV% (Coefficient of Variation): Std / Mean * 100
    - Typical Error: Std / sqrt(2)
    - ICC (Intraclass Correlation): Not implemented yet

    Good reliability:
    - CV% < 10%
    - Typical Error < 5% of mean
    """

    if metrics is None:
        metrics = [col for col in df.select_dtypes(include=[np.number]).columns]

    reliability_data = []

    for metric in metrics:
        metric_df = df[[athlete_column, metric]].dropna()

        if metric_df.empty:
            continue

        # For athletes with multiple tests
        athlete_stats = metric_df.groupby(athlete_column)[metric].agg(['mean', 'std', 'count'])
        athlete_stats = athlete_stats[athlete_stats['count'] >= 2]  # Need at least 2 tests

        if athlete_stats.empty:
            continue

        # Calculate CV%
        athlete_stats['cv_percent'] = (athlete_stats['std'] / athlete_stats['mean']) * 100

        # Calculate Typical Error
        athlete_stats['typical_error'] = athlete_stats['std'] / np.sqrt(2)
        athlete_stats['te_percent'] = (athlete_stats['typical_error'] / athlete_stats['mean']) * 100

        # Summary across all athletes
        summary = {
            'metric': metric.split('[')[0].strip(),
            'n_athletes': len(athlete_stats),
            'mean_cv_percent': athlete_stats['cv_percent'].mean(),
            'median_cv_percent': athlete_stats['cv_percent'].median(),
            'mean_te_percent': athlete_stats['te_percent'].mean(),
            'median_te_percent': athlete_stats['te_percent'].median(),
            'reliability_rating': 'Good' if athlete_stats['cv_percent'].median() < 10 else 'Moderate' if athlete_stats['cv_percent'].median() < 15 else 'Poor'
        }

        reliability_data.append(summary)

    reliability_df = pd.DataFrame(reliability_data)

    return reliability_df


def create_reliability_plot(reliability_df: pd.DataFrame) -> go.Figure:
    """
    Visualize reliability metrics
    """

    if reliability_df.empty:
        st.warning("No reliability data")
        return None

    fig = go.Figure()

    # CV% bars
    fig.add_trace(go.Bar(
        x=reliability_df['metric'],
        y=reliability_df['median_cv_percent'],
        name='CV%',
        marker_color='#0d4f3c',
        text=reliability_df['median_cv_percent'].round(1),
        textposition='outside'
    ))

    # Add reference line at 10% (good reliability threshold)
    fig.add_hline(
        y=10,
        line_dash="dash",
        line_color="green",
        annotation_text="Good reliability (<10%)",
        annotation_position="right"
    )

    fig.add_hline(
        y=15,
        line_dash="dash",
        line_color="orange",
        annotation_text="Moderate reliability",
        annotation_position="right"
    )

    fig.update_layout(
        title="Test-Retest Reliability (CV%)",
        xaxis_title="Metric",
        yaxis_title="Coefficient of Variation (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter",
        height=500
    )

    return fig


# ============================================================================
# RANKING WITH DATA LABELS
# ============================================================================

def create_labeled_ranking(
    df: pd.DataFrame,
    metric: str,
    athlete_column: str = 'Name',
    sport_column: str = 'athlete_sport',
    top_n: int = 15,
    title: Optional[str] = None
) -> go.Figure:
    """
    Create ranking chart with data labels on bars
    """

    # Get latest test per athlete
    latest_df = df.sort_values('recordedDateUtc').groupby(athlete_column).last().reset_index()

    # Filter valid data and sort - handle missing sport column
    has_sport = sport_column in latest_df.columns
    required_cols = [athlete_column, metric]
    if has_sport:
        required_cols.insert(1, sport_column)
    rank_df = latest_df[required_cols].dropna()
    rank_df = rank_df.sort_values(metric, ascending=False).head(top_n)

    if rank_df.empty:
        st.warning(f"No data for {metric}")
        return None

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=rank_df[athlete_column],
        y=rank_df[metric],
        marker_color='#0d4f3c',
        text=rank_df[metric].round(2),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=title or f"Top {top_n} - {metric}",
        xaxis_title="Athlete",
        yaxis_title=metric.split('[')[0].strip(),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter",
        height=500,
        showlegend=False
    )

    fig.update_xaxes(tickangle=-45)

    return fig


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("Advanced Visualization Module - Examples")
    print("=" * 60)
    print()
    print("1. Quadrant Analysis:")
    print("   fig = create_quadrant_plot(df, 'CMJ Height', 'Peak Power')")
    print()
    print("2. Parallel Coordinates:")
    print("   fig = create_parallel_coordinates(df, ['CMJ', 'SJ', 'RSI'])")
    print()
    print("3. Violin Plots:")
    print("   fig = create_violin_plot(df, 'Jump Height', group_by='sport')")
    print()
    print("4. Best-of-Day Trends:")
    print("   fig = create_best_of_day_trend(df, athletes=['A', 'B'], metric='CMJ')")
    print()
    print("5. Reliability Analysis:")
    print("   reliability = calculate_reliability_metrics(df)")
    print("   fig = create_reliability_plot(reliability)")
