"""
Plotly Chart Theming for VALD Performance Dashboard
Team Saudi Professional Theme
"""

from .colors import (
    TEAL_PRIMARY, TEAL_DARK, TEAL_LIGHT, GOLD_ACCENT, GRAY_BLUE,
    TEXT_PRIMARY, TEXT_SECONDARY, BORDER, CHART_COLORS, ZONE_COLORS
)


def get_plotly_layout():
    """
    Return consistent Plotly layout settings.

    Returns:
        dict: Layout configuration for Plotly figures
    """
    return {
        'font': {
            'family': 'Source Sans 3, Inter, sans-serif',
            'color': TEXT_PRIMARY,
            'size': 12,
        },
        'paper_bgcolor': 'white',
        'plot_bgcolor': 'white',
        'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
        'hovermode': 'closest',
        'legend': {
            'bgcolor': 'rgba(255, 255, 255, 0.9)',
            'bordercolor': BORDER,
            'borderwidth': 1,
            'font': {'size': 11},
        },
    }


def get_axis_style():
    """
    Return consistent axis styling.

    Returns:
        dict: Axis configuration for Plotly figures
    """
    return {
        'gridcolor': 'rgba(128, 128, 128, 0.12)',
        'linecolor': BORDER,
        'tickfont': {'size': 11, 'color': TEXT_SECONDARY},
        'titlefont': {'size': 12, 'color': TEXT_PRIMARY},
        'zeroline': False,
    }


def apply_team_saudi_theme(fig, title: str = None, show_legend: bool = True):
    """
    Apply Team Saudi styling to a Plotly figure.

    Args:
        fig: Plotly figure object
        title: Optional chart title
        show_legend: Whether to show legend (default True)

    Returns:
        fig: Styled Plotly figure
    """
    layout = get_plotly_layout()
    axis_style = get_axis_style()

    fig.update_layout(
        font=layout['font'],
        paper_bgcolor=layout['paper_bgcolor'],
        plot_bgcolor=layout['plot_bgcolor'],
        margin=layout['margin'],
        hovermode=layout['hovermode'],
        showlegend=show_legend,
        legend=layout['legend'] if show_legend else None,
    )

    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(
                    size=14,
                    family='Space Grotesk, Source Sans 3, sans-serif',
                    color=TEXT_PRIMARY,
                ),
                x=0,
                xanchor='left',
            )
        )

    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)

    return fig


def apply_minimal_theme(fig):
    """
    Apply minimal styling for small/inline charts.

    Args:
        fig: Plotly figure object

    Returns:
        fig: Styled Plotly figure
    """
    fig.update_layout(
        font={'family': 'Source Sans 3, sans-serif', 'color': TEXT_PRIMARY, 'size': 10},
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin={'l': 40, 'r': 20, 't': 30, 'b': 30},
        showlegend=False,
    )

    fig.update_xaxes(
        gridcolor='rgba(128, 128, 128, 0.1)',
        linecolor=BORDER,
        tickfont={'size': 9},
    )

    fig.update_yaxes(
        gridcolor='rgba(128, 128, 128, 0.1)',
        linecolor=BORDER,
        tickfont={'size': 9},
    )

    return fig


def get_line_colors(n: int = 1) -> list:
    """
    Get line colors for charts.

    Args:
        n: Number of colors needed

    Returns:
        list: List of color hex codes
    """
    if n == 1:
        return [TEAL_PRIMARY]
    return CHART_COLORS[:n]


def get_primary_color() -> str:
    """Return primary chart color."""
    return TEAL_PRIMARY


def get_accent_color() -> str:
    """Return accent chart color (gold)."""
    return GOLD_ACCENT


def get_benchmark_zones() -> dict:
    """
    Return benchmark zone colors for chart backgrounds.

    Returns:
        dict: Zone colors with transparency
    """
    return ZONE_COLORS


def add_pb_marker(fig, value: float, label: str = "PB"):
    """
    Add a Personal Best marker line to a figure.

    Args:
        fig: Plotly figure object
        value: PB value (y-axis position)
        label: Label text (default "PB")

    Returns:
        fig: Figure with PB marker added
    """
    fig.add_hline(
        y=value,
        line_dash="dash",
        line_color=GOLD_ACCENT,
        line_width=2,
        annotation_text=label,
        annotation_position="right",
        annotation_font_color=GOLD_ACCENT,
        annotation_font_size=11,
    )
    return fig


def add_benchmark_zone(fig, y_min: float, y_max: float, zone_type: str = 'good', label: str = None):
    """
    Add a benchmark zone (horizontal band) to a figure.

    Args:
        fig: Plotly figure object
        y_min: Zone lower bound
        y_max: Zone upper bound
        zone_type: Type of zone ('excellent', 'good', 'average')
        label: Optional zone label

    Returns:
        fig: Figure with zone added
    """
    color = ZONE_COLORS.get(zone_type, ZONE_COLORS['average'])

    fig.add_hrect(
        y0=y_min,
        y1=y_max,
        fillcolor=color,
        line_width=0,
        annotation_text=label if label else None,
        annotation_position="right",
        annotation_font_size=10,
        annotation_font_color=TEXT_SECONDARY,
    )
    return fig


def style_bar_chart(fig, color: str = None, show_values: bool = False):
    """
    Apply consistent styling to bar charts.

    Args:
        fig: Plotly figure object
        color: Bar color (default: teal primary)
        show_values: Whether to show values on bars

    Returns:
        fig: Styled figure
    """
    bar_color = color or TEAL_PRIMARY

    fig.update_traces(
        marker_color=bar_color,
        marker_line_width=0,
        textposition='outside' if show_values else 'none',
        textfont={'size': 10, 'color': TEXT_PRIMARY},
    )

    return apply_team_saudi_theme(fig)


def style_scatter_chart(fig, colors: list = None):
    """
    Apply consistent styling to scatter charts.

    Args:
        fig: Plotly figure object
        colors: List of colors for traces

    Returns:
        fig: Styled figure
    """
    trace_colors = colors or [TEAL_PRIMARY]

    for i, trace in enumerate(fig.data):
        color = trace_colors[i % len(trace_colors)]
        fig.update_traces(
            marker=dict(
                color=color,
                size=8,
                line=dict(width=1, color='white'),
            ),
            selector=dict(name=trace.name) if trace.name else None,
        )

    return apply_team_saudi_theme(fig)


def style_line_chart(fig, colors: list = None, line_width: int = 2):
    """
    Apply consistent styling to line charts.

    Args:
        fig: Plotly figure object
        colors: List of colors for lines
        line_width: Line width (default 2)

    Returns:
        fig: Styled figure
    """
    trace_colors = colors or CHART_COLORS

    for i, trace in enumerate(fig.data):
        color = trace_colors[i % len(trace_colors)]
        fig.update_traces(
            line=dict(color=color, width=line_width),
            selector=dict(name=trace.name) if trace.name else None,
        )

    return apply_team_saudi_theme(fig)


# Color constants for direct access
PRIMARY = TEAL_PRIMARY
SECONDARY = TEAL_LIGHT
ACCENT = GOLD_ACCENT
NEUTRAL = GRAY_BLUE
DANGER = '#dc3545'
INFO = '#0077B6'
