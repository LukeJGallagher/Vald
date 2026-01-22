"""
Team Saudi Theme Module for VALD Performance Dashboard

This module provides centralized styling including:
- Color palette and design tokens
- CSS stylesheet
- Reusable UI components
- Plotly chart theming

Usage:
    from theme import get_main_css, render_header, metric_card, apply_team_saudi_theme
    from theme import TEAL_PRIMARY, GOLD_ACCENT, COLORS
"""

# Color constants and palettes
from .colors import (
    # Primary colors
    TEAL_PRIMARY,
    TEAL_DARK,
    TEAL_LIGHT,
    GOLD_ACCENT,
    GRAY_BLUE,
    # UI colors
    BACKGROUND,
    SURFACE,
    SURFACE_HOVER,
    BORDER,
    BORDER_FOCUS,
    # Text colors
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    TEXT_INVERSE,
    # Status colors
    SUCCESS,
    WARNING,
    DANGER,
    INFO,
    # Collections
    COLORS,
    GRADIENTS,
    SESSION_COLORS,
    ZONE_COLORS,
    CHART_COLORS,
)

# CSS stylesheet functions
from .css_styles import (
    get_main_css,
    get_sidebar_css,
)

# UI component functions
from .components import (
    render_header,
    metric_card,
    metric_card_colored,
    status_badge,
    section_header,
    info_card,
    stat_row,
    render_divider,
    alert_box,
    progress_indicator,
    render_athlete_card,
)

# Plotly theming functions
from .plotly_theme import (
    apply_team_saudi_theme,
    apply_minimal_theme,
    get_plotly_layout,
    get_axis_style,
    get_line_colors,
    get_primary_color,
    get_accent_color,
    get_benchmark_zones,
    add_pb_marker,
    add_benchmark_zone,
    style_bar_chart,
    style_scatter_chart,
    style_line_chart,
    # Direct color access for charts
    PRIMARY,
    SECONDARY,
    ACCENT,
    NEUTRAL,
)

# Sport icon components
from .sport_icons import (
    render_sport_icon_grid,
    get_selected_sport,
    filter_by_selected_sport,
    get_sport_icon,
    get_icon_image_base64,
    render_selected_sport_header,
    SPORT_ICONS,
    ICONS_DIR,
)

__version__ = '1.0.0'
__all__ = [
    # Colors
    'TEAL_PRIMARY', 'TEAL_DARK', 'TEAL_LIGHT', 'GOLD_ACCENT', 'GRAY_BLUE',
    'BACKGROUND', 'SURFACE', 'BORDER', 'TEXT_PRIMARY', 'TEXT_SECONDARY',
    'SUCCESS', 'WARNING', 'DANGER', 'INFO',
    'COLORS', 'GRADIENTS', 'SESSION_COLORS', 'ZONE_COLORS', 'CHART_COLORS',
    # CSS
    'get_main_css', 'get_sidebar_css',
    # Components
    'render_header', 'metric_card', 'metric_card_colored', 'status_badge',
    'section_header', 'info_card', 'stat_row', 'render_divider',
    'alert_box', 'progress_indicator', 'render_athlete_card',
    # Plotly
    'apply_team_saudi_theme', 'apply_minimal_theme', 'get_plotly_layout',
    'get_line_colors', 'get_primary_color', 'get_accent_color',
    'get_benchmark_zones', 'add_pb_marker', 'add_benchmark_zone',
    'style_bar_chart', 'style_scatter_chart', 'style_line_chart',
    'PRIMARY', 'SECONDARY', 'ACCENT', 'NEUTRAL',
    # Sport Icons
    'render_sport_icon_grid', 'get_selected_sport', 'filter_by_selected_sport',
    'get_sport_icon', 'render_selected_sport_header', 'SPORT_ICONS',
]
