"""
Reusable Styled Components for VALD Performance Dashboard
Team Saudi Professional Theme
"""

import streamlit as st
from .colors import (
    TEAL_PRIMARY, TEAL_DARK, TEAL_LIGHT, GOLD_ACCENT, GRAY_BLUE,
    TEXT_PRIMARY, TEXT_SECONDARY, SURFACE, COLORS
)


def render_header(title: str, subtitle: str = None):
    """
    Render Team Saudi branded header.

    Args:
        title: Main header title
        subtitle: Optional subtitle text
    """
    subtitle_html = f'<p>{subtitle}</p>' if subtitle else ''

    st.markdown(f"""
    <div class="ts-header ts-animate-in">
        <h1>{title}</h1>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, status: str = None, delta: str = None):
    """
    Render a styled metric card.

    Args:
        label: Metric label text
        value: Metric value to display
        status: Status for border color ('excellent', 'good', 'warning', 'danger')
        delta: Optional delta/change text
    """
    border_color = {
        'excellent': TEAL_PRIMARY,
        'good': TEAL_LIGHT,
        'warning': GOLD_ACCENT,
        'danger': '#dc3545',
    }.get(status, TEAL_PRIMARY)

    delta_class = ''
    if delta:
        if delta.startswith('+') or delta.startswith('↑'):
            delta_class = 'positive'
        elif delta.startswith('-') or delta.startswith('↓'):
            delta_class = 'negative'

    delta_html = f'<div class="ts-metric-delta {delta_class}">{delta}</div>' if delta else ''

    st.markdown(f"""
    <div class="ts-metric-card" style="border-left: 4px solid {border_color};">
        <div class="ts-metric-label">{label}</div>
        <div class="ts-metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def metric_card_colored(label: str, value: str, status: str = 'good'):
    """
    Render a metric card with colored background based on status.

    Args:
        label: Metric label text
        value: Metric value to display
        status: Status for background color ('excellent', 'good', 'warning', 'danger', 'neutral')
    """
    colors = {
        'excellent': TEAL_PRIMARY,
        'good': TEAL_LIGHT,
        'warning': GOLD_ACCENT,
        'danger': '#dc3545',
        'neutral': GRAY_BLUE,
    }
    bg_color = colors.get(status, GRAY_BLUE)

    st.markdown(f"""
    <div style="background: {bg_color}; padding: 1.25rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <p style="color: rgba(255,255,255,0.85); margin: 0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px;">{label}</p>
        <p style="color: white; margin: 0.25rem 0 0 0; font-size: 1.75rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">{value}</p>
    </div>
    """, unsafe_allow_html=True)


def status_badge(text: str, status: str = 'success') -> str:
    """
    Return HTML for a status badge.

    Args:
        text: Badge text
        status: Status type ('success', 'warning', 'danger', 'info', 'neutral')

    Returns:
        HTML string for the badge
    """
    return f'<span class="ts-badge ts-badge-{status}">{text}</span>'


def section_header(title: str, icon: str = None):
    """
    Render a section header with optional icon.

    Args:
        title: Section title text
        icon: Optional emoji/icon to prepend
    """
    icon_html = f"{icon} " if icon else ""
    st.markdown(f"""
    <div class="ts-section-header">{icon_html}{title}</div>
    """, unsafe_allow_html=True)


def info_card(title: str, content: str, accent_color: str = None):
    """
    Render an info card with accent border.

    Args:
        title: Card title
        content: Card content text
        accent_color: Optional custom accent color (hex)
    """
    color = accent_color or TEAL_PRIMARY
    st.markdown(f"""
    <div class="ts-card" style="border-left-color: {color};">
        <h4 style="margin: 0 0 0.5rem 0; color: {TEXT_PRIMARY}; font-weight: 600;">{title}</h4>
        <p style="margin: 0; color: {TEXT_SECONDARY}; line-height: 1.5;">{content}</p>
    </div>
    """, unsafe_allow_html=True)


def stat_row(stats: list):
    """
    Render a row of small stat items.

    Args:
        stats: List of tuples [(label, value), ...]
    """
    cols = st.columns(len(stats))
    for i, (label, value) in enumerate(stats):
        with cols[i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 0.75rem;">
                <div style="font-size: 0.75rem; color: {TEXT_SECONDARY}; text-transform: uppercase;">{label}</div>
                <div style="font-size: 1.25rem; font-weight: 600; color: {TEAL_PRIMARY};">{value}</div>
            </div>
            """, unsafe_allow_html=True)


def render_divider():
    """Render a styled horizontal divider."""
    st.markdown('<hr class="ts-divider">', unsafe_allow_html=True)


def alert_box(message: str, alert_type: str = 'info'):
    """
    Render a styled alert box.

    Args:
        message: Alert message text
        alert_type: Type of alert ('info', 'success', 'warning', 'danger')
    """
    colors = {
        'info': ('#0077B6', 'rgba(0, 119, 182, 0.1)'),
        'success': (TEAL_PRIMARY, 'rgba(0, 113, 103, 0.1)'),
        'warning': (GOLD_ACCENT, 'rgba(160, 142, 102, 0.1)'),
        'danger': ('#dc3545', 'rgba(220, 53, 69, 0.1)'),
    }
    border_color, bg_color = colors.get(alert_type, colors['info'])

    icons = {
        'info': 'ℹ️',
        'success': '✓',
        'warning': '⚠️',
        'danger': '⚠',
    }
    icon = icons.get(alert_type, '')

    st.markdown(f"""
    <div style="
        background: {bg_color};
        border-left: 4px solid {border_color};
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    ">
        <span style="font-weight: 600;">{icon}</span> {message}
    </div>
    """, unsafe_allow_html=True)


def progress_indicator(value: float, max_value: float = 100, label: str = None):
    """
    Render a styled progress indicator.

    Args:
        value: Current value
        max_value: Maximum value (default 100)
        label: Optional label text
    """
    percentage = min(100, (value / max_value) * 100)

    # Color based on percentage
    if percentage >= 80:
        color = TEAL_PRIMARY
    elif percentage >= 60:
        color = TEAL_LIGHT
    elif percentage >= 40:
        color = GOLD_ACCENT
    else:
        color = '#dc3545'

    label_html = f'<span style="font-size: 0.85rem; color: {TEXT_SECONDARY};">{label}</span>' if label else ''

    st.markdown(f"""
    <div style="margin: 0.5rem 0;">
        {label_html}
        <div style="background: #e9ecef; border-radius: 10px; height: 10px; overflow: hidden; margin-top: 0.25rem;">
            <div style="background: {color}; width: {percentage}%; height: 100%; border-radius: 10px; transition: width 0.3s ease;"></div>
        </div>
        <span style="font-size: 0.75rem; color: {TEXT_SECONDARY};">{value:.1f} / {max_value:.1f}</span>
    </div>
    """, unsafe_allow_html=True)


def render_athlete_card(name: str, sport: str, stats: dict = None):
    """
    Render an athlete info card.

    Args:
        name: Athlete name
        sport: Sport/discipline
        stats: Optional dict of stat name -> value
    """
    stats_html = ''
    if stats:
        stats_items = ''.join([
            f'<div style="text-align: center;"><div style="font-size: 0.7rem; color: {TEXT_SECONDARY}; text-transform: uppercase;">{k}</div><div style="font-weight: 600; color: {TEAL_PRIMARY};">{v}</div></div>'
            for k, v in stats.items()
        ])
        stats_html = f'<div style="display: flex; justify-content: space-around; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e9ecef;">{stats_items}</div>'

    st.markdown(f"""
    <div class="ts-card">
        <h4 style="margin: 0; color: {TEXT_PRIMARY}; font-weight: 600;">{name}</h4>
        <p style="margin: 0.25rem 0 0 0; color: {TEXT_SECONDARY}; font-size: 0.9rem;">{sport}</p>
        {stats_html}
    </div>
    """, unsafe_allow_html=True)
