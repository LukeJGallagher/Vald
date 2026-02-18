"""
Sport Icon Buttons Component for VALD Performance Dashboard
Visual sport selection using Team Saudi official pictograms
"""

import streamlit as st
import pandas as pd
import os
import base64
from .colors import TEAL_PRIMARY, TEAL_DARK, GOLD_ACCENT, TEXT_SECONDARY

# Path to sport icon images
ICONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'sport_icons')

# Sport icons mapping - official Team Saudi pictograms
# Image files extracted from Team Saudi branding materials
SPORT_ICONS = {
    'Athletics': {
        'icon': '🏃',
        'image': 'athletics.png',
        'keywords': ['Athletics', 'Track', 'Field', 'Throws', 'Sprint', 'Jump'],
    },
    'Cycling': {
        'icon': '🚴',
        'image': 'cycling.png',
        'keywords': ['Cycling', 'Bike', 'BMX'],
    },
    'Taekwondo': {
        'icon': '🥋',
        'image': 'taekwondo.png',
        'keywords': ['Taekwondo', 'TKD'],
    },
    'Equestrian': {
        'icon': '🏇',
        'image': 'equestrian.png',
        'keywords': ['Equestrian', 'Horse', 'Dressage', 'Jumping'],
    },
    'Swimming': {
        'icon': '🏊',
        'image': 'swimming.png',
        'keywords': ['Swimming', 'Swim', 'Aquatics'],
    },
    'Fencing': {
        'icon': '🤺',
        'image': 'fencing.png',
        'keywords': ['Fencing', 'Epee', 'Foil', 'Sabre'],
    },
    'Karate': {
        'icon': '🥋',
        'image': 'karate.png',
        'keywords': ['Karate', 'Kata', 'Kumite'],
    },
    'Shooting': {
        'icon': '🎯',
        'image': 'shooting.png',
        'keywords': ['Shooting', 'Pistol', 'Rifle', '10m', '50m'],
    },
    'Gymnastics': {
        'icon': '🤸',
        'image': 'gymnastics.png',
        'keywords': ['Gymnastics', 'Artistic', 'Rhythmic'],
    },
    'Weightlifting': {
        'icon': '🏋️',
        'image': 'weightlifting.png',
        'keywords': ['Weightlifting', 'Olympic Lifting', 'Snatch', 'Clean'],
    },
    'Archery': {
        'icon': '🏹',
        'image': 'archery.png',
        'keywords': ['Archery', 'Bow'],
    },
    'Judo': {
        'icon': '🥋',
        'image': 'judo.png',
        'keywords': ['Judo'],
    },
    'Jiu Jitsu': {
        'icon': '🥋',
        'image': 'jiujitsu.png',
        'keywords': ['Jiu Jitsu', 'BJJ', 'Jiu-Jitsu'],
    },
    'Wrestling': {
        'icon': '🤼',
        'image': 'wrestling.png',
        'keywords': ['Wrestling', 'Greco', 'Freestyle'],
    },
    'Football': {
        'icon': '⚽',
        'image': None,
        'keywords': ['Football', 'Soccer'],
    },
    'Basketball': {
        'icon': '🏀',
        'image': None,
        'keywords': ['Basketball'],
    },
    'Volleyball': {
        'icon': '🏐',
        'image': None,
        'keywords': ['Volleyball', 'Beach Volleyball'],
    },
    'Tennis': {
        'icon': '🎾',
        'image': None,
        'keywords': ['Tennis'],
    },
    'Table Tennis': {
        'icon': '🏓',
        'image': None,
        'keywords': ['Table Tennis', 'Ping Pong'],
    },
    'Golf': {
        'icon': '⛳',
        'image': None,
        'keywords': ['Golf'],
    },
}


def get_icon_image_base64(sport_name: str) -> str:
    """Get base64 encoded image for a sport icon."""
    sport_lower = sport_name.lower()

    for sport, data in SPORT_ICONS.items():
        for keyword in data['keywords']:
            if keyword.lower() in sport_lower:
                if data.get('image'):
                    img_path = os.path.join(ICONS_DIR, data['image'])
                    if os.path.exists(img_path):
                        with open(img_path, 'rb') as f:
                            return base64.b64encode(f.read()).decode()
                break
    return None


def get_sport_icon(sport_name: str) -> str:
    """Get icon for a sport name, matching by keywords."""
    sport_lower = sport_name.lower()

    for sport, data in SPORT_ICONS.items():
        for keyword in data['keywords']:
            if keyword.lower() in sport_lower:
                return data['icon']

    # Default icon for unknown sports
    return '🏅'


def match_sport_to_category(sport_name: str) -> str:
    """Match a detailed sport name to its main category."""
    sport_lower = sport_name.lower()

    for category, data in SPORT_ICONS.items():
        for keyword in data['keywords']:
            if keyword.lower() in sport_lower:
                return category

    return sport_name  # Return original if no match


def render_sport_icon_grid(available_sports: list, selected_sport: str = None, key_prefix: str = "sport"):
    """
    Render a grid of sport icon buttons in the sidebar.

    Args:
        available_sports: List of sport names from the data
        selected_sport: Currently selected sport (or None for all)
        key_prefix: Prefix for button keys

    Returns:
        str: Selected sport name or None
    """
    # Initialize session state if needed
    if 'selected_sport_icon' not in st.session_state:
        st.session_state.selected_sport_icon = None

    # Group sports by category
    sport_categories = {}
    for sport in available_sports:
        category = match_sport_to_category(sport)
        if category not in sport_categories:
            sport_categories[category] = []
        if sport not in sport_categories[category]:
            sport_categories[category].append(sport)

    # Custom CSS for sport buttons
    st.markdown(f"""
    <style>
    .sport-icon-btn {{
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 70px;
        height: 70px;
        margin: 4px;
        padding: 8px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid transparent;
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
    }}

    .sport-icon-btn:hover {{
        background: rgba(255, 255, 255, 0.2);
        border-color: {GOLD_ACCENT};
        transform: translateY(-2px);
    }}

    .sport-icon-btn.selected {{
        background: {GOLD_ACCENT};
        border-color: white;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }}

    .sport-icon-btn .icon {{
        font-size: 1.5rem;
        margin-bottom: 2px;
    }}

    .sport-icon-btn .label {{
        font-size: 0.6rem;
        color: white;
        text-align: center;
        line-height: 1.1;
        max-width: 60px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}

    .sport-grid {{
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 4px;
        padding: 8px 0;
    }}

    .sport-all-btn {{
        width: 100%;
        padding: 8px 16px;
        margin: 8px 0;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid transparent;
        color: white;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: center;
    }}

    .sport-all-btn:hover {{
        background: rgba(255, 255, 255, 0.2);
        border-color: {GOLD_ACCENT};
    }}

    .sport-all-btn.selected {{
        background: {TEAL_PRIMARY};
        border-color: {GOLD_ACCENT};
    }}
    </style>
    """, unsafe_allow_html=True)

    # "All Sports" button
    all_selected = st.session_state.selected_sport_icon is None
    if st.button(
        "🏅 All Sports" if all_selected else "📋 All Sports",
        key=f"{key_prefix}_all",
        width='stretch',
        type="primary" if all_selected else "secondary"
    ):
        st.session_state.selected_sport_icon = None
        st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Sport category buttons using Streamlit columns
    categories = sorted(sport_categories.keys())

    # Create rows of 3 buttons each
    for i in range(0, len(categories), 3):
        row_categories = categories[i:i+3]
        cols = st.columns(len(row_categories))

        for j, category in enumerate(row_categories):
            with cols[j]:
                is_selected = st.session_state.selected_sport_icon == category

                # Try to get image, fall back to emoji
                img_b64 = get_icon_image_base64(category)
                icon = get_sport_icon(category)

                # Short label for button
                short_label = category[:8] + '..' if len(category) > 10 else category

                # If we have an image, show it with clickable container
                if img_b64:
                    border_color = GOLD_ACCENT if is_selected else 'transparent'
                    bg_color = GOLD_ACCENT if is_selected else 'rgba(255,255,255,0.1)'

                    # Render image button with HTML
                    st.markdown(f"""
                    <div style="
                        background: {bg_color};
                        border: 2px solid {border_color};
                        border-radius: 8px;
                        padding: 4px;
                        text-align: center;
                        margin-bottom: 4px;
                    ">
                        <img src="data:image/png;base64,{img_b64}" style="width: 48px; height: 48px; border-radius: 4px;">
                        <div style="color: white; font-size: 0.65rem; margin-top: 2px;">{short_label}</div>
                    </div>
                    """, unsafe_allow_html=True)

                btn_type = "primary" if is_selected else "secondary"
                if st.button(
                    f"{icon}\n{short_label}",
                    key=f"{key_prefix}_{category}",
                    type=btn_type,
                    width='stretch'
                ):
                    st.session_state.selected_sport_icon = category
                    st.rerun()

    return st.session_state.selected_sport_icon


def get_selected_sport() -> str:
    """Get the currently selected sport from session state."""
    return st.session_state.get('selected_sport_icon', None)


def filter_by_selected_sport(df, sport_column: str = 'athlete_sport'):
    """
    Filter a dataframe by the currently selected sport.

    Args:
        df: DataFrame to filter
        sport_column: Column name containing sport info

    Returns:
        Filtered DataFrame
    """
    selected = get_selected_sport()

    if selected is None or df.empty or sport_column not in df.columns:
        return df

    # Get the sport data for the selected category
    if selected in SPORT_ICONS:
        keywords = SPORT_ICONS[selected]['keywords']
        # Match any keyword
        mask = df[sport_column].str.lower().apply(
            lambda x: any(kw.lower() in str(x).lower() for kw in keywords) if pd.notna(x) else False
        )
        return df[mask]
    else:
        # Direct match
        return df[df[sport_column].str.contains(selected, case=False, na=False)]


def render_selected_sport_header():
    """Render a header showing the currently selected sport."""
    selected = get_selected_sport()

    if selected:
        icon = get_sport_icon(selected)
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%);
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        ">
            <span style="font-size: 1.5rem;">{icon}</span>
            <span style="color: white; font-weight: 600; font-size: 1.1rem;">{selected}</span>
            <span style="color: rgba(255,255,255,0.7); font-size: 0.85rem; margin-left: auto;">Filtered View</span>
        </div>
        """, unsafe_allow_html=True)
