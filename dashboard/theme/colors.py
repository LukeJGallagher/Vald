"""
Team Saudi Color Palette and Design Tokens
Official branding colors matching Team Saudi banner
"""

# Primary Brand Colors (Team Saudi - matching banner green)
TEAL_PRIMARY = '#255035'      # Saudi Green - exact color from banners
TEAL_DARK = '#1C3D28'         # Darker green for hover states
TEAL_LIGHT = '#2E6040'        # Lighter green for secondary
GOLD_ACCENT = '#a08e66'       # Gold - highlights, PB markers, secondary accents
GRAY_BLUE = '#78909C'         # Neutral, needs improvement

# UI Colors
BACKGROUND = '#f8f9fa'        # App background
SURFACE = '#ffffff'           # Card/container background
SURFACE_HOVER = '#f5f5f5'     # Hover states
BORDER = '#e9ecef'            # Subtle borders
BORDER_FOCUS = TEAL_PRIMARY   # Focus states

# Text Colors
TEXT_PRIMARY = '#1a1a1a'      # Headings
TEXT_SECONDARY = '#4b5563'    # Body text
TEXT_MUTED = '#6c757d'        # Captions, hints
TEXT_INVERSE = '#ffffff'      # On dark backgrounds

# Status Colors
SUCCESS = TEAL_PRIMARY        # Good/excellent (uses teal)
WARNING = '#FFB800'           # Caution (gold)
DANGER = '#dc3545'            # Alert/risk
INFO = '#0077B6'              # Informational

# Session type colors (for training data)
SESSION_COLORS = {
    'Training': TEAL_PRIMARY,     # Teal
    'Competition': '#FFB800',     # Gold/Yellow
    'Testing': '#0077B6',         # Blue
    'Warm-up': '#6c757d'          # Gray
}

# Benchmark zone colors (with transparency)
ZONE_COLORS = {
    'excellent': 'rgba(29, 77, 59, 0.20)',       # Saudi Green (excellent)
    'good': 'rgba(42, 106, 80, 0.15)',           # Light green (good)
    'average': 'rgba(120, 144, 156, 0.15)',      # Gray-blue (needs work)
}

# Gradients
GRADIENTS = {
    'header': f'linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%)',
    'sidebar': f'linear-gradient(180deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%)',
    'card_hover': 'linear-gradient(145deg, #ffffff 0%, #f9f9f9 100%)',
}

# Chart color sequence for multi-series
CHART_COLORS = [
    TEAL_PRIMARY,    # #1D4D3B
    GOLD_ACCENT,     # #a08e66
    '#0077B6',       # Blue
    '#FF6B6B',       # Coral
    '#9C27B0',       # Purple
    '#FF9800',       # Orange
]

# Color dictionary for easy access
COLORS = {
    'teal_primary': TEAL_PRIMARY,
    'teal_dark': TEAL_DARK,
    'teal_light': TEAL_LIGHT,
    'gold_accent': GOLD_ACCENT,
    'gray_blue': GRAY_BLUE,
    'background': BACKGROUND,
    'surface': SURFACE,
    'surface_hover': SURFACE_HOVER,
    'border': BORDER,
    'border_focus': BORDER_FOCUS,
    'text_primary': TEXT_PRIMARY,
    'text_secondary': TEXT_SECONDARY,
    'text_muted': TEXT_MUTED,
    'text_inverse': TEXT_INVERSE,
    'success': SUCCESS,
    'warning': WARNING,
    'danger': DANGER,
    'info': INFO,
}
