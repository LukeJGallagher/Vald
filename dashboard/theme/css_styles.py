"""
Main CSS Stylesheet for VALD Performance Dashboard
Team Saudi Professional Theme - inspired by tayb.sa
"""

from .colors import (
    TEAL_PRIMARY, TEAL_DARK, TEAL_LIGHT, GOLD_ACCENT, GRAY_BLUE,
    BACKGROUND, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, BORDER
)


def get_main_css():
    """Return the main CSS stylesheet as a string."""
    return f"""
    <style>
    /* ============================================
       1. CSS CUSTOM PROPERTIES (Variables)
       ============================================ */
    :root {{
        --teal-primary: {TEAL_PRIMARY};
        --teal-dark: {TEAL_DARK};
        --teal-light: {TEAL_LIGHT};
        --gold-accent: {GOLD_ACCENT};
        --gray-blue: {GRAY_BLUE};

        --bg-primary: {BACKGROUND};
        --bg-surface: {SURFACE};
        --text-primary: {TEXT_PRIMARY};
        --text-secondary: {TEXT_SECONDARY};
        --border: {BORDER};

        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
        --shadow-card: 0 4px 12px rgba(0, 0, 0, 0.08);
        --shadow-lg: 0 10px 25px rgba(0, 0, 0, 0.12);
        --shadow-hover: 0 8px 25px rgba(0, 113, 103, 0.15);

        --radius-sm: 4px;
        --radius-md: 8px;
        --radius-lg: 12px;
        --radius-xl: 16px;

        --transition-fast: 0.2s ease;
        --transition-medium: 0.3s ease;
    }}

    /* ============================================
       2. BASE STYLES & FONTS
       ============================================ */
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=Noto+Kufi+Arabic:wght@500;600&display=swap');

    .stApp {{
        font-family: 'Source Sans 3', 'Inter', sans-serif;
        background: var(--bg-primary);
    }}

    /* ============================================
       3. LAYOUT CONTAINERS
       ============================================ */
    .main .block-container {{
        padding: 1.5rem 2rem;
        max-width: 1400px;
    }}

    /* ============================================
       4. HEADER COMPONENTS
       ============================================ */
    .ts-header {{
        background: linear-gradient(135deg, var(--teal-primary) 0%, var(--teal-dark) 100%);
        padding: 1.75rem 2rem;
        border-radius: var(--radius-lg);
        margin-bottom: 1.5rem;
        border-top: 4px solid var(--gold-accent);
        position: relative;
        box-shadow: var(--shadow-card);
    }}

    .ts-header h1 {{
        color: white;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.85rem;
        margin: 0;
        letter-spacing: -0.5px;
    }}

    .ts-header p {{
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
    }}

    /* ============================================
       5. CARD COMPONENTS
       ============================================ */
    .ts-card {{
        background: var(--bg-surface);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-card);
        transition: transform var(--transition-fast), box-shadow var(--transition-medium);
        border-left: 4px solid var(--teal-primary);
    }}

    .ts-card:hover {{
        transform: translateY(-3px);
        box-shadow: var(--shadow-hover);
    }}

    .ts-metric-card {{
        background: var(--bg-surface);
        border-radius: var(--radius-lg);
        padding: 1.25rem;
        text-align: center;
        box-shadow: var(--shadow-card);
        transition: all var(--transition-medium);
        border-bottom: 3px solid transparent;
    }}

    .ts-metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: var(--shadow-hover);
        border-bottom: 3px solid var(--gold-accent);
    }}

    .ts-metric-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--teal-primary);
        font-family: 'Space Grotesk', sans-serif;
        line-height: 1.2;
    }}

    .ts-metric-label {{
        font-size: 0.8rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }}

    .ts-metric-delta {{
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }}

    .ts-metric-delta.positive {{
        color: var(--teal-primary);
    }}

    .ts-metric-delta.negative {{
        color: #dc3545;
    }}

    /* ============================================
       6. SIDEBAR STYLES
       ============================================ */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, var(--teal-primary) 0%, var(--teal-dark) 100%);
        border-right: 3px solid var(--gold-accent);
    }}

    [data-testid="stSidebar"] * {{
        color: white !important;
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: white !important;
        border-bottom: 2px solid var(--gold-accent);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }}

    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stMultiSelect > div > div {{
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.25);
        border-radius: var(--radius-md);
    }}

    [data-testid="stSidebar"] .stSelectbox > div > div:hover,
    [data-testid="stSidebar"] .stMultiSelect > div > div:hover {{
        background: rgba(255, 255, 255, 0.18);
        border-color: var(--gold-accent);
    }}

    [data-testid="stSidebar"] hr {{
        border-color: rgba(255, 255, 255, 0.2);
    }}

    /* ============================================
       7. TAB NAVIGATION
       ============================================ */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: #f5f5f5;
        padding: 10px 12px;
        border-radius: var(--radius-lg);
        border-bottom: 3px solid var(--gold-accent);
        overflow-x: auto;
        flex-wrap: nowrap;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 42px;
        padding: 0 18px;
        background: white;
        border-radius: var(--radius-md);
        font-weight: 600;
        font-size: 0.9rem;
        font-family: 'Source Sans 3', sans-serif;
        transition: all var(--transition-fast);
        border: 2px solid #e0e0e0;
        white-space: nowrap;
        color: var(--text-secondary);
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background: #fafafa;
        border-color: var(--gold-accent);
        color: var(--teal-primary);
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, var(--teal-primary) 0%, var(--teal-dark) 100%) !important;
        color: white !important;
        border-color: var(--gold-accent) !important;
        box-shadow: 0 4px 12px rgba(0, 113, 103, 0.25);
    }}

    /* Tab indicator line */
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: transparent !important;
    }}

    /* ============================================
       8. BUTTONS
       ============================================ */
    .stButton > button {{
        background: var(--teal-primary);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.55rem 1.25rem;
        font-weight: 600;
        font-size: 0.9rem;
        font-family: 'Source Sans 3', sans-serif;
        transition: all var(--transition-fast);
        box-shadow: 0 2px 8px rgba(0, 113, 103, 0.2);
    }}

    .stButton > button:hover {{
        background: var(--gold-accent);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(160, 142, 102, 0.35);
    }}

    .stButton > button:active {{
        transform: translateY(0);
    }}

    .stDownloadButton > button {{
        background: var(--gold-accent);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        font-weight: 600;
    }}

    .stDownloadButton > button:hover {{
        background: var(--teal-primary);
    }}

    /* ============================================
       9. STATUS BADGES
       ============================================ */
    .ts-badge {{
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }}

    .ts-badge-success {{
        background: var(--teal-primary);
        color: white;
    }}

    .ts-badge-warning {{
        background: var(--gold-accent);
        color: white;
    }}

    .ts-badge-danger {{
        background: #dc3545;
        color: white;
    }}

    .ts-badge-info {{
        background: #0077B6;
        color: white;
    }}

    .ts-badge-neutral {{
        background: var(--gray-blue);
        color: white;
    }}

    /* ============================================
       10. DATA TABLES
       ============================================ */
    .stDataFrame {{
        border-radius: var(--radius-md);
        overflow: hidden;
        box-shadow: var(--shadow-card);
    }}

    .stDataFrame thead tr th {{
        background: var(--teal-primary) !important;
        color: white !important;
        font-weight: 600;
        font-size: 0.85rem;
    }}

    .stDataFrame tbody tr:hover {{
        background: rgba(0, 113, 103, 0.05) !important;
    }}

    /* ============================================
       11. FORMS & INPUTS
       ============================================ */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {{
        border-radius: var(--radius-md);
        border: 2px solid #e0e0e0;
        transition: all var(--transition-fast);
        font-family: 'Source Sans 3', sans-serif;
    }}

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: var(--teal-primary);
        box-shadow: 0 0 0 3px rgba(0, 113, 103, 0.1);
    }}

    .stSelectbox > div > div {{
        border-radius: var(--radius-md);
    }}

    /* ============================================
       12. EXPANDERS
       ============================================ */
    .streamlit-expanderHeader {{
        background: var(--bg-surface);
        border-radius: var(--radius-md);
        font-weight: 600;
        color: var(--text-primary);
        border: 1px solid var(--border);
    }}

    .streamlit-expanderHeader:hover {{
        border-color: var(--teal-primary);
        color: var(--teal-primary);
    }}

    /* ============================================
       13. PLOTLY CHART CONTAINERS
       ============================================ */
    .js-plotly-plot {{
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-card);
        overflow: hidden;
    }}

    /* ============================================
       14. SCROLLBARS
       ============================================ */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: #f1f1f1;
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb {{
        background: var(--teal-primary);
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: var(--gold-accent);
    }}

    /* ============================================
       15. METRICS (Streamlit native)
       ============================================ */
    [data-testid="metric-container"] {{
        background: var(--bg-surface);
        border-radius: var(--radius-lg);
        padding: 1rem;
        box-shadow: var(--shadow-card);
        border-left: 4px solid var(--teal-primary);
    }}

    [data-testid="metric-container"] label {{
        color: var(--text-secondary) !important;
        font-weight: 500;
    }}

    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: var(--teal-primary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }}

    /* ============================================
       16. ALERTS & INFO BOXES
       ============================================ */
    .stAlert {{
        border-radius: var(--radius-md);
        border-left: 4px solid var(--teal-primary);
    }}

    /* ============================================
       17. ANIMATIONS
       ============================================ */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes slideIn {{
        from {{ opacity: 0; transform: translateX(-10px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}

    .ts-animate-in {{
        animation: fadeIn 0.4s ease-out;
    }}

    .ts-animate-slide {{
        animation: slideIn 0.3s ease-out;
    }}

    /* ============================================
       18. RESPONSIVE ADJUSTMENTS
       ============================================ */
    @media (max-width: 768px) {{
        .ts-header {{
            padding: 1.25rem;
        }}

        .ts-header h1 {{
            font-size: 1.4rem;
        }}

        .ts-metric-value {{
            font-size: 1.5rem;
        }}

        .main .block-container {{
            padding: 1rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            padding: 0 12px;
            font-size: 0.8rem;
        }}
    }}

    /* ============================================
       19. SECTION HEADERS
       ============================================ */
    .ts-section-header {{
        color: var(--text-primary);
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border);
    }}

    .ts-section-header::before {{
        content: '';
        display: inline-block;
        width: 4px;
        height: 1em;
        background: var(--teal-primary);
        margin-right: 0.5rem;
        vertical-align: middle;
    }}

    /* ============================================
       20. UTILITY CLASSES
       ============================================ */
    .ts-text-muted {{
        color: var(--text-secondary);
        font-size: 0.875rem;
    }}

    .ts-text-small {{
        font-size: 0.8rem;
    }}

    .ts-gold-accent {{
        color: var(--gold-accent);
    }}

    .ts-teal-accent {{
        color: var(--teal-primary);
    }}

    .ts-divider {{
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.5rem 0;
    }}
    </style>
    """


def get_sidebar_css():
    """Return additional sidebar-specific CSS."""
    return """
    <style>
    /* Additional sidebar enhancements */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.9rem;
    }

    [data-testid="stSidebar"] .stRadio > label {
        background: rgba(255, 255, 255, 0.08);
        padding: 0.5rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.25rem;
        transition: all 0.2s ease;
    }

    [data-testid="stSidebar"] .stRadio > label:hover {
        background: rgba(255, 255, 255, 0.15);
    }
    </style>
    """
