"""
WORLD-CLASS VALD PERFORMANCE DASHBOARD
Saudi National Team - Enhanced S&C Analysis

Features:
- 15+ sports with evidence-based benchmarks
- Multi-device support (ForceDecks, ForceFrame, NordBord)
- Risk scoring and readiness assessment
- Advanced asymmetry tracking
- Multi-athlete comparisons
- Sport-specific context and interpretation
- Force trace analysis with phase detection
- Advanced analytics (quadrant, parallel coords, violin plots)
- Reliability analysis (CV%, Typical Error)
- Best-of-day trend tracking
- Selective trace data fetching from API

Author: Performance Analysis Team
Version: 3.0
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import sys
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports (works locally and on Streamlit Cloud)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Import configuration
try:
    from config.sports_config import (
        SPORT_BENCHMARKS,
        get_sport_benchmarks,
        get_percentile_rank,
        get_metric_status,
        get_asymmetry_status,
        get_priority_metrics_for_sport,
        get_sport_context,
        TEST_TYPE_CONFIG,
        METRIC_DEFINITIONS,
        RISK_THRESHOLDS
    )
except ImportError:
    st.error("⚠️ Sports configuration not found. Please ensure config/sports_config.py exists.")
    st.stop()

# Import utilities
try:
    from utils.data_loader import (
        load_vald_data,
        load_all_devices,
        get_latest_test_per_athlete,
        calculate_asymmetry_index,
        get_metrics_from_test_type,
        filter_dataframe,
        get_test_summary_stats,
        push_to_github_repo,
        refresh_and_save_data
    )
    GITHUB_SYNC_AVAILABLE = True
except ImportError:
    GITHUB_SYNC_AVAILABLE = False
    st.warning("⚠️ Data loader utilities not found. Using basic loading...")

# Import advanced visualization modules
try:
    from utils.advanced_viz import (
        create_quadrant_plot,
        create_parallel_coordinates,
        create_violin_plot,
        get_best_of_day_per_athlete,
        create_best_of_day_trend,
        calculate_reliability_metrics,
        create_reliability_plot,
        create_labeled_ranking
    )
    ADVANCED_VIZ_AVAILABLE = True
except ImportError:
    ADVANCED_VIZ_AVAILABLE = False
    st.sidebar.warning("⚠️ Advanced visualization module not loaded")

# Import force trace visualization
try:
    from utils.force_trace_viz import (
        plot_force_trace,
        plot_multi_trial_overlay,
        plot_athlete_comparison,
        calculate_trace_metrics,
        detect_phases
    )
    FORCE_TRACE_AVAILABLE = True
except ImportError:
    FORCE_TRACE_AVAILABLE = False

# Import test-type specific modules (CMJ, IMTP, Throws analysis)
try:
    from utils.test_type_modules import (
        CMJAnalysisModule,
        IsometricSingleLegModule,
        IsometricDoubleLegModule,
        ThrowsTrainingModule,
        display_test_type_module
    )
    TEST_TYPE_MODULES_AVAILABLE = True
except ImportError:
    TEST_TYPE_MODULES_AVAILABLE = False
    st.sidebar.warning("⚠️ Test-type modules not loaded")

# Import advanced analysis (Elite Insights features)
try:
    from utils.advanced_analysis import (
        calculate_asymmetry,
        create_asymmetry_circle_plot,
        calculate_meaningful_change,
        create_meaningful_change_plot,
        calculate_tem_with_ci,
        calculate_percentile_rank,
        create_normative_benchmark_plot,
        calculate_zscore_with_context,
        create_zscore_distribution_plot,
        create_bilateral_force_curve
    )
    ELITE_INSIGHTS_AVAILABLE = True
except ImportError:
    ELITE_INSIGHTS_AVAILABLE = False

    # Fallback data loading
    @st.cache_data
    def load_vald_data(device='forcedecks'):
        file_paths = [
            f'{device}_allsports_with_athletes.csv',
            f'data/{device}_allsports_with_athletes.csv',
            f'data/master_files/{device}_allsports_with_athletes.csv',
        ]

        for file_path in file_paths:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if 'recordedDateUtc' in df.columns:
                    df['recordedDateUtc'] = pd.to_datetime(df['recordedDateUtc'])
                return df

        return pd.DataFrame()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def filter_performance_columns(df):
    """Filter dataframe to show only performance metrics relevant to coaches."""
    # Columns to EXCLUDE
    exclude_patterns = [
        'offset', 'Offset', 'UTC', 'utc', 'Id', 'id', 'guid', 'Guid',
        'tenantId', 'testId', 'trialId', 'athleteId', 'groupId',
        'timezone', 'sessionId', 'deviceId', 'softwareVersion',
        'notes', 'comment', 'flag', 'valid', 'created', 'updated', 'modified'
    ]

    # Keep essential ID columns
    keep_columns = ['Name', 'athlete_sport', 'testType']

    # Keep all performance metric columns
    performance_keywords = [
        'Peak', 'Jump', 'Force', 'Power', 'Height', 'Depth', 'Time', 'RFD',
        'Impulse', 'Load', 'Work', 'Velocity', 'Speed', 'Distance',
        'Asymmetry', 'Balance', 'Stability', 'RSI', 'Stiffness',
        'Concentric', 'Eccentric', 'Isometric', 'Flight', 'Contact',
        'BM', 'Body', 'Mass', 'Weight', 'Left', 'Right', 'Average', 'Mean'
    ]

    filtered_cols = []
    for col in df.columns:
        # Always keep essential ID columns
        if col in keep_columns:
            filtered_cols.append(col)
        # Exclude technical columns
        elif any(pattern in col for pattern in exclude_patterns):
            continue
        # Keep performance metrics
        elif any(keyword in col for keyword in performance_keywords):
            filtered_cols.append(col)
        # Keep recordedDateUtc but rename it
        elif col == 'recordedDateUtc':
            filtered_cols.append(col)

    return df[filtered_cols] if filtered_cols else df


def get_performance_metric_columns(numeric_cols):
    """Filter numeric columns to only include performance metrics (for dropdowns)."""
    exclude_patterns = [
        'offset', 'Offset', 'UTC', 'utc', 'Id', 'id', 'guid', 'Guid',
        'tenantId', 'testId', 'trialId', 'athleteId', 'groupId',
        'timezone', 'sessionId', 'deviceId', 'softwareVersion',
        'index', 'unnamed'
    ]

    return [col for col in numeric_cols if not any(pattern in col.lower() for pattern in
            [p.lower() for p in exclude_patterns])]



def get_oauth_token(client_id, client_secret, region='euw'):
    """Get OAuth token from VALD API."""
    try:
        # Use the correct VALD security token URL
        token_url = "https://security.valdperformance.com/connect/token"
        data = {
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret
        }
        response = requests.post(token_url, data=data, timeout=30)
        if response.status_code == 200:
            return response.json().get('access_token', '')
        return ''
    except Exception:
        return ''


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_athlete_names(tenant_id, client_id, client_secret, region='euw'):
    """Fetch athlete names from VALD API and create a mapping."""
    try:
        # Get fresh OAuth token
        token = get_oauth_token(client_id, client_secret, region)
        if not token:
            return {}

        base_url = f"https://prd-{region}-api-athlete.valdperformance.com/"
        url = f"{base_url}v2021q3/team/{tenant_id}/athletes/detailed"

        headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json'
        }

        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            athletes = response.json()
            mapping = {}

            athlete_list = athletes if isinstance(athletes, list) else athletes.get('athletes', [])

            for athlete in athlete_list:
                athlete_id = str(athlete.get('id', ''))
                given_name = athlete.get('givenName', '')
                family_name = athlete.get('familyName', '')
                full_name = f"{given_name} {family_name}".strip()

                if athlete_id and full_name:
                    mapping[athlete_id] = full_name

            return mapping
        else:
            return {}
    except Exception:
        return {}


def normalize_dataframe_columns(df, athlete_mapping=None):
    """Normalize column names to ensure Name and testType columns exist."""
    if df.empty:
        return df

    # Create Name column if it doesn't exist
    if 'Name' not in df.columns:
        # Priority 1: Use full_name column if it exists (already has real names from API)
        if 'full_name' in df.columns:
            df['Name'] = df['full_name'].fillna('')
            # For any missing names, try mapping or use profileId
            if 'profileId' in df.columns:
                def get_name(row):
                    if pd.notna(row.get('full_name')) and str(row.get('full_name')).strip():
                        return str(row['full_name']).strip()
                    pid = str(row.get('profileId', ''))
                    if athlete_mapping and pid in athlete_mapping:
                        return athlete_mapping[pid]
                    return f"Athlete_{pid[:8]}" if pid else 'Unknown'
                df['Name'] = df.apply(get_name, axis=1)
            else:
                df['Name'] = df['full_name'].fillna('Unknown')
        # Priority 2: Use profileId with mapping
        elif 'profileId' in df.columns:
            if athlete_mapping:
                df['Name'] = df['profileId'].apply(lambda x: athlete_mapping.get(str(x), f"Athlete_{str(x)[:8]}"))
            else:
                df['Name'] = df['profileId'].apply(lambda x: f"Athlete_{str(x)[:8]}")
        elif 'athleteId' in df.columns:
            if athlete_mapping:
                df['Name'] = df['athleteId'].apply(lambda x: athlete_mapping.get(str(x), f"Athlete_{str(x)[:8]}"))
            else:
                df['Name'] = df['athleteId'].apply(lambda x: f"Athlete_{str(x)[:8]}")
        else:
            df['Name'] = 'Unknown'

    # Create testType column if it doesn't exist
    if 'testType' not in df.columns:
        if 'testTypeName' in df.columns:
            df['testType'] = df['testTypeName']
        else:
            df['testType'] = 'Unknown'

    # Normalize date column
    if 'recordedDateUtc' not in df.columns:
        if 'testDateUtc' in df.columns:
            df['recordedDateUtc'] = pd.to_datetime(df['testDateUtc'])
        elif 'modifiedDateUtc' in df.columns:
            df['recordedDateUtc'] = pd.to_datetime(df['modifiedDateUtc'])

    return df


# ============================================================================
# VALD API CREDENTIAL LOADING
# ============================================================================

def load_env_credentials():
    """Load VALD API credentials from Streamlit secrets or .env file.

    Priority:
    1. Streamlit secrets (for Cloud deployment)
    2. Local .env file (for local development)
    """
    credentials = {
        'token': '',
        'tenant_id': '',
        'client_id': '',
        'client_secret': '',
        'region': 'euw'
    }

    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        if hasattr(st, 'secrets') and 'vald' in st.secrets:
            credentials['token'] = st.secrets['vald'].get('MANUAL_TOKEN', '')
            credentials['tenant_id'] = st.secrets['vald'].get('TENANT_ID', '')
            credentials['client_id'] = st.secrets['vald'].get('CLIENT_ID', '')
            credentials['client_secret'] = st.secrets['vald'].get('CLIENT_SECRET', '')
            credentials['region'] = st.secrets['vald'].get('VALD_REGION', 'euw')
            if credentials['tenant_id'] and (credentials['token'] or (credentials['client_id'] and credentials['client_secret'])):
                return credentials, True
    except Exception:
        pass  # Fall through to .env file

    # Try multiple possible locations for .env file (relative paths for cloud compatibility)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        # In same directory as dashboard (Streamlit Cloud)
        os.path.join(script_dir, '.env'),
        # Relative to dashboard directory
        os.path.join(os.path.dirname(script_dir), 'vald_api_pulls-main', 'forcedecks', '.env'),
        # In parent directory
        os.path.join(os.path.dirname(script_dir), '.env'),
    ]

    env_path = None
    for path in possible_paths:
        if os.path.exists(path):
            env_path = path
            break

    if not env_path:
        return credentials, False

    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    if key == 'MANUAL_TOKEN':
                        credentials['token'] = value
                    elif key == 'TENANT_ID':
                        credentials['tenant_id'] = value
                    elif key == 'CLIENT_ID':
                        credentials['client_id'] = value
                    elif key == 'CLIENT_SECRET':
                        credentials['client_secret'] = value
                    elif key == 'VALD_REGION':
                        credentials['region'] = value
        return credentials, True
    except Exception:
        return credentials, False


@st.cache_data(ttl=3600, show_spinner=False)
def get_force_trace(test_id, trial_id, token, tenant_id, region='euw'):
    """Fetch force trace data from VALD API for a specific trial. Cached for 1 hour."""
    import requests

    base_urls = {
        'euw': 'https://prd-euw-webapi-1.forcedecks.com',
        'use': 'https://prd-use-webapi-1.forcedecks.com',
        'aue': 'https://prd-aue-webapi-1.forcedecks.com'
    }

    base_url = base_urls.get(region, base_urls['euw'])
    url = f"{base_url}/api/trials/{trial_id}/trace"

    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }

    params = {'tenantId': tenant_id}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)

        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return None, f"Request failed: {str(e)}"


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Team Saudi | Performance Analysis Dashboard",
    page_icon="🇸🇦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://olympic.sa/team-saudi/',
        'Report a bug': None,
        'About': """
        ## Team Saudi Performance Analysis Dashboard

        **Version 3.0** | Saudi Olympic & Paralympic Committee

        World-class strength & conditioning analysis powered by VALD Performance technology.

        ---

        🏅 15+ Olympic & Paralympic Sports

        📊 Evidence-based benchmarks

        🔬 Advanced analytics & force trace analysis

        ---

        © 2025 Saudi Olympic & Paralympic Committee

        [olympic.sa](https://olympic.sa) | [valdperformance.com](https://valdperformance.com)
        """
    }
)

# ============================================================================
# CUSTOM CSS - OFFICIAL SAUDI OLYMPIC THEME
# Theme based on: https://olympic.sa/team-saudi/
# ============================================================================

st.markdown("""
<style>
    /* Import Official Saudi Olympic fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800&display=swap');

    /* CSS Variables - Official Team Saudi Green (from TEAM SAUDI GREEN.png) */
    :root {
        --saudi-green: #1D4D3B;
        --saudi-green-dark: #153829;
        --saudi-green-light: #2A6A50;
        --saudi-gold: #a08e66;
        --saudi-gold-alt: #9d8e65;
        --saudi-gold-hover: #998967;
        --saudi-text-primary: #111827;
        --saudi-text-secondary: #4b5563;
        --saudi-border: #eeeeee;
        --saudi-white: #ffffff;
        --saudi-gray: #f1f1f1;
        --saudi-light-gray: #f5f5f5;
        --saudi-shadow: rgba(0, 0, 0, 0.1);
        --saudi-shadow-strong: rgba(0, 0, 0, 0.2);
    }

    /* Main App Styling - Saudi Olympic Theme */
    .stApp {
        font-family: 'Poppins', 'Roboto', 'Tajawal', sans-serif;
        background: #f8f9fa;
        background-attachment: fixed;
    }

    /* Main container */
    .main .block-container {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    /* Header - Team Saudi Style */
    .main-header {
        background: linear-gradient(135deg, var(--saudi-green) 0%, var(--saudi-green-dark) 100%);
        color: white;
        padding: 3.5rem 2.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        border-top: 4px solid var(--saudi-gold);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.1) 0%, rgba(0, 0, 0, 0.3) 100%);
        pointer-events: none;
    }

    .main-header h1, .main-header h2, .main-header p {
        position: relative;
        z-index: 1;
    }

    /* Metric Cards - Premium Team Saudi Style */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f9f9f9 100%);
        border-radius: 12px;
        padding: 1.8rem;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: center;
        border-left: 5px solid var(--saudi-green);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 100px;
        height: 100px;
        background: linear-gradient(135deg, var(--saudi-green) 0%, transparent 70%);
        opacity: 0.05;
        border-radius: 0 12px 0 100%;
    }

    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 35px rgba(32, 81, 54, 0.25);
        border-left-color: var(--saudi-gold);
        border-left-width: 6px;
    }

    /* Sport Cards - Team Saudi Style */
    .sport-card {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 6px solid var(--sport-color, var(--saudi-green));
        transition: all 0.5s ease;
        position: relative;
    }

    .sport-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(32, 81, 54, 0.25);
    }

    .sport-card::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, var(--saudi-gold) 0%, transparent 100%);
        opacity: 0.1;
        border-radius: 0 10px 0 100%;
    }

    /* Risk Status Badges - Team Saudi Colors */
    .risk-low {
        background: var(--saudi-green);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(32, 81, 54, 0.3);
    }

    .risk-moderate {
        background: var(--saudi-gold);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(160, 142, 102, 0.3);
    }

    .risk-high {
        background: #d32f2f;
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(211, 47, 47, 0.3);
    }

    /* Performance Status - Gold Accents */
    .status-excellent {
        background: linear-gradient(135deg, var(--saudi-gold) 0%, #8a7656 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 2px 10px rgba(160, 142, 102, 0.3);
    }

    .status-good {
        background: linear-gradient(135deg, var(--saudi-green) 0%, var(--saudi-green-dark) 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 2px 10px rgba(0, 113, 103, 0.3);
    }

    .status-average {
        background: linear-gradient(135deg, var(--saudi-light-teal) 0%, #7a9e9a 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
    }

    .status-needs-attention {
        background: linear-gradient(135deg, #d32f2f 0%, #b71c1c 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Tabs Styling - Olympic Style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #f0f2f5;
        padding: 8px;
        border-radius: 10px;
        border-bottom: 3px solid var(--saudi-gold);
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: white;
        border-radius: 8px;
        color: #333333;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
        border: 2px solid #e0e0e0;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f8f9fa;
        border-color: var(--saudi-gold);
        color: var(--saudi-green);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--saudi-green) 0%, var(--saudi-green-dark) 100%);
        color: white !important;
        border-color: var(--saudi-gold);
        box-shadow: 0 4px 12px rgba(32, 81, 54, 0.3);
    }

    .stTabs [aria-selected="true"] p {
        color: white !important;
    }

    .stTabs [aria-selected="true"] span {
        color: white !important;
    }

    /* Sidebar - Stunning Green Design */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--saudi-green) 0%, var(--saudi-green-dark) 100%);
        border-right: 4px solid var(--saudi-gold);
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: white !important;
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: white !important;
    }

    /* Sidebar Headers */
    [data-testid="stSidebar"] h2 {
        color: var(--saudi-gold);
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        border-bottom: 3px solid var(--saudi-gold);
        padding-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: white !important;
    }

    /* Sidebar selectbox styling */
    [data-testid="stSidebar"] .stSelectbox label {
        color: white !important;
        font-weight: 600;
    }

    [data-testid="stSidebar"] .stMultiSelect label {
        color: white !important;
        font-weight: 600;
    }

    /* Sidebar - ALL text white for readability */
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] .stSelectbox span,
    [data-testid="stSidebar"] .stMultiSelect span,
    [data-testid="stSidebar"] .stDateInput label,
    [data-testid="stSidebar"] .stDateInput span,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSlider span,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stCheckbox span,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio span,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stNumberInput span,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stTextInput span,
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] .stFileUploader span,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white !important;
    }

    /* Sidebar expander text */
    [data-testid="stSidebar"] .streamlit-expanderHeader,
    [data-testid="stSidebar"] .streamlit-expanderHeader p,
    [data-testid="stSidebar"] .streamlit-expanderHeader span,
    [data-testid="stSidebar"] details summary span {
        color: white !important;
    }

    /* Sidebar info/warning/success boxes - keep readable */
    [data-testid="stSidebar"] .stAlert p,
    [data-testid="stSidebar"] .stAlert span {
        color: #333 !important;
    }

    /* Data Tables - Professional with White Background */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        background-color: white !important;
    }

    .stDataFrame table {
        background-color: white !important;
    }

    .stDataFrame thead tr th {
        background-color: white !important;
        color: #333 !important;
    }

    .stDataFrame tbody tr td {
        background-color: white !important;
        color: #333 !important;
    }

    .stDataFrame tbody tr:nth-child(even) {
        background-color: #f9f9f9 !important;
    }

    .stDataFrame tbody tr:hover {
        background-color: #f5f5f5 !important;
    }

    /* Buttons - Team Saudi Theme */
    .stButton > button {
        background-color: var(--saudi-green);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        transition: all 0.5s ease;
        box-shadow: 0 2px 8px rgba(32, 81, 54, 0.2);
    }

    .stButton > button:hover {
        background-color: var(--saudi-gold);
        box-shadow: 0 4px 15px rgba(160, 142, 102, 0.4);
        transform: translateY(-2px);
    }

    /* Download Buttons */
    .stDownloadButton > button {
        background-color: var(--saudi-gold);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.5s ease;
    }

    .stDownloadButton > button:hover {
        background-color: var(--saudi-green);
        transform: translateY(-2px);
    }

    /* Metrics - Streamlit Native */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: var(--saudi-green);
        font-family: 'Poppins', sans-serif;
    }

    [data-testid="stMetricDelta"] {
        font-size: 1rem;
        font-weight: 600;
    }

    /* Expanders - Team Saudi Style */
    .streamlit-expanderHeader {
        background-color: rgba(32, 81, 54, 0.05);
        border-left: 4px solid var(--saudi-green);
        border-radius: 8px;
        font-weight: 600;
        color: var(--saudi-text-primary);
        transition: all 0.3s ease;
    }

    .streamlit-expanderHeader:hover {
        background-color: rgba(32, 81, 54, 0.1);
        border-left-color: var(--saudi-gold);
    }

    /* Select Boxes */
    .stSelectbox > div > div {
        border-color: var(--saudi-light-gray);
        border-radius: 8px;
    }

    .stSelectbox > div > div:focus-within {
        border-color: var(--saudi-green);
        box-shadow: 0 0 0 2px rgba(32, 81, 54, 0.1);
    }

    /* Multiselect */
    .stMultiSelect > div > div {
        border-color: var(--saudi-light-gray);
        border-radius: 8px;
    }

    /* Text styling for better readability */
    h1, h2, h3 {
        font-family: 'Poppins', 'Tajawal', sans-serif;
        font-weight: 700;
        color: #1a1a1a;
    }

    h1 {
        color: var(--saudi-green);
    }

    h2 {
        color: var(--saudi-text-primary);
    }

    h3 {
        color: var(--saudi-text-secondary);
    }

    p, div, span, label {
        color: #333333;
    }

    /* Improve general text contrast */
    .stMarkdown {
        color: #2c3e50;
    }

    /* Arabic text optimization */
    [lang="ar"], .arabic-text {
        font-family: 'Tajawal', sans-serif;
        font-weight: 600;
    }

    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: var(--saudi-gray);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--saudi-green);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--saudi-gold);
    }

    /* Plotly Chart Containers */
    .js-plotly-plot {
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

# Header with logo
st.markdown("""
<div class="main-header">
""", unsafe_allow_html=True)

# Logo section with white backdrop for visibility
col1, col2, col3 = st.columns([2, 1, 2])
with col1:
    st.write("")  # Spacer
with col2:
    st.write("")  # Spacer
with col3:
    st.write("")  # Spacer

st.markdown("""
    <div style="height: 1rem;"></div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# LIVE API DATA REFRESH (for Streamlit Cloud)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_athlete_profiles(token, region, tenant_id):
    """Fetch all athlete profiles from VALD API to get names. Cached for 1 hour."""
    import requests

    profiles_url = f'https://prd-{region}-api-externalprofile.valdperformance.com/api/v1/profiles'
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }
    params = {'TenantId': tenant_id}

    try:
        response = requests.get(profiles_url, headers=headers, params=params, timeout=60)
        if response.status_code == 200:
            profiles = response.json()
            # Create mapping: profileId -> full name
            name_map = {}
            for profile in profiles:
                pid = profile.get('id') or profile.get('profileId')
                name = profile.get('fullName') or profile.get('name') or profile.get('displayName')
                if pid and name:
                    name_map[pid] = name
            return name_map
    except Exception:
        pass
    return {}


@st.cache_data(ttl=300, show_spinner="Fetching data from VALD API...")
def fetch_live_data_from_api(device='forcedecks'):
    """Fetch fresh data from VALD API using Streamlit secrets or .env credentials. Cached for 5 min."""
    import requests
    from datetime import datetime, timedelta, timezone

    # Try to get credentials from Streamlit secrets first
    try:
        if hasattr(st, 'secrets') and 'vald' in st.secrets:
            client_id = st.secrets['vald'].get('CLIENT_ID', '')
            client_secret = st.secrets['vald'].get('CLIENT_SECRET', '')
            tenant_id = st.secrets['vald'].get('TENANT_ID', '')
            region = st.secrets['vald'].get('VALD_REGION', 'euw')
        else:
            # Fall back to environment variables
            credentials, found = load_env_credentials()
            if not found:
                return None, "No API credentials found. Configure in Streamlit secrets or .env file."
            client_id = credentials.get('client_id', '')
            client_secret = credentials.get('client_secret', '')
            tenant_id = credentials['tenant_id']
            region = credentials['region']
    except Exception as e:
        return None, f"Error loading credentials: {str(e)}"

    if not client_id or not client_secret or not tenant_id:
        return None, "Missing API credentials (CLIENT_ID, CLIENT_SECRET, or TENANT_ID)"

    # Get OAuth token - use correct identity URL based on region
    region_token_urls = {
        'euw': "https://identity-euw.vald.com/connect/token",
        'use': "https://identity-use.vald.com/connect/token",
        'aue': "https://identity-aue.vald.com/connect/token"
    }
    token_url = region_token_urls.get(region, "https://identity-euw.vald.com/connect/token")
    token_data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
        'scope': 'vald-api'
    }

    try:
        token_response = requests.post(token_url, data=token_data, timeout=30)
        if token_response.status_code != 200:
            return None, f"Failed to get OAuth token: {token_response.status_code}"
        token = token_response.json().get('access_token')
    except Exception as e:
        return None, f"Token request failed: {str(e)}"

    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }

    # Set date range (last 35 days - API limit)
    now = datetime.now(timezone.utc)
    start_date = (now - timedelta(days=34)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date = now.strftime('%Y-%m-%dT%H:%M:%SZ')

    # API endpoints
    endpoints = {
        'forcedecks': f'https://prd-{region}-api-extforcedecks.valdperformance.com/tests',
        'forceframe': f'https://prd-{region}-api-externalforceframe.valdperformance.com/tests',
        'nordbord': f'https://prd-{region}-api-externalnordbord.valdperformance.com/tests',
    }

    if device not in endpoints:
        return None, f"Unknown device: {device}"

    url = endpoints[device]
    params = {
        'TenantId': tenant_id,
        'TestFromUtc': start_date,
        'TestToUtc': end_date,
        'ModifiedFromUtc': start_date,
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=60)
        if response.status_code == 200:
            data = response.json()
            # Handle dict with 'tests' key or list
            if isinstance(data, dict) and 'tests' in data:
                tests = data['tests']
            elif isinstance(data, list):
                tests = data
            else:
                tests = []

            if tests:
                df = pd.DataFrame(tests)

                # Fetch athlete profiles to get names
                if 'profileId' in df.columns:
                    profile_names = fetch_athlete_profiles(token, region, tenant_id)
                    if profile_names:
                        df['full_name'] = df['profileId'].map(profile_names)

                return df, None
            else:
                return pd.DataFrame(), "No data found in the specified date range"
        else:
            return None, f"API Error {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return None, f"API request failed: {str(e)}"


@st.cache_data(ttl=600, show_spinner="Fetching historical data...")
def fetch_historical_data_from_api(device='forcedecks', progress_callback=None):
    """
    Fetch ALL historical data from VALD API using the valdR pattern. Cached for 10 min.
    1. Get OAuth token
    2. Get all athletes
    3. For each athlete, get all tests (no date limit!)

    This bypasses the 35-day API limit by fetching per-athlete.
    Based on the official valdR package approach.
    """
    import requests
    import time

    # Try to get credentials from Streamlit secrets first
    try:
        if hasattr(st, 'secrets') and 'vald' in st.secrets:
            client_id = st.secrets['vald'].get('CLIENT_ID', '')
            client_secret = st.secrets['vald'].get('CLIENT_SECRET', '')
            region = st.secrets['vald'].get('VALD_REGION', 'euw')
        else:
            credentials, found = load_env_credentials()
            if not found:
                return None, "No API credentials found. Configure in Streamlit secrets or .env file."
            client_id = credentials.get('client_id', '')
            client_secret = credentials.get('client_secret', '')
            region = credentials['region']
    except Exception as e:
        return None, f"Error loading credentials: {str(e)}"

    if not client_id or not client_secret:
        return None, "Missing API credentials (CLIENT_ID, CLIENT_SECRET)"

    # Get OAuth token - using correct valdR identity URL
    region_token_urls = {
        'euw': "https://identity-euw.vald.com/connect/token",
        'use': "https://identity-use.vald.com/connect/token",
        'aue': "https://identity-aue.vald.com/connect/token"
    }
    token_url = region_token_urls.get(region, "https://identity-euw.vald.com/connect/token")

    token_data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
        'scope': 'vald-api'
    }

    try:
        token_response = requests.post(token_url, data=token_data, timeout=30)
        if token_response.status_code != 200:
            return None, f"Token error {token_response.status_code}: {token_response.text[:200]}"
        token = token_response.json().get('access_token')
    except Exception as e:
        return None, f"Token request failed: {str(e)}"

    headers = {'Authorization': f'Bearer {token}'}

    # API base URLs per device (valdR pattern)
    api_bases = {
        'forcedecks': f'https://prd-{region}-api-extforcedecks.valdperformance.com/api',
        'forceframe': f'https://prd-{region}-api-externalforceframe.valdperformance.com/api',
        'nordbord': f'https://prd-{region}-api-externalnordbord.valdperformance.com/api'
    }

    if device not in api_bases:
        return None, f"Unknown device: {device}"

    api_base = api_bases[device]

    # Step 1: Get all athletes
    if progress_callback:
        progress_callback("Fetching athlete list...")

    try:
        athletes_url = f"{api_base}/athletes"
        athletes_response = requests.get(athletes_url, headers=headers, timeout=60)
        if athletes_response.status_code != 200:
            return None, f"Failed to get athletes: {athletes_response.status_code}"
        athletes = athletes_response.json()
    except Exception as e:
        return None, f"Failed to fetch athletes: {str(e)}"

    if not athletes:
        return None, "No athletes found"

    # Step 2: Get all tests for each athlete
    all_tests = []
    total_athletes = len(athletes)

    for i, athlete in enumerate(athletes):
        athlete_id = athlete.get('id')
        athlete_name = f"{athlete.get('firstName', '')} {athlete.get('lastName', '')}".strip()

        if progress_callback:
            progress_callback(f"Fetching tests for athlete {i+1}/{total_athletes}: {athlete_name}")

        try:
            tests_url = f"{api_base}/athletes/{athlete_id}/tests"
            tests_response = requests.get(tests_url, headers=headers, timeout=30)

            if tests_response.status_code == 200:
                tests = tests_response.json()
                for test in tests:
                    test['athlete_name'] = athlete_name
                    test['athlete_id'] = athlete_id
                    test['full_name'] = athlete_name
                all_tests.extend(tests)
            elif tests_response.status_code == 404:
                continue  # No tests for this athlete
        except Exception:
            continue  # Skip failed athletes

        # Polite delay to avoid rate limiting (like valdR)
        time.sleep(0.3)

    if not all_tests:
        return pd.DataFrame(), "No tests found for any athletes"

    df = pd.DataFrame(all_tests)
    return df, None


# ============================================================================
# DATA LOADING WITH UPLOAD & REFRESH SUPPORT
# ============================================================================

# Initialize session state for uploaded data
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'local'

# Data management section in sidebar (before filters)
with st.sidebar.expander("📁 Data Management", expanded=False):
    # File uploader for Streamlit Cloud
    uploaded_file = st.file_uploader(
        "Upload CSV Data",
        type=['csv'],
        help="Upload your VALD ForceDecks CSV export",
        key="data_uploader"
    )

    if uploaded_file is not None:
        try:
            st.session_state.uploaded_data = pd.read_csv(uploaded_file)
            if 'recordedDateUtc' in st.session_state.uploaded_data.columns:
                st.session_state.uploaded_data['recordedDateUtc'] = pd.to_datetime(
                    st.session_state.uploaded_data['recordedDateUtc']
                )
            st.session_state.data_source = 'uploaded'
            st.success(f"✅ Loaded {len(st.session_state.uploaded_data)} records")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

    # Refresh button to clear cache
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", help="Clear cache and reload data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.uploaded_data = None
            st.session_state.data_source = 'local'
            st.rerun()

    with col2:
        if st.button("🗑️ Clear", help="Clear uploaded data", use_container_width=True):
            st.session_state.uploaded_data = None
            st.session_state.data_source = 'local'
            st.rerun()

    # Show current data source
    if st.session_state.data_source == 'uploaded':
        st.info("📤 Using uploaded data")
    elif st.session_state.data_source == 'api':
        st.info("🌐 Using live API data")
    else:
        st.info("💾 Using local data")

    # Live API Refresh section
    st.markdown("---")
    st.markdown("**🌐 Live API Refresh**")

    api_device = st.selectbox(
        "Device:",
        ["ForceDecks", "ForceFrame", "NordBord"],
        key="api_device_select"
    )

    if st.button("🔄 Fetch from API", help="Pull fresh data from VALD API and merge with historical data", use_container_width=True):
        device_map = {"ForceDecks": "forcedecks", "ForceFrame": "forceframe", "NordBord": "nordbord"}
        device_key = device_map[api_device]

        with st.spinner(f"Fetching {api_device} data from API..."):
            df_api, error = fetch_live_data_from_api(device_key)

            if error:
                st.error(f"API Error: {error}")
            elif df_api is not None and not df_api.empty:
                # Load existing historical data (use relative paths for cloud compatibility)
                script_dir = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(script_dir, 'data', f'{device_key}_allsports_with_athletes.csv')

                existing_df = None
                if os.path.exists(csv_path):
                    existing_df = pd.read_csv(csv_path, low_memory=False)

                new_count = len(df_api)

                if existing_df is not None and not existing_df.empty:
                    # Merge: keep all existing, add new records (avoid duplicates by testId)
                    if 'testId' in df_api.columns and 'testId' in existing_df.columns:
                        existing_ids = set(existing_df['testId'].astype(str))
                        new_records = df_api[~df_api['testId'].astype(str).isin(existing_ids)]
                        new_count = len(new_records)

                        if not new_records.empty:
                            # Add athlete_sport column to new records if not present
                            if 'athlete_sport' not in new_records.columns and 'athlete_sport' in existing_df.columns:
                                # Try to get athlete_sport from existing data based on profileId
                                if 'profileId' in new_records.columns:
                                    sport_map = existing_df.groupby('profileId')['athlete_sport'].first().to_dict()
                                    new_records['athlete_sport'] = new_records['profileId'].map(sport_map).fillna('Unknown')

                            merged_df = pd.concat([existing_df, new_records], ignore_index=True)
                        else:
                            merged_df = existing_df
                    else:
                        merged_df = pd.concat([existing_df, df_api], ignore_index=True)
                else:
                    merged_df = df_api

                # Save merged data to CSV (note: on Streamlit Cloud, this is ephemeral storage)
                try:
                    merged_df.to_csv(csv_path, index=False)
                    st.success(f"✅ Added {new_count} new records! Total: {len(merged_df)} records saved.")
                except Exception as e:
                    st.warning(f"Data loaded but save failed: {e}")

                # Store in session state based on device
                if device_key == 'forcedecks':
                    st.session_state.uploaded_data = merged_df
                    st.session_state.data_source = 'api'
                st.cache_data.clear()
                st.rerun()
            else:
                st.warning("No data returned from API")

    # Sync to Private GitHub Repo
    st.markdown("---")
    st.markdown("**☁️ Sync to GitHub**")
    st.caption("Save data to private repo for persistence")

    if GITHUB_SYNC_AVAILABLE:
        sync_device = st.selectbox(
            "Device to sync:",
            ["ForceDecks", "ForceFrame", "NordBord"],
            key="sync_device_select"
        )

        if st.button("☁️ Refresh & Save to GitHub", help="Fetch from API and save to private GitHub repo", use_container_width=True):
            device_map = {"ForceDecks": "forcedecks", "ForceFrame": "forceframe", "NordBord": "nordbord"}
            device_key = device_map[sync_device]

            with st.spinner(f"Fetching {sync_device} data and syncing to GitHub..."):
                df_synced, success = refresh_and_save_data(device_key)

                if success and not df_synced.empty:
                    st.success(f"✅ Synced {len(df_synced)} records to GitHub!")
                    st.cache_data.clear()
                    st.rerun()
                elif not df_synced.empty:
                    st.warning("Data fetched but GitHub sync failed. Check secrets.")
                else:
                    st.error("Failed to fetch data from API. Check VALD credentials.")
    else:
        st.info("GitHub sync requires data_loader with push_to_github_repo function")

    # Historical Refresh section - fetches ALL data (no date limit)
    st.markdown("---")
    st.markdown("**📜 Full Historical Refresh**")
    st.caption("Fetches ALL test data per athlete (bypasses 35-day limit)")

    hist_device = st.selectbox(
        "Device:",
        ["ForceDecks", "ForceFrame", "NordBord"],
        key="hist_device_select"
    )

    if st.button("📜 Full Historical Refresh", help="Pull ALL historical data from VALD API (takes longer)", use_container_width=True):
        device_map = {"ForceDecks": "forcedecks", "ForceFrame": "forceframe", "NordBord": "nordbord"}
        device_key = device_map[hist_device]

        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        def update_progress(message):
            status_placeholder.text(message)

        with st.spinner(f"Fetching ALL {hist_device} historical data..."):
            update_progress("Starting historical data fetch...")
            df_hist, error = fetch_historical_data_from_api(device_key, progress_callback=update_progress)

            if error:
                st.error(f"Historical Fetch Error: {error}")
            elif df_hist is not None and not df_hist.empty:
                # Save to CSV (replaces existing data with fresh download)
                script_dir = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(script_dir, 'data', f'{device_key}_allsports_with_athletes.csv')

                try:
                    df_hist.to_csv(csv_path, index=False)
                    st.success(f"✅ Historical refresh complete! {len(df_hist)} total records saved.")

                    # Get date range if available
                    date_col = 'recordedDateUtc' if 'recordedDateUtc' in df_hist.columns else 'testDateUtc' if 'testDateUtc' in df_hist.columns else None
                    if date_col:
                        df_hist[date_col] = pd.to_datetime(df_hist[date_col])
                        min_date = df_hist[date_col].min()
                        max_date = df_hist[date_col].max()
                        st.info(f"📅 Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

                    # Update session state
                    if device_key == 'forcedecks':
                        st.session_state.uploaded_data = df_hist
                        st.session_state.data_source = 'api'
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.warning(f"Data fetched but save failed: {e}")
                    st.session_state.uploaded_data = df_hist
                    st.session_state.data_source = 'api'
            else:
                st.warning("No historical data returned")

        progress_placeholder.empty()
        status_placeholder.empty()

# Load data based on source
with st.spinner("Loading performance data..."):
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
    else:
        df = load_vald_data('forcedecks')

# Load ForceFrame and NordBord data (uses data_loader with API fallback)
@st.cache_data(ttl=3600)
def load_forceframe_data():
    """Load ForceFrame data from CSV or API. Cached for 1 hour."""
    return load_vald_data('forceframe')

@st.cache_data(ttl=3600)
def load_nordbord_data():
    """Load NordBord data from CSV or API. Cached for 1 hour."""
    return load_vald_data('nordbord')

# Load ForceFrame and NordBord
df_forceframe = load_forceframe_data()
df_nordbord = load_nordbord_data()

# Normalize all dataframes to ensure required columns exist
# Try to load athlete names from JSON mapping file first (faster and more reliable)
def load_athlete_mapping_from_file():
    """Load athlete mapping from local JSON file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mapping_paths = [
        # In dashboard data folder (Streamlit Cloud)
        os.path.join(script_dir, 'data', 'profile_athlete_mapping.json'),
        # Relative path for local development
        os.path.join(os.path.dirname(script_dir), 'vald_api_pulls-main', 'forcedecks', 'data', 'master_files', 'profile_athlete_mapping.json'),
    ]

    for path in mapping_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    raw_mapping = json.load(f)
                # Convert to simple {profileId: name} format
                return {k: v.get('athlete_name', f"Athlete_{k[:8]}") for k, v in raw_mapping.items()}
            except Exception:
                continue
    return None

athlete_mapping = load_athlete_mapping_from_file()

# If no local mapping, try API as fallback
if not athlete_mapping:
    creds, creds_found = load_env_credentials()
    if creds_found and creds.get('tenant_id') and creds.get('client_id') and creds.get('client_secret'):
        athlete_mapping = fetch_athlete_names(
            creds['tenant_id'],
            creds['client_id'],
            creds['client_secret'],
            creds.get('region', 'euw')
        )

df = normalize_dataframe_columns(df, athlete_mapping)
df_forceframe = normalize_dataframe_columns(df_forceframe, athlete_mapping)
df_nordbord = normalize_dataframe_columns(df_nordbord, athlete_mapping)

if df.empty:
    # Check if API credentials are configured
    has_credentials = False
    try:
        if hasattr(st, 'secrets') and 'vald' in st.secrets:
            has_credentials = bool(st.secrets['vald'].get('TENANT_ID'))
    except Exception:
        pass

    if has_credentials:
        st.warning("""
        ### 🔄 Fetching Data from VALD API...

        Data is being loaded from the VALD API. This may take a moment on first load.

        If this persists, please check:
        - Your API credentials in Streamlit secrets
        - Your TENANT_ID is correct
        - The VALD API is accessible
        """)
        st.info("💡 **Tip:** Refresh the page if data doesn't appear within 30 seconds.")
    else:
        st.error("""
        ### ⚠️ API Credentials Required

        Please configure your VALD API credentials in Streamlit secrets:

        ```toml
        [vald]
        CLIENT_ID = "your_client_id"
        CLIENT_SECRET = "your_client_secret"
        TENANT_ID = "your_tenant_id"
        VALD_REGION = "euw"
        ```

        Or use a manual token:
        ```toml
        [vald]
        MANUAL_TOKEN = "your_bearer_token"
        TENANT_ID = "your_tenant_id"
        VALD_REGION = "euw"
        ```
        """)
    st.stop()

# ============================================================================
# SIDEBAR - FILTERS
# ============================================================================

# Sidebar branding with new logo
try:
    import base64
    import os
    # Try multiple locations for the logo
    sidebar_logo_path = os.path.join(os.path.dirname(__file__), 'Saudi logo.png')
    if not os.path.exists(sidebar_logo_path):
        sidebar_logo_path = os.path.join(os.path.dirname(__file__), 'team_saudi_logo.png')
    if not os.path.exists(sidebar_logo_path):
        sidebar_logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Saudi logo.png')
    if not os.path.exists(sidebar_logo_path):
        sidebar_logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'idxJjwCtfw_logos.png')

    if os.path.exists(sidebar_logo_path):
        with open(sidebar_logo_path, "rb") as f:
            sidebar_logo_data = base64.b64encode(f.read()).decode()
        st.sidebar.markdown(f"""
        <div style="text-align: center; padding: 1.5rem 1rem 2rem 1rem; border-bottom: 3px solid #a08e66; margin-bottom: 1.5rem;">
            <img src="data:image/png;base64,{sidebar_logo_data}"
                 alt="Team Saudi"
                 style="
                     width: 100%;
                     max-width: 200px;
                     height: auto;
                     filter: brightness(1.1);
                 ">
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback if logo not found
        st.sidebar.markdown("""
        <div style="text-align: center; padding: 1rem 0 1.5rem 0; border-bottom: 2px solid #a08e66; margin-bottom: 1rem;">
            <div style="font-size: 2rem;">🇸🇦</div>
            <p style="font-family: 'Poppins', sans-serif; font-size: 0.9rem; font-weight: 600; color: white; margin: 0.5rem 0 0 0; letter-spacing: 1px;">TEAM SAUDI</p>
        </div>
        """, unsafe_allow_html=True)
except Exception as e:
    # Fallback on error
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0 1.5rem 0; border-bottom: 2px solid #a08e66; margin-bottom: 1rem;">
        <div style="font-size: 2rem;">🇸🇦</div>
        <p style="font-family: 'Poppins', sans-serif; font-size: 0.9rem; font-weight: 600; color: white; margin: 0.5rem 0 0 0; letter-spacing: 1px;">TEAM SAUDI</p>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("## 🎯 Filters")

# Sport filter (only if athlete_sport column exists)
if 'athlete_sport' in df.columns:
    available_sports = sorted([s for s in df['athlete_sport'].unique() if pd.notna(s)])
    selected_sports = st.sidebar.multiselect(
        "Select Sports:",
        options=available_sports,
        default=available_sports[:5] if len(available_sports) >= 5 else available_sports
    )
else:
    available_sports = []
    selected_sports = []

# Athlete filter
if selected_sports and 'athlete_sport' in df.columns:
    filtered_athletes = df[df['athlete_sport'].isin(selected_sports)]['Name'].unique()
    selected_athletes = st.sidebar.multiselect(
        "Select Athletes (optional):",
        options=sorted(filtered_athletes),
        default=[]
    )
elif 'Name' in df.columns:
    # If no sport filter, show all athletes
    all_athletes = sorted(df['Name'].unique())
    selected_athletes = st.sidebar.multiselect(
        "Select Athletes (optional):",
        options=all_athletes,
        default=[]
    )
else:
    selected_athletes = []

# Test type filter
available_test_types = sorted(df['testType'].unique())
selected_test_types = st.sidebar.multiselect(
    "Select Test Types:",
    options=available_test_types,
    default=available_test_types
)

# Date range filter
if 'recordedDateUtc' in df.columns:
    min_date = df['recordedDateUtc'].min()
    max_date = df['recordedDateUtc'].max()

    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input(
            "Date Range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        date_range = None
else:
    date_range = None

# Apply filters
filtered_df = df.copy()

if selected_sports and 'athlete_sport' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['athlete_sport'].isin(selected_sports)]

if selected_athletes:
    filtered_df = filtered_df[filtered_df['Name'].isin(selected_athletes)]

if selected_test_types:
    filtered_df = filtered_df[filtered_df['testType'].isin(selected_test_types)]

if date_range and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df['recordedDateUtc'].dt.date >= start_date) &
        (filtered_df['recordedDateUtc'].dt.date <= end_date)
    ]

# Summary stats in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Selection")
st.sidebar.metric("Total Tests", len(filtered_df))
st.sidebar.metric("Athletes", filtered_df['Name'].nunique() if 'Name' in filtered_df.columns else 0)
st.sidebar.metric("Sports", filtered_df['athlete_sport'].nunique() if 'athlete_sport' in filtered_df.columns else 0)

# ============================================================================
# SIDEBAR NAVIGATION - Replaces horizontal tabs for easier navigation
# ============================================================================

# Horizontal tabs - reorganized with ForceFrame/NordBord visible
tabs = st.tabs([
    "🏠 Home", "🔲 ForceFrame", "🦵 NordBord", "🏃 Athlete", "🦘 CMJ",
    "💪 Iso", "🥏 Throws", "📉 Trace", "🏅 Sport", "⚠️ Risk",
    "🔀 Compare", "📈 Progress", "🏆 Rank", "🎯 Adv", "⭐ Insights", "📋 Data"
])

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================

with tabs[0]:
    st.markdown("## 🏠 Performance Overview")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_athletes = filtered_df['Name'].nunique() if 'Name' in filtered_df.columns else 0
        st.metric("👥 Total Athletes", total_athletes)

    with col2:
        total_tests = len(filtered_df)
        st.metric("🧪 Total Tests", total_tests)

    with col3:
        total_sports = filtered_df['athlete_sport'].nunique() if 'athlete_sport' in filtered_df.columns else 0
        st.metric("🏅 Sports", total_sports)

    with col4:
        if 'recordedDateUtc' in filtered_df.columns:
            # Handle timezone-aware datetime comparison
            now_utc = datetime.now(timezone.utc)
            seven_days_ago = now_utc - timedelta(days=7)
            last_7_days = filtered_df[filtered_df['recordedDateUtc'] >= seven_days_ago]
            st.metric("📅 Tests (7d)", len(last_7_days))
        else:
            st.metric("📅 Tests (7d)", "N/A")

    # Recent Activity
    st.markdown("### 📅 Recent Testing Activity")

    if 'recordedDateUtc' in filtered_df.columns:
        recent_tests = filtered_df.sort_values('recordedDateUtc', ascending=False).head(10)

        if not recent_tests.empty:
            # Only show essential columns - no technical IDs
            display_cols = ['Name', 'athlete_sport', 'testType', 'recordedDateUtc']
            display_cols = [col for col in display_cols if col in recent_tests.columns]

            st.dataframe(
                recent_tests[display_cols],
                use_container_width=True,
                hide_index=True
            )

    # Sport Distribution
    st.markdown("### 🏅 Sport Distribution")

    if 'athlete_sport' in filtered_df.columns:
        sport_counts = filtered_df['athlete_sport'].value_counts().reset_index()
        sport_counts.columns = ['Sport', 'Count']

        fig_sport_dist = px.bar(
            sport_counts,
            x='Sport',
            y='Count',
            title="Tests per Sport",
            color='Count',
            color_continuous_scale='Greens'
        )

        fig_sport_dist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )

        st.plotly_chart(fig_sport_dist, use_container_width=True)

# ============================================================================
# PAGE: ATHLETE PROFILE
# ============================================================================



with tabs[3]:
    st.markdown("## 🏃 Athlete Deep Dive")

    athlete_list = sorted(filtered_df['Name'].unique()) if 'Name' in filtered_df.columns else []

    if athlete_list:
        selected_athlete = st.selectbox("Select Athlete:", athlete_list)

        if selected_athlete:
            athlete_df = filtered_df[filtered_df['Name'] == selected_athlete].copy()

            # Athlete Header
            col1, col2, col3 = st.columns([2, 3, 3])

            with col1:
                st.markdown(f"### {selected_athlete}")
                if 'athlete_sport' in athlete_df.columns:
                    sport = athlete_df['athlete_sport'].iloc[0]
                    st.markdown(f"**Sport:** {sport}")

            with col2:
                st.metric("Total Tests", len(athlete_df))
                if 'recordedDateUtc' in athlete_df.columns:
                    last_test = athlete_df['recordedDateUtc'].max()
                    st.metric("Last Test", last_test.strftime('%d %b %y') if pd.notna(last_test) else "N/A")

            with col3:
                test_types = athlete_df['testType'].nunique()
                st.metric("Test Types", test_types)

            # Sport Context
            if 'athlete_sport' in athlete_df.columns:
                sport = athlete_df['athlete_sport'].iloc[0]
                sport_context = get_sport_context(sport)

                with st.expander(f"📖 {sport} Context & Benchmarks", expanded=False):
                    st.markdown(f"**Context:** {sport_context['context']}")

                    st.markdown("**Key Attributes:**")
                    for attr in sport_context['key_attributes']:
                        st.markdown(f"- {attr}")

                    st.markdown("**Priority Metrics:**")
                    for metric in sport_context['priority_metrics']:
                        st.markdown(f"- {metric}")

            # Latest Test Performance - Tile Layout
            st.markdown("### 🎯 Latest Test Performance")

            for test_type in athlete_df['testType'].unique():
                test_df = athlete_df[athlete_df['testType'] == test_type].sort_values('recordedDateUtc')

                if not test_df.empty:
                    latest_test = test_df.iloc[-1]

                    # Tile container with custom CSS - compact spacing
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(145deg, #ffffff 0%, #f9f9f9 100%);
                        border-radius: 12px;
                        padding: 1rem 1.5rem 0.5rem 1.5rem;
                        margin: 0.5rem 0;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
                        border-left: 5px solid var(--saudi-green);
                    ">
                        <h4 style="margin: 0 0 0.5rem 0; color: var(--saudi-green); font-family: 'Poppins', sans-serif;">
                            {test_type}
                        </h4>
                    </div>
                    """, unsafe_allow_html=True)

                    metrics = get_metrics_from_test_type(test_type)

                    # Display metrics in a 4-column grid
                    num_cols = 4
                    metric_cols = st.columns(num_cols)

                    for i, metric in enumerate(metrics[:8]):  # Show first 8 metrics
                        if metric in latest_test.index and pd.notna(latest_test[metric]):
                            with metric_cols[i % 4]:
                                value = latest_test[metric]

                                # Get status
                                if 'athlete_sport' in latest_test.index:
                                    sport = latest_test['athlete_sport']
                                    status = get_metric_status(value, sport, metric)
                                    percentile = get_percentile_rank(value, sport, metric)

                                    if percentile:
                                        st.metric(
                                            metric.split('[')[0].strip(),
                                            f"{value:.2f}",
                                            delta=f"{percentile:.0f}th percentile"
                                        )
                                    else:
                                        st.metric(metric.split('[')[0].strip(), f"{value:.2f}")
                                else:
                                    st.metric(metric.split('[')[0].strip(), f"{value:.2f}")

            # Progress Over Time
            st.markdown("### 📈 Progress Over Time")

            test_type_for_progress = st.selectbox(
                "Select test type for progress tracking:",
                athlete_df['testType'].unique()
            )

            if test_type_for_progress:
                progress_df = athlete_df[athlete_df['testType'] == test_type_for_progress].sort_values('recordedDateUtc')

                if len(progress_df) > 1:
                    metrics = get_metrics_from_test_type(test_type_for_progress)

                    for metric in metrics[:4]:  # Show top 4 metrics
                        if metric in progress_df.columns:
                            metric_data = progress_df[['recordedDateUtc', metric]].dropna()

                            if not metric_data.empty:
                                fig = px.line(
                                    metric_data,
                                    x='recordedDateUtc',
                                    y=metric,
                                    title=f"{metric} - Progress Over Time",
                                    markers=True
                                )

                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)'
                                )

                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 tests to show progress")

    else:
        st.warning("No athletes found with current filters")

# ============================================================================
# PAGE: CMJ ANALYSIS MODULE
# ============================================================================



with tabs[4]:
    st.markdown("## 🦘 CMJ (Countermovement Jump) Analysis")
    st.markdown("*Comprehensive bilateral analysis with research-backed insights*")

    if TEST_TYPE_MODULES_AVAILABLE:
        # Athlete selector - filter to only show athletes with CMJ data
        if 'Name' in filtered_df.columns and 'testType' in filtered_df.columns:
            # Filter to only athletes who have CMJ test data
            cmj_df = filtered_df[filtered_df['testType'].str.contains('CMJ', case=False, na=False)]
            athletes = sorted([a for a in cmj_df['Name'].unique() if pd.notna(a)])

            if athletes:
                selected_athlete = st.selectbox("Select Athlete:", athletes, key="cmj_athlete")

                # Filter athlete data
                athlete_df = filtered_df[filtered_df['Name'] == selected_athlete].copy()

                # Get sport
                sport = athlete_df['athlete_sport'].iloc[0] if 'athlete_sport' in athlete_df.columns else "Unknown"

                # Display CMJ modules
                CMJAnalysisModule.display_cmj_dashboard(athlete_df, selected_athlete)

                st.markdown("---")
                st.markdown("## Power-Focused Analysis")
                CMJAnalysisModule.display_cmj_power_focus(athlete_df, selected_athlete)

            else:
                st.warning("No athletes with CMJ test data available")
        else:
            st.warning("Name or testType column not found in data")
    else:
        st.error("CMJ Analysis module not available. Please check utils/test_type_modules.py")

# ============================================================================
# TAB 4: ISOMETRIC ANALYSIS MODULE
# ============================================================================



with tabs[5]:
    st.markdown("## 💪 Isometric Strength Analysis")
    st.markdown("*Single Leg & Double Leg IMTP (Isometric Mid-Thigh Pull)*")

    if TEST_TYPE_MODULES_AVAILABLE:
        # Subtabs for single vs double leg
        iso_subtabs = st.tabs(["Single Leg", "Double Leg"])

        # Athlete selector - filter to only show athletes with Isometric data
        if 'Name' in filtered_df.columns and 'testType' in filtered_df.columns:
            # Filter to only athletes who have Isometric test data (IMTP, etc.)
            iso_df = filtered_df[filtered_df['testType'].str.contains('Isometric|IMTP', case=False, na=False)]
            athletes = sorted([a for a in iso_df['Name'].unique() if pd.notna(a)])

            if athletes:
                selected_athlete = st.selectbox("Select Athlete:", athletes, key="iso_athlete")

                # Filter athlete data
                athlete_df = filtered_df[filtered_df['Name'] == selected_athlete].copy()

                with iso_subtabs[0]:
                    st.markdown("### Single Leg Isometric Peak Force")
                    IsometricSingleLegModule.display_single_leg_analysis(athlete_df, selected_athlete)

                with iso_subtabs[1]:
                    st.markdown("### Double Leg Isometric (IMTP)")
                    IsometricDoubleLegModule.display_double_leg_analysis(athlete_df, selected_athlete)

            else:
                st.warning("No athletes with Isometric test data available")
        else:
            st.warning("Name or testType column not found in data")
    else:
        st.error("Isometric Analysis module not available. Please check utils/test_type_modules.py")

# ============================================================================
# TAB 5: THROWS TRAINING DASHBOARD
# ============================================================================



with tabs[6]:
    st.markdown("## 🥏 Throws Training Dashboard")
    st.markdown("*Performance dashboard for Athletics athletes - tracking power, force, and technical consistency*")

    if TEST_TYPE_MODULES_AVAILABLE:
        # Filter for Athletics athletes (or all athletes if no sport column)
        if 'athlete_sport' in filtered_df.columns and 'Name' in filtered_df.columns:
            # Prefer Athletics athletes, but show all if none found
            athletics_df = filtered_df[
                filtered_df['athlete_sport'].str.contains('Athletics|Track|Field|Shot|Discus|Javelin|Hammer|Throws', case=False, na=False)
            ].copy()

            # Use athletics athletes if found, otherwise use all athletes
            display_df = athletics_df if not athletics_df.empty else filtered_df.copy()

            if not display_df.empty:
                athletes = sorted([a for a in display_df['Name'].unique() if pd.notna(a)])

                if athletes:
                    selected_athlete = st.selectbox("Select Athlete:", athletes, key="throws_athlete")

                    # Filter athlete data
                    athlete_df = display_df[display_df['Name'] == selected_athlete].copy()
                    sport = athlete_df['athlete_sport'].iloc[0] if 'athlete_sport' in athlete_df.columns else "Athletics"

                    # Display throws dashboard
                    ThrowsTrainingModule.display_throws_dashboard(athlete_df, selected_athlete, sport)

                else:
                    st.warning("No athletes with name data available")
            else:
                st.info("No athlete data available")
        else:
            st.warning("Required columns (Name, athlete_sport) not found in data")
    else:
        st.error("Throws Training module not available. Please check utils/test_type_modules.py")

# ============================================================================
# TAB 6: SPORT ANALYSIS
# ============================================================================



with tabs[8]:
    st.markdown("## 🏅 Sport-Specific Analysis")

    # If no sports selected in sidebar, show all available sports
    sports_to_display = selected_sports if selected_sports else (
        filtered_df['athlete_sport'].unique().tolist() if 'athlete_sport' in filtered_df.columns else []
    )

    if sports_to_display:
        for sport in sports_to_display:
            sport_df = filtered_df[filtered_df['athlete_sport'] == sport] if 'athlete_sport' in filtered_df.columns else filtered_df

            if not sport_df.empty:
                sport_context = get_sport_context(sport)

                st.markdown(f"### {sport}")

                # Sport Overview
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Athletes", sport_df['Name'].nunique())

                with col2:
                    st.metric("Total Tests", len(sport_df))

                with col3:
                    st.metric("Test Types", sport_df['testType'].nunique())

                # Context
                with st.expander(f"📖 {sport} Context"):
                    st.markdown(sport_context['context'])

                # Top Performers
                st.markdown(f"#### 🏆 Top Performers - {sport}")

                test_type = st.selectbox(
                    f"Select test type for {sport}:",
                    sport_df['testType'].unique(),
                    key=f"test_{sport}"
                )

                if test_type:
                    test_df = sport_df[sport_df['testType'] == test_type]
                    metrics = get_metrics_from_test_type(test_type)

                    if metrics and metrics[0] in test_df.columns:
                        primary_metric = metrics[0]

                        # Get latest test per athlete
                        latest = test_df.sort_values('recordedDateUtc').groupby('Name').last().reset_index()
                        latest = latest.sort_values(primary_metric, ascending=False)

                        top_10 = latest.head(10)[['Name', primary_metric]]

                        fig_top = px.bar(
                            top_10,
                            x='Name',
                            y=primary_metric,
                            title=f"Top 10 - {primary_metric}",
                            color=primary_metric,
                            color_continuous_scale='Greens'
                        )

                        fig_top.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )

                        st.plotly_chart(fig_top, use_container_width=True)

                st.markdown("---")

    else:
        st.info("Select sports in the sidebar to view analysis")

# ============================================================================
# TAB 7: RISK & READINESS
# ============================================================================



with tabs[9]:
    st.markdown("## ⚠️ Risk & Readiness Assessment")

    # Research-backed overview
    st.markdown("""
    This assessment uses evidence-based thresholds from peer-reviewed research to identify athletes
    who may be at elevated injury risk or require modified training loads.
    """)

    # Expandable research background section
    with st.expander("📚 Research Background & Evidence Base", expanded=False):
        st.markdown("""
        ### Asymmetry & Injury Risk Research

        **Key Studies Supporting These Thresholds:**

        1. **Impellizzeri et al. (2007)** - *Journal of Sports Sciences*
           - Found >15% strength asymmetry associated with **2.6x higher injury risk** in soccer players
           - Recommended threshold: <10% for low risk

        2. **Croisier et al. (2008)** - *American Journal of Sports Medicine*
           - Prospective study of 462 professional soccer players
           - Hamstring strength asymmetry >15% = **4.7x increased hamstring injury risk**
           - Threshold normalized after strength training intervention

        3. **Malone et al. (2018)** - *British Journal of Sports Medicine*
           - Meta-analysis of 7,000+ athletes across multiple sports
           - Consistent finding: **10-15% asymmetry = moderate concern**
           - >15% asymmetry = **significant injury predictor**

        4. **Bishop et al. (2021)** - *Sports Medicine*
           - Comprehensive review of inter-limb asymmetries
           - Task-specific thresholds vary by movement pattern
           - Jump asymmetries more predictive than isometric tests

        5. **Fort-Vanmeerhaeghe et al. (2016)** - *Physical Therapy in Sport*
           - Sport-specific asymmetry norms established
           - Rotational sports (tennis, fencing) show **higher functional asymmetry**

        ### Performance Decline Research

        **Detecting Overtraining & Fatigue:**

        - **Gathercole et al. (2015)** - CMJ height decreases >5% indicate acute fatigue
        - **Taylor et al. (2012)** - RSI (Reactive Strength Index) most sensitive to neuromuscular fatigue
        - **Claudino et al. (2017)** - Jump performance decline predicts injury within 7-14 days

        ### VALD ForceDecks Validation

        VALD ForceDecks has been validated against gold-standard force plates in:
        - McMahon et al. (2018) - *Journal of Strength & Conditioning Research*
        - Compensation for natural biological variation: ±3-5% day-to-day
        """)

    # Risk thresholds with explanations
    with st.expander("⚠️ Risk Level Thresholds Explained", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            ### 🟢 Low Risk
            **Asymmetry: <10%**

            **What this means:**
            - Within normal biological variation
            - No intervention required
            - Continue current training

            **Evidence:**
            - Day-to-day variation typically 3-5%
            - <10% considered "normal" across studies
            """)

        with col2:
            st.markdown("""
            ### 🟡 Moderate Risk
            **Asymmetry: 10-15%**

            **What this means:**
            - Above normal variation
            - Monitor closely over time
            - Consider unilateral work

            **Evidence:**
            - 1.5-2x elevated injury risk
            - Often responds to targeted training
            """)

        with col3:
            st.markdown("""
            ### 🔴 High Risk
            **Asymmetry: >15%**

            **What this means:**
            - Significant imbalance detected
            - Requires intervention
            - Screen for underlying issues

            **Evidence:**
            - 2.6-4.7x increased injury risk
            - May indicate compensation pattern
            """)

    # Sport-specific guidelines
    with st.expander("🏅 Sport-Specific Asymmetry Guidelines", expanded=False):
        st.markdown("""
        ### Understanding Sport Context

        **Not all asymmetry is problematic.** Some sports naturally develop asymmetry that may be:
        - **Functional** (helps performance)
        - **Structural** (developed over years of sport-specific loading)
        - **Compensatory** (may indicate injury risk)

        ---

        | Sport | Expected Asymmetry | Concern Threshold | Notes |
        |-------|-------------------|-------------------|-------|
        | **Fencing** | 15-25% (weapon arm) | >30% | Dominant side strength is functional |
        | **Tennis** | 10-20% (serving arm) | >25% | Monitor rotator cuff balance |
        | **Soccer** | 5-10% (kicking leg) | >15% | Higher thresholds for strikers |
        | **Basketball** | <10% | >12% | Bilateral demands require symmetry |
        | **Athletics (Sprinting)** | <8% | >10% | Small asymmetry affects performance |
        | **Swimming** | <10% | >15% | Stroke-dependent variation |
        | **Weightlifting** | <5% | >8% | Symmetry critical for technique |
        | **Cycling** | <5% | >10% | Power output affected by asymmetry |
        | **Combat Sports** | 10-15% (stance leg) | >20% | Stance-dependent loading |
        | **Volleyball** | 10-15% (hitting arm) | >20% | Monitor shoulder health |

        ---

        **Key Considerations:**

        1. **Unilateral Sports** (fencing, tennis, javelin)
           - Asymmetry is expected and often beneficial
           - Focus on maintaining range of motion
           - Monitor for excessive progression over time

        2. **Bilateral Sports** (weightlifting, swimming, cycling)
           - Symmetry directly affects performance
           - Lower tolerance for asymmetry
           - Address imbalances proactively

        3. **Field Sports** (soccer, rugby, AFL)
           - Moderate asymmetry acceptable
           - Kicking/dominant leg differences normal
           - Monitor for changes from baseline
        """)

    # Coaching interventions
    with st.expander("💡 Coaching Interventions & Action Plan", expanded=False):
        st.markdown("""
        ### Evidence-Based Intervention Strategies

        ---

        #### 🟢 Low Risk Athletes (<10% asymmetry)

        **Actions:**
        - ✅ Continue current program
        - ✅ Maintain bilateral training
        - ✅ Retest in 4-6 weeks

        ---

        #### 🟡 Moderate Risk Athletes (10-15% asymmetry)

        **Immediate Actions:**
        1. **Add unilateral exercises** - 2-3 sets targeting weaker limb
        2. **Deficit work** - Start weaker side first, match reps with stronger
        3. **Movement screening** - Check for mobility/stability restrictions

        **Example Protocol:**
        ```
        Week 1-4:
        - Single leg RDL: 3x8 weaker side, 3x6 stronger side
        - Bulgarian split squat: 3x10 each (weaker first)
        - Single leg hop series: 2x5 each

        Week 5-8:
        - Progress load on weaker side
        - Maintain stronger side
        - Retest asymmetry
        ```

        **Expected Timeline:** 4-8 weeks to see meaningful change

        ---

        #### 🔴 High Risk Athletes (>15% asymmetry)

        **Immediate Actions:**
        1. **Movement screening** - Rule out pain/dysfunction
        2. **Physio referral** - If pain or movement restriction present
        3. **Modify training load** - Reduce high-impact bilateral work
        4. **Daily unilateral focus** - Targeted activation exercises

        **Red Flags Requiring Medical Review:**
        - ⚠️ Asymmetry developed suddenly (>5% change in 2 weeks)
        - ⚠️ Associated with pain or discomfort
        - ⚠️ Movement compensation visible
        - ⚠️ Recent injury history on weaker side

        **Example Protocol:**
        ```
        Phase 1 (Week 1-2): Assessment
        - Daily single leg balance: 3x30s each
        - Activation work: glute bridges, clamshells
        - No heavy bilateral loading

        Phase 2 (Week 3-6): Loading
        - Progressive unilateral strength
        - 80% volume on weaker limb
        - Retest at week 4

        Phase 3 (Week 7+): Integration
        - Return to bilateral with monitoring
        - Maintain unilateral maintenance
        ```

        ---

        ### Performance Decline Interventions

        **If CMJ Height drops >5% from baseline:**

        | Decline | Duration | Likely Cause | Action |
        |---------|----------|--------------|--------|
        | 5-8% | 1-2 days | Acute fatigue | Reduce load 20%, prioritize recovery |
        | 5-8% | >3 days | Accumulated fatigue | Deload week, sleep audit |
        | >10% | Any | Overreaching/illness | Rest, medical screen if persists |
        | >15% | Any | Significant issue | Full rest, medical review |

        **Recovery Priorities:**
        1. Sleep (7-9 hours minimum)
        2. Nutrition (protein timing, hydration)
        3. Active recovery (light movement)
        4. Stress management
        """)

    st.markdown("---")

    # Current athlete risk assessment
    st.markdown("### 📊 Current Athlete Risk Status")

    # Check for asymmetry metrics
    asymmetry_cols = [col for col in filtered_df.columns if 'asymmetry' in col.lower() or 'asym' in col.lower()]

    if asymmetry_cols:
        # Summary metrics at top
        total_athletes = filtered_df['Name'].nunique()

        # Calculate risk counts across all asymmetry metrics
        all_high_risk = set()
        all_moderate_risk = set()
        all_low_risk = set()

        for col in asymmetry_cols:
            asym_cols = ['Name', col]
            if 'athlete_sport' in filtered_df.columns:
                asym_cols.insert(1, 'athlete_sport')
            asym_data = filtered_df[asym_cols].dropna()
            if not asym_data.empty:
                for _, row in asym_data.iterrows():
                    abs_val = abs(row[col])
                    if abs_val > 15:
                        all_high_risk.add(row['Name'])
                    elif abs_val > 10:
                        all_moderate_risk.add(row['Name'])
                    else:
                        all_low_risk.add(row['Name'])

        # Remove duplicates (athlete in high risk shouldn't be in moderate)
        all_moderate_risk = all_moderate_risk - all_high_risk
        all_low_risk = all_low_risk - all_high_risk - all_moderate_risk

        # Display summary cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Athletes", total_athletes)
        with col2:
            st.metric("🔴 High Risk", len(all_high_risk), help="Asymmetry >15% - requires intervention")
        with col3:
            st.metric("🟡 Moderate Risk", len(all_moderate_risk), help="Asymmetry 10-15% - monitor closely")
        with col4:
            st.metric("🟢 Low Risk", len(all_low_risk), help="Asymmetry <10% - continue program")

        st.markdown("---")

        # Detailed analysis by metric
        st.markdown("### ⚖️ Detailed Asymmetry Analysis")

        selected_asym_metric = st.selectbox(
            "Select asymmetry metric to analyze:",
            asymmetry_cols,
            key="risk_asym_metric"
        )

        if selected_asym_metric:
            asym_cols = ['Name', selected_asym_metric]
            if 'athlete_sport' in filtered_df.columns:
                asym_cols.insert(1, 'athlete_sport')
            asym_data = filtered_df[asym_cols].dropna()

            if not asym_data.empty:
                # Calculate risk status with sport context
                asym_data['abs_asymmetry'] = asym_data[selected_asym_metric].abs()
                asym_data['risk_status'] = asym_data.apply(
                    lambda row: get_asymmetry_status(row[selected_asym_metric], row.get('athlete_sport', 'Unknown'))[0],
                    axis=1
                )

                # Color-coded bar chart
                fig_risk = px.bar(
                    asym_data.sort_values('abs_asymmetry', ascending=True),
                    x='abs_asymmetry',
                    y='Name',
                    color='risk_status',
                    color_discrete_map={
                        'Low Risk': '#28a745',
                        'Moderate Risk': '#ffc107',
                        'High Risk': '#dc3545'
                    },
                    orientation='h',
                    title=f"Asymmetry Distribution: {selected_asym_metric}",
                    labels={'abs_asymmetry': 'Absolute Asymmetry (%)', 'Name': 'Athlete'}
                )

                # Add threshold lines
                fig_risk.add_vline(x=10, line_dash="dash", line_color="orange",
                                   annotation_text="10% threshold")
                fig_risk.add_vline(x=15, line_dash="dash", line_color="red",
                                   annotation_text="15% threshold")

                fig_risk.update_layout(
                    height=max(400, len(asym_data) * 25),
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig_risk, use_container_width=True)

                # High risk athletes table with action items
                high_risk = asym_data[asym_data['risk_status'] == 'High Risk'].copy()

                if not high_risk.empty:
                    st.error(f"⚠️ {len(high_risk)} athletes require attention for {selected_asym_metric}")

                    # Add intervention suggestions based on sport
                    if 'athlete_sport' in high_risk.columns:
                        high_risk['Suggested Action'] = high_risk['athlete_sport'].apply(
                            lambda sport: "Maintain current - functional for sport" if sport in ['Fencing', 'Tennis', 'Badminton'] and high_risk['abs_asymmetry'].max() < 25
                            else "Unilateral strength protocol + movement screen"
                        )
                        display_cols = ['Name', 'athlete_sport', selected_asym_metric, 'risk_status', 'Suggested Action']
                    else:
                        high_risk['Suggested Action'] = "Unilateral strength protocol + movement screen"
                        display_cols = ['Name', selected_asym_metric, 'risk_status', 'Suggested Action']

                    st.dataframe(
                        high_risk[display_cols].round(1),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Export option
                    csv = high_risk.to_csv(index=False)
                    st.download_button(
                        label="📥 Export High Risk Athletes",
                        data=csv,
                        file_name=f"high_risk_athletes_{selected_asym_metric}.csv",
                        mime="text/csv"
                    )
                else:
                    st.success("✅ No high-risk athletes detected for this metric")

                # Moderate risk athletes
                moderate_risk = asym_data[asym_data['risk_status'] == 'Moderate Risk']
                if not moderate_risk.empty:
                    with st.expander(f"🟡 View {len(moderate_risk)} Moderate Risk Athletes"):
                        st.dataframe(
                            moderate_risk[[c for c in ['Name', 'athlete_sport', selected_asym_metric, 'risk_status'] if c in moderate_risk.columns]].round(1),
                            use_container_width=True,
                            hide_index=True
                        )

    else:
        st.info("No asymmetry metrics found in current data selection. Select a test type that includes asymmetry data (e.g., CMJ, SL CMJ, Isometric tests).")

# ============================================================================
# TAB 8: COMPARISONS
# ============================================================================



with tabs[10]:
    st.markdown("## 🔀 Multi-Athlete Comparison")

    comparison_athletes = st.multiselect(
        "Select athletes to compare (up to 4):",
        options=sorted(filtered_df['Name'].unique()),
        max_selections=4
    )

    if len(comparison_athletes) >= 2:
        comparison_test = st.selectbox(
            "Select test type:",
            filtered_df['testType'].unique()
        )

        if comparison_test:
            metrics = get_metrics_from_test_type(comparison_test)

            comparison_df = filtered_df[
                (filtered_df['Name'].isin(comparison_athletes)) &
                (filtered_df['testType'] == comparison_test)
            ]

            # Get latest test for each athlete
            latest = comparison_df.sort_values('recordedDateUtc').groupby('Name').last().reset_index()

            if not latest.empty and metrics:
                # Create radar chart
                available_metrics = [m for m in metrics if m in latest.columns]

                if len(available_metrics) >= 3:
                    fig_radar = go.Figure()

                    for athlete in comparison_athletes:
                        athlete_data = latest[latest['Name'] == athlete]

                        if not athlete_data.empty:
                            values = [athlete_data[m].iloc[0] for m in available_metrics]

                            fig_radar.add_trace(go.Scatterpolar(
                                r=values,
                                theta=[m.split('[')[0].strip() for m in available_metrics],
                                fill='toself',
                                name=athlete
                            ))

                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        showlegend=True,
                        title=f"{comparison_test} - Athlete Comparison"
                    )

                    st.plotly_chart(fig_radar, use_container_width=True)

                # Side-by-side comparison table
                st.markdown("### 📊 Detailed Metrics")

                # Only show Name and performance metrics - no technical IDs
                comparison_table = latest[['Name'] + available_metrics]
                st.dataframe(
                    comparison_table.round(2),
                    use_container_width=True,
                    hide_index=True
                )

    else:
        st.info("Select at least 2 athletes to compare")

# ============================================================================
# TAB 9: PROGRESS TRACKING
# ============================================================================



with tabs[11]:
    st.markdown("## 📈 Progress Tracking")

    st.markdown("""
    Track athlete performance changes over time. Identify improvements, plateaus, and areas needing attention.
    """)

    # Athlete selection for progress tracking
    progress_athlete = st.selectbox(
        "Select Athlete:",
        sorted(filtered_df['Name'].unique()) if 'Name' in filtered_df.columns else [],
        key="progress_athlete"
    )

    if progress_athlete:
        athlete_progress_df = filtered_df[filtered_df['Name'] == progress_athlete].copy()

        if not athlete_progress_df.empty and 'recordedDateUtc' in athlete_progress_df.columns:
            # Convert and sort by date
            athlete_progress_df['Test Date'] = pd.to_datetime(athlete_progress_df['recordedDateUtc']).dt.date
            athlete_progress_df = athlete_progress_df.sort_values('Test Date')

            # Test type filter for progress
            progress_test_types = athlete_progress_df['testType'].unique().tolist()
            selected_progress_test = st.selectbox(
                "Select Test Type:",
                progress_test_types,
                key="progress_test_type"
            )

            if selected_progress_test:
                test_progress_df = athlete_progress_df[athlete_progress_df['testType'] == selected_progress_test]

                # Get metrics for this test type
                progress_metrics = get_metrics_from_test_type(selected_progress_test)
                available_progress_metrics = [m for m in progress_metrics if m in test_progress_df.columns]

                if available_progress_metrics:
                    selected_progress_metric = st.selectbox(
                        "Select Metric to Track:",
                        available_progress_metrics,
                        key="progress_metric"
                    )

                    if selected_progress_metric and selected_progress_metric in test_progress_df.columns:
                        metric_data = test_progress_df[['Test Date', selected_progress_metric]].dropna()

                        if len(metric_data) >= 2:
                            # Summary stats
                            col1, col2, col3, col4 = st.columns(4)

                            first_value = metric_data[selected_progress_metric].iloc[0]
                            latest_value = metric_data[selected_progress_metric].iloc[-1]
                            peak_value = metric_data[selected_progress_metric].max()
                            change_pct = ((latest_value - first_value) / first_value * 100) if first_value != 0 else 0

                            with col1:
                                st.metric("First Test", f"{first_value:.1f}")
                            with col2:
                                st.metric("Latest Test", f"{latest_value:.1f}",
                                         delta=f"{change_pct:+.1f}%")
                            with col3:
                                st.metric("Peak Value", f"{peak_value:.1f}")
                            with col4:
                                st.metric("Tests Recorded", len(metric_data))

                            # Progress chart with trend line
                            fig_progress = go.Figure()

                            # Actual values
                            fig_progress.add_trace(go.Scatter(
                                x=metric_data['Test Date'],
                                y=metric_data[selected_progress_metric],
                                mode='lines+markers',
                                name='Actual',
                                line=dict(color='#1D4D3B', width=3),
                                marker=dict(size=10)
                            ))

                            # Trend line (linear regression)
                            x_numeric = np.arange(len(metric_data))
                            y_values = metric_data[selected_progress_metric].values

                            if len(x_numeric) > 1:
                                z = np.polyfit(x_numeric, y_values, 1)
                                p = np.poly1d(z)
                                trend_values = p(x_numeric)

                                fig_progress.add_trace(go.Scatter(
                                    x=metric_data['Test Date'],
                                    y=trend_values,
                                    mode='lines',
                                    name='Trend',
                                    line=dict(color='orange', width=2, dash='dash')
                                ))

                                # Determine trend direction
                                slope = z[0]
                                if slope > 0:
                                    trend_text = "📈 **Improving** - Positive trend detected"
                                    trend_color = "green"
                                elif slope < 0:
                                    trend_text = "📉 **Declining** - Negative trend detected"
                                    trend_color = "red"
                                else:
                                    trend_text = "➡️ **Stable** - No significant change"
                                    trend_color = "gray"

                            fig_progress.update_layout(
                                title=f"{selected_progress_metric} Progress for {progress_athlete}",
                                xaxis_title="Date",
                                yaxis_title=selected_progress_metric,
                                hovermode='x unified',
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )

                            st.plotly_chart(fig_progress, use_container_width=True)

                            # Trend interpretation
                            if len(x_numeric) > 1:
                                if slope > 0:
                                    st.success(trend_text)
                                elif slope < 0:
                                    st.warning(trend_text)
                                else:
                                    st.info(trend_text)

                            # Progress table
                            with st.expander("📋 View All Test Results"):
                                display_cols = ['Test Date', selected_progress_metric]
                                if 'testType' in test_progress_df.columns:
                                    display_cols.insert(1, 'testType')
                                st.dataframe(
                                    test_progress_df[display_cols].round(2),
                                    use_container_width=True,
                                    hide_index=True
                                )

                        else:
                            st.info(f"Only {len(metric_data)} test recorded. Need at least 2 tests to show progress.")
                else:
                    st.info("No metrics available for this test type")
        else:
            st.info("No test data available for this athlete")

# ============================================================================
# TAB 10: TEST ANALYSIS
# ============================================================================



with tabs[12]:
    st.markdown("## 📊 Team Benchmarks & Rankings")

    st.markdown("""
    See where athletes rank within the team for each test type. Identify top performers and athletes needing development.
    """)

    if selected_test_types:
        benchmark_test = st.selectbox(
            "Select test type:",
            selected_test_types,
            key="benchmark_test"
        )

        if benchmark_test:
            test_data = filtered_df[filtered_df['testType'] == benchmark_test].copy()

            if not test_data.empty:
                # Get metrics for this test type
                metrics = get_metrics_from_test_type(benchmark_test)
                available_metrics = [m for m in metrics if m in test_data.columns]

                if available_metrics:
                    ranking_metric = st.selectbox(
                        "Rank by metric:",
                        available_metrics,
                        key="ranking_metric"
                    )

                    if ranking_metric:
                        # Get latest test for each athlete
                        if 'recordedDateUtc' in test_data.columns:
                            latest_tests = test_data.sort_values('recordedDateUtc').groupby('Name').last().reset_index()
                        else:
                            latest_tests = test_data.groupby('Name').last().reset_index()

                        if not latest_tests.empty and ranking_metric in latest_tests.columns:
                            # Calculate rankings
                            latest_tests['Rank'] = latest_tests[ranking_metric].rank(ascending=False, method='min').astype(int)
                            latest_tests = latest_tests.sort_values('Rank')

                            # Team stats
                            col1, col2, col3, col4 = st.columns(4)
                            metric_values = latest_tests[ranking_metric].dropna()

                            with col1:
                                st.metric("Team Average", f"{metric_values.mean():.1f}")
                            with col2:
                                st.metric("Team Best", f"{metric_values.max():.1f}")
                            with col3:
                                st.metric("Team Lowest", f"{metric_values.min():.1f}")
                            with col4:
                                st.metric("Athletes Tested", len(metric_values))

                            st.markdown("---")

                            # Rankings chart
                            fig_ranking = px.bar(
                                latest_tests.head(20),  # Top 20
                                x=ranking_metric,
                                y='Name',
                                orientation='h',
                                color=ranking_metric,
                                color_continuous_scale=['#dc3545', '#ffc107', '#28a745'],
                                title=f"Team Rankings: {ranking_metric}"
                            )

                            fig_ranking.update_layout(
                                yaxis={'categoryorder': 'total ascending'},
                                height=max(400, len(latest_tests.head(20)) * 30),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                showlegend=False
                            )

                            st.plotly_chart(fig_ranking, use_container_width=True)

                            # Percentile breakdown
                            st.markdown("### 📊 Performance Tiers")

                            col1, col2, col3 = st.columns(3)

                            p75 = metric_values.quantile(0.75)
                            p25 = metric_values.quantile(0.25)

                            top_tier = latest_tests[latest_tests[ranking_metric] >= p75]
                            mid_tier = latest_tests[(latest_tests[ranking_metric] >= p25) & (latest_tests[ranking_metric] < p75)]
                            dev_tier = latest_tests[latest_tests[ranking_metric] < p25]

                            with col1:
                                st.markdown("#### 🥇 Top Tier (75th+)")
                                st.write(f"Threshold: ≥{p75:.1f}")
                                if not top_tier.empty:
                                    for _, row in top_tier.iterrows():
                                        st.write(f"• {row['Name']}: {row[ranking_metric]:.1f}")
                                else:
                                    st.write("No athletes")

                            with col2:
                                st.markdown("#### 🥈 Mid Tier (25-75th)")
                                st.write(f"Range: {p25:.1f} - {p75:.1f}")
                                if not mid_tier.empty:
                                    for _, row in mid_tier.head(10).iterrows():
                                        st.write(f"• {row['Name']}: {row[ranking_metric]:.1f}")
                                    if len(mid_tier) > 10:
                                        st.write(f"...and {len(mid_tier)-10} more")
                                else:
                                    st.write("No athletes")

                            with col3:
                                st.markdown("#### 🎯 Development (<25th)")
                                st.write(f"Threshold: <{p25:.1f}")
                                if not dev_tier.empty:
                                    for _, row in dev_tier.iterrows():
                                        st.write(f"• {row['Name']}: {row[ranking_metric]:.1f}")
                                else:
                                    st.write("No athletes")

                            # Distribution histogram
                            with st.expander("📈 Distribution Analysis"):
                                fig_dist = px.histogram(
                                    latest_tests,
                                    x=ranking_metric,
                                    title=f"Distribution of {ranking_metric}",
                                    nbins=15,
                                    color_discrete_sequence=['#1D4D3B']
                                )

                                # Add percentile lines
                                fig_dist.add_vline(x=p25, line_dash="dash", line_color="orange",
                                                   annotation_text="25th %ile")
                                fig_dist.add_vline(x=p75, line_dash="dash", line_color="green",
                                                   annotation_text="75th %ile")
                                fig_dist.add_vline(x=metric_values.mean(), line_dash="solid", line_color="blue",
                                                   annotation_text="Mean")

                                fig_dist.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)'
                                )

                                st.plotly_chart(fig_dist, use_container_width=True)

                            # Full rankings table
                            with st.expander("📋 Full Rankings Table"):
                                display_cols = ['Rank', 'Name']
                                if 'athlete_sport' in latest_tests.columns:
                                    display_cols.append('athlete_sport')
                                display_cols.append(ranking_metric)

                                st.dataframe(
                                    latest_tests[display_cols].round(1),
                                    use_container_width=True,
                                    hide_index=True
                                )

                                # Export
                                csv = latest_tests[display_cols].to_csv(index=False)
                                st.download_button(
                                    label="📥 Export Rankings",
                                    data=csv,
                                    file_name=f"team_rankings_{benchmark_test}_{ranking_metric}.csv",
                                    mime="text/csv"
                                )
    else:
        st.info("Select test types from the sidebar to view team benchmarks")

# ============================================================================
# TAB 11: FORCE TRACE ANALYSIS
# ============================================================================



with tabs[7]:
    st.markdown("## 📉 Force Trace Analysis")

    # =========================================================================
    # TRIAL COMPARISON - Show available tests and allow selection
    # =========================================================================
    st.markdown("### 🔄 Compare Force Traces")

    # Load API credentials first
    env_creds, env_loaded = load_env_credentials()

    if env_loaded:
        st.success("✅ API credentials loaded from .env file")
    else:
        st.warning("⚠️ API credentials not found. Add MANUAL_TOKEN, TENANT_ID to your .env file")

    # Check if data has required columns for comparison
    has_test_ids = 'testId' in filtered_df.columns and 'trialId' in filtered_df.columns

    if has_test_ids and env_loaded:
        # Step 1: Select Athlete
        st.markdown("#### Step 1: Select Athlete")
        selected_trace_athlete = st.selectbox(
            "Choose athlete to view available tests:",
            sorted(filtered_df['Name'].unique()),
            key="trace_athlete_select"
        )

        # Get all tests for this athlete
        athlete_tests_df = filtered_df[filtered_df['Name'] == selected_trace_athlete].copy()

        if not athlete_tests_df.empty and 'recordedDateUtc' in athlete_tests_df.columns:
            # Sort by date descending
            athlete_tests_df = athlete_tests_df.sort_values('recordedDateUtc', ascending=False)

            # Create display table with key columns
            display_cols = ['testType', 'recordedDateUtc', 'testId', 'trialId']
            # Add any performance metrics that exist
            perf_cols = [c for c in athlete_tests_df.columns if any(k in c for k in ['Jump', 'Force', 'Peak', 'Height', 'RSI', 'Power'])]
            display_cols.extend(perf_cols[:3])  # Add up to 3 performance metrics

            available_cols = [c for c in display_cols if c in athlete_tests_df.columns]
            test_table = athlete_tests_df[available_cols].head(30).copy()

            # Format date column
            if 'recordedDateUtc' in test_table.columns:
                test_table['Date'] = pd.to_datetime(test_table['recordedDateUtc']).dt.strftime('%Y-%m-%d %H:%M')
                test_table = test_table.drop('recordedDateUtc', axis=1)

            # Shorten IDs for display
            if 'testId' in test_table.columns:
                test_table['testId_short'] = test_table['testId'].astype(str).str[:8] + '...'
            if 'trialId' in test_table.columns:
                test_table['trialId_short'] = test_table['trialId'].astype(str).str[:8] + '...'

            st.markdown(f"#### Available Tests for {selected_trace_athlete}")
            st.markdown(f"*Showing {len(test_table)} most recent tests*")

            # Show the table
            st.dataframe(
                test_table.round(2),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("---")

            # Step 2: Select Two Tests to Compare
            st.markdown("#### Step 2: Select Two Tests to Compare")

            # Create selection options with readable labels
            test_options = []
            for idx, row in athlete_tests_df.head(30).iterrows():
                date_str = row['recordedDateUtc'].strftime('%Y-%m-%d %H:%M') if pd.notna(row.get('recordedDateUtc')) else 'N/A'
                # Include a key metric if available
                metric_str = ""
                for col in ['Jump Height (Imp-Mom) [cm]', 'Peak Force [N]', 'jumpHeightImpMom[cm]']:
                    if col in row and pd.notna(row[col]):
                        metric_str = f" | {row[col]:.1f}"
                        break
                label = f"{row['testType']} - {date_str}{metric_str}"
                test_options.append({
                    'label': label,
                    'testId': str(row.get('testId', '')),
                    'trialId': str(row.get('trialId', '')),
                    'testType': row.get('testType', ''),
                    'date': date_str
                })

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🔵 Trial 1 (Blue line)**")
                test1_idx = st.selectbox(
                    "Select first test:",
                    range(len(test_options)),
                    format_func=lambda i: test_options[i]['label'],
                    key="trace_test1_select"
                )
                test1_data = test_options[test1_idx] if test_options else None

                if test1_data:
                    st.caption(f"Test ID: `{test1_data['testId'][:16]}...`")
                    st.caption(f"Trial ID: `{test1_data['trialId'][:16]}...`")

            with col2:
                st.markdown("**🟢 Trial 2 (Green line)**")
                # Default to second test if available
                default_idx2 = 1 if len(test_options) > 1 else 0
                test2_idx = st.selectbox(
                    "Select second test:",
                    range(len(test_options)),
                    index=default_idx2,
                    format_func=lambda i: test_options[i]['label'],
                    key="trace_test2_select"
                )
                test2_data = test_options[test2_idx] if test_options else None

                if test2_data:
                    st.caption(f"Test ID: `{test2_data['testId'][:16]}...`")
                    st.caption(f"Trial ID: `{test2_data['trialId'][:16]}...`")

            # Options
            st.markdown("#### Options")
            opt_col1, opt_col2 = st.columns(2)
            with opt_col1:
                normalize_time = st.checkbox("Normalize time (0-100%)", value=False, key="trace_normalize")
            with opt_col2:
                show_diff = st.checkbox("Show difference plot", value=False, key="trace_diff")

            # Step 3: Fetch and Compare
            st.markdown("---")
            st.markdown("#### Step 3: Fetch & Compare")

            if st.button("🔄 Fetch Force Traces from API", type="primary", key="fetch_traces_btn", use_container_width=True):
                if test1_data and test2_data:
                    with st.spinner("Fetching force traces from VALD API..."):
                        try:
                            # Fetch both traces
                            trace1 = get_force_trace(
                                test1_data['testId'],
                                test1_data['trialId'],
                                env_creds['token'],
                                env_creds['tenant_id'],
                                env_creds['region']
                            )
                            trace2 = get_force_trace(
                                test2_data['testId'],
                                test2_data['trialId'],
                                env_creds['token'],
                                env_creds['tenant_id'],
                                env_creds['region']
                            )

                            if trace1 is not None and trace2 is not None:
                                st.success(f"✅ Fetched {len(trace1)} points for Trial 1, {len(trace2)} points for Trial 2")

                                # Store in session state for persistence
                                st.session_state['trace1'] = trace1
                                st.session_state['trace2'] = trace2
                                st.session_state['trace1_label'] = test1_data['label']
                                st.session_state['trace2_label'] = test2_data['label']

                                # Extract time and force data
                                time1 = trace1['time_ms'].values if 'time_ms' in trace1.columns else np.arange(len(trace1))
                                force1 = trace1['force_n'].values if 'force_n' in trace1.columns else trace1.iloc[:, 0].values

                                time2 = trace2['time_ms'].values if 'time_ms' in trace2.columns else np.arange(len(trace2))
                                force2 = trace2['force_n'].values if 'force_n' in trace2.columns else trace2.iloc[:, 0].values

                                # Normalize time if selected
                                if normalize_time:
                                    time1 = np.linspace(0, 100, len(time1))
                                    time2 = np.linspace(0, 100, len(time2))
                                    x_title = "Movement Phase (%)"
                                else:
                                    x_title = "Time (ms)"

                                # Create overlay plot
                                fig = go.Figure()

                                fig.add_trace(go.Scatter(
                                    x=time1, y=force1,
                                    mode='lines',
                                    name=f"Trial 1: {test1_data['date']}",
                                    line=dict(color='#1f77b4', width=2.5)
                                ))

                                fig.add_trace(go.Scatter(
                                    x=time2, y=force2,
                                    mode='lines',
                                    name=f"Trial 2: {test2_data['date']}",
                                    line=dict(color='#2ca02c', width=2.5)
                                ))

                                fig.update_layout(
                                    title=f"Force Trace Comparison - {selected_trace_athlete}",
                                    xaxis_title=x_title,
                                    yaxis_title="Force (N)",
                                    height=500,
                                    hovermode='x unified',
                                    legend=dict(orientation='h', yanchor='bottom', y=1.02)
                                )

                                st.plotly_chart(fig, use_container_width=True)

                                # Metrics comparison
                                st.markdown("#### 📊 Comparison Metrics")
                                m1, m2, m3, m4 = st.columns(4)

                                peak1, peak2 = max(force1), max(force2)
                                avg1, avg2 = np.mean(force1), np.mean(force2)
                                impulse1 = np.trapz(force1, time1) / 1000  # Convert to N·s
                                impulse2 = np.trapz(force2, time2) / 1000

                                with m1:
                                    diff_peak = ((peak2 - peak1) / peak1 * 100) if peak1 > 0 else 0
                                    st.metric("Peak Force T1", f"{peak1:.0f} N")
                                    st.metric("Peak Force T2", f"{peak2:.0f} N", delta=f"{diff_peak:+.1f}%")

                                with m2:
                                    diff_avg = ((avg2 - avg1) / avg1 * 100) if avg1 > 0 else 0
                                    st.metric("Avg Force T1", f"{avg1:.0f} N")
                                    st.metric("Avg Force T2", f"{avg2:.0f} N", delta=f"{diff_avg:+.1f}%")

                                with m3:
                                    st.metric("Duration T1", f"{max(time1):.0f} ms")
                                    st.metric("Duration T2", f"{max(time2):.0f} ms")

                                with m4:
                                    diff_imp = ((impulse2 - impulse1) / impulse1 * 100) if impulse1 > 0 else 0
                                    st.metric("Impulse T1", f"{impulse1:.1f} N·s")
                                    st.metric("Impulse T2", f"{impulse2:.1f} N·s", delta=f"{diff_imp:+.1f}%")

                                # Difference plot if selected
                                if show_diff and len(force1) == len(force2):
                                    st.markdown("#### Difference Plot (Trial 1 - Trial 2)")
                                    force_diff = force1 - force2

                                    fig_diff = go.Figure()
                                    fig_diff.add_trace(go.Scatter(
                                        x=time1, y=force_diff,
                                        mode='lines',
                                        fill='tozeroy',
                                        fillcolor='rgba(214, 39, 40, 0.2)',
                                        line=dict(color='#d62728', width=2)
                                    ))
                                    fig_diff.add_hline(y=0, line_dash="dash", line_color="gray")
                                    fig_diff.update_layout(
                                        xaxis_title=x_title,
                                        yaxis_title="Force Difference (N)",
                                        height=300
                                    )
                                    st.plotly_chart(fig_diff, use_container_width=True)
                                elif show_diff:
                                    st.info("Difference plot requires traces of equal length. Try normalizing time.")

                                # Show raw data option
                                with st.expander("📋 View Raw Trace Data"):
                                    raw_col1, raw_col2 = st.columns(2)
                                    with raw_col1:
                                        st.markdown("**Trial 1 Data**")
                                        st.dataframe(trace1.head(50), use_container_width=True)
                                    with raw_col2:
                                        st.markdown("**Trial 2 Data**")
                                        st.dataframe(trace2.head(50), use_container_width=True)

                            else:
                                st.error("❌ Could not fetch traces. Check API credentials and test IDs.")
                                if trace1 is None:
                                    st.warning(f"Trial 1 failed: testId={test1_data['testId'][:20]}...")
                                if trace2 is None:
                                    st.warning(f"Trial 2 failed: testId={test2_data['testId'][:20]}...")

                        except Exception as e:
                            st.error(f"❌ Error fetching traces: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.error("Please select both tests to compare.")

        else:
            st.info("No tests available for this athlete with testId/trialId data.")

    elif not has_test_ids:
        st.warning("""
        **Your data doesn't have testId and trialId columns.**

        These are required to fetch force traces from the VALD API.

        **To fix this:**
        1. Use the VALD API pull scripts to download data with full metadata
        2. Ensure your export includes testId and trialId columns
        3. Re-upload the data with these columns included
        """)

        # Show what columns ARE available
        with st.expander("View available columns in your data"):
            st.write(list(filtered_df.columns))

    elif not env_loaded:
        st.error("""
        **API credentials not configured.**

        Add these to your `.env` file:
        ```
        MANUAL_TOKEN=your_vald_api_token
        TENANT_ID=your_tenant_id
        VALD_REGION=euw
        ```
        """)

    st.markdown("---")

    # =========================================================================
    # FORCE TRACE MODULE FEATURES (only if module available)
    # =========================================================================
    if not FORCE_TRACE_AVAILABLE:
        st.info("""
        **Advanced Force Trace Features** (requires utils/force_trace_viz.py):
        - Phase detection (eccentric, concentric, flight, landing)
        - Derived metrics (RFD, impulse, durations)
        - Multi-trial overlay with phase markers
        """)
    else:
        st.info("""
        **Force Trace Analysis** provides deep biomechanical insights by analyzing raw force-time curves.

        **Key Features:**
        - 🔬 **Phase Detection**: Automatically identify movement phases (unweighting, eccentric, concentric, flight, landing)
        - 📊 **Trial Consistency**: Overlay multiple trials to assess technique reproducibility
        - 👥 **Athlete Comparison**: Compare force profiles between athletes
        - 📈 **Derived Metrics**: Calculate RFD, impulse, and phase durations
        """)

        # Force Trace Demo Section
        st.markdown("### 🎯 Force Trace Visualization Demo")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Sample Force-Time Curve")

            # Generate sample CMJ force trace for demonstration
            time_ms = np.linspace(0, 2000, 2000)

            # Simulate a CMJ force profile
            bodyweight = 800  # N

            # Create realistic CMJ pattern
            force = np.ones_like(time_ms) * bodyweight

            # Quiet stance (0-300ms)
            force[:300] = bodyweight + np.random.normal(0, 5, 300)

            # Unweighting phase (300-500ms)
            unweight = np.linspace(0, -300, 200)
            force[300:500] = bodyweight + unweight

            # Eccentric braking (500-700ms)
            eccentric = np.linspace(-300, 400, 200)
            force[500:700] = bodyweight + eccentric

            # Concentric push (700-900ms)
            concentric = np.linspace(400, 600, 200)
            force[700:900] = bodyweight + concentric

            # Takeoff to flight (900-1100ms)
            takeoff = np.exp(-np.linspace(0, 5, 200)) * 1400
            force[900:1100] = takeoff

            # Flight phase (1100-1400ms) - zero force
            force[1100:1400] = 0

            # Landing (1400-1700ms)
            landing = np.exp(-np.linspace(0, 3, 300)) * 2000
            force[1400:1700] = landing

            # Recovery (1700-2000ms)
            force[1700:] = bodyweight + np.random.normal(0, 10, 300)

            # Create DataFrame
            sample_trace = pd.DataFrame({
                'time_ms': time_ms,
                'force_n': force
            })

            # Plot using the module function
            fig = plot_force_trace(
                sample_trace,
                title="Sample CMJ Force-Time Curve",
                show_phases=True,
                test_type='CMJ'
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Phase Metrics")

            # Calculate and display metrics
            metrics = calculate_trace_metrics(sample_trace, test_type='CMJ')

            st.metric("Peak Force", f"{metrics.get('peak_force', 0):.0f} N")
            st.metric("Average Force", f"{metrics.get('average_force', 0):.0f} N")

            if 'rfd_100ms' in metrics:
                st.metric("RFD (100ms)", f"{metrics['rfd_100ms']:.0f} N/s")

            if 'impulse' in metrics:
                st.metric("Impulse", f"{metrics['impulse']:.0f} N·s")

            # Phase durations
            st.markdown("**Phase Durations:**")
            for key, value in metrics.items():
                if 'duration' in key:
                    phase_name = key.replace('_duration_ms', '').replace('_', ' ').title()
                    st.write(f"- {phase_name}: {value:.0f} ms")

        # Multi-Trial Comparison Section
        st.markdown("---")
        st.markdown("### 🔄 Multi-Trial Overlay")

        st.info("""
        **Use Cases:**
        - Assess trial-to-trial consistency
        - Identify optimal vs. sub-optimal patterns
        - Monitor technique changes over time
        """)

        # Generate multiple trials for demo
        trials = []
        trial_labels = []

        for i in range(3):
            trial_trace = sample_trace.copy()
            # Add some variation
            trial_trace['force_n'] = trial_trace['force_n'] * (1 + np.random.uniform(-0.05, 0.05)) + np.random.normal(0, 10, len(trial_trace))
            trials.append(trial_trace)
            trial_labels.append(f"Trial {i+1}")

        fig_overlay = plot_multi_trial_overlay(trials, trial_labels, title="Trial-to-Trial Consistency")
        st.plotly_chart(fig_overlay, use_container_width=True)

        # Athlete Comparison Section
        st.markdown("---")
        st.markdown("### 👥 Athlete Force Profile Comparison")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Create sample traces for different athlete profiles
            athlete_traces = {}

            # Power athlete (higher peak, shorter phases)
            power_trace = sample_trace.copy()
            power_trace['force_n'] = power_trace['force_n'] * 1.15
            athlete_traces['Power Athlete'] = power_trace

            # Endurance athlete (lower peak, longer phases)
            endurance_trace = sample_trace.copy()
            endurance_trace['force_n'] = endurance_trace['force_n'] * 0.9
            athlete_traces['Endurance Athlete'] = endurance_trace

            fig_comparison = plot_athlete_comparison(
                athlete_traces,
                test_type='CMJ',
                title="Force Profile Comparison"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)

        with col2:
            st.markdown("**Profile Differences:**")
            st.write("- **Power**: Higher peak force, explosive concentric")
            st.write("- **Endurance**: Lower peak, efficient mechanics")

        # Live Data Fetching Section
        st.markdown("---")
        st.markdown("### 🔌 Fetch Live Force Trace Data")

        st.info("""
        **Select specific tests to fetch force trace data from VALD API.**
        Only selected traces will be downloaded to minimize API calls.
        """)

        # Load credentials (function defined at top of file)
        env_creds, env_loaded = load_env_credentials()

        # API Configuration
        if env_loaded:
            st.success("✅ **API Configuration loaded from .env file**")

            with st.expander("📋 View API Configuration", expanded=False):
                st.markdown(f"""
                **Tenant ID:** `{env_creds['tenant_id']}`
                **Region:** `{env_creds['region']}`
                **Token:** `{'*' * 20}...` (hidden for security)

                *Credentials automatically loaded from:*
                `vald_api_pulls-main/forcedecks/.env`
                """)

            # Use environment credentials
            api_token = env_creds['token']
            tenant_id = env_creds['tenant_id']
            region = env_creds['region']
        else:
            st.warning("⚠️ Could not load .env file. Please configure manually.")

            with st.expander("🔧 Manual API Configuration", expanded=True):
                col1, col2, col3 = st.columns(3)

                with col1:
                    api_token = st.text_input(
                        "API Token:",
                        type="password",
                        help="OAuth bearer token from VALD Hub",
                        key="api_token"
                    )

                with col2:
                    tenant_id = st.text_input(
                        "Tenant ID:",
                        help="Your organization's tenant identifier",
                        key="tenant_id"
                    )

                with col3:
                    region = st.selectbox(
                        "Region:",
                        options=['euw', 'use', 'aue'],
                        index=0,
                        help="euw (Europe), use (US East), aue (Australia)",
                        key="region"
                    )

        # Test Selection
        if 'testId' in filtered_df.columns and 'trialId' in filtered_df.columns:
            st.markdown("#### Select Test to Analyze")

            col1, col2 = st.columns(2)

            with col1:
                # Select athlete
                trace_athlete = st.selectbox(
                    "Select Athlete:",
                    options=sorted(filtered_df['Name'].unique()),
                    key="trace_athlete"
                )

            with col2:
                # Filter tests for selected athlete
                if trace_athlete:
                    athlete_tests = filtered_df[filtered_df['Name'] == trace_athlete].sort_values('recordedDateUtc', ascending=False)

                    if not athlete_tests.empty:
                        # Create test options with date and type
                        test_options = []
                        for _, row in athlete_tests.head(20).iterrows():
                            date_str = row['recordedDateUtc'].strftime('%Y-%m-%d') if pd.notna(row['recordedDateUtc']) else 'N/A'
                            test_label = f"{row['testType']} - {date_str}"
                            test_options.append((test_label, row['testId'], row.get('trialId', '')))

                        selected_test_label = st.selectbox(
                            "Select Test:",
                            options=[t[0] for t in test_options],
                            key="trace_test"
                        )

                        # Get selected test IDs
                        selected_test_info = next((t for t in test_options if t[0] == selected_test_label), None)

                        if selected_test_info:
                            st.markdown(f"**Test ID:** `{selected_test_info[1]}`")
                            if selected_test_info[2]:
                                st.markdown(f"**Trial ID:** `{selected_test_info[2]}`")

            # Fetch button
            if st.button("Fetch Force Trace", type="primary", key="fetch_trace"):
                if not api_token or not tenant_id:
                    st.error("Please configure API Token and Tenant ID in the configuration section above.")
                elif selected_test_info:
                    with st.spinner("Fetching force trace data..."):
                        try:
                            # Import the get_force_trace function
                            from utils.force_trace_viz import get_force_trace

                            trace_data = get_force_trace(
                                test_id=selected_test_info[1],
                                trial_id=selected_test_info[2] if selected_test_info[2] else selected_test_info[1],
                                token=api_token,
                                tenant_id=tenant_id,
                                region=region
                            )

                            if not trace_data.empty:
                                st.success(f"Successfully fetched {len(trace_data)} data points!")

                                # Store in session state for visualization
                                st.session_state['fetched_trace'] = trace_data
                                st.session_state['fetched_trace_info'] = {
                                    'athlete': trace_athlete,
                                    'test': selected_test_label
                                }

                                # Display the trace
                                fig_live = plot_force_trace(
                                    trace_data,
                                    title=f"{trace_athlete} - {selected_test_label}",
                                    show_phases=True,
                                    test_type='CMJ'
                                )
                                st.plotly_chart(fig_live, use_container_width=True)

                                # Display metrics
                                live_metrics = calculate_trace_metrics(trace_data, test_type='CMJ')

                                metric_cols = st.columns(4)
                                with metric_cols[0]:
                                    st.metric("Peak Force", f"{live_metrics.get('peak_force', 0):.0f} N")
                                with metric_cols[1]:
                                    st.metric("Avg Force", f"{live_metrics.get('average_force', 0):.0f} N")
                                with metric_cols[2]:
                                    if 'rfd_100ms' in live_metrics:
                                        st.metric("RFD", f"{live_metrics['rfd_100ms']:.0f} N/s")
                                with metric_cols[3]:
                                    if 'impulse' in live_metrics:
                                        st.metric("Impulse", f"{live_metrics['impulse']:.0f} N·s")

                                # Download option
                                csv = trace_data.to_csv(index=False)
                                st.download_button(
                                    label="Download Trace Data (CSV)",
                                    data=csv,
                                    file_name=f"{trace_athlete}_{selected_test_label.replace(' ', '_')}_trace.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning("No trace data returned. Check your API credentials and test IDs.")

                        except Exception as e:
                            st.error(f"Error fetching trace: {str(e)}")
                else:
                    st.error("Please select a test to fetch.")

            # Show previously fetched trace if available
            if 'fetched_trace' in st.session_state:
                with st.expander("📋 Previously Fetched Trace"):
                    info = st.session_state.get('fetched_trace_info', {})
                    st.markdown(f"**Athlete:** {info.get('athlete', 'N/A')}")
                    st.markdown(f"**Test:** {info.get('test', 'N/A')}")
                    st.markdown(f"**Data Points:** {len(st.session_state['fetched_trace'])}")

        else:
            st.warning("""
            **Test IDs not found in data.**

            To fetch live traces, your data needs `testId` and `trialId` columns.
            These are available when data is pulled directly from the VALD API.
            """)

        # Trial Comparison Section
        st.markdown("---")
        st.markdown("### 🔄 Compare Two Trials Side-by-Side")

        st.info("""
        **Smart filtering allows you to compare:**
        - Same athlete, different dates (progress over time)
        - Same athlete, different test types (CMJ vs SJ)
        - Different athletes, same test type (athlete comparison)
        - Left vs Right limb asymmetry
        """)

        # Load credentials for comparison
        env_creds_comp, env_loaded_comp = load_env_credentials()

        if env_loaded_comp:
            api_token_comp = env_creds_comp['token']
            tenant_id_comp = env_creds_comp['tenant_id']
            region_comp = env_creds_comp['region']
        else:
            api_token_comp = ""
            tenant_id_comp = ""
            region_comp = "euw"

        if 'testId' in filtered_df.columns and 'trialId' in filtered_df.columns:
            # Smart filter options
            col1, col2 = st.columns(2)

            with col1:
                comparison_mode = st.radio(
                    "Comparison Mode:",
                    ["Same Athlete - Progress Over Time",
                     "Same Athlete - Different Tests",
                     "Different Athletes - Same Test Type",
                     "Bilateral Asymmetry (Left vs Right)"],
                    key="comparison_mode"
                )

            with col2:
                show_difference = st.checkbox("Show Difference Plot", value=False, key="show_diff")
                normalize_time = st.checkbox("Normalize Time Axis", value=True, key="normalize_time")

            # Trial 1 Selection
            st.markdown("#### 🔵 Trial 1 Selection")
            col1a, col1b, col1c = st.columns(3)

            with col1a:
                if comparison_mode == "Different Athletes - Same Test Type":
                    athlete1 = st.selectbox(
                        "Athlete 1:",
                        options=sorted(filtered_df['Name'].unique()),
                        key="comp_athlete1"
                    )
                else:
                    athlete1 = st.selectbox(
                        "Athlete:",
                        options=sorted(filtered_df['Name'].unique()),
                        key="comp_athlete1_same"
                    )

            with col1b:
                # Filter tests for athlete 1
                if comparison_mode == "Different Athletes - Same Test Type":
                    athlete1_tests = filtered_df[filtered_df['Name'] == athlete1].sort_values('recordedDateUtc', ascending=False)
                else:
                    athlete1_tests = filtered_df[filtered_df['Name'] == athlete1].sort_values('recordedDateUtc', ascending=False)

                if not athlete1_tests.empty:
                    test_options1 = []
                    for _, row in athlete1_tests.head(20).iterrows():
                        date_str = row['recordedDateUtc'].strftime('%Y-%m-%d') if pd.notna(row['recordedDateUtc']) else 'N/A'
                        test_label = f"{row['testType']} - {date_str}"
                        test_options1.append((test_label, row['testId'], row.get('trialId', ''), row['testType']))

                    test1_label = st.selectbox(
                        "Test 1:",
                        options=[t[0] for t in test_options1],
                        key="comp_test1"
                    )
                    test1_info = next((t for t in test_options1 if t[0] == test1_label), None)

            with col1c:
                if test1_info:
                    st.markdown(f"**Test Type:** {test1_info[3]}")
                    st.markdown(f"**Test ID:** `{test1_info[1][:8]}...`")

            # Trial 2 Selection
            st.markdown("#### 🟢 Trial 2 Selection")
            col2a, col2b, col2c = st.columns(3)

            with col2a:
                if comparison_mode == "Different Athletes - Same Test Type":
                    # Filter to different athlete
                    other_athletes = [a for a in sorted(filtered_df['Name'].unique()) if a != athlete1]
                    athlete2 = st.selectbox(
                        "Athlete 2:",
                        options=other_athletes,
                        key="comp_athlete2"
                    )
                else:
                    athlete2 = athlete1  # Same athlete
                    st.markdown(f"**Athlete:** {athlete2}")

            with col2b:
                # Filter tests for athlete 2
                if comparison_mode == "Same Athlete - Progress Over Time":
                    # Show all dates
                    athlete2_tests = filtered_df[filtered_df['Name'] == athlete2].sort_values('recordedDateUtc', ascending=False)
                elif comparison_mode == "Same Athlete - Different Tests":
                    # Show all test types
                    athlete2_tests = filtered_df[filtered_df['Name'] == athlete2].sort_values('recordedDateUtc', ascending=False)
                elif comparison_mode == "Different Athletes - Same Test Type":
                    # Filter to same test type as trial 1
                    if test1_info:
                        athlete2_tests = filtered_df[
                            (filtered_df['Name'] == athlete2) &
                            (filtered_df['testType'] == test1_info[3])
                        ].sort_values('recordedDateUtc', ascending=False)
                    else:
                        athlete2_tests = filtered_df[filtered_df['Name'] == athlete2].sort_values('recordedDateUtc', ascending=False)
                else:  # Bilateral
                    athlete2_tests = filtered_df[filtered_df['Name'] == athlete2].sort_values('recordedDateUtc', ascending=False)

                if not athlete2_tests.empty:
                    test_options2 = []
                    for _, row in athlete2_tests.head(20).iterrows():
                        date_str = row['recordedDateUtc'].strftime('%Y-%m-%d') if pd.notna(row['recordedDateUtc']) else 'N/A'
                        test_label = f"{row['testType']} - {date_str}"
                        test_options2.append((test_label, row['testId'], row.get('trialId', ''), row['testType']))

                    # Filter out trial 1 if same athlete
                    if comparison_mode != "Different Athletes - Same Test Type":
                        test_options2 = [t for t in test_options2 if t[1] != test1_info[1]]

                    if test_options2:
                        test2_label = st.selectbox(
                            "Test 2:",
                            options=[t[0] for t in test_options2],
                            key="comp_test2"
                        )
                        test2_info = next((t for t in test_options2 if t[0] == test2_label), None)
                    else:
                        st.warning("No other tests available for comparison")
                        test2_info = None
                else:
                    st.warning("No tests available for this athlete")
                    test2_info = None

            with col2c:
                if test2_info:
                    st.markdown(f"**Test Type:** {test2_info[3]}")
                    st.markdown(f"**Test ID:** `{test2_info[1][:8]}...`")

            # Fetch and Compare Button
            if st.button("🔄 Fetch & Compare Trials", type="primary", key="compare_trials"):
                if not api_token_comp or not tenant_id_comp:
                    st.error("API credentials not configured. Please check .env file.")
                elif test1_info and test2_info:
                    with st.spinner("Fetching both force traces..."):
                        try:
                            from utils.force_trace_viz import get_force_trace

                            # Fetch Trial 1
                            trace1 = get_force_trace(
                                test_id=test1_info[1],
                                trial_id=test1_info[2] if test1_info[2] else test1_info[1],
                                token=api_token_comp,
                                tenant_id=tenant_id_comp,
                                region=region_comp
                            )

                            # Fetch Trial 2
                            trace2 = get_force_trace(
                                test_id=test2_info[1],
                                trial_id=test2_info[2] if test2_info[2] else test2_info[1],
                                token=api_token_comp,
                                tenant_id=tenant_id_comp,
                                region=region_comp
                            )

                            if not trace1.empty and not trace2.empty:
                                st.success(f"✅ Fetched {len(trace1)} + {len(trace2)} data points!")

                                # Store in session state
                                st.session_state['comparison_trace1'] = trace1
                                st.session_state['comparison_trace2'] = trace2
                                st.session_state['comparison_info'] = {
                                    'athlete1': athlete1,
                                    'athlete2': athlete2,
                                    'test1': test1_label,
                                    'test2': test2_label
                                }

                                # Create comparison visualization
                                st.markdown("---")
                                st.markdown("### 📊 Visual Comparison")

                                # Normalize time if requested
                                if normalize_time:
                                    trace1_norm = trace1.copy()
                                    trace2_norm = trace2.copy()

                                    # Normalize to 0-100% of movement
                                    trace1_norm['time_ms'] = (trace1_norm['time_ms'] - trace1_norm['time_ms'].min()) / (trace1_norm['time_ms'].max() - trace1_norm['time_ms'].min()) * 100
                                    trace2_norm['time_ms'] = (trace2_norm['time_ms'] - trace2_norm['time_ms'].min()) / (trace2_norm['time_ms'].max() - trace2_norm['time_ms'].min()) * 100
                                else:
                                    trace1_norm = trace1
                                    trace2_norm = trace2

                                # Create overlay plot
                                import plotly.graph_objects as go

                                fig_comp = go.Figure()

                                # Trial 1
                                fig_comp.add_trace(go.Scatter(
                                    x=trace1_norm['time_ms'] if 'time_ms' in trace1_norm.columns else trace1_norm.index,
                                    y=trace1_norm['force_n'] if 'force_n' in trace1_norm.columns else trace1_norm.iloc[:, 0],
                                    mode='lines',
                                    name=f"Trial 1: {athlete1}",
                                    line=dict(color='#1f77b4', width=2),
                                    hovertemplate='%{y:.0f} N<extra></extra>'
                                ))

                                # Trial 2
                                fig_comp.add_trace(go.Scatter(
                                    x=trace2_norm['time_ms'] if 'time_ms' in trace2_norm.columns else trace2_norm.index,
                                    y=trace2_norm['force_n'] if 'force_n' in trace2_norm.columns else trace2_norm.iloc[:, 0],
                                    mode='lines',
                                    name=f"Trial 2: {athlete2}",
                                    line=dict(color='#2ca02c', width=2),
                                    hovertemplate='%{y:.0f} N<extra></extra>'
                                ))

                                fig_comp.update_layout(
                                    title=f"Force Trace Comparison: {athlete1} vs {athlete2}",
                                    xaxis_title="Time (% of movement)" if normalize_time else "Time (ms)",
                                    yaxis_title="Force (N)",
                                    height=500,
                                    hovermode='x unified',
                                    template='plotly_white'
                                )

                                st.plotly_chart(fig_comp, use_container_width=True)

                                # Show difference plot if requested
                                if show_difference:
                                    st.markdown("#### Difference Analysis")

                                    # Interpolate to same time points for difference calculation
                                    from scipy import interpolate

                                    # Use Trial 1 time points as reference
                                    time_ref = trace1_norm['time_ms'].values if 'time_ms' in trace1_norm.columns else np.arange(len(trace1_norm))
                                    force1 = trace1_norm['force_n'].values if 'force_n' in trace1_norm.columns else trace1_norm.iloc[:, 0].values

                                    time2 = trace2_norm['time_ms'].values if 'time_ms' in trace2_norm.columns else np.arange(len(trace2_norm))
                                    force2 = trace2_norm['force_n'].values if 'force_n' in trace2_norm.columns else trace2_norm.iloc[:, 0].values

                                    # Interpolate trial 2 to match trial 1 time points
                                    f_interp = interpolate.interp1d(time2, force2, kind='linear', fill_value='extrapolate')
                                    force2_interp = f_interp(time_ref)

                                    # Calculate difference
                                    force_diff = force1 - force2_interp

                                    fig_diff = go.Figure()

                                    fig_diff.add_trace(go.Scatter(
                                        x=time_ref,
                                        y=force_diff,
                                        mode='lines',
                                        name='Difference (Trial 1 - Trial 2)',
                                        line=dict(color='#d62728', width=2),
                                        fill='tozeroy',
                                        fillcolor='rgba(214, 39, 40, 0.2)'
                                    ))

                                    fig_diff.add_hline(y=0, line_dash="dash", line_color="gray")

                                    fig_diff.update_layout(
                                        title="Force Difference (Trial 1 - Trial 2)",
                                        xaxis_title="Time (% of movement)" if normalize_time else "Time (ms)",
                                        yaxis_title="Force Difference (N)",
                                        height=350,
                                        template='plotly_white'
                                    )

                                    st.plotly_chart(fig_diff, use_container_width=True)

                                # Comparison Metrics
                                st.markdown("#### 📊 Comparison Metrics")

                                from utils.force_trace_viz import calculate_trace_metrics

                                metrics1 = calculate_trace_metrics(trace1, test_type=test1_info[3])
                                metrics2 = calculate_trace_metrics(trace2, test_type=test2_info[3])

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.markdown("**Peak Force**")
                                    peak1 = metrics1.get('peak_force', 0)
                                    peak2 = metrics2.get('peak_force', 0)
                                    diff_peak = ((peak1 - peak2) / peak2 * 100) if peak2 != 0 else 0
                                    st.metric("Trial 1", f"{peak1:.0f} N")
                                    st.metric("Trial 2", f"{peak2:.0f} N", delta=f"{diff_peak:+.1f}%")

                                with col2:
                                    st.markdown("**Average Force**")
                                    avg1 = metrics1.get('average_force', 0)
                                    avg2 = metrics2.get('average_force', 0)
                                    diff_avg = ((avg1 - avg2) / avg2 * 100) if avg2 != 0 else 0
                                    st.metric("Trial 1", f"{avg1:.0f} N")
                                    st.metric("Trial 2", f"{avg2:.0f} N", delta=f"{diff_avg:+.1f}%")

                                with col3:
                                    if 'impulse' in metrics1 and 'impulse' in metrics2:
                                        st.markdown("**Impulse**")
                                        imp1 = metrics1.get('impulse', 0)
                                        imp2 = metrics2.get('impulse', 0)
                                        diff_imp = ((imp1 - imp2) / imp2 * 100) if imp2 != 0 else 0
                                        st.metric("Trial 1", f"{imp1:.0f} N·s")
                                        st.metric("Trial 2", f"{imp2:.0f} N·s", delta=f"{diff_imp:+.1f}%")

                                # Export comparison data
                                st.markdown("---")
                                st.markdown("#### 💾 Export Comparison Data")

                                col1, col2 = st.columns(2)

                                with col1:
                                    csv1 = trace1.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Trial 1 CSV",
                                        data=csv1,
                                        file_name=f"trial1_{athlete1}_{test1_info[3]}.csv",
                                        mime="text/csv"
                                    )

                                with col2:
                                    csv2 = trace2.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Trial 2 CSV",
                                        data=csv2,
                                        file_name=f"trial2_{athlete2}_{test2_info[3]}.csv",
                                        mime="text/csv"
                                    )

                            else:
                                st.warning("One or both traces returned empty. Check API credentials and test IDs.")

                        except Exception as e:
                            st.error(f"Error during comparison: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.error("Please select both trials for comparison.")

            # Show previously compared traces if available
            if 'comparison_trace1' in st.session_state and 'comparison_trace2' in st.session_state:
                with st.expander("📋 View Saved Comparison"):
                    info = st.session_state.get('comparison_info', {})
                    st.markdown(f"**Trial 1:** {info.get('athlete1')} - {info.get('test1')}")
                    st.markdown(f"**Trial 2:** {info.get('athlete2')} - {info.get('test2')}")
                    st.markdown(f"**Data Points:** {len(st.session_state['comparison_trace1'])} + {len(st.session_state['comparison_trace2'])}")

        # General API Info
        with st.expander("📖 About Force Trace API"):
            st.markdown("""
            **API Endpoint:**
            ```
            GET /v2019q3/teams/{tenant}/tests/{testId}/trials/{trialId}/trace
            ```

            **Response Format:**
            - `time_ms`: Time in milliseconds
            - `force_n`: Total force in Newtons
            - `force_left_n`: Left force plate (bilateral tests)
            - `force_right_n`: Right force plate (bilateral tests)

            **Best Practices:**
            - Fetch traces only for tests you need to analyze in depth
            - Use test selection to minimize API calls
            - Cache results locally for repeated analysis

            Contact your VALD representative for API access credentials.
            """)

# ============================================================================
# TAB 12: ADVANCED ANALYTICS
# ============================================================================



with tabs[13]:
    st.markdown("## 🎯 Advanced Analytics")

    if not ADVANCED_VIZ_AVAILABLE:
        st.warning("""
        ### Advanced Visualization Module Not Available

        Please ensure `utils/advanced_viz.py` is in your dashboard directory.

        **Features available when enabled:**
        - Quadrant Analysis (2D performance mapping)
        - Parallel Coordinates (multi-metric profiles)
        - Violin Plots (distribution analysis)
        - Best-of-Day Trends
        - Reliability Analysis (CV%, Typical Error)
        """)
    else:
        # Create sub-tabs for different advanced analytics
        adv_subtabs = st.tabs([
            "📊 Quadrant Analysis",
            "📈 Parallel Coordinates",
            "🎻 Distribution Analysis",
            "📅 Best-of-Day",
            "🔬 Reliability"
        ])

        # ---- QUADRANT ANALYSIS ----
        with adv_subtabs[0]:
            st.markdown("### 📊 Quadrant Analysis")
            st.info("""
            **Quadrant Analysis** maps athletes on two performance dimensions to identify:
            - 🏆 **Elite** (Top-Right): High in both metrics
            - ⚠️ **Needs Focus** (Bottom-Left): Low in both metrics
            - 🔄 **Specialists**: High in one, low in other
            """)

            # Get available numeric columns for selection
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

            # Filter to meaningful metrics (exclude IDs, etc.)
            metric_options = get_performance_metric_columns(numeric_cols)

            if len(metric_options) >= 2:
                col1, col2 = st.columns(2)

                with col1:
                    x_metric = st.selectbox(
                        "X-Axis Metric:",
                        options=metric_options,
                        index=0,
                        key="quad_x"
                    )

                with col2:
                    y_metric = st.selectbox(
                        "Y-Axis Metric:",
                        options=metric_options,
                        index=min(1, len(metric_options)-1),
                        key="quad_y"
                    )

                if x_metric and y_metric:
                    fig_quad = create_quadrant_plot(
                        filtered_df,
                        x_metric=x_metric,
                        y_metric=y_metric,
                        athlete_column='Name',
                        sport_column='athlete_sport',
                        show_benchmarks=True,
                        color_by_sport=True
                    )

                    if fig_quad:
                        st.plotly_chart(fig_quad, use_container_width=True)

                        # Show quadrant summary
                        with st.expander("📋 Quadrant Summary"):
                            plot_cols = ['Name', x_metric, y_metric]
                            if 'athlete_sport' in filtered_df.columns:
                                plot_cols.insert(1, 'athlete_sport')
                            plot_df = filtered_df[plot_cols].dropna()
                            x_median = plot_df[x_metric].median()
                            y_median = plot_df[y_metric].median()

                            elite = plot_df[(plot_df[x_metric] > x_median) & (plot_df[y_metric] > y_median)]
                            needs_focus = plot_df[(plot_df[x_metric] < x_median) & (plot_df[y_metric] < y_median)]

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**🏆 Elite Performers:**")
                                if not elite.empty:
                                    for _, row in elite.head(10).iterrows():
                                        st.write(f"- {row['Name']} ({row.get('athlete_sport', 'N/A')})")
                                else:
                                    st.write("None identified")

                            with col2:
                                st.markdown("**⚠️ Needs Development:**")
                                if not needs_focus.empty:
                                    for _, row in needs_focus.head(10).iterrows():
                                        st.write(f"- {row['Name']} ({row.get('athlete_sport', 'N/A')})")
                                else:
                                    st.write("None identified")
            else:
                st.warning("Need at least 2 numeric metrics for quadrant analysis")

        # ---- PARALLEL COORDINATES ----
        with adv_subtabs[1]:
            st.markdown("### 📈 Parallel Coordinates")
            st.info("""
            **Parallel Coordinates** allow simultaneous comparison of multiple metrics.
            Perfect for identifying athlete profiles and patterns across dimensions.
            """)

            if len(metric_options) >= 3:
                selected_metrics = st.multiselect(
                    "Select Metrics (3-8 recommended):",
                    options=metric_options,
                    default=metric_options[:min(5, len(metric_options))],
                    key="parallel_metrics"
                )

                if len(selected_metrics) >= 3:
                    fig_parallel = create_parallel_coordinates(
                        filtered_df,
                        metrics=selected_metrics,
                        athlete_column='Name',
                        sport_column='athlete_sport',
                        color_by='sport',
                        title="Multi-Metric Athlete Profiles"
                    )

                    if fig_parallel:
                        st.plotly_chart(fig_parallel, use_container_width=True)

                        st.markdown("""
                        **How to Read:**
                        - Each vertical axis represents a metric
                        - Each line is an athlete
                        - Lines crossing at similar points indicate similar profiles
                        - Color indicates sport group
                        """)
                else:
                    st.warning("Select at least 3 metrics for parallel coordinates")
            else:
                st.warning("Need at least 3 numeric metrics for parallel coordinates")

        # ---- DISTRIBUTION ANALYSIS ----
        with adv_subtabs[2]:
            st.markdown("### 🎻 Distribution Analysis")
            st.info("""
            **Violin Plots** show the full distribution of metrics by group.
            Better than box plots for seeing bimodal distributions and density.
            """)

            if metric_options:
                col1, col2 = st.columns(2)

                with col1:
                    dist_metric = st.selectbox(
                        "Select Metric:",
                        options=metric_options,
                        key="dist_metric"
                    )

                with col2:
                    # Build group_by options based on available columns
                    group_options = []
                    if 'athlete_sport' in filtered_df.columns:
                        group_options.append('athlete_sport')
                    if 'testType' in filtered_df.columns:
                        group_options.append('testType')
                    if not group_options:
                        group_options = ['testType']  # fallback
                    group_by = st.selectbox(
                        "Group By:",
                        options=group_options,
                        key="dist_group"
                    )

                if dist_metric:
                    fig_violin = create_violin_plot(
                        filtered_df,
                        metric=dist_metric,
                        group_by=group_by,
                        show_box=True,
                        show_points=True
                    )

                    if fig_violin:
                        st.plotly_chart(fig_violin, use_container_width=True)

        # ---- BEST OF DAY ----
        with adv_subtabs[3]:
            st.markdown("### 📅 Best-of-Day Trends")
            st.info("""
            **Best-of-Day Analysis** tracks peak performance trends by:
            - Filtering out warm-up trials
            - Focusing on maximal efforts
            - Identifying performance progressions
            """)

            if 'recordedDateUtc' in filtered_df.columns and metric_options:
                col1, col2 = st.columns(2)

                with col1:
                    bod_metric = st.selectbox(
                        "Select Metric:",
                        options=metric_options,
                        key="bod_metric"
                    )

                with col2:
                    bod_athletes = st.multiselect(
                        "Select Athletes (up to 5):",
                        options=sorted(filtered_df['Name'].unique()),
                        max_selections=5,
                        key="bod_athletes"
                    )

                if bod_metric and len(bod_athletes) >= 1:
                    fig_bod = create_best_of_day_trend(
                        filtered_df,
                        athletes=bod_athletes,
                        metric=bod_metric,
                        date_column='recordedDateUtc',
                        athlete_column='Name',
                        higher_is_better=True
                    )

                    if fig_bod:
                        st.plotly_chart(fig_bod, use_container_width=True)
                else:
                    st.info("Select at least 1 athlete to view best-of-day trends")
            else:
                st.warning("Date column required for best-of-day analysis")

        # ---- RELIABILITY ANALYSIS ----
        with adv_subtabs[4]:
            st.markdown("### 🔬 Reliability Analysis")
            st.info("""
            **Test-Retest Reliability** helps you understand:
            - **CV% (Coefficient of Variation)**: Measurement consistency
            - **Typical Error**: Expected variation between tests

            **Interpretation:**
            - CV% < 10%: Good reliability
            - CV% 10-15%: Moderate reliability
            - CV% > 15%: Poor reliability (consider standardizing protocol)
            """)

            if metric_options:
                # Calculate reliability for selected metrics
                selected_reliability_metrics = st.multiselect(
                    "Select Metrics for Reliability Analysis:",
                    options=metric_options,
                    default=metric_options[:min(5, len(metric_options))],
                    key="reliability_metrics"
                )

                if selected_reliability_metrics:
                    reliability_df = calculate_reliability_metrics(
                        filtered_df,
                        athlete_column='Name',
                        metrics=selected_reliability_metrics
                    )

                    if not reliability_df.empty:
                        # Display reliability table
                        st.markdown("#### Reliability Summary")

                        display_df = reliability_df[[
                            'metric', 'n_athletes', 'median_cv_percent',
                            'mean_te_percent', 'reliability_rating'
                        ]].rename(columns={
                            'metric': 'Metric',
                            'n_athletes': 'Athletes (n)',
                            'median_cv_percent': 'CV% (Median)',
                            'mean_te_percent': 'TE% (Mean)',
                            'reliability_rating': 'Rating'
                        })

                        st.dataframe(
                            display_df.round(2),
                            use_container_width=True,
                            hide_index=True
                        )

                        # Visual
                        fig_reliability = create_reliability_plot(reliability_df)
                        if fig_reliability:
                            st.plotly_chart(fig_reliability, use_container_width=True)

                        # Interpretation
                        with st.expander("📖 Interpreting Reliability Metrics"):
                            st.markdown("""
                            **Coefficient of Variation (CV%)**
                            - Measures relative variability: `(SD / Mean) × 100`
                            - Lower is better (more consistent)
                            - Athletes with high CV% may need technique refinement

                            **Typical Error (TE)**
                            - Expected variation between tests: `SD / √2`
                            - Helps distinguish real change from noise
                            - A change > 2×TE is likely meaningful

                            **Recommendations:**
                            - If CV% > 15%: Review test protocol standardization
                            - High CV% in specific athletes: Address technique consistency
                            - Use TE to set meaningful change thresholds
                            """)
                    else:
                        st.warning("Need athletes with multiple tests to calculate reliability")
                else:
                    st.info("Select metrics for reliability analysis")
            else:
                st.warning("No numeric metrics available for reliability analysis")

# ============================================================================
# TAB 13: ELITE INSIGHTS (ORIGINAL)
# ============================================================================



with tabs[14]:
    st.markdown("## ⭐ Elite Insights - Advanced Analysis")
    st.markdown("Research-backed analysis from Patrick Ward (Nike), mattsams89, and elite sports science")

    if not ELITE_INSIGHTS_AVAILABLE:
        st.warning("⚠️ Elite Insights module not available. Check advanced_analysis.py installation.")
    elif filtered_df.empty:
        st.info("No data available. Please adjust filters.")
    else:
        # Sub-tabs for different analyses
        insight_tabs = st.tabs([
            "🔄 Asymmetry Analysis",
            "📊 Meaningful Change",
            "📏 Normative Benchmarks",
            "📉 Reliability (TEM)",
            "🎯 Z-Score Outliers"
        ])

        # ========== Asymmetry Analysis ==========
        with insight_tabs[0]:
            st.markdown("### Bilateral Asymmetry Analysis")
            st.markdown("**Threshold**: >15% requires intervention (VALD/NordBord standard)")

            # Select athlete
            athletes = sorted(filtered_df['Name'].dropna().unique())
            if athletes:
                selected_athlete = st.selectbox("Select Athlete:", athletes, key="asym_athlete")

                athlete_df = filtered_df[filtered_df['Name'] == selected_athlete].copy()

                # Look for bilateral metrics
                left_metrics = [col for col in athlete_df.columns if 'left' in col.lower() and 'force' in col.lower()]
                right_metrics = [col for col in athlete_df.columns if 'right' in col.lower() and 'force' in col.lower()]

                if left_metrics and right_metrics:
                    col1, col2 = st.columns(2)

                    with col1:
                        left_metric = st.selectbox("Left Metric:", left_metrics)

                    with col2:
                        right_metric = st.selectbox("Right Metric:", right_metrics)

                    # Calculate asymmetry for each test
                    asymmetry_data = []
                    for idx, row in athlete_df.iterrows():
                        left_val = row.get(left_metric)
                        right_val = row.get(right_metric)

                        if pd.notna(left_val) and pd.notna(right_val):
                            asym = calculate_asymmetry(left_val, right_val)
                            asymmetry_data.append({
                                'recordedDateUtc': row['recordedDateUtc'],
                                'testType': row['testType'],
                                'left_value': left_val,
                                'right_value': right_val,
                                **asym
                            })

                    if asymmetry_data:
                        asym_df = pd.DataFrame(asymmetry_data)

                        # Display current asymmetry status
                        latest = asym_df.iloc[-1]
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Latest Asymmetry", f"{latest['asymmetry_percent']:.1f}%",
                                     delta=None)

                        with col2:
                            st.metric("Dominant Side", latest['dominant_side'])

                        with col3:
                            status_color = "🔴" if latest['flag'] else "🟢"
                            st.metric("Status", f"{status_color} {latest['magnitude']}")

                        with col4:
                            avg_asym = asym_df['asymmetry_percent'].mean()
                            st.metric("Average", f"{avg_asym:.1f}%")

                        # Circle plot visualization
                        st.plotly_chart(
                            create_asymmetry_circle_plot(asym_df, selected_athlete),
                            use_container_width=True
                        )

                        # Data table
                        st.markdown("#### Test History")
                        display_df = asym_df[['recordedDateUtc', 'testType', 'asymmetry_percent',
                                              'dominant_side', 'magnitude']].copy()
                        display_df['recordedDateUtc'] = pd.to_datetime(display_df['recordedDateUtc']).dt.strftime('%Y-%m-%d')
                        # Already showing only performance columns
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No bilateral data available for selected metrics")
                else:
                    st.warning("No bilateral (left/right) force metrics found in data")
            else:
                st.info("No athletes available")

        # ========== Meaningful Change Detection ==========
        with insight_tabs[1]:
            st.markdown("### Meaningful Change Detection")
            st.markdown("**Method**: 0.6 × SD threshold (Patrick Ward - Nike Sports Research)")
            st.markdown("Uses *full dataset* SD to maintain context")

            athletes = sorted(filtered_df['Name'].dropna().unique())
            if athletes:
                selected_athlete = st.selectbox("Select Athlete:", athletes, key="change_athlete")
                athlete_df = filtered_df[filtered_df['Name'] == selected_athlete].copy().sort_values('recordedDateUtc')

                # Select metric
                numeric_cols = athlete_df.select_dtypes(include=[np.number]).columns.tolist()
                metric_cols = get_performance_metric_columns(numeric_cols)

                if metric_cols:
                    selected_metric = st.selectbox("Select Metric:", metric_cols, key="change_metric")

                    if len(athlete_df) >= 2:
                        # Calculate group SD from full dataset
                        sport = athlete_df['athlete_sport'].iloc[0] if 'athlete_sport' in athlete_df.columns else None
                        if sport and pd.notna(sport):
                            sport_df = filtered_df[filtered_df['athlete_sport'] == sport] if 'athlete_sport' in filtered_df.columns else filtered_df
                        else:
                            sport_df = filtered_df

                        group_sd = sport_df[selected_metric].std()

                        # Show plot
                        fig = create_meaningful_change_plot(athlete_df, selected_metric, sport)
                        st.plotly_chart(fig, use_container_width=True)

                        # Calculate changes
                        changes_data = []
                        for i in range(1, len(athlete_df)):
                            change = calculate_meaningful_change(
                                athlete_df.iloc[i][selected_metric],
                                athlete_df.iloc[i-1][selected_metric],
                                group_sd
                            )
                            changes_data.append({
                                'Date': athlete_df.iloc[i]['recordedDateUtc'].strftime('%Y-%m-%d'),
                                'Value': athlete_df.iloc[i][selected_metric],
                                'Change': f"{change['direction_symbol']} {abs(change['change_value']):.2f}",
                                'Change %': f"{change['change_percent']:.1f}%",
                                'Meaningful': '✓' if change['is_meaningful'] else '✗',
                                'Magnitude': change['magnitude']
                            })

                        if changes_data:
                            st.markdown("#### Change History")
                            # Already showing only performance columns - no technical IDs
                            st.dataframe(pd.DataFrame(changes_data), use_container_width=True, hide_index=True)

                            # Summary stats
                            meaningful_count = sum(1 for c in changes_data if c['Meaningful'] == '✓')
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Total Changes", len(changes_data))
                            with col2:
                                st.metric("Meaningful Changes", meaningful_count)
                            with col3:
                                st.metric("Group SD", f"{group_sd:.2f}")
                    else:
                        st.info("Need at least 2 tests for change detection")
                else:
                    st.warning("No numeric metrics available")
            else:
                st.info("No athletes available")

        # ========== Normative Benchmarking ==========
        with insight_tabs[2]:
            st.markdown("### Normative Benchmarking")
            st.markdown("**Percentile Zones**: <25% (Below Avg) | 25-50% (Average) | 50-75% (Good) | >75% (Excellent)")

            athletes = sorted(filtered_df['Name'].dropna().unique())
            if athletes:
                selected_athlete = st.selectbox("Select Athlete:", athletes, key="norm_athlete")
                athlete_df = filtered_df[filtered_df['Name'] == selected_athlete].copy()

                # Select sport for comparison
                sport = athlete_df['athlete_sport'].iloc[0] if 'athlete_sport' in athlete_df.columns and len(athlete_df) > 0 else None

                if sport and pd.notna(sport):
                    reference_df = filtered_df[filtered_df['athlete_sport'] == sport]

                    # Select metric
                    numeric_cols = athlete_df.select_dtypes(include=[np.number]).columns.tolist()
                    metric_cols = get_performance_metric_columns(numeric_cols)

                    if metric_cols:
                        selected_metric = st.selectbox("Select Metric:", metric_cols, key="norm_metric")

                        latest_value = athlete_df.sort_values('recordedDateUtc')[selected_metric].iloc[-1]

                        if pd.notna(latest_value):
                            # Calculate percentile
                            reference_values = reference_df[selected_metric].dropna().values

                            if len(reference_values) > 0:
                                rank = calculate_percentile_rank(latest_value, list(reference_values))

                                # Display metrics
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric("Athlete Value", f"{latest_value:.2f}")

                                with col2:
                                    st.metric("Percentile Rank", f"{rank['percentile']:.0f}th",
                                             help="Position within sport group")

                                with col3:
                                    zone_emoji = "🟢" if rank['zone'] == 'Excellent' else "🟡" if rank['zone'] in ['Good', 'Average'] else "🔴"
                                    st.metric("Zone", f"{zone_emoji} {rank['zone']}")

                                # Show distribution plot
                                fig = create_normative_benchmark_plot(latest_value, reference_df, selected_metric, sport)
                                st.plotly_chart(fig, use_container_width=True)

                                # Sport statistics
                                st.markdown(f"#### {sport} Statistics")
                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    st.metric("Mean", f"{reference_values.mean():.2f}")
                                with col2:
                                    st.metric("Median", f"{np.median(reference_values):.2f}")
                                with col3:
                                    st.metric("SD", f"{np.std(reference_values):.2f}")
                                with col4:
                                    st.metric("N", len(reference_values))
                            else:
                                st.warning("No reference data available for comparison")
                        else:
                            st.info("No data available for selected metric")
                    else:
                        st.warning("No numeric metrics available")
                else:
                    st.warning("Athlete sport not specified")
            else:
                st.info("No athletes available")

        # ========== Reliability Analysis (TEM) ==========
        with insight_tabs[3]:
            st.markdown("### Test Reliability Analysis")
            st.markdown("**TEM** (Typical Error Measurement) = SD(differences) / √2")
            st.markdown("**Reliability**: <5% Excellent | 5-10% Good | 10-15% Moderate | >15% Poor")

            # Select test type and metric
            test_types = sorted(filtered_df['testType'].dropna().unique())

            if test_types:
                selected_test = st.selectbox("Select Test Type:", test_types, key="tem_test")
                test_df = filtered_df[filtered_df['testType'] == selected_test].copy()

                # Select metric
                numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
                metric_cols = get_performance_metric_columns(numeric_cols)

                if metric_cols:
                    selected_metric = st.selectbox("Select Metric:", metric_cols, key="tem_metric")

                    tem_results = calculate_tem_with_ci(test_df, selected_metric)

                    if tem_results['tem'] is not None:
                        # Display TEM metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("TEM", f"{tem_results['tem']:.2f}")

                        with col2:
                            st.metric("TEM %", f"{tem_results['tem_percent']:.1f}%")

                        with col3:
                            reliability_emoji = "🟢" if tem_results['reliability'] == 'Excellent' else "🟡" if tem_results['reliability'] in ['Good', 'Moderate'] else "🔴"
                            st.metric("Reliability", f"{reliability_emoji} {tem_results['reliability']}")

                        with col4:
                            st.metric("95% CI", f"[{tem_results['ci_lower']:.2f}, {tem_results['ci_upper']:.2f}]")

                        # Interpretation
                        st.info(f"""
                        **Interpretation**: The typical error for {selected_metric} in {selected_test} tests is {tem_results['tem']:.2f}
                        ({tem_results['tem_percent']:.1f}% of mean), indicating **{tem_results['reliability']}** test reliability.

                        This means changes greater than {tem_results['tem'] * 1.96:.2f} (TEM × 1.96) can be considered real changes beyond measurement error.
                        """)

                        # Show data distribution
                        values = test_df[selected_metric].dropna()
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=values, nbinsx=30, name=selected_metric))
                        fig.update_layout(
                            title=f"{selected_metric} Distribution - {selected_test}",
                            xaxis_title=selected_metric,
                            yaxis_title="Frequency",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Insufficient data to calculate TEM (need at least 2 measurements)")
                else:
                    st.warning("No numeric metrics available")
            else:
                st.info("No test types available")

        # ========== Z-Score Outlier Detection ==========
        with insight_tabs[4]:
            st.markdown("### Z-Score Outlier Detection")
            st.markdown("**Method**: Z-scores calculated on full dataset before filtering (Patrick Ward approach)")
            st.markdown("**Thresholds**: |Z| > 2 (Very high/low) | |Z| > 1 (High/low)")

            # Select metric
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            metric_cols = get_performance_metric_columns(numeric_cols)

            if metric_cols:
                selected_metric = st.selectbox("Select Metric:", metric_cols, key="zscore_metric")

                # Calculate z-scores on full dataset
                df_with_z = calculate_zscore_with_context(df.copy(), selected_metric)

                # Show distribution
                fig = create_zscore_distribution_plot(df_with_z, selected_metric)
                st.plotly_chart(fig, use_container_width=True)

                # Filter to current selection
                filtered_with_z = df_with_z[df_with_z.index.isin(filtered_df.index)].copy()

                # Show outliers
                outliers = filtered_with_z[filtered_with_z['z_flag']].copy()

                if not outliers.empty:
                    st.markdown(f"#### Outliers Detected ({len(outliers)} tests)")

                    # Build display columns conditionally - handle missing columns
                    display_cols = ['Name']
                    if 'athlete_sport' in outliers.columns:
                        display_cols.append('athlete_sport')
                    if 'testType' in outliers.columns:
                        display_cols.append('testType')
                    display_cols.extend(['recordedDateUtc', selected_metric, 'z_score', 'z_interpretation'])
                    # Only include columns that exist
                    display_cols = [c for c in display_cols if c in outliers.columns]
                    outliers_display = outliers[display_cols].copy()
                    if 'recordedDateUtc' in outliers_display.columns:
                        outliers_display['recordedDateUtc'] = pd.to_datetime(outliers_display['recordedDateUtc']).dt.strftime('%Y-%m-%d')
                    outliers_display['z_score'] = outliers_display['z_score'].round(2)
                    outliers_display[selected_metric] = outliers_display[selected_metric].round(2)

                    st.dataframe(outliers_display.sort_values('z_score', ascending=False),
                                use_container_width=True, hide_index=True)

                    # Summary
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        very_high = len(outliers[outliers['z_score'] > 2])
                        st.metric("Very High (Z>2)", very_high)

                    with col2:
                        very_low = len(outliers[outliers['z_score'] < -2])
                        st.metric("Very Low (Z<-2)", very_low)

                    with col3:
                        outlier_rate = (len(outliers) / len(filtered_with_z)) * 100
                        st.metric("Outlier Rate", f"{outlier_rate:.1f}%")
                else:
                    st.success("✓ No outliers detected in current selection")

                # Dataset stats
                st.markdown("#### Dataset Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Mean (Full Dataset)", f"{df_with_z[selected_metric].mean():.2f}")
                with col2:
                    st.metric("SD (Full Dataset)", f"{df_with_z[selected_metric].std():.2f}")
                with col3:
                    st.metric("N (Full Dataset)", len(df_with_z))
                with col4:
                    st.metric("N (Filtered)", len(filtered_with_z))
            else:
                st.warning("No numeric metrics available")

# ============================================================================
# TAB 14: DATA TABLE
# ============================================================================



with tabs[15]:
    st.markdown("## 📋 Data Table View")
    st.markdown("Browse and export all testing data in table format")

    # Filters for data table
    col1, col2, col3 = st.columns(3)

    with col1:
        # Test type filter
        test_types = ['All'] + sorted(filtered_df['testType'].unique().tolist()) if 'testType' in filtered_df.columns else ['All']
        selected_test_type = st.selectbox("Test Type:", test_types, key="table_test_type")

    with col2:
        # Athlete filter
        athletes = ['All'] + sorted(filtered_df['Name'].unique().tolist()) if 'Name' in filtered_df.columns else ['All']
        selected_athlete_table = st.selectbox("Athlete:", athletes, key="table_athlete")

    with col3:
        # Sport filter
        sports = ['All'] + sorted(filtered_df['athlete_sport'].unique().tolist()) if 'athlete_sport' in filtered_df.columns else ['All']
        selected_sport_table = st.selectbox("Sport:", sports, key="table_sport")

    # Apply filters
    table_df = filtered_df.copy()

    if selected_test_type != 'All':
        table_df = table_df[table_df['testType'] == selected_test_type]

    if selected_athlete_table != 'All':
        table_df = table_df[table_df['Name'] == selected_athlete_table]

    if selected_sport_table != 'All' and 'athlete_sport' in table_df.columns:
        table_df = table_df[table_df['athlete_sport'] == selected_sport_table]

    # Column selection
    st.markdown("### Select Columns to Display")

    # Filter to performance columns only
    filtered_table_df = filter_performance_columns(table_df)

    # Get all column names (already filtered)
    all_columns = filtered_table_df.columns.tolist()

    # Default important columns
    default_cols = ['Name', 'athlete_sport', 'testType', 'recordedDateUtc']
    default_cols = [col for col in default_cols if col in all_columns]

    # Add metric columns that exist
    metric_keywords = ['Peak', 'Jump', 'Force', 'Power', 'Height', 'Depth', 'Time', 'RFD']
    metric_cols = [col for col in all_columns if any(keyword in col for keyword in metric_keywords)]

    suggested_cols = default_cols + metric_cols[:10]  # Default + first 10 metric columns
    suggested_cols = list(dict.fromkeys(suggested_cols))  # Remove duplicates while preserving order

    selected_columns = st.multiselect(
        "Choose columns:",
        options=all_columns,
        default=suggested_cols,
        key="table_columns"
    )

    if not selected_columns:
        st.warning("Please select at least one column to display")
    else:
        # Display table
        display_df = filtered_table_df[selected_columns].copy()

        # Format date column if it exists
        if 'recordedDateUtc' in display_df.columns:
            display_df['recordedDateUtc'] = pd.to_datetime(display_df['recordedDateUtc']).dt.strftime('%Y-%m-%d %H:%M')

        st.markdown(f"### Data Table ({len(display_df)} rows)")

        # Display dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            height=600,
            hide_index=True
        )

        # Download button
        st.markdown("### Export Data")

        col1, col2 = st.columns(2)

        with col1:
            # CSV export
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"vald_data_{selected_test_type}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        with col2:
            # Summary stats
            st.metric("Total Rows", len(display_df))

        # Quick stats
        if len(display_df) > 0:
            st.markdown("### Quick Statistics")

            stat_col1, stat_col2, stat_col3 = st.columns(3)

            with stat_col1:
                if 'Name' in display_df.columns:
                    st.metric("Unique Athletes", display_df['Name'].nunique())

            with stat_col2:
                if 'testType' in display_df.columns:
                    st.metric("Test Types", display_df['testType'].nunique())

            with stat_col3:
                if 'athlete_sport' in display_df.columns:
                    st.metric("Sports", display_df['athlete_sport'].nunique())

# ============================================================================
# PAGE: FORCEFRAME
# ============================================================================

with tabs[1]:
    st.markdown("## 🔲 ForceFrame Analysis")
    st.markdown("*Isometric strength testing across multiple joint positions*")

    if df_forceframe.empty:
        st.warning("No ForceFrame data available. Upload data or check data/forceframe_allsports.csv")
    else:
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            n_athletes = df_forceframe['athleteId'].nunique() if 'athleteId' in df_forceframe.columns else 0
            st.metric("Athletes Tested", n_athletes)

        with col2:
            st.metric("Total Tests", len(df_forceframe))

        with col3:
            n_types = df_forceframe['testTypeName'].nunique() if 'testTypeName' in df_forceframe.columns else 0
            st.metric("Test Types", n_types)

        with col4:
            if 'testDateUtc' in df_forceframe.columns:
                latest = df_forceframe['testDateUtc'].max()
                st.metric("Latest Test", latest.strftime('%Y-%m-%d') if pd.notna(latest) else "N/A")

        st.markdown("---")

        # Test type distribution
        if 'testTypeName' in df_forceframe.columns:
            st.markdown("### Test Type Distribution")
            test_counts = df_forceframe['testTypeName'].value_counts()

            fig = px.bar(
                x=test_counts.index,
                y=test_counts.values,
                labels={'x': 'Test Type', 'y': 'Count'},
                color=test_counts.values,
                color_continuous_scale=['#1D4D3B', '#a08e66']
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Force comparison
        st.markdown("### Left vs Right Force Comparison")

        left_col = 'innerLeftMaxForce' if 'innerLeftMaxForce' in df_forceframe.columns else None
        right_col = 'innerRightMaxForce' if 'innerRightMaxForce' in df_forceframe.columns else None

        if left_col and right_col:
            plot_df = df_forceframe[[left_col, right_col]].dropna()

            if len(plot_df) > 0:
                fig = px.scatter(
                    plot_df,
                    x=left_col,
                    y=right_col,
                    labels={left_col: 'Left Max Force (N)', right_col: 'Right Max Force (N)'}
                )
                max_val = max(plot_df[left_col].max(), plot_df[right_col].max())
                fig.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                             line=dict(color='red', dash='dash'))
                fig.update_traces(marker=dict(color='#1D4D3B', size=10))
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.markdown("### ForceFrame Data Table")
        display_cols = ['testDateUtc', 'testTypeName', 'innerLeftMaxForce', 'innerRightMaxForce',
                       'outerLeftMaxForce', 'outerRightMaxForce']
        available_cols = [c for c in display_cols if c in df_forceframe.columns]

        if available_cols:
            display_ff = df_forceframe[available_cols].copy()
            if 'testDateUtc' in display_ff.columns:
                display_ff['testDateUtc'] = pd.to_datetime(display_ff['testDateUtc']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(display_ff, use_container_width=True, height=400)

        csv_ff = df_forceframe.to_csv(index=False)
        st.download_button("📥 Download ForceFrame Data", csv_ff,
                          f"forceframe_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

# ============================================================================
# PAGE: NORDBORD
# ============================================================================

with tabs[2]:
    st.markdown("## 🦵 NordBord Analysis")
    st.markdown("*Nordic hamstring strength and asymmetry assessment*")

    if df_nordbord.empty:
        st.warning("No NordBord data available. Upload data or check data/nordbord_allsports.csv")
    else:
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            n_athletes = df_nordbord['athleteId'].nunique() if 'athleteId' in df_nordbord.columns else 0
            st.metric("Athletes Tested", n_athletes)

        with col2:
            st.metric("Total Tests", len(df_nordbord))

        with col3:
            if 'testTypeName' in df_nordbord.columns:
                test_types = df_nordbord['testTypeName'].unique()
                st.metric("Test Type", test_types[0] if len(test_types) > 0 else "N/A")

        with col4:
            if 'testDateUtc' in df_nordbord.columns:
                latest = df_nordbord['testDateUtc'].max()
                st.metric("Latest Test", latest.strftime('%Y-%m-%d') if pd.notna(latest) else "N/A")

        st.markdown("---")

        # Left vs Right scatter
        st.markdown("### Left vs Right Hamstring Force")

        if 'leftMaxForce' in df_nordbord.columns and 'rightMaxForce' in df_nordbord.columns:
            plot_df = df_nordbord[['leftMaxForce', 'rightMaxForce', 'testDateUtc']].dropna()

            if len(plot_df) > 0:
                fig = px.scatter(
                    plot_df, x='leftMaxForce', y='rightMaxForce',
                    labels={'leftMaxForce': 'Left Max Force (N)', 'rightMaxForce': 'Right Max Force (N)'}
                )
                max_val = max(plot_df['leftMaxForce'].max(), plot_df['rightMaxForce'].max())
                fig.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                             line=dict(color='red', dash='dash'))
                fig.update_traces(marker=dict(color='#1D4D3B', size=12))
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Asymmetry histogram
                plot_df['asymmetry'] = ((plot_df['rightMaxForce'] - plot_df['leftMaxForce']) /
                                        ((plot_df['rightMaxForce'] + plot_df['leftMaxForce']) / 2) * 100)

                st.markdown("### Asymmetry Distribution")
                fig_asym = px.histogram(plot_df, x='asymmetry', nbins=20,
                                       labels={'asymmetry': 'Asymmetry (%)'},
                                       color_discrete_sequence=['#1D4D3B'])
                fig_asym.add_vline(x=0, line_dash='dash', line_color='red')
                fig_asym.add_vline(x=10, line_dash='dot', line_color='orange', annotation_text='10%')
                fig_asym.add_vline(x=-10, line_dash='dot', line_color='orange')
                fig_asym.update_layout(height=400)
                st.plotly_chart(fig_asym, use_container_width=True)

        # Time series
        st.markdown("### Force Trend Over Time")

        if 'testDateUtc' in df_nordbord.columns and 'leftMaxForce' in df_nordbord.columns:
            time_df = df_nordbord.sort_values('testDateUtc')

            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=time_df['testDateUtc'], y=time_df['leftMaxForce'],
                name='Left', mode='markers+lines',
                marker=dict(color='#1D4D3B', size=8), line=dict(color='#1D4D3B')
            ))
            fig_time.add_trace(go.Scatter(
                x=time_df['testDateUtc'], y=time_df['rightMaxForce'],
                name='Right', mode='markers+lines',
                marker=dict(color='#a08e66', size=8), line=dict(color='#a08e66')
            ))
            fig_time.update_layout(
                xaxis_title='Date', yaxis_title='Max Force (N)', height=400,
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig_time, use_container_width=True)

        # Data table
        st.markdown("### NordBord Data Table")
        display_cols = ['testDateUtc', 'testTypeName', 'leftMaxForce', 'rightMaxForce',
                       'leftAvgForce', 'rightAvgForce', 'leftTorque', 'rightTorque', 'device']
        available_cols = [c for c in display_cols if c in df_nordbord.columns]

        if available_cols:
            display_nb = df_nordbord[available_cols].copy()
            if 'testDateUtc' in display_nb.columns:
                display_nb['testDateUtc'] = pd.to_datetime(display_nb['testDateUtc']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(display_nb, use_container_width=True, height=400)

        csv_nb = df_nordbord.to_csv(index=False)
        st.download_button("📥 Download NordBord Data", csv_nb,
                          f"nordbord_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
---
<div style="text-align: center; color: #ffffff; padding: 3rem 2rem; background: linear-gradient(135deg, #1D4D3B 0%, #153829 50%, #0F2A1E 100%); border-radius: 12px; margin-top: 3rem; position: relative; overflow: hidden; box-shadow: 0 8px 32px rgba(29, 77, 59, 0.3);">
    <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, #a08e66, #9d8e65, #a08e66);"></div>
    <h3 style="font-family: 'Poppins', sans-serif; font-size: 1.8rem; margin-bottom: 0.5rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase;">TEAM SAUDI</h3>
    <p style="font-size: 1.1rem; font-family: 'Tajawal', sans-serif; opacity: 0.95; font-weight: 500; margin-bottom: 0.5rem;">اللجنة الأولمبية والبارالمبية السعودية</p>
    <div style="background: rgba(160, 142, 102, 0.5); height: 1px; width: 200px; margin: 1rem auto;"></div>
    <p style="font-size: 0.95rem; opacity: 0.9; font-weight: 600;">Performance Analysis Dashboard v3.0</p>
    <p style="font-size: 0.85rem; opacity: 0.75; margin-top: 0.5rem;">Force Trace Analysis | Advanced Analytics | Reliability Metrics</p>
    <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1);">
        <p style="font-size: 0.75rem; opacity: 0.6; font-family: 'Roboto', sans-serif;">© 2025 Saudi Olympic & Paralympic Committee</p>
    </div>
</div>
""", unsafe_allow_html=True)
