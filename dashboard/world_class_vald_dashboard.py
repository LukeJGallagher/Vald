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
        # Primary source: vald-data repo
        vald_data_dir = r"c:\Users\l.gallagher\OneDrive - Team Saudi\Documents\Performance Analysis\vald-data\data"
        file_paths = [
            os.path.join(vald_data_dir, f'{device}_allsports_with_athletes.csv'),
            os.path.join(vald_data_dir, f'{device}_allsports.csv'),
            f'{device}_allsports_with_athletes.csv',
            f'data/{device}_allsports_with_athletes.csv',
            f'data/master_files/{device}_allsports_with_athletes.csv',
        ]

        for file_path in file_paths:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, low_memory=False)
                if 'recordedDateUtc' in df.columns:
                    df['recordedDateUtc'] = pd.to_datetime(df['recordedDateUtc'])
                if 'testDateUtc' in df.columns:
                    df['testDateUtc'] = pd.to_datetime(df['testDateUtc'], errors='coerce')
                # Create Name column from full_name if missing
                if 'Name' not in df.columns:
                    if 'full_name' in df.columns:
                        df['Name'] = df['full_name']
                    elif 'athleteId' in df.columns:
                        df['Name'] = df['athleteId'].apply(lambda x: f"Athlete_{str(x)[:8]}" if pd.notna(x) else "Unknown")
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
            # For any missing names, try mapping or use profileId/athleteId
            id_col = 'profileId' if 'profileId' in df.columns else ('athleteId' if 'athleteId' in df.columns else ('athlete_id' if 'athlete_id' in df.columns else None))
            if id_col:
                def get_name(row):
                    if pd.notna(row.get('full_name')) and str(row.get('full_name')).strip():
                        return str(row['full_name']).strip()
                    aid = str(row.get(id_col, ''))
                    if athlete_mapping and aid in athlete_mapping:
                        return athlete_mapping[aid]
                    return f"Athlete_{aid[:8]}" if aid else 'Unknown'
                df['Name'] = df.apply(get_name, axis=1)
            else:
                df['Name'] = df['full_name'].fillna('Unknown')
        # Priority 2: Use athleteId with mapping (ForceFrame/NordBord)
        elif 'athleteId' in df.columns:
            if athlete_mapping:
                df['Name'] = df['athleteId'].apply(lambda x: athlete_mapping.get(str(x), f"Athlete_{str(x)[:8]}"))
            else:
                df['Name'] = df['athleteId'].apply(lambda x: f"Athlete_{str(x)[:8]}")
        # Priority 3: Use profileId with mapping (ForceDecks)
        elif 'profileId' in df.columns:
            if athlete_mapping:
                df['Name'] = df['profileId'].apply(lambda x: athlete_mapping.get(str(x), f"Athlete_{str(x)[:8]}"))
            else:
                df['Name'] = df['profileId'].apply(lambda x: f"Athlete_{str(x)[:8]}")
        # Priority 4: Use athlete_id with mapping
        elif 'athlete_id' in df.columns:
            if athlete_mapping:
                df['Name'] = df['athlete_id'].apply(lambda x: athlete_mapping.get(str(x), f"Athlete_{str(x)[:8]}"))
            else:
                df['Name'] = df['athlete_id'].apply(lambda x: f"Athlete_{str(x)[:8]}")
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

    If MANUAL_TOKEN is not provided but CLIENT_ID and CLIENT_SECRET are,
    automatically fetches an OAuth token from the VALD API.
    """
    credentials = {
        'token': '',
        'tenant_id': '',
        'client_id': '',
        'client_secret': '',
        'region': 'euw'
    }

    found = False

    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        if hasattr(st, 'secrets') and 'vald' in st.secrets:
            credentials['token'] = st.secrets['vald'].get('MANUAL_TOKEN', '')
            credentials['tenant_id'] = st.secrets['vald'].get('TENANT_ID', '')
            credentials['client_id'] = st.secrets['vald'].get('CLIENT_ID', '')
            credentials['client_secret'] = st.secrets['vald'].get('CLIENT_SECRET', '')
            credentials['region'] = st.secrets['vald'].get('VALD_REGION', 'euw')
            if credentials['tenant_id'] and (credentials['token'] or (credentials['client_id'] and credentials['client_secret'])):
                found = True
    except Exception:
        pass  # Fall through to .env file

    # Try multiple possible locations for .env file (relative paths for cloud compatibility)
    if not found:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        possible_paths = [
            # config/local_secrets/.env (primary location for local development)
            os.path.join(parent_dir, 'config', 'local_secrets', '.env'),
            # In same directory as dashboard (Streamlit Cloud)
            os.path.join(script_dir, '.env'),
            # Relative to dashboard directory
            os.path.join(parent_dir, 'vald_api_pulls-main', 'forcedecks', '.env'),
            # In parent directory
            os.path.join(parent_dir, '.env'),
        ]

        env_path = None
        for path in possible_paths:
            if os.path.exists(path):
                env_path = path
                break

        if env_path:
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
                found = True
            except Exception:
                pass

    # If we have client credentials but no token, fetch OAuth token automatically
    if not credentials['token'] and credentials['client_id'] and credentials['client_secret']:
        oauth_token = get_oauth_token(
            credentials['client_id'],
            credentials['client_secret'],
            credentials['region']
        )
        if oauth_token:
            credentials['token'] = oauth_token
            found = True

    # Final check: do we have what we need?
    if credentials['tenant_id'] and credentials['token']:
        return credentials, True

    return credentials, found


@st.cache_data(ttl=3600, show_spinner=False)
def get_force_trace(test_id, trial_id, token, tenant_id, region='euw'):
    """Fetch force trace data from VALD API for a specific test. Cached for 1 hour.

    Uses the /v2019q3/teams/{teamId}/tests/{testId}/recording endpoint.
    Returns a DataFrame with force trace data, or empty DataFrame on failure.

    Response format from VALD API:
    - recordingDataHeader: column names like "Time", "Z Left", "Z Right", etc.
    - recordingData: array of data points
    - samplingFrequency: Hz (typically 200 or 1000)
    """
    import requests

    # External VALD API endpoints (official API per VALD documentation)
    ext_base_urls = {
        'euw': 'https://prd-euw-api-extforcedecks.valdperformance.com',
        'use': 'https://prd-use-api-extforcedecks.valdperformance.com',
        'aue': 'https://prd-aue-api-extforcedecks.valdperformance.com'
    }

    ext_base = ext_base_urls.get(region, ext_base_urls['euw'])

    # Primary endpoint: /v2019q3/teams/{teamId}/tests/{testId}/recording
    # This is the official VALD external API endpoint for force trace data
    recording_url = f"{ext_base}/v2019q3/teams/{tenant_id}/tests/{test_id}/recording"

    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }

    try:
        response = requests.get(recording_url, headers=headers, timeout=60)

        if response.status_code == 200:
            data = response.json()

            # Handle VALD recording response format
            # Contains: recordingDataHeader, recordingData, samplingFrequency, etc.
            if isinstance(data, dict):
                # Official VALD format with recordingData and recordingDataHeader
                if 'recordingData' in data and 'recordingDataHeader' in data:
                    headers_list = data['recordingDataHeader']
                    recording_data = data['recordingData']
                    sampling_freq = data.get('samplingFrequency', 200)

                    if recording_data and headers_list:
                        # Create DataFrame with proper column names
                        df = pd.DataFrame(recording_data, columns=headers_list)

                        # Calculate time in ms based on sampling frequency
                        if 'Time' not in df.columns:
                            df['time_ms'] = np.arange(len(df)) * (1000 / sampling_freq)
                        else:
                            df['time_ms'] = df['Time']

                        # Try to identify force columns (Z Left, Z Right, or combined)
                        force_cols = [c for c in df.columns if 'Z ' in c or 'Force' in c.lower()]
                        if force_cols:
                            # Combine left and right for total force if both exist
                            if 'Z Left' in df.columns and 'Z Right' in df.columns:
                                df['force_n'] = df['Z Left'] + df['Z Right']
                                df['force_left'] = df['Z Left']
                                df['force_right'] = df['Z Right']
                            elif force_cols:
                                df['force_n'] = df[force_cols[0]]

                        # Store metadata
                        df.attrs['sampling_frequency'] = sampling_freq
                        df.attrs['recording_type'] = data.get('recordingType', 'Unknown')
                        df.attrs['duration'] = data.get('duration', 0)

                        return df

                # Handle other dict formats (legacy/alternative)
                elif 'trace' in data:
                    trace_data = data['trace']
                    if isinstance(trace_data, list):
                        df = pd.DataFrame({
                            'time_ms': np.arange(len(trace_data)),
                            'force_n': trace_data
                        })
                        return df

                elif 'combinedForce' in data or 'leftForce' in data:
                    force_data = data.get('combinedForce') or data.get('leftForce', [])
                    if force_data:
                        df = pd.DataFrame({
                            'time_ms': np.arange(len(force_data)),
                            'force_n': force_data
                        })
                        return df

            # Handle list response (raw force values)
            elif isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], (int, float)):
                    df = pd.DataFrame({
                        'time_ms': np.arange(len(data)),
                        'force_n': data
                    })
                    return df
                elif isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                    return df

        # Log non-200 responses for debugging
        elif response.status_code == 401:
            st.warning("API authentication failed. Check your credentials.")
        elif response.status_code == 404:
            st.info(f"Recording not found for test {test_id[:8]}...")

    except requests.exceptions.Timeout:
        st.warning("API request timed out. Try again.")
    except requests.exceptions.RequestException as e:
        st.warning(f"API request error: {str(e)[:100]}")
    except Exception as e:
        st.warning(f"Error processing response: {str(e)[:100]}")

    # Return empty DataFrame if request fails
    return pd.DataFrame()


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

    /* Tabs Styling - Olympic Style with Scrollable Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f5;
        padding: 8px;
        border-radius: 10px;
        border-bottom: 3px solid var(--saudi-gold);
        /* Scrollable tabs */
        overflow-x: auto;
        overflow-y: hidden;
        flex-wrap: nowrap;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: thin;
        scrollbar-color: var(--saudi-green) #e0e0e0;
    }

    /* Custom scrollbar for tabs */
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        height: 6px;
    }

    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {
        background: #e0e0e0;
        border-radius: 3px;
    }

    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
        background: var(--saudi-green);
        border-radius: 3px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 0px 16px;
        background-color: white;
        border-radius: 8px;
        color: #333333;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
        border: 2px solid #e0e0e0;
        white-space: nowrap;
        flex-shrink: 0;
        min-width: fit-content;
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

# ============================================================================
# DATA SETUP - No sidebar filters (filters removed per user request)
# See docs/DASHBOARD_TABS_REFERENCE.md for filter restoration
# ============================================================================
filtered_df = df.copy()
filtered_forceframe = df_forceframe.copy()
filtered_nordbord = df_nordbord.copy()

# Variables for backward compatibility
selected_sports = []
selected_athletes = []
selected_test_types = df['testType'].dropna().unique().tolist() if 'testType' in df.columns else []

# Summary stats in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Data Summary")
st.sidebar.metric("Total Tests", len(filtered_df))
st.sidebar.metric("Athletes", filtered_df['Name'].nunique() if 'Name' in filtered_df.columns else 0)
st.sidebar.metric("Sports", filtered_df['athlete_sport'].nunique() if 'athlete_sport' in filtered_df.columns else 0)

# ============================================================================
# TABS - Streamlined (removed tabs preserved in docs/DASHBOARD_TABS_REFERENCE.md)
# ============================================================================

# New tab order: Home, Reports, ForceFrame, NordBord, Throws, Data Entry, Trace, Data
# Original indices: 0=Home, 1=Reports, 2=ForceFrame, 3=NordBord, 7=Throws, 8=Trace, 16=Data
# New indices:      0=Home, 1=Reports, 2=ForceFrame, 3=NordBord, 4=Throws, 5=Data Entry, 6=Trace, 7=Data
tabs = st.tabs([
    "🏠 Home", "📊 Reports", "🔲 ForceFrame", "🦵 NordBord", "🥏 Throws", "✏️ Data Entry", "📉 Trace", "📋 Data"
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
# PAGE: ATHLETE PROFILE (REMOVED - see docs/DASHBOARD_TABS_REFERENCE.md)
# ============================================================================

# ============================================================================
# PAGE: CMJ ANALYSIS MODULE (REMOVED - see docs/DASHBOARD_TABS_REFERENCE.md)
# ============================================================================

# ============================================================================
# TAB: ISOMETRIC ANALYSIS MODULE (REMOVED - see docs/DASHBOARD_TABS_REFERENCE.md)
# ============================================================================

# ============================================================================
# TAB 5: THROWS TRAINING DASHBOARD
# ============================================================================



with tabs[4]:  # Throws (was tabs[7])
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
# TAB: SPORT ANALYSIS (REMOVED - See docs/DASHBOARD_TABS_REFERENCE.md)
# ============================================================================

# ============================================================================
# TAB: RISK & READINESS (REMOVED - See docs/DASHBOARD_TABS_REFERENCE.md)
# ============================================================================

# ============================================================================
# TAB: COMPARISONS (REMOVED - See docs/DASHBOARD_TABS_REFERENCE.md)
# ============================================================================

# ============================================================================
# TAB: PROGRESS TRACKING (REMOVED - See docs/DASHBOARD_TABS_REFERENCE.md)
# ============================================================================

# ============================================================================
# TAB: RANKINGS (REMOVED - See docs/DASHBOARD_TABS_REFERENCE.md)
# ============================================================================

# ============================================================================
# TAB 5: DATA ENTRY - Training Distances & External Data
# ============================================================================

with tabs[5]:  # Data Entry
    st.markdown("## ✏️ Data Entry")
    st.markdown("*Record training distances, S&C metrics, and other external data*")

    # Initialize session state for training data
    if 'training_distances' not in st.session_state:
        st.session_state.training_distances = pd.DataFrame()

    # Initialize session state for S&C data
    if 'sc_upper_body' not in st.session_state:
        st.session_state.sc_upper_body = pd.DataFrame()

    if 'sc_lower_body' not in st.session_state:
        st.session_state.sc_lower_body = pd.DataFrame()

    # Load existing training data from CSV if available
    training_data_path = os.path.join(os.path.dirname(__file__), 'data', 'training_distances.csv')
    sc_upper_body_path = os.path.join(os.path.dirname(__file__), 'data', 'sc_upper_body.csv')
    sc_lower_body_path = os.path.join(os.path.dirname(__file__), 'data', 'sc_lower_body.csv')

    @st.cache_data(ttl=300)
    def load_training_distances():
        """Load training distances from CSV file."""
        if os.path.exists(training_data_path):
            df = pd.read_csv(training_data_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        return pd.DataFrame()

    @st.cache_data(ttl=300)
    def load_sc_upper_body():
        """Load S&C upper body data from CSV file."""
        if os.path.exists(sc_upper_body_path):
            df = pd.read_csv(sc_upper_body_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        return pd.DataFrame()

    @st.cache_data(ttl=300)
    def load_sc_lower_body():
        """Load S&C lower body data from CSV file."""
        if os.path.exists(sc_lower_body_path):
            df = pd.read_csv(sc_lower_body_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        return pd.DataFrame()

    # Load existing data
    if st.session_state.training_distances.empty:
        st.session_state.training_distances = load_training_distances()

    if st.session_state.sc_upper_body.empty:
        st.session_state.sc_upper_body = load_sc_upper_body()

    if st.session_state.sc_lower_body.empty:
        st.session_state.sc_lower_body = load_sc_lower_body()

    # Create sub-tabs for different entry types
    entry_tabs = st.tabs(["🥏 Throws Distance", "💪 S&C Upper Body", "🦵 S&C Lower Body", "📊 View Data", "📈 Charts"])

    # -------------------------------------------------------------------------
    # SUB-TAB: Throws Distance Entry Form
    # -------------------------------------------------------------------------
    with entry_tabs[0]:
        st.markdown("### 🥏 Record Throw Distance")

        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #007167 0%, #005a51 100%);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            color: white;
        ">
            <p style="margin: 0; font-size: 0.95rem;">
                📝 Record training throw distances for Athletics athletes. Data is saved locally and can be exported.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("throw_distance_form", clear_on_submit=True):
            col1, col2 = st.columns(2)

            with col1:
                # Get athlete list from existing data
                athletes_list = []
                if 'Name' in filtered_df.columns:
                    athletes_list = sorted([a for a in filtered_df['Name'].unique() if pd.notna(a)])

                selected_athlete = st.selectbox(
                    "Athlete *",
                    options=athletes_list if athletes_list else ["No athletes loaded"],
                    key="entry_athlete"
                )

                entry_date = st.date_input(
                    "Date *",
                    value=datetime.now().date(),
                    key="entry_date"
                )

                event_type = st.selectbox(
                    "Event *",
                    options=["Shot Put", "Discus", "Javelin", "Hammer"],
                    key="entry_event"
                )

            with col2:
                implement_weight = st.selectbox(
                    "Implement Weight (kg)",
                    options=["Competition", "3.0", "4.0", "5.0", "6.0", "7.0", "7.26", "Other"],
                    key="entry_implement"
                )

                if implement_weight == "Other":
                    custom_weight = st.number_input(
                        "Custom Weight (kg)",
                        min_value=0.5,
                        max_value=20.0,
                        value=4.0,
                        step=0.5,
                        key="entry_custom_weight"
                    )

                distance = st.number_input(
                    "Distance (m) *",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                    key="entry_distance"
                )

            # Additional fields
            col3, col4 = st.columns(2)

            with col3:
                session_type = st.selectbox(
                    "Session Type",
                    options=["Training", "Competition", "Testing", "Warm-up"],
                    key="entry_session"
                )

                # Location/Competition Name field - changes label based on session type
                location_label = "Competition Name" if session_type == "Competition" else "Training Location"
                location = st.text_input(
                    f"{location_label}",
                    placeholder="e.g., Arab Championships, King Fahd Stadium, Riyadh...",
                    key="entry_location"
                )

                attempt_number = st.number_input(
                    "Attempt #",
                    min_value=1,
                    max_value=20,
                    value=1,
                    key="entry_attempt"
                )

            with col4:
                conditions = st.selectbox(
                    "Conditions",
                    options=["Indoor", "Outdoor - Calm", "Outdoor - Windy", "Outdoor - Rain"],
                    key="entry_conditions"
                )

                notes = st.text_area(
                    "Notes (optional)",
                    placeholder="Technical cues, RPE, feelings...",
                    max_chars=500,
                    key="entry_notes"
                )

            submitted = st.form_submit_button("💾 Save Entry", use_container_width=True)

            if submitted:
                if distance > 0 and selected_athlete != "No athletes loaded":
                    # Determine weight value
                    weight_value = implement_weight
                    if implement_weight == "Other":
                        weight_value = str(custom_weight)

                    # Create new entry
                    new_entry = pd.DataFrame([{
                        'date': entry_date,
                        'athlete': selected_athlete,
                        'event': event_type,
                        'implement_kg': weight_value,
                        'distance_m': distance,
                        'session_type': session_type,
                        'location': location,
                        'attempt': attempt_number,
                        'conditions': conditions,
                        'notes': notes,
                        'created_at': datetime.now().isoformat()
                    }])

                    # Append to existing data
                    if st.session_state.training_distances.empty:
                        st.session_state.training_distances = new_entry
                    else:
                        st.session_state.training_distances = pd.concat(
                            [st.session_state.training_distances, new_entry],
                            ignore_index=True
                        )

                    # Save to CSV
                    os.makedirs(os.path.dirname(training_data_path), exist_ok=True)
                    st.session_state.training_distances.to_csv(training_data_path, index=False)

                    # Clear cache to reload data
                    load_training_distances.clear()

                    st.success(f"✅ Saved: {selected_athlete} - {event_type} - {distance}m")
                else:
                    st.error("Please enter a valid distance and select an athlete")

        # Quick entry for multiple throws
        st.markdown("---")
        st.markdown("### ⚡ Quick Entry (Multiple Throws)")
        st.markdown("*Paste data from a spreadsheet (Date, Athlete, Event, Weight, Distance, Session, Location)*")

        bulk_data = st.text_area(
            "Paste CSV data (one row per line, comma-separated)",
            placeholder="2024-01-15, Mohamed Tolo, Shot Put, 7.26, 18.5, Training, King Fahd Stadium\n2024-01-15, Mohamed Tolo, Shot Put, 7.26, 19.2, Competition, Arab Championships",
            key="bulk_entry"
        )

        if st.button("📥 Import Bulk Data", key="import_bulk"):
            if bulk_data.strip():
                try:
                    from io import StringIO
                    # Try to detect number of columns
                    first_line = bulk_data.strip().split('\n')[0]
                    num_cols = len(first_line.split(','))

                    if num_cols >= 7:
                        # Full format with session type and location
                        bulk_df = pd.read_csv(
                            StringIO(bulk_data),
                            names=['date', 'athlete', 'event', 'implement_kg', 'distance_m', 'session_type', 'location'],
                            skipinitialspace=True
                        )
                    elif num_cols >= 6:
                        # With session type but no location
                        bulk_df = pd.read_csv(
                            StringIO(bulk_data),
                            names=['date', 'athlete', 'event', 'implement_kg', 'distance_m', 'session_type'],
                            skipinitialspace=True
                        )
                        bulk_df['location'] = ''
                    else:
                        # Basic format
                        bulk_df = pd.read_csv(
                            StringIO(bulk_data),
                            names=['date', 'athlete', 'event', 'implement_kg', 'distance_m'],
                            skipinitialspace=True
                        )
                        bulk_df['session_type'] = 'Training'
                        bulk_df['location'] = ''

                    bulk_df['date'] = pd.to_datetime(bulk_df['date'])
                    if 'session_type' not in bulk_df.columns:
                        bulk_df['session_type'] = 'Training'
                    if 'location' not in bulk_df.columns:
                        bulk_df['location'] = ''
                    bulk_df['attempt'] = 1
                    bulk_df['conditions'] = ''
                    bulk_df['notes'] = ''
                    bulk_df['created_at'] = datetime.now().isoformat()

                    if st.session_state.training_distances.empty:
                        st.session_state.training_distances = bulk_df
                    else:
                        st.session_state.training_distances = pd.concat(
                            [st.session_state.training_distances, bulk_df],
                            ignore_index=True
                        )

                    # Save to CSV
                    os.makedirs(os.path.dirname(training_data_path), exist_ok=True)
                    st.session_state.training_distances.to_csv(training_data_path, index=False)
                    load_training_distances.clear()

                    st.success(f"✅ Imported {len(bulk_df)} entries")
                except Exception as e:
                    st.error(f"Error parsing data: {str(e)}")

    # -------------------------------------------------------------------------
    # SUB-TAB: S&C Upper Body (Bench Press, Pull Ups, etc.)
    # -------------------------------------------------------------------------
    with entry_tabs[1]:
        st.markdown("### 💪 S&C Upper Body Strength & Power")

        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #007167 0%, #005a51 100%);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            color: white;
        ">
            <p style="margin: 0; font-size: 0.95rem;">
                📝 Record upper body strength metrics: Bench Press, Pull Ups, Overhead Press, and more.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("sc_upper_body_form", clear_on_submit=True):
            sc_col1, sc_col2 = st.columns(2)

            with sc_col1:
                # Get athlete list from existing data
                sc_athletes_list = []
                if 'Name' in filtered_df.columns:
                    sc_athletes_list = sorted([a for a in filtered_df['Name'].unique() if pd.notna(a)])

                sc_selected_athlete = st.selectbox(
                    "Athlete *",
                    options=sc_athletes_list if sc_athletes_list else ["No athletes loaded"],
                    key="sc_entry_athlete"
                )

                sc_entry_date = st.date_input(
                    "Date *",
                    value=datetime.now().date(),
                    key="sc_entry_date"
                )

                sc_exercise = st.selectbox(
                    "Exercise *",
                    options=[
                        "Bench Press",
                        "Incline Bench Press",
                        "Overhead Press",
                        "Push Press",
                        "Pull Up",
                        "Weighted Pull Up",
                        "Lat Pulldown",
                        "Bent Over Row",
                        "Dumbbell Row",
                        "Shoulder Press (DB)",
                        "Floor Press",
                        "Close Grip Bench",
                        "Other"
                    ],
                    key="sc_entry_exercise"
                )

                if sc_exercise == "Other":
                    sc_custom_exercise = st.text_input(
                        "Custom Exercise Name",
                        key="sc_custom_exercise"
                    )

            with sc_col2:
                sc_weight = st.number_input(
                    "Weight (kg) *",
                    min_value=0.0,
                    max_value=500.0,
                    value=0.0,
                    step=2.5,
                    format="%.1f",
                    key="sc_entry_weight"
                )

                sc_reps = st.number_input(
                    "Reps *",
                    min_value=1,
                    max_value=100,
                    value=1,
                    key="sc_entry_reps"
                )

                sc_sets = st.number_input(
                    "Sets",
                    min_value=1,
                    max_value=20,
                    value=1,
                    key="sc_entry_sets"
                )

                sc_rpe = st.slider(
                    "RPE (Rate of Perceived Exertion)",
                    min_value=1,
                    max_value=10,
                    value=7,
                    key="sc_entry_rpe"
                )

            # Additional fields
            sc_col3, sc_col4 = st.columns(2)

            with sc_col3:
                sc_test_type = st.selectbox(
                    "Test Type",
                    options=["Working Set", "1RM Test", "3RM Test", "5RM Test", "AMRAP", "Warm-up"],
                    key="sc_entry_test_type"
                )

                sc_equipment = st.selectbox(
                    "Equipment",
                    options=["Barbell", "Dumbbell", "Machine", "Cable", "Bodyweight", "Kettlebell", "Other"],
                    key="sc_entry_equipment"
                )

            with sc_col4:
                sc_location = st.text_input(
                    "Location",
                    placeholder="e.g., King Fahd Stadium, SAOC Gym, Riyadh...",
                    key="sc_entry_location"
                )

                sc_notes = st.text_area(
                    "Notes (optional)",
                    placeholder="Technique cues, how it felt...",
                    max_chars=500,
                    key="sc_entry_notes"
                )

            sc_submitted = st.form_submit_button("💾 Save S&C Entry", use_container_width=True)

            if sc_submitted:
                if sc_weight > 0 and sc_selected_athlete != "No athletes loaded":
                    # Determine exercise name
                    exercise_name = sc_exercise
                    if sc_exercise == "Other" and sc_custom_exercise:
                        exercise_name = sc_custom_exercise

                    # Calculate estimated 1RM using Brzycki formula
                    if sc_reps > 0 and sc_reps <= 12:
                        estimated_1rm = sc_weight * (36 / (37 - sc_reps))
                    else:
                        estimated_1rm = sc_weight  # For higher reps, just use weight

                    # Create new entry
                    sc_new_entry = pd.DataFrame([{
                        'date': sc_entry_date,
                        'athlete': sc_selected_athlete,
                        'exercise': exercise_name,
                        'weight_kg': sc_weight,
                        'reps': sc_reps,
                        'sets': sc_sets,
                        'rpe': sc_rpe,
                        'test_type': sc_test_type,
                        'equipment': sc_equipment,
                        'estimated_1rm': round(estimated_1rm, 1),
                        'location': sc_location,
                        'notes': sc_notes,
                        'created_at': datetime.now().isoformat()
                    }])

                    # Append to existing data
                    if st.session_state.sc_upper_body.empty:
                        st.session_state.sc_upper_body = sc_new_entry
                    else:
                        st.session_state.sc_upper_body = pd.concat(
                            [st.session_state.sc_upper_body, sc_new_entry],
                            ignore_index=True
                        )

                    # Add row_id
                    st.session_state.sc_upper_body = st.session_state.sc_upper_body.reset_index(drop=True)
                    st.session_state.sc_upper_body['row_id'] = st.session_state.sc_upper_body.index

                    # Save to CSV
                    os.makedirs(os.path.dirname(sc_upper_body_path), exist_ok=True)
                    st.session_state.sc_upper_body.to_csv(sc_upper_body_path, index=False)

                    # Clear cache to reload data
                    load_sc_upper_body.clear()

                    st.success(f"✅ Saved: {sc_selected_athlete} - {exercise_name} - {sc_weight}kg x {sc_reps} (Est. 1RM: {estimated_1rm:.1f}kg)")
                else:
                    st.error("Please enter a valid weight and select an athlete")

        # Quick reference for 1RM estimation
        st.markdown("---")
        with st.expander("📊 1RM Estimation Reference"):
            st.markdown("""
            **Estimated 1RM** is calculated using the Brzycki formula:

            `1RM = Weight × (36 / (37 - Reps))`

            | Reps | % of 1RM |
            |------|----------|
            | 1    | 100%     |
            | 2    | 95%      |
            | 3    | 93%      |
            | 4    | 90%      |
            | 5    | 87%      |
            | 6    | 85%      |
            | 8    | 80%      |
            | 10   | 75%      |
            | 12   | 70%      |

            *Note: Estimates become less accurate above 10 reps*
            """)

        # Display recent S&C entries
        st.markdown("---")
        st.markdown("### 📋 Recent S&C Entries")

        sc_df = st.session_state.sc_upper_body

        if not sc_df.empty:
            # Show last 10 entries
            recent_sc = sc_df.sort_values('date', ascending=False).head(10)

            # Display columns
            display_cols = ['date', 'athlete', 'exercise', 'weight_kg', 'reps', 'sets', 'estimated_1rm', 'rpe', 'test_type']
            available_cols = [c for c in display_cols if c in recent_sc.columns]

            st.dataframe(
                recent_sc[available_cols],
                use_container_width=True,
                hide_index=True
            )

            # Summary stats
            st.markdown("#### 💪 Personal Bests (Estimated 1RM)")

            if 'estimated_1rm' in sc_df.columns:
                pb_df = sc_df.groupby(['athlete', 'exercise'])['estimated_1rm'].max().reset_index()
                pb_df.columns = ['Athlete', 'Exercise', 'Best Est. 1RM (kg)']

                st.dataframe(
                    pb_df.sort_values('Best Est. 1RM (kg)', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("📭 No S&C data recorded yet. Use the form above to add entries.")

    # -------------------------------------------------------------------------
    # SUB-TAB: S&C Lower Body (Squats, Deadlifts, etc.)
    # -------------------------------------------------------------------------
    with entry_tabs[2]:
        st.markdown("### 🦵 S&C Lower Body Strength & Power")

        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #007167 0%, #005a51 100%);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            color: white;
        ">
            <p style="margin: 0; font-size: 0.95rem;">
                📝 Record lower body strength metrics: Squats, Deadlifts, Hip Thrusts, and more.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("sc_lower_body_form", clear_on_submit=True):
            lb_col1, lb_col2 = st.columns(2)

            with lb_col1:
                # Get athlete list from existing data
                lb_athletes_list = []
                if 'Name' in filtered_df.columns:
                    lb_athletes_list = sorted([a for a in filtered_df['Name'].unique() if pd.notna(a)])

                lb_selected_athlete = st.selectbox(
                    "Athlete *",
                    options=lb_athletes_list if lb_athletes_list else ["No athletes loaded"],
                    key="lb_entry_athlete"
                )

                lb_entry_date = st.date_input(
                    "Date *",
                    value=datetime.now().date(),
                    key="lb_entry_date"
                )

                lb_exercise = st.selectbox(
                    "Exercise *",
                    options=[
                        "Back Squat",
                        "Front Squat",
                        "Deadlift",
                        "Romanian Deadlift (RDL)",
                        "Hip Thrust",
                        "Bulgarian Split Squat",
                        "Leg Press",
                        "Trap Bar Deadlift",
                        "Goblet Squat",
                        "Lunges",
                        "Step Ups",
                        "Nordic Hamstring Curl",
                        "Glute Ham Raise",
                        "Calf Raise",
                        "Other"
                    ],
                    key="lb_entry_exercise"
                )

                if lb_exercise == "Other":
                    lb_custom_exercise = st.text_input(
                        "Custom Exercise Name",
                        key="lb_custom_exercise"
                    )

            with lb_col2:
                lb_weight = st.number_input(
                    "Weight (kg) *",
                    min_value=0.0,
                    max_value=500.0,
                    value=0.0,
                    step=2.5,
                    format="%.1f",
                    key="lb_entry_weight"
                )

                lb_reps = st.number_input(
                    "Reps *",
                    min_value=1,
                    max_value=100,
                    value=1,
                    key="lb_entry_reps"
                )

                lb_sets = st.number_input(
                    "Sets",
                    min_value=1,
                    max_value=20,
                    value=1,
                    key="lb_entry_sets"
                )

                lb_rpe = st.slider(
                    "RPE (Rate of Perceived Exertion)",
                    min_value=1,
                    max_value=10,
                    value=7,
                    key="lb_entry_rpe"
                )

            # Additional fields
            lb_col3, lb_col4 = st.columns(2)

            with lb_col3:
                lb_test_type = st.selectbox(
                    "Test Type",
                    options=["Working Set", "1RM Test", "3RM Test", "5RM Test", "AMRAP", "Warm-up"],
                    key="lb_entry_test_type"
                )

                lb_equipment = st.selectbox(
                    "Equipment",
                    options=["Barbell", "Dumbbell", "Machine", "Cable", "Bodyweight", "Kettlebell", "Trap Bar", "Other"],
                    key="lb_entry_equipment"
                )

            with lb_col4:
                lb_location = st.text_input(
                    "Location",
                    placeholder="e.g., King Fahd Stadium, SAOC Gym, Riyadh...",
                    key="lb_entry_location"
                )

                lb_notes = st.text_area(
                    "Notes (optional)",
                    placeholder="Technique cues, how it felt...",
                    max_chars=500,
                    key="lb_entry_notes"
                )

            lb_submitted = st.form_submit_button("💾 Save S&C Entry", use_container_width=True)

            if lb_submitted:
                if lb_weight > 0 and lb_selected_athlete != "No athletes loaded":
                    # Determine exercise name
                    exercise_name = lb_exercise
                    if lb_exercise == "Other" and lb_custom_exercise:
                        exercise_name = lb_custom_exercise

                    # Calculate estimated 1RM using Brzycki formula
                    if lb_reps > 0 and lb_reps <= 12:
                        estimated_1rm = lb_weight * (36 / (37 - lb_reps))
                    else:
                        estimated_1rm = lb_weight  # For higher reps, just use weight

                    # Create new entry
                    lb_new_entry = pd.DataFrame([{
                        'date': lb_entry_date,
                        'athlete': lb_selected_athlete,
                        'exercise': exercise_name,
                        'weight_kg': lb_weight,
                        'reps': lb_reps,
                        'sets': lb_sets,
                        'rpe': lb_rpe,
                        'test_type': lb_test_type,
                        'equipment': lb_equipment,
                        'estimated_1rm': round(estimated_1rm, 1),
                        'location': lb_location,
                        'notes': lb_notes,
                        'created_at': datetime.now().isoformat()
                    }])

                    # Append to existing data
                    if st.session_state.sc_lower_body.empty:
                        st.session_state.sc_lower_body = lb_new_entry
                    else:
                        st.session_state.sc_lower_body = pd.concat(
                            [st.session_state.sc_lower_body, lb_new_entry],
                            ignore_index=True
                        )

                    # Add row_id
                    st.session_state.sc_lower_body = st.session_state.sc_lower_body.reset_index(drop=True)
                    st.session_state.sc_lower_body['row_id'] = st.session_state.sc_lower_body.index

                    # Save to CSV
                    os.makedirs(os.path.dirname(sc_lower_body_path), exist_ok=True)
                    st.session_state.sc_lower_body.to_csv(sc_lower_body_path, index=False)

                    # Clear cache to reload data
                    load_sc_lower_body.clear()

                    st.success(f"✅ Saved: {lb_selected_athlete} - {exercise_name} - {lb_weight}kg x {lb_reps} (Est. 1RM: {estimated_1rm:.1f}kg)")
                else:
                    st.error("Please enter a valid weight and select an athlete")

        # Quick reference for 1RM estimation
        st.markdown("---")
        with st.expander("📊 1RM Estimation Reference"):
            st.markdown("""
            **Estimated 1RM** is calculated using the Brzycki formula:

            `1RM = Weight × (36 / (37 - Reps))`

            | Reps | % of 1RM |
            |------|----------|
            | 1    | 100%     |
            | 2    | 95%      |
            | 3    | 93%      |
            | 4    | 90%      |
            | 5    | 87%      |
            | 6    | 85%      |
            | 8    | 80%      |
            | 10   | 75%      |
            | 12   | 70%      |

            *Note: Estimates become less accurate above 10 reps*
            """)

        # Display recent S&C Lower Body entries
        st.markdown("---")
        st.markdown("### 📋 Recent S&C Lower Body Entries")

        lb_df = st.session_state.sc_lower_body

        if not lb_df.empty:
            # Show last 10 entries
            recent_lb = lb_df.sort_values('date', ascending=False).head(10)

            # Display columns
            display_cols = ['date', 'athlete', 'exercise', 'weight_kg', 'reps', 'sets', 'estimated_1rm', 'rpe', 'test_type']
            available_cols = [c for c in display_cols if c in recent_lb.columns]

            st.dataframe(
                recent_lb[available_cols],
                use_container_width=True,
                hide_index=True
            )

            # Summary stats
            st.markdown("#### 🦵 Personal Bests (Estimated 1RM)")

            if 'estimated_1rm' in lb_df.columns:
                pb_lb_df = lb_df.groupby(['athlete', 'exercise'])['estimated_1rm'].max().reset_index()
                pb_lb_df.columns = ['Athlete', 'Exercise', 'Best Est. 1RM (kg)']

                st.dataframe(
                    pb_lb_df.sort_values('Best Est. 1RM (kg)', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("📭 No S&C Lower Body data recorded yet. Use the form above to add entries.")

    # -------------------------------------------------------------------------
    # SUB-TAB: View Data (with Edit/Delete) - Now with sub-tabs
    # -------------------------------------------------------------------------
    with entry_tabs[3]:
        st.markdown("### 📊 Recorded Training Data")

        # Create sub-tabs for different data types
        view_data_tabs = st.tabs(["🥏 Throws", "💪 S&C Upper Body", "🦵 S&C Lower Body"])

        # -----------------------------------------------------------------
        # VIEW DATA SUB-TAB: Throws
        # -----------------------------------------------------------------
        with view_data_tabs[0]:
            training_df = st.session_state.training_distances

            if not training_df.empty:
                # Ensure index is set for editing
                if 'row_id' not in training_df.columns:
                    training_df = training_df.reset_index(drop=True)
                    training_df['row_id'] = training_df.index
                    st.session_state.training_distances = training_df

                # Filters
                filter_col1, filter_col2, filter_col3 = st.columns(3)

                with filter_col1:
                    filter_athletes = ['All'] + sorted(training_df['athlete'].unique().tolist())
                    selected_filter_athlete = st.selectbox("Filter by Athlete:", filter_athletes, key="view_filter_athlete")

                with filter_col2:
                    filter_events = ['All'] + sorted(training_df['event'].unique().tolist())
                    selected_filter_event = st.selectbox("Filter by Event:", filter_events, key="view_filter_event")

                with filter_col3:
                    sort_options = ['Date (Newest)', 'Date (Oldest)', 'Distance (Best)', 'Distance (Worst)']
                    sort_by = st.selectbox("Sort by:", sort_options, key="view_sort")

                # Apply filters
                display_training_df = training_df.copy()

                if selected_filter_athlete != 'All':
                    display_training_df = display_training_df[display_training_df['athlete'] == selected_filter_athlete]

                if selected_filter_event != 'All':
                    display_training_df = display_training_df[display_training_df['event'] == selected_filter_event]

                # Apply sorting
                if 'date' in display_training_df.columns:
                    display_training_df['date'] = pd.to_datetime(display_training_df['date'])
                    if sort_by == 'Date (Newest)':
                        display_training_df = display_training_df.sort_values('date', ascending=False)
                    elif sort_by == 'Date (Oldest)':
                        display_training_df = display_training_df.sort_values('date', ascending=True)
                    elif sort_by == 'Distance (Best)':
                        display_training_df = display_training_df.sort_values('distance_m', ascending=False)
                    elif sort_by == 'Distance (Worst)':
                        display_training_df = display_training_df.sort_values('distance_m', ascending=True)

                # Display stats
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

                with stat_col1:
                    st.metric("Total Entries", len(display_training_df))
                with stat_col2:
                    st.metric("Athletes", display_training_df['athlete'].nunique())
                with stat_col3:
                    if 'distance_m' in display_training_df.columns and len(display_training_df) > 0:
                        st.metric("Best Distance", f"{display_training_df['distance_m'].max():.2f}m")
                with stat_col4:
                    if 'distance_m' in display_training_df.columns and len(display_training_df) > 0:
                        st.metric("Avg Distance", f"{display_training_df['distance_m'].mean():.2f}m")

                # Display table with color coding by session type
                st.markdown("#### Data Table")
                st.markdown("*🟢 Training | 🟡 Competition | 🔵 Testing | ⚪ Other*")

                # Add color column for display
                def get_session_color(session_type):
                    colors = {
                        'Training': '🟢',
                        'Competition': '🟡',
                        'Testing': '🔵',
                        'Warm-up': '⚪'
                    }
                    return colors.get(session_type, '⚪')

                display_df_styled = display_training_df.copy()
                if 'session_type' in display_df_styled.columns:
                    display_df_styled['Type'] = display_df_styled['session_type'].apply(get_session_color)
                    # Reorder columns to put Type first
                    cols = ['Type'] + [c for c in display_df_styled.columns if c != 'Type' and c != 'row_id' and c != 'created_at']
                    display_df_styled = display_df_styled[cols]

                st.dataframe(
                    display_df_styled,
                    use_container_width=True,
                    hide_index=True,
                    height=350
                )

                # EDIT / DELETE Section
                st.markdown("---")
                st.markdown("### ✏️ Edit or Delete Entry")

                # Create selection options
                row_options = []
                for idx, row in display_training_df.iterrows():
                    date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d') if pd.notna(row.get('date')) else 'N/A'
                    session_icon = get_session_color(row.get('session_type', ''))
                    label = f"{session_icon} {row['athlete']} | {row['event']} | {row['distance_m']:.2f}m | {date_str}"
                    row_options.append({'label': label, 'row_id': row.get('row_id', idx), 'idx': idx})

                if row_options:
                    selected_row_idx = st.selectbox(
                        "Select entry to edit/delete:",
                        range(len(row_options)),
                        format_func=lambda i: row_options[i]['label'],
                        key="edit_row_select"
                    )

                    selected_row_id = row_options[selected_row_idx]['row_id']
                    selected_row = training_df[training_df['row_id'] == selected_row_id].iloc[0]

                    edit_col1, edit_col2 = st.columns(2)

                    with edit_col1:
                        st.markdown("#### Edit Entry")

                        with st.form("edit_entry_form"):
                            edit_date = st.date_input(
                                "Date",
                                value=pd.to_datetime(selected_row['date']).date() if pd.notna(selected_row.get('date')) else datetime.now().date(),
                                key="edit_date"
                            )

                            edit_event = st.selectbox(
                                "Event",
                                options=["Shot Put", "Discus", "Javelin", "Hammer"],
                                index=["Shot Put", "Discus", "Javelin", "Hammer"].index(selected_row['event']) if selected_row['event'] in ["Shot Put", "Discus", "Javelin", "Hammer"] else 0,
                                key="edit_event"
                            )

                            edit_distance = st.number_input(
                                "Distance (m)",
                                min_value=0.0,
                                max_value=100.0,
                                value=float(selected_row['distance_m']),
                                step=0.01,
                                format="%.2f",
                                key="edit_distance"
                            )

                            edit_session = st.selectbox(
                                "Session Type",
                                options=["Training", "Competition", "Testing", "Warm-up"],
                                index=["Training", "Competition", "Testing", "Warm-up"].index(selected_row.get('session_type', 'Training')) if selected_row.get('session_type') in ["Training", "Competition", "Testing", "Warm-up"] else 0,
                                key="edit_session"
                            )

                            edit_location_label = "Competition Name" if edit_session == "Competition" else "Training Location"
                            edit_location = st.text_input(
                                edit_location_label,
                                value=str(selected_row.get('location', '')),
                                key="edit_location"
                            )

                            edit_implement = st.text_input(
                                "Implement (kg)",
                                value=str(selected_row.get('implement_kg', '')),
                                key="edit_implement"
                            )

                            edit_notes = st.text_area(
                                "Notes",
                                value=str(selected_row.get('notes', '')),
                                key="edit_notes"
                            )

                            save_edit = st.form_submit_button("💾 Save Changes", use_container_width=True)

                            if save_edit:
                                # Update the row
                                mask = st.session_state.training_distances['row_id'] == selected_row_id
                                st.session_state.training_distances.loc[mask, 'date'] = edit_date
                                st.session_state.training_distances.loc[mask, 'event'] = edit_event
                                st.session_state.training_distances.loc[mask, 'distance_m'] = edit_distance
                                st.session_state.training_distances.loc[mask, 'session_type'] = edit_session
                                st.session_state.training_distances.loc[mask, 'location'] = edit_location
                                st.session_state.training_distances.loc[mask, 'implement_kg'] = edit_implement
                                st.session_state.training_distances.loc[mask, 'notes'] = edit_notes

                                # Save to CSV
                                st.session_state.training_distances.to_csv(training_data_path, index=False)
                                load_training_distances.clear()
                                st.success("✅ Entry updated!")
                                st.rerun()

                    with edit_col2:
                        st.markdown("#### Delete Entry")

                        st.warning(f"""
                        **Selected Entry:**
                        - Athlete: {selected_row['athlete']}
                        - Event: {selected_row['event']}
                        - Distance: {selected_row['distance_m']:.2f}m
                        - Date: {pd.to_datetime(selected_row['date']).strftime('%Y-%m-%d') if pd.notna(selected_row.get('date')) else 'N/A'}
                        """)

                        delete_confirm = st.checkbox("⚠️ I confirm I want to delete this entry", key="delete_confirm")

                        if st.button("🗑️ Delete Entry", type="secondary", disabled=not delete_confirm, use_container_width=True):
                            # Remove the row
                            st.session_state.training_distances = st.session_state.training_distances[
                                st.session_state.training_distances['row_id'] != selected_row_id
                            ].reset_index(drop=True)

                            # Re-assign row IDs
                            st.session_state.training_distances['row_id'] = st.session_state.training_distances.index

                            # Save to CSV
                            if not st.session_state.training_distances.empty:
                                st.session_state.training_distances.to_csv(training_data_path, index=False)
                            else:
                                if os.path.exists(training_data_path):
                                    os.remove(training_data_path)

                            load_training_distances.clear()
                            st.success("✅ Entry deleted!")
                            st.rerun()

                # Export options
                st.markdown("---")
                st.markdown("### 📥 Export Data")
                exp_col1, exp_col2 = st.columns(2)

                with exp_col1:
                    csv_export = display_training_df.drop(columns=['row_id'], errors='ignore').to_csv(index=False)
                    st.download_button(
                        "📥 Download as CSV",
                        data=csv_export,
                        file_name=f"training_distances_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key="export_throws_csv"
                    )

                with exp_col2:
                    if st.button("🗑️ Clear ALL Throws Data", type="secondary", key="clear_throws_data"):
                        clear_confirm = st.checkbox("⚠️ Confirm delete ALL throws data", key="clear_all_throws_confirm")
                        if clear_confirm:
                            st.session_state.training_distances = pd.DataFrame()
                            if os.path.exists(training_data_path):
                                os.remove(training_data_path)
                            load_training_distances.clear()
                            st.success("All throws data cleared")
                            st.rerun()
            else:
                st.info("📭 No throws data recorded yet. Use the 'Throws Distance' tab to add entries.")

        # -----------------------------------------------------------------
        # VIEW DATA SUB-TAB: S&C Upper Body
        # -----------------------------------------------------------------
        with view_data_tabs[1]:
            upper_df = st.session_state.sc_upper_body

            if not upper_df.empty:
                # Ensure index is set for editing
                if 'row_id' not in upper_df.columns:
                    upper_df = upper_df.reset_index(drop=True)
                    upper_df['row_id'] = upper_df.index
                    st.session_state.sc_upper_body = upper_df

                # Filters
                uf_col1, uf_col2, uf_col3 = st.columns(3)

                with uf_col1:
                    upper_filter_athletes = ['All'] + sorted(upper_df['athlete'].unique().tolist())
                    upper_selected_athlete = st.selectbox("Filter by Athlete:", upper_filter_athletes, key="upper_view_filter_athlete")

                with uf_col2:
                    upper_filter_exercises = ['All'] + sorted(upper_df['exercise'].unique().tolist())
                    upper_selected_exercise = st.selectbox("Filter by Exercise:", upper_filter_exercises, key="upper_view_filter_exercise")

                with uf_col3:
                    upper_sort_options = ['Date (Newest)', 'Date (Oldest)', 'Weight (Heaviest)', 'Est. 1RM (Best)']
                    upper_sort_by = st.selectbox("Sort by:", upper_sort_options, key="upper_view_sort")

                # Apply filters
                display_upper_df = upper_df.copy()

                if upper_selected_athlete != 'All':
                    display_upper_df = display_upper_df[display_upper_df['athlete'] == upper_selected_athlete]

                if upper_selected_exercise != 'All':
                    display_upper_df = display_upper_df[display_upper_df['exercise'] == upper_selected_exercise]

                # Apply sorting
                if 'date' in display_upper_df.columns:
                    display_upper_df['date'] = pd.to_datetime(display_upper_df['date'])
                    if upper_sort_by == 'Date (Newest)':
                        display_upper_df = display_upper_df.sort_values('date', ascending=False)
                    elif upper_sort_by == 'Date (Oldest)':
                        display_upper_df = display_upper_df.sort_values('date', ascending=True)
                    elif upper_sort_by == 'Weight (Heaviest)':
                        display_upper_df = display_upper_df.sort_values('weight_kg', ascending=False)
                    elif upper_sort_by == 'Est. 1RM (Best)':
                        display_upper_df = display_upper_df.sort_values('estimated_1rm', ascending=False)

                # Display stats
                ustat_col1, ustat_col2, ustat_col3, ustat_col4 = st.columns(4)

                with ustat_col1:
                    st.metric("Total Entries", len(display_upper_df))
                with ustat_col2:
                    st.metric("Athletes", display_upper_df['athlete'].nunique())
                with ustat_col3:
                    if 'estimated_1rm' in display_upper_df.columns and len(display_upper_df) > 0:
                        st.metric("Best Est. 1RM", f"{display_upper_df['estimated_1rm'].max():.1f}kg")
                with ustat_col4:
                    if 'weight_kg' in display_upper_df.columns and len(display_upper_df) > 0:
                        st.metric("Max Weight", f"{display_upper_df['weight_kg'].max():.1f}kg")

                # Display table
                st.markdown("#### Data Table")
                display_cols = ['date', 'athlete', 'exercise', 'weight_kg', 'reps', 'sets', 'estimated_1rm', 'rpe', 'test_type', 'equipment']
                available_cols = [c for c in display_cols if c in display_upper_df.columns]

                st.dataframe(
                    display_upper_df[available_cols],
                    use_container_width=True,
                    hide_index=True,
                    height=350
                )

                # Export options
                st.markdown("---")
                st.markdown("### 📥 Export Data")

                csv_export_upper = display_upper_df.drop(columns=['row_id'], errors='ignore').to_csv(index=False)
                st.download_button(
                    "📥 Download as CSV",
                    data=csv_export_upper,
                    file_name=f"sc_upper_body_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="export_upper_csv"
                )
            else:
                st.info("📭 No S&C Upper Body data recorded yet. Use the 'S&C Upper Body' tab to add entries.")

        # -----------------------------------------------------------------
        # VIEW DATA SUB-TAB: S&C Lower Body
        # -----------------------------------------------------------------
        with view_data_tabs[2]:
            lower_df = st.session_state.sc_lower_body

            if not lower_df.empty:
                # Ensure index is set for editing
                if 'row_id' not in lower_df.columns:
                    lower_df = lower_df.reset_index(drop=True)
                    lower_df['row_id'] = lower_df.index
                    st.session_state.sc_lower_body = lower_df

                # Filters
                lf_col1, lf_col2, lf_col3 = st.columns(3)

                with lf_col1:
                    lower_filter_athletes = ['All'] + sorted(lower_df['athlete'].unique().tolist())
                    lower_selected_athlete = st.selectbox("Filter by Athlete:", lower_filter_athletes, key="lower_view_filter_athlete")

                with lf_col2:
                    lower_filter_exercises = ['All'] + sorted(lower_df['exercise'].unique().tolist())
                    lower_selected_exercise = st.selectbox("Filter by Exercise:", lower_filter_exercises, key="lower_view_filter_exercise")

                with lf_col3:
                    lower_sort_options = ['Date (Newest)', 'Date (Oldest)', 'Weight (Heaviest)', 'Est. 1RM (Best)']
                    lower_sort_by = st.selectbox("Sort by:", lower_sort_options, key="lower_view_sort")

                # Apply filters
                display_lower_df = lower_df.copy()

                if lower_selected_athlete != 'All':
                    display_lower_df = display_lower_df[display_lower_df['athlete'] == lower_selected_athlete]

                if lower_selected_exercise != 'All':
                    display_lower_df = display_lower_df[display_lower_df['exercise'] == lower_selected_exercise]

                # Apply sorting
                if 'date' in display_lower_df.columns:
                    display_lower_df['date'] = pd.to_datetime(display_lower_df['date'])
                    if lower_sort_by == 'Date (Newest)':
                        display_lower_df = display_lower_df.sort_values('date', ascending=False)
                    elif lower_sort_by == 'Date (Oldest)':
                        display_lower_df = display_lower_df.sort_values('date', ascending=True)
                    elif lower_sort_by == 'Weight (Heaviest)':
                        display_lower_df = display_lower_df.sort_values('weight_kg', ascending=False)
                    elif lower_sort_by == 'Est. 1RM (Best)':
                        display_lower_df = display_lower_df.sort_values('estimated_1rm', ascending=False)

                # Display stats
                lstat_col1, lstat_col2, lstat_col3, lstat_col4 = st.columns(4)

                with lstat_col1:
                    st.metric("Total Entries", len(display_lower_df))
                with lstat_col2:
                    st.metric("Athletes", display_lower_df['athlete'].nunique())
                with lstat_col3:
                    if 'estimated_1rm' in display_lower_df.columns and len(display_lower_df) > 0:
                        st.metric("Best Est. 1RM", f"{display_lower_df['estimated_1rm'].max():.1f}kg")
                with lstat_col4:
                    if 'weight_kg' in display_lower_df.columns and len(display_lower_df) > 0:
                        st.metric("Max Weight", f"{display_lower_df['weight_kg'].max():.1f}kg")

                # Display table
                st.markdown("#### Data Table")
                display_cols = ['date', 'athlete', 'exercise', 'weight_kg', 'reps', 'sets', 'estimated_1rm', 'rpe', 'test_type', 'equipment']
                available_cols = [c for c in display_cols if c in display_lower_df.columns]

                st.dataframe(
                    display_lower_df[available_cols],
                    use_container_width=True,
                    hide_index=True,
                    height=350
                )

                # Export options
                st.markdown("---")
                st.markdown("### 📥 Export Data")

                csv_export_lower = display_lower_df.drop(columns=['row_id'], errors='ignore').to_csv(index=False)
                st.download_button(
                    "📥 Download as CSV",
                    data=csv_export_lower,
                    file_name=f"sc_lower_body_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="export_lower_csv"
                )
            else:
                st.info("📭 No S&C Lower Body data recorded yet. Use the 'S&C Lower Body' tab to add entries.")

    # -------------------------------------------------------------------------
    # SUB-TAB: Charts - Now with sub-tabs
    # -------------------------------------------------------------------------
    with entry_tabs[4]:
        st.markdown("### 📈 Training Charts")

        # Create sub-tabs for different chart types
        chart_sub_tabs = st.tabs(["🥏 Throws", "💪 S&C Upper Body", "🦵 S&C Lower Body"])

        # -----------------------------------------------------------------
        # CHARTS SUB-TAB: Throws
        # -----------------------------------------------------------------
        with chart_sub_tabs[0]:
            throws_chart_df = st.session_state.training_distances

            if not throws_chart_df.empty and 'date' in throws_chart_df.columns:
                throws_chart_df['date'] = pd.to_datetime(throws_chart_df['date'])

                # Chart filters - Row 1: Athlete and Event
                tc_col1, tc_col2 = st.columns(2)

                with tc_col1:
                    throws_chart_athletes = sorted(throws_chart_df['athlete'].unique().tolist())
                    selected_throws_athlete = st.selectbox(
                        "Select Athlete:",
                        options=throws_chart_athletes,
                        key="throws_chart_athlete"
                    )

                with tc_col2:
                    throws_chart_events = sorted(throws_chart_df[throws_chart_df['athlete'] == selected_throws_athlete]['event'].unique().tolist())
                    selected_throws_event = st.selectbox(
                        "Select Event:",
                        options=throws_chart_events if throws_chart_events else ['No events'],
                        key="throws_chart_event"
                    )

                # Chart filters - Row 2: Implement Weight and Session Type
                tc_col3, tc_col4 = st.columns(2)

                # Get available implement weights for this athlete/event
                throws_athlete_event_df = throws_chart_df[
                    (throws_chart_df['athlete'] == selected_throws_athlete) &
                    (throws_chart_df['event'] == selected_throws_event)
                ]

                with tc_col3:
                    if 'implement_kg' in throws_athlete_event_df.columns:
                        throws_implement_weights = ['All'] + sorted([str(w) for w in throws_athlete_event_df['implement_kg'].unique() if pd.notna(w)])
                        selected_throws_implement = st.selectbox(
                            "Implement Weight:",
                            options=throws_implement_weights,
                            key="throws_chart_implement"
                        )
                    else:
                        selected_throws_implement = 'All'

                with tc_col4:
                    throws_session_options = ['All', 'Training', 'Competition', 'Testing', 'Warm-up']
                    selected_throws_session = st.selectbox(
                        "Session Type:",
                        options=throws_session_options,
                        key="throws_chart_session"
                    )

                # Filter data for charts
                filtered_throws_df = throws_chart_df[
                    (throws_chart_df['athlete'] == selected_throws_athlete) &
                    (throws_chart_df['event'] == selected_throws_event)
                ]

                # Apply implement weight filter
                if selected_throws_implement != 'All' and 'implement_kg' in filtered_throws_df.columns:
                    filtered_throws_df = filtered_throws_df[filtered_throws_df['implement_kg'].astype(str) == selected_throws_implement]

                # Apply session type filter
                if selected_throws_session != 'All' and 'session_type' in filtered_throws_df.columns:
                    filtered_throws_df = filtered_throws_df[filtered_throws_df['session_type'] == selected_throws_session]

                filtered_throws_df = filtered_throws_df.sort_values('date')

                if not filtered_throws_df.empty:
                    # Distance over time chart with color by session type
                    st.markdown("#### 📈 Distance Progression")
                    st.markdown("*🟢 Training | 🟡 Competition | 🔵 Testing | ⚪ Warm-up*")

                    fig_throws = go.Figure()

                    # Color mapping for session types
                    session_colors = {
                        'Training': '#007167',
                        'Competition': '#FFB800',
                        'Testing': '#0077B6',
                        'Warm-up': '#6c757d'
                    }

                    # Add scatter points colored by session type
                    if 'session_type' in filtered_throws_df.columns:
                        for session_type in filtered_throws_df['session_type'].unique():
                            session_data = filtered_throws_df[filtered_throws_df['session_type'] == session_type]
                            color = session_colors.get(session_type, '#007167')

                            fig_throws.add_trace(go.Scatter(
                                x=session_data['date'],
                                y=session_data['distance_m'],
                                mode='markers',
                                name=session_type,
                                marker=dict(
                                    size=12 if session_type == 'Competition' else 10,
                                    color=color,
                                    symbol='star' if session_type == 'Competition' else 'circle',
                                    line=dict(width=1, color='white')
                                ),
                                hovertemplate=f'<b>%{{x|%d %b %Y}}</b><br>{session_type}<br>Distance: %{{y:.2f}}m<extra></extra>'
                            ))

                        # Add connecting line
                        fig_throws.add_trace(go.Scatter(
                            x=filtered_throws_df['date'],
                            y=filtered_throws_df['distance_m'],
                            mode='lines',
                            name='Trend',
                            line=dict(color='rgba(0,113,103,0.3)', width=1),
                            hoverinfo='skip',
                            showlegend=False
                        ))
                    else:
                        fig_throws.add_trace(go.Scatter(
                            x=filtered_throws_df['date'],
                            y=filtered_throws_df['distance_m'],
                            mode='markers+lines',
                            name='Throws',
                            marker=dict(size=10, color='#007167'),
                            line=dict(color='#007167', width=2),
                            hovertemplate='<b>%{x|%d %b %Y}</b><br>Distance: %{y:.2f}m<extra></extra>'
                        ))

                    # Add best distance line
                    best_dist = filtered_throws_df['distance_m'].max()
                    fig_throws.add_hline(
                        y=best_dist,
                        line_dash="dash",
                        line_color="#a08e66",
                        annotation_text=f"PB: {best_dist:.2f}m",
                        annotation_position="top right"
                    )

                    fig_throws.update_layout(
                        title=f"{selected_throws_athlete} - {selected_throws_event} Progression",
                        xaxis_title="Date",
                        yaxis_title="Distance (m)",
                        height=450,
                        showlegend=True,
                        legend=dict(orientation='h', yanchor='bottom', y=1.02),
                        plot_bgcolor='white',
                        font=dict(family='Inter, sans-serif')
                    )
                    fig_throws.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    fig_throws.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

                    st.plotly_chart(fig_throws, use_container_width=True)

                    # Summary stats
                    st.markdown("#### 📋 Summary Statistics")
                    ts_col1, ts_col2, ts_col3, ts_col4 = st.columns(4)
                    with ts_col1:
                        st.metric("Total Throws", len(filtered_throws_df))
                    with ts_col2:
                        st.metric("Best Distance", f"{filtered_throws_df['distance_m'].max():.2f}m")
                    with ts_col3:
                        st.metric("Average", f"{filtered_throws_df['distance_m'].mean():.2f}m")
                    with ts_col4:
                        std_dev = filtered_throws_df['distance_m'].std()
                        st.metric("Std Dev", f"{std_dev:.2f}m" if pd.notna(std_dev) else "N/A")
                else:
                    st.info("No data available for the selected athlete and event")
            else:
                st.info("📭 No throws data available. Add entries in the 'Throws Distance' tab to see charts.")

        # -----------------------------------------------------------------
        # CHARTS SUB-TAB: S&C Upper Body
        # -----------------------------------------------------------------
        with chart_sub_tabs[1]:
            upper_chart_df = st.session_state.sc_upper_body

            if not upper_chart_df.empty and 'date' in upper_chart_df.columns:
                upper_chart_df['date'] = pd.to_datetime(upper_chart_df['date'])

                # Chart filters
                uc_col1, uc_col2 = st.columns(2)

                with uc_col1:
                    upper_chart_athletes = sorted(upper_chart_df['athlete'].unique().tolist())
                    selected_upper_athlete = st.selectbox(
                        "Select Athlete:",
                        options=upper_chart_athletes,
                        key="upper_chart_athlete"
                    )

                with uc_col2:
                    upper_exercises = sorted(upper_chart_df[upper_chart_df['athlete'] == selected_upper_athlete]['exercise'].unique().tolist())
                    selected_upper_exercise = st.selectbox(
                        "Select Exercise:",
                        options=upper_exercises if upper_exercises else ['No exercises'],
                        key="upper_chart_exercise"
                    )

                # Filter data
                filtered_upper_df = upper_chart_df[
                    (upper_chart_df['athlete'] == selected_upper_athlete) &
                    (upper_chart_df['exercise'] == selected_upper_exercise)
                ].sort_values('date')

                if not filtered_upper_df.empty:
                    # 1RM Progression Chart
                    st.markdown("#### 📈 Estimated 1RM Progression")

                    fig_upper_1rm = go.Figure()
                    fig_upper_1rm.add_trace(go.Scatter(
                        x=filtered_upper_df['date'],
                        y=filtered_upper_df['estimated_1rm'],
                        mode='markers+lines',
                        name='Estimated 1RM',
                        marker=dict(size=10, color='#007167'),
                        line=dict(color='#007167', width=2),
                        hovertemplate='<b>%{x|%d %b %Y}</b><br>Est. 1RM: %{y:.1f}kg<extra></extra>'
                    ))

                    best_upper_1rm = filtered_upper_df['estimated_1rm'].max()
                    fig_upper_1rm.add_hline(
                        y=best_upper_1rm,
                        line_dash="dash",
                        line_color="#a08e66",
                        annotation_text=f"Best: {best_upper_1rm:.1f}kg",
                        annotation_position="top right"
                    )

                    fig_upper_1rm.update_layout(
                        title=f"{selected_upper_athlete} - {selected_upper_exercise} 1RM Progression",
                        xaxis_title="Date",
                        yaxis_title="Estimated 1RM (kg)",
                        height=400,
                        plot_bgcolor='white',
                        font=dict(family='Inter, sans-serif')
                    )
                    fig_upper_1rm.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    fig_upper_1rm.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

                    st.plotly_chart(fig_upper_1rm, use_container_width=True)

                    # Summary stats
                    st.markdown("#### 📋 Summary Statistics")
                    us_col1, us_col2, us_col3, us_col4 = st.columns(4)
                    with us_col1:
                        st.metric("Total Sessions", len(filtered_upper_df))
                    with us_col2:
                        st.metric("Best Est. 1RM", f"{filtered_upper_df['estimated_1rm'].max():.1f}kg")
                    with us_col3:
                        st.metric("Max Weight Used", f"{filtered_upper_df['weight_kg'].max():.1f}kg")
                    with us_col4:
                        avg_rpe_upper = filtered_upper_df['rpe'].mean() if 'rpe' in filtered_upper_df.columns else None
                        st.metric("Avg RPE", f"{avg_rpe_upper:.1f}" if pd.notna(avg_rpe_upper) else "N/A")

                    # Personal Bests
                    st.markdown("#### 🏆 All Personal Bests")
                    athlete_upper_all = upper_chart_df[upper_chart_df['athlete'] == selected_upper_athlete]
                    if not athlete_upper_all.empty:
                        pb_upper_table = athlete_upper_all.groupby('exercise').agg({
                            'estimated_1rm': 'max',
                            'weight_kg': 'max',
                            'date': 'max'
                        }).reset_index()
                        pb_upper_table.columns = ['Exercise', 'Best Est. 1RM (kg)', 'Max Weight (kg)', 'Last Session']
                        pb_upper_table = pb_upper_table.sort_values('Best Est. 1RM (kg)', ascending=False)
                        st.dataframe(pb_upper_table, use_container_width=True, hide_index=True)
                else:
                    st.info("No data for selected athlete and exercise")
            else:
                st.info("📭 No S&C Upper Body data available. Add entries in the 'S&C Upper Body' tab to see charts.")

        # -----------------------------------------------------------------
        # CHARTS SUB-TAB: S&C Lower Body
        # -----------------------------------------------------------------
        with chart_sub_tabs[2]:
            lower_chart_df = st.session_state.sc_lower_body

            if not lower_chart_df.empty and 'date' in lower_chart_df.columns:
                lower_chart_df['date'] = pd.to_datetime(lower_chart_df['date'])

                # Chart filters
                lc_col1, lc_col2 = st.columns(2)

                with lc_col1:
                    lower_chart_athletes = sorted(lower_chart_df['athlete'].unique().tolist())
                    selected_lower_athlete = st.selectbox(
                        "Select Athlete:",
                        options=lower_chart_athletes,
                        key="lower_chart_athlete"
                    )

                with lc_col2:
                    lower_exercises = sorted(lower_chart_df[lower_chart_df['athlete'] == selected_lower_athlete]['exercise'].unique().tolist())
                    selected_lower_exercise = st.selectbox(
                        "Select Exercise:",
                        options=lower_exercises if lower_exercises else ['No exercises'],
                        key="lower_chart_exercise"
                    )

                # Filter data
                filtered_lower_df = lower_chart_df[
                    (lower_chart_df['athlete'] == selected_lower_athlete) &
                    (lower_chart_df['exercise'] == selected_lower_exercise)
                ].sort_values('date')

                if not filtered_lower_df.empty:
                    # 1RM Progression Chart
                    st.markdown("#### 📈 Estimated 1RM Progression")

                    fig_lower_1rm = go.Figure()
                    fig_lower_1rm.add_trace(go.Scatter(
                        x=filtered_lower_df['date'],
                        y=filtered_lower_df['estimated_1rm'],
                        mode='markers+lines',
                        name='Estimated 1RM',
                        marker=dict(size=10, color='#007167'),
                        line=dict(color='#007167', width=2),
                        hovertemplate='<b>%{x|%d %b %Y}</b><br>Est. 1RM: %{y:.1f}kg<extra></extra>'
                    ))

                    best_lower_1rm = filtered_lower_df['estimated_1rm'].max()
                    fig_lower_1rm.add_hline(
                        y=best_lower_1rm,
                        line_dash="dash",
                        line_color="#a08e66",
                        annotation_text=f"Best: {best_lower_1rm:.1f}kg",
                        annotation_position="top right"
                    )

                    fig_lower_1rm.update_layout(
                        title=f"{selected_lower_athlete} - {selected_lower_exercise} 1RM Progression",
                        xaxis_title="Date",
                        yaxis_title="Estimated 1RM (kg)",
                        height=400,
                        plot_bgcolor='white',
                        font=dict(family='Inter, sans-serif')
                    )
                    fig_lower_1rm.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    fig_lower_1rm.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

                    st.plotly_chart(fig_lower_1rm, use_container_width=True)

                    # Summary stats
                    st.markdown("#### 📋 Summary Statistics")
                    ls_col1, ls_col2, ls_col3, ls_col4 = st.columns(4)
                    with ls_col1:
                        st.metric("Total Sessions", len(filtered_lower_df))
                    with ls_col2:
                        st.metric("Best Est. 1RM", f"{filtered_lower_df['estimated_1rm'].max():.1f}kg")
                    with ls_col3:
                        st.metric("Max Weight Used", f"{filtered_lower_df['weight_kg'].max():.1f}kg")
                    with ls_col4:
                        avg_rpe_lower = filtered_lower_df['rpe'].mean() if 'rpe' in filtered_lower_df.columns else None
                        st.metric("Avg RPE", f"{avg_rpe_lower:.1f}" if pd.notna(avg_rpe_lower) else "N/A")

                    # Personal Bests
                    st.markdown("#### 🏆 All Personal Bests")
                    athlete_lower_all = lower_chart_df[lower_chart_df['athlete'] == selected_lower_athlete]
                    if not athlete_lower_all.empty:
                        pb_lower_table = athlete_lower_all.groupby('exercise').agg({
                            'estimated_1rm': 'max',
                            'weight_kg': 'max',
                            'date': 'max'
                        }).reset_index()
                        pb_lower_table.columns = ['Exercise', 'Best Est. 1RM (kg)', 'Max Weight (kg)', 'Last Session']
                        pb_lower_table = pb_lower_table.sort_values('Best Est. 1RM (kg)', ascending=False)
                        st.dataframe(pb_lower_table, use_container_width=True, hide_index=True)
                else:
                    st.info("No data for selected athlete and exercise")
            else:
                st.info("📭 No S&C Lower Body data available. Add entries in the 'S&C Lower Body' tab to see charts.")

# ============================================================================
# TAB 11: FORCE TRACE ANALYSIS
# ============================================================================



with tabs[6]:  # Trace (was tabs[8], now tabs[6])
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
    # Note: trialId is optional - the new API uses /recording endpoint which only needs testId
    has_test_ids = 'testId' in filtered_df.columns

    if has_test_ids and env_loaded:
        # Step 1: Select Athlete
        st.markdown("#### Step 1: Select Athlete")
        trace_athletes = sorted([n for n in filtered_df['Name'].unique() if pd.notna(n)])
        selected_trace_athlete = st.selectbox(
            "Choose athlete to view available tests:",
            trace_athletes if trace_athletes else ["No athletes available"],
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

                            # Check if we got valid DataFrames (not empty and not None)
                            trace1_valid = isinstance(trace1, pd.DataFrame) and not trace1.empty
                            trace2_valid = isinstance(trace2, pd.DataFrame) and not trace2.empty

                            if trace1_valid and trace2_valid:
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
                                # Use trapezoid (trapz removed in NumPy 2.0)
                                try:
                                    impulse1 = np.trapezoid(force1, time1) / 1000  # Convert to N·s
                                    impulse2 = np.trapezoid(force2, time2) / 1000
                                except AttributeError:
                                    impulse1 = np.trapz(force1, time1) / 1000
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

                                # Debug info
                                with st.expander("🔍 Debug Information"):
                                    st.markdown("**Trace 1 Status:**")
                                    if isinstance(trace1, pd.DataFrame):
                                        st.write(f"DataFrame with {len(trace1)} rows, columns: {list(trace1.columns)}")
                                    else:
                                        st.write(f"Type: {type(trace1)}, Value: {trace1}")

                                    st.markdown("**Trace 2 Status:**")
                                    if isinstance(trace2, pd.DataFrame):
                                        st.write(f"DataFrame with {len(trace2)} rows, columns: {list(trace2.columns)}")
                                    else:
                                        st.write(f"Type: {type(trace2)}, Value: {trace2}")

                                    st.markdown("**Request Details:**")
                                    st.code(f"""
Test 1: testId={test1_data['testId']}
        trialId={test1_data['trialId']}

Test 2: testId={test2_data['testId']}
        trialId={test2_data['trialId']}

Region: {env_creds['region']}
Token: {env_creds['token'][:20]}... (truncated)
Tenant: {env_creds['tenant_id']}
""")

                                st.info("💡 **Tip:** Force trace data may not be available for all tests. The API endpoint varies by test type and VALD version.")

                        except Exception as e:
                            st.error(f"❌ Error fetching traces: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.error("Please select both tests to compare.")

        else:
            st.info("No tests available for this athlete with testId data.")

    elif not has_test_ids:
        st.warning("""
        **Your data doesn't have a testId column.**

        This is required to fetch force traces from the VALD API.

        **To fix this:**
        1. Use the VALD API pull scripts to download data with full metadata
        2. Ensure your export includes the testId column
        3. Re-upload the data with this column included
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
    # MULTI-ATHLETE COMPARISON
    # =========================================================================
    st.markdown("### 👥 Multi-Athlete Force Trace Comparison")

    if not FORCE_TRACE_AVAILABLE:
        st.warning("Force trace module not available. Please ensure utils/force_trace_viz.py is present.")
    elif not has_test_ids:
        st.warning("Test IDs not available in data. Multi-athlete comparison requires testId column.")
    elif not env_loaded:
        st.warning("API credentials not configured. Please add credentials to your .env file.")
    else:
        st.markdown("*Select multiple athletes to compare their force traces side-by-side*")

        # Get athletes with test data (filter out NaN values)
        athletes_with_tests = sorted([n for n in filtered_df['Name'].unique() if pd.notna(n)])

        # Multi-select for athletes (up to 5)
        selected_athletes_compare = st.multiselect(
            "Select Athletes to Compare (max 5):",
            options=athletes_with_tests,
            max_selections=5,
            key="multi_athlete_compare"
        )

        if len(selected_athletes_compare) >= 2:
            # For each selected athlete, show their most recent test
            st.markdown("#### Select Test for Each Athlete")

            athlete_test_selections = {}

            cols = st.columns(len(selected_athletes_compare))

            for i, athlete in enumerate(selected_athletes_compare):
                with cols[i]:
                    st.markdown(f"**{athlete}**")

                    athlete_tests = filtered_df[filtered_df['Name'] == athlete].sort_values('recordedDateUtc', ascending=False)

                    if not athlete_tests.empty:
                        test_opts = []
                        for _, row in athlete_tests.head(10).iterrows():
                            date_str = row['recordedDateUtc'].strftime('%Y-%m-%d') if pd.notna(row.get('recordedDateUtc')) else 'N/A'
                            label = f"{row['testType']} - {date_str}"
                            test_opts.append({
                                'label': label,
                                'testId': str(row.get('testId', '')),
                                'trialId': str(row.get('trialId', ''))
                            })

                        if test_opts:
                            selected_idx = st.selectbox(
                                "Test:",
                                range(len(test_opts)),
                                format_func=lambda x, opts=test_opts: opts[x]['label'],
                                key=f"multi_compare_{athlete}"
                            )
                            athlete_test_selections[athlete] = test_opts[selected_idx]

            # Fetch and compare button
            if st.button("🔄 Fetch & Compare All Athletes", type="primary", key="fetch_multi_compare", use_container_width=True):
                if len(athlete_test_selections) >= 2:
                    with st.spinner("Fetching force traces for all athletes..."):
                        athlete_traces = {}
                        fetch_errors = []

                        for athlete, test_info in athlete_test_selections.items():
                            try:
                                trace = get_force_trace(
                                    test_info['testId'],
                                    test_info['trialId'],
                                    env_creds['token'],
                                    env_creds['tenant_id'],
                                    env_creds['region']
                                )
                                if trace is not None and not trace.empty:
                                    athlete_traces[athlete] = trace
                                else:
                                    fetch_errors.append(f"{athlete}: No data returned")
                            except Exception as e:
                                fetch_errors.append(f"{athlete}: {str(e)}")

                        if athlete_traces:
                            st.success(f"✅ Successfully fetched traces for {len(athlete_traces)} athletes")

                            # Plot comparison
                            fig_multi = plot_athlete_comparison(
                                athlete_traces,
                                test_type='CMJ',
                                title="Multi-Athlete Force Trace Comparison"
                            )
                            st.plotly_chart(fig_multi, use_container_width=True)

                            # Metrics comparison table
                            st.markdown("#### 📊 Comparison Metrics")
                            metrics_data = []
                            for athlete, trace in athlete_traces.items():
                                metrics = calculate_trace_metrics(trace, test_type='CMJ')
                                metrics_data.append({
                                    'Athlete': athlete,
                                    'Peak Force (N)': metrics.get('peak_force', 0),
                                    'Avg Force (N)': metrics.get('average_force', 0),
                                    'RFD (N/s)': metrics.get('rfd_100ms', 0),
                                    'Impulse (N·s)': metrics.get('impulse', 0)
                                })

                            metrics_df = pd.DataFrame(metrics_data)
                            st.dataframe(metrics_df.round(1), use_container_width=True, hide_index=True)

                        if fetch_errors:
                            with st.expander("⚠️ Fetch Errors"):
                                for err in fetch_errors:
                                    st.warning(err)
                else:
                    st.error("Please select tests for at least 2 athletes.")

        elif len(selected_athletes_compare) == 1:
            st.info("Select at least 2 athletes to compare their force traces.")
        else:
            st.info("Select athletes above to compare their force traces.")

    # Live Data Fetching Section
    st.markdown("---")
    st.markdown("### 🔌 Fetch Single Force Trace")

    st.info("""
    **Select specific tests to fetch force trace data from VALD API.**
    Only selected traces will be downloaded to minimize API calls.
    """)

    # Load credentials (function defined at top of file)
    env_creds_single, env_loaded_single = load_env_credentials()

    # API Configuration
    if env_loaded_single:
        st.success("✅ **API Configuration loaded**")

        # Use environment credentials
        api_token = env_creds_single['token']
        tenant_id = env_creds_single['tenant_id']
        region = env_creds_single['region']
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

    # Test Selection (trialId is optional - new API uses /recording endpoint)
    if 'testId' in filtered_df.columns:
        st.markdown("#### Select Test to Analyze")

        col1, col2 = st.columns(2)

        with col1:
            # Select athlete (filter out NaN values)
            trace_athlete = st.selectbox(
                "Select Athlete:",
                options=sorted([n for n in filtered_df['Name'].unique() if pd.notna(n)]),
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

        To fetch live traces, your data needs a `testId` column.
        This is available when data is pulled directly from the VALD API.
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

        if 'testId' in filtered_df.columns:
            # Smart filter options (trialId is optional - new API uses /recording)
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
                        options=sorted([n for n in filtered_df['Name'].unique() if pd.notna(n)]),
                        key="comp_athlete1"
                    )
                else:
                    athlete1 = st.selectbox(
                        "Athlete:",
                        options=sorted([n for n in filtered_df['Name'].unique() if pd.notna(n)]),
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
                    # Filter to different athlete (filter out NaN values)
                    other_athletes = [a for a in sorted([n for n in filtered_df['Name'].unique() if pd.notna(n)]) if a != athlete1]
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

                            # Check if we got valid DataFrames (not empty and not None/tuple)
                            trace1_valid = isinstance(trace1, pd.DataFrame) and not trace1.empty
                            trace2_valid = isinstance(trace2, pd.DataFrame) and not trace2.empty

                            if trace1_valid and trace2_valid:
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
# TAB: ADVANCED ANALYTICS (REMOVED - See docs/DASHBOARD_TABS_REFERENCE.md)
# ============================================================================

# ============================================================================
# TAB: ELITE INSIGHTS (REMOVED - See docs/DASHBOARD_TABS_REFERENCE.md)
# ============================================================================

# ============================================================================
# TAB 14: DATA TABLE
# ============================================================================



with tabs[7]:  # Data (was tabs[16], now tabs[7])
    st.markdown("## 📋 Data Table View")
    st.markdown("Browse and export all testing data in table format")

    # Filters for data table
    col1, col2, col3 = st.columns(3)

    with col1:
        # Test type filter (filter out NaN values)
        test_types = ['All'] + sorted([t for t in filtered_df['testType'].unique() if pd.notna(t)]) if 'testType' in filtered_df.columns else ['All']
        selected_test_type = st.selectbox("Test Type:", test_types, key="table_test_type")

    with col2:
        # Athlete filter (filter out NaN values)
        athletes = ['All'] + sorted([n for n in filtered_df['Name'].unique() if pd.notna(n)]) if 'Name' in filtered_df.columns else ['All']
        selected_athlete_table = st.selectbox("Athlete:", athletes, key="table_athlete")

    with col3:
        # Sport filter (filter out NaN values)
        sports = ['All'] + sorted([s for s in filtered_df['athlete_sport'].unique() if pd.notna(s)]) if 'athlete_sport' in filtered_df.columns else ['All']
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
# PAGE: SPORT REPORTS (NEW)
# ============================================================================

with tabs[1]:
    st.markdown("## 📊 Sport Reports")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #007167 0%, #005a51 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <p style="color: white; margin: 0; font-size: 0.95rem;">
            <strong>Group & Individual Reports</strong> • Benchmark zones • Trend analysis over time
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Import sport reports module
    try:
        from dashboard.utils.sport_reports import (
            create_group_report, create_group_report_v2, create_group_report_v3,
            create_individual_report, render_benchmark_legend, get_sport_benchmarks,
            create_shooting_group_report, create_shooting_individual_report
        )
        sport_reports_available = True
    except ImportError:
        try:
            from utils.sport_reports import (
                create_group_report, create_group_report_v2, create_group_report_v3,
                create_individual_report, render_benchmark_legend, get_sport_benchmarks,
                create_shooting_group_report, create_shooting_individual_report
            )
            sport_reports_available = True
        except ImportError:
            sport_reports_available = False
            st.error("Sport reports module not found. Please check utils/sport_reports.py")

    if sport_reports_available:
        # Sport selector
        available_sports = sorted(filtered_df['athlete_sport'].dropna().unique()) if 'athlete_sport' in filtered_df.columns else []

        if available_sports:
            selected_report_sport = st.selectbox(
                "Select Sport:",
                options=available_sports,
                key="report_sport_selector"
            )

            # Report type tabs - Group v1, v2, v3, Individual, Shooting, S&C Diagnostics, and Benchmark Settings
            report_tabs = st.tabs(["👥 Group v1", "📊 Group v2", "📈 Group v3", "🏃 Individual Report", "🎯 Shooting Balance", "🏋️ S&C Diagnostics", "⚙️ Benchmark Settings"])

            # Render benchmark legend
            render_benchmark_legend()

            # Filter data for selected sport (used by all tabs)
            sport_mask = filtered_df['athlete_sport'].str.contains(
                selected_report_sport.split()[0], case=False, na=False
            ) if 'athlete_sport' in filtered_df.columns else pd.Series([True] * len(filtered_df))

            sport_data = filtered_df[sport_mask].copy()

            # Filter ForceFrame and NordBord for sport
            # Note: ForceFrame/NordBord may have limited sport info from Profiles API
            # If sport filter returns empty, show all data instead
            sport_ff = filtered_forceframe.copy() if not filtered_forceframe.empty else pd.DataFrame()
            sport_nb = filtered_nordbord.copy() if not filtered_nordbord.empty else pd.DataFrame()

            if 'athlete_sport' in sport_ff.columns and not sport_ff.empty:
                sport_keyword = selected_report_sport.split()[0]
                filtered_ff = sport_ff[sport_ff['athlete_sport'].str.contains(
                    sport_keyword, case=False, na=False
                )]
                # Only apply filter if it returns results, otherwise show all
                if not filtered_ff.empty:
                    sport_ff = filtered_ff

            if 'athlete_sport' in sport_nb.columns and not sport_nb.empty:
                sport_keyword = selected_report_sport.split()[0]
                filtered_nb = sport_nb[sport_nb['athlete_sport'].str.contains(
                    sport_keyword, case=False, na=False
                )]
                # Only apply filter if it returns results, otherwise show all
                if not filtered_nb.empty:
                    sport_nb = filtered_nb

            with report_tabs[0]:
                # Group Report v1 - Bar charts with benchmark zones
                st.markdown("### Team Performance Overview")
                st.caption("Bar charts with shaded benchmark zones")

                create_group_report(
                    sport_data,
                    selected_report_sport,
                    forceframe_df=sport_ff if not sport_ff.empty else None,
                    nordbord_df=sport_nb if not sport_nb.empty else None
                )

            with report_tabs[1]:
                # Group Report v2 - Summary table with RAG status
                st.markdown("### Team Summary View")
                st.caption("Summary table with RAG status indicators")

                create_group_report_v2(
                    sport_data,
                    selected_report_sport,
                    forceframe_df=sport_ff if not sport_ff.empty else None,
                    nordbord_df=sport_nb if not sport_nb.empty else None
                )

            with report_tabs[2]:
                # Group Report v3 - Alternative visualizations
                st.markdown("### Alternative Visualizations")
                st.caption("Bullet charts, radar plots, and distribution analysis")

                create_group_report_v3(
                    sport_data,
                    selected_report_sport,
                    forceframe_df=sport_ff if not sport_ff.empty else None,
                    nordbord_df=sport_nb if not sport_nb.empty else None
                )

            with report_tabs[3]:
                # Individual Report
                st.markdown("### Individual Athlete Analysis")

                # Get athletes for selected sport
                sport_athletes = sorted(
                    filtered_df[sport_mask]['Name'].dropna().unique()
                ) if 'Name' in filtered_df.columns else []

                if sport_athletes:
                    selected_report_athlete = st.selectbox(
                        "Select Athlete:",
                        options=sport_athletes,
                        key="report_athlete_selector"
                    )

                    create_individual_report(
                        filtered_df,
                        selected_report_athlete,
                        selected_report_sport,
                        forceframe_df=filtered_forceframe if not filtered_forceframe.empty else None,
                        nordbord_df=filtered_nordbord if not filtered_nordbord.empty else None
                    )
                else:
                    st.info("No athletes found for the selected sport")

            with report_tabs[4]:
                # Shooting Balance Report - 10m Pistol
                st.markdown("### 🎯 10m Pistol - Balance Analysis")
                st.caption("Quiet Standing Balance tests for shooting athletes")

                # Create sub-tabs for Group and Individual
                shooting_tabs = st.tabs(["👥 Group Report", "🏃 Individual Report"])

                with shooting_tabs[0]:
                    # Group Shooting Report
                    create_shooting_group_report(filtered_df, "Shooting")

                with shooting_tabs[1]:
                    # Individual Shooting Report
                    # Get athletes who have QSB data
                    qsb_athletes = []
                    if 'testType' in filtered_df.columns:
                        qsb_df = filtered_df[filtered_df['testType'] == 'QSB']
                        if 'Name' in qsb_df.columns:
                            qsb_athletes = sorted(qsb_df['Name'].dropna().unique())
                        elif 'full_name' in qsb_df.columns:
                            qsb_athletes = sorted(qsb_df['full_name'].dropna().unique())

                    if qsb_athletes:
                        selected_shooting_athlete = st.selectbox(
                            "Select Athlete:",
                            options=qsb_athletes,
                            key="shooting_athlete_selector"
                        )

                        create_shooting_individual_report(
                            filtered_df,
                            selected_shooting_athlete,
                            "Shooting"
                        )
                    else:
                        st.info("No athletes with Quiet Standing Balance (QSB) tests found. Ensure athletes have completed QSB tests on ForceDecks.")

            with report_tabs[5]:
                # S&C Diagnostics Canvas
                try:
                    from dashboard.utils.snc_diagnostics import render_snc_diagnostics_tab
                except ImportError:
                    try:
                        from utils.snc_diagnostics import render_snc_diagnostics_tab
                    except ImportError:
                        render_snc_diagnostics_tab = None

                if render_snc_diagnostics_tab:
                    render_snc_diagnostics_tab(
                        filtered_df,
                        nordbord_df=filtered_nordbord if not filtered_nordbord.empty else None,
                        forceframe_df=filtered_forceframe if not filtered_forceframe.empty else None
                    )
                else:
                    st.error("S&C Diagnostics module not found. Please check utils/snc_diagnostics.py")

            with report_tabs[6]:
                # Benchmark Settings - S&C staff can edit VALD norms
                try:
                    from dashboard.utils.benchmark_database import render_benchmark_editor
                except ImportError:
                    try:
                        from utils.benchmark_database import render_benchmark_editor
                    except ImportError:
                        render_benchmark_editor = None

                if render_benchmark_editor:
                    render_benchmark_editor()
                else:
                    st.error("Benchmark database module not found. Please check utils/benchmark_database.py")
        else:
            st.warning("No sports found in the data. Check athlete_sport column.")

# ============================================================================
# PAGE: FORCEFRAME
# ============================================================================

with tabs[2]:
    st.markdown("## 🔲 ForceFrame Isometric Strength Analysis")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1D4D3B 0%, #2d6a5a 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <p style="color: white; margin: 0; font-size: 0.95rem;">
            <strong>35+ test positions</strong> • Bilateral strength assessment • Real-time asymmetry monitoring
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ForceFrame thresholds (from research)
    FORCEFRAME_THRESHOLDS = {
        'hip_adduction_min': 250,     # Minimum acceptable adduction force (N)
        'hip_abduction_min': 200,     # Minimum acceptable abduction force (N)
        'add_abd_ratio_max': 1.3,     # ADD:ABD ratio >1.3 = groin pain risk
        'asymmetry_threshold': 10,    # >10% asymmetry flag
    }

    # Use filtered data
    ff_data = filtered_forceframe if not filtered_forceframe.empty else df_forceframe

    if ff_data.empty:
        st.warning("No ForceFrame data available. Upload data or check data/forceframe_allsports.csv")
    else:
        # Create subtabs for ForceFrame - reorganized for better workflow
        ff_tabs = st.tabs(["📊 Overview", "🦴 Body Region", "🏃 Individual Athlete", "⚖️ Asymmetry Dashboard", "📈 Progression"])

        with ff_tabs[0]:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                n_athletes = ff_data['Name'].nunique() if 'Name' in ff_data.columns else (ff_data['athleteId'].nunique() if 'athleteId' in ff_data.columns else 0)
                st.metric("Athletes", n_athletes)

            with col2:
                st.metric("Total Tests", len(ff_data))

            with col3:
                n_types = ff_data['testTypeName'].nunique() if 'testTypeName' in ff_data.columns else 0
                st.metric("Test Types", n_types)

            with col4:
                if 'testDateUtc' in ff_data.columns:
                    latest = pd.to_datetime(ff_data['testDateUtc']).max()
                    st.metric("Latest Test", latest.strftime('%Y-%m-%d') if pd.notna(latest) else "N/A")

            st.markdown("---")

            # Test type distribution
            if 'testTypeName' in ff_data.columns:
                st.markdown("### Test Type Distribution")
                test_counts = ff_data['testTypeName'].value_counts()

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

            # Force comparison - Left vs Right
            st.markdown("### Left vs Right Force Comparison")

            left_col = 'innerLeftMaxForce' if 'innerLeftMaxForce' in ff_data.columns else None
            right_col = 'innerRightMaxForce' if 'innerRightMaxForce' in ff_data.columns else None

            if left_col and right_col:
                plot_df = ff_data[[left_col, right_col]].dropna()

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
            display_cols = ['Name', 'testDateUtc', 'testTypeName', 'innerLeftMaxForce', 'innerRightMaxForce',
                           'outerLeftMaxForce', 'outerRightMaxForce']
            available_cols = [c for c in display_cols if c in ff_data.columns]

            if available_cols:
                display_ff = ff_data[available_cols].copy()
                if 'testDateUtc' in display_ff.columns:
                    display_ff['testDateUtc'] = pd.to_datetime(display_ff['testDateUtc']).dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(display_ff, use_container_width=True, height=400)

            csv_ff = ff_data.to_csv(index=False)
            st.download_button("📥 Download ForceFrame Data", csv_ff,
                              f"forceframe_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

        # ========== BODY REGION TAB ==========
        with ff_tabs[1]:
            st.markdown("### 🦴 Body Region Analysis")
            st.markdown("*Grouped by anatomical region - Lower Body, Upper Body, Core/Neck*")

            # Define body region groupings
            body_regions = {
                'Lower Body - Hip': ['Hip AD/AB', 'Hip Flexion', 'Hip IR/ER', 'Hip Extension'],
                'Lower Body - Knee': ['Knee Extension', 'Knee Flexion'],
                'Lower Body - Ankle': ['Ankle Plantar Flexion', 'Ankle Dorsiflexion', 'Ankle IN/EV'],
                'Upper Body - Shoulder': ['Shoulder IR/ER', 'Shoulder Adduction', 'Shoulder Abduction',
                                          'Shoulder Extension', 'Shoulder Flexion'],
                'Upper Body - Elbow': ['Elbow Extension', 'Elbow Flexion'],
                'Core & Neck': ['Neck Extension', 'Neck Flexion', 'Neck Lateral Flexion', 'Chest Press']
            }

            if 'testTypeName' in ff_data.columns:
                # Map tests to regions
                def get_region(test_type):
                    for region, tests in body_regions.items():
                        for test in tests:
                            if test.lower() in str(test_type).lower():
                                return region
                    return 'Other'

                ff_data_region = ff_data.copy()
                ff_data_region['Body Region'] = ff_data_region['testTypeName'].apply(get_region)

                # Region summary cards
                st.markdown("#### Regional Test Distribution")
                region_counts = ff_data_region['Body Region'].value_counts()

                cols = st.columns(len(region_counts))
                region_colors = {
                    'Lower Body - Hip': '#e74c3c',
                    'Lower Body - Knee': '#3498db',
                    'Lower Body - Ankle': '#9b59b6',
                    'Upper Body - Shoulder': '#f39c12',
                    'Upper Body - Elbow': '#1abc9c',
                    'Core & Neck': '#34495e',
                    'Other': '#95a5a6'
                }

                for i, (region, count) in enumerate(region_counts.items()):
                    with cols[i]:
                        color = region_colors.get(region, '#1D4D3B')
                        st.markdown(f"""
                        <div style="background: {color}; padding: 1rem; border-radius: 8px; text-align: center; color: white;">
                            <h3 style="margin: 0; font-size: 2rem;">{count}</h3>
                            <p style="margin: 0; font-size: 0.8rem;">{region.split(' - ')[-1] if ' - ' in region else region}</p>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("---")

                # Select region for detailed view
                selected_region = st.selectbox(
                    "Select Body Region for Detailed Analysis:",
                    options=list(region_counts.index),
                    key="ff_region_select"
                )

                region_data = ff_data_region[ff_data_region['Body Region'] == selected_region]

                if not region_data.empty:
                    st.markdown(f"#### {selected_region} - Test Types")

                    # Show test types in this region
                    test_breakdown = region_data['testTypeName'].value_counts()

                    fig_region = px.bar(
                        x=test_breakdown.values,
                        y=test_breakdown.index,
                        orientation='h',
                        labels={'x': 'Number of Tests', 'y': 'Test Type'},
                        color=test_breakdown.values,
                        color_continuous_scale=['#1D4D3B', '#a08e66']
                    )
                    fig_region.update_layout(height=max(300, len(test_breakdown) * 40), showlegend=False)
                    st.plotly_chart(fig_region, use_container_width=True)

                    # Force comparison by test type
                    st.markdown(f"#### {selected_region} - Force Comparison")

                    force_cols = []
                    if 'outerLeftMaxForce' in region_data.columns:
                        force_cols = ['outerLeftMaxForce', 'outerRightMaxForce']
                    elif 'innerLeftMaxForce' in region_data.columns:
                        force_cols = ['innerLeftMaxForce', 'innerRightMaxForce']

                    if force_cols and len(region_data) > 0:
                        # Group by test type and get max forces
                        test_forces = region_data.groupby('testTypeName')[force_cols].max().reset_index()

                        fig_compare = go.Figure()
                        fig_compare.add_trace(go.Bar(
                            y=test_forces['testTypeName'],
                            x=test_forces[force_cols[0]],
                            name='Left',
                            orientation='h',
                            marker_color='#3498db'
                        ))
                        fig_compare.add_trace(go.Bar(
                            y=test_forces['testTypeName'],
                            x=test_forces[force_cols[1]],
                            name='Right',
                            orientation='h',
                            marker_color='#e74c3c'
                        ))
                        fig_compare.update_layout(
                            barmode='group',
                            xaxis_title='Peak Force (N)',
                            height=max(350, len(test_forces) * 50),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02)
                        )
                        st.plotly_chart(fig_compare, use_container_width=True)

                    # Athletes tested in this region
                    st.markdown(f"#### Athletes Tested - {selected_region}")
                    if 'Name' in region_data.columns:
                        athlete_summary = region_data.groupby('Name').agg({
                            'testTypeName': 'count',
                            force_cols[0] if force_cols else 'testId': 'max'
                        }).reset_index()
                        athlete_summary.columns = ['Athlete', 'Tests', 'Peak Force (N)' if force_cols else 'Tests']
                        st.dataframe(athlete_summary.sort_values('Tests', ascending=False), use_container_width=True, hide_index=True)
            else:
                st.warning("Test type information not available for body region grouping.")

        # ========== INDIVIDUAL ATHLETE TAB ==========
        with ff_tabs[2]:
            st.markdown("### 🏃 Individual Athlete Analysis")
            st.markdown("*Select an athlete and test type to view their ForceFrame performance profile with traffic light indicators*")

            if 'Name' in ff_data.columns and ff_data['Name'].nunique() > 0:
                # Get unique athlete names (filter out placeholder names)
                athlete_names = ff_data['Name'].dropna().unique()
                athlete_names = [n for n in athlete_names if not str(n).startswith('Athlete_')]

                if athlete_names:
                    col_sel1, col_sel2 = st.columns(2)

                    with col_sel1:
                        selected_ff_athlete = st.selectbox(
                            "Select Athlete:",
                            options=sorted(athlete_names),
                            key="ff_athlete_select"
                        )

                    # Get test types for this athlete
                    athlete_ff = ff_data[ff_data['Name'] == selected_ff_athlete].copy()

                    with col_sel2:
                        if 'testTypeName' in athlete_ff.columns:
                            test_types = sorted(athlete_ff['testTypeName'].dropna().unique())
                            selected_test_type = st.selectbox(
                                "Select Test Type:",
                                options=test_types,
                                key="ff_test_type_select"
                            )
                            # Filter by test type
                            athlete_ff = athlete_ff[athlete_ff['testTypeName'] == selected_test_type]
                        else:
                            selected_test_type = "All Tests"

                    if not athlete_ff.empty:
                        # Athlete summary with traffic light status cards
                        st.markdown(f"#### {selected_ff_athlete} - {selected_test_type}")

                        # Calculate key metrics for latest test
                        if 'testDateUtc' in athlete_ff.columns:
                            athlete_ff['testDateUtc'] = pd.to_datetime(athlete_ff['testDateUtc'])
                            athlete_ff = athlete_ff.sort_values('testDateUtc', ascending=False)
                            latest = athlete_ff.iloc[0]
                        else:
                            latest = athlete_ff.iloc[-1]

                        # Get force values
                        left_force = latest.get('innerLeftMaxForce', 0) or 0
                        right_force = latest.get('innerRightMaxForce', 0) or 0

                        # Calculate asymmetry
                        if left_force > 0 and right_force > 0:
                            asym = ((right_force - left_force) / ((right_force + left_force) / 2) * 100)
                        else:
                            asym = 0

                        # Determine traffic light colors
                        def get_asym_status_ff(asym_val):
                            """Return status color and text for ForceFrame asymmetry"""
                            if abs(asym_val) < 10:
                                return "#27AE60", "NORMAL", "✅"  # Green
                            elif abs(asym_val) < 15:
                                return "#F39C12", "CAUTION", "⚠️"  # Yellow
                            else:
                                return "#E74C3C", "AT RISK", "🔴"  # Red

                        asym_color, asym_status, asym_icon = get_asym_status_ff(asym)

                        # Traffic light status cards
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1D4D3B 0%, #153829 100%); padding: 1rem; border-radius: 12px; text-align: center; border-left: 4px solid #a08e66;">
                                <div style="font-size: 0.85rem; color: #a08e66; font-weight: 600;">TOTAL TESTS</div>
                                <div style="font-size: 2rem; font-weight: 700; color: white;">{len(athlete_ff)}</div>
                                <div style="font-size: 0.75rem; color: #8fb7b3;">{selected_test_type}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1D4D3B 0%, #153829 100%); padding: 1rem; border-radius: 12px; text-align: center; border-left: 4px solid #3498db;">
                                <div style="font-size: 0.85rem; color: #3498db; font-weight: 600;">LEFT PEAK</div>
                                <div style="font-size: 2rem; font-weight: 700; color: white;">{left_force:.0f}</div>
                                <div style="font-size: 0.75rem; color: #8fb7b3;">Newtons</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1D4D3B 0%, #153829 100%); padding: 1rem; border-radius: 12px; text-align: center; border-left: 4px solid #e74c3c;">
                                <div style="font-size: 0.85rem; color: #e74c3c; font-weight: 600;">RIGHT PEAK</div>
                                <div style="font-size: 2rem; font-weight: 700; color: white;">{right_force:.0f}</div>
                                <div style="font-size: 0.75rem; color: #8fb7b3;">Newtons</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with col4:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1D4D3B 0%, #153829 100%); padding: 1rem; border-radius: 12px; text-align: center; border-left: 4px solid {asym_color};">
                                <div style="font-size: 0.85rem; color: {asym_color}; font-weight: 600;">ASYMMETRY</div>
                                <div style="font-size: 2rem; font-weight: 700; color: white;">{abs(asym):.1f}%</div>
                                <div style="font-size: 0.75rem; color: {asym_color};">{asym_icon} {asym_status}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)

                        # Horizontal asymmetry bar chart (Patrick Ward style) - Test History
                        if 'testDateUtc' in athlete_ff.columns and len(athlete_ff) > 1:
                            st.markdown("#### 📊 Asymmetry History (Horizontal Bar Chart)")
                            st.markdown("*Each bar represents one test session - color indicates risk level*")

                            # Calculate asymmetry for each test
                            test_asym_data = []
                            for _, row in athlete_ff.iterrows():
                                left = row.get('innerLeftMaxForce', 0) or 0
                                right = row.get('innerRightMaxForce', 0) or 0
                                if left > 0 and right > 0:
                                    test_asym = ((right - left) / ((right + left) / 2) * 100)
                                    test_date = row['testDateUtc'].strftime('%Y-%m-%d') if pd.notna(row.get('testDateUtc')) else 'N/A'
                                    test_asym_data.append({
                                        'Date': test_date,
                                        'Asymmetry': test_asym,
                                        'Left': left,
                                        'Right': right
                                    })

                            if test_asym_data:
                                asym_history_df = pd.DataFrame(test_asym_data)

                                # Assign colors based on asymmetry level
                                def get_bar_color(val):
                                    if abs(val) < 10:
                                        return '#27AE60'  # Green
                                    elif abs(val) < 15:
                                        return '#F39C12'  # Yellow
                                    else:
                                        return '#E74C3C'  # Red

                                bar_colors = [get_bar_color(v) for v in asym_history_df['Asymmetry']]

                                fig_asym_hist = go.Figure()

                                # Add colored zones
                                fig_asym_hist.add_vrect(x0=-10, x1=10, fillcolor="rgba(39, 174, 96, 0.1)", layer="below", line_width=0)
                                fig_asym_hist.add_vrect(x0=-15, x1=-10, fillcolor="rgba(243, 156, 18, 0.1)", layer="below", line_width=0)
                                fig_asym_hist.add_vrect(x0=10, x1=15, fillcolor="rgba(243, 156, 18, 0.1)", layer="below", line_width=0)
                                fig_asym_hist.add_vrect(x0=-30, x1=-15, fillcolor="rgba(231, 76, 60, 0.1)", layer="below", line_width=0)
                                fig_asym_hist.add_vrect(x0=15, x1=30, fillcolor="rgba(231, 76, 60, 0.1)", layer="below", line_width=0)

                                # Add horizontal bars
                                fig_asym_hist.add_trace(go.Bar(
                                    y=asym_history_df['Date'],
                                    x=asym_history_df['Asymmetry'],
                                    orientation='h',
                                    marker_color=bar_colors,
                                    text=[f"{v:.1f}%" for v in asym_history_df['Asymmetry']],
                                    textposition='outside',
                                    hovertemplate='<b>%{y}</b><br>Asymmetry: %{x:.1f}%<extra></extra>'
                                ))

                                # Add vertical threshold lines
                                fig_asym_hist.add_vline(x=0, line_width=2, line_dash="solid", line_color="#666")
                                fig_asym_hist.add_vline(x=-10, line_width=1, line_dash="dash", line_color="#F39C12")
                                fig_asym_hist.add_vline(x=10, line_width=1, line_dash="dash", line_color="#F39C12")

                                fig_asym_hist.update_layout(
                                    title=f"{selected_ff_athlete} - Asymmetry Trend ({selected_test_type})",
                                    xaxis_title="Asymmetry (%) - Positive = Right Dominant",
                                    yaxis_title="Test Date",
                                    height=max(350, len(asym_history_df) * 40),
                                    showlegend=False,
                                    xaxis=dict(range=[-30, 30], zeroline=True),
                                    yaxis=dict(autorange="reversed"),
                                    plot_bgcolor='rgba(0,0,0,0)'
                                )

                                st.plotly_chart(fig_asym_hist, use_container_width=True)

                                # Legend
                                st.markdown("""
                                <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 0.5rem;">
                                    <span style="color: #27AE60;">● Normal (&lt;10%)</span>
                                    <span style="color: #F39C12;">● Caution (10-15%)</span>
                                    <span style="color: #E74C3C;">● At Risk (&gt;15%)</span>
                                </div>
                                """, unsafe_allow_html=True)

                        # Force time series - by test type
                        if 'testDateUtc' in athlete_ff.columns:
                            st.markdown(f"#### 📈 {selected_test_type} - Force Over Time")
                            athlete_ff_sorted = athlete_ff.sort_values('testDateUtc')

                            fig_time = go.Figure()

                            if 'innerLeftMaxForce' in athlete_ff_sorted.columns:
                                fig_time.add_trace(go.Scatter(
                                    x=athlete_ff_sorted['testDateUtc'],
                                    y=athlete_ff_sorted['innerLeftMaxForce'],
                                    mode='lines+markers',
                                    name='Left (Inner)',
                                    line=dict(color='#3498db', width=2)
                                ))

                            if 'innerRightMaxForce' in athlete_ff_sorted.columns:
                                fig_time.add_trace(go.Scatter(
                                    x=athlete_ff_sorted['testDateUtc'],
                                    y=athlete_ff_sorted['innerRightMaxForce'],
                                    mode='lines+markers',
                                    name='Right (Inner)',
                                    line=dict(color='#e74c3c', width=2)
                                ))

                            # Add outer forces if available
                            if 'outerLeftMaxForce' in athlete_ff_sorted.columns:
                                fig_time.add_trace(go.Scatter(
                                    x=athlete_ff_sorted['testDateUtc'],
                                    y=athlete_ff_sorted['outerLeftMaxForce'],
                                    mode='lines+markers',
                                    name='Left (Outer)',
                                    line=dict(color='#2980b9', width=2, dash='dash')
                                ))

                            if 'outerRightMaxForce' in athlete_ff_sorted.columns:
                                fig_time.add_trace(go.Scatter(
                                    x=athlete_ff_sorted['testDateUtc'],
                                    y=athlete_ff_sorted['outerRightMaxForce'],
                                    mode='lines+markers',
                                    name='Right (Outer)',
                                    line=dict(color='#c0392b', width=2, dash='dash')
                                ))

                            fig_time.update_layout(
                                xaxis_title="Date",
                                yaxis_title="Max Force (N)",
                                height=450,
                                hovermode='x unified',
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig_time, use_container_width=True)

                        # Athlete test history for this test type
                        st.markdown(f"#### 📋 {selected_test_type} - Test History")
                        hist_cols = ['testDateUtc', 'testTypeName', 'innerLeftMaxForce', 'innerRightMaxForce',
                                    'outerLeftMaxForce', 'outerRightMaxForce']
                        hist_available = [c for c in hist_cols if c in athlete_ff.columns]
                        if hist_available:
                            display_hist = athlete_ff[hist_available].copy()
                            if 'testDateUtc' in display_hist.columns:
                                display_hist['testDateUtc'] = pd.to_datetime(display_hist['testDateUtc']).dt.strftime('%Y-%m-%d')
                            st.dataframe(display_hist.round(1), use_container_width=True, hide_index=True)
                    else:
                        st.info("No data available for the selected athlete and test type.")
                else:
                    st.info("No athlete names available in ForceFrame data. Athlete mapping may not be configured.")
            else:
                st.info("No athlete names available in ForceFrame data.")

        # ========== YEARLY PROGRESSION TAB ==========
        with ff_tabs[4]:
            st.markdown("### 📈 Yearly Progression Analysis")
            st.markdown("*Track force development across the year*")

            if 'testDateUtc' in ff_data.columns and 'Name' in ff_data.columns:
                ff_data_copy = ff_data.copy()
                ff_data_copy['testDateUtc'] = pd.to_datetime(ff_data_copy['testDateUtc'])
                ff_data_copy['Month'] = ff_data_copy['testDateUtc'].dt.to_period('M').astype(str)
                ff_data_copy['Year'] = ff_data_copy['testDateUtc'].dt.year

                # Select athlete for progression
                prog_athlete = st.selectbox(
                    "Select Athlete for Progression:",
                    options=sorted(ff_data_copy['Name'].dropna().unique()),
                    key="ff_prog_athlete"
                )

                athlete_prog = ff_data_copy[ff_data_copy['Name'] == prog_athlete].copy()

                if not athlete_prog.empty and 'innerLeftMaxForce' in athlete_prog.columns:
                    # Monthly progression
                    monthly_avg = athlete_prog.groupby('Month').agg({
                        'innerLeftMaxForce': 'mean',
                        'innerRightMaxForce': 'mean'
                    }).reset_index()

                    fig_prog = go.Figure()

                    fig_prog.add_trace(go.Bar(
                        x=monthly_avg['Month'],
                        y=monthly_avg['innerLeftMaxForce'],
                        name='Left Force',
                        marker_color='#3498db'
                    ))

                    fig_prog.add_trace(go.Bar(
                        x=monthly_avg['Month'],
                        y=monthly_avg['innerRightMaxForce'],
                        name='Right Force',
                        marker_color='#e74c3c'
                    ))

                    fig_prog.update_layout(
                        title=f"{prog_athlete} - Monthly Force Progression",
                        xaxis_title="Month",
                        yaxis_title="Avg Max Force (N)",
                        barmode='group',
                        height=450
                    )
                    st.plotly_chart(fig_prog, use_container_width=True)

                    # Progress metrics
                    if len(athlete_prog) >= 2:
                        first_test = athlete_prog.iloc[0]
                        last_test = athlete_prog.iloc[-1]

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            change_left = ((last_test['innerLeftMaxForce'] - first_test['innerLeftMaxForce']) /
                                          first_test['innerLeftMaxForce'] * 100) if first_test['innerLeftMaxForce'] > 0 else 0
                            st.metric("Left Force Change", f"{change_left:+.1f}%",
                                     delta="Improved" if change_left > 0 else "Declined")
                        with col2:
                            change_right = ((last_test['innerRightMaxForce'] - first_test['innerRightMaxForce']) /
                                           first_test['innerRightMaxForce'] * 100) if first_test['innerRightMaxForce'] > 0 else 0
                            st.metric("Right Force Change", f"{change_right:+.1f}%",
                                     delta="Improved" if change_right > 0 else "Declined")
                        with col3:
                            st.metric("Testing Period",
                                     f"{(athlete_prog['testDateUtc'].max() - athlete_prog['testDateUtc'].min()).days} days")
            else:
                st.info("Date or athlete information not available for progression analysis.")

        # ========== ASYMMETRY DASHBOARD TAB ==========
        with ff_tabs[3]:
            st.markdown("### ⚖️ Bilateral Asymmetry Dashboard")
            st.markdown("""
            **Asymmetry Risk Thresholds:**
            - ✅ **<10%**: Normal bilateral balance
            - ⚠️ **10-15%**: Monitor - targeted training recommended
            - 🔴 **>15%**: High risk - intervention required
            """)

            if 'innerLeftMaxForce' in ff_data.columns and 'innerRightMaxForce' in ff_data.columns:
                # Calculate asymmetry for all tests
                cols_needed = ['Name', 'testDateUtc', 'innerLeftMaxForce', 'innerRightMaxForce']
                if 'testTypeName' in ff_data.columns:
                    cols_needed.insert(1, 'testTypeName')
                asym_df = ff_data[[c for c in cols_needed if c in ff_data.columns]].dropna().copy()

                asym_df['Asymmetry (%)'] = ((asym_df['innerRightMaxForce'] - asym_df['innerLeftMaxForce']) /
                                            ((asym_df['innerRightMaxForce'] + asym_df['innerLeftMaxForce']) / 2) * 100)

                asym_df['Risk Level'] = asym_df['Asymmetry (%)'].abs().apply(
                    lambda x: '🔴 High Risk' if x > 15 else ('⚠️ Moderate' if x > 10 else '✅ Normal')
                )

                # Team overview cards
                st.markdown("#### 🚦 Team Asymmetry Status")
                col1, col2, col3 = st.columns(3)

                with col1:
                    normal_pct = (asym_df['Asymmetry (%)'].abs() < 10).sum() / len(asym_df) * 100 if len(asym_df) > 0 else 0
                    st.markdown(f"""
                    <div style="background: #27ae60; padding: 1rem; border-radius: 8px; text-align: center; color: white;">
                        <h2 style="margin: 0;">{normal_pct:.0f}%</h2>
                        <p style="margin: 0;">Normal (<10%)</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    moderate_pct = ((asym_df['Asymmetry (%)'].abs() >= 10) & (asym_df['Asymmetry (%)'].abs() < 15)).sum() / len(asym_df) * 100 if len(asym_df) > 0 else 0
                    st.markdown(f"""
                    <div style="background: #f39c12; padding: 1rem; border-radius: 8px; text-align: center; color: white;">
                        <h2 style="margin: 0;">{moderate_pct:.0f}%</h2>
                        <p style="margin: 0;">Monitor (10-15%)</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    high_pct = (asym_df['Asymmetry (%)'].abs() >= 15).sum() / len(asym_df) * 100 if len(asym_df) > 0 else 0
                    st.markdown(f"""
                    <div style="background: #e74c3c; padding: 1rem; border-radius: 8px; text-align: center; color: white;">
                        <h2 style="margin: 0;">{high_pct:.0f}%</h2>
                        <p style="margin: 0;">High Risk (>15%)</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Horizontal Asymmetry Bar Chart (Patrick Ward style)
                st.markdown("#### 📊 Team Asymmetry Profile by Athlete")
                st.markdown("*Bars show left-right difference. Green = safe, Yellow = caution, Red = high risk*")

                if 'Name' in asym_df.columns and asym_df['Name'].nunique() > 0:
                    # Get latest test per athlete
                    asym_df['testDateUtc'] = pd.to_datetime(asym_df['testDateUtc'])
                    latest_asym = asym_df.sort_values('testDateUtc').groupby('Name').last().reset_index()
                    latest_asym = latest_asym.sort_values('Asymmetry (%)', key=abs, ascending=True)

                    # Color based on risk
                    def get_ff_asym_color(val):
                        abs_val = abs(val)
                        if abs_val < 10:
                            return '#27ae60'
                        elif abs_val < 15:
                            return '#f39c12'
                        else:
                            return '#e74c3c'

                    colors = [get_ff_asym_color(v) for v in latest_asym['Asymmetry (%)']]

                    fig_ff_asym = go.Figure()

                    # Safe zone shading
                    fig_ff_asym.add_vrect(x0=-10, x1=10, fillcolor="rgba(39, 174, 96, 0.1)",
                                         layer="below", line_width=0)

                    fig_ff_asym.add_trace(go.Bar(
                        y=latest_asym['Name'],
                        x=latest_asym['Asymmetry (%)'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"{v:.1f}%" for v in latest_asym['Asymmetry (%)']],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Asymmetry: %{x:.1f}%<extra></extra>'
                    ))

                    # Threshold lines
                    fig_ff_asym.add_vline(x=0, line_dash='solid', line_color='gray', line_width=1)
                    fig_ff_asym.add_vline(x=10, line_dash='dot', line_color='#f39c12', line_width=1)
                    fig_ff_asym.add_vline(x=-10, line_dash='dot', line_color='#f39c12', line_width=1)
                    fig_ff_asym.add_vline(x=15, line_dash='dash', line_color='#e74c3c', line_width=1)
                    fig_ff_asym.add_vline(x=-15, line_dash='dash', line_color='#e74c3c', line_width=1)

                    fig_ff_asym.update_layout(
                        height=max(300, len(latest_asym) * 35),
                        xaxis_title="Asymmetry (%) - Left ← → Right Dominant",
                        yaxis_title="",
                        showlegend=False,
                        xaxis=dict(range=[-30, 30], zeroline=True),
                        margin=dict(l=150)
                    )
                    st.plotly_chart(fig_ff_asym, use_container_width=True)

                st.markdown("---")

                # Asymmetry distribution histogram
                st.markdown("#### Asymmetry Distribution (All Tests)")
                fig_asym = px.histogram(
                    asym_df, x='Asymmetry (%)', nbins=25,
                    color_discrete_sequence=['#1D4D3B']
                )
                fig_asym.add_vline(x=0, line_dash='dash', line_color='gray', annotation_text='Balanced')
                fig_asym.add_vline(x=10, line_dash='dot', line_color='orange', annotation_text='+10%')
                fig_asym.add_vline(x=-10, line_dash='dot', line_color='orange', annotation_text='-10%')
                fig_asym.add_vline(x=15, line_dash='dot', line_color='red')
                fig_asym.add_vline(x=-15, line_dash='dot', line_color='red')
                fig_asym.update_layout(height=350)
                st.plotly_chart(fig_asym, use_container_width=True)

                # Athletes at risk table
                st.markdown("#### ⚠️ Athletes Requiring Attention")
                high_risk = asym_df[asym_df['Asymmetry (%)'].abs() > 10].copy()

                if not high_risk.empty:
                    high_risk['testDateUtc'] = pd.to_datetime(high_risk['testDateUtc'])
                    latest_risk = high_risk.sort_values('testDateUtc').groupby('Name').last().reset_index()
                    latest_risk = latest_risk.sort_values('Asymmetry (%)', key=abs, ascending=False)

                    display_cols = ['Name', 'Asymmetry (%)', 'Risk Level', 'innerLeftMaxForce', 'innerRightMaxForce']
                    if 'testTypeName' in latest_risk.columns:
                        display_cols.insert(1, 'testTypeName')
                    display_risk = latest_risk[[c for c in display_cols if c in latest_risk.columns]]
                    rename_map = {'Name': 'Athlete', 'testTypeName': 'Test Type', 'Asymmetry (%)': 'Asymmetry (%)',
                                  'Risk Level': 'Risk', 'innerLeftMaxForce': 'Left (N)', 'innerRightMaxForce': 'Right (N)'}
                    display_risk.columns = [rename_map.get(c, c) for c in display_risk.columns]
                    st.dataframe(display_risk.round(1), use_container_width=True, hide_index=True)
                else:
                    st.success("✅ All athletes within normal asymmetry thresholds!")
            else:
                st.warning("Left/Right force columns not available for asymmetry analysis.")

# ============================================================================
# PAGE: NORDBORD
# ============================================================================

with tabs[3]:
    st.markdown("## 🦵 NordBord Hamstring Analysis")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <p style="color: white; margin: 0; font-size: 0.95rem;">
            <strong>Gold standard</strong> for hamstring testing • Research-based injury thresholds
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Research-based thresholds (Timmins et al. 2016, Opar et al. 2015)
    NORDBORD_THRESHOLDS = {
        'injury_risk_absolute': 337,  # Below 337N = 4.4x injury risk (Timmins et al. 2016)
        'target_absolute': 400,       # Target for elite athletes
        'relative_good': 4.5,         # Good relative strength N/kg
        'relative_elite': 5.2,        # Elite male athletes N/kg
        'asymmetry_low': 10,          # <10% = low risk
        'asymmetry_moderate': 15,     # 10-15% = 2.4x risk
        'asymmetry_high': 20,         # >15% = 3.4x risk
    }

    # Use filtered data
    nb_data = filtered_nordbord if not filtered_nordbord.empty else df_nordbord

    if nb_data.empty:
        st.warning("No NordBord data available. Upload data or check data/nordbord_allsports.csv")
    else:
        # Create subtabs for NordBord - enhanced structure
        nb_tabs = st.tabs(["📊 Overview", "🎯 Strength Benchmarks", "🏃 Individual Athlete", "⚖️ Asymmetry Dashboard", "📈 Progression"])

        with nb_tabs[0]:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                n_athletes = nb_data['Name'].nunique() if 'Name' in nb_data.columns else (nb_data['athleteId'].nunique() if 'athleteId' in nb_data.columns else 0)
                st.metric("Athletes", n_athletes)

            with col2:
                st.metric("Total Tests", len(nb_data))

            with col3:
                if 'testTypeName' in nb_data.columns:
                    test_types = nb_data['testTypeName'].unique()
                    st.metric("Test Type", test_types[0] if len(test_types) > 0 else "Nordic")

            with col4:
                if 'testDateUtc' in nb_data.columns:
                    latest = pd.to_datetime(nb_data['testDateUtc']).max()
                    st.metric("Latest Test", latest.strftime('%Y-%m-%d') if pd.notna(latest) else "N/A")

            st.markdown("---")

            # Left vs Right scatter
            st.markdown("### Left vs Right Hamstring Force")

            if 'leftMaxForce' in nb_data.columns and 'rightMaxForce' in nb_data.columns:
                plot_df = nb_data[['leftMaxForce', 'rightMaxForce']].dropna()

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

            # Data table
            st.markdown("### NordBord Data Table")
            display_cols = ['Name', 'testDateUtc', 'testTypeName', 'leftMaxForce', 'rightMaxForce',
                           'leftAvgForce', 'rightAvgForce']
            available_cols = [c for c in display_cols if c in nb_data.columns]

            if available_cols:
                display_nb = nb_data[available_cols].copy()
                if 'testDateUtc' in display_nb.columns:
                    display_nb['testDateUtc'] = pd.to_datetime(display_nb['testDateUtc']).dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(display_nb, use_container_width=True, height=400)

            csv_nb = nb_data.to_csv(index=False)
            st.download_button("📥 Download NordBord Data", csv_nb,
                              f"nordbord_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

        # ========== STRENGTH BENCHMARKS TAB ==========
        with nb_tabs[1]:
            st.markdown("### 🎯 Hamstring Strength Benchmarks")
            st.markdown("""
            **Research-Based Injury Risk Thresholds:**

            | Metric | Threshold | Risk Level | Source |
            |--------|-----------|------------|--------|
            | Absolute Force | <337 N | **4.4x injury risk** | Timmins et al. 2016 |
            | Relative Force | <4.5 N/kg | Below average | VALD Population Data |
            | Asymmetry | >15% | **2.4x injury risk** | Opar et al. 2015 |
            | Asymmetry | >20% | **3.4x injury risk** | Opar et al. 2015 |

            **Elite Benchmarks:** Males 441±89N (5.2 N/kg) | Females 315±60N (4.8 N/kg)
            """)

            if 'leftMaxForce' in nb_data.columns and 'rightMaxForce' in nb_data.columns:
                # Calculate metrics for benchmarking
                nb_bench = nb_data.copy()
                nb_bench['Max Force'] = nb_bench[['leftMaxForce', 'rightMaxForce']].max(axis=1)
                nb_bench['Min Force'] = nb_bench[['leftMaxForce', 'rightMaxForce']].min(axis=1)
                nb_bench['Avg Force'] = nb_bench[['leftMaxForce', 'rightMaxForce']].mean(axis=1)
                nb_bench['Asymmetry (%)'] = (
                    (nb_bench['leftMaxForce'] - nb_bench['rightMaxForce']).abs() /
                    ((nb_bench['leftMaxForce'] + nb_bench['rightMaxForce']) / 2) * 100
                )

                # Calculate relative strength if body mass available
                if 'athlete_weight_kg' in nb_bench.columns:
                    nb_bench['Relative Strength (N/kg)'] = nb_bench['Avg Force'] / nb_bench['athlete_weight_kg']
                else:
                    nb_bench['Relative Strength (N/kg)'] = None

                # Injury risk classification
                INJURY_THRESHOLD = NORDBORD_THRESHOLDS['injury_risk_absolute']  # 337N

                # Status cards with traffic light system
                st.markdown("#### 🚦 Team Risk Dashboard")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    above_337 = (nb_bench['Min Force'] >= INJURY_THRESHOLD).sum()
                    pct_above = above_337 / len(nb_bench) * 100 if len(nb_bench) > 0 else 0
                    risk_color = '#27ae60' if pct_above >= 80 else '#f39c12' if pct_above >= 50 else '#e74c3c'
                    st.markdown(f"""
                    <div style="background: {risk_color}; padding: 1rem; border-radius: 8px; text-align: center; color: white;">
                        <h2 style="margin: 0;">{pct_above:.0f}%</h2>
                        <p style="margin: 0; font-size: 0.85rem;">Above 337N (Safe)</p>
                        <p style="margin: 0; font-size: 0.7rem; opacity: 0.8;">{above_337}/{len(nb_bench)} athletes</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    below_15 = (nb_bench['Asymmetry (%)'] < 15).sum()
                    pct_sym = below_15 / len(nb_bench) * 100 if len(nb_bench) > 0 else 0
                    sym_color = '#27ae60' if pct_sym >= 80 else '#f39c12' if pct_sym >= 50 else '#e74c3c'
                    st.markdown(f"""
                    <div style="background: {sym_color}; padding: 1rem; border-radius: 8px; text-align: center; color: white;">
                        <h2 style="margin: 0;">{pct_sym:.0f}%</h2>
                        <p style="margin: 0; font-size: 0.85rem;">Within 15% Asymmetry</p>
                        <p style="margin: 0; font-size: 0.7rem; opacity: 0.8;">{below_15}/{len(nb_bench)} athletes</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    avg_force = nb_bench[['leftMaxForce', 'rightMaxForce']].mean().mean()
                    force_color = '#27ae60' if avg_force >= 400 else '#f39c12' if avg_force >= INJURY_THRESHOLD else '#e74c3c'
                    st.markdown(f"""
                    <div style="background: {force_color}; padding: 1rem; border-radius: 8px; text-align: center; color: white;">
                        <h2 style="margin: 0;">{avg_force:.0f}N</h2>
                        <p style="margin: 0; font-size: 0.85rem;">Team Avg Force</p>
                        <p style="margin: 0; font-size: 0.7rem; opacity: 0.8;">Target: >400N</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    avg_asym = nb_bench['Asymmetry (%)'].mean()
                    asym_color = '#27ae60' if avg_asym < 10 else '#f39c12' if avg_asym < 15 else '#e74c3c'
                    st.markdown(f"""
                    <div style="background: {asym_color}; padding: 1rem; border-radius: 8px; text-align: center; color: white;">
                        <h2 style="margin: 0;">{avg_asym:.1f}%</h2>
                        <p style="margin: 0; font-size: 0.85rem;">Avg Asymmetry</p>
                        <p style="margin: 0; font-size: 0.7rem; opacity: 0.8;">Target: <10%</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Horizontal Asymmetry Bar Chart (Patrick Ward style)
                st.markdown("#### 📊 Team Asymmetry Profile")
                st.markdown("*Bars show left-right difference. Green = safe (<10%), Yellow = caution (10-15%), Red = high risk (>15%)*")

                if 'Name' in nb_bench.columns:
                    # Get latest test per athlete
                    if 'testDateUtc' in nb_bench.columns:
                        nb_bench['testDateUtc'] = pd.to_datetime(nb_bench['testDateUtc'])
                        latest_tests = nb_bench.sort_values('testDateUtc').groupby('Name').last().reset_index()
                    else:
                        latest_tests = nb_bench.groupby('Name').last().reset_index()

                    # Calculate signed asymmetry (positive = right dominant, negative = left dominant)
                    latest_tests['Signed Asymmetry'] = (
                        (latest_tests['rightMaxForce'] - latest_tests['leftMaxForce']) /
                        ((latest_tests['rightMaxForce'] + latest_tests['leftMaxForce']) / 2) * 100
                    )

                    # Sort by absolute asymmetry
                    latest_tests = latest_tests.sort_values('Signed Asymmetry', key=abs, ascending=True)

                    # Color based on risk level
                    def get_asym_color(val):
                        abs_val = abs(val)
                        if abs_val < 10:
                            return '#27ae60'  # Green - safe
                        elif abs_val < 15:
                            return '#f39c12'  # Yellow - caution
                        else:
                            return '#e74c3c'  # Red - high risk

                    colors = [get_asym_color(v) for v in latest_tests['Signed Asymmetry']]

                    fig_asym_bars = go.Figure()

                    # Add gray zone for safe range
                    fig_asym_bars.add_vrect(x0=-10, x1=10, fillcolor="rgba(39, 174, 96, 0.1)",
                                           layer="below", line_width=0)

                    fig_asym_bars.add_trace(go.Bar(
                        y=latest_tests['Name'],
                        x=latest_tests['Signed Asymmetry'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"{v:.1f}%" for v in latest_tests['Signed Asymmetry']],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Asymmetry: %{x:.1f}%<extra></extra>'
                    ))

                    # Add threshold lines
                    fig_asym_bars.add_vline(x=0, line_dash='solid', line_color='gray', line_width=1)
                    fig_asym_bars.add_vline(x=10, line_dash='dot', line_color='#f39c12', line_width=1)
                    fig_asym_bars.add_vline(x=-10, line_dash='dot', line_color='#f39c12', line_width=1)
                    fig_asym_bars.add_vline(x=15, line_dash='dash', line_color='#e74c3c', line_width=1)
                    fig_asym_bars.add_vline(x=-15, line_dash='dash', line_color='#e74c3c', line_width=1)

                    fig_asym_bars.update_layout(
                        height=max(300, len(latest_tests) * 35),
                        xaxis_title="Asymmetry (%) - Left ← → Right Dominant",
                        yaxis_title="",
                        showlegend=False,
                        xaxis=dict(range=[-30, 30], zeroline=True),
                        margin=dict(l=150)
                    )
                    st.plotly_chart(fig_asym_bars, use_container_width=True)

                st.markdown("---")

                # Force distribution with 337N threshold
                st.markdown("#### Force Distribution vs 337N Injury Threshold")
                col1, col2 = st.columns(2)

                with col1:
                    fig_left = px.histogram(
                        nb_bench, x='leftMaxForce', nbins=15,
                        labels={'leftMaxForce': 'Left Hamstring Force (N)'},
                        color_discrete_sequence=['#3498db']
                    )
                    fig_left.add_vline(x=INJURY_THRESHOLD, line_dash='dash', line_color='red',
                                       annotation_text=f'{INJURY_THRESHOLD}N Risk Threshold')
                    fig_left.add_vline(x=400, line_dash='dot', line_color='green',
                                       annotation_text='400N Target')
                    fig_left.update_layout(title='Left Hamstring', height=350)
                    st.plotly_chart(fig_left, use_container_width=True)

                with col2:
                    fig_right = px.histogram(
                        nb_bench, x='rightMaxForce', nbins=15,
                        labels={'rightMaxForce': 'Right Hamstring Force (N)'},
                        color_discrete_sequence=['#e74c3c']
                    )
                    fig_right.add_vline(x=INJURY_THRESHOLD, line_dash='dash', line_color='red',
                                        annotation_text=f'{INJURY_THRESHOLD}N Risk Threshold')
                    fig_right.add_vline(x=400, line_dash='dot', line_color='green',
                                        annotation_text='400N Target')
                    fig_right.update_layout(title='Right Hamstring', height=350)
                    st.plotly_chart(fig_right, use_container_width=True)

                # Athletes below threshold
                st.markdown("#### ⚠️ Athletes Below 337N Injury Threshold")
                if 'Name' in nb_bench.columns:
                    below_threshold = nb_bench[nb_bench['Min Force'] < INJURY_THRESHOLD].copy()
                    if not below_threshold.empty:
                        below_summary = below_threshold.groupby('Name').agg({
                            'leftMaxForce': 'max',
                            'rightMaxForce': 'max',
                            'Asymmetry (%)': 'mean'
                        }).reset_index()
                        below_summary.columns = ['Athlete', 'Left Peak (N)', 'Right Peak (N)', 'Avg Asymmetry (%)']
                        below_summary['Deficit to 337N'] = INJURY_THRESHOLD - below_summary[['Left Peak (N)', 'Right Peak (N)']].min(axis=1)
                        below_summary['Risk'] = below_summary['Deficit to 337N'].apply(
                            lambda x: '🔴 High' if x > 50 else ('⚠️ Moderate' if x > 20 else '🟡 Low')
                        )
                        below_summary = below_summary.sort_values('Deficit to 337N', ascending=False)
                        st.dataframe(below_summary.round(1), use_container_width=True, hide_index=True)
                    else:
                        st.success("✅ All athletes meet the 337N injury threshold!")
            else:
                st.warning("Force data columns not available for benchmarking.")

        # ========== INDIVIDUAL ATHLETE TAB ==========
        with nb_tabs[2]:
            st.markdown("### 🏃 Individual Athlete Analysis")
            st.markdown("*Select an athlete to view their Nordic hamstring profile with risk assessment*")

            if 'Name' in nb_data.columns and nb_data['Name'].nunique() > 0:
                # Get unique athlete names (filter out placeholder names)
                athlete_names = nb_data['Name'].dropna().unique()
                athlete_names = [n for n in athlete_names if not str(n).startswith('Athlete_')]

                if athlete_names:
                    selected_nb_athlete = st.selectbox(
                        "Select Athlete:",
                        options=sorted(athlete_names),
                        key="nb_athlete_select"
                    )

                    athlete_nb = nb_data[nb_data['Name'] == selected_nb_athlete].copy()

                    if not athlete_nb.empty:
                        st.markdown(f"#### {selected_nb_athlete} - Hamstring Profile")

                        # Calculate key metrics
                        peak_left = athlete_nb['leftMaxForce'].max() if 'leftMaxForce' in athlete_nb.columns else 0
                        peak_right = athlete_nb['rightMaxForce'].max() if 'rightMaxForce' in athlete_nb.columns else 0
                        min_force = min(peak_left, peak_right)
                        avg_force = (peak_left + peak_right) / 2 if (peak_left + peak_right) > 0 else 0

                        # Calculate asymmetry
                        if peak_left > 0 and peak_right > 0:
                            avg_asym = ((peak_right - peak_left) / ((peak_right + peak_left) / 2) * 100)
                        else:
                            avg_asym = 0

                        # Traffic light status cards (matching team style)
                        st.markdown("#### 🚦 Athlete Risk Status")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            # Force threshold status
                            force_ok = min_force >= NORDBORD_THRESHOLDS['injury_risk_absolute']
                            force_color = '#27ae60' if force_ok else '#e74c3c'
                            force_status = '✅ Safe' if force_ok else '🔴 At Risk'
                            st.markdown(f"""
                            <div style="background: {force_color}; padding: 1rem; border-radius: 8px; text-align: center; color: white;">
                                <h2 style="margin: 0;">{min_force:.0f}N</h2>
                                <p style="margin: 0; font-size: 0.85rem;">Min Force</p>
                                <p style="margin: 0; font-size: 0.7rem; opacity: 0.8;">{force_status} (>337N)</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"""
                            <div style="background: #3498db; padding: 1rem; border-radius: 8px; text-align: center; color: white;">
                                <h2 style="margin: 0;">{peak_left:.0f}N</h2>
                                <p style="margin: 0; font-size: 0.85rem;">Left Peak</p>
                                <p style="margin: 0; font-size: 0.7rem; opacity: 0.8;">Max recorded</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            st.markdown(f"""
                            <div style="background: #e74c3c; padding: 1rem; border-radius: 8px; text-align: center; color: white;">
                                <h2 style="margin: 0;">{peak_right:.0f}N</h2>
                                <p style="margin: 0; font-size: 0.85rem;">Right Peak</p>
                                <p style="margin: 0; font-size: 0.7rem; opacity: 0.8;">Max recorded</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col4:
                            # Asymmetry status
                            abs_asym = abs(avg_asym)
                            if abs_asym < 10:
                                asym_color = '#27ae60'
                                asym_status = '✅ Normal'
                            elif abs_asym < 15:
                                asym_color = '#f39c12'
                                asym_status = '⚠️ Monitor'
                            else:
                                asym_color = '#e74c3c'
                                asym_status = '🔴 High Risk'

                            direction = "R>L" if avg_asym > 0 else "L>R"
                            st.markdown(f"""
                            <div style="background: {asym_color}; padding: 1rem; border-radius: 8px; text-align: center; color: white;">
                                <h2 style="margin: 0;">{avg_asym:+.1f}%</h2>
                                <p style="margin: 0; font-size: 0.85rem;">Asymmetry ({direction})</p>
                                <p style="margin: 0; font-size: 0.7rem; opacity: 0.8;">{asym_status}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("---")

                        # Test history horizontal bar chart (like team profile)
                        st.markdown("#### 📊 Test History - Asymmetry Profile")
                        if 'testDateUtc' in athlete_nb.columns and len(athlete_nb) > 1:
                            athlete_nb['testDateUtc'] = pd.to_datetime(athlete_nb['testDateUtc'])
                            athlete_nb = athlete_nb.sort_values('testDateUtc')
                            athlete_nb['Test Date'] = athlete_nb['testDateUtc'].dt.strftime('%Y-%m-%d')
                            athlete_nb['Test Asymmetry'] = ((athlete_nb['rightMaxForce'] - athlete_nb['leftMaxForce']) /
                                                           ((athlete_nb['rightMaxForce'] + athlete_nb['leftMaxForce']) / 2) * 100)

                            # Color by risk
                            def get_test_color(val):
                                abs_val = abs(val)
                                if abs_val < 10:
                                    return '#27ae60'
                                elif abs_val < 15:
                                    return '#f39c12'
                                else:
                                    return '#e74c3c'

                            colors = [get_test_color(v) for v in athlete_nb['Test Asymmetry']]

                            fig_athlete_asym = go.Figure()
                            fig_athlete_asym.add_vrect(x0=-10, x1=10, fillcolor="rgba(39, 174, 96, 0.1)",
                                                      layer="below", line_width=0)

                            fig_athlete_asym.add_trace(go.Bar(
                                y=athlete_nb['Test Date'],
                                x=athlete_nb['Test Asymmetry'],
                                orientation='h',
                                marker_color=colors,
                                text=[f"{v:.1f}%" for v in athlete_nb['Test Asymmetry']],
                                textposition='outside'
                            ))

                            fig_athlete_asym.add_vline(x=0, line_dash='solid', line_color='gray', line_width=1)
                            fig_athlete_asym.add_vline(x=10, line_dash='dot', line_color='#f39c12', line_width=1)
                            fig_athlete_asym.add_vline(x=-10, line_dash='dot', line_color='#f39c12', line_width=1)
                            fig_athlete_asym.add_vline(x=15, line_dash='dash', line_color='#e74c3c', line_width=1)
                            fig_athlete_asym.add_vline(x=-15, line_dash='dash', line_color='#e74c3c', line_width=1)

                            fig_athlete_asym.update_layout(
                                height=max(250, len(athlete_nb) * 30),
                                xaxis_title="Asymmetry (%) - Left ← → Right Dominant",
                                yaxis_title="",
                                showlegend=False,
                                xaxis=dict(range=[-30, 30], zeroline=True),
                                margin=dict(l=100)
                            )
                            st.plotly_chart(fig_athlete_asym, use_container_width=True)

                        st.markdown("---")

                        # Force time series
                        if 'testDateUtc' in athlete_nb.columns:
                            st.markdown("#### Hamstring Force Over Time")
                            athlete_nb['testDateUtc'] = pd.to_datetime(athlete_nb['testDateUtc'])
                            athlete_nb = athlete_nb.sort_values('testDateUtc')

                            fig_time = go.Figure()

                            if 'leftMaxForce' in athlete_nb.columns:
                                fig_time.add_trace(go.Scatter(
                                    x=athlete_nb['testDateUtc'],
                                    y=athlete_nb['leftMaxForce'],
                                    mode='lines+markers',
                                    name='Left Hamstring',
                                    line=dict(color='#3498db', width=2),
                                    marker=dict(size=8)
                                ))

                            if 'rightMaxForce' in athlete_nb.columns:
                                fig_time.add_trace(go.Scatter(
                                    x=athlete_nb['testDateUtc'],
                                    y=athlete_nb['rightMaxForce'],
                                    mode='lines+markers',
                                    name='Right Hamstring',
                                    line=dict(color='#e74c3c', width=2),
                                    marker=dict(size=8)
                                ))

                            fig_time.update_layout(
                                xaxis_title="Date",
                                yaxis_title="Max Force (N)",
                                height=400,
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig_time, use_container_width=True)

                        # Asymmetry trend
                        if 'leftMaxForce' in athlete_nb.columns and 'rightMaxForce' in athlete_nb.columns:
                            st.markdown("#### Asymmetry Trend")
                            athlete_nb['Asymmetry'] = ((athlete_nb['rightMaxForce'] - athlete_nb['leftMaxForce']) /
                                                       ((athlete_nb['rightMaxForce'] + athlete_nb['leftMaxForce']) / 2) * 100)

                            fig_asym_trend = go.Figure()
                            fig_asym_trend.add_trace(go.Scatter(
                                x=athlete_nb['testDateUtc'],
                                y=athlete_nb['Asymmetry'],
                                mode='lines+markers',
                                name='Asymmetry',
                                line=dict(color='#1D4D3B', width=2),
                                fill='tozeroy',
                                fillcolor='rgba(29, 77, 59, 0.2)'
                            ))
                            fig_asym_trend.add_hline(y=0, line_dash='dash', line_color='gray')
                            fig_asym_trend.add_hline(y=10, line_dash='dot', line_color='orange', annotation_text='10% threshold')
                            fig_asym_trend.add_hline(y=-10, line_dash='dot', line_color='orange')
                            fig_asym_trend.update_layout(
                                yaxis_title="Asymmetry (%)",
                                height=350
                            )
                            st.plotly_chart(fig_asym_trend, use_container_width=True)
                else:
                    st.info("No athlete names available in NordBord data. Athlete mapping may not be configured.")
            else:
                st.info("No athlete names available in NordBord data.")

        # ========== YEARLY PROGRESSION TAB ==========
        with nb_tabs[4]:
            st.markdown("### 📈 Yearly Progression Analysis")
            st.markdown("*Track hamstring strength development across the year*")

            if 'testDateUtc' in nb_data.columns and 'Name' in nb_data.columns:
                nb_data_copy = nb_data.copy()
                nb_data_copy['testDateUtc'] = pd.to_datetime(nb_data_copy['testDateUtc'])
                nb_data_copy['Month'] = nb_data_copy['testDateUtc'].dt.to_period('M').astype(str)

                prog_nb_athlete = st.selectbox(
                    "Select Athlete for Progression:",
                    options=sorted(nb_data_copy['Name'].dropna().unique()),
                    key="nb_prog_athlete"
                )

                athlete_nb_prog = nb_data_copy[nb_data_copy['Name'] == prog_nb_athlete].copy()

                if not athlete_nb_prog.empty and 'leftMaxForce' in athlete_nb_prog.columns:
                    # Monthly progression
                    monthly_nb = athlete_nb_prog.groupby('Month').agg({
                        'leftMaxForce': 'max',
                        'rightMaxForce': 'max'
                    }).reset_index()

                    fig_nb_prog = go.Figure()

                    fig_nb_prog.add_trace(go.Bar(
                        x=monthly_nb['Month'],
                        y=monthly_nb['leftMaxForce'],
                        name='Left Hamstring (Peak)',
                        marker_color='#3498db'
                    ))

                    fig_nb_prog.add_trace(go.Bar(
                        x=monthly_nb['Month'],
                        y=monthly_nb['rightMaxForce'],
                        name='Right Hamstring (Peak)',
                        marker_color='#e74c3c'
                    ))

                    fig_nb_prog.update_layout(
                        title=f"{prog_nb_athlete} - Monthly Peak Force",
                        xaxis_title="Month",
                        yaxis_title="Peak Force (N)",
                        barmode='group',
                        height=450
                    )
                    st.plotly_chart(fig_nb_prog, use_container_width=True)

                    # Progress summary
                    if len(athlete_nb_prog) >= 2:
                        athlete_nb_prog = athlete_nb_prog.sort_values('testDateUtc')
                        first = athlete_nb_prog.iloc[0]
                        last = athlete_nb_prog.iloc[-1]

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            change = ((last['leftMaxForce'] - first['leftMaxForce']) / first['leftMaxForce'] * 100) if first['leftMaxForce'] > 0 else 0
                            st.metric("Left Change", f"{change:+.1f}%")
                        with col2:
                            change = ((last['rightMaxForce'] - first['rightMaxForce']) / first['rightMaxForce'] * 100) if first['rightMaxForce'] > 0 else 0
                            st.metric("Right Change", f"{change:+.1f}%")
                        with col3:
                            days = (athlete_nb_prog['testDateUtc'].max() - athlete_nb_prog['testDateUtc'].min()).days
                            st.metric("Period", f"{days} days")
            else:
                st.info("Date or athlete information not available.")

        # ========== ASYMMETRY ANALYSIS TAB ==========
        with nb_tabs[3]:
            st.markdown("### ⚖️ Bilateral Hamstring Asymmetry")
            st.markdown("""
            **Hamstring Injury Risk Thresholds:**
            - ✅ **< 10%**: Low injury risk
            - ⚠️ **10-15%**: Elevated risk - targeted training recommended
            - 🔴 **> 15%**: High injury risk - intervention required

            *Based on Nordic hamstring research (Opar et al., 2015)*
            """)

            if 'leftMaxForce' in nb_data.columns and 'rightMaxForce' in nb_data.columns:
                asym_nb = nb_data[['Name', 'testDateUtc', 'leftMaxForce', 'rightMaxForce']].dropna().copy()

                asym_nb['Asymmetry (%)'] = ((asym_nb['rightMaxForce'] - asym_nb['leftMaxForce']) /
                                            ((asym_nb['rightMaxForce'] + asym_nb['leftMaxForce']) / 2) * 100)

                asym_nb['Risk'] = asym_nb['Asymmetry (%)'].abs().apply(
                    lambda x: '🔴 High' if x > 15 else ('⚠️ Moderate' if x > 10 else '✅ Low')
                )

                # Distribution
                st.markdown("#### Asymmetry Distribution")
                fig_nb_asym = px.histogram(
                    asym_nb, x='Asymmetry (%)', nbins=25,
                    color_discrete_sequence=['#1D4D3B']
                )
                fig_nb_asym.add_vline(x=0, line_dash='dash', line_color='gray')
                fig_nb_asym.add_vline(x=10, line_dash='dot', line_color='orange')
                fig_nb_asym.add_vline(x=-10, line_dash='dot', line_color='orange')
                fig_nb_asym.add_vline(x=15, line_dash='dot', line_color='red')
                fig_nb_asym.add_vline(x=-15, line_dash='dot', line_color='red')
                fig_nb_asym.update_layout(height=400)
                st.plotly_chart(fig_nb_asym, use_container_width=True)

                # Risk athletes
                st.markdown("#### Athletes at Injury Risk")
                at_risk = asym_nb[asym_nb['Asymmetry (%)'].abs() > 10].copy()

                if not at_risk.empty and 'Name' in at_risk.columns:
                    at_risk['testDateUtc'] = pd.to_datetime(at_risk['testDateUtc'])
                    latest_risk = at_risk.sort_values('testDateUtc').groupby('Name').last().reset_index()
                    latest_risk = latest_risk.sort_values('Asymmetry (%)', key=abs, ascending=False)

                    st.dataframe(
                        latest_risk[['Name', 'Asymmetry (%)', 'Risk', 'leftMaxForce', 'rightMaxForce']].round(1),
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.success("✅ All athletes within safe asymmetry thresholds!")

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
