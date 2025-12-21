"""
Data Loading and Processing Utilities
VALD Performance Dashboard - Saudi National Team

Supports both local CSV files and VALD API fetching.
On Streamlit Cloud, data is fetched from API using secrets.
"""

import pandas as pd
import numpy as np
import os
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import streamlit as st


# ============================================================================
# API DATA FETCHING
# ============================================================================

def _get_vald_credentials():
    """Get VALD API credentials from Streamlit secrets or environment."""
    try:
        if hasattr(st, 'secrets') and 'vald' in st.secrets:
            return {
                'client_id': st.secrets['vald'].get('CLIENT_ID', ''),
                'client_secret': st.secrets['vald'].get('CLIENT_SECRET', ''),
                'tenant_id': st.secrets['vald'].get('TENANT_ID', ''),
                'region': st.secrets['vald'].get('VALD_REGION', 'euw'),
                'manual_token': st.secrets['vald'].get('MANUAL_TOKEN', ''),
            }
    except Exception:
        pass
    return None


def _get_oauth_token(client_id: str, client_secret: str) -> Optional[str]:
    """Get OAuth token from VALD security endpoint."""
    try:
        response = requests.post(
            'https://security.valdperformance.com/connect/token',
            data={
                'grant_type': 'client_credentials',
                'client_id': client_id,
                'client_secret': client_secret,
                'scope': 'forcedecks forceframe nordbord athletes'
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get('access_token')
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner="Fetching athlete profiles...")
def _fetch_athlete_profiles(token: str, region: str, tenant_id: str) -> Dict[str, dict]:
    """Fetch athlete profiles from VALD API. Returns {profileId: {name, sport, etc}}"""
    profiles_url = f'https://prd-{region}-api-externalprofile.valdperformance.com/api/v1/profiles'
    headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/json'}
    params = {'TenantId': tenant_id}

    try:
        response = requests.get(profiles_url, headers=headers, params=params, timeout=60)
        if response.status_code == 200:
            profiles = response.json()
            profile_map = {}
            for p in profiles:
                pid = p.get('id') or p.get('profileId')
                if pid:
                    profile_map[pid] = {
                        'Name': p.get('fullName') or p.get('name') or p.get('displayName', 'Unknown'),
                        'athlete_sport': p.get('sport') or p.get('primarySport') or 'Unknown',
                        'Groups': p.get('groups', []),
                    }
            return profile_map
    except Exception:
        pass
    return {}


@st.cache_data(ttl=1800, show_spinner="Fetching data from VALD API...")
def fetch_from_vald_api(device: str = 'forcedecks') -> pd.DataFrame:
    """
    Fetch data directly from VALD API.
    Used when local CSV files are not available (e.g., Streamlit Cloud).
    """
    credentials = _get_vald_credentials()
    if not credentials:
        return pd.DataFrame()

    # Get token
    token = credentials.get('manual_token')
    if not token and credentials.get('client_id') and credentials.get('client_secret'):
        token = _get_oauth_token(credentials['client_id'], credentials['client_secret'])

    if not token:
        return pd.DataFrame()

    region = credentials.get('region', 'euw')
    tenant_id = credentials.get('tenant_id')

    if not tenant_id:
        return pd.DataFrame()

    # API endpoints by device
    endpoints = {
        'forcedecks': f'https://prd-{region}-api-extforcedecks.valdperformance.com/v2019q3/teams/{tenant_id}/tests',
        'forceframe': f'https://prd-{region}-api-externalforceframe.valdperformance.com/v2020q1/teams/{tenant_id}/tests',
        'nordbord': f'https://prd-{region}-api-externalnordbord.valdperformance.com/v2019q3/teams/{tenant_id}/tests',
    }

    url = endpoints.get(device)
    if not url:
        return pd.DataFrame()

    headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/json'}

    # Fetch last 90 days of data
    from_date = (datetime.now(timezone.utc) - timedelta(days=90)).strftime('%Y-%m-%dT00:00:00.000Z')
    params = {'modifiedFromUtc': from_date}

    try:
        all_tests = []
        page = 1

        while True:
            params['page'] = page
            response = requests.get(url, headers=headers, params=params, timeout=60)

            if response.status_code != 200:
                break

            data = response.json()
            tests = data if isinstance(data, list) else data.get('data', [])

            if not tests:
                break

            all_tests.extend(tests)

            # Check for more pages
            if len(tests) < 100:  # Assuming page size of 100
                break
            page += 1

            # Safety limit
            if page > 50:
                break

        if not all_tests:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_tests)

        # Fetch athlete profiles for names
        profile_map = _fetch_athlete_profiles(token, region, tenant_id)

        # Add athlete names and sports from profiles
        if 'profileId' in df.columns and profile_map:
            df['Name'] = df['profileId'].map(lambda pid: profile_map.get(pid, {}).get('Name', 'Unknown'))
            df['athlete_sport'] = df['profileId'].map(lambda pid: profile_map.get(pid, {}).get('athlete_sport', 'Unknown'))

        # Parse dates
        date_columns = ['recordedDateUtc', 'testDateUtc', 'modifiedDateUtc']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        df['data_source'] = device

        return df

    except Exception as e:
        st.warning(f"Error fetching {device} data from API: {e}")
        return pd.DataFrame()


# ============================================================================
# MAIN DATA LOADER
# ============================================================================

@st.cache_data(ttl=3600)
def load_vald_data(device: str = 'forcedecks') -> pd.DataFrame:
    """
    Load VALD data from local files or API.

    Priority:
    1. Local CSV files (for local development)
    2. VALD API (for Streamlit Cloud deployment)
    """
    file_mapping = {
        'forcedecks': [
            'forcedecks_allsports_with_athletes.csv',
            'data/forcedecks_allsports_with_athletes.csv',
            'data/master_files/forcedecks_allsports_with_athletes.csv',
            '../data/master_files/forcedecks_allsports_with_athletes.csv',
        ],
        'forceframe': [
            'forceframe_allsports.csv',
            'data/forceframe_allsports.csv',
            'data/master_files/forceframe_allsports.csv',
            '../data/master_files/forceframe_allsports.csv',
        ],
        'nordbord': [
            'nordbord_allsports.csv',
            'data/nordbord_allsports.csv',
            'data/master_files/nordbord_allsports.csv',
            '../data/master_files/nordbord_allsports.csv',
        ]
    }

    file_paths = file_mapping.get(device, file_mapping['forcedecks'])

    # Try local files first
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)

                # Parse dates
                date_columns = ['recordedDateUtc', 'testDateUtc', 'modifiedDateUtc']
                for col in date_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')

                # Standardize group/sport column
                if 'Groups' in df.columns and 'athlete_sport' not in df.columns:
                    df['athlete_sport'] = df['Groups'].apply(extract_sport_from_group)

                # Add device source
                df['data_source'] = device

                return df

            except Exception as e:
                st.warning(f"Error loading {device} data from {file_path}: {e}")
                continue

    # No local files found - try API
    df = fetch_from_vald_api(device)
    if not df.empty:
        return df

    return pd.DataFrame()


def extract_sport_from_group(group_str: str) -> str:
    """Extract sport name from group string"""
    if pd.isna(group_str):
        return 'Unknown'

    group_lower = str(group_str).lower()

    # Sport mapping
    sport_mappings = {
        'fencing': 'Fencing',
        'epee': 'Epee',
        'sabre': 'Sabre',
        'rowing': 'Rowing',
        'coastal': 'Coastal',
        'athletics': 'Athletics',
        'horizontal jump': 'Athletics - Horizontal Jumps',
        'middle distance': 'Athletics - Middle distance',
        'decathlon': 'Decathlon',
        'swimming': 'Swimming',
        'para swimming': 'Para Swimming',
        'weightlifting': 'Weightlifting',
        'wrestling': 'Wrestling',
        'judo': 'Judo',
        'jiu-jitsu': 'Jiu-Jitsu',
        'jiu jitsu': 'Jiu-Jitsu',
        'shooting': 'Shooting',
        'snow': 'Snow Sports',
    }

    for key, sport in sport_mappings.items():
        if key in group_lower:
            return sport

    # Return cleaned group name if no mapping found
    return group_str.replace('_', ' ').title()


@st.cache_data
def load_all_devices() -> pd.DataFrame:
    """Load data from all devices and combine"""
    devices = ['forcedecks', 'forceframe', 'nordbord']

    all_data = []

    for device in devices:
        df = load_vald_data(device)
        if not df.empty:
            all_data.append(df)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True, sort=False)
        return combined

    return pd.DataFrame()


def get_latest_test_per_athlete(df: pd.DataFrame, test_type: Optional[str] = None) -> pd.DataFrame:
    """Get the latest test for each athlete"""
    if df.empty:
        return df

    # Filter by test type if specified
    if test_type:
        df = df[df['testType'] == test_type].copy()

    if df.empty:
        return df

    # Sort by date and get last test per athlete
    df_sorted = df.sort_values('recordedDateUtc')
    latest = df_sorted.groupby('Name').last().reset_index()

    return latest


def calculate_percentile_ranks(df: pd.DataFrame, metric: str, sport: Optional[str] = None) -> pd.Series:
    """Calculate percentile ranks for a metric"""
    if metric not in df.columns:
        return pd.Series([None] * len(df))

    # Filter by sport if specified
    if sport:
        comparison_df = df[df['athlete_sport'] == sport].copy()
    else:
        comparison_df = df.copy()

    if comparison_df.empty:
        return pd.Series([None] * len(df))

    # Calculate percentile ranks
    percentiles = comparison_df[metric].rank(pct=True) * 100

    return percentiles


def get_athlete_test_history(df: pd.DataFrame, athlete_name: str, test_type: Optional[str] = None) -> pd.DataFrame:
    """Get all tests for an athlete"""
    athlete_df = df[df['Name'] == athlete_name].copy()

    if test_type:
        athlete_df = athlete_df[athlete_df['testType'] == test_type].copy()

    # Sort by date
    if 'recordedDateUtc' in athlete_df.columns:
        athlete_df = athlete_df.sort_values('recordedDateUtc')

    return athlete_df


def calculate_metric_change(df: pd.DataFrame, metric: str, window: int = 2) -> pd.Series:
    """Calculate percentage change for a metric"""
    if metric not in df.columns or len(df) < window:
        return pd.Series([None] * len(df))

    values = df[metric].values
    changes = []

    for i in range(len(values)):
        if i < window - 1:
            changes.append(None)
        else:
            old_value = values[i - window + 1]
            new_value = values[i]

            if pd.notna(old_value) and pd.notna(new_value) and old_value != 0:
                change = ((new_value - old_value) / old_value) * 100
                changes.append(change)
            else:
                changes.append(None)

    return pd.Series(changes, index=df.index)


def get_sport_statistics(df: pd.DataFrame, sport: str, metric: str) -> Dict:
    """Get statistical summary for a sport and metric"""
    sport_df = df[df['athlete_sport'] == sport]

    if sport_df.empty or metric not in sport_df.columns:
        return {}

    values = sport_df[metric].dropna()

    if values.empty:
        return {}

    return {
        'count': len(values),
        'mean': values.mean(),
        'std': values.std(),
        'min': values.min(),
        'max': values.max(),
        'median': values.median(),
        'q25': values.quantile(0.25),
        'q75': values.quantile(0.75),
    }


def identify_outliers(df: pd.DataFrame, metric: str, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """Identify outliers in a metric"""
    if metric not in df.columns:
        return pd.Series([False] * len(df))

    values = df[metric].copy()

    if method == 'iqr':
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - (threshold * iqr)
        upper_bound = q3 + (threshold * iqr)

        outliers = (values < lower_bound) | (values > upper_bound)

    elif method == 'zscore':
        mean = values.mean()
        std = values.std()

        z_scores = abs((values - mean) / std)
        outliers = z_scores > threshold

    else:
        outliers = pd.Series([False] * len(df))

    return outliers


def calculate_asymmetry_index(left_value: float, right_value: float) -> float:
    """
    Calculate bilateral asymmetry index
    Formula: ((Left - Right) / ((Left + Right) / 2)) * 100
    """
    if pd.isna(left_value) or pd.isna(right_value):
        return None

    if left_value == 0 and right_value == 0:
        return 0.0

    average = (left_value + right_value) / 2

    if average == 0:
        return None

    asymmetry = ((left_value - right_value) / average) * 100

    return asymmetry


def get_metrics_from_test_type(test_type: str) -> List[str]:
    """Get common metrics for a test type"""
    test_metrics = {
        'CMJ': [
            'Jump Height (Flight Time) [cm]',
            'Jump Height (Imp-Mom)_Trial',
            'Peak Power / BM_Trial',
            'Peak Power [W]',
            'RSI-modified (Imp-Mom)_Trial',
            'RSI Modified',
            'Relative Peak Force [N/kg]',
            'Peak Force [N]',
            'Contraction Time [ms]',
        ],
        'SJ': [
            'Jump Height (Flight Time) [cm]',
            'Jump Height (Imp-Mom)_Trial',
            'Peak Power / BM_Trial',
            'Concentric Peak Force / BM_Trial',
            'Peak Force [N]',
        ],
        'IMTP': [
            'Peak Vertical Force / BM_Trial',
            'Peak Force [N]',
            'Relative Peak Force [N/kg]',
            'RFD - 100ms_Trial',
            'RFD - 200ms_Trial',
        ],
        'ISOT': [
            'Peak Vertical Force / BM_Trial',
            'Peak Force [N]',
            'Relative Peak Force [N/kg]',
            'RFD - 100ms_Trial',
            'RFD - 200ms_Trial',
        ],
        'ISOSQT': [
            'Peak Force [N]',
            'Relative Peak Force [N/kg]',
            'Peak Force Left [N]',
            'Peak Force Right [N]',
            'Asymmetry Index',
        ],
        'DJ': [
            'Jump Height (Flight Time) [cm]',
            'RSI Modified',
            'Contact Time [ms]',
            'Peak Power [W]',
        ],
        'SLCMJ': [
            'Jump Height (Flight Time) [cm]',
            'Jump Height Left [cm]',
            'Jump Height Right [cm]',
            'Peak Force [N]',
            'Asymmetry Index',
        ],
    }

    return test_metrics.get(test_type, [])


def filter_dataframe(
    df: pd.DataFrame,
    sports: Optional[List[str]] = None,
    athletes: Optional[List[str]] = None,
    test_types: Optional[List[str]] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None
) -> pd.DataFrame:
    """Apply multiple filters to dataframe"""
    filtered_df = df.copy()

    if sports:
        filtered_df = filtered_df[filtered_df['athlete_sport'].isin(sports)]

    if athletes:
        filtered_df = filtered_df[filtered_df['Name'].isin(athletes)]

    if test_types:
        filtered_df = filtered_df[filtered_df['testType'].isin(test_types)]

    if date_range and 'recordedDateUtc' in filtered_df.columns:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['recordedDateUtc'] >= start_date) &
            (filtered_df['recordedDateUtc'] <= end_date)
        ]

    return filtered_df


def get_available_metrics(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns that could be metrics"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Filter out system columns
    excluded = ['testId', 'profileId', 'tenantId', 'Reps', 'Reps (L)', 'Reps (R)']

    metrics = [col for col in numeric_cols if col not in excluded and not col.endswith('_id')]

    return sorted(metrics)


def calculate_z_score(df: pd.DataFrame, metric: str, sport: Optional[str] = None) -> pd.Series:
    """Calculate z-scores for a metric"""
    if metric not in df.columns:
        return pd.Series([None] * len(df))

    # Filter by sport if specified
    if sport:
        comparison_df = df[df['athlete_sport'] == sport].copy()
    else:
        comparison_df = df.copy()

    if comparison_df.empty or metric not in comparison_df.columns:
        return pd.Series([None] * len(df))

    mean = comparison_df[metric].mean()
    std = comparison_df[metric].std()

    if pd.isna(mean) or pd.isna(std) or std == 0:
        return pd.Series([None] * len(df))

    z_scores = (df[metric] - mean) / std

    return z_scores


def get_test_summary_stats(df: pd.DataFrame) -> Dict:
    """Get summary statistics for testing"""
    if df.empty:
        return {}

    stats = {
        'total_tests': len(df),
        'total_athletes': df['Name'].nunique() if 'Name' in df.columns else 0,
        'total_sports': df['athlete_sport'].nunique() if 'athlete_sport' in df.columns else 0,
        'test_types': df['testType'].nunique() if 'testType' in df.columns else 0,
    }

    if 'recordedDateUtc' in df.columns:
        df_with_dates = df[df['recordedDateUtc'].notna()]
        if not df_with_dates.empty:
            stats['date_range'] = {
                'start': df_with_dates['recordedDateUtc'].min(),
                'end': df_with_dates['recordedDateUtc'].max(),
                'days_span': (df_with_dates['recordedDateUtc'].max() - df_with_dates['recordedDateUtc'].min()).days
            }

    return stats


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")

    df = load_vald_data('forcedecks')

    if not df.empty:
        print(f"Loaded {len(df)} records")
        print(f"Columns: {df.columns.tolist()[:10]}")
        print(f"Sports: {df['athlete_sport'].unique()}")
    else:
        print("No data loaded")
