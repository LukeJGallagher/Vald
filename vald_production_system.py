"""
VALD Production System - Team Saudi
Unified data management for ForceDecks, ForceFrame, and NordBord

Implements all 5 critical actions:
1. Fresh API Data Pull - Incremental updates from all devices
2. API Credential Verification - Test OAuth and endpoints
3. Automated Update Scheduling - Configurable daily/weekly updates
4. Multi-Device Integration - ForceDecks, ForceFrame, NordBord + Athletes
5. API Change Monitoring - Check R package for updates

Version: 1.0
Author: Performance Analysis Team
Date: 2025-11-24
"""

import os
import sys
import json
import pickle
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from threading import Lock
import time
import logging
from logging.handlers import RotatingFileHandler
import re

# Add config to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config.vald_config import ValdConfig, DEVICE_CONFIG, SPORT_MAPPINGS
except ImportError:
    print("ERROR: config/vald_config.py not found!")
    sys.exit(1)


# ============================================================================
# LOGGING SETUP
# ============================================================================

class ValdLogger:
    """Structured logging with rotation"""

    def __init__(self, name: str, log_dir: str = 'logs'):
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console.setFormatter(console_format)

        # File handler with rotation
        file_handler = RotatingFileHandler(
            f'{log_dir}/vald_system_{datetime.now().strftime("%Y%m%d")}.log',
            maxBytes=10485760,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)

        self.logger.addHandler(console)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger


# ============================================================================
# OAUTH TOKEN MANAGER
# ============================================================================

class OAuthTokenManager:
    """Manages OAuth tokens with caching and auto-refresh"""

    def __init__(self, config: ValdConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.token_cache_file = config.TOKEN_CACHE_FILE
        self.token_url = config.TOKEN_URL

        # Ensure cache directory exists
        os.makedirs(os.path.dirname(self.token_cache_file), exist_ok=True)

    def get_token(self) -> str:
        """Get valid access token (cached or fresh)"""
        # Try manual token first
        if self.config.MANUAL_TOKEN:
            if self._test_token(self.config.MANUAL_TOKEN):
                self.logger.info("Using manual token")
                return self.config.MANUAL_TOKEN

        # Try cached token
        cached = self._get_cached_token()
        if cached:
            if self._test_token(cached):
                self.logger.info("Using cached token")
                return cached

        # Get fresh token
        self.logger.info("Requesting fresh OAuth token")
        return self._get_fresh_token()

    def _get_cached_token(self) -> Optional[str]:
        """Load token from cache if valid"""
        if not os.path.exists(self.token_cache_file):
            return None

        try:
            with open(self.token_cache_file, 'rb') as f:
                cache = pickle.load(f)

            if cache['expiry'] > datetime.now():
                return cache['token']
        except Exception as e:
            self.logger.warning(f"Cache load failed: {e}")

        return None

    def _get_fresh_token(self) -> str:
        """Request new OAuth token from API"""
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.config.CLIENT_ID,
            'client_secret': self.config.CLIENT_SECRET,
            'audience': self.config.AUDIENCE,
        }

        try:
            response = requests.post(
                self.token_url,
                headers=headers,
                data=data,
                timeout=self.config.REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                token_data = response.json()
                access_token = token_data.get('access_token')
                expires_in = token_data.get('expires_in', 7200)

                # Cache token (refresh 5 min before expiry)
                expiry_time = datetime.now() + timedelta(seconds=expires_in - 300)

                with open(self.token_cache_file, 'wb') as f:
                    pickle.dump({
                        'token': access_token,
                        'expiry': expiry_time
                    }, f)

                self.logger.info(f"New token obtained (expires in {expires_in}s)")
                return access_token
            else:
                raise Exception(f"Token request failed: {response.status_code}")

        except Exception as e:
            self.logger.error(f"OAuth token error: {e}")
            raise

    def _test_token(self, token: str) -> bool:
        """Test if token is valid"""
        try:
            # Test with profiles endpoint
            url = self.config.get_endpoint('profiles') + 'profiles'
            headers = {'Authorization': f'Bearer {token}'}
            params = {'tenantId': self.config.TENANT_ID}

            response = requests.get(url, headers=headers, params=params, timeout=10)
            return response.status_code == 200

        except:
            return False


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """Thread-safe rate limiting with queue tracking"""

    def __init__(self, calls_per_window: int = 12, window_seconds: float = 5.0):
        self.calls_per_window = calls_per_window
        self.window_seconds = window_seconds
        self.call_times = deque()
        self.lock = Lock()

    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = time.time()

            # Remove old calls outside window
            while self.call_times and self.call_times[0] < now - self.window_seconds:
                self.call_times.popleft()

            # Check if we need to wait
            if len(self.call_times) >= self.calls_per_window:
                sleep_time = self.window_seconds - (now - self.call_times[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()

            # Record this call
            self.call_times.append(now)

            # Small delay between all calls
            time.sleep(0.3)


# ============================================================================
# BASE DEVICE ADAPTER
# ============================================================================

class BaseDeviceAdapter:
    """Base class for all device adapters"""

    def __init__(self, config: ValdConfig, token_manager: OAuthTokenManager,
                 rate_limiter: RateLimiter, logger: logging.Logger):
        self.config = config
        self.token_manager = token_manager
        self.rate_limiter = rate_limiter
        self.logger = logger
        self.device_name = None  # Set by subclass

    def safe_api_call(self, url: str, params: Optional[Dict] = None,
                      headers: Optional[Dict] = None, method: str = 'GET') -> Optional[requests.Response]:
        """Make rate-limited API call with retry logic"""
        token = self.token_manager.get_token()

        if headers is None:
            headers = {}
        headers['Authorization'] = f'Bearer {token}'

        for attempt in range(self.config.MAX_RETRIES):
            try:
                self.rate_limiter.wait_if_needed()

                if method == 'GET':
                    response = requests.get(url, headers=headers, params=params,
                                          timeout=self.config.REQUEST_TIMEOUT)
                else:
                    response = requests.post(url, headers=headers, json=params,
                                           timeout=self.config.REQUEST_TIMEOUT)

                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = 15 + (attempt * 10)
                    self.logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Handle auth errors
                if response.status_code == 401:
                    self.logger.warning("Token expired, refreshing...")
                    token = self.token_manager._get_fresh_token()
                    headers['Authorization'] = f'Bearer {token}'
                    continue

                if response.status_code in [200, 204]:
                    return response

                self.logger.warning(f"API call failed: {response.status_code}")
                return None

            except Exception as e:
                self.logger.error(f"API call error (attempt {attempt + 1}): {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)

        return None

    def get_last_update_date(self) -> str:
        """Get last update date from existing CSV"""
        master_file = os.path.join(self.config.MASTER_DIR,
                                   DEVICE_CONFIG[self.device_name]['master_file'])

        if os.path.exists(master_file):
            try:
                df = pd.read_csv(master_file)
                if 'modifiedDateUtc' in df.columns and len(df) > 0:
                    last_date = pd.to_datetime(df['modifiedDateUtc']).max()
                    return last_date.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            except Exception as e:
                self.logger.warning(f"Could not read last update: {e}")

        return self.config.DEFAULT_START_DATE

    def save_data(self, df: pd.DataFrame, backup: bool = True):
        """Save data with optional backup"""
        master_file = os.path.join(self.config.MASTER_DIR,
                                   DEVICE_CONFIG[self.device_name]['master_file'])

        # Backup existing file
        if backup and os.path.exists(master_file):
            backup_file = os.path.join(
                self.config.BACKUP_DIR,
                f"{self.device_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            try:
                os.rename(master_file, backup_file)
                self.logger.info(f"Backed up to: {backup_file}")
            except Exception as e:
                self.logger.warning(f"Backup failed: {e}")

        # Save new data
        df.to_csv(master_file, index=False, encoding='utf-8')
        self.logger.info(f"Saved {len(df)} records to {master_file}")


# ============================================================================
# FORCEDECKS ADAPTER
# ============================================================================

class ForceDecksAdapter(BaseDeviceAdapter):
    """ForceDecks device with comprehensive trials"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_name = 'forcedecks'
        self.base_url = self.config.get_endpoint('forcedecks')

    def fetch_tests(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch tests with comprehensive trial data"""
        if start_date is None:
            start_date = self.get_last_update_date()

        self.logger.info(f"Fetching ForceDecks tests from {start_date}")

        all_tests = []
        modified_from_utc = start_date

        while True:
            url = f"{self.base_url}tests"
            params = {
                'tenantId': self.config.TENANT_ID,
                'modifiedFromUtc': modified_from_utc
            }

            response = self.safe_api_call(url, params=params)

            if response is None or response.status_code == 204:
                self.logger.info("No more tests to fetch")
                break

            try:
                data = response.json()
                tests = data.get('tests', [])

                if not tests:
                    break

                all_tests.extend(tests)
                modified_from_utc = tests[-1]['modifiedDateUtc']
                self.logger.info(f"Fetched {len(tests)} tests, total: {len(all_tests)}")

            except Exception as e:
                self.logger.error(f"Error parsing tests: {e}")
                break

        if not all_tests:
            self.logger.info("No new tests found")
            return pd.DataFrame()

        # Fetch trial data for each test
        if self.config.ENABLE_TRIAL_FETCHING:
            all_tests = self._fetch_trials_for_tests(all_tests)

        # Convert to DataFrame
        df = pd.json_normalize(all_tests)
        self.logger.info(f"Loaded {len(df)} ForceDecks tests")

        return df

    def _fetch_trials_for_tests(self, tests: List[Dict]) -> List[Dict]:
        """Fetch comprehensive trial data for all tests"""
        self.logger.info(f"Fetching trials for {len(tests)} tests...")

        for i, test in enumerate(tests):
            test_id = test.get('testId')

            if not test_id:
                continue

            if (i + 1) % 10 == 0:
                self.logger.info(f"Progress: {i + 1}/{len(tests)} tests")

            url = f"{self.base_url}v2019q3/teams/{self.config.TENANT_ID}/tests/{test_id}/trials"

            response = self.safe_api_call(url)

            if response is None:
                continue

            try:
                trial_data = response.json()

                # Store trial IDs for potential trace fetching
                test['trial_ids'] = []

                # Extract metrics from trials
                for trial in trial_data:
                    trial_id = trial.get('id')
                    if trial_id:
                        test['trial_ids'].append(trial_id)

                    results = trial.get('results', [])
                    for result in results:
                        metric_name = result.get('definition', {}).get('name')
                        limb = result.get('limb', 'Trial')
                        value = result.get('value')

                        if metric_name and value is not None:
                            field_name = f"{metric_name}_{limb}"
                            test[field_name] = value

            except Exception as e:
                self.logger.warning(f"Trial fetch failed for {test_id}: {e}")

        return tests

    def fetch_force_traces(self, test_ids: List[str], output_dir: str = 'data/force_traces') -> Dict:
        """
        Fetch raw force-time traces for specific tests

        SELECTIVE USE: Only fetch traces when needed (large data)
        Use for: Biomechanics analysis, phase analysis, athlete comparisons

        Parameters:
        -----------
        test_ids : List[str]
            List of test IDs to fetch traces for
        output_dir : str
            Directory to save trace data

        Returns:
        --------
        Dict with test_id -> trace data mapping
        """
        os.makedirs(output_dir, exist_ok=True)

        self.logger.info(f"Fetching force traces for {len(test_ids)} tests...")
        trace_data = {}

        for test_id in test_ids:
            # Get trials for this test
            url = f"{self.base_url}v2019q3/teams/{self.config.TENANT_ID}/tests/{test_id}/trials"
            response = self.safe_api_call(url)

            if response is None:
                continue

            try:
                trials = response.json()
                test_traces = []

                for trial in trials:
                    trial_id = trial.get('id')
                    if not trial_id:
                        continue

                    # Fetch trace for this trial
                    trace_url = f"{self.base_url}v2019q3/teams/{self.config.TENANT_ID}/tests/{test_id}/trials/{trial_id}/trace"
                    trace_response = self.safe_api_call(trace_url)

                    if trace_response and trace_response.status_code == 200:
                        trace = trace_response.json()
                        test_traces.append({
                            'trial_id': trial_id,
                            'trace_data': trace
                        })

                        self.logger.info(f"Fetched trace for test {test_id}, trial {trial_id}")

                if test_traces:
                    trace_data[test_id] = test_traces

                    # Save to JSON
                    import json
                    output_file = os.path.join(output_dir, f"{test_id}_traces.json")
                    with open(output_file, 'w') as f:
                        json.dump(test_traces, f)

            except Exception as e:
                self.logger.error(f"Trace fetch failed for {test_id}: {e}")

        self.logger.info(f"Fetched traces for {len(trace_data)} tests")
        return trace_data


# ============================================================================
# FORCEFRAME ADAPTER
# ============================================================================

class ForceFrameAdapter(BaseDeviceAdapter):
    """ForceFrame device adapter"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_name = 'forceframe'
        self.base_url = self.config.get_endpoint('forceframe')

    def fetch_tests(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch ForceFrame tests"""
        if start_date is None:
            start_date = self.get_last_update_date()

        self.logger.info(f"Fetching ForceFrame tests from {start_date}")

        all_tests = []
        modified_from_utc = start_date

        while True:
            url = f"{self.base_url}tests"
            params = {
                'TenantId': self.config.TENANT_ID,
                'ModifiedFromUtc': modified_from_utc
            }

            response = self.safe_api_call(url, params=params)

            if response is None or response.status_code == 204:
                break

            try:
                data = response.json()
                tests = data.get('tests', [])

                if not tests:
                    break

                all_tests.extend(tests)
                modified_from_utc = tests[-1]['modifiedDateUtc']
                self.logger.info(f"Fetched {len(tests)} tests, total: {len(all_tests)}")

            except Exception as e:
                self.logger.error(f"Error parsing tests: {e}")
                break

        if not all_tests:
            self.logger.info("No new ForceFrame tests found")
            return pd.DataFrame()

        df = pd.json_normalize(all_tests)
        self.logger.info(f"Loaded {len(df)} ForceFrame tests")

        return df


# ============================================================================
# NORDBORD ADAPTER
# ============================================================================

class NordBordAdapter(BaseDeviceAdapter):
    """NordBord device adapter"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_name = 'nordbord'
        self.base_url = self.config.get_endpoint('nordbord')

    def fetch_tests(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch NordBord tests"""
        self.logger.info("Fetching NordBord tests")

        all_tests = []
        page = 1

        while True:
            url = f"{self.base_url}tests"
            params = {
                'TenantId': self.config.TENANT_ID,
                'Page': page
            }

            response = self.safe_api_call(url, params=params)

            if response is None or response.status_code == 204:
                break

            try:
                tests = response.json()

                if not tests:
                    break

                all_tests.extend(tests)
                self.logger.info(f"Fetched page {page}, total: {len(all_tests)} tests")
                page += 1

            except Exception as e:
                self.logger.error(f"Error parsing tests: {e}")
                break

        if not all_tests:
            self.logger.info("No new NordBord tests found")
            return pd.DataFrame()

        # Deduplicate by ID
        unique_tests = {test['id']: test for test in all_tests}.values()
        df = pd.json_normalize(list(unique_tests))

        self.logger.info(f"Loaded {len(df)} unique NordBord tests")

        return df


# ============================================================================
# ATHLETE DATA MANAGER
# ============================================================================

class AthleteDataManager:
    """Manage athlete data fetching and integration"""

    def __init__(self, config: ValdConfig, token_manager: OAuthTokenManager,
                 rate_limiter: RateLimiter, logger: logging.Logger):
        self.config = config
        self.token_manager = token_manager
        self.rate_limiter = rate_limiter
        self.logger = logger
        self.profiles_url = config.get_endpoint('profiles')

    def fetch_athletes(self) -> pd.DataFrame:
        """Fetch athlete/profile details from Profiles API"""
        self.logger.info("Fetching athlete data from Profiles API...")

        # Use the External Profiles API endpoint
        url = f"{self.profiles_url}api/v1/profiles"

        token = self.token_manager.get_token()
        headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json'
        }
        params = {'TenantId': self.config.TENANT_ID}

        self.rate_limiter.wait_if_needed()

        try:
            response = requests.get(url, headers=headers, params=params,
                                   timeout=self.config.REQUEST_TIMEOUT)

            if response.status_code == 200:
                profiles = response.json()

                if isinstance(profiles, list):
                    # Normalize the profile data
                    athletes = []
                    for p in profiles:
                        athlete = {
                            'id': p.get('id') or p.get('profileId'),
                            'profileId': p.get('id') or p.get('profileId'),
                            'givenName': p.get('givenName', ''),
                            'familyName': p.get('familyName', ''),
                            'fullName': p.get('fullName') or p.get('name') or p.get('displayName', ''),
                            'sport': p.get('sport') or p.get('primarySport', 'Unknown'),
                            'email': p.get('email', ''),
                            'externalId': p.get('externalId', ''),
                            'sex': p.get('sex', ''),
                            'dateOfBirth': p.get('dateOfBirth', ''),
                            'weightInKG': p.get('weightInKG'),
                            'heightInCM': p.get('heightInCM'),
                        }
                        # Build full name if not provided
                        if not athlete['fullName']:
                            athlete['fullName'] = f"{athlete['givenName']} {athlete['familyName']}".strip()
                        athletes.append(athlete)

                    df = pd.DataFrame(athletes)
                    self.logger.info(f"Loaded {len(df)} athlete profiles")
                    return df
                else:
                    self.logger.error("Unexpected profile data format")
                    return pd.DataFrame()
            else:
                self.logger.error(f"Profile fetch failed: {response.status_code}")
                # Log response for debugging
                self.logger.error(f"Response: {response.text[:500] if response.text else 'No response body'}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Profile fetch error: {e}")
            return pd.DataFrame()

    def create_athlete_mapping(self, athletes_df: pd.DataFrame) -> Dict:
        """Create lookup dictionaries for athlete matching"""
        mapping = {
            'by_name': {},
            'by_id': {},
            'by_profile_id': {},
            'by_external_id': {}
        }

        for _, athlete in athletes_df.iterrows():
            athlete_dict = athlete.to_dict()

            # Name lookup - try fullName first, then construct from parts
            full_name = athlete.get('fullName', '')
            if not full_name:
                full_name = f"{athlete.get('givenName', '')} {athlete.get('familyName', '')}".strip()
            if full_name:
                mapping['by_name'][full_name.lower()] = athlete_dict

            # ID lookups - profile ID (used by ForceDecks)
            profile_id = athlete.get('profileId') or athlete.get('id')
            if profile_id and pd.notna(profile_id):
                mapping['by_id'][str(profile_id)] = athlete_dict
                mapping['by_profile_id'][str(profile_id)] = athlete_dict

            # External ID lookup
            if 'externalId' in athlete and pd.notna(athlete['externalId']):
                mapping['by_external_id'][str(athlete['externalId'])] = athlete_dict

        self.logger.info(f"Created athlete mapping with {len(mapping['by_name'])} athletes")
        return mapping

    def enrich_with_athlete_data(self, df: pd.DataFrame, athlete_mapping: Dict) -> pd.DataFrame:
        """Enrich test data with athlete information"""
        self.logger.info("Enriching data with athlete details...")

        enriched_rows = []

        for _, row in df.iterrows():
            athlete = None

            # Try matching by profileId (ForceDecks)
            if 'profileId' in row and pd.notna(row.get('profileId')):
                athlete = athlete_mapping['by_profile_id'].get(str(row['profileId']))

            # Try matching by athleteId (ForceFrame/NordBord)
            if not athlete and 'athleteId' in row and pd.notna(row.get('athleteId')):
                athlete = athlete_mapping['by_id'].get(str(row['athleteId']))

            # Try matching by athlete_id (alternate column name)
            if not athlete and 'athlete_id' in row and pd.notna(row.get('athlete_id')):
                athlete = athlete_mapping['by_id'].get(str(row['athlete_id']))

            # Try matching by name
            if not athlete and 'Name' in row and pd.notna(row.get('Name')):
                name_lower = str(row['Name']).lower()
                athlete = athlete_mapping['by_name'].get(name_lower)

            if athlete:
                # Add athlete name if not already present
                if 'full_name' not in row or pd.isna(row.get('full_name')):
                    given = athlete.get('givenName', '')
                    family = athlete.get('familyName', '')
                    row['full_name'] = f"{given} {family}".strip()

                row['athlete_sport'] = athlete.get('sport', 'Unknown')
                row['athlete_sex'] = athlete.get('sex', 'Unknown')
                row['athlete_dob'] = athlete.get('dateOfBirth')
                row['athlete_weight_kg'] = athlete.get('weightInKG')
                row['athlete_height_cm'] = athlete.get('heightInCM')
                row['athlete_position'] = athlete.get('sportSpecificPosition')
                row['athlete_email'] = athlete.get('email', '')

            enriched_rows.append(row)

        enriched_df = pd.DataFrame(enriched_rows)
        matched = enriched_df['athlete_sport'].notna().sum()
        self.logger.info(f"Matched {matched}/{len(enriched_df)} records with athlete data")

        return enriched_df


# ============================================================================
# API HEALTH MONITOR
# ============================================================================

class APIHealthMonitor:
    """Monitor API health and check for changes"""

    def __init__(self, config: ValdConfig, token_manager: OAuthTokenManager,
                 rate_limiter: RateLimiter, logger: logging.Logger):
        self.config = config
        self.token_manager = token_manager
        self.rate_limiter = rate_limiter
        self.logger = logger

    def verify_credentials(self) -> Dict:
        """Verify API credentials and endpoints"""
        self.logger.info("Verifying API credentials...")

        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'oauth': False,
            'endpoints': {}
        }

        # Test OAuth
        try:
            token = self.token_manager.get_token()
            results['oauth'] = bool(token)
            results['token_length'] = len(token) if token else 0
        except Exception as e:
            results['oauth_error'] = str(e)

        # Test endpoints
        endpoints_to_test = ['forcedecks', 'forceframe', 'nordbord', 'athletes', 'profiles']

        for endpoint_name in endpoints_to_test:
            try:
                url = self.config.get_endpoint(endpoint_name)
                token = self.token_manager.get_token()
                headers = {'Authorization': f'Bearer {token}'}

                self.rate_limiter.wait_if_needed()

                response = requests.get(url, headers=headers, timeout=10)
                results['endpoints'][endpoint_name] = {
                    'status_code': response.status_code,
                    'accessible': response.status_code in [200, 204],
                    'url': url
                }

            except Exception as e:
                results['endpoints'][endpoint_name] = {
                    'error': str(e),
                    'accessible': False
                }

        return results

    def check_r_package_updates(self) -> Dict:
        """Check VALD R package for API updates"""
        self.logger.info("Checking VALD R package for updates...")

        r_package_url = "https://cran.r-project.org/web/packages/valdr/index.html"

        try:
            response = requests.get(r_package_url, timeout=10)

            if response.status_code == 200:
                version_match = re.search(r'Version:\s*</td><td>([0-9.]+)', response.text)
                date_match = re.search(r'Published:\s*</td><td>([0-9-]+)', response.text)

                if version_match and date_match:
                    return {
                        'version': version_match.group(1),
                        'published_date': date_match.group(1),
                        'url': r_package_url,
                        'status': 'available'
                    }

            return {'status': 'error', 'message': 'Could not fetch R package version'}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}


# ============================================================================
# UNIFIED VALD SYSTEM (Main Orchestrator)
# ============================================================================

class UnifiedValdSystem:
    """Main system orchestrator - All 5 actions"""

    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = ValdConfig.from_env(config_path)
        self.config.validate()
        self.config.ensure_directories()

        # Setup logging
        logger_setup = ValdLogger('vald_system', self.config.LOG_DIR)
        self.logger = logger_setup.get_logger()

        # Initialize core components
        self.token_manager = OAuthTokenManager(self.config, self.logger)
        self.rate_limiter = RateLimiter(
            self.config.RATE_LIMIT_CALLS,
            self.config.RATE_LIMIT_WINDOW
        )

        # Initialize device adapters
        self.devices = {
            'forcedecks': ForceDecksAdapter(
                self.config, self.token_manager, self.rate_limiter, self.logger
            )
        }

        if self.config.ENABLE_FORCEFRAME:
            self.devices['forceframe'] = ForceFrameAdapter(
                self.config, self.token_manager, self.rate_limiter, self.logger
            )

        if self.config.ENABLE_NORDBORD:
            self.devices['nordbord'] = NordBordAdapter(
                self.config, self.token_manager, self.rate_limiter, self.logger
            )

        # Initialize athlete manager
        self.athlete_manager = AthleteDataManager(
            self.config, self.token_manager, self.rate_limiter, self.logger
        )

        # Initialize health monitor
        self.health_monitor = APIHealthMonitor(
            self.config, self.token_manager, self.rate_limiter, self.logger
        )

    # ACTION 1: Update Data from API
    def update_all_devices(self):
        """Fetch fresh data from all devices"""
        print("=" * 80)
        print("ACTION 1: FRESH API DATA PULL")
        print("=" * 80)
        self.logger.info("Starting multi-device update...")

        for device_name, adapter in self.devices.items():
            try:
                print(f"\nUpdating {device_name.upper()}...")
                df = adapter.fetch_tests()

                if not df.empty:
                    adapter.save_data(df, backup=self.config.ENABLE_BACKUP)
                    print(f"SUCCESS: {len(df)} tests saved")
                else:
                    print(f"INFO: No new {device_name} data")

            except Exception as e:
                self.logger.error(f"{device_name} update failed: {e}")
                print(f"ERROR: {device_name} update failed - {e}")

        print("\nAll devices updated!")

    # ACTION 2: Verify API Credentials
    def verify_credentials(self):
        """Test OAuth and endpoint accessibility"""
        print("=" * 80)
        print("ACTION 2: API CREDENTIAL VERIFICATION")
        print("=" * 80)

        results = self.health_monitor.verify_credentials()

        print(f"\nTimestamp: {results['timestamp']}")
        print(f"\nOAuth Token: {'VALID' if results['oauth'] else 'INVALID'}")

        if results['oauth']:
            print(f"Token length: {results.get('token_length', 0)} characters")

        print("\nEndpoint Accessibility:")
        for endpoint, data in results['endpoints'].items():
            status = "ACCESSIBLE" if data.get('accessible') else "FAILED"
            print(f"  {endpoint}: {status} (Code: {data.get('status_code', 'N/A')})")

        return results

    # ACTION 3: Schedule Automated Updates
    def schedule_updates(self, frequency: str = 'daily', time_str: str = '02:00'):
        """Setup automated scheduling"""
        print("=" * 80)
        print("ACTION 3: AUTOMATED UPDATE SCHEDULING")
        print("=" * 80)
        print(f"\nSchedule: {frequency} at {time_str}")
        print("\nNOTE: Requires utils/vald_scheduler.py to be implemented")
        print("See documentation for Windows Task Scheduler or cron setup")

    # ACTION 4: Integrate Athlete Data
    def integrate_athlete_data(self, device_name: str = 'forcedecks'):
        """Fetch athletes and enrich device data"""
        print("=" * 80)
        print("ACTION 4: ATHLETE DATA INTEGRATION")
        print("=" * 80)

        # Fetch athletes
        print("\nFetching athlete data...")
        athletes_df = self.athlete_manager.fetch_athletes()

        if athletes_df.empty:
            print("ERROR: No athlete data found")
            return

        print(f"Loaded {len(athletes_df)} athletes")

        # Create mapping
        athlete_mapping = self.athlete_manager.create_athlete_mapping(athletes_df)

        # Load device data
        master_file = os.path.join(self.config.MASTER_DIR,
                                   DEVICE_CONFIG[device_name]['master_file'])

        if not os.path.exists(master_file):
            print(f"ERROR: No {device_name} data found")
            return

        df = pd.read_csv(master_file)
        print(f"\nEnriching {len(df)} {device_name} records...")

        # Enrich data
        enriched_df = self.athlete_manager.enrich_with_athlete_data(df, athlete_mapping)

        # Save enriched data
        self.devices[device_name].save_data(enriched_df, backup=True)

        print(f"\nSUCCESS: Enriched data saved")
        print(f"Sports found: {enriched_df['athlete_sport'].value_counts().to_dict()}")

    # ACTION 5: Check for API Updates
    def check_api_updates(self):
        """Check R package for API changes"""
        print("=" * 80)
        print("ACTION 5: API CHANGE MONITORING")
        print("=" * 80)

        r_package = self.health_monitor.check_r_package_updates()

        if r_package.get('status') == 'available':
            print("\nVALD R Package (valdr):")
            print(f"  Version: {r_package['version']}")
            print(f"  Published: {r_package['published_date']}")
            print(f"  URL: {r_package['url']}")
        else:
            print(f"\nWARNING: Could not check R package - {r_package.get('message')}")

        print("\nNOTE: Compare with your Python implementation")
        print("See dashboard/utils/vald_version_monitor.py for detailed comparison")

    # BONUS: Fetch Force Traces (Selective)
    def fetch_selected_traces(self, athlete_names: Optional[List[str]] = None,
                             test_types: Optional[List[str]] = None,
                             max_tests: int = 10):
        """
        Fetch force traces for selected athletes/tests

        This is SELECTIVE - only fetches what you need due to large data size

        Parameters:
        -----------
        athlete_names : List[str], optional
            Filter by athlete names
        test_types : List[str], optional
            Filter by test types (e.g., ['CMJ', 'SJ'])
        max_tests : int
            Maximum number of tests to fetch traces for (default: 10)
        """
        print("=" * 80)
        print("BONUS: SELECTIVE FORCE TRACE FETCHING")
        print("=" * 80)

        # Load existing ForceDecks data to find test IDs
        master_file = os.path.join(self.config.MASTER_DIR,
                                   'forcedecks_allsports_with_athletes.csv')

        if not os.path.exists(master_file):
            print("ERROR: No ForceDecks data found. Run update first.")
            return

        import pandas as pd
        df = pd.read_csv(master_file)

        # Apply filters
        if athlete_names:
            df = df[df['Name'].isin(athlete_names)]
        if test_types:
            df = df[df['testType'].isin(test_types)]

        # Get most recent tests
        df = df.sort_values('recordedDateUtc', ascending=False).head(max_tests)

        test_ids = df['testId'].tolist()

        print(f"\nFetching traces for {len(test_ids)} tests...")
        if athlete_names:
            print(f"Athletes: {', '.join(athlete_names)}")
        if test_types:
            print(f"Test types: {', '.join(test_types)}")

        # Fetch traces
        trace_data = self.devices['forcedecks'].fetch_force_traces(test_ids)

        print(f"\nSUCCESS: Fetched {len(trace_data)} force traces")
        print(f"Saved to: data/force_traces/")

        return trace_data


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Interactive CLI"""
    import argparse

    parser = argparse.ArgumentParser(description='VALD Production System - Team Saudi')

    parser.add_argument('--update-all', action='store_true',
                       help='Update all devices (Action 1)')
    parser.add_argument('--verify-credentials', action='store_true',
                       help='Verify API credentials (Action 2)')
    parser.add_argument('--schedule', type=str, choices=['daily', 'weekly'],
                       help='Setup automated scheduling (Action 3)')
    parser.add_argument('--integrate-athletes', type=str,
                       help='Integrate athlete data for device (Action 4)')
    parser.add_argument('--check-api-updates', action='store_true',
                       help='Check R package for updates (Action 5)')
    parser.add_argument('--config', type=str,
                       help='Path to .env config file')

    args = parser.parse_args()

    # Initialize system
    system = UnifiedValdSystem(config_path=args.config)

    # Execute actions
    if args.update_all:
        system.update_all_devices()

    if args.verify_credentials:
        system.verify_credentials()

    if args.schedule:
        system.schedule_updates(frequency=args.schedule)

    if args.integrate_athletes:
        system.integrate_athlete_data(device_name=args.integrate_athletes)

    if args.check_api_updates:
        system.check_api_updates()

    # If no arguments, show interactive menu
    if not any(vars(args).values()):
        print("\n" + "=" * 80)
        print("VALD PRODUCTION SYSTEM - TEAM SAUDI")
        print("=" * 80)
        print("\n1. Update All Devices (Fresh Data Pull)")
        print("2. Verify API Credentials")
        print("3. Setup Automated Scheduling")
        print("4. Integrate Athlete Data")
        print("5. Check for API Updates")
        print("6. Run All Actions")
        print("0. Exit")

        choice = input("\nSelect action (0-6): ")

        if choice == '1':
            system.update_all_devices()
        elif choice == '2':
            system.verify_credentials()
        elif choice == '3':
            system.schedule_updates()
        elif choice == '4':
            device = input("Enter device (forcedecks/forceframe/nordbord): ")
            system.integrate_athlete_data(device)
        elif choice == '5':
            system.check_api_updates()
        elif choice == '6':
            system.verify_credentials()
            system.update_all_devices()
            system.integrate_athlete_data('forcedecks')
            system.check_api_updates()


if __name__ == "__main__":
    main()
