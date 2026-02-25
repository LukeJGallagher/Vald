"""
Centralized VALD Configuration
Production-ready settings for Team Saudi VALD Data System
"""

import os
import time
import requests
from dataclasses import dataclass, field
from typing import Dict, Optional, List
try:
    from dotenv import load_dotenv
except ImportError:
    # dotenv not available (e.g. GitHub Actions) - env vars set directly
    def load_dotenv(*args, **kwargs):
        pass


@dataclass
class ValdConfig:
    """Centralized configuration for VALD API system"""

    # Authentication (Auth0 - Updated Feb 2026)
    CLIENT_ID: str
    CLIENT_SECRET: str
    TENANT_ID: str
    TOKEN_URL: str = 'https://auth.prd.vald.com/oauth/token'
    AUDIENCE: str = 'vald-api-external'
    MANUAL_TOKEN: Optional[str] = None

    # Region
    REGION: str = 'euw'

    # Rate Limiting
    RATE_LIMIT_CALLS: int = 12
    RATE_LIMIT_WINDOW: float = 5.0
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_FACTOR: float = 2.0

    # Directories
    DATA_DIR: str = 'data'
    MASTER_DIR: str = 'data/master_files'
    CACHE_DIR: str = 'data/cache'
    LOG_DIR: str = 'logs'
    BACKUP_DIR: str = 'data/backups'

    # Feature Flags
    ENABLE_ATHLETE_INTEGRATION: bool = True
    ENABLE_TRIAL_FETCHING: bool = True
    ENABLE_BACKUP: bool = True
    ENABLE_MONITORING: bool = True
    ENABLE_FORCEFRAME: bool = True
    ENABLE_NORDBORD: bool = True

    # Pagination
    PAGE_SIZE: int = 100
    DEFAULT_START_DATE: str = '2024-01-01T00:00:00.000Z'

    # Token Management
    TOKEN_CACHE_FILE: str = 'data/cache/token_cache.pkl'
    TOKEN_REFRESH_BUFFER: int = 300  # Refresh 5 min before expiry

    # Logging
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_MAX_BYTES: int = 10485760  # 10MB
    LOG_BACKUP_COUNT: int = 5

    # Email Notifications (optional)
    ENABLE_EMAIL_NOTIFICATIONS: bool = False
    EMAIL_FROM: Optional[str] = None
    EMAIL_TO: Optional[List[str]] = None
    SMTP_SERVER: Optional[str] = None
    SMTP_PORT: int = 587

    # Scheduling
    SCHEDULE_ENABLED: bool = False
    SCHEDULE_TIME: str = '02:00'  # 2 AM daily
    SCHEDULE_FREQUENCY: str = 'daily'  # daily, weekly, custom

    @classmethod
    def from_env(cls, env_path: Optional[str] = None):
        """Load configuration from .env file"""
        # Try multiple .env locations
        env_locations = [
            env_path,
            '.env',
            'config/local_secrets/.env',
            '../config/local_secrets/.env',
            'vald_api_pulls-main/forcedecks/.env',
            '../vald_api_pulls-main/forcedecks/.env',
        ]

        env_loaded = False
        for loc in env_locations:
            if loc and os.path.exists(loc):
                load_dotenv(loc)
                env_loaded = True
                print(f"Loaded config from: {loc}")
                break

        if not env_loaded:
            print("WARNING: No .env file found, using environment variables")

        return cls(
            CLIENT_ID=os.getenv('CLIENT_ID', ''),
            CLIENT_SECRET=os.getenv('CLIENT_SECRET', ''),
            TENANT_ID=os.getenv('TENANT_ID', ''),
            TOKEN_URL=os.getenv('TOKEN_URL', 'https://auth.prd.vald.com/oauth/token'),
            AUDIENCE=os.getenv('VALD_AUDIENCE', 'vald-api-external'),
            MANUAL_TOKEN=os.getenv('MANUAL_TOKEN'),
            REGION=os.getenv('VALD_REGION', 'euw'),
            ENABLE_EMAIL_NOTIFICATIONS=os.getenv('ENABLE_EMAIL', 'false').lower() == 'true',
            EMAIL_FROM=os.getenv('EMAIL_FROM'),
            EMAIL_TO=os.getenv('EMAIL_TO', '').split(',') if os.getenv('EMAIL_TO') else None,
            SMTP_SERVER=os.getenv('SMTP_SERVER'),
        )

    def validate(self) -> bool:
        """Validate required configuration"""
        required = ['CLIENT_ID', 'CLIENT_SECRET', 'TENANT_ID']
        missing = [f for f in required if not getattr(self, f)]

        if missing:
            raise ValueError(f"Missing required config: {', '.join(missing)}")

        return True

    def get_endpoint(self, service: str, version: Optional[str] = None) -> str:
        """Generate endpoint URL for a service"""
        endpoints = {
            'forcedecks': f'https://prd-{self.REGION}-api-extforcedecks.valdperformance.com/',
            'forceframe': f'https://prd-{self.REGION}-api-externalforceframe.valdperformance.com/',
            'nordbord': f'https://prd-{self.REGION}-api-externalnordbord.valdperformance.com/',
            'profiles': f'https://prd-{self.REGION}-api-externalprofile.valdperformance.com/',
            'athletes': f'https://prd-{self.REGION}-api-athlete.valdperformance.com/',  # Note: singular 'athlete'
            'tenants': f'https://prd-{self.REGION}-api-externaltenants.valdperformance.com/',
            'security': self.TOKEN_URL.rsplit('/', 1)[0] + '/'  # Auth0: https://auth.prd.vald.com/
        }

        base_url = endpoints.get(service)
        if not base_url:
            raise ValueError(f"Unknown service: {service}")

        if version:
            return f"{base_url}{version}/"

        return base_url

    def ensure_directories(self):
        """Create required directories if they don't exist"""
        dirs = [
            self.DATA_DIR,
            self.MASTER_DIR,
            self.CACHE_DIR,
            self.LOG_DIR,
            self.BACKUP_DIR,
        ]

        for d in dirs:
            os.makedirs(d, exist_ok=True)


# ============================================================================
# CENTRALIZED TOKEN MANAGEMENT (Auth0 - Feb 2026)
# ============================================================================
# All scripts and dashboard should use get_vald_token() instead of
# making their own token requests. Auth0 rate-limits token requests,
# so caching is mandatory.

_token_cache = {
    'access_token': None,
    'expires_at': 0,  # Unix timestamp
}


def get_vald_token(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    token_url: str = 'https://auth.prd.vald.com/oauth/token',
    audience: str = 'vald-api-external',
) -> str:
    """Get a cached OAuth token from VALD Auth0.

    Uses in-memory cache. Auth0 rate-limits token requests, so tokens
    are reused until 5 minutes before expiry.

    Args:
        client_id: VALD client ID. Falls back to CLIENT_ID env var.
        client_secret: VALD client secret. Falls back to CLIENT_SECRET env var.
        token_url: Auth0 token endpoint.
        audience: Auth0 audience parameter.

    Returns:
        Access token string.

    Raises:
        Exception: If token request fails.
    """
    global _token_cache

    # Return cached token if still valid (with 5 min buffer)
    if _token_cache['access_token'] and time.time() < _token_cache['expires_at'] - 300:
        return _token_cache['access_token']

    # Resolve credentials from env if not provided
    cid = client_id or os.getenv('CLIENT_ID', '') or os.getenv('VALD_CLIENT_ID', '')
    csecret = client_secret or os.getenv('CLIENT_SECRET', '') or os.getenv('VALD_CLIENT_SECRET', '')
    url = os.getenv('TOKEN_URL', token_url)
    aud = os.getenv('VALD_AUDIENCE', audience)

    if not cid or not csecret:
        raise ValueError("Missing VALD credentials (CLIENT_ID / CLIENT_SECRET)")

    response = requests.post(
        url,
        data={
            'grant_type': 'client_credentials',
            'client_id': cid,
            'client_secret': csecret,
            'audience': aud,
        },
        headers={'Content-Type': 'application/x-www-form-urlencoded'},
        timeout=30,
    )

    if response.status_code != 200:
        raise Exception(f"Auth0 token error {response.status_code}: {response.text}")

    token_data = response.json()
    access_token = token_data['access_token']
    expires_in = token_data.get('expires_in', 7200)

    # Cache the token
    _token_cache = {
        'access_token': access_token,
        'expires_at': time.time() + expires_in,
    }

    return access_token


# Device-specific settings
DEVICE_CONFIG = {
    'forcedecks': {
        'api_endpoint': 'forcedecks',
        'supports_trials': True,
        'trial_endpoint_template': '/v2019q3/teams/{tenant_id}/tests/{test_id}/trials',
        'pagination_type': 'cursor',  # cursor or page
        'cursor_field': 'modifiedFromUtc',
        'master_file': 'forcedecks_allsports_with_athletes.csv',
        'metrics': [
            'Jump Height (Flight Time) [cm]',
            'Peak Power / BM_Trial',
            'RSI-modified (Imp-Mom)_Trial',
            'Peak Force [N]',
            'Relative Peak Force [N/kg]',
        ]
    },
    'forceframe': {
        'api_endpoint': 'forceframe',
        'supports_trials': False,
        'pagination_type': 'cursor',
        'cursor_field': 'modifiedFromUtc',
        'master_file': 'forceframe_allsports_with_athletes.csv',
        'test_types': ['Pull', 'Squeeze'],
        'metrics': [
            'L Max Force (N)',
            'R Max Force (N)',
            'Max Imbalance',
            'L Max Ratio',
            'R Max Ratio',
        ]
    },
    'nordbord': {
        'api_endpoint': 'nordbord',
        'supports_trials': False,
        'pagination_type': 'page',
        'page_param': 'Page',
        'master_file': 'nordbord_allsports_with_athletes.csv',
        'metrics': [
            'L Max Force (N)',
            'R Max Force (N)',
            'Max Imbalance (%)',
            'L Max Torque (Nm)',
            'R Max Torque (Nm)',
        ]
    }
}


# Test type configurations
TEST_TYPE_METRICS = {
    'CMJ': [
        'Jump Height (Flight Time) [cm]',
        'Jump Height (Imp-Mom)_Trial',
        'Peak Power / BM_Trial',
        'RSI-modified (Imp-Mom)_Trial',
        'Peak Force [N]',
    ],
    'SJ': [
        'Jump Height (Flight Time) [cm]',
        'Peak Power / BM_Trial',
        'Concentric Peak Force / BM_Trial',
    ],
    'IMTP': [
        'Peak Vertical Force / BM_Trial',
        'Peak Force [N]',
        'RFD - 100ms_Trial',
        'RFD - 200ms_Trial',
    ],
    'DJ': [
        'Jump Height (Flight Time) [cm]',
        'RSI Modified',
        'Contact Time [ms]',
    ],
}


# Sport mappings for group extraction
SPORT_MAPPINGS = {
    'fencing': 'Fencing',
    'epee': 'Fencing - Epee',
    'sabre': 'Fencing - Sabre',
    'rowing': 'Rowing',
    'coastal': 'Rowing - Coastal',
    'athletics': 'Athletics',
    'horizontal jump': 'Athletics - Horizontal Jumps',
    'middle distance': 'Athletics - Middle Distance',
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


# Validation schemas
REQUIRED_FIELDS = {
    'test_data': [
        'testId',
        'profileId',
        'recordedDateUtc',
        'testType',
    ],
    'athlete_data': [
        'id',
        'givenName',
        'familyName',
    ],
    'enriched_data': [
        'testId',
        'profileId',
        'Name',
        'recordedDateUtc',
        'testType',
        'athlete_sport',
    ]
}


if __name__ == "__main__":
    # Test configuration loading
    config = ValdConfig.from_env()
    config.validate()
    config.ensure_directories()

    print("Configuration loaded successfully!")
    print(f"Region: {config.REGION}")
    print(f"Tenant ID: {config.TENANT_ID[:8]}...")
    print(f"Rate Limit: {config.RATE_LIMIT_CALLS} calls / {config.RATE_LIMIT_WINDOW}s")
    print(f"\nEndpoints:")
    for service in ['forcedecks', 'forceframe', 'nordbord', 'athletes']:
        print(f"  {service}: {config.get_endpoint(service)}")
