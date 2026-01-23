"""
Athlete Enrichment Script (v2)
Enriches ForceFrame and NordBord data using the VALD Profiles API as primary source.

Priority:
1. External Profiles API (has ALL 606+ athletes)
2. ForceDecks CSV as fallback (only has athletes who did ForceDecks tests)

ForceFrame/NordBord use 'athleteId' which equals 'profileId' in Profiles API.
"""

import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Add config directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'config'))

# Import centralized category mapping
try:
    from vald_categories import get_sport_from_groups, SKIP_GROUPS, ALL_SPORTS
    print("Using centralized category mapping from config/vald_categories.py")
except ImportError:
    print("WARNING: Could not import vald_categories, using inline fallback")
    SKIP_GROUPS = {'ARCHIVED', 'Staff', 'TBC', 'All Athletes'}
    ALL_SPORTS = ['Athletics', 'Fencing', 'Swimming', 'Taekwondo', 'Wrestling']
    def get_sport_from_groups(group_names):
        for name in group_names:
            if name and name not in SKIP_GROUPS:
                return name
        return 'Unknown'

# Try multiple .env locations (same as vald_config.py)
ENV_LOCATIONS = [
    os.path.join(SCRIPT_DIR, '.env'),
    os.path.join(SCRIPT_DIR, 'config', 'local_secrets', '.env'),
    os.path.join(SCRIPT_DIR, '..', 'config', 'local_secrets', '.env'),
    os.path.join(SCRIPT_DIR, 'vald_api_pulls-main', 'forcedecks', '.env'),
    os.path.join(SCRIPT_DIR, '..', 'vald_api_pulls-main', 'forcedecks', '.env'),
]

env_loaded = False
for loc in ENV_LOCATIONS:
    if os.path.exists(loc):
        load_dotenv(loc)
        print(f"Loaded credentials from: {loc}")
        env_loaded = True
        break

if not env_loaded:
    print("WARNING: No .env file found, using environment variables")

# Data paths
DATA_DIR = r"c:\Users\l.gallagher\OneDrive - Team Saudi\Documents\Performance Analysis\vald-data\data"

FORCEDECKS_FILE = os.path.join(DATA_DIR, "forcedecks_allsports_with_athletes.csv")
FORCEFRAME_FILE = os.path.join(DATA_DIR, "forceframe_allsports.csv")
NORDBORD_FILE = os.path.join(DATA_DIR, "nordbord_allsports.csv")
PROFILES_CACHE = os.path.join(SCRIPT_DIR, "profiles_cache.json")

# API Configuration
REGION = os.getenv('REGION', os.getenv('VALD_REGION', 'euw'))
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
TENANT_ID = os.getenv('TENANT_ID')


def get_oauth_token():
    """Get OAuth token from VALD security endpoint."""
    token_url = 'https://security.valdperformance.com/connect/token'

    response = requests.post(token_url, data={
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    })

    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        print(f"    Token request failed: {response.status_code}")
        return None


def fetch_groups_mapping(token):
    """Fetch group ID to name mapping from VALD Tenants API (Kenny's approach)."""
    url = f'https://prd-{REGION}-api-externaltenants.valdperformance.com/groups'
    headers = {'Authorization': f'Bearer {token}'}
    params = {'TenantId': TENANT_ID}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=60)
        if response.status_code == 200:
            data = response.json()
            groups = data.get('groups', data) if isinstance(data, dict) else data
            return {g['id']: g['name'] for g in groups if 'id' in g and 'name' in g}
    except Exception as e:
        print(f"    Groups fetch error: {e}")
    return {}


def fetch_individual_profile(token, profile_id):
    """Fetch individual profile to get groupIds (Kenny's approach).
    The bulk /profiles endpoint does NOT return groupIds.
    The individual /profiles/{id} endpoint DOES return groupIds.
    """
    url = f'https://prd-{REGION}-api-externalprofile.valdperformance.com/profiles/{profile_id}'
    headers = {'Authorization': f'Bearer {token}'}
    params = {'TenantId': TENANT_ID}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


# NOTE: derive_sport_from_groups is now imported from config/vald_categories.py
# as get_sport_from_groups() - provides centralized mapping


def fetch_profiles_from_api():
    """Fetch all athlete profiles from VALD External Profiles API."""
    print("\n[1] Fetching profiles from VALD Profiles API...")

    if not all([CLIENT_ID, CLIENT_SECRET, TENANT_ID]):
        print("    WARNING: Missing API credentials in .env file")
        return None

    # Get OAuth token
    token = get_oauth_token()
    if not token:
        print("    Failed to get OAuth token")
        return None

    # Fetch profiles
    profiles_url = f'https://prd-{REGION}-api-externalprofile.valdperformance.com/profiles'
    headers = {'Authorization': f'Bearer {token}'}
    params = {'TenantId': TENANT_ID}

    response = requests.get(profiles_url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        profiles = data.get('profiles', [])
        print(f"    Fetched {len(profiles)} profiles from API")

        # Cache profiles for future use
        with open(PROFILES_CACHE, 'w') as f:
            json.dump(profiles, f, indent=2)
        print(f"    Cached profiles to: {PROFILES_CACHE}")

        return profiles
    else:
        print(f"    API request failed: {response.status_code}")
        return None


def load_cached_profiles():
    """Load profiles from cache file if available."""
    if os.path.exists(PROFILES_CACHE):
        with open(PROFILES_CACHE, 'r') as f:
            profiles = json.load(f)
        print(f"    Loaded {len(profiles)} profiles from cache")
        return profiles
    return None


def build_athlete_lookup(profiles=None, forcedecks_df=None, token=None, groups_map=None):
    """Build athlete lookup dictionary from profiles and/or ForceDecks data.

    Uses Kenny's approach: fetch individual profiles to get groupIds,
    then map groups to sports.
    """
    import time
    athlete_lookup = {}

    # Priority 1: Use Profiles API data with group enrichment
    if profiles:
        print(f"    Processing {len(profiles)} profiles...")

        # First pass: build basic lookup
        for profile in profiles:
            pid = str(profile.get('profileId', ''))
            if pid:
                given_name = profile.get('givenName', '') or ''
                family_name = profile.get('familyName', '') or ''
                full_name = f"{given_name} {family_name}".strip()

                athlete_lookup[pid] = {
                    'full_name': full_name,
                    'givenName': given_name,
                    'familyName': family_name,
                    'dateOfBirth': profile.get('dateOfBirth', ''),
                    'externalId': profile.get('externalId', ''),
                    'athlete_sport': 'Unknown',
                    'groups': [],
                    'athlete_sex': '',
                    'athlete_weight_kg': None,
                    'athlete_height_cm': None,
                }

        print(f"    Built lookup with {len(athlete_lookup)} profiles from API")

        # Second pass: fetch individual profiles to get groupIds (Kenny's approach)
        if token and groups_map:
            print(f"    Fetching individual profiles for group membership (Kenny's approach)...")
            enriched = 0
            for i, pid in enumerate(athlete_lookup.keys()):
                if (i + 1) % 50 == 0:
                    print(f"      Progress: {i + 1}/{len(athlete_lookup)}")

                # Rate limiting
                time.sleep(0.1)

                individual_profile = fetch_individual_profile(token, pid)
                if individual_profile:
                    group_ids = individual_profile.get('groupIds', [])
                    if group_ids:
                        group_names = [groups_map.get(gid, 'Unknown') for gid in group_ids]
                        sport = get_sport_from_groups(group_names)
                        athlete_lookup[pid]['groups'] = group_names
                        athlete_lookup[pid]['athlete_sport'] = sport
                        enriched += 1

            print(f"    Enriched {enriched} profiles with group/sport info")

    # Priority 2: Enrich/fallback with ForceDecks data (has sport, sex, biometrics)
    if forcedecks_df is not None and len(forcedecks_df) > 0:
        athlete_cols = ['profileId', 'full_name', 'athlete_sport', 'athlete_sex',
                        'athlete_weight_kg', 'athlete_height_cm']
        available_cols = [c for c in athlete_cols if c in forcedecks_df.columns]

        if 'profileId' in available_cols:
            fd_athletes = forcedecks_df[available_cols].drop_duplicates(subset=['profileId'])
            fd_athletes = fd_athletes.dropna(subset=['profileId'])

            enriched_count = 0
            added_count = 0

            for _, row in fd_athletes.iterrows():
                pid = str(row['profileId'])

                if pid in athlete_lookup:
                    # Enrich existing profile with ForceDecks data
                    # Only use ForceDecks sport if API didn't provide one (Unknown)
                    existing_sport = athlete_lookup[pid].get('athlete_sport', 'Unknown')
                    fd_sport = row.get('athlete_sport', '')
                    if existing_sport == 'Unknown' and fd_sport and fd_sport != 'Unknown':
                        athlete_lookup[pid]['athlete_sport'] = fd_sport
                    if row.get('athlete_sex'):
                        athlete_lookup[pid]['athlete_sex'] = row.get('athlete_sex', '')
                    if pd.notna(row.get('athlete_weight_kg')):
                        athlete_lookup[pid]['athlete_weight_kg'] = row.get('athlete_weight_kg')
                    if pd.notna(row.get('athlete_height_cm')):
                        athlete_lookup[pid]['athlete_height_cm'] = row.get('athlete_height_cm')
                    enriched_count += 1
                else:
                    # Add athlete not in Profiles API
                    athlete_lookup[pid] = {
                        'full_name': row.get('full_name', ''),
                        'athlete_sport': row.get('athlete_sport', ''),
                        'athlete_sex': row.get('athlete_sex', ''),
                        'athlete_weight_kg': row.get('athlete_weight_kg'),
                        'athlete_height_cm': row.get('athlete_height_cm'),
                    }
                    added_count += 1

            print(f"    Enriched {enriched_count} profiles with ForceDecks data")
            if added_count > 0:
                print(f"    Added {added_count} athletes from ForceDecks (not in API)")

    return athlete_lookup


def enrich_dataframe(df, athlete_lookup, device_name):
    """Add athlete columns to a dataframe."""
    if 'athleteId' not in df.columns:
        print(f"    WARNING: No 'athleteId' column in {device_name} data")
        return df

    def get_athlete_info(athlete_id, field, fallback=''):
        match = athlete_lookup.get(str(athlete_id), {})
        value = match.get(field, fallback)
        if pd.isna(value) or value == '':
            if field == 'full_name':
                # Fallback to shortened athleteId
                return f"Athlete_{str(athlete_id)[:8]}" if pd.notna(athlete_id) else "Unknown"
            return fallback
        return value

    df['full_name'] = df['athleteId'].apply(lambda x: get_athlete_info(x, 'full_name'))
    df['athlete_sport'] = df['athleteId'].apply(lambda x: get_athlete_info(x, 'athlete_sport', 'Unknown'))
    df['athlete_sex'] = df['athleteId'].apply(lambda x: get_athlete_info(x, 'athlete_sex', ''))

    # Count real matches (not fallback names)
    real_names = df['full_name'].apply(lambda x: not str(x).startswith('Athlete_') and x != 'Unknown')
    matched = real_names.sum()
    print(f"    Matched {matched}/{len(df)} records with real athlete names")

    return df


def main():
    print("=" * 70)
    print("ATHLETE ENRICHMENT (v3 - Kenny's Approach)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Step 0: Get OAuth token (needed for group/profile enrichment)
    print("\n[0] Getting OAuth token...")
    token = get_oauth_token()
    if not token:
        print("    WARNING: Failed to get token - will skip group enrichment")

    # Step 0b: Fetch groups mapping (Kenny's approach)
    groups_map = {}
    if token:
        print("\n[0b] Fetching groups mapping (Kenny's approach)...")
        groups_map = fetch_groups_mapping(token)
        print(f"    Found {len(groups_map)} groups")
        for gid, gname in groups_map.items():
            print(f"      - {gname}")

    # Step 1: Try to fetch profiles from API
    profiles = fetch_profiles_from_api()

    # Fallback to cached profiles if API fails
    if profiles is None:
        print("\n    Trying cached profiles...")
        profiles = load_cached_profiles()

    # Step 2: Load ForceDecks data for additional athlete info (sport, sex, biometrics)
    print("\n[2] Loading ForceDecks data for additional athlete info...")
    forcedecks_df = None
    if os.path.exists(FORCEDECKS_FILE):
        forcedecks_df = pd.read_csv(FORCEDECKS_FILE)
        print(f"    Loaded {len(forcedecks_df)} ForceDecks records")
    else:
        print(f"    ForceDecks file not found (optional)")

    # Step 3: Build combined athlete lookup with group enrichment
    print("\n[3] Building athlete lookup with group enrichment (Kenny's approach)...")
    if profiles is None and forcedecks_df is None:
        print("ERROR: No athlete data available (API failed and no ForceDecks file)")
        return

    athlete_lookup = build_athlete_lookup(profiles, forcedecks_df, token=token, groups_map=groups_map)
    print(f"    Total athletes in lookup: {len(athlete_lookup)}")

    # Print sport distribution from lookup
    sports_count = {}
    for pid, info in athlete_lookup.items():
        sport = info.get('athlete_sport', 'Unknown')
        sports_count[sport] = sports_count.get(sport, 0) + 1
    print("\n    Sport distribution from profiles:")
    for sport, count in sorted(sports_count.items(), key=lambda x: -x[1]):
        print(f"      {sport}: {count}")

    # Step 4: Process ForceDecks (add names to ForceDecks itself)
    print("\n[4] Processing ForceDecks...")
    if forcedecks_df is not None and not forcedecks_df.empty:
        # Add full_name column using profileId
        if 'profileId' in forcedecks_df.columns:
            forcedecks_df['full_name'] = forcedecks_df['profileId'].apply(
                lambda x: athlete_lookup.get(str(x), {}).get('full_name', f"Athlete_{str(x)[:8]}")
            )
            forcedecks_df['athlete_sport'] = forcedecks_df['profileId'].apply(
                lambda x: athlete_lookup.get(str(x), {}).get('athlete_sport', 'Unknown')
            )
            matched = forcedecks_df['full_name'].apply(lambda x: not str(x).startswith('Athlete_')).sum()
            print(f"    Matched {matched}/{len(forcedecks_df)} records with real athlete names")

            # Save enriched ForceDecks file
            output_file = FORCEDECKS_FILE.replace('.csv', '_enriched.csv')
            forcedecks_df.to_csv(output_file, index=False)
            print(f"    Saved: {output_file}")

            # Also save to vald-data repo location
            vald_data_output = FORCEDECKS_FILE.replace('Vald/data/master_files', 'vald-data/data')
            if os.path.exists(os.path.dirname(vald_data_output)):
                forcedecks_df.to_csv(vald_data_output, index=False)
                print(f"    Saved: {vald_data_output}")
    else:
        print("    SKIP: No ForceDecks data loaded")

    # Step 5: Process ForceFrame
    print("\n[5] Processing ForceFrame...")
    if os.path.exists(FORCEFRAME_FILE):
        ff_df = pd.read_csv(FORCEFRAME_FILE)
        print(f"    Loaded {len(ff_df)} ForceFrame records")

        ff_df = enrich_dataframe(ff_df, athlete_lookup, "ForceFrame")

        # Save enriched file
        output_file = FORCEFRAME_FILE.replace('.csv', '_with_athletes.csv')
        ff_df.to_csv(output_file, index=False)
        print(f"    Saved: {output_file}")
    else:
        print(f"    SKIP: ForceFrame file not found")

    # Step 6: Process NordBord
    print("\n[6] Processing NordBord...")
    if os.path.exists(NORDBORD_FILE):
        nb_df = pd.read_csv(NORDBORD_FILE)
        print(f"    Loaded {len(nb_df)} NordBord records")

        nb_df = enrich_dataframe(nb_df, athlete_lookup, "NordBord")

        # Save enriched file
        output_file = NORDBORD_FILE.replace('.csv', '_with_athletes.csv')
        nb_df.to_csv(output_file, index=False)
        print(f"    Saved: {output_file}")
    else:
        print(f"    SKIP: NordBord file not found")

    print("\n" + "=" * 70)
    print("ENRICHMENT COMPLETE!")
    print("=" * 70)
    print("\nNext: Restart the dashboard to load enriched files")
    print("      streamlit run dashboard/world_class_vald_dashboard.py")


if __name__ == "__main__":
    main()
