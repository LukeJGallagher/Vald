"""
Local sync script to fetch fresh VALD data and save directly to vald-data location.

Usage:
    python local_sync.py              # Incremental sync (new data only)
    python local_sync.py --force-all  # Full sync from 2020
    python local_sync.py --days 7     # Sync last 7 days
"""
import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import json
import argparse
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment from local secrets
from dotenv import load_dotenv
env_path = Path(r"c:\Users\l.gallagher\OneDrive - Team Saudi\Documents\Performance Analysis\Vald\config\local_secrets\.env")
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded credentials from: {env_path}")
else:
    print(f"WARNING: .env file not found at {env_path}")

# Sync state file - tracks last sync date
SYNC_STATE_FILE = Path(__file__).parent.parent / 'config' / 'sync_state.json'

# Output directory - vald-data repo location
OUTPUT_DIR = Path(r"c:\Users\l.gallagher\OneDrive - Team Saudi\Documents\Performance Analysis\vald-data\data")

# Import centralized sport mapping from config
from config.vald_categories import GROUP_TO_CATEGORY, SKIP_GROUPS


def load_sync_state():
    """Load last sync state from file."""
    if SYNC_STATE_FILE.exists():
        try:
            with open(SYNC_STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {'last_sync': None, 'devices': {}}


def save_sync_state(state):
    """Save sync state to file."""
    SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNC_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def get_sync_from_date(force_all=False, days=None):
    """Get the date to sync from based on options."""
    if force_all:
        return '2020-01-01T00:00:00.000Z'

    if days:
        from_date = datetime.utcnow() - timedelta(days=days)
        return from_date.strftime('%Y-%m-%dT00:00:00.000Z')

    # Incremental: use last sync date or default to 30 days ago
    state = load_sync_state()
    if state.get('last_sync'):
        return state['last_sync']

    # Default: last 30 days for first run
    from_date = datetime.utcnow() - timedelta(days=30)
    return from_date.strftime('%Y-%m-%dT00:00:00.000Z')


# Import centralized function
from config.vald_categories import get_sport_from_groups


def get_token():
    """Get OAuth token."""
    client_id = os.environ.get('CLIENT_ID', '') or os.environ.get('VALD_CLIENT_ID', '')
    client_secret = os.environ.get('CLIENT_SECRET', '') or os.environ.get('VALD_CLIENT_SECRET', '')

    response = requests.post(
        'https://security.valdperformance.com/connect/token',
        data={
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret
        },
        timeout=30
    )
    if response.status_code != 200:
        raise Exception(f"Token error: {response.status_code} - {response.text}")
    return response.json()['access_token']


def fetch_groups(token, region, tenant_id):
    """Fetch groups from VALD API."""
    url = f'https://prd-{region}-api-externaltenants.valdperformance.com/groups'
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(url, headers=headers, params={'TenantId': tenant_id}, timeout=60)

    if response.status_code == 200:
        data = response.json()
        groups = data.get('groups', data) if isinstance(data, dict) else data
        return {g['id']: g['name'] for g in groups if 'id' in g and 'name' in g}
    print(f"    Groups API error: {response.status_code} - {response.text[:200]}")
    return {}


def fetch_profiles(token, region, tenant_id):
    """Fetch athlete profiles."""
    url = f'https://prd-{region}-api-externalprofile.valdperformance.com/profiles'
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(url, headers=headers, params={'TenantId': tenant_id}, timeout=60)

    if response.status_code == 200:
        data = response.json()
        profiles = data.get('profiles', data) if isinstance(data, dict) else data
        result = {}
        for p in profiles:
            pid = p.get('profileId') or p.get('id')
            given = p.get('givenName', '') or ''
            family = p.get('familyName', '') or ''
            full_name = f"{given} {family}".strip() or 'Unknown'
            sex = p.get('sex') or p.get('gender') or ''
            result[pid] = {'full_name': full_name, 'athlete_sport': 'Unknown', 'athlete_sex': sex, 'groupIds': []}
        return result, profiles
    print(f"    Profiles API error: {response.status_code} - {response.text[:200]}")
    return {}, []


def fetch_individual_profile(token, region, tenant_id, profile_id):
    """Fetch individual profile to get groupIds."""
    url = f'https://prd-{region}-api-externalprofile.valdperformance.com/profiles/{profile_id}'
    headers = {'Authorization': f'Bearer {token}'}

    try:
        time.sleep(0.1)  # Rate limiting
        response = requests.get(url, headers=headers, params={'TenantId': tenant_id}, timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    return None


def enrich_profiles_with_groups(profiles_dict, raw_profiles, groups_map, token, region, tenant_id):
    """Enrich profiles with sport from groups (Kenny's approach)."""
    profiles_needing_fetch = []

    # First pass: check which profiles need individual fetch
    for profile in raw_profiles:
        pid = profile.get('profileId') or profile.get('id')
        if not pid or pid not in profiles_dict:
            continue
        group_ids = profile.get('groupIds', [])
        if group_ids:
            group_names = [groups_map.get(gid, 'Unknown') for gid in group_ids]
            profiles_dict[pid]['athlete_sport'] = get_sport_from_groups(group_names)
            profiles_dict[pid]['groups'] = '|'.join(group_names)
        else:
            profiles_needing_fetch.append(pid)

    print(f"    Need to fetch {len(profiles_needing_fetch)} individual profiles for groups...")

    # Second pass: fetch individual profiles
    for i, pid in enumerate(profiles_needing_fetch):
        if (i + 1) % 50 == 0:
            print(f"      Progress: {i + 1}/{len(profiles_needing_fetch)}")

        profile_data = fetch_individual_profile(token, region, tenant_id, pid)
        if profile_data:
            group_ids = profile_data.get('groupIds', [])
            if group_ids:
                group_names = [groups_map.get(gid, 'Unknown') for gid in group_ids]
                profiles_dict[pid]['athlete_sport'] = get_sport_from_groups(group_names)
                profiles_dict[pid]['groups'] = '|'.join(group_names)

    return profiles_dict


def fetch_trial_data(token, region, tenant_id, test_id, device='forcedecks'):
    """Fetch detailed trial data for a single test."""
    device_urls = {
        'forcedecks': f'https://prd-{region}-api-extforcedecks.valdperformance.com',
        'forceframe': f'https://prd-{region}-api-externalforceframe.valdperformance.com',
        'nordbord': f'https://prd-{region}-api-externalnordbord.valdperformance.com'
    }

    base_url = device_urls.get(device, device_urls['forcedecks'])
    url = f'{base_url}/v2019q3/teams/{tenant_id}/tests/{test_id}/trials'
    headers = {'Authorization': f'Bearer {token}'}

    try:
        time.sleep(0.15)  # Rate limiting
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            trials = data.get('trials', data) if isinstance(data, dict) else data
            return trials if isinstance(trials, list) else []
    except Exception as e:
        pass
    return []


def flatten_trial_metrics(trials):
    """Flatten trial metrics into a single dict of aggregated values.

    VALD trial data format:
    [{'id': '...', 'results': [{'resultId': 123, 'value': 0.5, 'definition': {'result': 'JUMP_HEIGHT'}}]}]
    """
    if not trials:
        return {}

    # Collect all metrics from all trials
    all_metrics = {}

    for trial in trials:
        # Get results from trial - VALD uses a list of result objects
        results = trial.get('results', [])

        if isinstance(results, list):
            # VALD format: list of {'value': X, 'definition': {'result': 'METRIC_NAME'}}
            for result in results:
                if not isinstance(result, dict):
                    continue

                value = result.get('value')
                if not isinstance(value, (int, float)) or value is None:
                    continue

                # Get metric name from definition
                definition = result.get('definition', {})
                metric_name = definition.get('result', '') if isinstance(definition, dict) else ''

                if not metric_name:
                    # Fallback to resultId
                    metric_name = f"metric_{result.get('resultId', 'unknown')}"

                # Track limb if available
                limb = result.get('limb', 'Trial')
                if limb and limb != 'Trial':
                    metric_key = f"{metric_name}_{limb}"
                else:
                    metric_key = metric_name

                if metric_key not in all_metrics:
                    all_metrics[metric_key] = []
                all_metrics[metric_key].append(value)

        elif isinstance(results, dict):
            # Older format: dict of metric_name: value
            for key, value in results.items():
                if isinstance(value, (int, float)) and value is not None:
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

    # Average the metrics (or take best for performance metrics)
    flattened = {}
    for key, values in all_metrics.items():
        if values:
            key_lower = key.lower()
            # For jump height/power/velocity, use max (best performance)
            if any(x in key_lower for x in ['height', 'power', 'velocity', 'peak', 'max']):
                flattened[key] = max(values)
            else:
                flattened[key] = sum(values) / len(values)

    return flattened


def fetch_forcedecks(token, region, tenant_id, from_date='2020-01-01T00:00:00.000Z', fetch_trials=True):
    """Fetch ForceDecks tests with optional trial data."""
    url = f'https://prd-{region}-api-extforcedecks.valdperformance.com/tests'
    headers = {'Authorization': f'Bearer {token}'}

    all_tests = []
    modified_from = from_date

    while True:
        time.sleep(0.5)
        params = {'tenantId': tenant_id, 'modifiedFromUtc': modified_from}
        response = requests.get(url, headers=headers, params=params, timeout=120)

        if response.status_code == 204:
            break
        if response.status_code != 200:
            print(f"    API error: {response.status_code}")
            break

        data = response.json()
        tests = data.get('tests', data) if isinstance(data, dict) else data

        if not tests or not isinstance(tests, list):
            break

        all_tests.extend(tests)
        print(f"    Fetched {len(tests)} tests (total: {len(all_tests)})")

        if len(tests) < 50:
            break

        last_modified = tests[-1].get('modifiedDateUtc')
        if not last_modified or last_modified == modified_from:
            break
        modified_from = last_modified

    # Fetch trial data for detailed metrics
    if fetch_trials and all_tests:
        print(f"\n    Fetching trial data for {len(all_tests)} tests...")
        for i, test in enumerate(all_tests):
            test_id = test.get('id') or test.get('testId')
            if test_id:
                trials = fetch_trial_data(token, region, tenant_id, test_id, 'forcedecks')
                if trials:
                    metrics = flatten_trial_metrics(trials)
                    test.update(metrics)

            if (i + 1) % 100 == 0:
                print(f"      Trial progress: {i + 1}/{len(all_tests)}")

        print(f"    Trial fetching complete!")

    return all_tests


def fetch_forceframe(token, region, tenant_id, from_date='2020-01-01T00:00:00.000Z'):
    """Fetch ForceFrame tests (requires all date parameters).

    IMPORTANT: ForceFrame API only allows 6-month date ranges.
    We iterate in 6-month chunks to fetch all historical data.
    """
    url = f'https://prd-{region}-api-externalforceframe.valdperformance.com/tests'
    headers = {'Authorization': f'Bearer {token}'}

    all_tests = []

    # Parse the from_date and iterate in 6-month chunks
    start = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
    end = datetime.now(timezone.utc)
    chunk_months = 5  # Use 5 months to stay safely under 6-month limit

    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=chunk_months * 30), end)

        time.sleep(0.5)
        # ForceFrame requires TenantId, TestFromUtc, TestToUtc, AND ModifiedFromUtc
        params = {
            'TenantId': tenant_id,
            'TestFromUtc': current_start.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'TestToUtc': current_end.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'ModifiedFromUtc': current_start.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        }
        response = requests.get(url, headers=headers, params=params, timeout=120)

        if response.status_code == 204:
            # No data for this period, move to next chunk
            current_start = current_end
            continue
        if response.status_code != 200:
            print(f"    API error: {response.status_code} - {response.text[:200]}")
            current_start = current_end
            continue

        data = response.json()
        tests = data.get('tests', data) if isinstance(data, dict) else data

        if tests and isinstance(tests, list):
            all_tests.extend(tests)
            print(f"    Fetched {len(tests)} tests from {current_start.strftime('%Y-%m')} to {current_end.strftime('%Y-%m')} (total: {len(all_tests)})")

        current_start = current_end

    if not all_tests:
        print("    No ForceFrame data found")

    return all_tests


def fetch_nordbord(token, region, tenant_id, from_date='2020-01-01T00:00:00.000Z'):
    """Fetch NordBord tests (requires all date parameters).

    IMPORTANT: NordBord API only allows 6-month date ranges.
    We iterate in 6-month chunks to fetch all historical data.
    """
    url = f'https://prd-{region}-api-externalnordbord.valdperformance.com/tests'
    headers = {'Authorization': f'Bearer {token}'}

    all_tests = []

    # Parse the from_date and iterate in 6-month chunks
    start = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
    end = datetime.now(timezone.utc)
    chunk_months = 5  # Use 5 months to stay safely under 6-month limit

    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=chunk_months * 30), end)

        time.sleep(0.5)
        # NordBord requires TenantId, TestFromUtc, TestToUtc, AND ModifiedFromUtc
        params = {
            'TenantId': tenant_id,
            'TestFromUtc': current_start.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'TestToUtc': current_end.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'ModifiedFromUtc': current_start.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        }
        response = requests.get(url, headers=headers, params=params, timeout=120)

        if response.status_code == 204:
            # No data for this period, move to next chunk
            current_start = current_end
            continue
        if response.status_code != 200:
            print(f"    API error: {response.status_code} - {response.text[:200]}")
            current_start = current_end
            continue

        data = response.json()
        tests = data.get('tests', data) if isinstance(data, dict) else data

        if tests and isinstance(tests, list):
            all_tests.extend(tests)
            print(f"    Fetched {len(tests)} tests from {current_start.strftime('%Y-%m')} to {current_end.strftime('%Y-%m')} (total: {len(all_tests)})")

        current_start = current_end

    if not all_tests:
        print("    No NordBord data found")

    return all_tests


def merge_with_existing(new_df, output_file, id_column='id'):
    """Merge new data with existing CSV file, removing duplicates."""
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file)
            print(f"    Existing data: {len(existing_df)} rows")

            # Find a valid ID column
            if id_column not in new_df.columns:
                for col in ['id', 'testId', 'athleteId', 'profileId']:
                    if col in new_df.columns:
                        id_column = col
                        break

            if id_column in new_df.columns and id_column in existing_df.columns:
                # Combine and remove duplicates, keeping the latest
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=[id_column], keep='last')
                print(f"    After merge: {len(combined)} rows (added {len(combined) - len(existing_df)} new)")
                return combined
            else:
                # No ID column - just append
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                return combined
        except Exception as e:
            print(f"    Could not load existing file: {e}")

    return new_df


def fetch_dynamo(token, region, tenant_id, from_date='2020-01-01T00:00:00.000Z'):
    """Fetch DynaMo (grip strength/strength testing) data.

    Uses the v2022q2 API endpoint with pagination.
    Flattens repetitionTypeSummaries into columns for easier analysis.
    """
    # Correct endpoint: extdynamo (not externaldynamo)
    url = f'https://prd-{region}-api-extdynamo.valdperformance.com/v2022q2/teams/{tenant_id}/tests'
    headers = {'Authorization': f'Bearer {token}'}

    all_tests = []
    page = 1

    try:
        while True:
            time.sleep(0.5)
            params = {
                'testFromUTC': from_date,
                'testToUTC': '2030-12-31T23:59:59Z',
                'includeRepSummaries': 'true',
                'page': page
            }
            response = requests.get(url, headers=headers, params=params, timeout=30)

            if response.status_code == 204:
                break
            if response.status_code != 200:
                print(f"    API error: {response.status_code} - {response.text[:200]}")
                break

            data = response.json()
            items = data.get('items', [])
            total_pages = data.get('totalPages', 1)
            total_items = data.get('totalItems', 0)

            if not items:
                break

            # Flatten each test - extract key metrics from repetitionTypeSummaries
            for test in items:
                flat_test = {
                    'id': test.get('id'),
                    'athleteId': test.get('athleteId'),
                    'teamId': test.get('teamId'),
                    'testCategory': test.get('testCategory'),
                    'bodyRegion': test.get('bodyRegion'),
                    'movement': test.get('movement'),
                    'position': test.get('position'),
                    'laterality': test.get('laterality'),
                    'startTimeUTC': test.get('startTimeUTC'),
                    'durationSeconds': test.get('durationSeconds'),
                    'analysedDateUTC': test.get('analysedDateUTC'),
                }

                # Extract metrics from first rep summary (primary metric)
                rep_summaries = test.get('repetitionTypeSummaries', [])
                if rep_summaries:
                    rep = rep_summaries[0]
                    flat_test['repCount'] = rep.get('repCount')
                    flat_test['maxForceNewtons'] = rep.get('maxForceNewtons')
                    flat_test['avgForceNewtons'] = rep.get('avgForceNewtons')
                    flat_test['maxImpulseNewtonSeconds'] = rep.get('maxImpulseNewtonSeconds')
                    flat_test['avgImpulseNewtonSeconds'] = rep.get('avgImpulseNewtonSeconds')
                    flat_test['maxRFD'] = rep.get('maxRateOfForceDevelopmentNewtonsPerSecond')
                    flat_test['avgRFD'] = rep.get('avgRateOfForceDevelopmentNewtonsPerSecond')
                    flat_test['maxROM'] = rep.get('maxRangeOfMotionDegrees')
                    flat_test['avgROM'] = rep.get('avgRangeOfMotionDegrees')

                all_tests.append(flat_test)

            print(f"    Page {page}/{total_pages}: {len(items)} tests (total: {len(all_tests)}/{total_items})")

            if page >= total_pages:
                break
            page += 1

    except requests.exceptions.ConnectionError:
        print("    DynaMo API not available (connection failed)")
    except Exception as e:
        print(f"    DynaMo fetch error: {e}")

    return all_tests


def main():
    """Main sync execution."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sync VALD data locally')
    parser.add_argument('--force-all', action='store_true', help='Fetch all data from 2020 (ignore last sync)')
    parser.add_argument('--days', type=int, help='Fetch data from last N days')
    args = parser.parse_args()

    # Determine sync from date
    from_date = get_sync_from_date(force_all=args.force_all, days=args.days)

    print("=" * 70)
    print("LOCAL VALD DATA SYNC")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if args.force_all:
        print("\n    Mode: FULL SYNC (fetching all data from 2020)")
    elif args.days:
        print(f"\n    Mode: LAST {args.days} DAYS")
    else:
        print("\n    Mode: INCREMENTAL (new data only)")
    print(f"    From date: {from_date}")

    region = os.environ.get('VALD_REGION', 'euw')
    tenant_id = os.environ.get('TENANT_ID', '') or os.environ.get('VALD_TENANT_ID', '')

    print(f"\n    Region: {region}")
    print(f"    Tenant ID: {tenant_id}")

    print(f"\n[1] Getting OAuth token...")
    token = get_token()
    print("    Token obtained!")

    print(f"\n[2] Fetching groups...")
    groups_map = fetch_groups(token, region, tenant_id)
    print(f"    Found {len(groups_map)} groups")

    # Show group names for debugging
    swimming_groups = [name for name in groups_map.values() if 'swim' in name.lower()]
    print(f"    Swimming-related groups: {swimming_groups}")

    print(f"\n[3] Fetching profiles...")
    profiles, raw_profiles = fetch_profiles(token, region, tenant_id)
    print(f"    Found {len(profiles)} profiles")

    print(f"\n[4] Enriching profiles with groups (Kenny's approach)...")
    profiles = enrich_profiles_with_groups(profiles, raw_profiles, groups_map, token, region, tenant_id)

    # Count sports
    sport_counts = {}
    for p in profiles.values():
        sport = p.get('athlete_sport', 'Unknown')
        sport_counts[sport] = sport_counts.get(sport, 0) + 1
    print(f"\n    Sport distribution:")
    for sport, count in sorted(sport_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"      {sport}: {count}")

    # Find Swimming athletes
    swimming_athletes = [(pid, p['full_name']) for pid, p in profiles.items() if p.get('athlete_sport') == 'Swimming']
    print(f"\n    Swimming athletes ({len(swimming_athletes)}):")
    for pid, name in swimming_athletes:
        print(f"      - {name} ({pid})")

    print(f"\n[5] Fetching ForceDecks data...")
    forcedecks_tests = fetch_forcedecks(token, region, tenant_id, from_date=from_date)
    print(f"    Total ForceDecks tests: {len(forcedecks_tests)}")

    # Count unique athletes
    fd_athletes = set()
    for t in forcedecks_tests:
        pid = t.get('athleteId') or t.get('profileId')
        if pid:
            fd_athletes.add(pid)
    print(f"    Unique athletes with tests: {len(fd_athletes)}")

    # Check for Swimming athletes' tests
    swimming_test_count = 0
    for t in forcedecks_tests:
        pid = t.get('athleteId') or t.get('profileId')
        if pid and profiles.get(pid, {}).get('athlete_sport') == 'Swimming':
            swimming_test_count += 1
    print(f"    Swimming tests found: {swimming_test_count}")

    if forcedecks_tests:
        df = pd.DataFrame(forcedecks_tests)
        id_col = 'athleteId' if 'athleteId' in df.columns else 'profileId'

        output_file = OUTPUT_DIR / 'forcedecks_allsports_with_athletes.csv'
        # Merge with existing data instead of overwriting
        df = merge_with_existing(df, output_file, id_column='id')

        # Re-enrich ALL rows after merge (fixes "Unknown" sport for old data)
        df['full_name'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('full_name', f'Athlete_{str(pid)[:8]}'))
        df['athlete_sport'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('athlete_sport', 'Unknown'))
        df['athlete_sex'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('athlete_sex', ''))
        df['Name'] = df['full_name']

        df.to_csv(output_file, index=False)
        print(f"    Saved: {output_file}")

        # Sport distribution in tests
        if 'athlete_sport' in df.columns:
            test_sports = df['athlete_sport'].value_counts()
            print(f"\n    Test count by sport:")
            for sport, count in test_sports.head(15).items():
                print(f"      {sport}: {count}")

    # --- ForceFrame, NordBord, DynaMo: wrap each in try/except so one failure doesn't block others ---
    # Refresh token before each device since ForceDecks trial fetch can take 40+ min

    forceframe_tests = []
    print(f"\n[6] Fetching ForceFrame data...")
    try:
        print("    Refreshing token...")
        token = get_token()
        forceframe_tests = fetch_forceframe(token, region, tenant_id, from_date=from_date)
        print(f"    Total ForceFrame tests: {len(forceframe_tests)}")
    except Exception as e:
        print(f"    ForceFrame fetch FAILED (continuing): {e}")

    if forceframe_tests:
        df = pd.DataFrame(forceframe_tests)
        id_col = 'athleteId' if 'athleteId' in df.columns else 'profileId'

        output_file = OUTPUT_DIR / 'forceframe_allsports_with_athletes.csv'
        df = merge_with_existing(df, output_file, id_column='id')

        # Re-enrich ALL rows after merge (fixes "Unknown" sport for old data)
        df['full_name'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('full_name', f'Athlete_{str(pid)[:8]}'))
        df['athlete_sport'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('athlete_sport', 'Unknown'))
        df['athlete_sex'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('athlete_sex', ''))
        df['Name'] = df['full_name']

        df.to_csv(output_file, index=False)
        print(f"    Saved: {output_file}")

    nordbord_tests = []
    print(f"\n[7] Fetching NordBord data...")
    try:
        print("    Refreshing token...")
        token = get_token()
        nordbord_tests = fetch_nordbord(token, region, tenant_id, from_date=from_date)
        print(f"    Total NordBord tests: {len(nordbord_tests)}")
    except Exception as e:
        print(f"    NordBord fetch FAILED (continuing): {e}")

    if nordbord_tests:
        df = pd.DataFrame(nordbord_tests)
        id_col = 'athleteId' if 'athleteId' in df.columns else 'profileId'

        output_file = OUTPUT_DIR / 'nordbord_allsports_with_athletes.csv'
        df = merge_with_existing(df, output_file, id_column='id')

        # Re-enrich ALL rows after merge (fixes "Unknown" sport for old data)
        df['full_name'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('full_name', f'Athlete_{str(pid)[:8]}'))
        df['athlete_sport'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('athlete_sport', 'Unknown'))
        df['athlete_sex'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('athlete_sex', ''))
        df['Name'] = df['full_name']

        df.to_csv(output_file, index=False)
        print(f"    Saved: {output_file}")

    dynamo_tests = []
    print(f"\n[8] Fetching DynaMo data...")
    try:
        print("    Refreshing token...")
        token = get_token()
        dynamo_tests = fetch_dynamo(token, region, tenant_id, from_date=from_date)
        print(f"    Total DynaMo tests: {len(dynamo_tests)}")
    except Exception as e:
        print(f"    DynaMo fetch FAILED (continuing): {e}")

    if dynamo_tests:
        df = pd.DataFrame(dynamo_tests)
        id_col = 'athleteId' if 'athleteId' in df.columns else 'profileId'

        output_file = OUTPUT_DIR / 'dynamo_allsports_with_athletes.csv'
        df = merge_with_existing(df, output_file, id_column='id')

        # Re-enrich ALL rows after merge (fixes "Unknown" sport for old data)
        df['full_name'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('full_name', f'Athlete_{str(pid)[:8]}'))
        df['athlete_sport'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('athlete_sport', 'Unknown'))
        df['athlete_sex'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('athlete_sex', ''))
        df['Name'] = df['full_name']

        df.to_csv(output_file, index=False)
        print(f"    Saved: {output_file}")

    # Save sync state for future incremental syncs
    sync_state = {
        'last_sync': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.000Z'),
        'devices': {
            'forcedecks': len(forcedecks_tests),
            'forceframe': len(forceframe_tests),
            'nordbord': len(nordbord_tests),
            'dynamo': len(dynamo_tests)
        }
    }
    save_sync_state(sync_state)
    print(f"\n    Sync state saved to: {SYNC_STATE_FILE}")

    print("\n" + "=" * 70)
    print("SYNC COMPLETE!")
    print("=" * 70)
    print(f"\nData saved to: {OUTPUT_DIR}")
    print(f"\nUsage:")
    print("  python local_sync.py              # Incremental (new data only)")
    print("  python local_sync.py --force-all  # Full sync from 2020")
    print("  python local_sync.py --days 7     # Sync last 7 days")
    print("\nNext: Copy to dashboard and run")
    print("  cp ../vald-data/data/*.csv dashboard/data/")
    print("  streamlit run dashboard/world_class_vald_dashboard.py")


if __name__ == '__main__':
    main()
