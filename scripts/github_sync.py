"""
GitHub Actions script to fetch VALD data and export to CSV.
Used by .github/workflows/sync-vald-data.yml

Sports are derived from VALD Groups mapped to Categories.
Group names (Epee, Foil, Sabre) -> Category (Fencing)
"""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time

# ============================================================================
# VALD Group to Category (Sport) Mapping
# Categories are the actual sports. Groups are sub-divisions within sports.
# ============================================================================

GROUP_TO_CATEGORY = {
    # Fencing (3 groups)
    'Epee': 'Fencing', 'Epee ': 'Fencing', 'Foil': 'Fencing', 'Sabre': 'Fencing',
    # Athletics (5 groups)
    'Athletics - Horizontal Jumps': 'Athletics', 'Athletics - Middle distance': 'Athletics',
    'Short Sprints': 'Athletics', 'Throwers': 'Athletics', 'Decathlon': 'Athletics',
    # Wrestling (4 groups)
    'Freestyle': 'Wrestling', 'Greco Roman': 'Wrestling', 'GS': 'Wrestling', 'RUS': 'Wrestling',
    # Taekwondo (4 groups)
    'TKD Junior Female': 'Taekwondo', 'TKD Junior Male': 'Taekwondo',
    'TKD Senior Female': 'Taekwondo', 'TKD Senior Male': 'Taekwondo',
    # Swimming
    'SOTC Swimming': 'Swimming',
    # Para Sports
    'Para Swimming': 'Para Swimming', 'Para Sprints': 'Para Athletics',
    'Para TKD': 'Para Taekwondo', 'Para Cycling': 'Para Cycling', 'Wheel Chair': 'Wheelchair Sports',
    # Individual Sports (group = category)
    'Karate': 'Karate', 'Coastal': 'Rowing', 'Pistol 10m': 'Shooting',
    'Snow Sports': 'Snow Sports', 'Equestrian': 'Equestrian',
    'Judo': 'Judo', 'Jiu-Jitsu': 'Jiu-Jitsu', 'Weightlifting': 'Weightlifting',
    # Excluded groups
    'ARCHIVED': None, 'Staff': None, 'TBC': None, 'All Athletes': None,
}

SKIP_GROUPS = {
    'ARCHIVED', 'Staff', 'TBC', 'All Athletes', 'All athletes',
    'VALD HQ', 'Test Group', 'Performance Staff', 'Coaches', 'Medical', 'Admin'
}


def get_sport_from_groups(group_names):
    """Get sport (category) from group names using centralized mapping."""
    for name in group_names:
        if name in SKIP_GROUPS:
            continue
        if name in GROUP_TO_CATEGORY:
            category = GROUP_TO_CATEGORY[name]
            if category:
                return category
        if name and name not in SKIP_GROUPS:
            return name
    return 'Unknown'


def get_token():
    """Get OAuth token from VALD API."""
    client_id = os.environ.get('VALD_CLIENT_ID', '')
    client_secret = os.environ.get('VALD_CLIENT_SECRET', '')

    print(f"Client ID length: {len(client_id)}")
    print(f"Client Secret length: {len(client_secret)}")

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
        print(f"Token error: {response.status_code} - {response.text}")
        raise Exception("Failed to get OAuth token")
    return response.json()['access_token']


def fetch_groups(token, region, tenant_id):
    """Fetch group names from VALD Tenants API.
    Returns {group_id: group_name} mapping.
    """
    # Try Tenants API first (newer API)
    url = f'https://prd-{region}-api-externaltenants.valdperformance.com/groups'
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(url, headers=headers, params={'TenantId': tenant_id}, timeout=60)

    if response.status_code == 200:
        data = response.json()
        groups = data.get('groups', data) if isinstance(data, dict) else data
        result = {g['id']: g['name'] for g in groups if 'id' in g and 'name' in g}
        print(f"Fetched {len(result)} groups from Tenants API")
        return result

    # Fallback to legacy groupnames endpoint
    url = f'https://prd-{region}-api-extforcedecks.valdperformance.com/groupnames'
    response = requests.get(url, headers=headers, params={'TenantId': tenant_id}, timeout=60)

    if response.status_code == 200:
        data = response.json()
        groups = data.get('groups', data) if isinstance(data, dict) else data
        result = {g['id']: g['name'] for g in groups if 'id' in g and 'name' in g}
        print(f"Fetched {len(result)} groups from legacy API")
        return result

    print(f"Groups fetch error: {response.status_code}")
    return {}


def fetch_individual_profile(token, region, tenant_id, profile_id, rate_limit_delay=0.1):
    """Fetch individual profile to get groupIds.
    Individual profile endpoint returns groupIds, bulk endpoint does not.
    """
    url = f'https://prd-{region}-api-externalprofile.valdperformance.com/profiles/{profile_id}'
    headers = {'Authorization': f'Bearer {token}'}

    try:
        time.sleep(rate_limit_delay)  # Rate limiting
        response = requests.get(url, headers=headers, params={'TenantId': tenant_id}, timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching profile {profile_id}: {e}")
    return None


def fetch_profiles(token, region, tenant_id):
    """Fetch athlete profiles from VALD Profiles API."""
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
            full_name = f"{given} {family}".strip() or p.get('fullName') or 'Unknown'
            # Note: bulk profiles endpoint does NOT return sport
            # Sport is derived from group membership in enrich_with_groups()
            result[pid] = {
                'full_name': full_name,
                'athlete_sport': 'Unknown',  # Will be enriched later from groups
                'groupIds': []  # Will be populated if available
            }
        return result, profiles  # Return raw profiles too for group lookup
    print(f"Profiles error: {response.status_code}")
    return {}, []


def enrich_with_groups(profiles_dict, raw_profiles, groups_map, token, region, tenant_id, fetch_individual=True):
    """Enrich profiles with sport from group membership.

    Priority sport groups (in order):
    1. Sports like 'Athletics', 'Fencing', 'Swimming', etc.
    2. Exclude generic groups like 'All Athletes', 'VALD HQ'

    If fetch_individual=True, fetches individual profiles to get groupIds
    (the bulk endpoint doesn't return groupIds).
    """
    # Generic groups to skip when determining sport
    generic_groups = {
        'All Athletes', 'All athletes', 'VALD HQ', 'Test Group',
        'SOTC Performance', 'Saudi Sports', 'Performance Staff',
        'Coaches', 'Medical', 'Admin'
    }

    profiles_with_groups = 0
    profiles_needing_fetch = []

    # First pass: check which profiles already have groupIds
    for profile in raw_profiles:
        pid = profile.get('profileId') or profile.get('id')
        if not pid or pid not in profiles_dict:
            continue

        group_ids = profile.get('groupIds', [])

        if group_ids:
            profiles_with_groups += 1
            group_names = [groups_map.get(gid, 'Unknown') for gid in group_ids]
            sport = get_sport_from_groups(group_names)
            profiles_dict[pid]['athlete_sport'] = sport
            profiles_dict[pid]['groups'] = '|'.join(group_names)
        else:
            profiles_needing_fetch.append(pid)

    print(f"Found {profiles_with_groups} profiles with group info from bulk endpoint")
    print(f"Need to fetch individual profiles for {len(profiles_needing_fetch)} athletes")

    # Second pass: fetch individual profiles to get groupIds
    if fetch_individual and profiles_needing_fetch:
        print(f"Fetching individual profiles for group membership...")
        fetched = 0
        for i, pid in enumerate(profiles_needing_fetch):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(profiles_needing_fetch)}")

            profile_data = fetch_individual_profile(token, region, tenant_id, pid)

            if profile_data:
                group_ids = profile_data.get('groupIds', [])
                if group_ids:
                    fetched += 1
                    group_names = [groups_map.get(gid, 'Unknown') for gid in group_ids]
                    sport = get_sport_from_groups(group_names)
                    profiles_dict[pid]['athlete_sport'] = sport
                    profiles_dict[pid]['groups'] = '|'.join(group_names)

        print(f"Fetched group info for {fetched} additional profiles")
        profiles_with_groups += fetched

    print(f"Total: Enriched {profiles_with_groups} profiles with group/sport info")
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

    all_metrics = {}

    for trial in trials:
        results = trial.get('results', [])

        if isinstance(results, list):
            for result in results:
                if not isinstance(result, dict):
                    continue

                value = result.get('value')
                if not isinstance(value, (int, float)) or value is None:
                    continue

                definition = result.get('definition', {})
                metric_name = definition.get('result', '') if isinstance(definition, dict) else ''

                if not metric_name:
                    metric_name = f"metric_{result.get('resultId', 'unknown')}"

                limb = result.get('limb', 'Trial')
                if limb and limb != 'Trial':
                    metric_key = f"{metric_name}_{limb}"
                else:
                    metric_key = metric_name

                if metric_key not in all_metrics:
                    all_metrics[metric_key] = []
                all_metrics[metric_key].append(value)

        elif isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, (int, float)) and value is not None:
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

    flattened = {}
    for key, values in all_metrics.items():
        if values:
            key_lower = key.lower()
            if any(x in key_lower for x in ['height', 'power', 'velocity', 'peak', 'max']):
                flattened[key] = max(values)
            else:
                flattened[key] = sum(values) / len(values)

    return flattened


def fetch_device_data(token, region, tenant_id, device, fetch_trials=True):
    """Fetch test data from a VALD device API."""
    base_urls = {
        'forcedecks': f'https://prd-{region}-api-extforcedecks.valdperformance.com/tests',
        'forceframe': f'https://prd-{region}-api-externalforceframe.valdperformance.com/tests',
        'nordbord': f'https://prd-{region}-api-externalnordbord.valdperformance.com/tests',
    }

    url = base_urls[device]
    headers = {'Authorization': f'Bearer {token}'}
    # Fetch ALL historical data from 2020 to get complete dataset
    from_date = '2020-01-01T00:00:00.000Z'

    all_tests = []

    if device == 'forcedecks':
        # ForceDecks uses cursor-based pagination with modifiedFromUtc
        # API returns {'tests': [...]} not just a list
        modified_from = from_date
        while True:
            time.sleep(0.5)  # Rate limit
            params = {'tenantId': tenant_id, 'modifiedFromUtc': modified_from}
            response = requests.get(url, headers=headers, params=params, timeout=120)
            if response.status_code == 204:
                print(f"{device}: No more data (204)")
                break
            if response.status_code != 200:
                print(f"{device} API error: {response.status_code} - {response.text[:200]}")
                break
            data = response.json()
            # Handle both {'tests': [...]} and [...] response formats
            tests = data.get('tests', data) if isinstance(data, dict) else data
            if not tests or not isinstance(tests, list):
                print(f"{device}: Empty or invalid response")
                break
            all_tests.extend(tests)
            print(f"{device}: Fetched {len(tests)} tests (total: {len(all_tests)})")
            # Get the last modified date for next page
            if len(tests) < 50:
                break
            last_modified = tests[-1].get('modifiedDateUtc')
            if not last_modified or last_modified == modified_from:
                break
            modified_from = last_modified

        # Fetch trial data for detailed metrics (ForceDecks only)
        if fetch_trials and all_tests:
            print(f"{device}: Fetching trial data for {len(all_tests)} tests...")
            for i, test in enumerate(all_tests):
                test_id = test.get('id') or test.get('testId')
                if test_id:
                    trials = fetch_trial_data(token, region, tenant_id, test_id, device)
                    if trials:
                        metrics = flatten_trial_metrics(trials)
                        test.update(metrics)

                if (i + 1) % 100 == 0:
                    print(f"{device}: Trial progress: {i + 1}/{len(all_tests)}")

            print(f"{device}: Trial fetching complete!")

    elif device == 'forceframe':
        # ForceFrame uses cursor-based pagination
        # API may return {'tests': [...]} or [...]
        modified_from = from_date
        while True:
            time.sleep(0.5)  # Rate limit
            params = {'TenantId': tenant_id, 'ModifiedFromUtc': modified_from}
            response = requests.get(url, headers=headers, params=params, timeout=120)
            if response.status_code == 204:
                print(f"{device}: No more data (204)")
                break
            if response.status_code != 200:
                print(f"{device} API error: {response.status_code} - {response.text[:200]}")
                break
            data = response.json()
            tests = data.get('tests', data) if isinstance(data, dict) else data
            if not tests or not isinstance(tests, list):
                print(f"{device}: Empty or invalid response")
                break
            all_tests.extend(tests)
            print(f"{device}: Fetched {len(tests)} tests (total: {len(all_tests)})")
            if len(tests) < 50:
                break
            last_modified = tests[-1].get('modifiedDateUtc')
            if not last_modified or last_modified == modified_from:
                break
            modified_from = last_modified

    elif device == 'nordbord':
        # NordBord uses page-based pagination
        # API may return {'tests': [...]} or [...]
        page = 1
        while page < 100:
            time.sleep(0.5)  # Rate limit
            params = {'TenantId': tenant_id, 'Page': page}
            response = requests.get(url, headers=headers, params=params, timeout=120)
            if response.status_code == 204:
                print(f"{device}: No more data (204)")
                break
            if response.status_code != 200:
                print(f"{device} API error: {response.status_code} - {response.text[:200]}")
                break
            data = response.json()
            tests = data.get('tests', data) if isinstance(data, dict) else data
            if not tests or not isinstance(tests, list):
                print(f"{device}: Empty or invalid response")
                break
            all_tests.extend(tests)
            print(f"{device}: Fetched {len(tests)} tests (total: {len(all_tests)})")
            if len(tests) < 50:
                break
            page += 1

    return all_tests


def merge_with_existing(new_df, existing_path, id_column='testId'):
    """Merge new data with existing CSV, removing duplicates."""
    if os.path.exists(existing_path):
        try:
            existing_df = pd.read_csv(existing_path)
            print(f"  Existing data: {len(existing_df)} rows")

            # Combine and remove duplicates based on test ID
            if id_column in new_df.columns and id_column in existing_df.columns:
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=[id_column], keep='last')
                print(f"  After merge: {len(combined)} rows")
                return combined
            else:
                # If no ID column, just append
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                return combined
        except Exception as e:
            print(f"  Could not read existing file: {e}")

    return new_df


def main():
    """Main execution."""
    region = os.environ.get('VALD_REGION', 'euw')
    tenant_id = os.environ.get('VALD_TENANT_ID', '')

    print(f"Region: {region}")
    print(f"Tenant ID: {tenant_id}")

    print("Getting OAuth token...")
    token = get_token()
    print("Token obtained successfully!")

    print("Fetching groups (for sport mapping)...")
    groups_map = fetch_groups(token, region, tenant_id)
    print(f"Found {len(groups_map)} groups")

    print("Fetching athlete profiles...")
    profiles, raw_profiles = fetch_profiles(token, region, tenant_id)
    print(f"Found {len(profiles)} profiles")

    # Enrich profiles with sport from groups
    # fetch_individual=True will query each profile individually to get groupIds
    # This is slower but ensures all athletes get their sport assigned correctly
    profiles = enrich_with_groups(profiles, raw_profiles, groups_map, token, region, tenant_id, fetch_individual=True)

    # Also try to load existing sport data from ForceDecks (for historical context)
    existing_sports = {}
    forcedecks_path = 'private_data_repo/data/forcedecks_allsports_with_athletes.csv'
    if os.path.exists(forcedecks_path):
        try:
            existing_df = pd.read_csv(forcedecks_path)
            if 'profileId' in existing_df.columns and 'athlete_sport' in existing_df.columns:
                sport_mapping = existing_df[['profileId', 'athlete_sport']].dropna().drop_duplicates()
                for _, row in sport_mapping.iterrows():
                    pid = str(row['profileId'])
                    sport = row['athlete_sport']
                    if pid and sport and sport not in ['Unknown', '', 'nan']:
                        existing_sports[pid] = sport
                print(f"Loaded {len(existing_sports)} existing sport mappings from ForceDecks")
        except Exception as e:
            print(f"Could not load existing sports: {e}")

    # Apply existing sports as fallback
    for pid, sport in existing_sports.items():
        if pid in profiles and profiles[pid].get('athlete_sport') == 'Unknown':
            profiles[pid]['athlete_sport'] = sport

    os.makedirs('data_export', exist_ok=True)

    for device in ['forcedecks', 'forceframe', 'nordbord']:
        print(f"\nFetching {device} data...")
        tests = fetch_device_data(token, region, tenant_id, device)
        print(f"Found {len(tests)} {device} tests from API")

        filename = f'data_export/{device}_allsports_with_athletes.csv'
        existing_path = f'private_data_repo/data/{device}_allsports_with_athletes.csv'

        if tests:
            df = pd.DataFrame(tests)
            id_col = 'athleteId' if 'athleteId' in df.columns else 'profileId'
            if id_col in df.columns:
                df['full_name'] = df[id_col].map(
                    lambda pid: profiles.get(pid, {}).get('full_name', f'Athlete_{str(pid)[:8]}')
                )
                df['athlete_sport'] = df[id_col].map(
                    lambda pid: profiles.get(pid, {}).get('athlete_sport', 'Unknown')
                )

            # Merge with existing data
            df = merge_with_existing(df, existing_path)

            # Report sport distribution
            if 'athlete_sport' in df.columns:
                sports = df['athlete_sport'].value_counts()
                print(f"Sport distribution: {dict(sports.head(10))}")

            df.to_csv(filename, index=False)
            print(f"Saved {filename} ({len(df)} total rows)")
        elif os.path.exists(existing_path):
            # No new data, but keep existing
            import shutil
            shutil.copy(existing_path, filename)
            print(f"No new {device} data, kept existing file")

    print("\nData export complete!")


if __name__ == '__main__':
    main()
