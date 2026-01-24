"""
Local sync script to fetch fresh VALD data and save directly to vald-data location.
This script fetches ALL historical data (back to 2020) to ensure no tests are missed.
"""
import os
import sys
import requests
import pandas as pd
from datetime import datetime
import time
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

# Output directory - vald-data repo location
OUTPUT_DIR = Path(r"c:\Users\l.gallagher\OneDrive - Team Saudi\Documents\Performance Analysis\vald-data\data")

# Group to sport mapping
GROUP_TO_CATEGORY = {
    'Epee': 'Fencing', 'Epee ': 'Fencing', 'Foil': 'Fencing', 'Sabre': 'Fencing',
    'Athletics - Horizontal Jumps': 'Athletics', 'Athletics - Middle distance': 'Athletics',
    'Short Sprints': 'Athletics', 'Throwers': 'Athletics', 'Decathlon': 'Athletics',
    'Freestyle': 'Wrestling', 'Greco Roman': 'Wrestling', 'GS': 'Wrestling', 'RUS': 'Wrestling',
    'TKD Junior Female': 'Taekwondo', 'TKD Junior Male': 'Taekwondo',
    'TKD Senior Female': 'Taekwondo', 'TKD Senior Male': 'Taekwondo',
    'SOTC Swimming': 'Swimming',
    'Para Swimming': 'Para Swimming', 'Para Sprints': 'Para Athletics',
    'Para TKD': 'Para Taekwondo', 'Para Cycling': 'Para Cycling', 'Wheel Chair': 'Wheelchair Sports',
    'Karate': 'Karate', 'Coastal': 'Rowing', 'Pistol 10m': 'Shooting',
    'Weightlifting': 'Weightlifting', 'Judo': 'Judo', 'Jiu-Jitsu': 'Jiu-Jitsu',
    'ARCHIVED': None, 'Staff': None, 'TBC': None, 'All Athletes': None,
}

SKIP_GROUPS = {'ARCHIVED', 'Staff', 'TBC', 'All Athletes', 'All athletes', 'VALD HQ', 'Test Group'}


def get_sport_from_groups(group_names):
    """Derive sport from group names."""
    for name in group_names:
        if name in SKIP_GROUPS:
            continue
        if name in GROUP_TO_CATEGORY:
            cat = GROUP_TO_CATEGORY[name]
            if cat:
                return cat
        if name and name not in SKIP_GROUPS:
            return name
    return 'Unknown'


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
            result[pid] = {'full_name': full_name, 'athlete_sport': 'Unknown', 'groupIds': []}
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


def fetch_forcedecks(token, region, tenant_id):
    """Fetch ALL ForceDecks tests from 2020."""
    url = f'https://prd-{region}-api-extforcedecks.valdperformance.com/tests'
    headers = {'Authorization': f'Bearer {token}'}
    from_date = '2020-01-01T00:00:00.000Z'

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

    return all_tests


def fetch_forceframe(token, region, tenant_id):
    """Fetch ALL ForceFrame tests."""
    url = f'https://prd-{region}-api-externalforceframe.valdperformance.com/tests'
    headers = {'Authorization': f'Bearer {token}'}
    from_date = '2020-01-01T00:00:00.000Z'

    all_tests = []
    modified_from = from_date

    while True:
        time.sleep(0.5)
        # Try lowercase first, then uppercase if that fails
        params = {'tenantId': tenant_id, 'modifiedFromUtc': modified_from}
        response = requests.get(url, headers=headers, params=params, timeout=120)

        if response.status_code == 400:
            # Try uppercase parameters
            params = {'TenantId': tenant_id, 'ModifiedFromUtc': modified_from}
            response = requests.get(url, headers=headers, params=params, timeout=120)

        if response.status_code == 204:
            break
        if response.status_code != 200:
            print(f"    API error: {response.status_code} - {response.text[:200]}")
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

    return all_tests


def fetch_nordbord(token, region, tenant_id):
    """Fetch ALL NordBord tests."""
    url = f'https://prd-{region}-api-externalnordbord.valdperformance.com/tests'
    headers = {'Authorization': f'Bearer {token}'}

    all_tests = []
    page = 1

    while page < 100:
        time.sleep(0.5)
        # Try lowercase first, then uppercase
        params = {'tenantId': tenant_id, 'page': page}
        response = requests.get(url, headers=headers, params=params, timeout=120)

        if response.status_code == 400:
            params = {'TenantId': tenant_id, 'Page': page}
            response = requests.get(url, headers=headers, params=params, timeout=120)

        if response.status_code == 204:
            break
        if response.status_code != 200:
            print(f"    API error: {response.status_code} - {response.text[:200]}")
            break

        data = response.json()
        tests = data.get('tests', data) if isinstance(data, dict) else data

        if not tests or not isinstance(tests, list):
            break

        all_tests.extend(tests)
        print(f"    Fetched {len(tests)} tests (total: {len(all_tests)})")

        if len(tests) < 50:
            break
        page += 1

    return all_tests


def main():
    """Main sync execution."""
    print("=" * 70)
    print("LOCAL VALD DATA SYNC")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

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

    print(f"\n[5] Fetching ForceDecks data (all historical)...")
    forcedecks_tests = fetch_forcedecks(token, region, tenant_id)
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

        df['full_name'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('full_name', f'Athlete_{str(pid)[:8]}'))
        df['athlete_sport'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('athlete_sport', 'Unknown'))

        output_file = OUTPUT_DIR / 'forcedecks_allsports_with_athletes.csv'
        df.to_csv(output_file, index=False)
        print(f"    Saved: {output_file}")

        # Sport distribution in tests
        if 'athlete_sport' in df.columns:
            test_sports = df['athlete_sport'].value_counts()
            print(f"\n    Test count by sport:")
            for sport, count in test_sports.head(15).items():
                print(f"      {sport}: {count}")

    print(f"\n[6] Fetching ForceFrame data...")
    forceframe_tests = fetch_forceframe(token, region, tenant_id)
    print(f"    Total ForceFrame tests: {len(forceframe_tests)}")

    if forceframe_tests:
        df = pd.DataFrame(forceframe_tests)
        id_col = 'athleteId' if 'athleteId' in df.columns else 'profileId'

        df['full_name'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('full_name', f'Athlete_{str(pid)[:8]}'))
        df['athlete_sport'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('athlete_sport', 'Unknown'))

        output_file = OUTPUT_DIR / 'forceframe_allsports_with_athletes.csv'
        df.to_csv(output_file, index=False)
        print(f"    Saved: {output_file}")

    print(f"\n[7] Fetching NordBord data...")
    nordbord_tests = fetch_nordbord(token, region, tenant_id)
    print(f"    Total NordBord tests: {len(nordbord_tests)}")

    if nordbord_tests:
        df = pd.DataFrame(nordbord_tests)
        id_col = 'athleteId' if 'athleteId' in df.columns else 'profileId'

        df['full_name'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('full_name', f'Athlete_{str(pid)[:8]}'))
        df['athlete_sport'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('athlete_sport', 'Unknown'))

        output_file = OUTPUT_DIR / 'nordbord_allsports_with_athletes.csv'
        df.to_csv(output_file, index=False)
        print(f"    Saved: {output_file}")

    print("\n" + "=" * 70)
    print("SYNC COMPLETE!")
    print("=" * 70)
    print(f"\nData saved to: {OUTPUT_DIR}")
    print("\nNext: Run the dashboard to verify Swimming athletes appear")
    print("      streamlit run dashboard/world_class_vald_dashboard.py")


if __name__ == '__main__':
    main()
