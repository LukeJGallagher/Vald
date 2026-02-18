"""Quick re-enrichment script: fetches profiles from VALD API and re-maps sports."""
import requests, json, os, time, sys
from dotenv import load_dotenv
import pandas as pd

load_dotenv('config/local_secrets/.env')

CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
TENANT_ID = os.getenv('TENANT_ID')
REGION = os.getenv('VALD_REGION', 'euw')

# Get token
token_url = 'https://security.valdperformance.com/connect/token'
r = requests.post(token_url, data={
    'grant_type': 'client_credentials',
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET
})
token = r.json()['access_token']
print('Token obtained')

# Fetch groups
headers = {'Authorization': f'Bearer {token}'}
groups_url = f'https://prd-{REGION}-api-externaltenants.valdperformance.com/groups'
r = requests.get(groups_url, headers=headers, params={'TenantId': TENANT_ID})
groups_data = r.json()
# Handle both list and dict response formats
if isinstance(groups_data, list):
    group_map = {g['id']: g['name'] for g in groups_data}
elif isinstance(groups_data, dict) and 'groups' in groups_data:
    group_map = {g['id']: g['name'] for g in groups_data['groups']}
else:
    print(f'Unexpected groups response type: {type(groups_data)}')
    print(f'Response preview: {str(groups_data)[:500]}')
    group_map = {}
print(f'Found {len(group_map)} groups')

# Fetch bulk profiles for IDs
profiles_url = f'https://prd-{REGION}-api-externalprofile.valdperformance.com/profiles'
r = requests.get(profiles_url, headers=headers, params={'TenantId': TENANT_ID})
profiles_response = r.json()
# Handle both list and dict response formats
if isinstance(profiles_response, dict) and 'profiles' in profiles_response:
    all_profiles = profiles_response['profiles']
elif isinstance(profiles_response, list):
    all_profiles = profiles_response
else:
    all_profiles = []
    print(f'Unexpected profiles response: {type(profiles_response)}')
print(f'Found {len(all_profiles)} profiles')

# Fetch individual profiles for group memberships
profile_groups = {}
for i, p in enumerate(all_profiles):
    pid = p['profileId']
    try:
        r = requests.get(f'{profiles_url}/{pid}', headers=headers, params={'TenantId': TENANT_ID})
        if r.status_code == 200:
            pdata = r.json()
            gids = pdata.get('groupIds', [])
            gnames = [group_map.get(gid, 'Unknown') for gid in gids]
            full_name = f"{p.get('givenName', '')} {p.get('familyName', '')}".strip()
            sex = pdata.get('sex', '')
            profile_groups[pid] = {
                'full_name': full_name,
                'group_names': gnames,
                'sex': sex
            }
    except Exception as e:
        pass

    if (i + 1) % 50 == 0:
        print(f'  Progress: {i+1}/{len(all_profiles)}')
        sys.stdout.flush()

print(f'Fetched group info for {len(profile_groups)} profiles')

# Save profiles cache for future use
cache_file = 'config/profiles_cache.json'
with open(cache_file, 'w') as f:
    json.dump(profile_groups, f)
print(f'Saved profiles cache to {cache_file}')

# Import centralized sport mapping
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.vald_categories import GROUP_TO_CATEGORY, SKIP_GROUPS


def get_sport(pid):
    pdata = profile_groups.get(str(pid), {})
    groups = pdata.get('group_names', [])
    if not groups:
        return 'Unknown'
    for gn in groups:
        gn_stripped = gn.strip()
        if gn_stripped in SKIP_GROUPS:
            continue
        if gn_stripped in GROUP_TO_CATEGORY:
            cat = GROUP_TO_CATEGORY[gn_stripped]
            if cat is not None:
                return cat
    for gn in groups:
        gn_stripped = gn.strip()
        if gn_stripped not in SKIP_GROUPS:
            return gn_stripped
    return 'Unknown'


def get_name(pid):
    pdata = profile_groups.get(str(pid), {})
    return pdata.get('full_name', f'Athlete_{str(pid)[:8]}')


def get_sex(pid):
    pdata = profile_groups.get(str(pid), {})
    return pdata.get('sex', '')


# Re-enrich all CSVs
for csv_file in ['forcedecks_allsports_with_athletes.csv', 'forceframe_allsports_with_athletes.csv',
                 'nordbord_allsports_with_athletes.csv', 'dynamo_allsports_with_athletes.csv']:

    filepath = f'dashboard/data/{csv_file}'
    vald_filepath = f'../vald-data/data/{csv_file}'

    if not os.path.exists(filepath):
        print(f'SKIP: {csv_file} not in dashboard/data/')
        continue

    df = pd.read_csv(filepath, low_memory=False)

    # Find profile ID column
    id_col = None
    for col in ['profileId', 'athleteId']:
        if col in df.columns:
            id_col = col
            break

    if not id_col:
        print(f'SKIP: {csv_file} no profile ID column')
        continue

    # Re-enrich
    df['athlete_sport'] = df[id_col].map(get_sport)
    df['full_name'] = df[id_col].map(get_name)
    df['athlete_sex'] = df[id_col].map(get_sex)

    # Also set Name column for dashboard compatibility
    if 'Name' not in df.columns:
        df['Name'] = df['full_name']
    else:
        df['Name'] = df['full_name']

    # Save to both locations
    df.to_csv(filepath, index=False)
    if os.path.exists(os.path.dirname(vald_filepath)):
        df.to_csv(vald_filepath, index=False)

    print(f'\n{csv_file}: {len(df)} rows')
    print(df['athlete_sport'].value_counts().to_string())

print('\nDone!')
