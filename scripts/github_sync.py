"""
GitHub Actions script to fetch VALD data and export to CSV.
Used by .github/workflows/sync-vald-data.yml
"""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone


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
            result[pid] = {'full_name': full_name, 'athlete_sport': p.get('sport', 'Unknown')}
        return result
    print(f"Profiles error: {response.status_code}")
    return {}


def fetch_device_data(token, region, tenant_id, device):
    """Fetch test data from a VALD device API."""
    base_urls = {
        'forcedecks': f'https://prd-{region}-api-extforcedecks.valdperformance.com/tests',
        'forceframe': f'https://prd-{region}-api-externalforceframe.valdperformance.com/tests',
        'nordbord': f'https://prd-{region}-api-externalnordbord.valdperformance.com/tests',
    }

    url = base_urls[device]
    headers = {'Authorization': f'Bearer {token}'}
    from_date = (datetime.now(timezone.utc) - timedelta(days=365)).strftime('%Y-%m-%dT00:00:00.000Z')

    all_tests = []

    if device == 'forcedecks':
        # ForceDecks uses cursor-based pagination with modifiedFromUtc
        modified_from = from_date
        while True:
            params = {'tenantId': tenant_id, 'modifiedFromUtc': modified_from}
            response = requests.get(url, headers=headers, params=params, timeout=60)
            if response.status_code == 204:
                print(f"{device}: No more data (204)")
                break
            if response.status_code != 200:
                print(f"{device} API error: {response.status_code} - {response.text[:200]}")
                break
            tests = response.json()
            if not tests:
                break
            all_tests.extend(tests)
            print(f"{device}: Fetched {len(tests)} tests (total: {len(all_tests)})")
            # Get the last modified date for next page
            if len(tests) < 100:
                break
            last_modified = tests[-1].get('modifiedDateUtc')
            if not last_modified or last_modified == modified_from:
                break
            modified_from = last_modified

    elif device == 'forceframe':
        # ForceFrame uses cursor-based pagination
        modified_from = from_date
        while True:
            params = {'TenantId': tenant_id, 'ModifiedFromUtc': modified_from}
            response = requests.get(url, headers=headers, params=params, timeout=60)
            if response.status_code == 204:
                print(f"{device}: No more data (204)")
                break
            if response.status_code != 200:
                print(f"{device} API error: {response.status_code} - {response.text[:200]}")
                break
            tests = response.json()
            if not tests:
                break
            all_tests.extend(tests)
            print(f"{device}: Fetched {len(tests)} tests (total: {len(all_tests)})")
            if len(tests) < 100:
                break
            last_modified = tests[-1].get('modifiedDateUtc')
            if not last_modified or last_modified == modified_from:
                break
            modified_from = last_modified

    elif device == 'nordbord':
        # NordBord uses page-based pagination
        page = 1
        while page < 100:
            params = {'TenantId': tenant_id, 'Page': page}
            response = requests.get(url, headers=headers, params=params, timeout=60)
            if response.status_code == 204:
                print(f"{device}: No more data (204)")
                break
            if response.status_code != 200:
                print(f"{device} API error: {response.status_code} - {response.text[:200]}")
                break
            tests = response.json()
            if not tests:
                break
            all_tests.extend(tests)
            print(f"{device}: Fetched {len(tests)} tests (total: {len(all_tests)})")
            if len(tests) < 100:
                break
            page += 1

    return all_tests


def main():
    """Main execution."""
    region = os.environ.get('VALD_REGION', 'euw')
    tenant_id = os.environ.get('VALD_TENANT_ID', '')

    print(f"Region: {region}")
    print(f"Tenant ID: {tenant_id}")

    print("Getting OAuth token...")
    token = get_token()
    print("Token obtained successfully!")

    print("Fetching athlete profiles...")
    profiles = fetch_profiles(token, region, tenant_id)
    print(f"Found {len(profiles)} profiles")

    os.makedirs('data_export', exist_ok=True)

    for device in ['forcedecks', 'forceframe', 'nordbord']:
        print(f"Fetching {device} data...")
        tests = fetch_device_data(token, region, tenant_id, device)
        print(f"Found {len(tests)} {device} tests")

        if tests:
            df = pd.DataFrame(tests)
            id_col = 'athleteId' if 'athleteId' in df.columns else 'profileId'
            if id_col in df.columns:
                df['full_name'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('full_name', f'Athlete_{str(pid)[:8]}'))
                df['athlete_sport'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('athlete_sport', 'Unknown'))

            filename = f'data_export/{device}_allsports_with_athletes.csv'
            df.to_csv(filename, index=False)
            print(f"Saved {filename}")

    print("Data export complete!")


if __name__ == '__main__':
    main()
