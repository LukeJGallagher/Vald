# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Team Saudi VALD Performance Analysis System - A Python/Streamlit analytics platform for Olympic & Paralympic athletes. Fetches data from VALD's API (ForceDecks, ForceFrame, NordBord) and provides advanced analytics.

**Deployment**: Streamlit Cloud (public dashboard) + Private GitHub repo (athlete data with PII)

**Two Repositories:**
- `Vald/` - Dashboard code and scripts (this repo)
- `vald-data/` - Private repo storing CSV files with athlete PII (must push enriched data here for Streamlit Cloud)

## Common Commands

```bash
# Run dashboard locally
cd dashboard && streamlit run world_class_vald_dashboard.py

# Local sync with detailed trial metrics (preferred for full data with all metrics)
python scripts/local_sync.py

# Copy synced data to dashboard directory (dashboard loads from here FIRST)
cp ../vald-data/data/forcedecks_allsports_with_athletes.csv dashboard/data/

# Full data update from VALD API (legacy - metadata only, no trial metrics)
python vald_production_system.py --update-all

# Verify API credentials
python vald_production_system.py --verify

# Push to both GitHub repos (main and master branches)
git push origin main && git push origin main:master
```

## GitHub Actions

Automated data sync runs daily at 6 AM UTC via `.github/workflows/sync-vald-data.yml`.
Required GitHub secrets: `VALD_CLIENT_ID`, `VALD_CLIENT_SECRET`, `VALD_TENANT_ID`, `DATA_REPO_TOKEN`, `DATA_REPO`

## Architecture

### Data Loading Priority (data_loader.py)
1. `dashboard/data/*.csv` - Local dashboard directory (checked FIRST)
2. `../vald-data/data/*.csv` - Private vald-data repo directory
3. Private GitHub repo via token (Streamlit Cloud)
4. VALD API direct fetch (fallback - last 90 days only)

**Important**: After syncing data with `local_sync.py`, copy files to `dashboard/data/` for local testing.

### Key Functions
- `load_vald_data(device)` - Main loader with fallback chain
- `fetch_from_github_repo(device)` - Reads from private `vald-data` repo
- `fetch_from_vald_api(device)` - Direct VALD API fetch
- `push_to_github_repo(df, device)` - Saves refreshed data to private repo
- `refresh_and_save_data(device)` - Fetch from API + push to GitHub

### Streamlit Cloud Secrets
```toml
[github]
GITHUB_TOKEN = "github_pat_..."
DATA_REPO = "LukeJGallagher/vald-data"

[vald]
CLIENT_ID = "..."
CLIENT_SECRET = "..."
TENANT_ID = "..."
VALD_REGION = "euw"
```

### Git Branch Setup
- Streamlit Cloud deploys from `master` branch
- Local development uses `main` branch
- Always push to both: `git push origin main:master`

## API Endpoints

```
OAuth: https://security.valdperformance.com/connect/token (use empty scope!)
ForceDecks: https://prd-{REGION}-api-extforcedecks.valdperformance.com/
ForceFrame: https://prd-{REGION}-api-externalforceframe.valdperformance.com/
NordBord: https://prd-{REGION}-api-externalnordbord.valdperformance.com/
Profiles: https://prd-{REGION}-api-externalprofile.valdperformance.com/
Tenants: https://prd-{REGION}-api-externaltenants.valdperformance.com/
```
Regions: `euw` (Europe), `use` (US East), `aue` (Australia)

**OAuth Note**: Use empty scope for client credentials grant. DO NOT include scope parameter.

### API Key Methods (from Kenny's vald-aspire)
- **Get Profiles**: `/profiles?TenantId={tenant_id}` - Returns all athlete profiles with names
- **Get Tests**: `/v2019q3/teams/{tenant_id}/tests?modifiedFromUtc={date}` - Test data with pagination
- **Get Trials**: `/v2019q3/teams/{tenant_id}/tests/{test_id}/trials` - Individual trial data with all metrics
- **Get Groups**: Tenants API `/groups?TenantId={tenant_id}` - Team/group information

### Trial Data Format
The `/tests` endpoint returns metadata only (19 columns). To get detailed metrics (700+ columns like jump height, power, force), fetch trials:
```python
# Trial response format
[{
    'id': '...',
    'results': [{
        'resultId': 123,
        'value': 0.45,  # The actual metric value
        'limb': 'Trial',  # or 'Left', 'Right' for bilateral
        'definition': {'result': 'JUMP_HEIGHT'}  # Metric name
    }]
}]
```

### API Differences by Device

**ForceDecks** - Uses cursor pagination:
```python
params = {'tenantId': tenant_id, 'modifiedFromUtc': '2020-01-01T00:00:00.000Z'}
```

**ForceFrame/NordBord** - Requires date range parameters:
```python
params = {'TenantId': tenant_id, 'TestFromUtc': '...', 'TestToUtc': '...'}
```

### API Limitation
VALD API can only pull **6 months of data maximum** per API call. Use staggered date ranges for historical data.

## Athlete Name & Sport Mapping

ForceFrame/NordBord use `athleteId` which equals `profileId` in the Profiles API.

**Important**: The bulk `/profiles` endpoint returns names but **NOT** sport or groupIds. To get sports:
1. Fetch groups from Tenants API: `/groups?TenantId=...` → `{id: name}` mapping
2. Fetch individual profile: `/profiles/{profileId}?TenantId=...` → returns `groupIds` array
3. Map groupIds to group names (e.g., "Athletics", "Fencing", "SOTC Swimming")

The `Groups` column in ForceDecks data IS the sport. Sports like "Swimming" come from VALD group membership.

### Data Sync Scripts

**`scripts/local_sync.py`** (Recommended for local development):
- Fetches all historical ForceDecks tests from 2020
- Fetches trial data for detailed metrics (700+ columns)
- Enriches with athlete names and sports via Kenny's approach
- Outputs to `vald-data/data/forcedecks_allsports_with_athletes.csv`

**`scripts/github_sync.py`**:
- Used by GitHub Actions for automated daily updates
- Fetches individual profiles to get groupIds for sport assignment

**`enrich_from_forcedecks.py`** (Legacy):
- Enriches existing data with athlete names from Profiles API

Local sync workflow:
```bash
# 1. Sync all data with trial metrics
python scripts/local_sync.py

# 2. Copy to dashboard directory
cp ../vald-data/data/forcedecks_allsports_with_athletes.csv dashboard/data/

# 3. Run dashboard
cd dashboard && streamlit run world_class_vald_dashboard.py
```

## Important Compatibility Notes

### NumPy 2.0+
Streamlit Cloud uses NumPy 2.0 where `np.trapz` was removed. Use:
```python
try:
    result = np.trapezoid(y, x)
except AttributeError:
    result = np.trapz(y, x)  # Fallback for older NumPy
```

### Streamlit Deprecations
`use_container_width` is deprecated (remove after 2025-12-31). Use `width='stretch'` instead.

## Key Analytics Methods

**Asymmetry Monitoring**: Flags bilateral imbalances >10%

**Adaptive Ranges**: EM-based baseline with meaningful change threshold at 0.6×SD

**F-V-P Profiles**: Linear regression (F = F0 - slope × V) for power optimization

## Branding

Team Saudi colors (Official Saudi Flag Green - PMS 3425 C):
- Primary Green: `#005430` - Main brand, headers, buttons
- Gold Accent: `#a08e66` - Highlights, PB markers
- Dark Green: `#003d1f` - Hover states, gradients
- Light Green: `#2A8F5C` - Secondary positive

Header gradient: `linear-gradient(135deg, #005430 0%, #003d1f 100%)`

Theme files: `dashboard/theme/colors.py`, `C:\Users\l.gallagher\.claude\branding\team_saudi\THEME_GUIDE.md`

## Credentials Location

Local testing: `config/local_secrets/.env`
Production: Streamlit Cloud Secrets dashboard

## Troubleshooting

**Metrics/charts blank (only athlete names showing):**
1. The `/tests` API only returns metadata (19 columns). Need trial data for metrics.
2. Run `python scripts/local_sync.py` to fetch trial data with all metrics
3. Copy to dashboard: `cp ../vald-data/data/forcedecks_allsports_with_athletes.csv dashboard/data/`
4. Restart dashboard

**Athlete names not showing (shows "Athlete_XXXXXX"):**
1. Run `python scripts/local_sync.py` (includes athlete enrichment)
2. Copy data to `dashboard/data/` directory
3. Restart Streamlit (cache is 1 hour)

**Sport filter showing no athletes (e.g., Swimming = 0):**
1. Bulk profiles API doesn't return groupIds - need Kenny's approach
2. Run `python scripts/local_sync.py` which fetches individual profiles for groups
3. Group mapping: "SOTC Swimming" → "Swimming", "Epee/Foil/Sabre" → "Fencing"

**API Authentication Errors (401):**
1. Check `.env` credentials in `config/local_secrets/`
2. Verify region setting matches your VALD account (euw/use/aue)
3. Test with: `python vald_production_system.py --verify`

**ForceFrame/NordBord API 400 errors:**
- These APIs require `TestFromUtc` and `TestToUtc` parameters (unlike ForceDecks)
- Check parameter casing: some use `TenantId`, others `tenantId`

**Rate Limiting (429):**
- System auto-handles with 12 calls per 5 seconds
- Adjust in `config/vald_config.py` if needed
