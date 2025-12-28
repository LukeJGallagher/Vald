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

# Full data update from VALD API
python vald_production_system.py --update-all

# Verify API credentials
python vald_production_system.py --verify

# Push to both GitHub repos (main and master branches)
git push origin main && git push origin main:master
```

## Architecture

### Data Loading Priority (data_loader.py)
1. Local CSV files (development)
2. Private GitHub repo via token (Streamlit Cloud - has full history)
3. VALD API direct fetch (fallback - last 90 days only)

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
OAuth: https://security.valdperformance.com/connect/token
ForceDecks: https://prd-{REGION}-api-extforcedecks.valdperformance.com/
ForceFrame: https://prd-{REGION}-api-externalforceframe.valdperformance.com/
NordBord: https://prd-{REGION}-api-externalnordbord.valdperformance.com/
Profiles: https://prd-{REGION}-api-externalprofile.valdperformance.com/
Tenants: https://prd-{REGION}-api-externaltenants.valdperformance.com/
```
Regions: `euw` (Europe), `use` (US East), `aue` (Australia)

### API Key Methods (from Kenny's vald-aspire)
- **Get Profiles**: `/profiles?TenantId={tenant_id}` - Returns all athlete profiles with names
- **Get Tests**: `/v2019q3/teams/{tenant_id}/tests?modifiedFromUtc={date}` - Test data with pagination
- **Get Trials**: `/v2019q3/teams/{tenant_id}/tests/{test_id}/trials` - Individual trial data
- **Get Groups**: Tenants API `/groups?TenantId={tenant_id}` - Team/group information

### API Limitation
VALD API can only pull **6 months of data maximum** per API call. Use staggered date ranges for historical data.

## Athlete Name Mapping

ForceFrame/NordBord use `athleteId` which equals `profileId` in the Profiles API.

**Enrichment script**: `enrich_from_forcedecks.py`
1. Fetches all profiles from Profiles API (606+ athletes)
2. Creates `full_name` from `givenName` + `familyName`
3. Enriches ForceFrame/NordBord CSVs with real names
4. Dashboard creates `Name` column from `full_name`

Run after API data updates:
```bash
python enrich_from_forcedecks.py

# Then push enriched files to vald-data repo for Streamlit Cloud
cd ../vald-data
git add data/*_with_athletes.csv && git commit -m "Update enriched data" && git push
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

Team Saudi colors:
- Primary Teal: `#007167`
- Gold Accent: `#a08e66`
- Dark Teal: `#005a51`

## Credentials Location

Local testing: `config/local_secrets/.env`
Production: Streamlit Cloud Secrets dashboard

## Troubleshooting

**Athlete names not showing (shows "Athlete_XXXXXX"):**
1. Run `python enrich_from_forcedecks.py` to fetch profiles and enrich data
2. Push `*_with_athletes.csv` files to `vald-data` repo
3. Click "🔄 Refresh" button in dashboard or restart Streamlit (cache is 1 hour)

**API Authentication Errors (401):**
1. Check `.env` credentials in `config/local_secrets/`
2. Verify region setting matches your VALD account (euw/use/aue)
3. Test with: `python vald_production_system.py --verify`

**Rate Limiting (429):**
- System auto-handles with 12 calls per 5 seconds
- Adjust in `config/vald_config.py` if needed
