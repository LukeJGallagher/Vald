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
DynaMo: https://prd-{REGION}-api-extdynamo.valdperformance.com/
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

# 2. Copy ALL synced files to dashboard directory
cp ../vald-data/data/*.csv dashboard/data/

# 3. Run dashboard
cd dashboard && streamlit run world_class_vald_dashboard.py
```

### Data Enrichment Pattern (CRITICAL)
When syncing data, athlete names and sports must be enriched AFTER merging with existing data, not before. This ensures old rows get updated with the latest profile info.

**Correct pattern (in local_sync.py and github_sync.py):**
```python
# 1. Merge first
df = merge_with_existing(df, output_file, id_column='id')

# 2. THEN re-enrich ALL rows (fixes "Unknown" sport for old data)
df['full_name'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('full_name', f'Athlete_{str(pid)[:8]}'))
df['athlete_sport'] = df[id_col].map(lambda pid: profiles.get(pid, {}).get('athlete_sport', 'Unknown'))

# 3. Save
df.to_csv(output_file, index=False)
```

**Wrong pattern (enriches before merge, old rows keep stale values):**
```python
df['full_name'] = df[id_col].map(...)  # Enriches new data only
df = merge_with_existing(...)           # Old rows keep "Unknown"
```

## Manual Test Data Files

Located in `dashboard/data/`:

| File | Content | Key Columns |
|------|---------|-------------|
| `broad_jump.csv` | Broad jump distances | date, athlete, distance_cm, attempt, session_type, notes |
| `sc_lower_body.csv` | Lower body strength (Squat, Deadlift, Front Squat) | date, athlete, exercise, weight_kg, reps, sets, rpe, estimated_1rm |
| `sc_upper_body.csv` | Upper body strength (Bench, Pull Up, OHP) | date, athlete, exercise, weight_kg, reps, sets, rpe, estimated_1rm |
| `power_tests.csv` | Power tests (Peak, Repeat, Glycolytic) | date, athlete, test_type, peak_wattage, avg_wattage, body_mass_kg |
| `aerobic_tests.csv` | Aerobic capacity | date, athlete, avg_wattage, body_mass_kg, avg_relative_wattage |
| `trunk_endurance.csv` | Trunk endurance tests | date, athlete, supine_sec, prone_sec, lateral_left_sec, lateral_right_sec |
| `training_distances.csv` | Training load (throws) | date, athlete, event, implement_kg, distance_m |

**Required Filtering Columns** (added to all manual test files for dashboard compatibility):
- `athlete_sport` - Sport category for filtering (e.g., "Staff", "Athletics", "Fencing")
- `Name` - Full athlete name (matches VALD profile name format)
- `recordedDateUtc` - Date for time-based filtering (same as `date` column)

These columns enable consistent filtering across VALD data and manual test data in S&C Diagnostics tabs.

## S&C Diagnostics Tab Structure

```python
test_tabs = st.tabs([
    "📊 IMTP",           # Ranked Bar - ForceDecks IMTP tests
    "🦘 CMJ",            # Ranked Bar - ForceDecks CMJ tests
    "🦵 SL Tests",       # Side-by-Side - SL ISO Squat, SL IMTP, SL CMJ, SL DJ, SL Jump, SL Hop, Ash Test
    "💪 NordBord",       # Side-by-Side - Nordic hamstring (Left/Right)
    "🏃 10:5 Hop",       # Ranked Bar - HJ, SLHJ, RSHIP, RSKIP, RSAIP
    "🔄 Quadrant Tests", # Stacked - Trunk, Neck, Shoulder, Hip (ForceFrame)
    "🏋️ Strength RM",    # Ranked Bar - Manual Entry (sc_lower_body.csv, sc_upper_body.csv)
    "🦘 Broad Jump",     # Ranked Bar - Manual Entry (broad_jump.csv)
    "🏃 Fitness Tests",  # Ranked Bar - 6 Min Aerobic, VO2 Max, Yo-Yo, 30-15 IFT
    "💥 Plyo Pushup",    # Ranked Bar - ForceDecks PPU tests (upper body power)
    "✊ DynaMo",          # Ranked Bar - Grip strength
    "⚖️ Balance"         # Ranked Bar - QSB, SLSB for Shooting
])
```

### VALD Test Type Codes (ForceDecks)
- **IMTP**: Isometric Mid-Thigh Pull
- **CMJ**: Countermovement Jump
- **PPU**: Plyo Pushup (upper body power)
  - Key metric: `PUSHUP_HEIGHT` (cm)
  - Also available: `FLIGHT_TIME`, `BODYMASS_RELATIVE_TAKEOFF_POWER`
- **HJ**: Hop Jump, **SLHJ**: Single Leg Hop Jump
- **DJ**: Drop Jump, **SJ**: Squat Jump
- **QSB**: Quiet Stance Balance, **SLSB**: Single Leg Stance Balance
- **ISOT**: Isometric tests, **SLISOT**: Single Leg Isometric
- **SLISOSQT**: Single Leg Isometric Squat

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

## Sport-Specific Test Types

### Shooting (10m Pistol)
**Balance Tests** - Quiet standing 30 sec, open eyes
- Test types: `QSB` (Quiet Static Balance), `SLSB` (Single Leg Static Balance)
- Located in S&C Diagnostics → "⚖️ Balance" tab
- Key metrics (VALD stores in m, display in mm):
  - `BAL_COP_TOTAL_EXCURSION` - Total CoP excursion (mm) - lower is better
  - `BAL_COP_MEAN_VELOCITY` - Mean CoP velocity (mm/s) - lower is better
  - `BAL_COP_ELLIPSE_AREA` - CoP 95% ellipse area (mm²) - smaller is better

### Hop Tests (All Sports)
- Test types: `HJ` (Hop Jump), `SLHJ` (Single Leg Hop Jump), `RSHIP`, `RSKIP`, `RSAIP`
- Located in S&C Diagnostics → "🏃 10:5 Hop" tab
- Key metric: RSI (Reactive Strength Index)

### Common Test Type Codes
| Code | Full Name | Primary Sports |
|------|-----------|----------------|
| CMJ | Counter Movement Jump | All |
| IMTP | Isometric Mid-Thigh Pull | All |
| HJ | Hop Jump | All |
| QSB | Quiet Static Balance | Shooting |
| SLSB | Single Leg Static Balance | Shooting |
| ISOT | Isometric Test | Various |
| SLJ | Single Leg Jump | All |

## Important Fixes (Don't Revert!)

### API Endpoint Corrections
- **DynaMo**: Use `prd-{region}-api-extdynamo.valdperformance.com` (NOT `externaldynamo`)
  - Endpoint: `/v2022q2/teams/{tenant_id}/tests`
  - Uses page-based pagination with `includeRepSummaries=true`
  - Key metric: `maxForceNewtons` for grip strength
  - Filter by `movement == 'GripSqueeze'` for grip tests (excludes trunk, ankle tests)
  - **Note**: DynaMo athletes often show "Unknown" sport because their profiles may not have VALD group memberships
- **ForceFrame/NordBord**: Require ALL date params: `TestFromUtc`, `TestToUtc`, `ModifiedFromUtc`

### Unit Conversions (S&C Diagnostics)
- **Balance metrics**: VALD stores in meters, display in mm
  - Total Excursion: multiply by 1000 (m → mm)
  - Mean Velocity: multiply by 1000 (m/s → mm/s)
  - Ellipse Area: multiply by 1,000,000 (m² → mm²)
- **RSI metrics**: Some stored as 0-100 scale, display as 0-1 scale (divide by 100 if median > 10)

### S&C Diagnostics Tab Filters
- **10:5 Hop tab**: Filter by `testType.isin(['HJ', 'SLHJ', 'RSHIP', 'RSKIP', 'RSAIP'])`
  - NOT `str.contains('Hop')` - that misses HJ, RSHIP codes
- **Balance tab**: Filter by `testType.isin(['QSB', 'SLSB'])`
  - Primarily for Shooting (10m Pistol) athletes

## Current Data State (Jan 2026)

### VALD Device Data (after full sync)
| Device | Tests | Key Sports |
|--------|-------|------------|
| ForceDecks | 2,102 | Fencing (643), Taekwondo (255), Karate (217), Wrestling (190), Weightlifting (178) |
| ForceFrame | 203 | Karate (52), Fencing (32), Athletics (22), Jiu-Jitsu (19), Para Athletics (12) |
| NordBord | 165 | Various sports |
| DynaMo | 837 | Fencing (280), Weightlifting (109), TKD (91), Wrestling (68), Karate (29) |

### Sport Athlete Counts (650 profiles)
- Fencing: 142 | Karate: 75 | Athletics: 61 | Taekwondo: 48 | Para Athletics: 45
- Judo: 40 | Weightlifting: 34 | Jiu-Jitsu: 32 | Wrestling: 31 | Shooting: 22
- Swimming: 17 | Rowing: 17 | Equestrian: 7 | Snow Sports: 4

### Manual Test Data (Staff only)
- `sc_lower_body.csv`, `sc_upper_body.csv` - Luke Gallagher, Paul Stretch strength data
- `broad_jump.csv`, `power_tests.csv`, `aerobic_tests.csv`, `trunk_endurance.csv`
- All files have `athlete_sport`, `Name`, `recordedDateUtc` columns for filtering

## S&C Diagnostics Canvas Overview

### Tier 1 Tests (Primary)
| Test | Group Chart | Individual Chart | Key Metric | Unit | Source |
|------|-------------|------------------|------------|------|--------|
| IMTP | Ranked Bar | Line + Squad Avg | Relative Peak Force | N/Kg | VALD |
| CMJ | Ranked Bar | Line + Squad Avg | Relative Peak Power | W/Kg | VALD |
| 6 Minute Aerobic | Ranked Bar | Line + Squad Avg | Avg Relative Wattage | W/Kg | Manual |

### Tier 2 Tests (Secondary)
| Test | Group Chart | Individual Chart | Key Metric 1 | Key Metric 2 | Source |
|------|-------------|------------------|--------------|--------------|--------|
| SL ISO Squat & SL IMTP | Ranked side-by-side | Dual line + % diff | Rel Peak Force (N/Kg) | % Difference | VALD |
| Strength RM | Ranked Bar | Multi Line | Rel Strength (RM/BM) | ABS Strength (Kg) | Manual |
| SL CMJ | Ranked side-by-side | Dual line + % diff | Rel Peak Power R&L | Height (cm) | VALD |
| Broad Jump | Ranked Bar | Line + Squad Avg | Distance (cm) | - | Manual |
| 10:5 Hop Test | Ranked Bar | Line + Squad Avg | RSI (absolute) | - | VALD |
| Peak Power (10s) | Ranked Bar | Line + Squad Avg | Peak Rel Wattage | W/Kg | Manual |
| Repeat Power (10x6s) | Ranked Bar | Line + Squad Avg | Peak Rel Wattage | % Fade | Manual |
| Glycolytic Power (3min) | Ranked Bar | Line + Squad Avg | Peak Rel Wattage | W/Kg | Manual |

### Chart Types
1. **Ranked Bar Chart** - Single value tests (IMTP, CMJ, etc.)
2. **Ranked Side-by-Side Bar** - Bilateral/unilateral tests (SL ISO Squat, Nordic L/R)
3. **Stacked Multi-Variable Bar** - Quadrant tests (Trunk, 4-Way Neck, Shoulder IR/ER, Hip Add/Abd)

### S&C Diagnostics Tab Structure (snc_diagnostics.py)
```python
test_tabs = st.tabs([
    "📊 IMTP",           # Ranked Bar - ForceDecks IMTP tests
    "🦘 CMJ",            # Ranked Bar - ForceDecks CMJ tests
    "🦵 SL Tests",       # Side-by-Side - SL ISO Squat, SL IMTP, SL CMJ, SL DJ, SL Jump, SL Hop, Ash Test
    "💪 NordBord",       # Side-by-Side - Nordic hamstring (Left/Right)
    "🏃 10:5 Hop",       # Ranked Bar - HJ, SLHJ, RSHIP, RSKIP, RSAIP
    "🔄 Quadrant Tests", # Stacked - Trunk, 4-Way Neck, Shoulder IR/ER, Hip Add/Abd (ForceFrame)
    "🏋️ Strength RM",    # Ranked Bar - Manual Entry (Back Squat, Bench, Deadlift, etc.)
    "🦘 Broad Jump",     # Ranked Bar - Manual Entry
    "🏃 Fitness Tests",  # Ranked Bar - 6 Min Aerobic, VO2 Max, Yo-Yo, 30-15 IFT (Manual)
    "💥 Plyo Pushup",    # Ranked Bar - ForceDecks PP tests (Upper Body Power)
    "✊ DynaMo",          # Ranked Bar - Grip strength (DynaMo device)
    "⚖️ Balance"         # Ranked Bar - QSB, SLSB for Shooting athletes
])
```

### Reporting Levels
Each test has two views:
- **👥 Group View** - Ranked bar charts with squad average line + benchmark reference
- **🏃 Individual View** - Line charts tracking change over time, multi-athlete selection

**Individual View Pattern (snc_diagnostics.py):**
```python
# Get ALL historical data for trends (not just date-filtered data)
all_data = base_df.copy()  # e.g., all_imtp, all_ppu, grip_df

# Still apply sport/gender filters (but NOT date filter)
if sport != 'All' and 'athlete_sport' in all_data.columns:
    all_data = all_data[all_data['athlete_sport'] == sport]
if gender != 'All' and 'athlete_sex' in all_data.columns:
    all_data = all_data[all_data['athlete_sex'] == gender]

# Use all_data for line chart (shows full history)
fig = create_individual_line_chart(all_data, selected_athletes, ...)
```
