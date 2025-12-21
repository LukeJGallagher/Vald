# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Team Saudi VALD Performance Analysis System - A Python-based strength and conditioning analytics platform for Olympic & Paralympic athletes. Fetches data from VALD's API (ForceDecks, ForceFrame, NordBord) and provides advanced analytics through a Streamlit dashboard.

## Project Structure

```
Vald/
├── dashboard/                    # Streamlit app (deployment target)
│   ├── world_class_vald_dashboard.py
│   ├── requirements.txt
│   ├── utils/                    # Analytics modules
│   ├── .streamlit/               # Streamlit config
│   └── TEAM_SAUDI_THEME/         # Branding assets
├── config/
│   └── vald_config.py            # Centralized configuration
├── data/
│   ├── master_files/             # Consolidated CSV data
│   ├── cache/                    # Token cache
│   └── backups/                  # Data backups
├── logs/                         # System logs
├── vald_production_system.py     # Core API system
├── integrate_athletes.py         # Athlete data enrichment
├── interactive_trace_selector.py # Selective trace fetching
└── README.md
```

## Common Commands

### Data Fetching
```bash
# Full update from all VALD devices
python vald_production_system.py --update-all

# Update specific device
python vald_production_system.py --update forcedecks

# Verify API credentials
python vald_production_system.py --verify

# Integrate athlete data (sports/biometrics)
python integrate_athletes.py

# Check for VALD API changes
python vald_production_system.py --check-updates
```

### Dashboard
```bash
cd dashboard
pip install -r requirements.txt
streamlit run world_class_vald_dashboard.py
```
Dashboard runs at http://localhost:8501

### Interactive Force Trace Fetching
```bash
python interactive_trace_selector.py
```
Selective trace fetching - avoids downloading all traces (3GB+).

## Architecture

### Core Components

**vald_production_system.py** - Main orchestrator with 5 actions: data pull, credential verification, scheduling, athlete integration, and API monitoring. Contains key classes:
- `OAuthTokenManager` - Token caching with auto-refresh
- `RateLimiter` - Queue-based rate limiting (12 calls/5 seconds)
- `UnifiedValdSystem` - Coordinates all operations
- Device handlers: `ForceDecksHandler`, `ForceFrameHandler`, `NordBordHandler`

**config/vald_config.py** - Centralized configuration via `ValdConfig` dataclass. Loads from `.env` file. Contains:
- API credentials and region settings
- Rate limiting parameters
- Directory paths
- Device-specific settings (`DEVICE_CONFIG`)
- Test type metric definitions (`TEST_TYPE_METRICS`)

**dashboard/world_class_vald_dashboard.py** - 16-tab Streamlit app for data visualization and analysis.

**dashboard/utils/** - Modular analytics:
- `adaptive_ranges.py` - EM algorithm for meaningful change detection (Kenny McMillan method)
- `force_velocity_power.py` - F-V-P profile calculations (F = F0 - slope × V)
- `force_trace_viz.py` - Force-time curve visualization with phase detection
- `data_loader.py` - Data loading and asymmetry calculations
- `advanced_analysis.py` - Elite insights (Z-scores, percentiles)

### Data Flow
```
VALD API → vald_production_system.py → data/master_files/*.csv → dashboard/
```

### Key Data Files
- `data/master_files/forcedecks_allsports_with_athletes.csv` - Enriched ForceDecks data
- `data/master_files/forceframe_allsports_with_athletes.csv` - ForceFrame data
- `data/master_files/nordbord_allsports_with_athletes.csv` - NordBord data
- `data/force_traces/` - Test-specific force-time trace JSON files
- `data/cache/token_cache.pkl` - OAuth token persistence

## Environment Setup

Create `.env` in project root:
```
CLIENT_ID=your_vald_client_id
CLIENT_SECRET=your_vald_secret
TENANT_ID=your_team_id
VALD_REGION=euw
```

Regions: `euw` (Europe West), `use` (US East), `apse` (Asia Pacific)

## API Endpoints

ForceDecks: `https://prd-{REGION}-api-extforcedecks.valdperformance.com/`
ForceFrame: `https://prd-{REGION}-api-externalforceframe.valdperformance.com/`
NordBord: `https://prd-{REGION}-api-externalnordbord.valdperformance.com/`
Athletes: `https://prd-{REGION}-api-athletes.valdperformance.com/`
OAuth: `https://security.valdperformance.com/connect/token`

## Key Analytics Methods

**Asymmetry Monitoring**: Flags bilateral imbalances >10%

**Adaptive Ranges**: EM-based baseline: μ_t = α × x_t + (1 - α) × μ_(t-1), with meaningful change threshold at 0.6×SD

**F-V-P Profiles**: Linear regression (F = F0 - slope × V) for power optimization

## Branding

Team Saudi colors:
- Primary Teal: `#007167`
- Gold Accent: `#a08e66`
- Dark Teal: `#005a51`
