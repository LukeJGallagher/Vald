# VALD Dashboard - Complete Tab Reference Guide

This document provides comprehensive documentation of all dashboard tabs, their functionality, data sources, and API integration. Use this as a reference for restoring removed tabs in the future.

---

## Overview

**Dashboard File:** `dashboard/world_class_vald_dashboard.py`
**Version:** 3.0
**Data Sources:** ForceDecks, ForceFrame, NordBord (via VALD API or CSV files)

---

## Data Loading Architecture

### Priority Order:
1. **Local CSV files** - Development mode
2. **Private GitHub repo** (`vald-data`) - Streamlit Cloud with historical data
3. **VALD API direct fetch** - Fallback (90 days max)

### Key Files:
- `dashboard/utils/data_loader.py` - Main data loading logic
- `dashboard/config/sports_config.py` - Sport benchmarks and metrics
- `dashboard/utils/advanced_viz.py` - Advanced visualizations
- `dashboard/utils/force_trace_viz.py` - Force trace analysis
- `dashboard/utils/test_type_modules.py` - CMJ, IMTP, Throws modules
- `dashboard/utils/advanced_analysis.py` - Elite insights features
- `dashboard/utils/sport_reports.py` - Sport-specific group/individual reports

### API Endpoints:
```
OAuth:      https://security.valdperformance.com/connect/token
ForceDecks: https://prd-{REGION}-api-extforcedecks.valdperformance.com/
ForceFrame: https://prd-{REGION}-api-externalforceframe.valdperformance.com/
NordBord:   https://prd-{REGION}-api-externalnordbord.valdperformance.com/
Profiles:   https://prd-{REGION}-api-externalprofile.valdperformance.com/
```

---

## Tab Index Reference

Original tab order (before removal):
```python
tabs = st.tabs([
    "Home",        # [0]  - KEEP
    "Reports",     # [1]  - KEEP
    "ForceFrame",  # [2]  - KEEP
    "NordBord",    # [3]  - KEEP
    "Athlete",     # [4]  - REMOVED
    "CMJ",         # [5]  - REMOVED
    "Iso",         # [6]  - REMOVED
    "Throws",      # [7]  - KEEP
    "Trace",       # [8]  - KEEP
    "Sport",       # [9]  - REMOVED
    "Risk",        # [10] - REMOVED
    "Compare",     # [11] - REMOVED
    "Progress",    # [12] - REMOVED
    "Rank",        # [13] - REMOVED
    "Adv",         # [14] - REMOVED
    "Insights",    # [15] - REMOVED
    "Data"         # [16] - KEEP
])
```

---

## REMOVED TABS - Full Code Reference

### 1. Athlete Tab (tabs[4]) - Lines ~1888-2031

**Purpose:** Individual athlete deep-dive with test history, performance summary, and trends.

**Features:**
- Athlete selector dropdown
- Recent test summary table
- Performance metrics cards (Jump Height, Power, Force, RSI)
- Trend charts over time
- Test history table

**Data Requirements:**
- `filtered_df` - ForceDecks data
- Columns: `Name`, `athlete_sport`, `testType`, `recordedDateUtc`
- Performance columns from `get_metric_column()`

**Key Code Pattern:**
```python
with tabs[4]:
    st.markdown("## Athlete Profile")
    selected_athlete = st.selectbox("Select Athlete:", athletes)
    athlete_df = filtered_df[filtered_df['Name'] == selected_athlete]
    # Display summary cards, charts, tables
```

---

### 2. CMJ Tab (tabs[5]) - Lines ~2032-2071

**Purpose:** Counter Movement Jump deep analysis using CMJAnalysisModule.

**Features:**
- Jump height trends
- Power analysis
- RSI modified calculations
- Asymmetry tracking
- Force trace visualization

**Data Requirements:**
- `filtered_df` filtered to CMJ test types
- Requires `TEST_TYPE_MODULES_AVAILABLE = True`
- Uses `CMJAnalysisModule` from `utils/test_type_modules.py`

**Key Code Pattern:**
```python
with tabs[5]:
    st.markdown("## CMJ Analysis")
    if TEST_TYPE_MODULES_AVAILABLE:
        cmj_df = filtered_df[filtered_df['testType'].str.contains('CMJ|Counter', case=False, na=False)]
        display_test_type_module(CMJAnalysisModule, cmj_df, benchmarks)
```

---

### 3. Iso Tab (tabs[6]) - Lines ~2072-2112

**Purpose:** Isometric testing analysis (IMTP, single-leg, double-leg).

**Features:**
- Sub-tabs: Single Leg, Double Leg
- Peak force analysis
- RFD (Rate of Force Development)
- Asymmetry metrics
- Force-time curves

**Data Requirements:**
- `filtered_df` filtered to Isometric test types
- Uses `IsometricSingleLegModule`, `IsometricDoubleLegModule`

**Key Code Pattern:**
```python
with tabs[6]:
    st.markdown("## Isometric Analysis")
    iso_df = filtered_df[filtered_df['testType'].str.contains('IMTP|ISOT|Isometric', case=False, na=False)]
    iso_subtabs = st.tabs(["Single Leg", "Double Leg"])
    # Display modules in each subtab
```

---

### 4. Sport Tab (tabs[9]) - Lines ~2156-2237

**Purpose:** Sport-specific group reports with benchmarks.

**Features:**
- Sport selector
- Group Report v1/v2/v3 variants
- Individual athlete reports
- Sport-specific benchmarks from `sports_config.py`
- Benchmark zones visualization

**Data Requirements:**
- `filtered_df`, `filtered_forceframe`, `filtered_nordbord`
- Sport benchmarks from `SPORT_BENCHMARKS`
- Uses `sport_reports.py` module

**Key Code Pattern:**
```python
with tabs[9]:
    st.markdown("## Sport-Specific Reports")
    from utils.sport_reports import create_group_report, create_individual_report
    sport = st.selectbox("Select Sport:", available_sports)
    report_tabs = st.tabs(["Group Report", "Individual Report"])
    # Render reports with benchmarks
```

---

### 5. Risk Tab (tabs[10]) - Lines ~2238-2632

**Purpose:** Risk assessment and injury prevention dashboard.

**Features:**
- Risk score calculation
- Asymmetry-based flagging (>10% threshold)
- NordBord injury threshold (337N)
- ForceFrame imbalance detection
- Risk priority matrix
- Athlete risk cards with RAG status

**Data Requirements:**
- All device data
- `RISK_THRESHOLDS` from config
- Asymmetry calculations

**Key Code Pattern:**
```python
with tabs[10]:
    st.markdown("## Risk Assessment")
    # Calculate risk scores based on:
    # - Asymmetry > 10%
    # - NordBord < 337N
    # - ForceFrame imbalances
    # Display risk matrix and athlete cards
```

---

### 6. Compare Tab (tabs[11]) - Lines ~2633-2706

**Purpose:** Multi-athlete comparison tool.

**Features:**
- Select 2-4 athletes for comparison
- Side-by-side metric comparison
- Radar charts overlaid
- Bar chart comparisons
- Performance table

**Data Requirements:**
- `filtered_df` with multiple athletes
- Metric columns for comparison

**Key Code Pattern:**
```python
with tabs[11]:
    st.markdown("## Athlete Comparison")
    compare_athletes = st.multiselect("Select athletes to compare:", athletes, max_selections=4)
    # Create comparison visualizations
```

---

### 7. Progress Tab (tabs[12]) - Lines ~2707-2858

**Purpose:** Longitudinal progress tracking.

**Features:**
- Time period selection (30d, 90d, 180d, 1y)
- Trend analysis with rolling averages
- Change detection (improvement/decline)
- Progress sparklines
- Milestone tracking

**Data Requirements:**
- Historical test data
- Date columns for time-series analysis

**Key Code Pattern:**
```python
with tabs[12]:
    st.markdown("## Progress Tracking")
    time_period = st.selectbox("Time Period:", ["30 Days", "90 Days", "180 Days", "1 Year"])
    # Calculate trends and display progress charts
```

---

### 8. Rank Tab (tabs[13]) - Lines ~2859-3031

**Purpose:** Athlete rankings and percentile analysis.

**Features:**
- Metric-specific rankings
- Percentile calculations
- Sport-relative rankings
- Leaderboards with medals
- Ranking trends over time

**Data Requirements:**
- Latest test per athlete
- Benchmark data for percentiles
- Uses `create_labeled_ranking()` from advanced_viz

**Key Code Pattern:**
```python
with tabs[13]:
    st.markdown("## Rankings")
    rank_metric = st.selectbox("Rank by:", available_metrics)
    rankings = calculate_rankings(filtered_df, rank_metric)
    # Display leaderboard with medals
```

---

### 9. Advanced Tab (tabs[14]) - Lines ~4048-4371

**Purpose:** Advanced analytics and statistical analysis.

**Features:**
- Sub-tabs: Quadrant, Parallel, Violin, Reliability, Best-of-Day
- Quadrant plot (power vs technique)
- Parallel coordinates (multi-metric)
- Violin plots (distribution)
- Reliability metrics (CV%, ICC, TEM)
- Best-of-day trend analysis

**Data Requirements:**
- All ForceDecks data
- Requires `ADVANCED_VIZ_AVAILABLE = True`
- Uses `utils/advanced_viz.py` functions

**Key Code Pattern:**
```python
with tabs[14]:
    st.markdown("## Advanced Analytics")
    adv_subtabs = st.tabs(["Quadrant", "Parallel", "Violin", "Reliability", "Best-of-Day"])
    # Render each analysis type
```

---

### 10. Insights Tab (tabs[15]) - Lines ~4372-4766

**Purpose:** Elite-level insights with statistical rigor.

**Features:**
- Sub-tabs: Asymmetry, Meaningful Change, Normative, Z-Score
- Asymmetry circle plot (bilateral comparison)
- Meaningful change detection (vs TEM/SWC)
- Normative benchmarks with percentiles
- Z-score distribution analysis
- Bilateral force curve overlay

**Data Requirements:**
- All ForceDecks data
- Requires `ELITE_INSIGHTS_AVAILABLE = True`
- Uses `utils/advanced_analysis.py` functions

**Key Code Pattern:**
```python
with tabs[15]:
    st.markdown("## Elite Insights")
    insight_tabs = st.tabs(["Asymmetry", "Meaningful Change", "Normative", "Z-Score"])
    # Render each insight type
```

---

## SIDEBAR FILTERS (REMOVED)

**Location:** Lines ~1682-1789

**Components:**
1. **Sport Filter** - Multi-select for sports
2. **Athlete Filter** - Multi-select based on sport selection
3. **Test Type Filter** - Multi-select for test types
4. **Date Range Filter** - Date input for start/end dates

**Filter Logic:**
```python
# Sport filter
selected_sports = st.sidebar.multiselect("Select Sports:", available_sports)

# Athlete filter (filtered by sport)
selected_athletes = st.sidebar.multiselect("Select Athletes:", filtered_athletes)

# Test type filter
selected_test_types = st.sidebar.multiselect("Select Test Types:", available_test_types)

# Date range filter
date_range = st.sidebar.date_input("Date Range:", value=(min_date, max_date))

# Apply filters to dataframes
filtered_df = apply_filters(df, selected_sports, selected_athletes, selected_test_types, date_range)
```

---

## KEPT TABS

### Home (tabs[0])
- Overview dashboard with KPIs
- Recent testing activity
- Test distribution charts

### Reports (tabs[1])
- Sport Reports integration
- Group and Individual reports
- Benchmark visualizations

### ForceFrame (tabs[2])
- Body region analysis
- Shoulder/Hip testing
- Asymmetry dashboard
- Progression tracking

### NordBord (tabs[3])
- Hamstring strength analysis
- 337N injury threshold
- Bilateral comparison
- Progression tracking

### Throws (tabs[7])
- Throws training module
- Uses `ThrowsTrainingModule`

### Trace (tabs[8])
- Force trace visualization
- Phase detection
- Multi-trial overlay
- Athlete comparison

### Data (tabs[16])
- Raw data export
- Refresh data from API
- GitHub sync status

---

## Restoring Removed Tabs

To restore a removed tab:

1. **Find the tab code** in this document or git history
2. **Add the tab name** to the `st.tabs([...])` list
3. **Add the tab content** with `with tabs[INDEX]:`
4. **Verify imports** - ensure required modules are imported
5. **Test locally** before deploying

### Import Requirements by Tab:

| Tab | Required Imports |
|-----|------------------|
| Athlete | Basic (no special imports) |
| CMJ | `TEST_TYPE_MODULES_AVAILABLE`, `CMJAnalysisModule` |
| Iso | `TEST_TYPE_MODULES_AVAILABLE`, `IsometricSingleLegModule`, `IsometricDoubleLegModule` |
| Sport | `sport_reports` module |
| Risk | `RISK_THRESHOLDS` from config |
| Compare | Basic (no special imports) |
| Progress | Basic (no special imports) |
| Rank | `ADVANCED_VIZ_AVAILABLE`, `create_labeled_ranking` |
| Advanced | `ADVANCED_VIZ_AVAILABLE`, `advanced_viz` functions |
| Insights | `ELITE_INSIGHTS_AVAILABLE`, `advanced_analysis` functions |

---

## API Integration Details

### Authentication:
```python
# OAuth2 Client Credentials Flow
response = requests.post(
    'https://security.valdperformance.com/connect/token',
    data={
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'scope': 'forcedecks forceframe nordbord athletes'
    }
)
token = response.json()['access_token']
```

### Fetching Test Data:
```python
# ForceDecks example
url = f'https://prd-{region}-api-extforcedecks.valdperformance.com/tests'
headers = {'Authorization': f'Bearer {token}'}
params = {'tenantId': tenant_id, 'modifiedFromUtc': from_date}
response = requests.get(url, headers=headers, params=params)
tests = response.json()
```

### Pagination:
- **ForceDecks/ForceFrame**: Cursor-based (`modifiedFromUtc`)
- **NordBord**: Page-based (`Page` parameter)

### Rate Limiting:
- 12 calls per 5 seconds
- Handled in `vald_production_system.py`

---

## Configuration Files

### sports_config.py
- `SPORT_BENCHMARKS` - Per-sport benchmark values
- `RISK_THRESHOLDS` - Risk assessment thresholds
- `TEST_TYPE_CONFIG` - Test type definitions
- `METRIC_DEFINITIONS` - Metric metadata

### Streamlit Secrets (for Cloud):
```toml
[github]
GITHUB_TOKEN = "github_pat_..."
DATA_REPO = "username/vald-data"

[vald]
CLIENT_ID = "..."
CLIENT_SECRET = "..."
TENANT_ID = "..."
VALD_REGION = "euw"
```

---

## Color Scheme

| Color | Hex | Usage |
|-------|-----|-------|
| Primary Teal | `#007167` | Main accent, good performance |
| Dark Teal | `#005a51` | Headers, emphasis |
| Coral | `#FF6B6B` | Secondary (bilateral Right) |
| Light Teal | `#4DB6AC` | Average performance |
| Gray-Blue | `#78909C` | Needs improvement |
| Gold (Legacy) | `#a08e66` | Brand accent (removed from V3) |

---

*Document Created: January 2026*
*For restoration assistance, reference git history or contact the Performance Analysis Team.*
