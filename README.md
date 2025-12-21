# Team Saudi VALD Performance Analysis System

**World-class strength and conditioning dashboard for Olympic & Paralympic athletes**

Built with Python, Streamlit, and the VALD Performance API (ForceDecks, ForceFrame, NordBord).

---

## Quick Start

### 1. Initial Setup

```bash
# Install dependencies (if needed)
pip install requests pandas streamlit plotly scipy

# Set credentials in .env file
CLIENT_ID=your_vald_client_id
CLIENT_SECRET=your_vald_secret
TENANT_ID=your_team_id
```

### 2. Fetch Latest Data

```bash
# Full update (ForceDecks, ForceFrame, NordBord)
python vald_production_system.py --update-all

# Or update specific device
python vald_production_system.py --update forcedecks
```

### 3. Integrate Athlete Data

```bash
# Add athlete sports and biometrics
python integrate_athletes.py
```

### 4. Launch Dashboard

```bash
cd dashboard
streamlit run world_class_vald_dashboard.py
```

Dashboard will open at: http://localhost:8501

---

## System Architecture

```
vald/
├── config/
│   └── vald_config.py          # Centralized configuration
├── data/
│   ├── master/                 # Consolidated data files
│   ├── raw/                    # Device-specific raw data
│   └── force_traces/           # Force-time trace data
├── dashboard/
│   ├── world_class_vald_dashboard.py    # Main Streamlit app
│   └── utils/
│       ├── adaptive_ranges.py           # EM adaptive monitoring
│       ├── force_velocity_power.py      # F-V-P profiles
│       └── force_trace_viz.py           # Trace visualization
├── vald_production_system.py   # Core API system (5 actions)
├── interactive_trace_selector.py    # Selective trace fetching
├── integrate_athletes.py       # Quick athlete integration
└── .env                        # Credentials (not in git)
```

---

## Core Features

### Production System (5 Actions)

**Action 1: Update All Data**
```bash
python vald_production_system.py --update-all
```
- Fetches from ForceDecks, ForceFrame, NordBord
- Comprehensive trial data extraction
- Rate limiting (12 calls/5s)
- OAuth token caching

**Action 2: Verify Credentials**
```bash
python vald_production_system.py --verify
```
- Tests API authentication
- Checks token validity
- Validates endpoints

**Action 3: Automated Scheduling**
```bash
python vald_production_system.py --schedule
```
- Daily automated updates
- Configurable schedule (default: 6 AM)
- Background operation

**Action 4: Integrate Athletes**
```bash
python vald_production_system.py --integrate-athletes forcedecks
```
- Fetches 451+ athletes with sports/biometrics
- Matches to test profiles
- Enriches test data

**Action 5: Monitor API Changes**
```bash
python vald_production_system.py --check-updates
```
- Compares with VALD R package
- Detects new endpoints
- Flags breaking changes

### Advanced Analytics

**Force-Velocity-Power Profiles**
- F-V linear regression (F = F0 - slope × V)
- Pmax calculation and optimization
- F-V imbalance detection
- Training recommendations
- Based on: [Optimum Sports Performance](https://optimumsportsperformance.com/blog/r-tips-tricks-force-velocity-power-profile-graphs-in-r-shiny/)

**Adaptive Range Monitoring**
- EM (Expectation-Maximization) algorithm
- Baseline: μ_t = α × x_t + (1 - α) × μ_(t-1)
- Adjustable confidence intervals
- Meaningful change detection (0.6×SD threshold)
- Based on: [Kenny McMillan PhD research](https://bjsm.bmj.com/content/56/24/1451)

**Interactive Trace Selector**
```bash
python interactive_trace_selector.py
```
- Show athlete summary first
- Filter by sport/test type
- Selective trace fetching
- Avoids downloading 3GB+ blindly

### Dashboard (6 Streamlined Tabs)

**Tab 1: Overview Dashboard**
- KPIs (total tests, athletes, sports)
- Recent activity timeline
- Quick athlete search
- System health

**Tab 2: Athlete Profiles**
- Personal bests across test types
- Force-Velocity-Power profiles
- Longitudinal tracking
- Test history
- Download reports (PDF)

**Tab 3: Team & Sport Analysis**
- Sport-specific rankings
- Team vs. elite benchmarks
- Multi-athlete comparisons
- Adaptive sport insights

**Tab 4: Risk & Readiness**
- Asymmetry monitoring (>10% flagged)
- Adaptive ranges (EM approximation)
- Meaningful change detection
- Fatigue indicators
- Readiness dashboard (Green/Amber/Red)

**Tab 5: Testing & Analysis**
- Force-time curves
- Phase detection (eccentric/concentric/landing)
- Multi-trial overlays
- RFD, impulse-momentum metrics
- Export trace data

**Tab 6: Elite Insights**
- Multi-metric EM dashboards
- F-V-P team comparisons
- Advanced statistics (Z-scores, percentiles)
- Research tools
- Data management

---

## Team Saudi Branding

**Colors:**
```python
PRIMARY_TEAL = '#007167'      # Main brand
GOLD_ACCENT = '#a08e66'       # Secondary
DARK_TEAL = '#005a51'         # Darker variant
LIGHT_TEAL = '#8fb7b3'        # Lighter variant
BACKGROUND = '#f8f9fa'        # Clean background
TEXT_DARK = '#1a1a1a'         # High contrast
```

**Dashboard Style:**
- Clean, professional interface
- Team Saudi logo integration
- Consistent color scheme
- High-contrast text for readability

---

## Data Flow

1. **API Fetch** → Raw JSON from VALD endpoints
2. **Trial Extraction** → Comprehensive metrics per trial
3. **Master CSV** → Consolidated device data
4. **Athlete Integration** → Enrich with sports/biometrics
5. **Dashboard** → Interactive Streamlit visualization
6. **Trace Selector** → On-demand force-time data

---

## Performance Metrics

- **Data Pull**: ~1,562 tests in ~60 minutes (with rate limiting)
- **Dashboard Load**: <5 seconds
- **Athlete Integration**: ~451 athletes in ~2 minutes
- **Trace Fetching**: ~5 seconds per test (selective)

---

## Configuration

All settings in `config/vald_config.py`:

```python
@dataclass
class ValdConfig:
    # Authentication
    CLIENT_ID: str
    CLIENT_SECRET: str
    TENANT_ID: str

    # Region
    REGION: str = 'euw'  # Europe West

    # Rate Limiting
    RATE_LIMIT_CALLS: int = 12
    RATE_LIMIT_WINDOW: float = 5.0

    # Feature Flags
    ENABLE_FORCE_TRACES: bool = True
    ENABLE_FV_PROFILES: bool = True
    ENABLE_ADAPTIVE_RANGES: bool = True
```

---

## Troubleshooting

### Authentication Errors

**Problem**: `401 Unauthorized`

**Solution**:
1. Check `.env` credentials
2. Verify region setting (euw/use/apse)
3. Test with: `python vald_production_system.py --verify`

### Rate Limiting

**Problem**: `429 Too Many Requests`

**Solution**:
- System auto-handles with queue-based limiting
- Default: 12 calls per 5 seconds
- Adjust in `config/vald_config.py` if needed

### Missing Data

**Problem**: No tests in dashboard

**Solution**:
1. Run fresh data pull: `python vald_production_system.py --update-all`
2. Check date range in config (default: 2024-01-01)
3. Verify TENANT_ID matches your team

### Force Traces Not Loading

**Problem**: Trace plots empty

**Solution**:
1. Use Interactive Trace Selector first: `python interactive_trace_selector.py`
2. Fetches traces selectively (not all 1,562 tests)
3. Check `data/force_traces/` for JSON files

---

## Research & Literature

This system implements evidence-based sports science methods:

1. **Force-Velocity-Power Profiles**
   - Samozino et al. (2008) - "Optimal force-velocity profile"
   - Morin & Samozino (2016) - "Interpreting power-force-velocity profiles"
   - Optimum Sports Performance resources

2. **Adaptive Range Monitoring**
   - Kenny McMillan PhD (2022) - "Comparison of methods to generate adaptive reference ranges"
   - Patrick Ward (Nike) - Meaningful change detection (0.6×SD)
   - EM algorithm for baseline estimation

3. **CMJ Testing Protocols**
   - VALD ForceDecks best practices
   - Bilateral asymmetry thresholds (>10%)
   - RSI-modified reliability

---

## Daily Workflow

**Morning (Automated):**
```bash
# Scheduled at 6 AM
python vald_production_system.py --update-all
python integrate_athletes.py
```

**During Training:**
- Launch dashboard: `streamlit run dashboard/world_class_vald_dashboard.py`
- View athlete profiles
- Check readiness flags
- Monitor asymmetries

**Post-Testing:**
- Fetch new traces: `python interactive_trace_selector.py`
- Analyze force-time curves
- Generate athlete reports

**Weekly Review:**
- Check for API updates: `python vald_production_system.py --check-updates`
- Review adaptive ranges
- Export data for analysis

---

## File Locations

**Master Data:**
- `data/master/forcedecks_master.csv` - All ForceDecks tests
- `data/master/forcedecks_allsports_with_athletes.csv` - Enriched with athlete data
- `data/master/forceframe_master.csv` - All ForceFrame tests
- `data/master/nordbord_master.csv` - All NordBord tests

**Raw Data:**
- `data/raw/forcedecks/` - JSON files per date
- `data/raw/forceframe/` - JSON files per date
- `data/raw/nordbord/` - JSON files per date

**Force Traces:**
- `data/force_traces/` - Test-specific trace JSON

**Logs:**
- `logs/vald_system.log` - System operations
- `logs/api_changes.log` - API update checks

---

## API Endpoints

**ForceDecks:**
```
https://prd-euw-api-extforcedecks.valdperformance.com/
├── v2019q3/teams/{TENANT_ID}/tests
├── v2019q3/teams/{TENANT_ID}/tests/{testId}/trials
└── v2019q3/teams/{TENANT_ID}/tests/{testId}/trials/{trialId}/trace
```

**Athletes:**
```
https://prd-euw-api-athletes.valdperformance.com/
└── v2023q2/teams/{TENANT_ID}/athletes
```

**Security (OAuth):**
```
https://security.valdperformance.com/
└── connect/token
```

---

## Support & Documentation

- **VALD Official Docs**: [VALD Performance Documentation](https://valdperformance.com/developers/)
- **VALD R Package**: [GitHub - valdperformance/vald-hub-data-extractor](https://github.com/valdperformance/vald-hub-data-extractor)
- **Internal Guide**: `QUICK_START_GUIDE.md`
- **Trace Selector Guide**: `TRACE_SELECTOR_GUIDE.md`
- **Dashboard Plan**: `DASHBOARD_STREAMLINE_PLAN.md`

---

## Version History

**v2.0 (Current)**
- Unified production system with 5 core actions
- Interactive trace selector
- F-V-P profile analysis
- Adaptive range monitoring (EM approximation)
- Dashboard streamlined to 6 tabs
- Team Saudi branding integrated

**v1.0 (Legacy)**
- Basic API fetching
- 11-tab dashboard
- Manual athlete matching

---

## License

Team Saudi Olympic & Paralympic Committee - Internal Use Only

---

**Built with excellence for Saudi Arabia's Olympic champions.**
