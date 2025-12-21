# Team Saudi VALD Performance Dashboard

**Saudi National Team - World-Class Strength & Conditioning Analysis**

Version 3.0 | Built with Streamlit + Plotly

---

## Quick Start

### Local Development
```bash
pip install -r requirements.txt
streamlit run world_class_vald_dashboard.py
```

### Streamlit Cloud Deployment
1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set the main file as `world_class_vald_dashboard.py`
5. Add your VALD API credentials in **Secrets** (see below)

---

## Features

### Devices Supported
- **ForceDecks** - Jump and force plate analysis (CMJ, SJ, IMTP, DJ)
- **ForceFrame** - Isometric strength testing (19 joint positions)
- **NordBord** - Nordic hamstring strength assessment

### Dashboard Tabs
1. **Overview** - KPIs, recent activity, test distribution
2. **Athlete** - Individual deep dive with progress tracking
3. **CMJ** - Countermovement jump analysis
4. **Isometric** - Single/double leg isometric testing
5. **Throws** - Medicine ball throw analysis
6. **Force Trace** - Raw force-time curve visualization
7. **Sport** - Sport-specific benchmarks and context
8. **Risk** - Risk and readiness assessment
9. **Compare** - Multi-athlete radar chart comparison
10. **Progress** - Longitudinal tracking
11. **Rankings** - Percentile rankings
12. **Advanced** - Quadrant analysis, parallel coordinates
13. **Insights** - Elite-level metrics (asymmetry, meaningful change)
14. **Data** - Raw data table with export
15. **ForceFrame** - ForceFrame-specific analysis
16. **NordBord** - NordBord-specific analysis

### Sports Supported (15+)
Athletics, Fencing, Rowing, Swimming, Para Swimming, Weightlifting, Wrestling, Judo, Jiu-Jitsu, Shooting, Snow Sports, and more

---

## Configuration

### Streamlit Secrets (for Cloud Deployment)

Add the following to your Streamlit Cloud app secrets:

```toml
[vald]
CLIENT_ID = "your_client_id"
CLIENT_SECRET = "your_client_secret"
TENANT_ID = "your_tenant_id"
VALD_REGION = "euw"
```

### Local Development (.env file)

Create a `.env` file (not committed to git):
```
CLIENT_ID=your_client_id
CLIENT_SECRET=your_client_secret
TENANT_ID=your_tenant_id
VALD_REGION=euw
```

---

## Data Refresh

### Option 1: Upload CSV
Use the Data Management panel in the sidebar to upload CSV files exported from VALD Hub.

### Option 2: Live API Refresh
1. Configure your VALD API credentials in Streamlit Secrets
2. Use the "Fetch from API" button in the Data Management panel
3. Select the device (ForceDecks, ForceFrame, or NordBord)
4. Data will be refreshed from the VALD API (last 35 days)

---

## Project Structure

```
dashboard/
├── world_class_vald_dashboard.py  # Main dashboard
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
├── .streamlit/
│   ├── config.toml               # Streamlit theme config
│   └── secrets.toml.example      # Secrets template
├── config/
│   └── sports_config.py          # Sports benchmarks
├── utils/
│   ├── data_loader.py            # Data loading utilities
│   ├── advanced_viz.py           # Advanced visualizations
│   ├── force_trace_viz.py        # Force trace analysis
│   ├── test_type_modules.py      # Test-specific modules
│   └── advanced_analysis.py      # Elite insights
├── data/                          # Data files (gitignored)
│   ├── forcedecks_allsports_with_athletes.csv
│   ├── forceframe_allsports.csv
│   └── nordbord_allsports.csv
└── team_saudi_logo.png           # Logo assets
```

---

## Technology Stack

- **Streamlit** - Dashboard framework
- **Plotly** - Interactive visualizations
- **Pandas** - Data processing
- **SciPy/Statsmodels** - Statistical analysis
- **Requests** - VALD API integration

---

## Support

For issues or feature requests, contact the Performance Analysis Team.

---

© 2025 Saudi Olympic & Paralympic Committee | Performance Analysis Dashboard v3.0
