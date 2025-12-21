# Team Saudi Streamlit Theme

A reusable Streamlit theme and component library for Team Saudi applications.

## Quick Start

1. Copy the `.streamlit` folder to your project root
2. Copy the logo file (`Saudi logo.png` or `team_saudi_logo.png`)
3. Import the theme components in your app

## Folder Structure

```
your_project/
├── .streamlit/
│   ├── config.toml          # Theme colors and settings
│   └── secrets.toml         # API credentials (create from template)
├── Saudi logo.png           # Team logo
├── requirements.txt         # Dependencies
└── your_app.py              # Your Streamlit app
```

## Theme Colors

| Color | Hex | Usage |
|-------|-----|-------|
| Saudi Green (Primary) | `#1D4D3B` | Headers, buttons, primary charts |
| Saudi Green (Light) | `#2A6A50` | Hover states, secondary elements |
| Saudi Green (Dark) | `#153829` | Sidebar, dark backgrounds |
| Saudi Gold | `#A08E66` | Accents, borders |

## Components

### Header with Logo

```python
import streamlit as st
import base64
import os

def render_header():
    """Render Team Saudi header with logo."""
    logo_path = "Saudi logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        st.markdown(f'''
        <div style="background: linear-gradient(135deg, #1D4D3B 0%, #153829 100%);
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;
                    border-bottom: 4px solid #A08E66;">
            <div style="display: flex; align-items: center; gap: 1.5rem;">
                <img src="data:image/png;base64,{logo_data}"
                     style="height: 80px; width: auto;">
                <div>
                    <h1 style="color: white; margin: 0; font-size: 2rem;">
                        Team Saudi Performance Dashboard
                    </h1>
                    <p style="color: #A08E66; margin: 0.5rem 0 0 0;">
                        Powered by VALD Performance
                    </p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
```

### Sidebar Logo

```python
def render_sidebar_logo():
    """Render logo in sidebar."""
    logo_path = "Saudi logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        st.sidebar.markdown(f'''
        <div style="text-align: center; padding: 1.5rem 1rem 2rem 1rem;
                    border-bottom: 3px solid #A08E66; margin-bottom: 1.5rem;">
            <img src="data:image/png;base64,{logo_data}"
                 style="width: 100%; max-width: 200px; height: auto;">
        </div>
        ''', unsafe_allow_html=True)
```

### Status Colors

```python
# Success (green)
st.success("✅ Operation completed successfully")

# Warning (yellow)
st.warning("⚠️ Attention required")

# Error (red)
st.error("🔴 Error occurred")

# Info (blue)
st.info("ℹ️ Information message")
```

### Risk Indicators

```python
def risk_badge(level):
    """Return colored risk badge."""
    colors = {
        'low': ('🟢', '#28a745'),
        'moderate': ('🟡', '#ffc107'),
        'high': ('🔴', '#dc3545')
    }
    icon, color = colors.get(level, ('⚪', '#6c757d'))
    return f"{icon} {level.title()} Risk"
```

## Streamlit Cloud Deployment

### secrets.toml

Create `.streamlit/secrets.toml` (never commit to git):

```toml
[vald]
MANUAL_TOKEN = "your_api_token"
TENANT_ID = "your_tenant_id"
VALD_REGION = "euw"
```

### requirements.txt

```
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.17.0
numpy>=1.24.0
requests>=2.31.0
python-dateutil>=2.8.2
scipy>=1.11.0
```

## File Checklist for Deployment

Essential files:
- [ ] `.streamlit/config.toml` - Theme configuration
- [ ] `requirements.txt` - Python dependencies
- [ ] `your_app.py` - Main application
- [ ] Logo file (PNG)

On Streamlit Cloud:
- [ ] Add secrets in dashboard settings
- [ ] Connect GitHub repository

## Support

Saudi Olympic & Paralympic Committee
https://olympic.sa/team-saudi/
