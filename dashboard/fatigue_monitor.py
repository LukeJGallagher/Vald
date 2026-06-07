"""
CMJ Modified Fatigue Monitor — Team Saudi VALD Dashboard
=========================================================

Standalone Streamlit app for S&C coaches. Tracks Peak Power (W/kg) from the
modified CMJ protocol (hands-on-hips, no arm swing) against three concurrent
Smallest Worthwhile Change benchmarks over a rolling 7-day baseline.

Run locally:
    cd dashboard && streamlit run fatigue_monitor.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make sibling utils importable when run as `streamlit run fatigue_monitor.py`
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import load_vald_data, refresh_and_save_data
from utils.snc_diagnostics import (
    get_persisted_athlete_selection,
    get_persisted_selectbox_index,
)


# ── Team Saudi Brand Colours (May 2024 Guidelines) ──────────────────────────
ELITE_GREEN = "#235036"
ENABLER_GREEN = "#69c399"
DISCIPLINE_GREEN = "#18342a"
STAMINA_GREEN = "#c3d9d1"
VICTORY_GOLD = "#ebce83"
VISIONARY_LAVENDER = "#9263aa"
STATUS_DANGER = "#dc3545"
STATUS_WARNING = "#FFB800"

METRIC_COL = "BODYMASS_RELATIVE_TAKEOFF_POWER"   # Peak Power, W/kg
METRIC_LABEL = "Peak Power (W/kg)"
BASELINE_DAYS = 7
SWC_FACTOR = 0.2          # Hopkins small effect
CV_FLAG_MULTIPLIER = 1.5  # changes > 1.5 × baseline SD count as meaningful


# ── Page Setup ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CMJ Modified Fatigue Monitor — Team Saudi",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: #ffffff; }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {ELITE_GREEN} 0%, {DISCIPLINE_GREEN} 100%);
        }}
        [data-testid="stSidebar"] * {{ color: white !important; }}
        h1, h2, h3, h4 {{ color: {ELITE_GREEN}; font-family: Arial, sans-serif; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, {ELITE_GREEN} 0%, {DISCIPLINE_GREEN} 100%);
         padding: 1.5rem 1.75rem; border-radius: 10px; margin-bottom: 1.25rem;
         border-left: 5px solid {VICTORY_GOLD};
         box-shadow: 0 4px 12px rgba(35, 80, 54, 0.25);">
        <h2 style="color: white; margin: 0; font-size: 1.7rem;">
            CMJ Modified — Fatigue Monitor
        </h2>
        <p style="color: {VICTORY_GOLD}; margin: 0.4rem 0 0 0; font-size: 1rem;">
            Peak Power tracking vs. Smallest Worthwhile Change (rolling {BASELINE_DAYS}-day baseline)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Data Load ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner="Loading CMJ Modified data…")
def load_cmj_data() -> pd.DataFrame:
    df = load_vald_data("forcedecks")
    if df is None or df.empty:
        return pd.DataFrame()

    df = df[df["testType"] == "CMJ"].copy()
    if df.empty or METRIC_COL not in df.columns:
        return pd.DataFrame()

    df[METRIC_COL] = pd.to_numeric(df[METRIC_COL], errors="coerce")
    df = df.dropna(subset=[METRIC_COL])

    df["recordedDateUtc"] = pd.to_datetime(df["recordedDateUtc"], errors="coerce", utc=True)
    df = df.dropna(subset=["recordedDateUtc"])
    df["date"] = df["recordedDateUtc"].dt.tz_convert(None).dt.normalize()

    if "Name" not in df.columns or df["Name"].notna().sum() < len(df) * 0.5:
        df["Name"] = df.get("full_name", pd.Series(index=df.index, dtype=object))
    df = df.dropna(subset=["Name"])

    if "athlete_sport" not in df.columns:
        df["athlete_sport"] = "Unknown"
    df["athlete_sport"] = df["athlete_sport"].fillna("Unknown")

    return df[["Name", "athlete_sport", "date", METRIC_COL]].rename(
        columns={METRIC_COL: "peak_power"}
    )


df = load_cmj_data()
if df.empty:
    st.error(
        "No CMJ Modified data available. Run `python scripts/local_sync.py` "
        "then copy the CSV into `dashboard/data/` and reload."
    )
    st.stop()


# ── Sidebar — Live Refresh + Filters ────────────────────────────────────────
st.sidebar.markdown(
    f"<h3 style='color: {VICTORY_GOLD}; margin-top: 0;'>Live data</h3>",
    unsafe_allow_html=True,
)

last_refresh = st.session_state.get("fm_last_refresh", "Never (using cached load)")
st.sidebar.markdown(
    f"<p style='color:rgba(255,255,255,0.78); font-size:0.78rem; margin:0 0 0.5rem 0;'>"
    f"Latest test in data: <b>{df['date'].max().strftime('%Y-%m-%d')}</b><br>"
    f"Last manual refresh: <b>{last_refresh}</b></p>",
    unsafe_allow_html=True,
)

if st.sidebar.button(
    "Refresh from VALD API",
    type="primary",
    use_container_width=True,
    key="fm_refresh_btn",
    help="Pull the latest ForceDecks tests from VALD.",
):
    with st.spinner("Pulling latest CMJ data from VALD..."):
        try:
            fresh_df, ok = refresh_and_save_data("forcedecks")
            load_cmj_data.clear()
            st.session_state["fm_last_refresh"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            if ok and not fresh_df.empty:
                st.sidebar.success(f"Refreshed: {len(fresh_df)} ForceDecks tests")
            elif not fresh_df.empty:
                st.sidebar.warning(f"Fetched {len(fresh_df)} but couldn't push to repo")
            else:
                st.sidebar.error("No data returned from VALD — check credentials.")
        except Exception as exc:
            st.sidebar.error(f"Refresh failed: {exc}")
    st.rerun()

st.sidebar.markdown(
    f"<hr style='border-color: rgba(255,255,255,0.15); margin: 0.75rem 0;'>"
    f"<h3 style='color: {VICTORY_GOLD}; margin-top: 0;'>Filters</h3>",
    unsafe_allow_html=True,
)

# Group filter — quick-pick presets + clean multiselect
all_groups_raw = sorted(df["athlete_sport"].dropna().unique().tolist())
clean_groups = [g for g in all_groups_raw if g and g != "Unknown"]

GROUPS_KEY = "fm_groups"
if GROUPS_KEY not in st.session_state:
    st.session_state[GROUPS_KEY] = []  # empty == all

selected_groups = st.sidebar.multiselect(
    "Groups",
    clean_groups,
    key=GROUPS_KEY,
    placeholder="Empty = all groups",
)
effective_groups = selected_groups if selected_groups else clean_groups

st.sidebar.markdown(
    f"<p style='color:rgba(255,255,255,0.65); font-size:0.75rem; margin:0.25rem 0 0.5rem 0;'>"
    f"Showing {len(effective_groups)} of {len(clean_groups)} groups</p>",
    unsafe_allow_html=True,
)

max_date = df["date"].max().date()
min_date = df["date"].min().date()
default_start = max(min_date, max_date - timedelta(days=42))
date_range = st.sidebar.date_input(
    "Window",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
    key="fm_dates",
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = default_start, max_date

include_first_test = st.sidebar.checkbox(
    "Show first-time testers", value=True, key="fm_include_first",
    help="Show athletes with only one CMJ. They appear with a 'baseline pending' badge.",
)

mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
mask &= df["athlete_sport"].isin(effective_groups)
window = df.loc[mask].copy()

if window.empty:
    st.warning("No CMJ tests in the selected window. Widen date range or change groups.")
    st.stop()

athletes = sorted(window["Name"].unique().tolist())
default_athletes = get_persisted_athlete_selection("fm_athletes", athletes)
selected_athletes = st.sidebar.multiselect(
    "Athletes (Individual tab)", athletes, default=default_athletes, key="fm_athletes",
)

st.sidebar.markdown(
    f"<p style='color: rgba(255,255,255,0.75); font-size: 0.8rem; margin-top: 1.5rem;'>"
    f"Baseline: rolling {BASELINE_DAYS} days per athlete. SWC factor: {SWC_FACTOR} × SD.</p>",
    unsafe_allow_html=True,
)


# ── SWC Calculations ────────────────────────────────────────────────────────
def latest_vs_baseline(athlete_df: pd.DataFrame, baseline_days: int = BASELINE_DAYS):
    if len(athlete_df) < 2:
        return None
    athlete_df = athlete_df.sort_values("date")
    latest = athlete_df.iloc[-1]
    cutoff = latest["date"] - timedelta(days=baseline_days)
    baseline = athlete_df[(athlete_df["date"] >= cutoff) & (athlete_df["date"] < latest["date"])]
    if baseline.empty:
        baseline = athlete_df.iloc[:-1].tail(3)
    if baseline.empty:
        return None
    return {
        "latest_date": latest["date"],
        "latest": float(latest["peak_power"]),
        "baseline_mean": float(baseline["peak_power"].mean()),
        "baseline_sd": float(baseline["peak_power"].std(ddof=1)) if len(baseline) > 1 else 0.0,
        "baseline_n": int(len(baseline)),
    }


def squad_between_swc(window_df: pd.DataFrame) -> float:
    means = window_df.groupby("Name")["peak_power"].mean()
    if len(means) < 2:
        return 0.0
    return SWC_FACTOR * float(means.std(ddof=1))


def status_from_change(change: float, threshold: float) -> str:
    if threshold <= 0:
        return "insufficient data"
    if change <= -threshold:
        return "fatigued"
    if change >= threshold:
        return "supercompensating"
    return "trivial"


STATUS_COLOURS = {
    "fatigued": STATUS_DANGER,
    "trivial": ENABLER_GREEN,
    "supercompensating": VICTORY_GOLD,
    "insufficient data": "#9aa5a1",
}

between_swc = squad_between_swc(window)

rows = []
for name, ath_df in window.groupby("Name"):
    stats = latest_vs_baseline(ath_df)
    sport_label = ath_df["athlete_sport"].iloc[-1]
    if stats is None:
        if not include_first_test:
            continue
        latest = ath_df.sort_values("date").iloc[-1]
        rows.append({
            "Athlete": name, "Sport": sport_label,
            "Latest test": latest["date"].strftime("%Y-%m-%d"),
            "Latest": float(latest["peak_power"]),
            "Baseline mean": np.nan, "Baseline SD": np.nan, "Baseline n": 0,
            "Change": np.nan, "Change %": np.nan, "CV %": np.nan,
            "Hopkins (between)": "insufficient data",
            "Individual (within)": "insufficient data",
            "Typical error (CV)": "insufficient data",
        })
        continue
    change = stats["latest"] - stats["baseline_mean"]
    within_swc = SWC_FACTOR * stats["baseline_sd"]
    cv_pct = (stats["baseline_sd"] / stats["baseline_mean"] * 100) if stats["baseline_mean"] else 0.0
    cv_threshold = CV_FLAG_MULTIPLIER * stats["baseline_sd"]
    rows.append({
        "Athlete": name, "Sport": sport_label,
        "Latest test": stats["latest_date"].strftime("%Y-%m-%d"),
        "Latest": stats["latest"],
        "Baseline mean": stats["baseline_mean"],
        "Baseline SD": stats["baseline_sd"],
        "Baseline n": stats["baseline_n"],
        "Change": change,
        "Change %": (change / stats["baseline_mean"] * 100) if stats["baseline_mean"] else 0.0,
        "CV %": cv_pct,
        "Hopkins (between)": status_from_change(change, between_swc),
        "Individual (within)": status_from_change(change, within_swc),
        "Typical error (CV)": status_from_change(change, cv_threshold),
    })

summary = pd.DataFrame(rows)
if summary.empty:
    st.warning("No athletes with CMJ tests in the selected window.")
    st.stop()


# ── Shared helpers ──────────────────────────────────────────────────────────
def kpi(label: str, value: str, accent: str = ELITE_GREEN):
    st.markdown(
        f"""
        <div style="background: {accent}; padding: 0.9rem; border-radius: 8px; text-align: center;
             box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <p style="color: rgba(255,255,255,0.85); margin: 0; font-size: 0.78rem;
                 text-transform: uppercase; letter-spacing: 0.05em;">{label}</p>
            <p style="color: white; margin: 0.25rem 0 0 0; font-size: 1.4rem; font-weight: 700;">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {ELITE_GREEN} 0%, {DISCIPLINE_GREEN} 100%);
             padding: 0.9rem 1.1rem; border-radius: 8px; margin: 1.5rem 0 0.75rem 0;
             border-left: 4px solid {VICTORY_GOLD};">
            <h3 style="color: white; margin: 0; font-size: 1.1rem;">{title}</h3>
            <p style="color: rgba(255,255,255,0.85); margin: 0.2rem 0 0 0; font-size: 0.85rem;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(s: str) -> str:
    colour = STATUS_COLOURS.get(s, "#9aa5a1")
    label = {"fatigued": "FATIGUED", "trivial": "Trivial",
             "supercompensating": "Supercomp.",
             "insufficient data": "pending"}[s]
    return (f"<span style='background:{colour}; color:white; padding:2px 8px; "
            f"border-radius:10px; font-size:0.75rem; font-weight:600;'>{label}</span>")


def per_athlete_trajectory(ath_full_df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling per-row baseline and SWC band columns. Leak-free."""
    ath_full_df = ath_full_df.sort_values("date").copy()
    baselines, sd_band = [], []
    for _, row in ath_full_df.iterrows():
        cutoff = row["date"] - timedelta(days=BASELINE_DAYS)
        past = ath_full_df[(ath_full_df["date"] >= cutoff) & (ath_full_df["date"] < row["date"])]
        if len(past) >= 2:
            baselines.append(past["peak_power"].mean())
            sd_band.append(past["peak_power"].std(ddof=1))
        else:
            baselines.append(np.nan)
            sd_band.append(np.nan)
    ath_full_df["baseline"] = baselines
    ath_full_df["sd_band"] = sd_band
    ath_full_df["swc_band"] = ath_full_df["sd_band"] * SWC_FACTOR
    return ath_full_df


# ── Tabs ────────────────────────────────────────────────────────────────────
tab_group, tab_individual = st.tabs(["Group view", "Individual progression"])


# ═══════════════════════════════ GROUP TAB ═════════════════════════════════
with tab_group:
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: kpi("Athletes monitored", f"{len(summary)}")
    with k2: kpi("CMJ tests in window", f"{len(window)}", ENABLER_GREEN)
    with k3: kpi("Squad SWC (W/kg)", f"{between_swc:.2f}", VICTORY_GOLD if between_swc else "#9aa5a1")
    with k4:
        n_fatigued = int((summary["Hopkins (between)"] == "fatigued").sum())
        kpi("Flagged (Hopkins)", f"{n_fatigued}", STATUS_DANGER if n_fatigued else ENABLER_GREEN)
    with k5:
        kpi("Most recent test", summary["Latest test"].max(), ELITE_GREEN)

    # ── Latest Test Day Snapshot ────────────────────────────────────────────
    snapshot_dates = sorted(summary["Latest test"].unique(), reverse=True)
    section_header(
        "Latest Test Day Snapshot",
        "Athletes who tested on the selected day — Δ Peak Power, three SWC verdicts",
    )

    snap_col1, snap_col2 = st.columns([1, 3])
    with snap_col1:
        snapshot_day = st.selectbox("Test day", snapshot_dates, index=0, key="fm_snapshot_day")
    with snap_col2:
        name_search = st.text_input(
            "Find athlete (optional)", value="",
            placeholder="Type a name fragment — e.g. Youcef",
            key="fm_name_search",
        ).strip()

    snap = summary[summary["Latest test"] == snapshot_day].copy()
    if name_search:
        snap = snap[snap["Athlete"].str.contains(name_search, case=False, na=False)]

    if snap.empty:
        st.info(f"No matching athletes tested on {snapshot_day}.")
    else:
        BADGE_LABEL = {
            "fatigued": "FATIGUED", "trivial": "Trivial",
            "supercompensating": "Supercomp.",
            "insufficient data": "BASELINE PENDING",
        }

        def snapshot_card(row: pd.Series) -> str:
            change = row["Change"]
            change_pct = row["Change %"]
            is_pending = pd.isna(change)
            verdicts = [row["Hopkins (between)"], row["Individual (within)"], row["Typical error (CV)"]]
            if "fatigued" in verdicts:
                accent = STATUS_DANGER
                bg = "linear-gradient(90deg, rgba(220,53,69,0.10) 0%, rgba(220,53,69,0.02) 100%)"
            elif "supercompensating" in verdicts:
                accent = VICTORY_GOLD
                bg = "linear-gradient(90deg, rgba(235,206,131,0.18) 0%, rgba(235,206,131,0.04) 100%)"
            elif "trivial" in verdicts:
                accent = ENABLER_GREEN
                bg = "white"
            else:
                accent = "#9aa5a1"
                bg = "rgba(154,165,161,0.06)"
            if is_pending:
                delta_colour = "#6c757d"
                delta_text = "<span style='font-style:italic;'>baseline pending</span>"
            else:
                delta_colour = STATUS_DANGER if change < 0 else ENABLER_GREEN if change > 0 else "#6c757d"
                delta_text = (f"{change:+.2f}<span style='font-size:0.7rem; font-weight:400;'>"
                              f" W/kg ({change_pct:+.1f}%)</span>")
            baseline_text = (
                "first CMJ in window<br><span style='color:#6c757d; font-size:0.72rem;'>need 2+ tests</span>"
                if is_pending
                else f"{row['Baseline mean']:.2f} W/kg<span style='color:#6c757d; font-size:0.72rem;'> (n={row['Baseline n']})</span>"
            )

            def badge(label: str, status: str) -> str:
                col = STATUS_COLOURS.get(status, "#9aa5a1")
                return (
                    f"<div style='display:flex; justify-content:space-between; "
                    f"align-items:center; padding:3px 0;'>"
                    f"<span style='color:#3a4a42; font-size:0.74rem;'>{label}</span>"
                    f"<span style='background:{col}; color:white; padding:2px 8px; "
                    f"border-radius:10px; font-size:0.7rem; font-weight:600;'>"
                    f"{BADGE_LABEL[status]}</span></div>"
                )

            return f"""
            <div style="background: {bg}; border-left: 5px solid {accent};
                 border-radius: 8px; padding: 0.85rem 1rem;
                 box-shadow: 0 2px 6px rgba(0,0,0,0.06); margin-bottom: 0.6rem;">
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <div style="font-weight:700; color:{ELITE_GREEN}; font-size:0.98rem;">{row['Athlete']}</div>
                    <div style="color:#6c757d; font-size:0.75rem;">{row['Sport']}</div>
                </div>
                <div style="display:flex; gap:1.2rem; align-items:baseline; margin-top:0.35rem;">
                    <div>
                        <div style="color:#6c757d; font-size:0.7rem; text-transform:uppercase;
                             letter-spacing:0.04em;">Latest</div>
                        <div style="color:{DISCIPLINE_GREEN}; font-weight:700; font-size:1.25rem;">
                            {row['Latest']:.2f}<span style="font-size:0.7rem; color:#6c757d;
                            font-weight:400;"> W/kg</span>
                        </div>
                    </div>
                    <div>
                        <div style="color:#6c757d; font-size:0.7rem; text-transform:uppercase;
                             letter-spacing:0.04em;">Δ vs baseline</div>
                        <div style="color:{delta_colour}; font-weight:700; font-size:1.25rem;">{delta_text}</div>
                    </div>
                    <div>
                        <div style="color:#6c757d; font-size:0.7rem; text-transform:uppercase;
                             letter-spacing:0.04em;">Baseline</div>
                        <div style="color:{DISCIPLINE_GREEN}; font-size:0.92rem;">{baseline_text}</div>
                    </div>
                </div>
                <div style="margin-top:0.5rem; border-top:1px solid #eef2f0; padding-top:0.4rem;">
                    {badge("Hopkins (between)", row["Hopkins (between)"])}
                    {badge("Individual (within)", row["Individual (within)"])}
                    {badge("Typical error (CV)", row["Typical error (CV)"])}
                </div>
            </div>
            """

        snap["_is_fatigued"] = (snap["Hopkins (between)"] == "fatigued").astype(int)
        snap = snap.sort_values(["_is_fatigued", "Change"], ascending=[False, True], na_position="last").reset_index(drop=True)
        sc = st.columns(2)
        for i, (_, row) in enumerate(snap.iterrows()):
            with sc[i % 2]:
                st.markdown(snapshot_card(row), unsafe_allow_html=True)

    # ── Group Peak Power Chart ──────────────────────────────────────────────
    section_header(
        "Group Peak Power vs Baseline",
        f"Coloured by Hopkins between-athlete SWC (±{between_swc:.2f} W/kg threshold)",
    )

    ranked = summary.dropna(subset=["Change"]).sort_values("Change", ascending=True)
    if ranked.empty:
        st.info("No athletes have a baseline yet — bar chart hidden.")
    else:
        bar_colours = [STATUS_COLOURS[s] for s in ranked["Hopkins (between)"]]
        fig_group = go.Figure()
        fig_group.add_trace(go.Bar(
            x=ranked["Change"], y=ranked["Athlete"], orientation="h",
            marker=dict(color=bar_colours, line=dict(color=DISCIPLINE_GREEN, width=1.2)),
            text=[f"{c:+.2f}" for c in ranked["Change"]], textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>Change vs baseline: %{x:.2f} W/kg<br>"
                "Latest: %{customdata[0]:.2f} W/kg<br>Baseline: %{customdata[1]:.2f} W/kg<br>"
                "Baseline n: %{customdata[2]}<extra></extra>"
            ),
            customdata=ranked[["Latest", "Baseline mean", "Baseline n"]].values,
        ))
        if between_swc:
            fig_group.add_vline(x=-between_swc, line_dash="dash", line_color=STATUS_DANGER,
                                annotation_text="-SWC (fatigue)", annotation_position="top")
            fig_group.add_vline(x=between_swc, line_dash="dash", line_color=VICTORY_GOLD,
                                annotation_text="+SWC", annotation_position="top")
        fig_group.add_vline(x=0, line_color=DISCIPLINE_GREEN, line_width=1)
        fig_group.update_layout(
            height=max(360, 26 * len(ranked) + 80),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial, sans-serif", color="#333"),
            xaxis=dict(title="Δ Peak Power vs baseline (W/kg)", showgrid=True, gridcolor="#e7ece9"),
            yaxis=dict(title=""),
            margin=dict(l=10, r=10, t=40, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_group, width="stretch")

    # ── Three SWC Methods Side-by-Side ──────────────────────────────────────
    section_header(
        "Smallest Worthwhile Change — three views",
        "Compare how each baseline assumption flags the same athletes",
    )

    st.markdown(
        """
        <style>
            table.swc-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
            table.swc-table th { background: #f1f5f3; color: #235036; text-align: left;
                padding: 6px 8px; border-bottom: 1px solid #e0e6e3; }
            table.swc-table td { padding: 6px 8px; border-bottom: 1px solid #f0f3f1;
                color: #18342a; }
            table.swc-table tr:hover td { background: #fafdfb; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def render_method_table(col, title: str, blurb: str, status_col: str, swc_value: str):
        with col:
            st.markdown(
                f"""
                <div style="background: white; border-left: 4px solid {ELITE_GREEN};
                     padding: 0.75rem 0.9rem; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
                     margin-bottom: 0.5rem;">
                    <div style="color: {ELITE_GREEN}; font-weight: 700;">{title}</div>
                    <div style="color: #666; font-size: 0.8rem;">{blurb}</div>
                    <div style="color: {DISCIPLINE_GREEN}; font-size: 0.78rem; margin-top: 0.25rem;">
                        Threshold: <b>{swc_value}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            view = summary[["Athlete", "Change", "Change %", status_col]].copy()
            view = view.sort_values("Change", ascending=True, na_position="last").reset_index(drop=True)
            view["Δ W/kg"] = view["Change"].map(lambda x: "—" if pd.isna(x) else f"{x:+.2f}")
            view["Δ %"] = view["Change %"].map(lambda x: "—" if pd.isna(x) else f"{x:+.1f}%")
            view["Status"] = view[status_col].map(status_badge)
            view = view[["Athlete", "Δ W/kg", "Δ %", "Status"]]
            st.markdown(view.to_html(escape=False, index=False, classes="swc-table", border=0),
                        unsafe_allow_html=True)

    cc1, cc2, cc3 = st.columns(3)
    render_method_table(
        cc1, "Hopkins (between-athlete)",
        "0.2 × SD across squad baseline means. Same threshold for everyone.",
        "Hopkins (between)", f"±{between_swc:.2f} W/kg",
    )
    render_method_table(
        cc2, "Individual (within-athlete)",
        "0.2 × SD of each athlete's own rolling baseline.",
        "Individual (within)", "per-athlete (see expander)",
    )
    render_method_table(
        cc3, "Typical error (CV)",
        f"Change exceeding {CV_FLAG_MULTIPLIER} × baseline SD.",
        "Typical error (CV)", f"{CV_FLAG_MULTIPLIER} × baseline SD",
    )

    with st.expander("Full athlete table (numeric)"):
        detail = summary.copy()
        for col in ["Latest", "Baseline mean", "Baseline SD", "Change"]:
            detail[col] = detail[col].round(2)
        detail["Change %"] = detail["Change %"].round(1)
        detail["CV %"] = detail["CV %"].round(1)
        st.dataframe(detail, width="stretch", hide_index=True)


# ═══════════════════════════════ INDIVIDUAL TAB ═════════════════════════════
with tab_individual:
    if not selected_athletes:
        st.info("Pick one or more athletes from the sidebar to load the progression view.")
    else:
        for idx, athlete in enumerate(selected_athletes):
            ath_all = df[df["Name"] == athlete].sort_values("date")
            if ath_all.empty:
                st.warning(f"No CMJ data for {athlete}.")
                continue

            ath_with_baseline = per_athlete_trajectory(ath_all)
            n_tests = len(ath_all)
            latest_row = ath_all.iloc[-1]
            first_test = ath_all["date"].min()
            best_ever = ath_all["peak_power"].max()
            mean_ever = ath_all["peak_power"].mean()
            days_since = (pd.Timestamp.utcnow().tz_convert(None).normalize() - latest_row["date"]).days

            # Baseline at the latest test for the headline Δ
            stats = latest_vs_baseline(ath_all)
            if stats:
                delta = stats["latest"] - stats["baseline_mean"]
                delta_pct = (delta / stats["baseline_mean"] * 100) if stats["baseline_mean"] else 0.0
                within_swc_val = SWC_FACTOR * stats["baseline_sd"]
                verdict_h = status_from_change(delta, between_swc)
                verdict_w = status_from_change(delta, within_swc_val)
                verdict_c = status_from_change(delta, CV_FLAG_MULTIPLIER * stats["baseline_sd"])
                baseline_text_kpi = f"{stats['baseline_mean']:.2f}"
                delta_text_kpi = f"{delta:+.2f}"
                delta_accent = STATUS_DANGER if delta <= -between_swc else VICTORY_GOLD if delta >= between_swc else ENABLER_GREEN
            else:
                delta = delta_pct = within_swc_val = None
                verdict_h = verdict_w = verdict_c = "insufficient data"
                baseline_text_kpi = "—"
                delta_text_kpi = "n/a"
                delta_accent = "#9aa5a1"

            section_header(
                f"{athlete}  ·  {ath_all['athlete_sport'].iloc[-1]}",
                f"{n_tests} CMJ tests since {first_test.strftime('%Y-%m-%d')} · last test {days_since}d ago",
            )

            # KPI strip — per-athlete
            i1, i2, i3, i4, i5, i6 = st.columns(6)
            with i1: kpi("Latest", f"{latest_row['peak_power']:.2f}", ELITE_GREEN)
            with i2: kpi("Δ vs baseline", delta_text_kpi, delta_accent)
            with i3: kpi("Baseline mean", baseline_text_kpi, ENABLER_GREEN)
            with i4: kpi("Best ever", f"{best_ever:.2f}", VICTORY_GOLD)
            with i5: kpi("Career mean", f"{mean_ever:.2f}", DISCIPLINE_GREEN)
            with i6: kpi("Days since", str(days_since), STATUS_DANGER if days_since > 14 else ELITE_GREEN)

            # Verdict badges row
            st.markdown(
                f"""
                <div style="margin: 0.5rem 0 0.75rem 0; padding: 0.5rem 0.75rem;
                     background: rgba(35,80,54,0.04); border-radius: 6px; display:flex; gap:0.75rem;
                     align-items:center; flex-wrap:wrap;">
                    <span style="color:#3a4a42; font-size:0.78rem; font-weight:600;">SWC verdict (latest):</span>
                    <span>Hopkins {status_badge(verdict_h)}</span>
                    <span>Individual {status_badge(verdict_w)}</span>
                    <span>Typical-error {status_badge(verdict_c)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Progression chart with shaded SWC band
            fig = go.Figure()
            valid = ath_with_baseline.dropna(subset=["baseline"])

            # Light grey shaded ±1 SD band (context)
            if not valid.empty:
                fig.add_trace(go.Scatter(
                    x=valid["date"], y=valid["baseline"] + valid["sd_band"],
                    line=dict(color="#c3d9d1", width=0), mode="lines",
                    showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=valid["date"], y=valid["baseline"] - valid["sd_band"],
                    line=dict(color="#c3d9d1", width=0), mode="lines",
                    fill="tonexty", fillcolor="rgba(195,217,209,0.35)",
                    name="±1 SD band", hoverinfo="skip",
                ))
                # Gold SWC band on top
                fig.add_trace(go.Scatter(
                    x=valid["date"], y=valid["baseline"] + valid["swc_band"],
                    line=dict(color=VICTORY_GOLD, width=0), mode="lines",
                    showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=valid["date"], y=valid["baseline"] - valid["swc_band"],
                    line=dict(color=VICTORY_GOLD, width=0), mode="lines",
                    fill="tonexty", fillcolor="rgba(235,206,131,0.35)",
                    name=f"±SWC ({SWC_FACTOR}×SD)", hoverinfo="skip",
                ))
                # Baseline mean line
                fig.add_trace(go.Scatter(
                    x=valid["date"], y=valid["baseline"],
                    line=dict(color=DISCIPLINE_GREEN, width=1.5, dash="dot"),
                    mode="lines", name="Rolling baseline",
                    hovertemplate="Baseline: %{y:.2f} W/kg<extra></extra>",
                ))

            # Actual test points - colour-coded by verdict relative to that day's baseline
            point_colours = []
            for _, row in ath_with_baseline.iterrows():
                if pd.isna(row["baseline"]):
                    point_colours.append("#9aa5a1")
                    continue
                d = row["peak_power"] - row["baseline"]
                thresh = SWC_FACTOR * (row["sd_band"] or 0)
                if thresh > 0 and d <= -thresh:
                    point_colours.append(STATUS_DANGER)
                elif thresh > 0 and d >= thresh:
                    point_colours.append(VICTORY_GOLD)
                else:
                    point_colours.append(ENABLER_GREEN)

            fig.add_trace(go.Scatter(
                x=ath_with_baseline["date"], y=ath_with_baseline["peak_power"],
                mode="lines+markers", name="Peak Power",
                line=dict(color=ELITE_GREEN, width=2.4),
                marker=dict(size=10, color=point_colours,
                            line=dict(color=DISCIPLINE_GREEN, width=1.5)),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.2f} W/kg<extra></extra>",
            ))

            fig.update_layout(
                height=460,
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Arial, sans-serif", color="#333"),
                xaxis=dict(title="Date", showgrid=True, gridcolor="#e7ece9"),
                yaxis=dict(title=METRIC_LABEL, showgrid=True, gridcolor="#e7ece9"),
                margin=dict(l=10, r=10, t=20, b=30),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, width="stretch", key=f"fm_ind_{athlete}_{idx}")

            # Recent tests table
            with st.expander(f"Recent CMJ tests — {athlete}"):
                recent = ath_with_baseline.tail(15).copy()
                recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
                recent["peak_power"] = recent["peak_power"].round(2)
                recent["baseline"] = recent["baseline"].round(2)
                recent["swc_band"] = recent["swc_band"].round(2)
                recent = recent.rename(columns={
                    "date": "Date", "peak_power": "Peak Power (W/kg)",
                    "baseline": "Baseline", "swc_band": "±SWC",
                })[["Date", "Peak Power (W/kg)", "Baseline", "±SWC"]].iloc[::-1]
                st.dataframe(recent, width="stretch", hide_index=True)

            if idx < len(selected_athletes) - 1:
                st.markdown("<hr style='border-color:#e7ece9; margin: 1.5rem 0;'>", unsafe_allow_html=True)


# ── Footer (shared) ─────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="color: #6c757d; font-size: 0.78rem; margin-top: 1.25rem;
         padding-top: 0.75rem; border-top: 1px solid #e0e6e3;">
        Test type: VALD ForceDecks CMJ (modified protocol, hands-on-hips). Metric:
        body-mass relative takeoff peak power. Baseline window: {BASELINE_DAYS} days.
        Hopkins SWC factor: {SWC_FACTOR}. CV flag multiplier: {CV_FLAG_MULTIPLIER}.
    </div>
    """,
    unsafe_allow_html=True,
)
