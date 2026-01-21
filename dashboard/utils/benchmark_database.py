"""
Benchmark Database Module

Provides editable VALD-based benchmarks with audit logging for S&C staff.
Benchmarks are stored in JSON and can be edited through the dashboard.

Features:
- Default VALD normative data benchmarks
- Per-sport/group customizable benchmarks
- Full audit trail (who changed, when, why)
- Gender-specific benchmarks
"""

import json
import os
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List, Any
from pathlib import Path

# Team Saudi Brand Colors
TEAL_PRIMARY = '#007167'
GOLD_ACCENT = '#a08e66'
TEAL_DARK = '#005a51'

# Default data directory
DATA_DIR = Path(__file__).parent.parent / "data"
BENCHMARK_FILE = DATA_DIR / "benchmarks.json"
AUDIT_LOG_FILE = DATA_DIR / "benchmark_audit_log.json"

# VALD Normative Data - Default benchmarks based on VALD research
# Sources: VALD Performance normative databases, published research
VALD_NORMS = {
    "CMJ": {
        "name": "Counter Movement Jump",
        "metrics": {
            "Jump Height (Imp-Mom)_Trial": {
                "name": "Jump Height (Impulse-Momentum)",
                "unit": "m",
                "display_multiplier": 100,  # Convert to cm
                "display_unit": "cm",
                "male": {
                    "elite": 0.45,      # 45cm
                    "good": 0.38,       # 38cm
                    "average": 0.32,    # 32cm
                    "below_average": 0.25
                },
                "female": {
                    "elite": 0.35,      # 35cm
                    "good": 0.30,       # 30cm
                    "average": 0.25,    # 25cm
                    "below_average": 0.20
                }
            },
            "Peak Power / BM_Trial": {
                "name": "Relative Peak Power",
                "unit": "W/kg",
                "display_multiplier": 1,
                "display_unit": "W/kg",
                "male": {
                    "elite": 60.0,
                    "good": 52.0,
                    "average": 45.0,
                    "below_average": 38.0
                },
                "female": {
                    "elite": 50.0,
                    "good": 43.0,
                    "average": 37.0,
                    "below_average": 30.0
                }
            },
            "RSI-modified_Trial": {
                "name": "Reactive Strength Index (Modified)",
                "unit": "",
                "display_multiplier": 1,
                "display_unit": "",
                "male": {
                    "elite": 0.70,
                    "good": 0.55,
                    "average": 0.45,
                    "below_average": 0.35
                },
                "female": {
                    "elite": 0.55,
                    "good": 0.45,
                    "average": 0.35,
                    "below_average": 0.28
                }
            }
        }
    },
    "IMTP": {
        "name": "Isometric Mid-Thigh Pull",
        "metrics": {
            "Peak Force / BM_Trial": {
                "name": "Relative Peak Force",
                "unit": "N/kg",
                "display_multiplier": 1,
                "display_unit": "N/kg",
                "male": {
                    "elite": 40.0,
                    "good": 35.0,
                    "average": 30.0,
                    "below_average": 25.0
                },
                "female": {
                    "elite": 32.0,
                    "good": 28.0,
                    "average": 24.0,
                    "below_average": 20.0
                }
            },
            "RFD - 200ms_Trial": {
                "name": "Rate of Force Development (0-200ms)",
                "unit": "N/s",
                "display_multiplier": 1,
                "display_unit": "N/s",
                "male": {
                    "elite": 8000,
                    "good": 6500,
                    "average": 5000,
                    "below_average": 3500
                },
                "female": {
                    "elite": 6000,
                    "good": 4800,
                    "average": 3600,
                    "below_average": 2500
                }
            }
        }
    },
    "SLISOSQT": {
        "name": "Single Leg Isometric Squat",
        "metrics": {
            "Peak Vertical Force_Left": {
                "name": "Peak Force (Left)",
                "unit": "N",
                "display_multiplier": 1,
                "display_unit": "N",
                "male": {"elite": 1200, "good": 1000, "average": 850, "below_average": 700},
                "female": {"elite": 900, "good": 750, "average": 620, "below_average": 500}
            },
            "Peak Vertical Force_Right": {
                "name": "Peak Force (Right)",
                "unit": "N",
                "display_multiplier": 1,
                "display_unit": "N",
                "male": {"elite": 1200, "good": 1000, "average": 850, "below_average": 700},
                "female": {"elite": 900, "good": 750, "average": 620, "below_average": 500}
            }
        },
        "asymmetry_threshold": 10.0  # Flag if L/R diff > 10%
    },
    "SLIMTP": {
        "name": "Single Leg Isometric Mid-Thigh Pull",
        "metrics": {
            "Peak Vertical Force_Left": {
                "name": "Peak Force (Left)",
                "unit": "N",
                "display_multiplier": 1,
                "display_unit": "N",
                "male": {"elite": 1400, "good": 1150, "average": 950, "below_average": 750},
                "female": {"elite": 1050, "good": 880, "average": 720, "below_average": 580}
            },
            "Peak Vertical Force_Right": {
                "name": "Peak Force (Right)",
                "unit": "N",
                "display_multiplier": 1,
                "display_unit": "N",
                "male": {"elite": 1400, "good": 1150, "average": 950, "below_average": 750},
                "female": {"elite": 1050, "good": 880, "average": 720, "below_average": 580}
            }
        },
        "asymmetry_threshold": 10.0
    },
    "DJ": {
        "name": "Drop Jump",
        "metrics": {
            "RSI_Trial": {
                "name": "Reactive Strength Index",
                "unit": "",
                "display_multiplier": 1,
                "display_unit": "",
                "male": {"elite": 2.5, "good": 2.0, "average": 1.5, "below_average": 1.2},
                "female": {"elite": 2.0, "good": 1.6, "average": 1.2, "below_average": 0.9}
            },
            "Contact Time_Trial": {
                "name": "Ground Contact Time",
                "unit": "s",
                "display_multiplier": 1000,
                "display_unit": "ms",
                "male": {"elite": 0.18, "good": 0.22, "average": 0.26, "below_average": 0.30},  # Lower is better
                "female": {"elite": 0.20, "good": 0.24, "average": 0.28, "below_average": 0.32}
            }
        }
    },
    "NordBord": {
        "name": "Nordic Hamstring",
        "source": "nordbord",
        "metrics": {
            "leftMaxForce": {
                "name": "Left Max Force",
                "unit": "N",
                "display_multiplier": 1,
                "display_unit": "N",
                "male": {"elite": 450, "good": 380, "average": 320, "below_average": 260},
                "female": {"elite": 350, "good": 300, "average": 250, "below_average": 200},
                "injury_threshold": 337  # Below this = increased injury risk
            },
            "rightMaxForce": {
                "name": "Right Max Force",
                "unit": "N",
                "display_multiplier": 1,
                "display_unit": "N",
                "male": {"elite": 450, "good": 380, "average": 320, "below_average": 260},
                "female": {"elite": 350, "good": 300, "average": 250, "below_average": 200},
                "injury_threshold": 337
            }
        },
        "asymmetry_threshold": 15.0  # Flag if L/R diff > 15%
    },
    "ForceFrame": {
        "name": "Force Frame (Isometric Testing)",
        "source": "forceframe",
        "test_types": {
            "Trunk Quadrant": {
                "asymmetry_threshold": 10.0,
                "positions": ["Anterior", "Posterior", "Left", "Right"]
            },
            "Hip Adduction/Abduction": {
                "asymmetry_threshold": 15.0,
                "sides": ["Left", "Right"]
            },
            "Shoulder IR/ER": {
                "asymmetry_threshold": 10.0,
                "sides": ["Left", "Right"]
            }
        }
    }
}


def ensure_data_dir():
    """Ensure the data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_benchmarks() -> Dict:
    """
    Load benchmarks from file, falling back to VALD norms if not found.
    """
    ensure_data_dir()

    if BENCHMARK_FILE.exists():
        try:
            with open(BENCHMARK_FILE, 'r') as f:
                custom_benchmarks = json.load(f)
            # Merge with VALD norms (custom overrides defaults)
            benchmarks = VALD_NORMS.copy()
            _deep_merge(benchmarks, custom_benchmarks)
            return benchmarks
        except (json.JSONDecodeError, IOError):
            return VALD_NORMS.copy()

    return VALD_NORMS.copy()


def _deep_merge(base: Dict, override: Dict) -> None:
    """Deep merge override into base dict (modifies base in place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def save_benchmarks(benchmarks: Dict, user: str, reason: str) -> bool:
    """
    Save benchmarks to file and log the change.

    Args:
        benchmarks: The benchmark data to save
        user: Username of person making the change
        reason: Reason for the change

    Returns:
        True if successful, False otherwise
    """
    ensure_data_dir()

    try:
        # Save benchmarks
        with open(BENCHMARK_FILE, 'w') as f:
            json.dump(benchmarks, f, indent=2)

        # Log the change
        log_change(user, "UPDATE", benchmarks, reason)

        return True
    except IOError as e:
        st.error(f"Failed to save benchmarks: {e}")
        return False


def log_change(user: str, action: str, data: Any, reason: str) -> None:
    """
    Log a benchmark change to the audit log.

    Args:
        user: Username of person making the change
        action: Type of action (UPDATE, RESET, etc.)
        data: The data that was changed
        reason: Reason for the change
    """
    ensure_data_dir()

    # Load existing log
    log_entries = []
    if AUDIT_LOG_FILE.exists():
        try:
            with open(AUDIT_LOG_FILE, 'r') as f:
                log_entries = json.load(f)
        except (json.JSONDecodeError, IOError):
            log_entries = []

    # Add new entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user": user,
        "action": action,
        "reason": reason,
        "summary": _summarize_changes(data)
    }
    log_entries.append(entry)

    # Keep last 500 entries
    if len(log_entries) > 500:
        log_entries = log_entries[-500:]

    # Save log
    try:
        with open(AUDIT_LOG_FILE, 'w') as f:
            json.dump(log_entries, f, indent=2)
    except IOError:
        pass  # Non-critical, don't fail on log write errors


def _summarize_changes(data: Any) -> str:
    """Create a brief summary of what changed."""
    if isinstance(data, dict):
        test_types = list(data.keys())[:3]
        return f"Updated benchmarks for: {', '.join(test_types)}"
    return "Benchmark update"


def get_audit_log(limit: int = 50) -> List[Dict]:
    """
    Get recent audit log entries.

    Args:
        limit: Maximum number of entries to return

    Returns:
        List of audit log entries (newest first)
    """
    if not AUDIT_LOG_FILE.exists():
        return []

    try:
        with open(AUDIT_LOG_FILE, 'r') as f:
            entries = json.load(f)
        return list(reversed(entries[-limit:]))
    except (json.JSONDecodeError, IOError):
        return []


def reset_to_vald_norms(user: str, reason: str = "Reset to VALD defaults") -> bool:
    """
    Reset all benchmarks to VALD normative defaults.

    Args:
        user: Username of person making the change
        reason: Reason for the reset

    Returns:
        True if successful
    """
    ensure_data_dir()

    try:
        # Remove custom benchmarks file
        if BENCHMARK_FILE.exists():
            BENCHMARK_FILE.unlink()

        # Log the reset
        log_change(user, "RESET", {"action": "reset_to_defaults"}, reason)

        return True
    except IOError as e:
        st.error(f"Failed to reset benchmarks: {e}")
        return False


def get_benchmark_for_test(
    test_type: str,
    metric: str,
    gender: str = "male",
    level: str = "good"
) -> Optional[float]:
    """
    Get a specific benchmark value.

    Args:
        test_type: Test type code (CMJ, IMTP, etc.)
        metric: Metric column name
        gender: 'male' or 'female'
        level: 'elite', 'good', 'average', or 'below_average'

    Returns:
        Benchmark value or None if not found
    """
    benchmarks = load_benchmarks()

    try:
        test_config = benchmarks.get(test_type, {})
        metrics = test_config.get("metrics", {})
        metric_config = metrics.get(metric, {})
        gender_values = metric_config.get(gender.lower(), {})
        return gender_values.get(level)
    except (KeyError, TypeError):
        return None


def get_injury_threshold(test_type: str, metric: str) -> Optional[float]:
    """
    Get injury risk threshold for a metric if defined.
    """
    benchmarks = load_benchmarks()

    try:
        return benchmarks[test_type]["metrics"][metric].get("injury_threshold")
    except (KeyError, TypeError):
        return None


def get_asymmetry_threshold(test_type: str) -> float:
    """
    Get asymmetry threshold for a test type.
    Default is 10% if not specified.
    """
    benchmarks = load_benchmarks()

    try:
        return benchmarks[test_type].get("asymmetry_threshold", 10.0)
    except (KeyError, TypeError):
        return 10.0


def render_benchmark_editor():
    """
    Render the benchmark editor interface for S&C staff.
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, #007167 0%, #005a51 100%);
                padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0;">⚙️ Benchmark Settings</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Edit VALD normative benchmarks - changes are logged with audit trail
        </p>
    </div>
    """, unsafe_allow_html=True)

    # User identification
    col1, col2 = st.columns([1, 2])
    with col1:
        user_name = st.text_input(
            "Your Name (for audit log):",
            key="benchmark_editor_user",
            placeholder="e.g., John Smith"
        )

    if not user_name:
        st.warning("Please enter your name to edit benchmarks (required for audit trail)")
        return

    # Load current benchmarks
    benchmarks = load_benchmarks()

    # Tabs for different sections
    editor_tabs = st.tabs(["📊 Edit Benchmarks", "📋 Audit Log", "🔄 Reset Options"])

    with editor_tabs[0]:
        _render_benchmark_edit_form(benchmarks, user_name)

    with editor_tabs[1]:
        _render_audit_log()

    with editor_tabs[2]:
        _render_reset_options(user_name)


def _render_benchmark_edit_form(benchmarks: Dict, user_name: str):
    """Render the benchmark editing form."""

    # Select test type to edit
    test_types = list(benchmarks.keys())
    selected_test = st.selectbox(
        "Select Test Type:",
        test_types,
        format_func=lambda x: benchmarks.get(x, {}).get("name", x),
        key="benchmark_edit_test_select"
    )

    if not selected_test:
        return

    test_config = benchmarks.get(selected_test, {})
    metrics = test_config.get("metrics", {})

    if not metrics:
        st.info(f"No editable metrics found for {selected_test}")
        return

    st.markdown(f"### {test_config.get('name', selected_test)} Benchmarks")

    # Edit each metric
    updated_values = {}

    for metric_key, metric_config in metrics.items():
        metric_name = metric_config.get("name", metric_key)
        unit = metric_config.get("display_unit", metric_config.get("unit", ""))

        st.markdown(f"**{metric_name}** ({unit})")

        col1, col2 = st.columns(2)

        with col1:
            st.caption("Male Benchmarks")
            male_values = metric_config.get("male", {})
            for level in ["elite", "good", "average", "below_average"]:
                current = male_values.get(level, 0)
                multiplier = metric_config.get("display_multiplier", 1)
                display_val = current * multiplier if multiplier != 1 else current

                new_val = st.number_input(
                    f"{level.replace('_', ' ').title()}:",
                    value=float(display_val),
                    key=f"bench_{selected_test}_{metric_key}_male_{level}",
                    step=0.1
                )

                # Store back in original units
                if multiplier != 1:
                    new_val = new_val / multiplier

                if f"male_{metric_key}" not in updated_values:
                    updated_values[f"male_{metric_key}"] = {}
                updated_values[f"male_{metric_key}"][level] = new_val

        with col2:
            st.caption("Female Benchmarks")
            female_values = metric_config.get("female", {})
            for level in ["elite", "good", "average", "below_average"]:
                current = female_values.get(level, 0)
                multiplier = metric_config.get("display_multiplier", 1)
                display_val = current * multiplier if multiplier != 1 else current

                new_val = st.number_input(
                    f"{level.replace('_', ' ').title()}:",
                    value=float(display_val),
                    key=f"bench_{selected_test}_{metric_key}_female_{level}",
                    step=0.1
                )

                if multiplier != 1:
                    new_val = new_val / multiplier

                if f"female_{metric_key}" not in updated_values:
                    updated_values[f"female_{metric_key}"] = {}
                updated_values[f"female_{metric_key}"][level] = new_val

        st.divider()

    # Asymmetry threshold if applicable
    if "asymmetry_threshold" in test_config:
        st.markdown("**Asymmetry Threshold**")
        new_asym = st.number_input(
            "Flag asymmetry above (%):",
            value=float(test_config.get("asymmetry_threshold", 10.0)),
            min_value=0.0,
            max_value=50.0,
            step=1.0,
            key=f"bench_{selected_test}_asymmetry"
        )
        updated_values["asymmetry_threshold"] = new_asym

    # Save changes
    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        change_reason = st.text_input(
            "Reason for change:",
            placeholder="e.g., Updated based on new research / sport-specific adjustment",
            key="benchmark_change_reason"
        )

    with col2:
        st.write("")  # Spacer
        st.write("")
        if st.button("💾 Save Changes", type="primary", key="save_benchmarks"):
            if not change_reason:
                st.error("Please provide a reason for the change")
            else:
                # Reconstruct the benchmarks with updates
                updated_benchmarks = benchmarks.copy()

                # Apply metric updates
                for key, values in updated_values.items():
                    if key == "asymmetry_threshold":
                        updated_benchmarks[selected_test]["asymmetry_threshold"] = values
                    elif key.startswith("male_"):
                        metric_key = key[5:]
                        if metric_key in updated_benchmarks[selected_test]["metrics"]:
                            updated_benchmarks[selected_test]["metrics"][metric_key]["male"] = values
                    elif key.startswith("female_"):
                        metric_key = key[7:]
                        if metric_key in updated_benchmarks[selected_test]["metrics"]:
                            updated_benchmarks[selected_test]["metrics"][metric_key]["female"] = values

                if save_benchmarks(updated_benchmarks, user_name, change_reason):
                    st.success("✅ Benchmarks saved successfully!")
                    st.rerun()


def _render_audit_log():
    """Render the audit log view."""
    st.markdown("### 📋 Change History")

    entries = get_audit_log(50)

    if not entries:
        st.info("No changes logged yet")
        return

    # Convert to dataframe for display
    df = pd.DataFrame(entries)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')

    # Style the dataframe
    st.dataframe(
        df[['timestamp', 'user', 'action', 'reason', 'summary']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.TextColumn("Date/Time", width="medium"),
            "user": st.column_config.TextColumn("Changed By", width="medium"),
            "action": st.column_config.TextColumn("Action", width="small"),
            "reason": st.column_config.TextColumn("Reason", width="large"),
            "summary": st.column_config.TextColumn("Summary", width="large")
        }
    )

    # Export option
    if st.button("📥 Export Log (CSV)"):
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "benchmark_audit_log.csv",
            "text/csv"
        )


def _render_reset_options(user_name: str):
    """Render reset/restore options."""
    st.markdown("### 🔄 Reset Options")

    st.warning("⚠️ These actions cannot be undone!")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Reset to VALD Defaults**")
        st.caption("Restore all benchmarks to original VALD normative values")

        reset_reason = st.text_input(
            "Reason for reset:",
            placeholder="e.g., Starting fresh / Incorrect values",
            key="reset_reason"
        )

        if st.button("🔄 Reset All Benchmarks", type="secondary"):
            if reset_reason:
                if reset_to_vald_norms(user_name, reset_reason):
                    st.success("✅ Benchmarks reset to VALD defaults")
                    st.rerun()
            else:
                st.error("Please provide a reason for the reset")


def get_benchmark_status(
    value: float,
    test_type: str,
    metric: str,
    gender: str = "male"
) -> str:
    """
    Get status indicator based on benchmark comparison.

    Returns: 'elite', 'good', 'average', 'below_average', or 'unknown'
    """
    benchmarks = load_benchmarks()

    try:
        metric_config = benchmarks[test_type]["metrics"][metric]
        gender_values = metric_config.get(gender.lower(), {})

        elite = gender_values.get("elite", float('inf'))
        good = gender_values.get("good", float('inf'))
        average = gender_values.get("average", float('inf'))

        # Check if lower is better (e.g., contact time)
        if elite < good < average:
            # Lower is better
            if value <= elite:
                return "elite"
            elif value <= good:
                return "good"
            elif value <= average:
                return "average"
            else:
                return "below_average"
        else:
            # Higher is better (default)
            if value >= elite:
                return "elite"
            elif value >= good:
                return "good"
            elif value >= average:
                return "average"
            else:
                return "below_average"

    except (KeyError, TypeError):
        return "unknown"


def get_status_color(status: str) -> str:
    """Get color for status indicator."""
    colors = {
        "elite": TEAL_PRIMARY,      # Teal
        "good": "#009688",          # Light teal
        "average": GOLD_ACCENT,     # Gold/warning
        "below_average": "#dc3545", # Red
        "unknown": "#6c757d"        # Gray
    }
    return colors.get(status, "#6c757d")


def get_status_emoji(status: str) -> str:
    """Get emoji for status indicator."""
    emojis = {
        "elite": "🟢",
        "good": "🟢",
        "average": "🟡",
        "below_average": "🔴",
        "unknown": "⚪"
    }
    return emojis.get(status, "⚪")
