"""
Sport-Specific Configuration and Benchmarks
Saudi National Team - VALD Performance Dashboard

This module contains sport-specific contexts, benchmarks, and metric priorities
for strength and conditioning analysis across different sports.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class SportBenchmarks:
    """Benchmark values for a specific sport"""
    sport_name: str

    # Jump metrics (ForceDecks)
    cmj_height_excellent: float  # cm
    cmj_height_good: float
    cmj_height_average: float

    rsi_excellent: float  # Reactive Strength Index
    rsi_good: float
    rsi_average: float

    peak_force_excellent: float  # N/kg
    peak_force_good: float
    peak_force_average: float

    # Asymmetry thresholds
    asymmetry_caution: float = 10.0  # %
    asymmetry_risk: float = 15.0  # %

    # Sport-specific emphasis
    priority_metrics: List[str] = None
    key_attributes: List[str] = None

    # Normative data context
    context: str = ""


# ============================================================================
# SPORT-SPECIFIC BENCHMARKS
# ============================================================================

SPORT_BENCHMARKS = {

    # ------------------------------------------------------------------------
    # ATHLETICS - HORIZONTAL JUMPS (Long Jump, Triple Jump)
    # ------------------------------------------------------------------------
    "Athletics - Horizontal Jumps": SportBenchmarks(
        sport_name="Athletics - Horizontal Jumps",
        cmj_height_excellent=50.0,
        cmj_height_good=45.0,
        cmj_height_average=40.0,
        rsi_excellent=2.5,
        rsi_good=2.0,
        rsi_average=1.5,
        peak_force_excellent=35.0,  # N/kg
        peak_force_good=30.0,
        peak_force_average=25.0,
        asymmetry_caution=8.0,
        asymmetry_risk=12.0,
        priority_metrics=[
            "Jump Height (Flight Time) [cm]",
            "RSI Modified",
            "Relative Peak Force [N/kg]",
            "Peak Power [W]",
            "Contraction Time [ms]"
        ],
        key_attributes=[
            "Explosive power",
            "Elastic strength",
            "Minimal asymmetry critical",
            "High RSI for sprint-jump performance"
        ],
        context="Horizontal jumpers require exceptional elastic power and reactive strength. "
                "Jump height >45cm and RSI >2.0 are typical for elite level. "
                "Asymmetry should be <10% to prevent injury in high-impact landings."
    ),

    # ------------------------------------------------------------------------
    # ATHLETICS - MIDDLE DISTANCE (800m, 1500m)
    # ------------------------------------------------------------------------
    "Athletics - Middle distance": SportBenchmarks(
        sport_name="Athletics - Middle distance",
        cmj_height_excellent=42.0,
        cmj_height_good=38.0,
        cmj_height_average=35.0,
        rsi_excellent=2.0,
        rsi_good=1.7,
        rsi_average=1.4,
        peak_force_excellent=28.0,
        peak_force_good=25.0,
        peak_force_average=22.0,
        asymmetry_caution=8.0,
        asymmetry_risk=12.0,
        priority_metrics=[
            "Jump Height (Flight Time) [cm]",
            "RSI Modified",
            "Relative Peak Force [N/kg]",
            "Contact Time [ms]"
        ],
        key_attributes=[
            "Moderate power output",
            "Efficient force application",
            "Running economy focus",
            "Lower asymmetry critical"
        ],
        context="Middle distance runners need good reactive strength for running economy. "
                "Lower jump heights than sprinters but excellent RSI. "
                "Bilateral symmetry crucial for injury prevention over high volumes."
    ),

    # ------------------------------------------------------------------------
    # FENCING - EPEE
    # ------------------------------------------------------------------------
    "Epee": SportBenchmarks(
        sport_name="Epee",
        cmj_height_excellent=42.0,
        cmj_height_good=38.0,
        cmj_height_average=35.0,
        rsi_excellent=2.2,
        rsi_good=1.8,
        rsi_average=1.5,
        peak_force_excellent=30.0,
        peak_force_good=27.0,
        peak_force_average=24.0,
        asymmetry_caution=12.0,  # Higher tolerance for fencing
        asymmetry_risk=18.0,
        priority_metrics=[
            "Jump Height (Flight Time) [cm]",
            "Lunge Distance",
            "Unilateral Peak Force (Lead Leg)",
            "Asymmetry Index",
            "RFD (Rate of Force Development)"
        ],
        key_attributes=[
            "Explosive lunge power",
            "Unilateral strength (lead leg dominant)",
            "Quick force application",
            "Expected asymmetry (sport-specific)"
        ],
        context="Fencers develop asymmetry due to dominant leg lunging. "
                "15-20% asymmetry may be functional, not pathological. "
                "Focus on explosive single-leg strength and reactive ability."
    ),

    # ------------------------------------------------------------------------
    # FENCING - SABRE
    # ------------------------------------------------------------------------
    "Sabre": SportBenchmarks(
        sport_name="Sabre",
        cmj_height_excellent=43.0,
        cmj_height_good=39.0,
        cmj_height_average=36.0,
        rsi_excellent=2.3,
        rsi_good=1.9,
        rsi_average=1.6,
        peak_force_excellent=31.0,
        peak_force_good=28.0,
        peak_force_average=25.0,
        asymmetry_caution=12.0,
        asymmetry_risk=18.0,
        priority_metrics=[
            "Jump Height (Flight Time) [cm]",
            "RSI Modified",
            "Peak Power [W]",
            "Unilateral Jump Height"
        ],
        key_attributes=[
            "Higher power than epee (more dynamic)",
            "Explosive upper body integration",
            "Quick direction changes",
            "Functional asymmetry expected"
        ],
        context="Sabre is more dynamic than epee with more explosive movements. "
                "Slightly higher power outputs expected. "
                "Asymmetry tolerance similar to epee due to sport demands."
    ),

    # ------------------------------------------------------------------------
    # ROWING
    # ------------------------------------------------------------------------
    "Rowing": SportBenchmarks(
        sport_name="Rowing",
        cmj_height_excellent=45.0,
        cmj_height_good=40.0,
        cmj_height_average=36.0,
        rsi_excellent=2.0,
        rsi_good=1.7,
        rsi_average=1.4,
        peak_force_excellent=32.0,
        peak_force_good=28.0,
        peak_force_average=25.0,
        asymmetry_caution=6.0,  # Very low tolerance
        asymmetry_risk=10.0,
        priority_metrics=[
            "Peak Force [N]",
            "Relative Peak Force [N/kg]",
            "Impulse [Ns]",
            "Jump Height (Flight Time) [cm]",
            "Asymmetry Index"
        ],
        key_attributes=[
            "High absolute force production",
            "Bilateral symmetry critical",
            "Force endurance",
            "Lower RSI emphasis"
        ],
        context="Rowers require high absolute strength and perfect bilateral symmetry. "
                "Asymmetry >10% can affect boat balance and efficiency. "
                "Focus on total force production and bilateral consistency."
    ),

    # ------------------------------------------------------------------------
    # SWIMMING
    # ------------------------------------------------------------------------
    "Swimming": SportBenchmarks(
        sport_name="Swimming",
        cmj_height_excellent=38.0,
        cmj_height_good=34.0,
        cmj_height_average=30.0,
        rsi_excellent=1.8,
        rsi_good=1.5,
        rsi_average=1.2,
        peak_force_excellent=26.0,
        peak_force_good=23.0,
        peak_force_average=20.0,
        asymmetry_caution=8.0,
        asymmetry_risk=12.0,
        priority_metrics=[
            "Jump Height (Flight Time) [cm]",
            "Peak Power [W]",
            "Rate of Force Development",
            "Asymmetry Index"
        ],
        key_attributes=[
            "Lower emphasis on jumping ability",
            "Power for starts and turns",
            "Bilateral symmetry important",
            "Explosive push-off strength"
        ],
        context="Swimmers have lower jump metrics than other sports but still need power for starts/turns. "
                "CMJ >35cm is good for swimmers. "
                "Symmetry important for stroke efficiency."
    ),

    # ------------------------------------------------------------------------
    # PARA SWIMMING
    # ------------------------------------------------------------------------
    "Para Swimming": SportBenchmarks(
        sport_name="Para Swimming",
        cmj_height_excellent=None,  # Highly variable by classification
        cmj_height_good=None,
        cmj_height_average=None,
        rsi_excellent=None,
        rsi_good=None,
        rsi_average=None,
        peak_force_excellent=None,
        peak_force_good=None,
        peak_force_average=None,
        asymmetry_caution=None,  # Classification-dependent
        asymmetry_risk=None,
        priority_metrics=[
            "Individual baseline tracking",
            "% change from baseline",
            "Classification-specific metrics"
        ],
        key_attributes=[
            "Highly individualized assessment",
            "Classification-specific norms",
            "Focus on individual progress",
            "Functional movement patterns"
        ],
        context="Para athletes require individualized assessment based on classification. "
                "Focus on personal baselines and percentage improvements rather than absolute values. "
                "Asymmetry may be functional depending on impairment."
    ),

    # ------------------------------------------------------------------------
    # WEIGHTLIFTING (Olympic Lifting)
    # ------------------------------------------------------------------------
    "Weightlifting": SportBenchmarks(
        sport_name="Weightlifting",
        cmj_height_excellent=50.0,  # Very high
        cmj_height_good=46.0,
        cmj_height_average=42.0,
        rsi_excellent=2.4,
        rsi_good=2.0,
        rsi_average=1.7,
        peak_force_excellent=38.0,
        peak_force_good=34.0,
        peak_force_average=30.0,
        asymmetry_caution=6.0,
        asymmetry_risk=10.0,
        priority_metrics=[
            "Jump Height (Flight Time) [cm]",
            "Peak Power [W]",
            "Peak Force [N]",
            "Rate of Force Development",
            "Asymmetry Index"
        ],
        key_attributes=[
            "Highest power outputs",
            "Explosive triple extension",
            "Bilateral symmetry critical",
            "Strong correlation with lift performance"
        ],
        context="Weightlifters have the highest jump heights and power outputs of all sports. "
                "CMJ >45cm typical for international level. "
                "Perfect symmetry required for clean & jerk technique."
    ),

    # ------------------------------------------------------------------------
    # SHOOTING
    # ------------------------------------------------------------------------
    "Shooting": SportBenchmarks(
        sport_name="Shooting",
        cmj_height_excellent=35.0,  # Lower emphasis
        cmj_height_good=31.0,
        cmj_height_average=28.0,
        rsi_excellent=1.6,
        rsi_good=1.3,
        rsi_average=1.0,
        peak_force_excellent=24.0,
        peak_force_good=21.0,
        peak_force_average=18.0,
        asymmetry_caution=10.0,
        asymmetry_risk=15.0,
        priority_metrics=[
            "Isometric Strength",
            "Stability Metrics",
            "Postural Control",
            "Bilateral Balance"
        ],
        key_attributes=[
            "Postural stability emphasis",
            "Lower power requirements",
            "Isometric strength important",
            "Core stability focus"
        ],
        context="Shooters require less explosive power but excellent stability and control. "
                "Lower CMJ values expected. "
                "Focus on isometric strength and postural stability for shooting platform."
    ),

    # ------------------------------------------------------------------------
    # WRESTLING
    # ------------------------------------------------------------------------
    "Wrestling": SportBenchmarks(
        sport_name="Wrestling",
        cmj_height_excellent=46.0,
        cmj_height_good=42.0,
        cmj_height_average=38.0,
        rsi_excellent=2.2,
        rsi_good=1.9,
        rsi_average=1.6,
        peak_force_excellent=33.0,
        peak_force_good=30.0,
        peak_force_average=27.0,
        asymmetry_caution=10.0,
        asymmetry_risk=15.0,
        priority_metrics=[
            "Peak Force [N]",
            "Jump Height (Flight Time) [cm]",
            "Isometric Strength",
            "Bilateral Strength Balance",
            "Rate of Force Development"
        ],
        key_attributes=[
            "High power for takedowns",
            "Isometric strength for holds",
            "Bilateral strength important",
            "Weight class considerations"
        ],
        context="Wrestlers need explosive power for takedowns and isometric strength for control. "
                "Bilateral strength balance important for technique. "
                "Relative strength (N/kg) critical due to weight classes."
    ),

    # ------------------------------------------------------------------------
    # JUDO
    # ------------------------------------------------------------------------
    "Judo": SportBenchmarks(
        sport_name="Judo",
        cmj_height_excellent=45.0,
        cmj_height_good=41.0,
        cmj_height_average=37.0,
        rsi_excellent=2.1,
        rsi_good=1.8,
        rsi_average=1.5,
        peak_force_excellent=32.0,
        peak_force_good=29.0,
        peak_force_average=26.0,
        asymmetry_caution=10.0,
        asymmetry_risk=15.0,
        priority_metrics=[
            "Peak Power [W]",
            "Jump Height (Flight Time) [cm]",
            "Grip Strength (if available)",
            "Unilateral Strength",
            "Rate of Force Development"
        ],
        key_attributes=[
            "Explosive throwing power",
            "Unilateral strength for grips",
            "Quick force application",
            "Weight class specific"
        ],
        context="Judokas require explosive power for throws and strong unilateral capacity. "
                "Power production in multiple planes important. "
                "Grip strength and upper body integration crucial."
    ),

    # ------------------------------------------------------------------------
    # JIU-JITSU
    # ------------------------------------------------------------------------
    "Jiu-Jitsu": SportBenchmarks(
        sport_name="Jiu-Jitsu",
        cmj_height_excellent=43.0,
        cmj_height_good=39.0,
        cmj_height_average=36.0,
        rsi_excellent=2.0,
        rsi_good=1.7,
        rsi_average=1.4,
        peak_force_excellent=30.0,
        peak_force_good=27.0,
        peak_force_average=24.0,
        asymmetry_caution=10.0,
        asymmetry_risk=15.0,
        priority_metrics=[
            "Isometric Strength",
            "Peak Force [N]",
            "Force Endurance",
            "Jump Height (Flight Time) [cm]"
        ],
        key_attributes=[
            "Isometric strength for holds",
            "Force endurance",
            "Less explosive than judo",
            "Ground-based power"
        ],
        context="Jiu-jitsu athletes need sustained force production and isometric strength. "
                "Less emphasis on explosive power than judo. "
                "Force endurance and repetitive strength important."
    ),

    # ------------------------------------------------------------------------
    # SNOW SPORTS
    # ------------------------------------------------------------------------
    "Snow Sports": SportBenchmarks(
        sport_name="Snow Sports",
        cmj_height_excellent=48.0,
        cmj_height_good=44.0,
        cmj_height_average=40.0,
        rsi_excellent=2.3,
        rsi_good=2.0,
        rsi_average=1.7,
        peak_force_excellent=34.0,
        peak_force_good=30.0,
        peak_force_average=27.0,
        asymmetry_caution=8.0,
        asymmetry_risk=12.0,
        priority_metrics=[
            "Jump Height (Flight Time) [cm]",
            "RSI Modified",
            "Peak Power [W]",
            "Eccentric Strength",
            "Asymmetry Index"
        ],
        key_attributes=[
            "High eccentric demands",
            "Reactive strength crucial",
            "Unilateral control",
            "Injury prevention focus"
        ],
        context="Snow sports athletes need exceptional reactive strength and eccentric control. "
                "High RSI for landing absorption in alpine skiing. "
                "Single-leg strength critical for injury prevention."
    ),

    # ------------------------------------------------------------------------
    # DECATHLON
    # ------------------------------------------------------------------------
    "Decathlon": SportBenchmarks(
        sport_name="Decathlon",
        cmj_height_excellent=52.0,  # Elite multi-event
        cmj_height_good=48.0,
        cmj_height_average=44.0,
        rsi_excellent=2.6,
        rsi_good=2.2,
        rsi_average=1.9,
        peak_force_excellent=36.0,
        peak_force_good=32.0,
        peak_force_average=28.0,
        asymmetry_caution=7.0,
        asymmetry_risk=10.0,
        priority_metrics=[
            "Jump Height (Flight Time) [cm]",
            "Peak Power [W]",
            "RSI Modified",
            "Peak Force [N]",
            "Rate of Force Development"
        ],
        key_attributes=[
            "Highest all-around athleticism",
            "Elite power production",
            "Excellent RSI",
            "Versatile strength qualities"
        ],
        context="Decathletes display the highest overall athleticism with elite jump and power metrics. "
                "CMJ >50cm typical for world-class level. "
                "Require balance of all strength qualities."
    ),

    # ------------------------------------------------------------------------
    # COASTAL (Rowing variant)
    # ------------------------------------------------------------------------
    "Coastal": SportBenchmarks(
        sport_name="Coastal",
        cmj_height_excellent=43.0,
        cmj_height_good=39.0,
        cmj_height_average=35.0,
        rsi_excellent=1.9,
        rsi_good=1.6,
        rsi_average=1.3,
        peak_force_excellent=30.0,
        peak_force_good=27.0,
        peak_force_average=24.0,
        asymmetry_caution=8.0,
        asymmetry_risk=12.0,
        priority_metrics=[
            "Peak Force [N]",
            "Force Endurance",
            "Jump Height (Flight Time) [cm]",
            "Asymmetry Index"
        ],
        key_attributes=[
            "High force production",
            "Endurance component",
            "Bilateral symmetry",
            "Variable conditions adaptation"
        ],
        context="Coastal rowing requires similar strength to traditional rowing but with more variability. "
                "Slightly lower absolute force but higher adaptability. "
                "Bilateral symmetry still important but slightly more tolerance."
    ),

    # ------------------------------------------------------------------------
    # GENERAL ATHLETICS (when not specified)
    # ------------------------------------------------------------------------
    "Athletics": SportBenchmarks(
        sport_name="Athletics",
        cmj_height_excellent=45.0,
        cmj_height_good=40.0,
        cmj_height_average=36.0,
        rsi_excellent=2.2,
        rsi_good=1.8,
        rsi_average=1.5,
        peak_force_excellent=30.0,
        peak_force_good=27.0,
        peak_force_average=24.0,
        asymmetry_caution=8.0,
        asymmetry_risk=12.0,
        priority_metrics=[
            "Jump Height (Flight Time) [cm]",
            "RSI Modified",
            "Peak Power [W]",
            "Peak Force [N]"
        ],
        key_attributes=[
            "General track & field",
            "Power-speed emphasis",
            "Reactive strength",
            "Low asymmetry"
        ],
        context="General athletics benchmarks for unspecified events. "
                "Use event-specific benchmarks when available."
    ),
}


# ============================================================================
# METRIC DEFINITIONS & INTERPRETATION
# ============================================================================

METRIC_DEFINITIONS = {
    "Jump Height (Flight Time) [cm]": {
        "description": "Vertical jump height calculated from flight time",
        "unit": "cm",
        "higher_is_better": True,
        "category": "Power",
        "interpretation": "Primary indicator of lower body power. Elite athletes: >45cm, Good: >40cm, Average: >35cm"
    },
    "RSI Modified": {
        "description": "Reactive Strength Index - ratio of jump height to ground contact time",
        "unit": "ratio",
        "higher_is_better": True,
        "category": "Reactive Strength",
        "interpretation": "Measures elastic strength. Elite: >2.0, Good: >1.7, Average: >1.4. Critical for running economy and plyometric ability"
    },
    "Peak Force [N]": {
        "description": "Maximum force produced during jump",
        "unit": "N",
        "higher_is_better": True,
        "category": "Strength",
        "interpretation": "Absolute strength production. Compare to body weight for relative force (N/kg)"
    },
    "Relative Peak Force [N/kg]": {
        "description": "Peak force normalized to body weight",
        "unit": "N/kg",
        "higher_is_better": True,
        "category": "Relative Strength",
        "interpretation": "Elite: >30 N/kg, Good: >27 N/kg. Critical for weight class sports and jumping ability"
    },
    "Peak Power [W]": {
        "description": "Maximum power output during jump",
        "unit": "W",
        "higher_is_better": True,
        "category": "Power",
        "interpretation": "Combination of force and velocity. Highest in weightlifters and jumpers"
    },
    "Rate of Force Development": {
        "description": "How quickly force is produced (first 200ms)",
        "unit": "N/s",
        "higher_is_better": True,
        "category": "Explosive Strength",
        "interpretation": "Critical for quick movements. Important for combat sports, fencing, and sprinting"
    },
    "Asymmetry Index": {
        "description": "Difference between left and right limb performance",
        "unit": "%",
        "higher_is_better": False,
        "category": "Bilateral Balance",
        "interpretation": "<10% is ideal, 10-15% caution, >15% high injury risk (except fencing where some asymmetry is functional)"
    },
    "Contraction Time [ms]": {
        "description": "Time from movement initiation to takeoff",
        "unit": "ms",
        "higher_is_better": False,
        "category": "Speed",
        "interpretation": "Shorter is better. Elite: <600ms. Indicates neuromuscular efficiency"
    },
    "Contact Time [ms]": {
        "description": "Ground contact time during reactive movements",
        "unit": "ms",
        "higher_is_better": False,
        "category": "Reactive Strength",
        "interpretation": "Shorter contact with higher RSI indicates better elastic strength. Elite runners: <200ms"
    },
}


# ============================================================================
# RISK THRESHOLDS
# ============================================================================

RISK_THRESHOLDS = {
    "asymmetry_high_risk": 15.0,  # %
    "asymmetry_moderate_risk": 10.0,  # %
    "performance_drop_significant": -10.0,  # % from baseline
    "performance_drop_concern": -5.0,  # %
    "minimum_tests_for_trend": 3,  # Need at least 3 tests to identify trend
    "days_between_tests_optimal": 7,  # Ideal testing frequency
    "days_between_tests_minimum": 3,  # Don't test more frequently
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_sport_benchmarks(sport_name: str) -> SportBenchmarks:
    """Get benchmarks for a specific sport"""
    # Handle non-string inputs
    if not isinstance(sport_name, str):
        sport_name = str(sport_name) if sport_name is not None else "Athletics"

    # Try exact match first
    if sport_name in SPORT_BENCHMARKS:
        return SPORT_BENCHMARKS[sport_name]

    # Try partial match (case insensitive)
    sport_lower = sport_name.lower()
    for key in SPORT_BENCHMARKS:
        if sport_lower in key.lower() or key.lower() in sport_lower:
            return SPORT_BENCHMARKS[key]

    # Return general athletics as default
    return SPORT_BENCHMARKS["Athletics"]


def get_percentile_rank(value: float, sport: str, metric: str) -> float:
    """
    Calculate percentile rank for a metric within sport context
    Returns 0-100 percentile
    """
    benchmarks = get_sport_benchmarks(sport)

    # Map metric to benchmark attribute
    if "Jump Height" in metric or "jump height" in metric.lower():
        excellent = benchmarks.cmj_height_excellent
        good = benchmarks.cmj_height_good
        average = benchmarks.cmj_height_average
    elif "RSI" in metric:
        excellent = benchmarks.rsi_excellent
        good = benchmarks.rsi_good
        average = benchmarks.rsi_average
    elif "Relative" in metric and "Force" in metric:
        excellent = benchmarks.peak_force_excellent
        good = benchmarks.peak_force_good
        average = benchmarks.peak_force_average
    else:
        return 50.0  # Default to 50th percentile if unknown

    # Handle None values (para athletes, etc.)
    if excellent is None or value is None:
        return None

    # Calculate percentile (simplified linear interpolation)
    if value >= excellent:
        return 90.0 + min(10.0, (value - excellent) / excellent * 100)
    elif value >= good:
        # Between 70th-90th percentile
        pct_range = (value - good) / (excellent - good)
        return 70.0 + (pct_range * 20.0)
    elif value >= average:
        # Between 40th-70th percentile
        pct_range = (value - average) / (good - average)
        return 40.0 + (pct_range * 30.0)
    else:
        # Below 40th percentile
        return max(5.0, 40.0 * (value / average))


def get_metric_status(value: float, sport: str, metric: str) -> str:
    """
    Get status label for a metric value
    Returns: 'Excellent', 'Good', 'Average', 'Below Average', 'Poor'
    """
    percentile = get_percentile_rank(value, sport, metric)

    if percentile is None:
        return "N/A"
    elif percentile >= 85:
        return "Excellent"
    elif percentile >= 65:
        return "Good"
    elif percentile >= 35:
        return "Average"
    elif percentile >= 15:
        return "Below Average"
    else:
        return "Needs Attention"


def get_asymmetry_status(asymmetry_value: float, sport: str) -> tuple:
    """
    Get asymmetry status and color
    Returns: (status_text, color_code)
    """
    benchmarks = get_sport_benchmarks(sport)

    if asymmetry_value is None:
        return ("Unknown", "gray")

    abs_asymmetry = abs(asymmetry_value)

    if abs_asymmetry >= benchmarks.asymmetry_risk:
        return ("High Risk", "red")
    elif abs_asymmetry >= benchmarks.asymmetry_caution:
        return ("Caution", "orange")
    else:
        return ("Acceptable", "green")


def get_priority_metrics_for_sport(sport: str) -> List[str]:
    """Get the priority metrics for a specific sport"""
    benchmarks = get_sport_benchmarks(sport)
    return benchmarks.priority_metrics if benchmarks.priority_metrics else []


def get_sport_context(sport: str) -> Dict[str, Any]:
    """Get complete context for a sport including benchmarks and interpretation"""
    benchmarks = get_sport_benchmarks(sport)

    return {
        "sport_name": benchmarks.sport_name,
        "benchmarks": {
            "cmj_height": {
                "excellent": benchmarks.cmj_height_excellent,
                "good": benchmarks.cmj_height_good,
                "average": benchmarks.cmj_height_average
            },
            "rsi": {
                "excellent": benchmarks.rsi_excellent,
                "good": benchmarks.rsi_good,
                "average": benchmarks.rsi_average
            },
            "peak_force": {
                "excellent": benchmarks.peak_force_excellent,
                "good": benchmarks.peak_force_good,
                "average": benchmarks.peak_force_average
            }
        },
        "asymmetry_thresholds": {
            "caution": benchmarks.asymmetry_caution,
            "risk": benchmarks.asymmetry_risk
        },
        "priority_metrics": benchmarks.priority_metrics,
        "key_attributes": benchmarks.key_attributes,
        "context": benchmarks.context
    }


# ============================================================================
# TEST TYPE CONFIGURATIONS
# ============================================================================

TEST_TYPE_CONFIG = {
    "CMJ": {
        "full_name": "Countermovement Jump",
        "description": "Bilateral jump with arm swing - measures total body power",
        "key_metrics": ["Jump Height (Flight Time) [cm]", "Peak Power [W]", "RSI Modified"],
        "color": "#1f77b4"
    },
    "SJ": {
        "full_name": "Squat Jump",
        "description": "Bilateral jump from static position - measures concentric strength",
        "key_metrics": ["Jump Height (Flight Time) [cm]", "Peak Force [N]", "Rate of Force Development"],
        "color": "#ff7f0e"
    },
    "ISOSQT": {
        "full_name": "Isometric Squat",
        "description": "Static squat hold - measures isometric strength",
        "key_metrics": ["Peak Force [N]", "Relative Peak Force [N/kg]", "RFD"],
        "color": "#2ca02c"
    },
    "SLCMJ": {
        "full_name": "Single Leg Countermovement Jump",
        "description": "Unilateral jump - measures single leg power and asymmetry",
        "key_metrics": ["Jump Height (Flight Time) [cm]", "Asymmetry Index", "Peak Force [N]"],
        "color": "#d62728"
    },
    "DJ": {
        "full_name": "Drop Jump",
        "description": "Plyometric jump from height - measures reactive strength",
        "key_metrics": ["RSI Modified", "Contact Time [ms]", "Jump Height (Flight Time) [cm]"],
        "color": "#9467bd"
    },
}


if __name__ == "__main__":
    # Test the configuration
    print("Sport Benchmarks Loaded:")
    print(f"Total sports configured: {len(SPORT_BENCHMARKS)}")

    # Test a specific sport
    weightlifting = get_sport_benchmarks("Weightlifting")
    print(f"\nWeightlifting Benchmarks:")
    print(f"  CMJ Excellent: {weightlifting.cmj_height_excellent}cm")
    print(f"  Priority Metrics: {weightlifting.priority_metrics}")

    # Test percentile calculation
    test_value = 48.0  # cm
    percentile = get_percentile_rank(test_value, "Weightlifting", "Jump Height")
    status = get_metric_status(test_value, "Weightlifting", "Jump Height")
    print(f"\nTest: 48cm CMJ for Weightlifter")
    print(f"  Percentile: {percentile:.1f}")
    print(f"  Status: {status}")
