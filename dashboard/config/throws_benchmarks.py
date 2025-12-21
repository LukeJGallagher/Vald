"""
Extended Benchmarks for Throwing Events
Team Saudi - VALD Performance Analysis

Detailed benchmarks for Shot Put, Discus, Javelin, Hammer
Based on research and elite athlete profiles

Key Sources:
- Young et al. (2015) - Physical characteristics of elite throwers
- Zaras et al. (2013) - Power-force-velocity profiles in throwers
- Morriss et al. (1997) - Kinematics and kinetics in javelin throwing
"""

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ThrowsSpecificBenchmarks:
    """Detailed benchmarks for specific throwing events"""
    event_name: str

    # Jump Metrics
    cmj_height_elite: float  # cm - International/Olympic level
    cmj_height_national: float  # cm - National competitive
    cmj_height_development: float  # cm - Development athletes

    # Power Metrics
    peak_power_elite: float  # W
    peak_power_national: float
    peak_power_development: float

    power_to_weight_elite: float  # W/kg
    power_to_weight_national: float
    power_to_weight_development: float

    # Force Metrics
    peak_force_elite: float  # N
    peak_force_national: float
    peak_force_development: float

    relative_force_elite: float  # N/kg
    relative_force_national: float
    relative_force_development: float

    # Event-Specific
    throw_distance_elite: float  # meters
    throw_distance_national: float
    throw_distance_development: float

    implement_weight: str  # e.g., "7.26kg (men), 4kg (women)"

    # Training emphasis
    primary_qualities: List[str]
    secondary_qualities: List[str]

    # Technical notes
    biomechanical_notes: str


# ============================================================================
# SHOT PUT BENCHMARKS
# ============================================================================

SHOT_PUT_BENCHMARKS = ThrowsSpecificBenchmarks(
    event_name="Shot Put",

    # Jump Metrics (Shot putters typically highest jumps among throwers)
    cmj_height_elite=52.0,  # >50cm typical for elite
    cmj_height_national=48.0,
    cmj_height_development=44.0,

    # Power (Highest absolute power requirements)
    peak_power_elite=6500,  # W - Elite shot putters
    peak_power_national=5800,
    peak_power_development=5000,

    power_to_weight_elite=70,  # W/kg
    power_to_weight_national=65,
    power_to_weight_development=58,

    # Force (Very high force demands)
    peak_force_elite=4000,  # N
    peak_force_national=3600,
    peak_force_development=3200,

    relative_force_elite=38,  # N/kg
    relative_force_national=35,
    relative_force_development=31,

    # Performance
    throw_distance_elite=20.0,  # meters (men - 7.26kg)
    throw_distance_national=18.0,
    throw_distance_development=16.0,

    implement_weight="7.26kg (men), 4kg (women)",

    # Training
    primary_qualities=[
        "Maximal strength (squats, deadlifts)",
        "Explosive power (Olympic lifts)",
        "Rate of force development",
        "Upper body power (bench throws)"
    ],
    secondary_qualities=[
        "Rotational mobility",
        "Core stability",
        "Technical efficiency (glide vs rotational)"
    ],

    biomechanical_notes="""
    Shot Put Characteristics:
    - Shortest throw duration (~0.3s from start to release)
    - Highest force application requirements
    - Bilateral power critical (both glide and rotational techniques)
    - Upper/lower body integration essential
    - Relative strength important despite large body mass
    - Power production in <200ms window

    Key Performance Indicators:
    1. CMJ height >48cm (correlates r=0.72 with throw distance)
    2. Peak power >5500W
    3. Back squat >2.2x body weight
    4. Clean >1.4x body weight
    """
)


# ============================================================================
# DISCUS BENCHMARKS
# ============================================================================

DISCUS_BENCHMARKS = ThrowsSpecificBenchmarks(
    event_name="Discus",

    # Jump Metrics (Slightly lower than shot, higher rotation demands)
    cmj_height_elite=48.0,
    cmj_height_national=44.0,
    cmj_height_development=40.0,

    # Power (High rotational power)
    peak_power_elite=6000,
    peak_power_national=5400,
    peak_power_development=4800,

    power_to_weight_elite=65,
    power_to_weight_national=60,
    power_to_weight_development=55,

    # Force
    peak_force_elite=3800,
    peak_force_national=3400,
    peak_force_development=3000,

    relative_force_elite=36,
    relative_force_national=33,
    relative_force_development=29,

    # Performance
    throw_distance_elite=65.0,  # meters (men - 2kg)
    throw_distance_national=58.0,
    throw_distance_development=52.0,

    implement_weight="2kg (men), 1kg (women)",

    # Training
    primary_qualities=[
        "Rotational power (medicine ball throws)",
        "Lower body power (rotational jumps)",
        "Core rotational strength",
        "Multi-directional force application"
    ],
    secondary_qualities=[
        "Hip mobility/flexibility",
        "Shoulder stability",
        "Balance during rotation",
        "Timing and rhythm"
    ],

    biomechanical_notes="""
    Discus Characteristics:
    - Multi-turn rotational technique (1.5-2 turns)
    - Sequential force summation (ground → legs → core → arm)
    - Asymmetrical loading (lead leg dominance)
    - Long acceleration path (~2.5m diameter circle)
    - Release speed >25 m/s for elite

    Key Performance Indicators:
    1. Rotational power tests (medicine ball throws)
    2. Single-leg jump asymmetry (some asymmetry expected)
    3. Hip rotation ROM >70° each direction
    4. CMJ >45cm with good RSI (>2.0)

    Training Note: 10-15% asymmetry acceptable due to rotational demands
    """
)


# ============================================================================
# JAVELIN BENCHMARKS
# ============================================================================

JAVELIN_BENCHMARKS = ThrowsSpecificBenchmarks(
    event_name="Javelin",

    # Jump Metrics (Highest among throwers - run-up component)
    cmj_height_elite=54.0,  # Javelin throwers often jump highest
    cmj_height_national=50.0,
    cmj_height_development=46.0,

    # Power (Speed-strength emphasis)
    peak_power_elite=5800,
    peak_power_national=5200,
    peak_power_development=4600,

    power_to_weight_elite=68,  # Higher relative power (lighter athletes)
    power_to_weight_national=62,
    power_to_weight_development=56,

    # Force (Lower absolute, higher relative)
    peak_force_elite=3400,
    peak_force_national=3100,
    peak_force_development=2800,

    relative_force_elite=37,
    relative_force_national=34,
    relative_force_development=30,

    # Performance
    throw_distance_elite=85.0,  # meters (men - 800g)
    throw_distance_national=75.0,
    throw_distance_development=68.0,

    implement_weight="800g (men), 600g (women)",

    # Training
    primary_qualities=[
        "Sprint speed (run-up)",
        "Explosive upper body power",
        "Shoulder strength and stability",
        "Unilateral lower body power"
    ],
    secondary_qualities=[
        "Thoracic mobility",
        "Deceleration capacity (shoulder health)",
        "Reactive strength (RSI >2.2)",
        "Core anti-rotation strength"
    ],

    biomechanical_notes="""
    Javelin Characteristics:
    - Run-up approach (12-18 strides) → requires sprint speed
    - Overhead throwing pattern (unique among throws)
    - Extreme shoulder external rotation and extension
    - Lower body → upper body force transfer critical
    - Release velocity >30 m/s (elite men)

    Key Performance Indicators:
    1. 30m sprint time <4.2s (correlates with run-up speed)
    2. CMJ >50cm with excellent RSI
    3. Single-leg CMJ >85% of bilateral
    4. Overhead medicine ball throw distance

    Injury Prevention:
    - Shoulder deceleration strength critical
    - Lower body asymmetry should be <12% (run-up efficiency)
    - Thoracic rotation >50° (protect shoulder)
    """
)


# ============================================================================
# HAMMER BENCHMARKS
# ============================================================================

HAMMER_BENCHMARKS = ThrowsSpecificBenchmarks(
    event_name="Hammer",

    # Jump Metrics (Moderate - not primary quality)
    cmj_height_elite=46.0,
    cmj_height_national=42.0,
    cmj_height_development=38.0,

    # Power (Very high rotational power)
    peak_power_elite=6200,
    peak_power_national=5600,
    peak_power_development=5000,

    power_to_weight_elite=62,
    power_to_weight_national=57,
    power_to_weight_development=52,

    # Force (High force tolerance)
    peak_force_elite=4200,  # Highest among all throws
    peak_force_national=3800,
    peak_force_development=3400,

    relative_force_elite=37,
    relative_force_national=34,
    relative_force_development=30,

    # Performance
    throw_distance_elite=78.0,  # meters (men - 7.26kg)
    throw_distance_national=70.0,
    throw_distance_development=62.0,

    implement_weight="7.26kg (men), 4kg (women)",

    # Training
    primary_qualities=[
        "Rotational strength and power",
        "Eccentric force absorption (3-4 turns)",
        "Grip and forearm strength",
        "Balance and proprioception during rotation"
    ],
    secondary_qualities=[
        "Core stability (anti-flexion)",
        "Hip and ankle mobility",
        "Centrifugal force tolerance",
        "Visual-spatial awareness (turning)"
    ],

    biomechanical_notes="""
    Hammer Throw Characteristics:
    - Multiple rotations (3-4 turns) before release
    - Highest centrifugal forces of all throws (~400kg pulling force)
    - Longest acceleration path (3-4 turns × 2.135m diameter)
    - Counter-balancing implement weight during turns
    - Release speed >28 m/s

    Key Performance Indicators:
    1. Rotational power tests (standing + turning)
    2. Grip strength endurance (hold heavy implement through turns)
    3. Single-leg balance (eyes closed >15s each leg)
    4. Peak force >3800N (withstand centrifugal loading)

    Training Note:
    - Asymmetry up to 12% acceptable (rotational sport)
    - Focus on rotational patterns vs vertical jump
    - Eccentric strength critical for deceleration between turns
    """
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_throws_benchmark(event: str) -> ThrowsSpecificBenchmarks:
    """Get benchmarks for specific throwing event"""
    event_lower = event.lower().strip()

    if 'shot' in event_lower or 'put' in event_lower:
        return SHOT_PUT_BENCHMARKS
    elif 'discus' in event_lower:
        return DISCUS_BENCHMARKS
    elif 'javelin' in event_lower:
        return JAVELIN_BENCHMARKS
    elif 'hammer' in event_lower:
        return HAMMER_BENCHMARKS
    else:
        # Return shot put as default for general "throws"
        return SHOT_PUT_BENCHMARKS


def get_performance_level(metric_value: float, elite: float, national: float, development: float, higher_is_better: bool = True) -> str:
    """
    Determine performance level based on benchmarks

    Returns: 'Elite', 'National', 'Development', or 'Below Development'
    """
    if higher_is_better:
        if metric_value >= elite:
            return 'Elite'
        elif metric_value >= national:
            return 'National'
        elif metric_value >= development:
            return 'Development'
        else:
            return 'Below Development'
    else:
        # For metrics where lower is better (not common in throws)
        if metric_value <= elite:
            return 'Elite'
        elif metric_value <= national:
            return 'National'
        elif metric_value <= development:
            return 'Development'
        else:
            return 'Below Development'


def compare_to_benchmark(athlete_metrics: Dict[str, float], event: str) -> Dict[str, str]:
    """
    Compare athlete metrics to benchmarks

    Parameters:
    -----------
    athlete_metrics : Dict[str, float]
        {'cmj_height': 48.5, 'peak_power': 5600, etc.}
    event : str
        'Shot Put', 'Discus', 'Javelin', 'Hammer'

    Returns:
    --------
    Dict with performance levels for each metric
    """
    benchmark = get_throws_benchmark(event)
    results = {}

    metric_map = {
        'cmj_height': (benchmark.cmj_height_elite, benchmark.cmj_height_national, benchmark.cmj_height_development),
        'peak_power': (benchmark.peak_power_elite, benchmark.peak_power_national, benchmark.peak_power_development),
        'power_to_weight': (benchmark.power_to_weight_elite, benchmark.power_to_weight_national, benchmark.power_to_weight_development),
        'peak_force': (benchmark.peak_force_elite, benchmark.peak_force_national, benchmark.peak_force_development),
        'relative_force': (benchmark.relative_force_elite, benchmark.relative_force_national, benchmark.relative_force_development),
    }

    for metric_name, athlete_value in athlete_metrics.items():
        if metric_name in metric_map:
            elite, national, dev = metric_map[metric_name]
            level = get_performance_level(athlete_value, elite, national, dev)
            results[metric_name] = level

    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Throws-Specific Benchmarks Module")
    print("=" * 80)

    # Example: Shot Put athlete assessment
    athlete_data = {
        'cmj_height': 48.5,
        'peak_power': 5600,
        'power_to_weight': 64,
        'peak_force': 3500,
        'relative_force': 34
    }

    print("\nExample: Shot Put Athlete Assessment")
    print("-" * 80)

    benchmark = get_throws_benchmark('Shot Put')
    comparison = compare_to_benchmark(athlete_data, 'Shot Put')

    print(f"Event: {benchmark.event_name}")
    print(f"Implement: {benchmark.implement_weight}")
    print(f"\nAthlete Metrics vs Benchmarks:")

    for metric, level in comparison.items():
        print(f"  {metric}: {athlete_data[metric]:.1f} → {level}")

    print(f"\nPrimary Training Qualities:")
    for quality in benchmark.primary_qualities:
        print(f"  • {quality}")

    print("\n" + "=" * 80)
    print("Module loaded successfully!")
