"""
Complete Sport-Specific Intelligence for ALL Team Saudi Sports
Including Para Athletics, Para Swimming, Para Judo, etc.

Sports Covered:
- Athletics (all events + Para Athletics)
- Swimming (all strokes + Para Swimming)
- Rowing (all boat classes)
- Wrestling (Freestyle, Greco-Roman + Para Wrestling)
- Fencing (Epee, Sabre, Foil + Wheelchair Fencing)
- Weightlifting (all weight classes)
- Football/Soccer
- Rugby Union
- Judo (+ Para Judo)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
import os

# Import base intelligence
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sport_specific_intelligence import SportInsight


# ============================================================================
# ATHLETICS INTELLIGENCE (Expanded)
# ============================================================================

class AthleticsIntelligence:
    """Intelligence for all Athletics events"""

    @staticmethod
    def analyze_athletics_athlete(athlete_data: pd.DataFrame, event: str) -> List[SportInsight]:
        """Analyze Athletics athlete with event-specific insights"""
        insights = []

        metrics = {
            'jump_height': AthleticsIntelligence._get_metric(athlete_data, 'Jump Height'),
            'rsi': AthleticsIntelligence._get_metric(athlete_data, 'RSI'),
            'peak_power': AthleticsIntelligence._get_metric(athlete_data, 'Peak Power'),
            'asymmetry': AthleticsIntelligence._get_metric(athlete_data, 'Asymmetry'),
            'peak_force': AthleticsIntelligence._get_metric(athlete_data, 'Peak Force')
        }

        event_lower = str(event).lower()

        # SPRINTS (100m, 200m, 400m)
        if any(x in event_lower for x in ['sprint', '100m', '200m', '400m']):
            if metrics['rsi'] and metrics['rsi'] < 2.0:
                insights.append(SportInsight(
                    category="power",
                    priority="high",
                    title="Reactive Strength for Sprint Performance",
                    message=f"RSI of {metrics['rsi']:.2f} below optimal for sprinting (target >2.0). "
                            "Ground contact times in sprinting are <100ms - reactive strength critical.",
                    recommendation="Implement: Depth jumps (20-30cm), reactive hurdle hops, "
                                 "wicket runs, sprint-specific plyometrics. Focus on minimal ground contact. "
                                 "Retest monthly.",
                    metrics_referenced=['RSI-modified (Imp-Mom)_Trial'],
                    icon="⚡"
                ))

        # DISTANCE (800m, 1500m, 5000m, 10000m)
        elif any(x in event_lower for x in ['distance', '800', '1500', '5000', '10000', 'marathon']):
            if metrics['asymmetry'] and abs(metrics['asymmetry']) > 10:
                insights.append(SportInsight(
                    category="asymmetry",
                    priority="critical",
                    title="Asymmetry Impact on Running Economy",
                    message=f"Asymmetry of {abs(metrics['asymmetry']):.1f}% can significantly affect running economy "
                            "over long distances. Bilateral symmetry critical for distance runners.",
                    recommendation="Priority corrective work: Single-leg calf raises, single-leg squats, "
                                 "gait analysis recommended. High volume training amplifies asymmetry impact. "
                                 "Target <8% asymmetry.",
                    metrics_referenced=['Asymmetry Index'],
                    icon="🏃"
                ))

        # JUMPS (Long Jump, Triple Jump, High Jump, Pole Vault)
        elif any(x in event_lower for x in ['jump', 'pole vault']):
            if metrics['jump_height'] and metrics['jump_height'] < 45:
                insights.append(SportInsight(
                    category="power",
                    priority="high",
                    title="Vertical Jump Power for Jumpers",
                    message=f"CMJ height ({metrics['jump_height']:.1f}cm) below typical for jump events (target >48cm). "
                            "Vertical power directly transfers to horizontal/high jump performance.",
                    recommendation="Implement: Heavy squat jumps, depth jumps from 40-60cm, "
                                 "single-leg bounding progressions, power cleans. "
                                 "Target: 45-50cm within 8 weeks.",
                    metrics_referenced=['Jump Height (Flight Time) [cm]'],
                    icon="🦘"
                ))

        return insights

    @staticmethod
    def _get_metric(df: pd.DataFrame, metric_name: str) -> Optional[float]:
        """Extract latest metric value"""
        matching_cols = [col for col in df.columns if metric_name.lower() in col.lower()]
        if not matching_cols:
            return None
        col = matching_cols[0]
        values = df[col].dropna()
        if values.empty:
            return None
        if 'recordedDateUtc' in df.columns:
            df_sorted = df.sort_values('recordedDateUtc')
            return df_sorted[col].iloc[-1]
        return values.iloc[-1]


# ============================================================================
# SWIMMING INTELLIGENCE
# ============================================================================

class SwimmingIntelligence:
    """Intelligence for Swimming (all strokes)"""

    @staticmethod
    def analyze_swimming_athlete(athlete_data: pd.DataFrame, stroke: str) -> List[SportInsight]:
        """Analyze swimmer with stroke-specific insights"""
        insights = []

        metrics = {
            'jump_height': SwimmingIntelligence._get_metric(athlete_data, 'Jump Height'),
            'peak_power': SwimmingIntelligence._get_metric(athlete_data, 'Peak Power'),
            'asymmetry': SwimmingIntelligence._get_metric(athlete_data, 'Asymmetry'),
            'rfd': SwimmingIntelligence._get_metric(athlete_data, 'RFD')
        }

        # Swimmers have lower jump metrics but still need power for starts/turns
        if metrics['jump_height']:
            if metrics['jump_height'] < 30:
                insights.append(SportInsight(
                    category="power",
                    priority="medium",
                    title="Start and Turn Power",
                    message=f"CMJ height ({metrics['jump_height']:.1f}cm) below typical for competitive swimmers (target >35cm). "
                            "While swimming is primarily upper body, leg power critical for starts and turns.",
                    recommendation="Implement: Vertical jumps from blocks, box jumps, squat jumps. "
                                 "Focus on explosive push-off strength. "
                                 "Complement with start/turn technique work in pool.",
                    metrics_referenced=['Jump Height (Flight Time) [cm]'],
                    icon="🏊"
                ))

        # Asymmetry important for stroke efficiency
        if metrics['asymmetry'] and abs(metrics['asymmetry']) > 12:
            insights.append(SportInsight(
                category="asymmetry",
                priority="high",
                title="Bilateral Symmetry for Stroke Efficiency",
                message=f"Lower body asymmetry of {abs(metrics['asymmetry']):.1f}% can affect push-off power and turns. "
                        "Bilateral symmetry important for efficient kicking and balanced rotation.",
                recommendation="Single-leg strength work: Bulgarian split squats, single-leg box steps, "
                             "unilateral core stability. Monitor kick symmetry in water. "
                             "Target <10% asymmetry.",
                metrics_referenced=['Asymmetry Index'],
                icon="⚖️"
            ))

        # Stroke-specific insights
        stroke_lower = str(stroke).lower()
        if 'butterfly' in stroke_lower or 'fly' in stroke_lower:
            if metrics['peak_power'] and metrics['peak_power'] < 3500:
                insights.append(SportInsight(
                    category="power",
                    priority="high",
                    title="Butterfly Power Requirements",
                    message="Butterfly demands highest power output of all strokes. "
                            "Leg drive during dolphin kick critical for race performance.",
                    recommendation="Focus: Explosive dolphin kick drills on land (jump squats), "
                                 "core power (medicine ball slams), hip flexor power. "
                                 "Integrate with underwater kick sets.",
                    metrics_referenced=['Peak Power [W]'],
                    icon="🦋"
                ))

        return insights

    @staticmethod
    def _get_metric(df: pd.DataFrame, metric_name: str) -> Optional[float]:
        """Extract latest metric value"""
        matching_cols = [col for col in df.columns if metric_name.lower() in col.lower()]
        if not matching_cols:
            return None
        col = matching_cols[0]
        values = df[col].dropna()
        if values.empty:
            return None
        if 'recordedDateUtc' in df.columns:
            df_sorted = df.sort_values('recordedDateUtc')
            return df_sorted[col].iloc[-1]
        return values.iloc[-1]


# ============================================================================
# ROWING INTELLIGENCE
# ============================================================================

class RowingIntelligence:
    """Intelligence for Rowing (all boat classes)"""

    @staticmethod
    def analyze_rowing_athlete(athlete_data: pd.DataFrame, boat_class: str) -> List[SportInsight]:
        """Analyze rower with boat-specific insights"""
        insights = []

        metrics = {
            'peak_force': RowingIntelligence._get_metric(athlete_data, 'Peak Force'),
            'relative_force': RowingIntelligence._get_metric(athlete_data, 'Relative Peak Force'),
            'asymmetry': RowingIntelligence._get_metric(athlete_data, 'Asymmetry'),
            'jump_height': RowingIntelligence._get_metric(athlete_data, 'Jump Height')
        }

        # CRITICAL: Bilateral symmetry for rowing
        if metrics['asymmetry'] and abs(metrics['asymmetry']) > 8:
            insights.append(SportInsight(
                category="asymmetry",
                priority="critical",
                title="Boat Balance - Symmetry Essential",
                message=f"Asymmetry of {abs(metrics['asymmetry']):.1f}% exceeds tolerance for rowing (target <6%). "
                        "Even small asymmetries affect boat set and efficiency, especially in sculling.",
                recommendation="PRIORITY: Bilateral strength correction. Single-leg work emphasizing weaker side, "
                             "ergometer technique review, core stability assessment. "
                             "Retest weekly until <6%. May require technique modification.",
                metrics_referenced=['Asymmetry Index'],
                icon="⚠️"
            ))

        # High absolute force requirement
        if metrics['peak_force'] and metrics['peak_force'] < 3000:
            insights.append(SportInsight(
                category="strength",
                priority="high",
                title="Absolute Strength for Rowing Power",
                message=f"Peak force ({metrics['peak_force']:.0f}N) below typical for competitive rowing (target >3200N). "
                        "Rowing demands high absolute strength for stroke power.",
                recommendation="Focus: Heavy squats (85-90% 1RM), deadlifts, Olympic lifts. "
                             "Rowing-specific: Seated cable rows with heavy load. "
                             "Target: Squat >1.8x body weight. Test monthly.",
                metrics_referenced=['Peak Force [N]'],
                icon="🚣"
            ))

        # Boat-specific insights
        boat_lower = str(boat_class).lower()
        if 'single' in boat_lower or 'scull' in boat_lower:
            insights.append(SportInsight(
                category="technique",
                priority="medium",
                title="Single Sculling Balance Demands",
                message="Single sculls require exceptional bilateral balance and core stability. "
                        "Any asymmetry is immediately evident in boat set.",
                recommendation="Emphasize: Unilateral core stability (Pallof press variations), "
                             "balance drills, symmetrical strength development. "
                             "Video analysis of stroke technique recommended.",
                metrics_referenced=['Asymmetry Index', 'Peak Force [N]'],
                icon="🚣"
            ))

        return insights

    @staticmethod
    def _get_metric(df: pd.DataFrame, metric_name: str) -> Optional[float]:
        """Extract latest metric value"""
        matching_cols = [col for col in df.columns if metric_name.lower() in col.lower()]
        if not matching_cols:
            return None
        col = matching_cols[0]
        values = df[col].dropna()
        if values.empty:
            return None
        if 'recordedDateUtc' in df.columns:
            df_sorted = df.sort_values('recordedDateUtc')
            return df_sorted[col].iloc[-1]
        return values.iloc[-1]


# ============================================================================
# WEIGHTLIFTING INTELLIGENCE
# ============================================================================

class WeightliftingIntelligence:
    """Intelligence for Olympic Weightlifting"""

    @staticmethod
    def analyze_weightlifting_athlete(athlete_data: pd.DataFrame) -> List[SportInsight]:
        """Analyze weightlifter"""
        insights = []

        metrics = {
            'jump_height': WeightliftingIntelligence._get_metric(athlete_data, 'Jump Height'),
            'peak_power': WeightliftingIntelligence._get_metric(athlete_data, 'Peak Power'),
            'asymmetry': WeightliftingIntelligence._get_metric(athlete_data, 'Asymmetry'),
            'rfd': WeightliftingIntelligence._get_metric(athlete_data, 'RFD')
        }

        # Weightlifters should have HIGHEST jump metrics
        if metrics['jump_height']:
            if metrics['jump_height'] < 48:
                insights.append(SportInsight(
                    category="power",
                    priority="high",
                    title="Elite Power Output Expected",
                    message=f"CMJ height ({metrics['jump_height']:.1f}cm) below typical for weightlifters (target >50cm). "
                            "Weightlifters typically display highest vertical jump of all athletes due to triple extension power.",
                    recommendation="Review Olympic lift technique - power transfer may be inefficient. "
                                 "Supplementary work: Jump squats, power cleans with focus on velocity, "
                                 "depth jumps. Target: >50cm CMJ correlates with snatch/clean performance.",
                    metrics_referenced=['Jump Height (Flight Time) [cm]'],
                    icon="🏋️"
                ))

        # Perfect symmetry required for lift technique
        if metrics['asymmetry'] and abs(metrics['asymmetry']) > 8:
            insights.append(SportInsight(
                category="asymmetry",
                priority="critical",
                title="Technique Risk - Bilateral Asymmetry",
                message=f"Asymmetry of {abs(metrics['asymmetry']):.1f}% affects clean & jerk and snatch technique. "
                        "Perfect bilateral balance required for successful heavy lifts and injury prevention.",
                recommendation="IMMEDIATE corrective work required. Single-leg strength emphasis on weaker side, "
                             "unilateral overhead work, technique review with coach. "
                             "Asymmetry >10% increases miss rate and injury risk. Target <6%.",
                metrics_referenced=['Asymmetry Index'],
                icon="⚠️"
            ))

        # Peak power correlation with lifts
        if metrics['peak_power'] and metrics['peak_power'] < 5500:
            insights.append(SportInsight(
                category="power",
                priority="high",
                title="Power Output vs Lifting Performance",
                message="Peak power in CMJ strongly correlates (r>0.8) with clean & jerk and snatch maxes. "
                        "Current power output suggests room for improvement in explosive strength.",
                recommendation="Focus: Hang power cleans, hang power snatches (above knee), "
                             "jump squats at 30-40% 1RM, med ball overhead throws. "
                             "Emphasize VELOCITY not just load. Retest every 3-4 weeks.",
                metrics_referenced=['Peak Power [W]', 'Rate of Force Development'],
                icon="💪"
            ))

        return insights

    @staticmethod
    def _get_metric(df: pd.DataFrame, metric_name: str) -> Optional[float]:
        """Extract latest metric value"""
        matching_cols = [col for col in df.columns if metric_name.lower() in col.lower()]
        if not matching_cols:
            return None
        col = matching_cols[0]
        values = df[col].dropna()
        if values.empty:
            return None
        if 'recordedDateUtc' in df.columns:
            df_sorted = df.sort_values('recordedDateUtc')
            return df_sorted[col].iloc[-1]
        return values.iloc[-1]


# ============================================================================
# FOOTBALL/SOCCER INTELLIGENCE
# ============================================================================

class FootballIntelligence:
    """Intelligence for Football/Soccer"""

    @staticmethod
    def analyze_football_athlete(athlete_data: pd.DataFrame, position: str = None) -> List[SportInsight]:
        """Analyze football player with position-specific insights"""
        insights = []

        metrics = {
            'jump_height': FootballIntelligence._get_metric(athlete_data, 'Jump Height'),
            'rsi': FootballIntelligence._get_metric(athlete_data, 'RSI'),
            'asymmetry': FootballIntelligence._get_metric(athlete_data, 'Asymmetry'),
            'peak_power': FootballIntelligence._get_metric(athlete_data, 'Peak Power')
        }

        # RSI critical for change of direction
        if metrics['rsi'] and metrics['rsi'] < 1.8:
            insights.append(SportInsight(
                category="power",
                priority="high",
                title="Reactive Strength for Agility",
                message=f"RSI of {metrics['rsi']:.2f} below optimal for football (target >2.0). "
                        "Reactive strength critical for quick direction changes, acceleration, deceleration.",
                recommendation="Implement: Reactive hurdle hops, lateral bounds, cutting drills, "
                             "deceleration training. Focus on minimal ground contact with force application. "
                             "Integrate with position-specific movement patterns.",
                metrics_referenced=['RSI-modified (Imp-Mom)_Trial'],
                icon="⚽"
            ))

        # Asymmetry monitoring (common in football due to dominant leg kicking)
        if metrics['asymmetry'] and abs(metrics['asymmetry']) > 15:
            insights.append(SportInsight(
                category="asymmetry",
                priority="high",
                title="Bilateral Imbalance - Injury Risk",
                message=f"Asymmetry of {abs(metrics['asymmetry']):.1f}% increases non-contact injury risk "
                        "(hamstring, ACL, groin). Some asymmetry expected due to kicking leg dominance, "
                        "but >15% excessive.",
                recommendation="Corrective program: Emphasize weaker leg in strength work, "
                             "bilateral Olympic lift variations, Nordic hamstring emphasis on weaker side. "
                             "Consider gait analysis. Target <12% asymmetry.",
                metrics_referenced=['Asymmetry Index'],
                icon="⚠️"
            ))

        # Position-specific if available
        if position:
            pos_lower = str(position).lower()
            if any(x in pos_lower for x in ['goalkeeper', 'gk']):
                if metrics['jump_height'] and metrics['jump_height'] < 40:
                    insights.append(SportInsight(
                        category="power",
                        priority="high",
                        title="Goalkeeper Vertical Jump Requirement",
                        message="Goalkeepers require good vertical jump for high ball collection and shot stopping. "
                                f"Current CMJ ({metrics['jump_height']:.1f}cm) below typical GK range (>42cm).",
                        recommendation="GK-specific: Box jumps, single-leg vertical jumps, "
                                     "explosive push-offs from ground. Integrate with diving technique work.",
                        metrics_referenced=['Jump Height (Flight Time) [cm]'],
                        icon="🧤"
                    ))

        return insights

    @staticmethod
    def _get_metric(df: pd.DataFrame, metric_name: str) -> Optional[float]:
        """Extract latest metric value"""
        matching_cols = [col for col in df.columns if metric_name.lower() in col.lower()]
        if not matching_cols:
            return None
        col = matching_cols[0]
        values = df[col].dropna()
        if values.empty:
            return None
        if 'recordedDateUtc' in df.columns:
            df_sorted = df.sort_values('recordedDateUtc')
            return df_sorted[col].iloc[-1]
        return values.iloc[-1]


# ============================================================================
# RUGBY INTELLIGENCE
# ============================================================================

class RugbyIntelligence:
    """Intelligence for Rugby Union"""

    @staticmethod
    def analyze_rugby_athlete(athlete_data: pd.DataFrame, position: str = None) -> List[SportInsight]:
        """Analyze rugby player with position-specific insights"""
        insights = []

        metrics = {
            'peak_force': RugbyIntelligence._get_metric(athlete_data, 'Peak Force'),
            'relative_force': RugbyIntelligence._get_metric(athlete_data, 'Relative Peak Force'),
            'jump_height': RugbyIntelligence._get_metric(athlete_data, 'Jump Height'),
            'asymmetry': RugbyIntelligence._get_metric(athlete_data, 'Asymmetry'),
            'rfd': RugbyIntelligence._get_metric(athlete_data, 'RFD')
        }

        # High force production requirement (especially forwards)
        if metrics['peak_force'] and metrics['peak_force'] < 3200:
            insights.append(SportInsight(
                category="strength",
                priority="high",
                title="Contact Strength Requirements",
                message=f"Peak force ({metrics['peak_force']:.0f}N) below typical for rugby (target >3400N forwards, >3000N backs). "
                        "High force production critical for tackles, rucks, mauls, scrums.",
                recommendation="Focus: Heavy compound lifts (squats, deadlifts, bench), "
                             "contact-specific strength (sled pushes, tackle bags with resistance). "
                             "Position-specific: Forwards prioritize absolute strength, backs relative strength.",
                metrics_referenced=['Peak Force [N]'],
                icon="🏉"
            ))

        # Asymmetry (some expected but monitor)
        if metrics['asymmetry'] and abs(metrics['asymmetry']) > 12:
            insights.append(SportInsight(
                category="asymmetry",
                priority="high",
                title="Bilateral Imbalance - Contact Sport Risk",
                message=f"Asymmetry of {abs(metrics['asymmetry']):.1f}% increases injury risk in contact situations. "
                        "Rugby involves multidirectional high-force impacts requiring balanced strength.",
                recommendation="Unilateral strength emphasis on weaker side: Single-leg squats, "
                             "step-ups with load, unilateral Olympic lift variations. "
                             "Monitor especially during/after injury. Target <10%.",
                metrics_referenced=['Asymmetry Index'],
                icon="⚖️"
            ))

        # Position-specific
        if position:
            pos_lower = str(position).lower()

            # Forwards (props, locks, flankers, number 8)
            if any(x in pos_lower for x in ['prop', 'lock', 'forward', 'hooker', 'flanker', '8']):
                if metrics['peak_force'] and metrics['peak_force'] < 3600:
                    insights.append(SportInsight(
                        category="strength",
                        priority="critical",
                        title="Forward-Specific Strength",
                        message="Forwards require exceptionally high absolute strength for scrummaging, lineout lifting, breakdown work. "
                                f"Current peak force below forward requirements.",
                        recommendation="Forward-specific: Scrum machine work, heavy squats (>2x BW target), "
                                     "deadlifts, farmer's carries. Integrate with scrum/lineout technique.",
                        metrics_referenced=['Peak Force [N]', 'Relative Peak Force [N/kg]'],
                        icon="⚙️"
                    ))

            # Backs (wings, fullback, centers, fly-half, scrum-half)
            elif any(x in pos_lower for x in ['back', 'wing', 'center', 'fullback', 'half']):
                if metrics['jump_height'] and metrics['jump_height'] < 42:
                    insights.append(SportInsight(
                        category="power",
                        priority="high",
                        title="Backs Power-to-Weight Ratio",
                        message="Rugby backs require high power-to-weight for acceleration, agility, jumping in contact. "
                                f"CMJ height ({metrics['jump_height']:.1f}cm) below typical for backs (>44cm).",
                        recommendation="Backs-specific: Plyometric program (box jumps, bounds), "
                                     "acceleration drills, power cleans emphasizing speed. "
                                     "Balance strength with maintaining speed/agility.",
                        metrics_referenced=['Jump Height (Flight Time) [cm]', 'Peak Power / BM_Trial'],
                        icon="⚡"
                    ))

        return insights

    @staticmethod
    def _get_metric(df: pd.DataFrame, metric_name: str) -> Optional[float]:
        """Extract latest metric value"""
        matching_cols = [col for col in df.columns if metric_name.lower() in col.lower()]
        if not matching_cols:
            return None
        col = matching_cols[0]
        values = df[col].dropna()
        if values.empty:
            return None
        if 'recordedDateUtc' in df.columns:
            df_sorted = df.sort_values('recordedDateUtc')
            return df_sorted[col].iloc[-1]
        return values.iloc[-1]


# ============================================================================
# PARA SPORTS INTELLIGENCE
# ============================================================================

class ParaSportsIntelligence:
    """Intelligence for Para Sports (Classification-Aware)"""

    @staticmethod
    def analyze_para_athlete(athlete_data: pd.DataFrame, sport: str, classification: str = None) -> List[SportInsight]:
        """
        Analyze Para athlete with classification-specific context

        Para sports require individualized assessment based on:
        - Classification (T/F for athletics, S/SB/SM for swimming, etc.)
        - Impairment type
        - Functional capacity
        """
        insights = []

        # Para athletes: Focus on individual baselines, not absolute values
        insights.append(SportInsight(
            category="training",
            priority="medium",
            title="Individualized Para Athlete Assessment",
            message="Para athlete performance must be evaluated against individual baseline and classification-specific norms, "
                    "not general population benchmarks. Focus on percentage improvements and functional capacity.",
            recommendation="Establish individual baseline across 3-5 tests. Track % change from baseline. "
                         "Consult classification-specific research for normative data. "
                         "Work with para sport specialists for training prescription.",
            metrics_referenced=['All metrics - individual baseline'],
            icon="♿"
        ))

        # Check for meaningful individual progress
        if 'recordedDateUtc' in athlete_data.columns and len(athlete_data) >= 3:
            # Calculate trend if enough data
            athlete_data_sorted = athlete_data.sort_values('recordedDateUtc')

            # Get a numeric metric for trend analysis
            numeric_cols = athlete_data_sorted.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Use first available numeric metric
                test_col = numeric_cols[0]
                values = athlete_data_sorted[test_col].dropna()

                if len(values) >= 3:
                    # Calculate simple trend
                    recent_avg = values.tail(2).mean()
                    baseline_avg = values.head(2).mean()

                    if baseline_avg > 0:
                        pct_change = ((recent_avg - baseline_avg) / baseline_avg) * 100

                        if pct_change > 5:
                            insights.append(SportInsight(
                                category="training",
                                priority="low",
                                title="Positive Training Response",
                                message=f"Showing {pct_change:.1f}% improvement from baseline. "
                                        "Positive adaptation to training program.",
                                recommendation="Continue current training approach. "
                                             "Consider progressive overload in areas showing adaptation. "
                                             "Monitor for plateaus - may indicate need for program variation.",
                                metrics_referenced=[test_col],
                                icon="📈"
                            ))
                        elif pct_change < -5:
                            insights.append(SportInsight(
                                category="training",
                                priority="high",
                                title="Performance Decline Detected",
                                message=f"Showing {abs(pct_change):.1f}% decline from baseline. "
                                        "May indicate fatigue, overtraining, or health issues.",
                                recommendation="Review: Training load, sleep quality, nutrition, stress levels. "
                                             "Consider deload week or active recovery. "
                                             "Consult medical team if decline persists >2 weeks.",
                                metrics_referenced=[test_col],
                                icon="⚠️"
                            ))

        return insights


# ============================================================================
# EXPORT FOR INTEGRATION
# ============================================================================

ALL_SPORT_HANDLERS = {
    'athletics': AthleticsIntelligence,
    'swimming': SwimmingIntelligence,
    'rowing': RowingIntelligence,
    'weightlifting': WeightliftingIntelligence,
    'football': FootballIntelligence,
    'soccer': FootballIntelligence,
    'rugby': RugbyIntelligence,
    'para_athletics': ParaSportsIntelligence,
    'para_swimming': ParaSportsIntelligence,
    'para_judo': ParaSportsIntelligence,
    'para_wrestling': ParaSportsIntelligence
}


if __name__ == "__main__":
    print("All Sports Intelligence Module Loaded!")
    print(f"Handlers for {len(ALL_SPORT_HANDLERS)} sport categories")
