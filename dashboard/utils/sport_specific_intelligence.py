"""
Sport-Specific Intelligence Module
Team Saudi - VALD Performance Analysis

Provides dynamic, sport-specific insights and recommendations based on:
- Sport biomechanics and movement patterns
- Test data analysis
- Evidence-based training principles
- Team Saudi athlete context

Built on research and best practices for:
- Throws (Shot Put, Discus, Javelin, Hammer)
- Fencing (Epee, Sabre, Foil)
- Wrestling & Combat Sports
- Athletics (Sprints, Jumps, Distance)
- Rowing, Swimming, Weightlifting, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.sports_config import get_sport_benchmarks, SPORT_BENCHMARKS, get_percentile_rank
except ImportError:
    print("Warning: sports_config.py not found. Using fallback benchmarks.")
    SPORT_BENCHMARKS = {}


@dataclass
class SportInsight:
    """Container for a sport-specific insight"""
    category: str  # "strength", "power", "asymmetry", "technique", "training"
    priority: str  # "critical", "high", "medium", "low"
    title: str
    message: str
    recommendation: str
    metrics_referenced: List[str]
    icon: str = "💡"


# ============================================================================
# THROWS-SPECIFIC INTELLIGENCE
# ============================================================================

class ThrowsIntelligence:
    """
    Specialized intelligence for throwing events
    Based on biomechanical demands of shot put, discus, javelin, hammer
    """

    SPORT_MAPPING = {
        'Shot Put': 'shot_put',
        'Discus': 'discus',
        'Discus Throw': 'discus',
        'Javelin': 'javelin',
        'Javelin Throw': 'javelin',
        'Hammer': 'hammer',
        'Hammer Throw': 'hammer',
        'Throws': 'general_throws'
    }

    @staticmethod
    def get_throw_type(sport: str) -> str:
        """Identify specific throw discipline"""
        sport_str = str(sport).strip()
        return ThrowsIntelligence.SPORT_MAPPING.get(sport_str, 'general_throws')

    @staticmethod
    def analyze_throws_athlete(athlete_data: pd.DataFrame, sport: str) -> List[SportInsight]:
        """Generate throws-specific insights for an athlete"""
        insights = []
        throw_type = ThrowsIntelligence.get_throw_type(sport)

        # Get key metrics
        metrics = {
            'peak_force': ThrowsIntelligence._get_metric(athlete_data, 'Peak Force'),
            'peak_power': ThrowsIntelligence._get_metric(athlete_data, 'Peak Power'),
            'jump_height': ThrowsIntelligence._get_metric(athlete_data, 'Jump Height'),
            'rfd': ThrowsIntelligence._get_metric(athlete_data, 'RFD'),
            'asymmetry': ThrowsIntelligence._get_metric(athlete_data, 'Asymmetry')
        }

        # 1. ROTATIONAL POWER (Critical for Discus & Hammer)
        if throw_type in ['discus', 'hammer']:
            if metrics['peak_power'] is not None:
                if metrics['peak_power'] < 4500:  # W
                    insights.append(SportInsight(
                        category="power",
                        priority="critical",
                        title="Rotational Power Development Needed",
                        message=f"Peak power ({metrics['peak_power']:.0f}W) below optimal for {throw_type}. "
                                f"Rotational throws require >5000W for competitive throwing.",
                        recommendation="Focus: Medicine ball rotational throws, Olympic lift variations (especially power cleans), "
                                     "plyometric rotational jumps. Target: 10-15% increase in peak power.",
                        metrics_referenced=['Peak Power [W]'],
                        icon="🔄"
                    ))

        # 2. EXPLOSIVE STRENGTH (Critical for Shot Put)
        if throw_type == 'shot_put':
            if metrics['rfd'] is not None:
                # RFD critical for shot put
                insights.append(SportInsight(
                    category="power",
                    priority="high",
                    title="Rate of Force Development",
                    message=f"Shot put demands maximal explosive strength in <0.3s. "
                            f"Current RFD metrics indicate {'good' if metrics['rfd'] > 3000 else 'needs improvement'} explosive capacity.",
                    recommendation="Implement: Heavy squat jumps (70-80% BW), depth jumps from 30-40cm, "
                                 "plyometric push-ups, medicine ball chest passes (emphasis on acceleration).",
                    metrics_referenced=['Rate of Force Development', 'Peak Force [N]'],
                    icon="💥"
                ))

        # 3. UNILATERAL STRENGTH (Important for Javelin)
        if throw_type == 'javelin':
            if metrics['asymmetry'] is not None:
                if abs(metrics['asymmetry']) > 15:
                    insights.append(SportInsight(
                        category="asymmetry",
                        priority="high",
                        title="Significant Lower Body Asymmetry",
                        message=f"Asymmetry of {abs(metrics['asymmetry']):.1f}% detected. "
                                f"While javelin is inherently asymmetrical (throwing arm dominant), "
                                f"lower body asymmetry >15% can compromise run-up efficiency.",
                        recommendation="Address with: Single-leg RDLs, Bulgarian split squats, "
                                     "single-leg box jumps. Monitor weekly - maintain <12% asymmetry.",
                        metrics_referenced=['Asymmetry Index'],
                        icon="⚠️"
                    ))

        # 4. GENERAL THROWS - Power-to-Weight Ratio
        if metrics['peak_power'] and metrics.get('body_weight'):
            power_to_weight = metrics['peak_power'] / metrics['body_weight']
            if power_to_weight < 60:  # W/kg
                insights.append(SportInsight(
                    category="power",
                    priority="medium",
                    title="Power-to-Weight Ratio",
                    message=f"Current power-to-weight ratio: {power_to_weight:.1f} W/kg. "
                            f"Elite throwers typically achieve >65 W/kg.",
                    recommendation="Dual focus: (1) Increase absolute power via heavy compound lifts, "
                                 "(2) Optimize body composition if carrying excess mass. "
                                 "Monitor monthly progression.",
                    metrics_referenced=['Peak Power / BM_Trial'],
                    icon="⚖️"
                ))

        # 5. BILATERAL FORCE PRODUCTION (All throws)
        if metrics['peak_force'] is not None:
            force_percentile = get_percentile_rank(metrics['peak_force'], sport, 'Peak Force')
            if force_percentile and force_percentile < 50:
                insights.append(SportInsight(
                    category="strength",
                    priority="high",
                    title="Foundational Strength Development",
                    message=f"Peak force below 50th percentile for {throw_type}. "
                            f"Absolute strength is the foundation for all throwing events.",
                    recommendation="Prioritize: Back squats (3-5 reps @ 85-90% 1RM), deadlifts, "
                                 "trap bar jumps. Build base strength before progressing to speed-strength work. "
                                 "Test every 4-6 weeks.",
                    metrics_referenced=['Peak Force [N]', 'Relative Peak Force [N/kg]'],
                    icon="🏋️"
                ))

        return insights

    @staticmethod
    def _get_metric(df: pd.DataFrame, metric_name: str) -> Optional[float]:
        """Extract metric value from dataframe (latest or mean)"""
        # Find column containing metric name
        matching_cols = [col for col in df.columns if metric_name.lower() in col.lower()]
        if not matching_cols:
            return None

        col = matching_cols[0]
        values = df[col].dropna()
        if values.empty:
            return None

        # Return latest value
        if 'recordedDateUtc' in df.columns:
            df_sorted = df.sort_values('recordedDateUtc')
            return df_sorted[col].iloc[-1]
        return values.iloc[-1]


# ============================================================================
# FENCING-SPECIFIC INTELLIGENCE
# ============================================================================

class FencingIntelligence:
    """
    Specialized intelligence for fencing (Epee, Sabre, Foil)
    Key: Unilateral lunge power, reactive strength, asymmetry is functional
    """

    @staticmethod
    def analyze_fencing_athlete(athlete_data: pd.DataFrame, sport: str) -> List[SportInsight]:
        """Generate fencing-specific insights"""
        insights = []

        # Get key metrics
        metrics = {
            'jump_height': FencingIntelligence._get_metric(athlete_data, 'Jump Height'),
            'rsi': FencingIntelligence._get_metric(athlete_data, 'RSI'),
            'peak_power': FencingIntelligence._get_metric(athlete_data, 'Peak Power'),
            'asymmetry': FencingIntelligence._get_metric(athlete_data, 'Asymmetry'),
            'rfd': FencingIntelligence._get_metric(athlete_data, 'RFD')
        }

        # 1. FUNCTIONAL ASYMMETRY CONTEXT
        if metrics['asymmetry'] is not None:
            abs_asym = abs(metrics['asymmetry'])

            if abs_asym < 10:
                # Unexpectedly low asymmetry for fencer
                insights.append(SportInsight(
                    category="technique",
                    priority="medium",
                    title="Lower Than Expected Asymmetry",
                    message=f"Asymmetry of {abs_asym:.1f}% is lower than typical for fencers (12-18%). "
                            f"This may indicate underdeveloped lead leg dominance or bilateral training emphasis.",
                    recommendation="Consider: More sport-specific lunge training to develop lead leg dominance. "
                                 "Some functional asymmetry (12-15%) is expected and beneficial for fencing performance. "
                                 "Assess lunge technique on-piste.",
                    metrics_referenced=['Asymmetry Index'],
                    icon="🤺"
                ))
            elif abs_asym > 20:
                # Excessive asymmetry even for fencing
                insights.append(SportInsight(
                    category="asymmetry",
                    priority="critical",
                    title="Excessive Asymmetry - Injury Risk",
                    message=f"Asymmetry of {abs_asym:.1f}% exceeds functional range even for fencing. "
                            f"While 12-18% is expected, >20% increases injury risk (knee, hip, lower back).",
                    recommendation="Implement corrective work: Single-leg squats on non-dominant leg, "
                                 "rear leg emphasis in lunges, unilateral RDLs. "
                                 "Add bilateral strength work to balance development. Monitor bi-weekly.",
                    metrics_referenced=['Asymmetry Index'],
                    icon="⚠️"
                ))

        # 2. REACTIVE STRENGTH FOR EXPLOSIVE LUNGES
        if metrics['rsi'] is not None:
            if metrics['rsi'] < 1.8:
                insights.append(SportInsight(
                    category="power",
                    priority="high",
                    title="Reactive Strength Development",
                    message=f"RSI of {metrics['rsi']:.2f} below optimal for explosive lunge attacks (target >1.9). "
                            f"Fencing requires rapid force application for effective lunges and recoveries.",
                    recommendation="Focus: Depth jumps (20-30cm height), single-leg reactive hops, "
                                 "lunge jumps with minimal ground contact, ankle stiffness drills. "
                                 "2-3x per week, monitor RSI progression.",
                    metrics_referenced=['RSI-modified (Imp-Mom)_Trial', 'Contact Time [ms]'],
                    icon="⚡"
                ))

        # 3. SABRE-SPECIFIC: Higher Power Demands
        if 'sabre' in sport.lower():
            if metrics['peak_power'] is not None and metrics['peak_power'] < 4000:
                insights.append(SportInsight(
                    category="power",
                    priority="high",
                    title="Sabre Power Requirements",
                    message="Sabre fencing is more dynamic than epee, requiring higher power outputs for "
                            "explosive attacks and rapid direction changes. Current power metrics suggest "
                            "room for improvement.",
                    recommendation="Implement: Box jumps with immediate second jump, broad jumps, "
                                 "weighted lunge jumps, medicine ball slam variations. "
                                 "Combine with speed-strength work (30-50% loads, maximal velocity).",
                    metrics_referenced=['Peak Power [W]', 'Peak Power / BM_Trial'],
                    icon="🗡️"
                ))

        # 4. RATE OF FORCE DEVELOPMENT
        if metrics['rfd'] is not None:
            # High RFD critical for quick attacks
            insights.append(SportInsight(
                category="power",
                priority="medium",
                title="Attack Speed Development",
                message="Fencing attacks occur in <300ms. Rate of force development directly correlates "
                        "with lunge speed and attack success rate.",
                recommendation="Drills: Isometric holds with explosive concentric (pause squats), "
                             "ballistic push-offs from lunge position, overspeed eccentric lunges. "
                             "Combine with on-piste attack speed training.",
                metrics_referenced=['Rate of Force Development'],
                icon="⏱️"
            ))

        return insights

    @staticmethod
    def _get_metric(df: pd.DataFrame, metric_name: str) -> Optional[float]:
        """Extract metric value"""
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
# WRESTLING & COMBAT SPORTS INTELLIGENCE
# ============================================================================

class CombatSportsIntelligence:
    """Intelligence for Wrestling, Judo, Jiu-Jitsu"""

    @staticmethod
    def analyze_combat_athlete(athlete_data: pd.DataFrame, sport: str) -> List[SportInsight]:
        """Generate combat sports-specific insights"""
        insights = []

        metrics = {
            'peak_force': CombatSportsIntelligence._get_metric(athlete_data, 'Peak Force'),
            'relative_force': CombatSportsIntelligence._get_metric(athlete_data, 'Relative Peak Force'),
            'jump_height': CombatSportsIntelligence._get_metric(athlete_data, 'Jump Height'),
            'asymmetry': CombatSportsIntelligence._get_metric(athlete_data, 'Asymmetry'),
            'rfd': CombatSportsIntelligence._get_metric(athlete_data, 'RFD')
        }

        # 1. WEIGHT CLASS OPTIMIZATION
        if metrics['relative_force'] is not None:
            if metrics['relative_force'] < 28:
                insights.append(SportInsight(
                    category="strength",
                    priority="critical",
                    title="Relative Strength for Weight Class",
                    message=f"Relative peak force ({metrics['relative_force']:.1f} N/kg) below optimal for weight class sports. "
                            f"Target >30 N/kg for competitive advantage in wrestling/combat.",
                    recommendation="Prioritize: Relative strength development via bodyweight + moderate load exercises. "
                                 "Reduce absolute mass if body composition allows. "
                                 "Focus: Pull-ups (weighted), pistol squats, explosive push-ups. "
                                 "Monitor body composition monthly.",
                    metrics_referenced=['Relative Peak Force [N/kg]'],
                    icon="⚖️"
                ))

        # 2. BILATERAL STRENGTH BALANCE
        if metrics['asymmetry'] is not None and abs(metrics['asymmetry']) > 12:
            insights.append(SportInsight(
                category="asymmetry",
                priority="high",
                title="Bilateral Strength Imbalance",
                message=f"Asymmetry of {abs(metrics['asymmetry']):.1f}% can affect wrestling/grappling technique. "
                        f"Balanced bilateral strength critical for takedown success and defensive stability.",
                recommendation="Corrective protocol: Unilateral strength work emphasizing weaker side. "
                             "Single-leg RDLs, split squats, single-arm work. "
                             "Reassess after 3-4 weeks of corrective work. Target <10% asymmetry.",
                metrics_referenced=['Asymmetry Index'],
                icon="⚖️"
            ))

        # 3. EXPLOSIVE TAKEDOWN POWER (Wrestling/Judo specific)
        if 'wrestling' in sport.lower() or 'judo' in sport.lower():
            if metrics['jump_height'] is not None and metrics['jump_height'] < 40:
                insights.append(SportInsight(
                    category="power",
                    priority="high",
                    title="Takedown Power Development",
                    message=f"Jump height ({metrics['jump_height']:.1f}cm) suggests limited explosive hip extension power. "
                            f"Critical for effective takedowns and throw execution.",
                    recommendation="Training focus: Box jumps, broad jumps, power cleans, " "med ball overhead throws. "
                                 "Integrate with takedown drills - explosive level changes. "
                                 "Target: 42-45cm CMJ within 8-12 weeks.",
                    metrics_referenced=['Jump Height (Flight Time) [cm]', 'Peak Power [W]'],
                    icon="🤼"
                ))

        # 4. ISOMETRIC STRENGTH (Critical for holds/control)
        if 'jiu-jitsu' in sport.lower() or 'judo' in sport.lower():
            insights.append(SportInsight(
                category="strength",
                priority="medium",
                title="Isometric Strength for Control",
                message="Grappling requires sustained isometric contractions for maintaining positions and submissions. "
                        "Consider supplementing with isometric strength assessments.",
                recommendation="Add isometric protocols: Isometric mid-thigh pull, plank variations with load, "
                             "wall sits with holds >30s. Develop time-under-tension capacity. "
                             "Correlates with grip endurance and positional control.",
                metrics_referenced=['Peak Force [N]'],
                icon="🥋"
            ))

        return insights

    @staticmethod
    def _get_metric(df: pd.DataFrame, metric_name: str) -> Optional[float]:
        """Extract metric value"""
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
# MASTER SPORT INTELLIGENCE ENGINE
# ============================================================================

class SportIntelligenceEngine:
    """
    Master engine that routes to sport-specific intelligence modules
    """

    SPORT_HANDLERS = {
        'throws': ['Shot Put', 'Discus', 'Javelin', 'Hammer', 'Throws', 'Discus Throw', 'Javelin Throw', 'Hammer Throw'],
        'fencing': ['Epee', 'Sabre', 'Foil', 'Fencing'],
        'wrestling': ['Wrestling', 'Greco-Roman Wrestling', 'Freestyle Wrestling', 'Para Wrestling'],
        'judo': ['Judo', 'Para Judo'],
        'jiu-jitsu': ['Jiu-Jitsu', 'Brazilian Jiu-Jitsu'],
        'athletics': ['Athletics', 'Track and Field'],
        'para_athletics': ['Para Athletics', 'Para Track and Field'],
        'swimming': ['Swimming'],
        'para_swimming': ['Para Swimming'],
        'rowing': ['Rowing', 'Coastal'],
        'weightlifting': ['Weightlifting', 'Olympic Weightlifting'],
        'football': ['Football', 'FootballSoccer', 'Soccer'],
        'rugby': ['Rugby', 'RugbyUnion', 'Rugby Union']
    }

    @classmethod
    def analyze_athlete(cls, athlete_data: pd.DataFrame, sport: str) -> List[SportInsight]:
        """
        Generate sport-specific insights for an athlete

        Parameters:
        -----------
        athlete_data : pd.DataFrame
            Athlete's test data
        sport : str
            Sport/discipline name

        Returns:
        --------
        List[SportInsight]
            Prioritized list of insights and recommendations
        """
        sport_str = str(sport).strip()
        insights = []

        # Import additional handlers
        try:
            from all_sports_intelligence import (
                AthleticsIntelligence, SwimmingIntelligence, RowingIntelligence,
                WeightliftingIntelligence, FootballIntelligence, RugbyIntelligence,
                ParaSportsIntelligence
            )
            extended_handlers_available = True
        except ImportError:
            extended_handlers_available = False

        # Route to appropriate handler
        if cls._is_sport_type(sport_str, 'throws'):
            insights = ThrowsIntelligence.analyze_throws_athlete(athlete_data, sport_str)

        elif cls._is_sport_type(sport_str, 'fencing'):
            insights = FencingIntelligence.analyze_fencing_athlete(athlete_data, sport_str)

        elif cls._is_sport_type(sport_str, 'wrestling') or cls._is_sport_type(sport_str, 'judo') or cls._is_sport_type(sport_str, 'jiu-jitsu'):
            insights = CombatSportsIntelligence.analyze_combat_athlete(athlete_data, sport_str)

        # Extended handlers (if available)
        elif extended_handlers_available:
            if cls._is_sport_type(sport_str, 'para_athletics') or cls._is_sport_type(sport_str, 'para_swimming') or \
               cls._is_sport_type(sport_str, 'para_judo') or cls._is_sport_type(sport_str, 'para_wrestling'):
                insights = ParaSportsIntelligence.analyze_para_athlete(athlete_data, sport_str)

            elif cls._is_sport_type(sport_str, 'athletics'):
                insights = AthleticsIntelligence.analyze_athletics_athlete(athlete_data, sport_str)

            elif cls._is_sport_type(sport_str, 'swimming'):
                insights = SwimmingIntelligence.analyze_swimming_athlete(athlete_data, sport_str)

            elif cls._is_sport_type(sport_str, 'rowing'):
                insights = RowingIntelligence.analyze_rowing_athlete(athlete_data, sport_str)

            elif cls._is_sport_type(sport_str, 'weightlifting'):
                insights = WeightliftingIntelligence.analyze_weightlifting_athlete(athlete_data)

            elif cls._is_sport_type(sport_str, 'football'):
                insights = FootballIntelligence.analyze_football_athlete(athlete_data)

            elif cls._is_sport_type(sport_str, 'rugby'):
                insights = RugbyIntelligence.analyze_rugby_athlete(athlete_data)

            else:
                # Generic insights for other sports
                insights = cls._generate_generic_insights(athlete_data, sport_str)
        else:
            # Generic insights if extended handlers not available
            insights = cls._generate_generic_insights(athlete_data, sport_str)

        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        insights.sort(key=lambda x: priority_order.get(x.priority, 99))

        return insights

    @classmethod
    def _is_sport_type(cls, sport: str, sport_type: str) -> bool:
        """Check if sport belongs to a category"""
        sport_lower = sport.lower()
        handlers = cls.SPORT_HANDLERS.get(sport_type, [])
        return any(handler.lower() in sport_lower for handler in handlers)

    @classmethod
    def _generate_generic_insights(cls, athlete_data: pd.DataFrame, sport: str) -> List[SportInsight]:
        """Generate general insights for sports without specific handlers"""
        insights = []

        # Use sport benchmarks from config
        benchmarks = get_sport_benchmarks(sport)

        # Check jump height
        jump_cols = [col for col in athlete_data.columns if 'jump height' in col.lower()]
        if jump_cols and benchmarks.cmj_height_excellent:
            latest_jump = athlete_data[jump_cols[0]].dropna()
            if not latest_jump.empty:
                latest_value = latest_jump.iloc[-1]
                percentile = get_percentile_rank(latest_value, sport, 'Jump Height')

                if percentile and percentile < 50:
                    insights.append(SportInsight(
                        category="power",
                        priority="medium",
                        title="Lower Body Power Development",
                        message=f"Jump height ({latest_value:.1f}cm) below 50th percentile for {sport}. "
                                f"Target: {benchmarks.cmj_height_good:.1f}cm (Good), {benchmarks.cmj_height_excellent:.1f}cm (Excellent).",
                        recommendation="Implement progressive plyometric program: Box jumps, depth jumps, "
                                     "squat jumps. Combine with strength training (squats, deadlifts). "
                                     "Retest every 4 weeks.",
                        metrics_referenced=['Jump Height (Flight Time) [cm]'],
                        icon="📈"
                    ))

        return insights


# ============================================================================
# HELPER FUNCTIONS FOR DASHBOARD INTEGRATION
# ============================================================================

def display_insights_streamlit(insights: List[SportInsight], sport: str):
    """
    Display insights in Streamlit with nice formatting

    Usage in dashboard:
    ```
    insights = SportIntelligenceEngine.analyze_athlete(athlete_df, sport)
    display_insights_streamlit(insights, sport)
    ```
    """
    import streamlit as st

    if not insights:
        st.info(f"✅ No critical insights for {sport}. Performance metrics within expected ranges.")
        return

    st.markdown(f"### 🎯 Sport-Specific Insights for {sport}")
    st.caption("*AI-generated insights based on biomechanical demands and performance data*")

    for insight in insights:
        # Color code by priority
        if insight.priority == 'critical':
            st.error(f"{insight.icon} **{insight.title}**")
        elif insight.priority == 'high':
            st.warning(f"{insight.icon} **{insight.title}**")
        else:
            st.info(f"{insight.icon} **{insight.title}**")

        with st.expander("View Details"):
            st.markdown(f"**Analysis:**\n{insight.message}")
            st.markdown(f"**Recommendation:**\n{insight.recommendation}")
            st.caption(f"*Metrics: {', '.join(insight.metrics_referenced)}*")


def get_quick_insights_summary(athlete_data: pd.DataFrame, sport: str) -> Dict[str, int]:
    """
    Get quick summary of insights by priority

    Returns:
    --------
    {'critical': 2, 'high': 3, 'medium': 1, 'low': 0}
    """
    insights = SportIntelligenceEngine.analyze_athlete(athlete_data, sport)

    summary = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    for insight in insights:
        summary[insight.priority] += 1

    return summary


# ============================================================================
# TESTING & EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("Sport-Specific Intelligence Module")
    print("=" * 80)

    # Example: Generate insights for a hypothetical throws athlete
    print("\nExample: Shot Put Athlete with Low Power")

    # Simulated data
    sample_data = pd.DataFrame({
        'recordedDateUtc': ['2024-01-01', '2024-01-15'],
        'Peak Force [N]': [2800, 2850],
        'Peak Power [W]': [4200, 4300],
        'Jump Height (Flight Time) [cm]': [38, 39],
        'Asymmetry Index': [8, 7],
        'athlete_sport': ['Shot Put', 'Shot Put']
    })

    insights = SportIntelligenceEngine.analyze_athlete(sample_data, 'Shot Put')

    for i, insight in enumerate(insights, 1):
        print(f"\n{i}. [{insight.priority.upper()}] {insight.title}")
        print(f"   {insight.message}")
        print(f"   → {insight.recommendation}")

    print("\n" + "=" * 80)
    print("Module loaded successfully!")
    print("Import in dashboard: from utils.sport_specific_intelligence import SportIntelligenceEngine")
