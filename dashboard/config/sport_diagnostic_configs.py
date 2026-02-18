"""
Sport-Specific Diagnostic Configurations
Saudi National Team - VALD Performance Dashboard

Defines which tests, metrics, and thresholds to display in each sport's
diagnostic tab. Each sport config drives the 3-column diagnostic layout:
  Column 1: Asymmetry profile (ForceDecks bilateral tests)
  Column 2: ForceFrame / isometric profiles
  Column 3: Key metrics summary card

Below the 3-column layout, extra_sections render additional test views
(hop tests, balance, strength RM, plyo pushup, etc.).

Each sport config can also include:
  - benchmarks: Sport-specific performance benchmarks (elite/national, M/F)
  - performance_ratios: Calculated ratios (EUR, IR:ER, Add:Abd, etc.)
  - injury_risk_indicators: Auto-flagging thresholds from peer-reviewed research
  - context_notes: Research-backed sport science notes with citations

IMPORTANT: sub_sports values must match the athlete_sport column values
produced by config.vald_categories.GROUP_TO_CATEGORY mapping. These are
the sub-category strings (e.g., 'Fencing - Epee', 'Athletics - Jumps').

Research sources:
  - Turner et al. 2014 (fencing force-power profiling)
  - Bottoms et al. 2011 (fencing weapon-specific jump performance)
  - Opar et al. 2015 BJSM (NordBord hamstring injury thresholds)
  - Byram et al. 2010 (shoulder IR:ER injury risk)
  - Suchomel et al. 2016 (IMTP importance for performance)
  - Loturco et al. 2015-2022 (combat sport power profiling)
  - West et al. 2011 (swimming force-performance relationships)
  - Cormie et al. 2011 (force-velocity-power review)
  - Comfort et al. 2014 (IMTP reliability and normative data)
"""

from typing import Dict, List, Any, Optional

import pandas as pd


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def has_sport_data(df: Optional[pd.DataFrame], sub_sports: List[str]) -> bool:
    """Check if any athletes from these sub-sports have data in the DataFrame."""
    if df is None or df.empty or 'athlete_sport' not in df.columns:
        return False
    return df['athlete_sport'].isin(sub_sports).any()


def get_config_for_athlete_sport(athlete_sport: str) -> Optional[Dict[str, Any]]:
    """Look up the diagnostic config for a given athlete_sport value.

    Tries exact sub_sports match first, then falls back to parent sport
    matching (e.g., 'Fencing - Epee' matches the 'fencing' config).

    Returns None if no config found.
    """
    for key, config in SPORT_DIAGNOSTIC_CONFIGS.items():
        if athlete_sport in config['sub_sports']:
            return config

    # Fallback: check if the athlete_sport starts with a known parent
    athlete_lower = athlete_sport.lower() if athlete_sport else ''
    for key, config in SPORT_DIAGNOSTIC_CONFIGS.items():
        if athlete_lower.startswith(key):
            return config

    return None


def get_all_configured_sports() -> List[str]:
    """Return a sorted list of all sport display names that have configs."""
    return sorted([cfg['display_name'] for cfg in SPORT_DIAGNOSTIC_CONFIGS.values()])


# ============================================================================
# SPORT DIAGNOSTIC CONFIGS
# ============================================================================

SPORT_DIAGNOSTIC_CONFIGS: Dict[str, Dict[str, Any]] = {

    # ========================================================================
    # FENCING (142 athletes - richest data)
    # Test types available: CMJ (219), HJ (191), IMTP (178), SLJ, SLHJ,
    #   SLISOT, SJ, SLCMRJ, SLSTICR, ISOT
    # ========================================================================
    'fencing': {
        'display_name': 'Fencing',
        'icon': '\u2694\uFE0F',
        'sub_sports': [
            'Fencing - Epee', 'Fencing - Foil', 'Fencing - Sabre', 'Fencing - SOTC',
            'Fencing', 'Epee', 'Sabre',
        ],

        # ---- Asymmetry profile (Column 1) ----
        'asymmetry_metrics': [
            {
                'name': 'IMTP Peak Force L/R',
                'test_types': ['IMTP'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Peak Concentric Force L/R',
                'test_types': ['CMJ', 'ABCMJ'],
                'left_col': 'PEAK_CONCENTRIC_FORCE_Left',
                'right_col': 'PEAK_CONCENTRIC_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Concentric Impulse L/R',
                'test_types': ['CMJ', 'ABCMJ'],
                'left_col': 'CONCENTRIC_IMPULSE_Left',
                'right_col': 'CONCENTRIC_IMPULSE_Right',
                'unit': 'Ns',
            },
            {
                'name': 'SL CMJ Peak Force',
                'test_types': ['SLCMJ'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'Concentric RFD L/R',
                'test_types': ['CMJ', 'ABCMJ'],
                'left_col': 'CONCENTRIC_RFD_Left',
                'right_col': 'CONCENTRIC_RFD_Right',
                'unit': 'N/s',
            },
            {
                'name': 'Eccentric Braking RFD L/R',
                'test_types': ['CMJ', 'ABCMJ'],
                'left_col': 'ECCENTRIC_BRAKING_RFD_Left',
                'right_col': 'ECCENTRIC_BRAKING_RFD_Right',
                'unit': 'N/s',
            },
            {
                'name': 'Force at Zero Velocity L/R',
                'test_types': ['CMJ', 'ABCMJ'],
                'left_col': 'FORCE_AT_ZERO_VELOCITY_Left',
                'right_col': 'FORCE_AT_ZERO_VELOCITY_Right',
                'unit': 'N',
            },
            {
                'name': 'Landing RFD L/R',
                'test_types': ['CMJ', 'ABCMJ'],
                'left_col': 'LANDING_RFD_Left',
                'right_col': 'LANDING_RFD_Right',
                'unit': 'N/s',
            },
        ],
        'asymmetry_thresholds': {'caution': 15.0, 'risk': 20.0},
        'asymmetry_note': (
            'Fencing athletes typically show 15-20% lead-leg dominance - '
            'this is expected and functional. Flag only if >20%. '
            'Concentric RFD asymmetry is particularly relevant for lunge initiation speed.'
        ),

        # ---- ForceFrame sections (Column 2) ----
        'forceframe_sections': [
            {
                'name': 'Shoulder IR/ER',
                'test_pattern': r'Shoulder.*Internal.*External|Shoulder.*Rotation',
                'inner_label': 'Internal Rotation',
                'outer_label': 'External Rotation',
            },
            {
                'name': 'Hip Add/Abd',
                'test_pattern': r'Hip.*Adduction.*Abduction|Hip.*Add.*Abd',
                'inner_label': 'Adduction',
                'outer_label': 'Abduction',
            },
        ],

        # ---- DynaMo grip ----
        'dynamo_section': {
            'enabled': True,
            'movement_filter': 'GripSqueeze',
            'title': 'Grip Strength (DynaMo)',
        },

        # ---- NordBord ----
        'nordbord_section': {
            'enabled': True,
            'title': 'Nordic Hamstring Strength',
        },

        # ---- Key metrics (Column 3) ----
        'key_metrics': [
            {
                'label': 'CMJ Height',
                'test_types': ['CMJ', 'ABCMJ'],
                'col': 'JUMP_HEIGHT_IMP_MOM',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'IMTP Net Peak Force',
                'test_types': ['IMTP'],
                'col': 'NET_PEAK_VERTICAL_FORCE',
                'unit': 'N',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'CMJ Rel. Power',
                'test_types': ['CMJ', 'ABCMJ'],
                'col': 'BODYMASS_RELATIVE_TAKEOFF_POWER',
                'unit': 'W/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'RSI',
                'test_types': ['HJ', 'SLHJ', 'RSHIP', 'RSKIP', 'RSAIP'],
                'col': 'HOP_BEST_RSI',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
            {
                'label': 'Concentric RFD',
                'test_types': ['CMJ', 'ABCMJ'],
                'col': 'CONCENTRIC_RFD',
                'unit': 'N/s',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'Eccentric Decel RFD',
                'test_types': ['CMJ', 'ABCMJ'],
                'col': 'ECCENTRIC_DECEL_RFD',
                'unit': 'N/s',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'RSI Modified',
                'test_types': ['CMJ', 'ABCMJ'],
                'col': 'RSI_MODIFIED',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
            {
                'label': 'Countermovement Depth',
                'test_types': ['CMJ', 'ABCMJ'],
                'col': 'COUNTERMOVEMENT_DEPTH',
                'unit': 'm',
                'format': '.3f',
                'higher_is_better': False,
            },
        ],

        # ---- Extra sections below 3-column layout ----
        'extra_sections': [
            {
                'type': 'hop_test',
                'title': '10:5 Hop / Reactive Strength',
                'test_types': ['HJ', 'SLHJ', 'RSHIP', 'RSKIP', 'RSAIP'],
                'metric_key': 'rsi',
            },
            {
                'type': 'eur',
                'title': 'Eccentric Utilization Ratio (CMJ:SJ)',
                'test_types_num': ['CMJ'],
                'test_types_den': ['SJ'],
                'col_num': 'JUMP_HEIGHT_IMP_MOM',
                'col_den': 'JUMP_HEIGHT_IMP_MOM',
            },
        ],

        # ---- Benchmarks (research-backed) ----
        'benchmarks': {
            'cmj_height': {
                'elite_m': (40, 48), 'national_m': (34, 40),
                'elite_f': (30, 38), 'national_f': (25, 30),
                'unit': 'cm',
                'source': 'Turner et al., 2014; Tsolakis & Vagenas, 2010',
            },
            'imtp_relative': {
                'elite_m': (35, 42), 'national_m': (30, 36),
                'elite_f': (28, 35), 'national_f': (25, 30),
                'unit': 'N/kg',
                'source': 'Suchomel et al., 2016',
            },
            'rsi': {
                'elite_m': (2.0, None), 'national_m': (1.5, 2.0),
                'elite_f': (1.6, None), 'national_f': (1.2, 1.6),
                'unit': '',
                'source': 'Turner et al., 2014',
            },
            'concentric_rfd': {
                'elite_m': (8000, 14000), 'national_m': (5000, 8000),
                'elite_f': (5000, 9000), 'national_f': (3500, 5000),
                'unit': 'N/s',
                'source': 'Turner et al., 2014',
            },
        },

        # ---- Performance ratios ----
        'performance_ratios': [
            {
                'name': 'Eccentric Utilization Ratio',
                'description': 'CMJ:SJ height ratio. Measures stretch-shortening cycle efficiency.',
                'formula': 'CMJ Height / SJ Height',
                'optimal': (1.08, 1.20),
                'concern_low': 1.05,
                'concern_high': 1.25,
                'test_types_num': ['CMJ'],
                'test_types_den': ['SJ'],
                'col_num': 'JUMP_HEIGHT_IMP_MOM',
                'col_den': 'JUMP_HEIGHT_IMP_MOM',
            },
            {
                'name': 'Shoulder IR:ER Ratio (Weapon Arm)',
                'description': 'Internal:External rotation strength ratio from ForceFrame. Weapon arm critical.',
                'source': 'forceframe',
                'test_pattern': 'Shoulder',
                'optimal': (1.50, 2.00),
                'concern_high': 2.00,
                'risk_high': 2.50,
                'citation': 'Byram et al., 2010; Ellenbecker & Davies, 2000',
            },
            {
                'name': 'Hip Add:Abd Ratio',
                'description': 'Adduction:Abduction strength ratio for groin injury prevention.',
                'source': 'forceframe',
                'test_pattern': 'Hip',
                'optimal': (0.80, 1.20),
                'concern_low': 0.80,
                'concern_high': 1.30,
                'citation': 'Tyler et al., 2001',
            },
        ],

        # ---- Injury risk indicators ----
        'injury_risk_indicators': [
            {
                'name': 'Hamstring Injury Risk',
                'test': 'NordBord',
                'metric': 'absolute_force_per_leg',
                'threshold_male': 337,
                'threshold_female': 256,
                'direction': 'below',
                'severity': 'high',
                'source': 'Opar et al., 2015 BJSM',
            },
            {
                'name': 'Shoulder Impingement Risk (Weapon Arm)',
                'test': 'ForceFrame',
                'pattern': 'Shoulder',
                'metric': 'ir_er_ratio',
                'threshold': 2.50,
                'direction': 'above',
                'severity': 'high',
                'source': 'Byram et al., 2010',
            },
            {
                'name': 'Groin Injury Risk (Hip Add:Abd)',
                'test': 'ForceFrame',
                'pattern': 'Hip',
                'metric': 'add_abd_ratio',
                'threshold': 0.80,
                'direction': 'below',
                'severity': 'moderate',
                'source': 'Tyler et al., 2001',
            },
            {
                'name': 'Lead-Leg Excessive Asymmetry',
                'test': 'ForceDecks',
                'test_types': ['CMJ'],
                'metric': 'concentric_impulse_asym',
                'threshold': 20.0,
                'direction': 'above',
                'severity': 'moderate',
                'source': 'Turner et al., 2014',
            },
        ],

        'benchmarks_key': 'Epee',
        'context_notes': [
            'CMJ relative peak power is the strongest discriminator between elite and sub-elite fencers (Turner et al., 2014).',
            'RFD 0-100ms from IMTP predicts attack initiation speed better than peak force alone.',
            'Sabre fencers produce higher jump heights than Foil > Epee (Bottoms et al., 2011).',
            'Lead-leg 15-20% asymmetry is functionally expected due to lunge mechanics.',
            'Weapon arm shoulder ER:IR > 2.5:1 warrants targeted ER strengthening (Byram et al., 2010).',
            'Grip endurance decay >20% over 10 reps indicates forearm overuse risk for epee.',
            'Eccentric decel RFD reflects ability to arrest lunge momentum - critical for repeated attacks.',
            'Concentric RFD has stronger correlation with lunge velocity than peak force (r=0.82 vs r=0.67).',
        ],
    },

    # ========================================================================
    # KARATE (75 athletes)
    # Test types available: ABCMJ (60), IMTP (40), CMJ (39), HJ (39),
    #   PPU (38), SLISOT (38), SJ (20)
    # ========================================================================
    'karate': {
        'display_name': 'Karate',
        'icon': '\U0001F94B',
        'sub_sports': ['Karate'],

        'asymmetry_metrics': [
            {
                'name': 'IMTP Peak Force L/R',
                'test_types': ['IMTP'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'ABCMJ Peak Concentric Force L/R',
                'test_types': ['ABCMJ', 'CMJ'],
                'left_col': 'PEAK_CONCENTRIC_FORCE_Left',
                'right_col': 'PEAK_CONCENTRIC_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'ABCMJ Concentric Impulse L/R',
                'test_types': ['ABCMJ', 'CMJ'],
                'left_col': 'CONCENTRIC_IMPULSE_Left',
                'right_col': 'CONCENTRIC_IMPULSE_Right',
                'unit': 'Ns',
            },
            {
                'name': 'Concentric RFD L/R',
                'test_types': ['ABCMJ', 'CMJ'],
                'left_col': 'CONCENTRIC_RFD_Left',
                'right_col': 'CONCENTRIC_RFD_Right',
                'unit': 'N/s',
            },
            {
                'name': 'Eccentric Decel RFD L/R',
                'test_types': ['ABCMJ', 'CMJ'],
                'left_col': 'ECCENTRIC_DECEL_RFD_Left',
                'right_col': 'ECCENTRIC_DECEL_RFD_Right',
                'unit': 'N/s',
            },
            {
                'name': 'Force at Zero Velocity L/R',
                'test_types': ['ABCMJ', 'CMJ'],
                'left_col': 'FORCE_AT_ZERO_VELOCITY_Left',
                'right_col': 'FORCE_AT_ZERO_VELOCITY_Right',
                'unit': 'N',
            },
        ],
        'asymmetry_thresholds': {'caution': 10.0, 'risk': 15.0},
        'asymmetry_note': (
            'Karate is a bilateral sport - asymmetry >10% should be investigated. '
            'Concentric RFD asymmetry is the #1 discriminator of technique imbalance '
            '(Loturco et al., 2016). Bilateral balance expected for both Kumite and Kata.'
        ),

        'forceframe_sections': [
            {
                'name': 'Shoulder IR/ER',
                'test_pattern': r'Shoulder.*Internal.*External|Shoulder.*Rotation',
                'inner_label': 'Internal Rotation',
                'outer_label': 'External Rotation',
            },
            {
                'name': 'Hip Add/Abd',
                'test_pattern': r'Hip.*Adduction.*Abduction|Hip.*Add.*Abd',
                'inner_label': 'Adduction',
                'outer_label': 'Abduction',
            },
            {
                'name': 'Trunk Flex/Ext',
                'test_pattern': r'Trunk.*Flexion.*Extension|Trunk.*Flex.*Ext',
                'inner_label': 'Flexion',
                'outer_label': 'Extension',
            },
        ],

        'dynamo_section': {
            'enabled': True,
            'movement_filter': 'GripSqueeze',
            'title': 'Grip Strength (DynaMo)',
        },

        'nordbord_section': {
            'enabled': True,
            'title': 'Nordic Hamstring Strength',
        },

        'key_metrics': [
            {
                'label': 'ABCMJ Height',
                'test_types': ['ABCMJ', 'CMJ'],
                'col': 'JUMP_HEIGHT_IMP_MOM',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'IMTP Rel. Peak Force',
                'test_types': ['IMTP'],
                'col': 'ISO_BM_REL_FORCE_PEAK',
                'unit': 'N/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'ABCMJ Rel. Power',
                'test_types': ['ABCMJ', 'CMJ'],
                'col': 'BODYMASS_RELATIVE_TAKEOFF_POWER',
                'unit': 'W/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'Concentric RFD',
                'test_types': ['ABCMJ', 'CMJ'],
                'col': 'CONCENTRIC_RFD',
                'unit': 'N/s',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'RSI Modified',
                'test_types': ['ABCMJ', 'CMJ'],
                'col': 'RSI_MODIFIED',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
            {
                'label': 'PPU Height',
                'test_types': ['PPU'],
                'col': 'PUSHUP_HEIGHT',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'Concentric Duration',
                'test_types': ['ABCMJ', 'CMJ'],
                'col': 'CONCENTRIC_DURATION',
                'unit': 's',
                'format': '.3f',
                'higher_is_better': False,
            },
        ],

        'extra_sections': [
            {
                'type': 'hop_test',
                'title': '10:5 Hop / Reactive Strength',
                'test_types': ['HJ', 'SLHJ', 'RSHIP', 'RSKIP', 'RSAIP'],
                'metric_key': 'rsi',
            },
            {
                'type': 'ppu',
                'title': 'Plyo Pushup (Upper Body Power)',
                'test_types': ['PPU'],
                'metric_key': 'peak_force',
            },
            {
                'type': 'eur',
                'title': 'Eccentric Utilization Ratio (ABCMJ:SJ)',
                'test_types_num': ['ABCMJ', 'CMJ'],
                'test_types_den': ['SJ'],
                'col_num': 'JUMP_HEIGHT_IMP_MOM',
                'col_den': 'JUMP_HEIGHT_IMP_MOM',
            },
        ],

        # ---- Benchmarks ----
        'benchmarks': {
            'cmj_height': {
                'elite_m': (45, 55), 'national_m': (38, 45),
                'elite_f': (32, 40), 'national_f': (27, 32),
                'unit': 'cm',
                'source': 'Loturco et al., 2016; Chaabene et al., 2012',
            },
            'imtp_relative': {
                'elite_m': (33, 40), 'national_m': (28, 33),
                'elite_f': (27, 33), 'national_f': (23, 27),
                'unit': 'N/kg',
                'source': 'Loturco et al., 2016',
            },
            'ppu_height': {
                'elite_m': (14, 20), 'national_m': (10, 14),
                'elite_f': (8, 13), 'national_f': (5, 8),
                'unit': 'cm',
                'source': 'Loturco et al., 2017',
            },
            'rsi': {
                'elite_m': (2.0, None), 'national_m': (1.5, 2.0),
                'elite_f': (1.5, None), 'national_f': (1.2, 1.5),
                'unit': '',
                'source': 'Chaabene et al., 2012',
            },
        },

        # ---- Performance ratios ----
        'performance_ratios': [
            {
                'name': 'Eccentric Utilization Ratio',
                'description': 'ABCMJ:SJ height ratio. Karate athletes with EUR 1.08-1.15 show optimal SSC utilization.',
                'formula': 'ABCMJ Height / SJ Height',
                'optimal': (1.08, 1.20),
                'concern_low': 1.05,
                'concern_high': 1.25,
                'test_types_num': ['ABCMJ', 'CMJ'],
                'test_types_den': ['SJ'],
                'col_num': 'JUMP_HEIGHT_IMP_MOM',
                'col_den': 'JUMP_HEIGHT_IMP_MOM',
            },
            {
                'name': 'Trunk Ext:Flex Ratio',
                'description': 'Trunk extension:flexion ratio. Correlates with rotational punch/kick power.',
                'source': 'forceframe',
                'test_pattern': 'Trunk',
                'optimal': (1.0, 1.3),
                'concern_low': 0.8,
                'concern_high': 1.5,
                'citation': 'Chaabene et al., 2012',
            },
            {
                'name': 'Hip Add:Abd Ratio',
                'description': 'Adduction:Abduction ratio for groin protection during high kicks.',
                'source': 'forceframe',
                'test_pattern': 'Hip',
                'optimal': (0.80, 1.20),
                'concern_low': 0.80,
                'concern_high': 1.30,
                'citation': 'Tyler et al., 2001',
            },
        ],

        # ---- Injury risk indicators ----
        'injury_risk_indicators': [
            {
                'name': 'Hamstring Injury Risk',
                'test': 'NordBord',
                'metric': 'absolute_force_per_leg',
                'threshold_male': 337,
                'threshold_female': 256,
                'direction': 'below',
                'severity': 'high',
                'source': 'Opar et al., 2015 BJSM',
            },
            {
                'name': 'Bilateral Force Imbalance',
                'test': 'ForceDecks',
                'test_types': ['ABCMJ', 'CMJ'],
                'metric': 'concentric_impulse_asym',
                'threshold': 15.0,
                'direction': 'above',
                'severity': 'moderate',
                'source': 'Impellizzeri et al., 2007',
            },
        ],

        'benchmarks_key': None,
        'context_notes': [
            'Karate uses ABCMJ (Abalakov CMJ) - arm swing included for ecological validity.',
            'CMJ concentric mean power correlates r=0.74 with WKF ranking (Loturco et al., 2016).',
            'RFD 0-100ms is the #1 discriminator between medal and non-medal Kumite athletes.',
            'Bilateral balance expected (<10% asymmetry) - unlike unilateral sports.',
            'PPU height 14-20cm (elite M) reflects upper body explosive power for punching.',
            'Trunk rotation strength (ForceFrame) correlates strongly with punch force (r=0.68).',
            'SJ data available (n=20) - calculate EUR for SSC efficiency monitoring.',
            'SL ISO Squat (SLISOT, n=38) provides unilateral strength screening for stance stability.',
        ],
    },

    # ========================================================================
    # TAEKWONDO (48 athletes)
    # Test types available: CMJ (92), DJ (85), SLDJ (78) -- DJ and SLDJ
    #   are KEY for TKD reactive strength
    # ========================================================================
    'taekwondo': {
        'display_name': 'Taekwondo',
        'icon': '\U0001F94B',
        'sub_sports': ['Taekwondo', 'Taekwondo - Junior', 'Taekwondo - Senior'],

        'asymmetry_metrics': [
            {
                'name': 'IMTP Peak Force L/R',
                'test_types': ['IMTP'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Peak Concentric Force L/R',
                'test_types': ['CMJ'],
                'left_col': 'PEAK_CONCENTRIC_FORCE_Left',
                'right_col': 'PEAK_CONCENTRIC_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Concentric Impulse L/R',
                'test_types': ['CMJ'],
                'left_col': 'CONCENTRIC_IMPULSE_Left',
                'right_col': 'CONCENTRIC_IMPULSE_Right',
                'unit': 'Ns',
            },
            {
                'name': 'Concentric RFD L/R',
                'test_types': ['CMJ'],
                'left_col': 'CONCENTRIC_RFD_Left',
                'right_col': 'CONCENTRIC_RFD_Right',
                'unit': 'N/s',
            },
            {
                'name': 'Eccentric Braking RFD L/R',
                'test_types': ['CMJ'],
                'left_col': 'ECCENTRIC_BRAKING_RFD_Left',
                'right_col': 'ECCENTRIC_BRAKING_RFD_Right',
                'unit': 'N/s',
            },
            {
                'name': 'DJ Active Stiffness L/R',
                'test_types': ['DJ', 'SLDJ'],
                'left_col': 'ACTIVE_STIFFNESS_Left',
                'right_col': 'ACTIVE_STIFFNESS_Right',
                'unit': 'N/m',
            },
            {
                'name': 'DJ Peak Driveoff Force L/R',
                'test_types': ['DJ', 'SLDJ'],
                'left_col': 'PEAK_DRIVEOFF_FORCE_Left',
                'right_col': 'PEAK_DRIVEOFF_FORCE_Right',
                'unit': 'N',
            },
        ],
        'asymmetry_thresholds': {'caution': 12.0, 'risk': 18.0},
        'asymmetry_note': (
            'Kicking leg 8-12% asymmetry is functionally expected. '
            'Flag at 18% - excessive asymmetry increases ACL/hamstring risk during kicks. '
            'DJ asymmetry is particularly important for landing from spinning kicks.'
        ),

        'forceframe_sections': [],

        'dynamo_section': {
            'enabled': True,
            'movement_filter': 'GripSqueeze',
            'title': 'Grip Strength (DynaMo) - Clinch Technique',
        },

        'nordbord_section': {
            'enabled': True,
            'title': 'Nordic Hamstring Strength - Critical for Kick Protection',
        },

        'key_metrics': [
            {
                'label': 'CMJ Height',
                'test_types': ['CMJ'],
                'col': 'JUMP_HEIGHT_IMP_MOM',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'IMTP Rel. Peak Force',
                'test_types': ['IMTP'],
                'col': 'ISO_BM_REL_FORCE_PEAK',
                'unit': 'N/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'CMJ Rel. Power',
                'test_types': ['CMJ'],
                'col': 'BODYMASS_RELATIVE_TAKEOFF_POWER',
                'unit': 'W/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'DJ RSI',
                'test_types': ['DJ', 'SLDJ'],
                'col': 'RSI',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
            {
                'label': 'DJ Active Stiffness',
                'test_types': ['DJ', 'SLDJ'],
                'col': 'ACTIVE_STIFFNESS',
                'unit': 'N/m',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'Concentric RFD',
                'test_types': ['CMJ'],
                'col': 'CONCENTRIC_RFD',
                'unit': 'N/s',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'Eccentric Decel RFD',
                'test_types': ['CMJ'],
                'col': 'ECCENTRIC_DECEL_RFD',
                'unit': 'N/s',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'RSI Modified',
                'test_types': ['CMJ'],
                'col': 'RSI_MODIFIED',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
        ],

        'extra_sections': [
            {
                'type': 'drop_jump',
                'title': 'Drop Jump / Reactive Strength (DJ & SLDJ)',
                'test_types': ['DJ', 'SLDJ'],
                'metric_key': 'rsi',
            },
        ],

        # ---- Benchmarks ----
        'benchmarks': {
            'cmj_height': {
                'elite_m': (42, 52), 'national_m': (36, 42),
                'elite_f': (30, 38), 'national_f': (25, 30),
                'unit': 'cm',
                'source': 'Santos & Franchini, 2018; Loturco et al., 2022',
            },
            'imtp_relative': {
                'elite_m': (32, 38), 'national_m': (27, 32),
                'elite_f': (26, 32), 'national_f': (22, 26),
                'unit': 'N/kg',
                'source': 'Bridge et al., 2014',
            },
            'rsi': {
                'elite_m': (2.0, None), 'national_m': (1.5, 2.0),
                'elite_f': (1.6, None), 'national_f': (1.2, 1.6),
                'unit': '',
                'source': 'Santos & Franchini, 2018',
            },
            'dj_rsi': {
                'elite_m': (1.8, 2.5), 'national_m': (1.3, 1.8),
                'elite_f': (1.4, 2.0), 'national_f': (1.0, 1.4),
                'unit': '',
                'source': 'Suchomel et al., 2015',
            },
        },

        # ---- Performance ratios ----
        'performance_ratios': [
            {
                'name': 'DJ:CMJ Height Ratio',
                'description': 'Drop jump vs CMJ height. Reflects reactive strength capacity relative to slow SSC.',
                'formula': 'DJ Height / CMJ Height',
                'optimal': (0.85, 1.05),
                'concern_low': 0.70,
                'concern_high': None,
                'test_types_num': ['DJ'],
                'test_types_den': ['CMJ'],
                'col_num': 'JUMP_HEIGHT_IMP_MOM',
                'col_den': 'JUMP_HEIGHT_IMP_MOM',
            },
        ],

        # ---- Injury risk indicators ----
        'injury_risk_indicators': [
            {
                'name': 'Hamstring Injury Risk (Kicking)',
                'test': 'NordBord',
                'metric': 'absolute_force_per_leg',
                'threshold_male': 370,
                'threshold_female': 280,
                'direction': 'below',
                'severity': 'high',
                'source': 'Opar et al., 2015 BJSM; higher threshold due to ballistic kicking demands',
            },
            {
                'name': 'NordBord L/R Asymmetry',
                'test': 'NordBord',
                'metric': 'bilateral_asymmetry',
                'threshold': 15.0,
                'direction': 'above',
                'severity': 'high',
                'source': 'Croisier et al., 2008',
            },
            {
                'name': 'DJ Landing Force Asymmetry',
                'test': 'ForceDecks',
                'test_types': ['DJ', 'SLDJ'],
                'metric': 'peak_impact_force_asym',
                'threshold': 15.0,
                'direction': 'above',
                'severity': 'moderate',
                'source': 'Hewit et al., 2012',
            },
        ],

        'benchmarks_key': None,
        'context_notes': [
            'TKD has the RICHEST DJ/SLDJ data (85 DJ, 78 SLDJ tests) - reactive strength is primary performance driver.',
            'Kicking leg 8-12% asymmetry is functionally expected (Santos & Franchini, 2018).',
            'NordBord is CRITICAL for hamstring protection during ballistic roundhouse kicks.',
            'Hip flexion strength >3.0 N/kg correlates with kick height and speed.',
            'DJ RSI reflects fast SSC capacity needed for rapid kick exchanges.',
            'Active stiffness from DJ predicts landing control during spinning techniques.',
            'Grip strength (DynaMo) relevant for clinching in close-range exchanges.',
            'CMJ concentric RFD predicts kick initiation speed (r=0.71, Bridge et al., 2014).',
        ],
    },

    # ========================================================================
    # WRESTLING (31 athletes)
    # Test types available: CMJ (61), HJ (65), IMTP (58), PPU (40),
    #   SLISOSQT (31), RSHIP (27), SJ (10)
    # ========================================================================
    'wrestling': {
        'display_name': 'Wrestling',
        'icon': '\U0001F93C',
        'sub_sports': ['Wrestling', 'Wrestling - Freestyle', 'Wrestling - Greco Roman'],

        'asymmetry_metrics': [
            {
                'name': 'IMTP Peak Force L/R',
                'test_types': ['IMTP'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Peak Concentric Force L/R',
                'test_types': ['CMJ'],
                'left_col': 'PEAK_CONCENTRIC_FORCE_Left',
                'right_col': 'PEAK_CONCENTRIC_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'SL ISO Squat Peak Force L/R',
                'test_types': ['SLISOSQT', 'SLISOT'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'Concentric RFD L/R',
                'test_types': ['CMJ'],
                'left_col': 'CONCENTRIC_RFD_Left',
                'right_col': 'CONCENTRIC_RFD_Right',
                'unit': 'N/s',
            },
            {
                'name': 'CMJ Stiffness L/R',
                'test_types': ['CMJ'],
                'left_col': 'CMJ_STIFFNESS_Left',
                'right_col': 'CMJ_STIFFNESS_Right',
                'unit': 'N/m',
            },
            {
                'name': 'Eccentric Braking RFD L/R',
                'test_types': ['CMJ'],
                'left_col': 'ECCENTRIC_BRAKING_RFD_Left',
                'right_col': 'ECCENTRIC_BRAKING_RFD_Right',
                'unit': 'N/s',
            },
        ],
        'asymmetry_thresholds': {'caution': 10.0, 'risk': 15.0},
        'asymmetry_note': (
            'Bilateral balance is critical for takedown technique and defense. '
            'Flag any asymmetry >10%. SL ISO Squat asymmetry is the most '
            'functionally relevant test for wrestling stance stability.'
        ),

        'forceframe_sections': [
            {
                'name': 'Shoulder IR/ER',
                'test_pattern': r'Shoulder.*Internal.*External|Shoulder.*Rotation',
                'inner_label': 'Internal Rotation',
                'outer_label': 'External Rotation',
            },
            {
                'name': 'Hip Add/Abd',
                'test_pattern': r'Hip.*Adduction.*Abduction|Hip.*Add.*Abd',
                'inner_label': 'Adduction',
                'outer_label': 'Abduction',
            },
            {
                'name': '4-Way Neck',
                'test_pattern': r'4.*Way.*Neck|Neck',
                'inner_label': 'Flexion/Extension',
                'outer_label': 'Lateral Flexion',
            },
        ],

        'dynamo_section': {
            'enabled': True,
            'movement_filter': 'GripSqueeze',
            'title': 'Grip Strength (DynaMo) - Critical for Wrestling',
        },

        'nordbord_section': {
            'enabled': True,
            'title': 'Nordic Hamstring Strength',
        },

        'key_metrics': [
            {
                'label': 'IMTP Rel. Peak Force',
                'test_types': ['IMTP'],
                'col': 'ISO_BM_REL_FORCE_PEAK',
                'unit': 'N/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'IMTP Abs Peak Force',
                'test_types': ['IMTP'],
                'col': 'PEAK_VERTICAL_FORCE',
                'unit': 'N',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'CMJ Height',
                'test_types': ['CMJ'],
                'col': 'JUMP_HEIGHT_IMP_MOM',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'CMJ Rel. Power',
                'test_types': ['CMJ'],
                'col': 'BODYMASS_RELATIVE_TAKEOFF_POWER',
                'unit': 'W/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'Grip Strength',
                'test_types': [],
                'col': 'maxForceNewtons',
                'unit': 'N',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'PPU Height',
                'test_types': ['PPU'],
                'col': 'PUSHUP_HEIGHT',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'RSI (Hop)',
                'test_types': ['HJ', 'SLHJ', 'RSHIP'],
                'col': 'HOP_BEST_RSI',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
            {
                'label': 'RSI Modified',
                'test_types': ['CMJ'],
                'col': 'RSI_MODIFIED',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
        ],

        'extra_sections': [
            {
                'type': 'hop_test',
                'title': '10:5 Hop / Reactive Strength',
                'test_types': ['HJ', 'SLHJ', 'RSHIP', 'RSKIP', 'RSAIP'],
                'metric_key': 'rsi',
            },
            {
                'type': 'ppu',
                'title': 'Plyo Pushup (Upper Body Power)',
                'test_types': ['PPU'],
                'metric_key': 'peak_force',
            },
            {
                'type': 'strength_rm',
                'title': 'Strength RM (Manual Entry)',
                'test_types': None,
                'metric_key': 'peak_force',
            },
            {
                'type': 'neck_strength',
                'title': '4-Way Neck Strength (ForceFrame) - Concussion Prevention',
                'test_pattern': r'4.*Way.*Neck|Neck',
            },
            {
                'type': 'eur',
                'title': 'Eccentric Utilization Ratio (CMJ:SJ)',
                'test_types_num': ['CMJ'],
                'test_types_den': ['SJ'],
                'col_num': 'JUMP_HEIGHT_IMP_MOM',
                'col_den': 'JUMP_HEIGHT_IMP_MOM',
            },
        ],

        # ---- Benchmarks ----
        'benchmarks': {
            'cmj_height': {
                'elite_m': (40, 50), 'national_m': (35, 40),
                'elite_f': (28, 35), 'national_f': (23, 28),
                'unit': 'cm',
                'source': 'Chaabene et al., 2017; Loturco et al., 2018',
            },
            'imtp_absolute': {
                'elite_m': (2800, 4200), 'national_m': (2200, 2800),
                'elite_f': (1800, 2600), 'national_f': (1400, 1800),
                'unit': 'N',
                'source': 'Comfort et al., 2014',
            },
            'imtp_relative': {
                'elite_m': (36, 45), 'national_m': (30, 36),
                'elite_f': (28, 35), 'national_f': (24, 28),
                'unit': 'N/kg',
                'source': 'Comfort et al., 2014',
            },
            'grip_relative': {
                'elite_m': (7.5, None), 'national_m': (6.0, 7.5),
                'elite_f': (6.0, None), 'national_f': (4.5, 6.0),
                'unit': 'N/kg BM',
                'source': 'Gerodimos et al., 2013',
            },
            'nordbord_per_leg': {
                'elite_m': (420, None), 'national_m': (337, 420),
                'elite_f': (300, None), 'national_f': (256, 300),
                'unit': 'N',
                'source': 'Opar et al., 2015',
            },
        },

        # ---- Performance ratios ----
        'performance_ratios': [
            {
                'name': 'Eccentric Utilization Ratio',
                'description': 'CMJ:SJ height ratio. Wrestling athletes typically 1.08-1.15.',
                'formula': 'CMJ Height / SJ Height',
                'optimal': (1.08, 1.15),
                'concern_low': 1.05,
                'concern_high': 1.20,
                'test_types_num': ['CMJ'],
                'test_types_den': ['SJ'],
                'col_num': 'JUMP_HEIGHT_IMP_MOM',
                'col_den': 'JUMP_HEIGHT_IMP_MOM',
            },
            {
                'name': 'Shoulder IR:ER Ratio',
                'description': 'Both arms equally vulnerable in wrestling. Balance is key.',
                'source': 'forceframe',
                'test_pattern': 'Shoulder',
                'optimal': (1.33, 1.67),
                'concern_high': 2.0,
                'risk_high': 2.5,
                'citation': 'Byram et al., 2010',
            },
        ],

        # ---- Injury risk indicators ----
        'injury_risk_indicators': [
            {
                'name': 'Hamstring Injury Risk',
                'test': 'NordBord',
                'metric': 'absolute_force_per_leg',
                'threshold_male': 420,
                'threshold_female': 300,
                'direction': 'below',
                'severity': 'high',
                'source': 'Opar et al., 2015; higher threshold for explosive takedowns',
            },
            {
                'name': 'Neck Strength Deficit (Concussion Risk)',
                'test': 'ForceFrame',
                'pattern': 'Neck',
                'metric': 'min_direction_force',
                'threshold_male': 150,
                'threshold_female': 100,
                'direction': 'below',
                'severity': 'high',
                'source': 'Eckner et al., 2014; Collins et al., 2014',
            },
            {
                'name': 'NordBord L/R Asymmetry',
                'test': 'NordBord',
                'metric': 'bilateral_asymmetry',
                'threshold': 10.0,
                'direction': 'above',
                'severity': 'moderate',
                'source': 'Croisier et al., 2008',
            },
        ],

        'benchmarks_key': 'Wrestling',
        'context_notes': [
            'Grip strength >7.5 N/kg BM is the threshold for international-level wrestling (Gerodimos et al., 2013).',
            'IMTP absolute force 2800-4200N for elite males reflects takedown power demand.',
            '4-Way Neck strength is CRITICAL for concussion prevention - a unique wrestling priority.',
            'Pull>Push ratio expected due to clinch-dominant fighting style.',
            'NordBord >420N per leg recommended given explosive takedown demands.',
            'SJ data available (n=10) - calculate EUR for SSC efficiency monitoring.',
            'SL ISO Squat (SLISOSQT, n=31) assesses single-leg stance stability for shot defense.',
            'PPU (n=40) reflects upper body explosive power needed for throws and takedowns.',
        ],
    },

    # ========================================================================
    # JUDO (40 athletes)
    # Test types available: CMJ (15), IMTP (10), ISOT (9), RSKIP (6),
    #   SLISOSQT (2), SJ (1)
    # ========================================================================
    'judo': {
        'display_name': 'Judo',
        'icon': '\U0001F94B',
        'sub_sports': ['Judo'],

        'asymmetry_metrics': [
            {
                'name': 'IMTP Peak Force L/R',
                'test_types': ['IMTP'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Peak Concentric Force L/R',
                'test_types': ['CMJ'],
                'left_col': 'PEAK_CONCENTRIC_FORCE_Left',
                'right_col': 'PEAK_CONCENTRIC_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Concentric Impulse L/R',
                'test_types': ['CMJ'],
                'left_col': 'CONCENTRIC_IMPULSE_Left',
                'right_col': 'CONCENTRIC_IMPULSE_Right',
                'unit': 'Ns',
            },
            {
                'name': 'Eccentric Braking RFD L/R',
                'test_types': ['CMJ'],
                'left_col': 'ECCENTRIC_BRAKING_RFD_Left',
                'right_col': 'ECCENTRIC_BRAKING_RFD_Right',
                'unit': 'N/s',
            },
            {
                'name': 'Force at Zero Velocity L/R',
                'test_types': ['CMJ'],
                'left_col': 'FORCE_AT_ZERO_VELOCITY_Left',
                'right_col': 'FORCE_AT_ZERO_VELOCITY_Right',
                'unit': 'N',
            },
        ],
        'asymmetry_thresholds': {'caution': 10.0, 'risk': 15.0},
        'asymmetry_note': (
            'Judo is a bilateral combat sport - some throw-side dominance is expected '
            'but >10% indicates strength deficit that may limit technique versatility.'
        ),

        'forceframe_sections': [],

        'dynamo_section': None,

        'nordbord_section': None,

        'key_metrics': [
            {
                'label': 'CMJ Height',
                'test_types': ['CMJ'],
                'col': 'JUMP_HEIGHT_IMP_MOM',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'IMTP Rel. Peak Force',
                'test_types': ['IMTP'],
                'col': 'ISO_BM_REL_FORCE_PEAK',
                'unit': 'N/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'IMTP Net Peak Force',
                'test_types': ['IMTP'],
                'col': 'NET_PEAK_VERTICAL_FORCE',
                'unit': 'N',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'CMJ Rel. Power',
                'test_types': ['CMJ'],
                'col': 'BODYMASS_RELATIVE_TAKEOFF_POWER',
                'unit': 'W/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'Concentric RFD',
                'test_types': ['CMJ'],
                'col': 'CONCENTRIC_RFD',
                'unit': 'N/s',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'RSI Modified',
                'test_types': ['CMJ'],
                'col': 'RSI_MODIFIED',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
            {
                'label': 'Ecc:Con Peak Power Ratio',
                'test_types': ['CMJ'],
                'col': 'ECC_CON_PEAK_POWER_RATIO',
                'unit': '',
                'format': '.2f',
                'higher_is_better': False,
            },
        ],

        'extra_sections': [
            {
                'type': 'hop_test',
                'title': 'Reactive Strength (RSKIP)',
                'test_types': ['RSKIP', 'HJ', 'SLHJ', 'RSHIP', 'RSAIP'],
                'metric_key': 'rsi',
            },
        ],

        # ---- Benchmarks ----
        'benchmarks': {
            'cmj_height': {
                'elite_m': (40, 50), 'national_m': (34, 40),
                'elite_f': (28, 35), 'national_f': (23, 28),
                'unit': 'cm',
                'source': 'Franchini et al., 2011; Detanico et al., 2012',
            },
            'imtp_relative': {
                'elite_m': (34, 42), 'national_m': (28, 34),
                'elite_f': (27, 33), 'national_f': (23, 27),
                'unit': 'N/kg',
                'source': 'Franchini et al., 2011',
            },
        },

        # ---- Performance ratios ----
        'performance_ratios': [
            {
                'name': 'Eccentric Utilization Ratio',
                'description': 'CMJ:SJ height. Judo EUR typically 1.08-1.15 reflecting moderate SSC reliance.',
                'formula': 'CMJ Height / SJ Height',
                'optimal': (1.08, 1.15),
                'concern_low': 1.05,
                'concern_high': 1.20,
                'test_types_num': ['CMJ'],
                'test_types_den': ['SJ'],
                'col_num': 'JUMP_HEIGHT_IMP_MOM',
                'col_den': 'JUMP_HEIGHT_IMP_MOM',
                'note': 'Limited SJ data (n=1) - collect more for reliable EUR calculation.',
            },
        ],

        # ---- Injury risk indicators ----
        'injury_risk_indicators': [
            {
                'name': 'Shoulder Injury Risk (Throwing Arm)',
                'test': 'ForceDecks',
                'test_types': ['ISOT'],
                'metric': 'peak_force_asymmetry',
                'threshold': 15.0,
                'direction': 'above',
                'severity': 'moderate',
                'source': 'Kawamura & Daigo, 2002',
            },
        ],

        'benchmarks_key': 'Judo',
        'context_notes': [
            'Grip endurance is the #1 discriminator of judo success (Bonitch-Gongora et al., 2012).',
            'RFD 0-200ms from IMTP predicts throwing success better than peak force (Detanico et al., 2012).',
            'EUR (CMJ:SJ) of 1.08-1.15 is typical - higher values suggest over-reliance on elastic energy.',
            'Weight class considerations make relative strength (N/kg, W/kg) more important than absolute.',
            'Isometric testing (ISOT, n=9) reflects holding strength critical for gripping battles.',
            'RSKIP test (n=6) available for reactive skip assessment - unique to judo footwork.',
            'Eccentric braking RFD reflects ability to resist throws and control landing from hip throws.',
        ],
    },

    # ========================================================================
    # JIU-JITSU (32 athletes)
    # Test types available: CMJ (8), SLJ (7), ISOSQT (2), HJ (2), PPU (2)
    # ========================================================================
    'jiu-jitsu': {
        'display_name': 'Jiu-Jitsu',
        'icon': '\U0001F94B',
        'sub_sports': ['Jiu-Jitsu'],

        'asymmetry_metrics': [
            {
                'name': 'IMTP Peak Force L/R',
                'test_types': ['IMTP'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Peak Concentric Force L/R',
                'test_types': ['CMJ'],
                'left_col': 'PEAK_CONCENTRIC_FORCE_Left',
                'right_col': 'PEAK_CONCENTRIC_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Concentric Impulse L/R',
                'test_types': ['CMJ'],
                'left_col': 'CONCENTRIC_IMPULSE_Left',
                'right_col': 'CONCENTRIC_IMPULSE_Right',
                'unit': 'Ns',
            },
            {
                'name': 'Eccentric Braking RFD L/R',
                'test_types': ['CMJ'],
                'left_col': 'ECCENTRIC_BRAKING_RFD_Left',
                'right_col': 'ECCENTRIC_BRAKING_RFD_Right',
                'unit': 'N/s',
            },
        ],
        'asymmetry_thresholds': {'caution': 10.0, 'risk': 15.0},
        'asymmetry_note': (
            'Bilateral shoulder conditioning is critical - both arms vulnerable to submission '
            'locks. Hip adduction asymmetry relates to guard retention ability.'
        ),

        'forceframe_sections': [
            {
                'name': 'Shoulder IR/ER',
                'test_pattern': r'Shoulder.*Internal.*External|Shoulder.*Rotation',
                'inner_label': 'Internal Rotation',
                'outer_label': 'External Rotation',
            },
            {
                'name': 'Hip Add/Abd',
                'test_pattern': r'Hip.*Adduction.*Abduction|Hip.*Add.*Abd',
                'inner_label': 'Adduction',
                'outer_label': 'Abduction',
            },
        ],

        'dynamo_section': None,

        'nordbord_section': None,

        'key_metrics': [
            {
                'label': 'CMJ Height',
                'test_types': ['CMJ'],
                'col': 'JUMP_HEIGHT_IMP_MOM',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'IMTP Rel. Peak Force',
                'test_types': ['IMTP'],
                'col': 'ISO_BM_REL_FORCE_PEAK',
                'unit': 'N/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'CMJ Rel. Power',
                'test_types': ['CMJ'],
                'col': 'BODYMASS_RELATIVE_TAKEOFF_POWER',
                'unit': 'W/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'Ecc:Con Peak Power Ratio',
                'test_types': ['CMJ'],
                'col': 'ECC_CON_PEAK_POWER_RATIO',
                'unit': '',
                'format': '.2f',
                'higher_is_better': False,
            },
            {
                'label': 'PPU Height',
                'test_types': ['PPU'],
                'col': 'PUSHUP_HEIGHT',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
        ],

        'extra_sections': [
            {
                'type': 'ppu',
                'title': 'Plyo Pushup (Upper Body Power)',
                'test_types': ['PPU'],
                'metric_key': 'peak_force',
            },
        ],

        # ---- Benchmarks ----
        'benchmarks': {
            'cmj_height': {
                'elite_m': (38, 46), 'national_m': (32, 38),
                'elite_f': (26, 33), 'national_f': (22, 26),
                'unit': 'cm',
                'source': 'Diaz-Lara et al., 2014',
            },
            'imtp_relative': {
                'elite_m': (32, 40), 'national_m': (27, 32),
                'elite_f': (26, 32), 'national_f': (22, 26),
                'unit': 'N/kg',
                'source': 'Estimated from combat sport norms',
            },
            'ppu_height': {
                'elite_m': (12, 18), 'national_m': (8, 12),
                'elite_f': (6, 10), 'national_f': (3, 6),
                'unit': 'cm',
                'source': 'Estimated from combat sport norms',
            },
        },

        # ---- Performance ratios ----
        'performance_ratios': [
            {
                'name': 'Shoulder IR:ER Ratio (Both Arms)',
                'description': 'Both arms equally at risk from arm locks and kimura. Balance critical.',
                'source': 'forceframe',
                'test_pattern': 'Shoulder',
                'optimal': (1.33, 1.67),
                'concern_high': 2.0,
                'risk_high': 2.5,
                'citation': 'Byram et al., 2010',
            },
            {
                'name': 'Hip Add:Abd Ratio',
                'description': 'Hip adduction strength underpins closed guard retention.',
                'source': 'forceframe',
                'test_pattern': 'Hip',
                'optimal': (0.90, 1.30),
                'concern_low': 0.80,
                'concern_high': 1.40,
                'citation': 'Tyler et al., 2001',
            },
        ],

        # ---- Injury risk indicators ----
        'injury_risk_indicators': [
            {
                'name': 'Shoulder Impingement Risk',
                'test': 'ForceFrame',
                'pattern': 'Shoulder',
                'metric': 'ir_er_ratio',
                'threshold': 2.0,
                'direction': 'above',
                'severity': 'high',
                'source': 'Byram et al., 2010; elevated risk due to arm-lock exposure',
            },
        ],

        'benchmarks_key': 'Jiu-Jitsu',
        'context_notes': [
            'Isometric endurance and force maintenance are more important than peak explosive power.',
            'Grip endurance >50s sustained hold is a key discriminator (Diaz-Lara et al., 2014).',
            'Bilateral shoulder conditioning is critical - both arms vulnerable to arm locks and kimura.',
            'Hip adduction (guard) strength index: Add force / BM is a BJJ-specific metric.',
            'Ground-based power and hip mobility underpin guard passing and sweeps.',
            'PPU (n=2) and SLJ (n=7) provide limited but useful upper/lower body power data.',
            'Isometric squat (ISOSQT, n=2) reflects isometric strength for positional control.',
        ],
    },

    # ========================================================================
    # SWIMMING (17 athletes)
    # Test types available: HJ (9), CMJ (7), IMTP (6), SHLDISOI (6),
    #   SLJ (5), SHLDISOY (5), SHLDISOT (5), PPU (4), SLISOSQT (3)
    # ========================================================================
    'swimming': {
        'display_name': 'Swimming',
        'icon': '\U0001F3CA',
        'sub_sports': ['Swimming'],

        'asymmetry_metrics': [
            {
                'name': 'IMTP Peak Force L/R',
                'test_types': ['IMTP'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Peak Concentric Force L/R',
                'test_types': ['CMJ'],
                'left_col': 'PEAK_CONCENTRIC_FORCE_Left',
                'right_col': 'PEAK_CONCENTRIC_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Concentric Impulse L/R',
                'test_types': ['CMJ'],
                'left_col': 'CONCENTRIC_IMPULSE_Left',
                'right_col': 'CONCENTRIC_IMPULSE_Right',
                'unit': 'Ns',
            },
            {
                'name': 'Concentric RFD L/R',
                'test_types': ['CMJ'],
                'left_col': 'CONCENTRIC_RFD_Left',
                'right_col': 'CONCENTRIC_RFD_Right',
                'unit': 'N/s',
            },
            {
                'name': 'Eccentric Decel RFD L/R',
                'test_types': ['CMJ'],
                'left_col': 'ECCENTRIC_DECEL_RFD_Left',
                'right_col': 'ECCENTRIC_DECEL_RFD_Right',
                'unit': 'N/s',
            },
            {
                'name': 'Peak Takeoff Force L/R',
                'test_types': ['CMJ'],
                'left_col': 'PEAK_TAKEOFF_FORCE_Left',
                'right_col': 'PEAK_TAKEOFF_FORCE_Right',
                'unit': 'N',
            },
        ],
        'asymmetry_thresholds': {'caution': 5.0, 'risk': 10.0},
        'asymmetry_note': (
            'STRICTEST bilateral symmetry requirement of any sport. '
            'Flag ANY asymmetry >5%. Even 3-5% asymmetry affects stroke efficiency '
            'and can cause compensatory patterns leading to shoulder injury (West et al., 2011).'
        ),

        'forceframe_sections': [
            {
                'name': 'Shoulder IR/ER',
                'test_pattern': r'Shoulder.*Internal.*External|Shoulder.*Rotation',
                'inner_label': 'Internal Rotation',
                'outer_label': 'External Rotation',
            },
        ],

        'dynamo_section': None,

        'nordbord_section': {
            'enabled': True,
            'title': 'Nordic Hamstring Strength',
        },

        'key_metrics': [
            {
                'label': 'CMJ Height',
                'test_types': ['CMJ'],
                'col': 'JUMP_HEIGHT_IMP_MOM',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'IMTP Net Peak Force',
                'test_types': ['IMTP'],
                'col': 'NET_PEAK_VERTICAL_FORCE',
                'unit': 'N',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'CMJ Rel. Power',
                'test_types': ['CMJ'],
                'col': 'BODYMASS_RELATIVE_TAKEOFF_POWER',
                'unit': 'W/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'RSI (Hop)',
                'test_types': ['HJ', 'SLHJ'],
                'col': 'HOP_BEST_RSI',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
            {
                'label': 'PPU Height',
                'test_types': ['PPU'],
                'col': 'PUSHUP_HEIGHT',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'Concentric Duration',
                'test_types': ['CMJ'],
                'col': 'CONCENTRIC_DURATION',
                'unit': 's',
                'format': '.3f',
                'higher_is_better': False,
            },
            {
                'label': 'RSI Modified',
                'test_types': ['CMJ'],
                'col': 'RSI_MODIFIED',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
        ],

        'extra_sections': [
            {
                'type': 'hop_test',
                'title': '10:5 Hop / Reactive Strength',
                'test_types': ['HJ', 'SLHJ', 'RSHIP', 'RSKIP', 'RSAIP'],
                'metric_key': 'rsi',
            },
            {
                'type': 'ppu',
                'title': 'Plyo Pushup (Upper Body Power)',
                'test_types': ['PPU'],
                'metric_key': 'peak_force',
            },
            {
                'type': 'shoulder_isometric',
                'title': 'Shoulder Isometric Tests (SHLDISO)',
                'test_types': ['SHLDISOI', 'SHLDISOY', 'SHLDISOT'],
                'metric_key': 'peak_force',
            },
        ],

        # ---- Benchmarks ----
        'benchmarks': {
            'cmj_height': {
                'elite_m': (42, 50), 'national_m': (36, 42),
                'elite_f': (30, 38), 'national_f': (25, 30),
                'unit': 'cm',
                'source': 'West et al., 2011; Bishop et al., 2013',
            },
            'imtp_relative': {
                'elite_m': (32, 40), 'national_m': (27, 32),
                'elite_f': (26, 33), 'national_f': (22, 26),
                'unit': 'N/kg',
                'source': 'Comfort et al., 2014',
            },
            'shoulder_er_ir_ratio': {
                'elite_m': (0.66, None), 'national_m': (0.60, 0.66),
                'elite_f': (0.66, None), 'national_f': (0.60, 0.66),
                'unit': 'ratio (ER/IR)',
                'source': 'Ellenbecker & Davies, 2000; critical injury prevention threshold',
            },
        },

        # ---- Performance ratios ----
        'performance_ratios': [
            {
                'name': 'Shoulder ER:IR Ratio',
                'description': 'External:Internal rotation ratio. ER/IR >0.66 is critical for swimmer shoulder.',
                'source': 'forceframe',
                'test_pattern': 'Shoulder',
                'optimal': (0.66, 0.80),
                'concern_low': 0.60,
                'risk_low': 0.50,
                'citation': 'Ellenbecker & Davies, 2000; Beach et al., 1992',
                'note': 'Expressed as ER/IR (inverse of IR:ER). Below 0.60 = high injury risk.',
            },
        ],

        # ---- Injury risk indicators ----
        'injury_risk_indicators': [
            {
                'name': 'Swimmer Shoulder Risk (ER:IR too low)',
                'test': 'ForceFrame',
                'pattern': 'Shoulder',
                'metric': 'er_ir_ratio',
                'threshold': 0.60,
                'direction': 'below',
                'severity': 'high',
                'source': 'Ellenbecker & Davies, 2000; Beach et al., 1992',
            },
            {
                'name': 'Bilateral Asymmetry (Stroke Efficiency)',
                'test': 'ForceDecks',
                'test_types': ['CMJ'],
                'metric': 'concentric_impulse_asym',
                'threshold': 5.0,
                'direction': 'above',
                'severity': 'moderate',
                'source': 'West et al., 2011',
            },
            {
                'name': 'Shoulder Isometric Imbalance',
                'test': 'ForceDecks',
                'test_types': ['SHLDISOI', 'SHLDISOY', 'SHLDISOT'],
                'metric': 'peak_force_asymmetry',
                'threshold': 10.0,
                'direction': 'above',
                'severity': 'moderate',
                'source': 'Shoulder isometric L/R imbalance assessment',
            },
        ],

        'benchmarks_key': 'Swimming',
        'context_notes': [
            'Bilateral symmetry is the STRICTEST of any sport - flag >5% asymmetry (West et al., 2011).',
            'Shoulder ER:IR ratio >0.66 is the CRITICAL injury prevention threshold for swimmers.',
            'CMJ correlates with start/turn performance at r=0.56-0.71 (West et al., 2011).',
            'SHLD isometric tests available (SHLDISOI n=6, SHLDISOY n=5, SHLDISOT n=5) - unique to swimming.',
            'PPU (n=4) reflects upper body power for starts and pull phase of stroke.',
            'SL ISO Squat (SLISOSQT, n=3) assesses single-leg push for wall turns.',
            'Lower jump heights expected vs land-based sports - context is critical for interpretation.',
            'Hop RSI (n=9) reflects elastic recoil capacity for wall push-off during turns.',
        ],
    },

    # ========================================================================
    # ATHLETICS (61 athletes - multiple disciplines)
    # Test types available: CMJ (50+), HJ (19+), IMTP (16+), SLISOT (14+),
    #   SLJ (10+), ISOT (10+), PPU varies by discipline
    # ========================================================================
    'athletics': {
        'display_name': 'Athletics',
        'icon': '\U0001F3C3',
        'sub_sports': [
            'Athletics', 'Athletics - Jumps', 'Athletics - Middle Distance',
            'Athletics - Multi Events', 'Athletics - Sprints', 'Athletics - Throws',
        ],

        'asymmetry_metrics': [
            {
                'name': 'IMTP Peak Force L/R',
                'test_types': ['IMTP'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Peak Concentric Force L/R',
                'test_types': ['CMJ'],
                'left_col': 'PEAK_CONCENTRIC_FORCE_Left',
                'right_col': 'PEAK_CONCENTRIC_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Concentric Impulse L/R',
                'test_types': ['CMJ'],
                'left_col': 'CONCENTRIC_IMPULSE_Left',
                'right_col': 'CONCENTRIC_IMPULSE_Right',
                'unit': 'Ns',
            },
            {
                'name': 'SL CMJ Peak Force',
                'test_types': ['SLCMJ'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'Concentric RFD L/R',
                'test_types': ['CMJ'],
                'left_col': 'CONCENTRIC_RFD_Left',
                'right_col': 'CONCENTRIC_RFD_Right',
                'unit': 'N/s',
            },
            {
                'name': 'Eccentric Braking RFD L/R',
                'test_types': ['CMJ'],
                'left_col': 'ECCENTRIC_BRAKING_RFD_Left',
                'right_col': 'ECCENTRIC_BRAKING_RFD_Right',
                'unit': 'N/s',
            },
            {
                'name': 'Force at Zero Velocity L/R',
                'test_types': ['CMJ'],
                'left_col': 'FORCE_AT_ZERO_VELOCITY_Left',
                'right_col': 'FORCE_AT_ZERO_VELOCITY_Right',
                'unit': 'N',
            },
            {
                'name': 'Landing RFD L/R',
                'test_types': ['CMJ'],
                'left_col': 'LANDING_RFD_Left',
                'right_col': 'LANDING_RFD_Right',
                'unit': 'N/s',
            },
        ],
        'asymmetry_thresholds': {'caution': 8.0, 'risk': 12.0},
        'asymmetry_note': (
            'Low asymmetry (<8%) is critical for sprinters and jumpers. '
            'Throwers may show 10-15% expected asymmetry in throwing arm/leg. '
            'Concentric RFD asymmetry is the strongest predictor of sprint injury risk.'
        ),

        'forceframe_sections': [
            {
                'name': 'Shoulder IR/ER',
                'test_pattern': r'Shoulder.*Internal.*External|Shoulder.*Rotation',
                'inner_label': 'Internal Rotation',
                'outer_label': 'External Rotation',
            },
        ],

        'dynamo_section': None,

        'nordbord_section': {
            'enabled': True,
            'title': 'Nordic Hamstring Strength - Critical for Sprinters',
        },

        'key_metrics': [
            {
                'label': 'CMJ Height',
                'test_types': ['CMJ'],
                'col': 'JUMP_HEIGHT_IMP_MOM',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'IMTP Rel. Peak Force',
                'test_types': ['IMTP'],
                'col': 'ISO_BM_REL_FORCE_PEAK',
                'unit': 'N/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'CMJ Rel. Power',
                'test_types': ['CMJ'],
                'col': 'BODYMASS_RELATIVE_TAKEOFF_POWER',
                'unit': 'W/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'RSI (Hop)',
                'test_types': ['HJ', 'SLHJ', 'RSHIP', 'RSKIP', 'RSAIP'],
                'col': 'HOP_BEST_RSI',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
            {
                'label': 'Concentric RFD',
                'test_types': ['CMJ'],
                'col': 'CONCENTRIC_RFD',
                'unit': 'N/s',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'Eccentric Decel RFD',
                'test_types': ['CMJ'],
                'col': 'ECCENTRIC_DECEL_RFD',
                'unit': 'N/s',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'RSI Modified',
                'test_types': ['CMJ'],
                'col': 'RSI_MODIFIED',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
            {
                'label': 'Flight:Contact Ratio',
                'test_types': ['CMJ'],
                'col': 'FLIGHT_CONTRACTION_TIME_RATIO',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
        ],

        'extra_sections': [
            {
                'type': 'hop_test',
                'title': '10:5 Hop / Reactive Strength',
                'test_types': ['HJ', 'SLHJ', 'RSHIP', 'RSKIP', 'RSAIP'],
                'metric_key': 'rsi',
            },
            {
                'type': 'ppu',
                'title': 'Plyo Pushup (Upper Body Power - Throwers)',
                'test_types': ['PPU'],
                'metric_key': 'peak_force',
            },
        ],

        # ---- Benchmarks ----
        'benchmarks': {
            'cmj_height_sprinters': {
                'elite_m': (50, 62), 'national_m': (42, 50),
                'elite_f': (35, 42), 'national_f': (30, 35),
                'unit': 'cm',
                'source': 'Loturco et al., 2015; Comfort et al., 2014',
            },
            'cmj_height_jumpers': {
                'elite_m': (55, 65), 'national_m': (45, 55),
                'elite_f': (38, 46), 'national_f': (32, 38),
                'unit': 'cm',
                'source': 'Cormie et al., 2011',
            },
            'cmj_height_throwers': {
                'elite_m': (42, 52), 'national_m': (36, 42),
                'elite_f': (30, 38), 'national_f': (26, 30),
                'unit': 'cm',
                'source': 'Zaras et al., 2013',
            },
            'cmj_height_mid_distance': {
                'elite_m': (38, 46), 'national_m': (32, 38),
                'elite_f': (28, 35), 'national_f': (24, 28),
                'unit': 'cm',
                'source': 'Estimated from athletics norms',
            },
            'imtp_relative': {
                'elite_m': (38, 48), 'national_m': (32, 38),
                'elite_f': (30, 38), 'national_f': (26, 30),
                'unit': 'N/kg',
                'source': 'Comfort et al., 2014; Suchomel et al., 2016',
            },
            'rsi_sprinters': {
                'elite_m': (2.50, None), 'national_m': (2.0, 2.50),
                'elite_f': (2.0, None), 'national_f': (1.5, 2.0),
                'unit': '',
                'source': 'Comfort et al., 2014',
            },
            'nordbord_per_leg_sprinters': {
                'elite_m': (450, None), 'national_m': (337, 450),
                'elite_f': (300, None), 'national_f': (256, 300),
                'unit': 'N',
                'source': 'Opar et al., 2015; higher for sprinters',
            },
            'ppu_throwers': {
                'elite_m': (20, 28), 'national_m': (15, 20),
                'elite_f': (12, 18), 'national_f': (8, 12),
                'unit': 'cm',
                'source': 'Estimated from athletics norms',
            },
        },

        # ---- Performance ratios ----
        'performance_ratios': [
            {
                'name': 'Eccentric Utilization Ratio (Sprinters/Jumpers)',
                'description': 'CMJ:SJ height. Sprinters typically 1.10-1.25.',
                'formula': 'CMJ Height / SJ Height',
                'optimal': (1.10, 1.25),
                'concern_low': 1.05,
                'concern_high': 1.30,
                'test_types_num': ['CMJ'],
                'test_types_den': ['SJ'],
                'col_num': 'JUMP_HEIGHT_IMP_MOM',
                'col_den': 'JUMP_HEIGHT_IMP_MOM',
                'note': 'SJ data limited - collect for sprinters/jumpers when possible.',
            },
            {
                'name': 'Shoulder IR:ER Ratio (Throwers)',
                'description': 'Throwing arm IR:ER. Critical for javelin and shot put shoulder health.',
                'source': 'forceframe',
                'test_pattern': 'Shoulder',
                'optimal': (1.50, 2.00),
                'concern_high': 2.00,
                'risk_high': 2.50,
                'citation': 'Byram et al., 2010',
            },
        ],

        # ---- Injury risk indicators ----
        'injury_risk_indicators': [
            {
                'name': 'Hamstring Injury Risk (Sprinters)',
                'test': 'NordBord',
                'metric': 'absolute_force_per_leg',
                'threshold_male': 450,
                'threshold_female': 300,
                'direction': 'below',
                'severity': 'high',
                'source': 'Opar et al., 2015; elevated threshold for high-speed running',
            },
            {
                'name': 'NordBord L/R Asymmetry',
                'test': 'NordBord',
                'metric': 'bilateral_asymmetry',
                'threshold': 10.0,
                'direction': 'above',
                'severity': 'high',
                'source': 'Croisier et al., 2008',
            },
            {
                'name': 'Concentric RFD Asymmetry',
                'test': 'ForceDecks',
                'test_types': ['CMJ'],
                'metric': 'concentric_rfd_asym',
                'threshold': 10.0,
                'direction': 'above',
                'severity': 'moderate',
                'source': 'Bishop et al., 2018',
            },
            {
                'name': 'Shoulder Injury Risk (Throwers)',
                'test': 'ForceFrame',
                'pattern': 'Shoulder',
                'metric': 'ir_er_ratio',
                'threshold': 2.50,
                'direction': 'above',
                'severity': 'high',
                'source': 'Byram et al., 2010',
            },
        ],

        'benchmarks_key': 'Athletics',
        'context_notes': [
            'Sprinters: CMJ 50-62cm elite M. RSI >2.50. NordBord >450N per leg. RFD 0-50ms >12000 N/s.',
            'Jumpers: Highest CMJ (55-65cm) and RSI requirements. Reactive strength is #1 priority.',
            'Throwers: PPU >20cm elite M. Shoulder IR:ER critical. Rotational power > vertical jump.',
            'Middle Distance: Lower power needs but CMJ monitors neuromuscular fatigue during training blocks.',
            'Nordic hamstring strength is THE critical injury prevention metric for sprinters.',
            'Concentric RFD asymmetry >10% is a stronger sprint injury predictor than peak force asymmetry.',
            'EUR (CMJ:SJ) of 1.10-1.25 typical for sprinters - indicates effective elastic energy use.',
            'SL ISOT (n=14+) and SLJ (n=10+) provide unilateral assessment for takeoff leg strength.',
        ],
    },

    # ========================================================================
    # ROWING (17 athletes)
    # Test types available: CMJ (42), IMTP (27), HJ (24), SLISOT (11),
    #   SLSTICR (9), ISOT (8)
    # ========================================================================
    'rowing': {
        'display_name': 'Rowing',
        'icon': '\U0001F6A3',
        'sub_sports': ['Rowing', 'Rowing - Classic', 'Rowing - Coastal', 'Coastal'],

        'asymmetry_metrics': [
            {
                'name': 'IMTP Peak Force L/R',
                'test_types': ['IMTP'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Peak Concentric Force L/R',
                'test_types': ['CMJ'],
                'left_col': 'PEAK_CONCENTRIC_FORCE_Left',
                'right_col': 'PEAK_CONCENTRIC_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Concentric Impulse L/R',
                'test_types': ['CMJ'],
                'left_col': 'CONCENTRIC_IMPULSE_Left',
                'right_col': 'CONCENTRIC_IMPULSE_Right',
                'unit': 'Ns',
            },
            {
                'name': 'Eccentric Braking RFD L/R',
                'test_types': ['CMJ'],
                'left_col': 'ECCENTRIC_BRAKING_RFD_Left',
                'right_col': 'ECCENTRIC_BRAKING_RFD_Right',
                'unit': 'N/s',
            },
            {
                'name': 'Force at Zero Velocity L/R',
                'test_types': ['CMJ'],
                'left_col': 'FORCE_AT_ZERO_VELOCITY_Left',
                'right_col': 'FORCE_AT_ZERO_VELOCITY_Right',
                'unit': 'N',
            },
        ],
        'asymmetry_thresholds': {'caution': 5.0, 'risk': 10.0},
        'asymmetry_note': (
            'Bilateral symmetry is critical for boat balance and stroke efficiency. '
            '<5% asymmetry target for crew boat rowers (sweep rowers may show some asymmetry). '
            'Asymmetry >10% directly affects boat tracking and speed.'
        ),

        'forceframe_sections': [],

        'dynamo_section': None,

        'nordbord_section': None,

        'key_metrics': [
            {
                'label': 'CMJ Height',
                'test_types': ['CMJ'],
                'col': 'JUMP_HEIGHT_IMP_MOM',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'IMTP Abs Peak Force',
                'test_types': ['IMTP'],
                'col': 'PEAK_VERTICAL_FORCE',
                'unit': 'N',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'IMTP Rel. Peak Force',
                'test_types': ['IMTP'],
                'col': 'ISO_BM_REL_FORCE_PEAK',
                'unit': 'N/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'IMTP Net Peak Force',
                'test_types': ['IMTP'],
                'col': 'NET_PEAK_VERTICAL_FORCE',
                'unit': 'N',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'RSI (Hop)',
                'test_types': ['HJ', 'SLHJ'],
                'col': 'HOP_BEST_RSI',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
            {
                'label': 'Concentric Duration',
                'test_types': ['CMJ'],
                'col': 'CONCENTRIC_DURATION',
                'unit': 's',
                'format': '.3f',
                'higher_is_better': False,
            },
            {
                'label': 'Ecc:Con Peak Power Ratio',
                'test_types': ['CMJ'],
                'col': 'ECC_CON_PEAK_POWER_RATIO',
                'unit': '',
                'format': '.2f',
                'higher_is_better': False,
            },
        ],

        'extra_sections': [
            {
                'type': 'hop_test',
                'title': '10:5 Hop / Reactive Strength',
                'test_types': ['HJ', 'SLHJ', 'RSHIP', 'RSKIP', 'RSAIP'],
                'metric_key': 'rsi',
            },
        ],

        # ---- Benchmarks ----
        'benchmarks': {
            'imtp_absolute': {
                'elite_m': (4000, None), 'national_m': (3200, 4000),
                'elite_f': (2800, None), 'national_f': (2200, 2800),
                'unit': 'N',
                'source': 'Lawton et al., 2011; McGregor et al., 2014',
            },
            'imtp_relative': {
                'elite_m': (38, 48), 'national_m': (32, 38),
                'elite_f': (30, 38), 'national_f': (26, 30),
                'unit': 'N/kg',
                'source': 'Lawton et al., 2011',
            },
            'cmj_height': {
                'elite_m': (40, 50), 'national_m': (35, 40),
                'elite_f': (28, 36), 'national_f': (24, 28),
                'unit': 'cm',
                'source': 'Lawton et al., 2011',
            },
        },

        # ---- Performance ratios ----
        'performance_ratios': [
            {
                'name': 'Trunk Ext:Flex Ratio',
                'description': 'Trunk extension:flexion. Rowers need 1.0-1.3 for low back protection.',
                'source': 'forceframe',
                'test_pattern': 'Trunk',
                'optimal': (1.0, 1.3),
                'concern_low': 0.8,
                'concern_high': 1.5,
                'citation': 'Wilson et al., 2010',
            },
        ],

        # ---- Injury risk indicators ----
        'injury_risk_indicators': [
            {
                'name': 'Low Back Injury Risk (Trunk Imbalance)',
                'test': 'ForceFrame',
                'pattern': 'Trunk',
                'metric': 'ext_flex_ratio',
                'threshold': 0.8,
                'direction': 'below',
                'severity': 'high',
                'source': 'Wilson et al., 2010; low back is #1 rowing injury',
            },
            {
                'name': 'Bilateral Leg Asymmetry',
                'test': 'ForceDecks',
                'test_types': ['CMJ'],
                'metric': 'concentric_impulse_asym',
                'threshold': 8.0,
                'direction': 'above',
                'severity': 'moderate',
                'source': 'Crew boat balance requirement',
            },
        ],

        'benchmarks_key': 'Rowing',
        'context_notes': [
            'IMTP absolute force >4000N for elite heavyweight males - strongest absolute force requirement.',
            'Bilateral symmetry <5% target for crew boat rowers - affects boat tracking and speed.',
            'Trunk extension:flexion ratio of 1.0-1.3 is protective for low back (#1 rowing injury).',
            'CMJ is primarily a neuromuscular fatigue monitoring tool, not a direct performance predictor.',
            'Absolute force matters more than relative force for heavyweight rowers.',
            'SLISOT (n=11) and ISOT (n=8) provide isometric endurance data relevant to catch position.',
            'Hop tests (n=24) available for reactive strength monitoring.',
            'Force endurance matters more than peak power for 2000m race distance (6-8 min effort).',
        ],
    },

    # ========================================================================
    # PARA ATHLETICS (45 athletes - classification-dependent)
    # Test types available: CMJ (14), HJ (1), IMTP (1)
    # ========================================================================
    'para_athletics': {
        'display_name': 'Para Athletics',
        'icon': '\u267F',
        'sub_sports': [
            'Para Athletics', 'Para Swimming', 'Para Taekwondo',
            'Para Cycling', 'Wheelchair Sports',
        ],

        'asymmetry_metrics': [
            {
                'name': 'IMTP Peak Force L/R',
                'test_types': ['IMTP'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Peak Concentric Force L/R',
                'test_types': ['CMJ'],
                'left_col': 'PEAK_CONCENTRIC_FORCE_Left',
                'right_col': 'PEAK_CONCENTRIC_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Concentric Impulse L/R',
                'test_types': ['CMJ'],
                'left_col': 'CONCENTRIC_IMPULSE_Left',
                'right_col': 'CONCENTRIC_IMPULSE_Right',
                'unit': 'Ns',
            },
        ],
        'asymmetry_thresholds': {'caution': None, 'risk': None},
        'asymmetry_note': (
            'Asymmetry may be EXPECTED and FUNCTIONAL depending on classification. '
            'Do NOT apply standard asymmetry thresholds. '
            'Track individual % improvement over time using SWC (0.6 x SD) as threshold. '
            'Z-scores relative to individual baseline are more meaningful than absolute values.'
        ),

        'forceframe_sections': [
            {
                'name': 'Shoulder IR/ER',
                'test_pattern': r'Shoulder.*Internal.*External|Shoulder.*Rotation',
                'inner_label': 'Internal Rotation',
                'outer_label': 'External Rotation',
            },
        ],

        'dynamo_section': None,

        'nordbord_section': None,

        'key_metrics': [
            {
                'label': 'CMJ Height',
                'test_types': ['CMJ'],
                'col': 'JUMP_HEIGHT_IMP_MOM',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'CMJ Rel. Power',
                'test_types': ['CMJ'],
                'col': 'BODYMASS_RELATIVE_TAKEOFF_POWER',
                'unit': 'W/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'IMTP Peak Force',
                'test_types': ['IMTP'],
                'col': 'PEAK_VERTICAL_FORCE',
                'unit': 'N',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'RSI Modified',
                'test_types': ['CMJ'],
                'col': 'RSI_MODIFIED',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
            {
                'label': 'PPU Height',
                'test_types': ['PPU'],
                'col': 'PUSHUP_HEIGHT',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
        ],

        'extra_sections': [
            {
                'type': 'ppu',
                'title': 'Plyo Pushup (Upper Body Power - Wheelchair Athletes)',
                'test_types': ['PPU'],
                'metric_key': 'peak_force',
            },
        ],

        # ---- Benchmarks ----
        # NO population benchmarks - use individual tracking methods
        'benchmarks': {},

        # ---- Performance ratios ----
        'performance_ratios': [
            {
                'name': 'Shoulder IR:ER Ratio (Wheelchair Athletes)',
                'description': 'Critical for wheelchair propulsion injury prevention. Applicable to wheelchair classes.',
                'source': 'forceframe',
                'test_pattern': 'Shoulder',
                'optimal': (1.33, 1.67),
                'concern_high': 2.0,
                'risk_high': 2.5,
                'citation': 'Burnham et al., 1993; wheelchair-specific shoulder health',
            },
        ],

        # ---- Injury risk indicators ----
        'injury_risk_indicators': [
            {
                'name': 'Shoulder Overuse Risk (Wheelchair Athletes)',
                'test': 'ForceFrame',
                'pattern': 'Shoulder',
                'metric': 'ir_er_ratio',
                'threshold': 2.0,
                'direction': 'above',
                'severity': 'high',
                'source': 'Burnham et al., 1993; wheelchair propulsion shoulder demands',
            },
            {
                'name': 'Individual SWC Regression',
                'test': 'ForceDecks',
                'test_types': ['CMJ'],
                'metric': 'individual_swc',
                'threshold': None,
                'direction': 'below',
                'severity': 'moderate',
                'source': 'Use SWC (0.6 x individual SD) to detect meaningful change',
                'note': 'Classification-dependent - no universal threshold',
            },
        ],

        'benchmarks_key': None,
        'context_notes': [
            'NO population benchmarks exist - track individual % improvement over time.',
            'Use SWC (Smallest Worthwhile Change = 0.6 x individual SD) for meaningful change detection.',
            'Z-scores relative to individual baseline are more meaningful than absolute comparisons.',
            'Classification-dependent: what is "normal" varies enormously by class.',
            'Asymmetry may be expected and functional depending on impairment type and classification.',
            'Wheelchair athletes: PPU and grip strength are primary performance drivers.',
            'Shoulder IR:ER ratio is critical for wheelchair propulsion injury prevention.',
            'CMJ data (n=14) available for ambulant Para athletes - use for individual tracking only.',
        ],
    },

    # ========================================================================
    # WEIGHTLIFTING (34 athletes, 178 ForceDecks tests)
    # ========================================================================
    'weightlifting': {
        'display_name': 'Weightlifting',
        'icon': '\U0001F3CB\uFE0F',
        'sub_sports': ['Weightlifting'],

        'asymmetry_metrics': [
            {
                'name': 'IMTP Peak Force L/R',
                'test_types': ['IMTP'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Peak Concentric Force L/R',
                'test_types': ['CMJ'],
                'left_col': 'PEAK_CONCENTRIC_FORCE_Left',
                'right_col': 'PEAK_CONCENTRIC_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Concentric Impulse L/R',
                'test_types': ['CMJ'],
                'left_col': 'CONCENTRIC_IMPULSE_Left',
                'right_col': 'CONCENTRIC_IMPULSE_Right',
                'unit': 'Ns',
            },
            {
                'name': 'Concentric RFD L/R',
                'test_types': ['CMJ'],
                'left_col': 'CONCENTRIC_RFD_Left',
                'right_col': 'CONCENTRIC_RFD_Right',
                'unit': 'N/s',
            },
            {
                'name': 'Force at Zero Velocity L/R',
                'test_types': ['CMJ'],
                'left_col': 'FORCE_AT_ZERO_VELOCITY_Left',
                'right_col': 'FORCE_AT_ZERO_VELOCITY_Right',
                'unit': 'N',
            },
            {
                'name': 'Eccentric Decel RFD L/R',
                'test_types': ['CMJ'],
                'left_col': 'ECCENTRIC_DECEL_RFD_Left',
                'right_col': 'ECCENTRIC_DECEL_RFD_Right',
                'unit': 'N/s',
            },
        ],
        'asymmetry_thresholds': {'caution': 6.0, 'risk': 10.0},
        'asymmetry_note': (
            'STRICTEST asymmetry threshold after swimming. Perfect bilateral symmetry is '
            'required for clean & jerk and snatch technique. Flag any asymmetry >6%. '
            'Force at zero velocity asymmetry is particularly diagnostic of catch imbalance.'
        ),

        'forceframe_sections': [],

        'dynamo_section': {
            'enabled': True,
            'movement_filter': 'GripSqueeze',
            'title': 'Grip Strength (DynaMo)',
        },

        'nordbord_section': None,

        'key_metrics': [
            {
                'label': 'CMJ Height',
                'test_types': ['CMJ'],
                'col': 'JUMP_HEIGHT_IMP_MOM',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'IMTP Abs Peak Force',
                'test_types': ['IMTP'],
                'col': 'PEAK_VERTICAL_FORCE',
                'unit': 'N',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'IMTP Rel. Peak Force',
                'test_types': ['IMTP'],
                'col': 'ISO_BM_REL_FORCE_PEAK',
                'unit': 'N/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'CMJ Rel. Power',
                'test_types': ['CMJ'],
                'col': 'BODYMASS_RELATIVE_TAKEOFF_POWER',
                'unit': 'W/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'Concentric RFD',
                'test_types': ['CMJ'],
                'col': 'CONCENTRIC_RFD',
                'unit': 'N/s',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'RSI Modified',
                'test_types': ['CMJ'],
                'col': 'RSI_MODIFIED',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
            {
                'label': 'Countermovement Depth',
                'test_types': ['CMJ'],
                'col': 'COUNTERMOVEMENT_DEPTH',
                'unit': 'm',
                'format': '.3f',
                'higher_is_better': False,
            },
        ],

        'extra_sections': [
            {
                'type': 'strength_rm',
                'title': 'Strength RM (Manual Entry)',
                'test_types': None,
                'metric_key': 'peak_force',
            },
        ],

        # ---- Benchmarks ----
        'benchmarks': {
            'cmj_height': {
                'elite_m': (45, 58), 'national_m': (38, 45),
                'elite_f': (32, 40), 'national_f': (27, 32),
                'unit': 'cm',
                'source': 'Carlock et al., 2004; Vizcaya et al., 2009',
            },
            'imtp_relative': {
                'elite_m': (34, 45), 'national_m': (28, 34),
                'elite_f': (28, 36), 'national_f': (24, 28),
                'unit': 'N/kg',
                'source': 'Beckham et al., 2013; Haff et al., 2005',
            },
            'cmj_relative_power': {
                'elite_m': (55, 70), 'national_m': (45, 55),
                'elite_f': (42, 55), 'national_f': (35, 42),
                'unit': 'W/kg',
                'source': 'Carlock et al., 2004',
            },
        },

        # ---- Performance ratios ----
        'performance_ratios': [
            {
                'name': 'IMTP:CMJ Relationship',
                'description': 'IMTP peak force correlates r=0.93 with snatch and r=0.91 with clean & jerk total.',
                'formula': 'IMTP Peak Force / Body Mass',
                'optimal': (34, 45),
                'concern_low': 28,
                'test_types_num': ['IMTP'],
                'col_num': 'ISO_BM_REL_FORCE_PEAK',
                'citation': 'Beckham et al., 2013',
            },
        ],

        # ---- Injury risk indicators ----
        'injury_risk_indicators': [
            {
                'name': 'Bilateral Imbalance (Technique Risk)',
                'test': 'ForceDecks',
                'test_types': ['CMJ'],
                'metric': 'concentric_impulse_asym',
                'threshold': 6.0,
                'direction': 'above',
                'severity': 'moderate',
                'source': 'Bilateral symmetry critical for overhead lifts',
            },
            {
                'name': 'Force at Zero Velocity Asymmetry',
                'test': 'ForceDecks',
                'test_types': ['CMJ'],
                'metric': 'force_at_zero_velocity_asym',
                'threshold': 8.0,
                'direction': 'above',
                'severity': 'high',
                'source': 'Reflects catch-phase imbalance in snatch/clean',
            },
        ],

        'benchmarks_key': 'Weightlifting',
        'context_notes': [
            'Weightlifters typically produce the highest CMJ and relative power values of all sports.',
            'CMJ peak power correlates r=0.90 with competition total (Carlock et al., 2004).',
            'IMTP peak force correlates r=0.93 with snatch and r=0.91 with C&J (Beckham et al., 2013).',
            'Perfect bilateral symmetry required - 6% threshold is strictest after swimming.',
            'Force at zero velocity asymmetry is diagnostic of catch-phase imbalance.',
            'Countermovement depth reflects squat depth strategy - monitors movement quality.',
            'CMJ concentric RFD correlates with second pull velocity in snatch.',
            'DynaMo grip strength monitors hook grip capacity under fatigue.',
        ],
    },

    # ========================================================================
    # SHOOTING (22 athletes)
    # Test types: QSB, SLSB (balance), some CMJ/IMTP
    # ========================================================================
    'shooting': {
        'display_name': 'Shooting',
        'icon': '\U0001F3AF',
        'sub_sports': ['Shooting'],

        'asymmetry_metrics': [
            {
                'name': 'IMTP Peak Force L/R',
                'test_types': ['IMTP'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'Balance CoP Total Excursion L/R',
                'test_types': ['SLSB'],
                'left_col': 'BAL_COP_TOTAL_EXCURSION_Left',
                'right_col': 'BAL_COP_TOTAL_EXCURSION_Right',
                'unit': 'mm',
            },
            {
                'name': 'Balance CoP Mean Velocity L/R',
                'test_types': ['SLSB'],
                'left_col': 'BAL_COP_MEAN_VELOCITY_Left',
                'right_col': 'BAL_COP_MEAN_VELOCITY_Right',
                'unit': 'mm/s',
            },
        ],
        'asymmetry_thresholds': {'caution': 10.0, 'risk': 15.0},
        'asymmetry_note': (
            'Balance L/R asymmetry on SLSB reflects stance leg stability. '
            'IMTP asymmetry thresholds are less critical for shooting than balance metrics.'
        ),

        'forceframe_sections': [],

        'dynamo_section': None,

        'nordbord_section': None,

        'key_metrics': [
            {
                'label': 'IMTP Rel. Peak Force',
                'test_types': ['IMTP'],
                'col': 'ISO_BM_REL_FORCE_PEAK',
                'unit': 'N/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'CMJ Height',
                'test_types': ['CMJ'],
                'col': 'JUMP_HEIGHT_IMP_MOM',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'QSB CoP Total Excursion',
                'test_types': ['QSB'],
                'col': 'BAL_COP_TOTAL_EXCURSION',
                'unit': 'mm',
                'format': '.1f',
                'higher_is_better': False,
            },
            {
                'label': 'QSB CoP Mean Velocity',
                'test_types': ['QSB'],
                'col': 'BAL_COP_MEAN_VELOCITY',
                'unit': 'mm/s',
                'format': '.2f',
                'higher_is_better': False,
            },
            {
                'label': 'QSB CoP Ellipse Area',
                'test_types': ['QSB'],
                'col': 'BAL_COP_ELLIPSE_AREA',
                'unit': 'mm2',
                'format': '.1f',
                'higher_is_better': False,
            },
            {
                'label': 'QSB ML Range',
                'test_types': ['QSB'],
                'col': 'BAL_COP_RANGE_MEDLAT',
                'unit': 'mm',
                'format': '.1f',
                'higher_is_better': False,
            },
            {
                'label': 'QSB AP Range',
                'test_types': ['QSB'],
                'col': 'BAL_COP_RANGE_ANTPOST',
                'unit': 'mm',
                'format': '.1f',
                'higher_is_better': False,
            },
        ],

        'extra_sections': [
            {
                'type': 'balance',
                'title': 'Quiet Stance Balance (QSB)',
                'test_types': ['QSB', 'SLSB'],
                'metric_key': 'balance',
            },
        ],

        # ---- Benchmarks ----
        'benchmarks': {
            'qsb_cop_excursion': {
                'elite_m': (None, 400), 'national_m': (400, 550),
                'elite_f': (None, 380), 'national_f': (380, 520),
                'unit': 'mm (lower is better)',
                'source': 'Shooting-specific norms; Mon-Lopez et al., 2019',
            },
            'qsb_cop_velocity': {
                'elite_m': (None, 12), 'national_m': (12, 18),
                'elite_f': (None, 11), 'national_f': (11, 16),
                'unit': 'mm/s (lower is better)',
                'source': 'Mon-Lopez et al., 2019',
            },
            'qsb_ellipse_area': {
                'elite_m': (None, 150), 'national_m': (150, 250),
                'elite_f': (None, 130), 'national_f': (130, 220),
                'unit': 'mm2 (lower is better)',
                'source': 'Mon-Lopez et al., 2019',
            },
        },

        # ---- Performance ratios ----
        'performance_ratios': [],

        # ---- Injury risk indicators ----
        'injury_risk_indicators': [
            {
                'name': 'Postural Control Decline',
                'test': 'ForceDecks',
                'test_types': ['QSB'],
                'metric': 'cop_velocity_increase',
                'threshold': None,
                'direction': 'above',
                'severity': 'moderate',
                'source': 'Track individual SWC - CoP velocity increase indicates fatigue or detraining',
            },
        ],

        'benchmarks_key': 'Shooting',
        'context_notes': [
            'Postural stability and balance are THE primary performance drivers for 10m pistol.',
            'Lower power requirements than any other sport - isometric strength and core stability are key.',
            'Balance metrics (QSB/SLSB): LOWER CoP excursion and velocity = BETTER stability.',
            'VALD balance data stored in meters, display in mm (multiply by 1000).',
            'CoP ellipse area 95% is the gold standard for stance quality assessment.',
            'ML (medio-lateral) CoP range correlates more strongly with pistol scores than AP range.',
            'Track balance trends over time - improvement is typically slow (months, not weeks).',
            'Isometric strength and trunk endurance underpin stance holding capacity.',
        ],
    },

    # ========================================================================
    # EQUESTRIAN (7 athletes)
    # Test types available: CMJ (1), HJ (1) - minimal VALD data
    # ========================================================================
    'equestrian': {
        'display_name': 'Equestrian',
        'icon': '\U0001F3C7',
        'sub_sports': ['Equestrian'],

        'asymmetry_metrics': [
            {
                'name': 'IMTP Peak Force L/R',
                'test_types': ['IMTP'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Peak Concentric Force L/R',
                'test_types': ['CMJ'],
                'left_col': 'PEAK_CONCENTRIC_FORCE_Left',
                'right_col': 'PEAK_CONCENTRIC_FORCE_Right',
                'unit': 'N',
            },
        ],
        'asymmetry_thresholds': {'caution': 5.0, 'risk': 10.0},
        'asymmetry_note': (
            'STRICTEST asymmetry threshold alongside swimming and weightlifting. '
            '<5% asymmetry required for balanced rider position. '
            'Asymmetric leg strength causes unintentional cues to the horse.'
        ),

        'forceframe_sections': [],

        'dynamo_section': None,

        'nordbord_section': None,

        'key_metrics': [
            {
                'label': 'IMTP Rel. Peak Force',
                'test_types': ['IMTP'],
                'col': 'ISO_BM_REL_FORCE_PEAK',
                'unit': 'N/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'CMJ Height',
                'test_types': ['CMJ'],
                'col': 'JUMP_HEIGHT_IMP_MOM',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'CMJ Rel. Power',
                'test_types': ['CMJ'],
                'col': 'BODYMASS_RELATIVE_TAKEOFF_POWER',
                'unit': 'W/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
        ],

        'extra_sections': [
            {
                'type': 'balance',
                'title': 'Quiet Stance Balance (Rider Stability)',
                'test_types': ['QSB', 'SLSB'],
                'metric_key': 'balance',
            },
        ],

        # ---- Benchmarks ----
        'benchmarks': {},  # Limited equestrian-specific research benchmarks

        # ---- Performance ratios ----
        'performance_ratios': [],

        # ---- Injury risk indicators ----
        'injury_risk_indicators': [
            {
                'name': 'Rider Bilateral Imbalance',
                'test': 'ForceDecks',
                'test_types': ['CMJ'],
                'metric': 'concentric_impulse_asym',
                'threshold': 5.0,
                'direction': 'above',
                'severity': 'moderate',
                'source': 'Equestrian-specific - asymmetry causes unintentional horse cues',
            },
        ],

        'benchmarks_key': None,
        'context_notes': [
            'Core stability and isometric endurance are THE primary physical demands.',
            '<5% bilateral asymmetry is among the strictest in any sport - affects horse communication.',
            'Hip adduction strength is critical for maintaining seat position and leg contact.',
            'Trunk endurance (flexion, extension, lateral) underpins sustained riding posture.',
            'Balance (QSB) is directly relevant to rider stability and effectiveness.',
            'Lower overall power demands compared to other sports - focus on endurance and stability.',
            'Very limited VALD data (CMJ n=1, HJ n=1) - track individual trends over time.',
        ],
    },

    # ========================================================================
    # SNOW SPORTS (4 athletes)
    # Test types available: SLISOSQT (3), HJ (3), SLJ (2), CMJ (2),
    #   IMTP (2), ISOT (2)
    # ========================================================================
    'snow_sports': {
        'display_name': 'Snow Sports',
        'icon': '\u26F7\uFE0F',
        'sub_sports': ['Snow Sports'],

        'asymmetry_metrics': [
            {
                'name': 'IMTP Peak Force L/R',
                'test_types': ['IMTP'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'CMJ Peak Concentric Force L/R',
                'test_types': ['CMJ'],
                'left_col': 'PEAK_CONCENTRIC_FORCE_Left',
                'right_col': 'PEAK_CONCENTRIC_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'SL ISO Squat Peak Force L/R',
                'test_types': ['SLISOSQT'],
                'left_col': 'PEAK_VERTICAL_FORCE_Left',
                'right_col': 'PEAK_VERTICAL_FORCE_Right',
                'unit': 'N',
            },
            {
                'name': 'Eccentric Decel RFD L/R',
                'test_types': ['CMJ'],
                'left_col': 'ECCENTRIC_DECEL_RFD_Left',
                'right_col': 'ECCENTRIC_DECEL_RFD_Right',
                'unit': 'N/s',
            },
        ],
        'asymmetry_thresholds': {'caution': 8.0, 'risk': 12.0},
        'asymmetry_note': (
            'Single-leg strength asymmetry >10% is the #1 modifiable ACL risk factor. '
            'SL ISO Squat asymmetry is the most functionally relevant test for skiing. '
            'Eccentric decel RFD asymmetry reflects landing control imbalance.'
        ),

        'forceframe_sections': [],

        'dynamo_section': None,

        'nordbord_section': None,

        'key_metrics': [
            {
                'label': 'CMJ Height',
                'test_types': ['CMJ'],
                'col': 'JUMP_HEIGHT_IMP_MOM',
                'unit': 'cm',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'IMTP Rel. Peak Force',
                'test_types': ['IMTP'],
                'col': 'ISO_BM_REL_FORCE_PEAK',
                'unit': 'N/kg',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'RSI (Hop)',
                'test_types': ['HJ', 'SLHJ', 'RSHIP', 'RSKIP', 'RSAIP'],
                'col': 'HOP_BEST_RSI',
                'unit': '',
                'format': '.2f',
                'higher_is_better': True,
            },
            {
                'label': 'Eccentric Decel RFD',
                'test_types': ['CMJ'],
                'col': 'ECCENTRIC_DECEL_RFD',
                'unit': 'N/s',
                'format': '.0f',
                'higher_is_better': True,
            },
            {
                'label': 'Ecc Braking Impulse',
                'test_types': ['CMJ'],
                'col': 'ECCENTRIC_BRAKING_IMPULSE',
                'unit': 'Ns',
                'format': '.1f',
                'higher_is_better': True,
            },
            {
                'label': 'Lower Limb Stiffness',
                'test_types': ['CMJ'],
                'col': 'LOWER_LIMB_STIFFNESS',
                'unit': 'N/m',
                'format': '.0f',
                'higher_is_better': True,
            },
        ],

        'extra_sections': [
            {
                'type': 'hop_test',
                'title': '10:5 Hop / Reactive Strength',
                'test_types': ['HJ', 'SLHJ', 'RSHIP', 'RSKIP', 'RSAIP'],
                'metric_key': 'rsi',
            },
        ],

        # ---- Benchmarks ----
        'benchmarks': {
            'cmj_height': {
                'elite_m': (42, 52), 'national_m': (36, 42),
                'elite_f': (30, 38), 'national_f': (26, 30),
                'unit': 'cm',
                'source': 'Jordan et al., 2018; Patterson et al., 2014',
            },
            'rsi_moguls': {
                'elite_m': (3.0, None), 'national_m': (2.2, 3.0),
                'elite_f': (2.4, None), 'national_f': (1.8, 2.4),
                'unit': '',
                'source': 'Mogul-specific - reflects rapid absorption/takeoff demands',
            },
            'nordbord_per_leg': {
                'elite_m': (400, None), 'national_m': (337, 400),
                'elite_f': (280, None), 'national_f': (256, 280),
                'unit': 'N',
                'source': 'Opar et al., 2015; <10% asymmetry critical for ACL risk',
            },
        },

        # ---- Performance ratios ----
        'performance_ratios': [],

        # ---- Injury risk indicators ----
        'injury_risk_indicators': [
            {
                'name': 'ACL Risk - SL Strength Asymmetry',
                'test': 'ForceDecks',
                'test_types': ['SLISOSQT'],
                'metric': 'peak_force_asymmetry',
                'threshold': 10.0,
                'direction': 'above',
                'severity': 'high',
                'source': 'Hewett et al., 2005; SL asymmetry is #1 modifiable ACL risk factor',
            },
            {
                'name': 'Hamstring Injury Risk',
                'test': 'NordBord',
                'metric': 'absolute_force_per_leg',
                'threshold_male': 400,
                'threshold_female': 280,
                'direction': 'below',
                'severity': 'high',
                'source': 'Opar et al., 2015',
            },
            {
                'name': 'NordBord L/R Asymmetry (ACL Risk)',
                'test': 'NordBord',
                'metric': 'bilateral_asymmetry',
                'threshold': 10.0,
                'direction': 'above',
                'severity': 'high',
                'source': 'Croisier et al., 2008; linked to ACL risk in skiing',
            },
        ],

        'benchmarks_key': 'Snow Sports',
        'context_notes': [
            'Eccentric strength is the #1 physical priority - landing absorption defines snow sport performance.',
            'ACL prevention is CRITICAL - SL asymmetry >10% is the primary modifiable risk factor (Hewett et al., 2005).',
            'NordBord >400N with <10% asymmetry recommended for ACL protection.',
            'RSI >3.0 for mogul skiers reflects rapid absorption/takeoff demands.',
            'Lower limb stiffness reflects shock absorption capacity during high-GRF landings.',
            'Eccentric decel RFD is the most sport-specific CMJ metric for snow sports.',
            'SL ISO Squat (SLISOSQT, n=3) provides the most functionally relevant unilateral assessment.',
            'Isometric tests (ISOT, n=2) monitor sustained force production for technical positions.',
        ],
    },
}


# ============================================================================
# LOOKUP UTILITIES
# ============================================================================

def get_diagnostic_config(sport_key: str) -> Optional[Dict[str, Any]]:
    """Get diagnostic config by sport key (lowercase).

    Args:
        sport_key: Lowercase sport identifier (e.g., 'fencing', 'karate').

    Returns:
        Config dict or None if not found.
    """
    return SPORT_DIAGNOSTIC_CONFIGS.get(sport_key)


def get_config_by_display_name(display_name: str) -> Optional[Dict[str, Any]]:
    """Look up config by display name (e.g., 'Fencing', 'Karate').

    Returns the first config whose display_name matches (case-insensitive).
    """
    name_lower = display_name.lower() if display_name else ''
    for key, config in SPORT_DIAGNOSTIC_CONFIGS.items():
        if config['display_name'].lower() == name_lower:
            return config
    return None


def get_sport_key_for_athlete(athlete_sport: str) -> Optional[str]:
    """Return the SPORT_DIAGNOSTIC_CONFIGS key for a given athlete_sport value.

    Useful for looking up a config when you have the athlete_sport column value.
    """
    if not athlete_sport:
        return None
    for key, config in SPORT_DIAGNOSTIC_CONFIGS.items():
        if athlete_sport in config['sub_sports']:
            return key
    # Fallback: prefix match
    athlete_lower = athlete_sport.lower()
    for key in SPORT_DIAGNOSTIC_CONFIGS:
        if athlete_lower.startswith(key.replace('_', ' ')):
            return key
    return None


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == '__main__':
    print('Sport Diagnostic Configs Loaded')
    print(f'Total sports configured: {len(SPORT_DIAGNOSTIC_CONFIGS)}')
    print()

    for key, cfg in SPORT_DIAGNOSTIC_CONFIGS.items():
        n_asym = len(cfg['asymmetry_metrics'])
        n_ff = len(cfg['forceframe_sections'])
        n_km = len(cfg['key_metrics'])
        n_extra = len(cfg['extra_sections'])
        has_grip = cfg['dynamo_section'] is not None and cfg['dynamo_section'].get('enabled')
        has_nb = cfg['nordbord_section'] is not None and cfg['nordbord_section'].get('enabled')
        thresholds = cfg['asymmetry_thresholds']
        n_bench = len(cfg.get('benchmarks', {}))
        n_ratios = len(cfg.get('performance_ratios', []))
        n_injury = len(cfg.get('injury_risk_indicators', []))
        n_notes = len(cfg.get('context_notes', []))

        print(
            f"  {cfg['display_name']:20s} | "
            f"sub_sports={len(cfg['sub_sports']):2d} | "
            f"asym={n_asym} | ff={n_ff} | km={n_km} | extra={n_extra} | "
            f"grip={'Y' if has_grip else 'N'} | nb={'Y' if has_nb else 'N'} | "
            f"bench={n_bench} | ratios={n_ratios} | injury={n_injury} | "
            f"notes={n_notes} | "
            f"thresholds=({thresholds['caution']}, {thresholds['risk']})"
        )

    # Verify all sub_sports are unique across configs (no overlaps)
    all_subs = []
    for cfg in SPORT_DIAGNOSTIC_CONFIGS.values():
        all_subs.extend(cfg['sub_sports'])
    duplicates = [s for s in all_subs if all_subs.count(s) > 1]
    if duplicates:
        print(f'\nWARNING: Duplicate sub_sports across configs: {set(duplicates)}')
    else:
        print('\nAll sub_sports are unique across configs.')

    # Summary of new fields
    total_benchmarks = sum(len(c.get('benchmarks', {})) for c in SPORT_DIAGNOSTIC_CONFIGS.values())
    total_ratios = sum(len(c.get('performance_ratios', [])) for c in SPORT_DIAGNOSTIC_CONFIGS.values())
    total_injury = sum(len(c.get('injury_risk_indicators', [])) for c in SPORT_DIAGNOSTIC_CONFIGS.values())
    total_notes = sum(len(c.get('context_notes', [])) for c in SPORT_DIAGNOSTIC_CONFIGS.values())
    print(f'\nTotal benchmarks: {total_benchmarks}')
    print(f'Total performance ratios: {total_ratios}')
    print(f'Total injury risk indicators: {total_injury}')
    print(f'Total context notes: {total_notes}')
