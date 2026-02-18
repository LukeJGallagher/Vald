"""
VALD Group to Category (Sport) Mapping - CENTRALIZED

Categories are the actual sports with sub-group detail.
Groups are VALD's internal divisions within sports.
This mapping is based on Team Saudi's VALD account structure.

VALD Hierarchy: Categories (Sports) -> Groups -> Athletes

IMPORTANT: This is the SINGLE SOURCE OF TRUTH for sport mapping.
All sync scripts (local_sync.py, github_sync.py, reenrich_sports.py)
import from here. Do NOT duplicate this mapping elsewhere.
"""

from typing import List, Optional


# Complete mapping of VALD Groups to their sport sub-category
# Uses detailed sub-groups (e.g., "Fencing - Epee" not just "Fencing")
GROUP_TO_CATEGORY = {
    # Fencing (split by weapon)
    'Fencing - Epee ': 'Fencing - Epee', 'Epee': 'Fencing - Epee', 'Epee ': 'Fencing - Epee',
    'Fencing - Epee - Mens - 2025': 'Fencing - Epee', 'Fencing - Epee - Womens - 2025': 'Fencing - Epee',
    'Fencing - Foil': 'Fencing - Foil', 'Foil': 'Fencing - Foil',
    'Fencing - Foil - Mens - 2025': 'Fencing - Foil', 'Fencing - Foil - Womens - 2025': 'Fencing - Foil',
    'Fencing - Sabre': 'Fencing - Sabre', 'Sabre': 'Fencing - Sabre',
    'Fencing - Sabre - Mens - 2025': 'Fencing - Sabre', 'Fencing - Sabre - Womens - 2025': 'Fencing - Sabre',
    'Fencing - SOTC - 2026': 'Fencing - SOTC',

    # Athletics (split by discipline)
    'Athletics - Horizontal Jumps': 'Athletics - Jumps',
    'Athletics - Middle distance': 'Athletics - Middle Distance',
    'Athletics - Multi events': 'Athletics - Multi Events',
    'Athletics - Short Sprints': 'Athletics - Sprints',
    'Athletics - Throwers': 'Athletics - Throws',
    'Athletics - TBC': 'Athletics',
    'Short Sprints': 'Athletics - Sprints',
    'Throwers': 'Athletics - Throws',
    'Decathlon': 'Athletics - Multi Events',

    # Wrestling (split by style)
    'Freestyle': 'Wrestling - Freestyle', 'GS': 'Wrestling - Freestyle', 'RUS': 'Wrestling - Freestyle',
    'Greco Roman': 'Wrestling - Greco Roman',
    'Wrestling - Freestyle': 'Wrestling - Freestyle', 'Wrestling - Greco Roman': 'Wrestling - Greco Roman',

    # Taekwondo (split by level)
    'TKD Junior Female': 'Taekwondo - Junior', 'TKD Junior Male': 'Taekwondo - Junior',
    'TKD Senior Female': 'Taekwondo - Senior', 'TKD Senior Male': 'Taekwondo - Senior',
    'TKD TBC': 'Taekwondo',

    # Swimming
    'SOTC Swimming': 'Swimming', 'Swimming TBC': 'Swimming',

    # Rowing (split by type)
    'Rowing - Classic': 'Rowing - Classic', 'Rowing - Coastal': 'Rowing - Coastal',
    'Coastal': 'Rowing - Coastal',

    # Para Sports
    'Para Swimming': 'Para Swimming', 'Para Sprints': 'Para Athletics', 'Para TBC': 'Para Athletics',
    'Para TKD': 'Para Taekwondo', 'Para Cycling': 'Para Cycling', 'Wheel Chair': 'Wheelchair Sports',

    # Combat sports (no sub-groups)
    'Karate': 'Karate', 'Karate TBC': 'Karate',
    'Judo': 'Judo', 'Judo TBC': 'Judo',
    'Jiu-Jitsu': 'Jiu-Jitsu', 'Jiu Jitsu TBC': 'Jiu-Jitsu',

    # Other sports (no sub-groups)
    'Weightlifting': 'Weightlifting', 'Weightlifting TBC': 'Weightlifting',
    'Weightlifting 2026': 'Weightlifting',
    'Pistol 10m': 'Shooting', 'Shooting TBC': 'Shooting',
    'Equestrian': 'Equestrian', 'Equestrian TBC': 'Equestrian',
    'Snow Sports': 'Snow Sports',

    # Exclude from sport assignment (return None)
    'ARCHIVED': None, 'Staff': None, 'TBC': None,
    'All Athletes': None, 'All athletes': None,
}

# Groups to skip when determining sport
SKIP_GROUPS = {
    'ARCHIVED', 'Staff', 'TBC', 'All Athletes', 'All athletes',
    'VALD HQ', 'Test Group', 'Performance Staff', 'Coaches',
    'Medical', 'Admin', 'SOTC Performance', 'Unknown'
}

# Map sub-categories to their parent sport
SUBCATEGORY_TO_PARENT = {
    # Fencing
    'Fencing - Epee': 'Fencing',
    'Fencing - Foil': 'Fencing',
    'Fencing - Sabre': 'Fencing',
    'Fencing - SOTC': 'Fencing',
    # Athletics
    'Athletics - Jumps': 'Athletics',
    'Athletics - Middle Distance': 'Athletics',
    'Athletics - Multi Events': 'Athletics',
    'Athletics - Sprints': 'Athletics',
    'Athletics - Throws': 'Athletics',
    'Athletics': 'Athletics',
    # Wrestling
    'Wrestling - Freestyle': 'Wrestling',
    'Wrestling - Greco Roman': 'Wrestling',
    # Taekwondo
    'Taekwondo - Junior': 'Taekwondo',
    'Taekwondo - Senior': 'Taekwondo',
    'Taekwondo': 'Taekwondo',
    # Rowing
    'Rowing - Classic': 'Rowing',
    'Rowing - Coastal': 'Rowing',
    # Sports without sub-groups (map to themselves)
    'Karate': 'Karate',
    'Judo': 'Judo',
    'Jiu-Jitsu': 'Jiu-Jitsu',
    'Swimming': 'Swimming',
    'Shooting': 'Shooting',
    'Weightlifting': 'Weightlifting',
    'Equestrian': 'Equestrian',
    'Snow Sports': 'Snow Sports',
    # Para Sports
    'Para Swimming': 'Para Swimming',
    'Para Athletics': 'Para Athletics',
    'Para Taekwondo': 'Para Taekwondo',
    'Para Cycling': 'Para Cycling',
    'Wheelchair Sports': 'Wheelchair Sports',
}

# All known sports for UI dropdown (alphabetical, includes parent + sub-groups)
ALL_SPORTS = sorted(set(SUBCATEGORY_TO_PARENT.keys()))


def get_parent_sport(sub_sport: str) -> str:
    """Get parent sport from sub-sport name.

    Returns the parent sport (e.g., "Fencing" from "Fencing - Epee").
    If no mapping found, returns the input unchanged.
    """
    return SUBCATEGORY_TO_PARENT.get(sub_sport, sub_sport)


def get_subcategories_for_parent(parent_sport: str) -> List[str]:
    """Get all sub-categories for a parent sport.

    e.g., get_subcategories_for_parent("Fencing") returns
    ["Fencing - Epee", "Fencing - Foil", "Fencing - Sabre", "Fencing - SOTC"]
    """
    return sorted([
        sub for sub, parent in SUBCATEGORY_TO_PARENT.items()
        if parent == parent_sport and sub != parent
    ])


def matches_sport_filter(athlete_sport: str, filter_sport: str) -> bool:
    """Check if an athlete's sport matches a filter selection.

    Handles both parent sports and sub-groups:
    - filter="Fencing" matches "Fencing - Epee", "Fencing - Foil", etc.
    - filter="Fencing - Epee" matches only "Fencing - Epee"
    """
    if not athlete_sport or not filter_sport:
        return False
    if athlete_sport == filter_sport:
        return True
    # Check if filter is a parent sport and athlete is a sub-category
    parent = SUBCATEGORY_TO_PARENT.get(athlete_sport, athlete_sport)
    return parent == filter_sport


def get_sport_from_groups(group_names: list) -> str:
    """
    Given a list of VALD group names, return the athlete's sport (sub-category).

    Priority:
    1. Direct mapping from GROUP_TO_CATEGORY
    2. If no match, use group name as sport (if not in SKIP_GROUPS)
    3. Return 'Unknown' if no valid sport found

    Args:
        group_names: List of VALD group names the athlete belongs to

    Returns:
        Sport name (sub-category) or 'Unknown'
    """
    if not group_names:
        return 'Unknown'

    for group_name in group_names:
        gn = group_name.strip() if group_name else ''
        # Skip administrative groups
        if gn in SKIP_GROUPS:
            continue

        # Check direct mapping first
        if gn in GROUP_TO_CATEGORY:
            category = GROUP_TO_CATEGORY[gn]
            if category:  # Not None (excluded groups)
                return category

        # Use group name as sport if not in skip list
        if gn and gn not in SKIP_GROUPS:
            return gn

    return 'Unknown'
