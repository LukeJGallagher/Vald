"""
VALD Group to Category (Sport) Mapping

Categories are the actual sports. Groups are sub-divisions within sports.
This mapping is based on Team Saudi's VALD account structure.

VALD Hierarchy: Categories (Sports) -> Groups -> Athletes
"""

# Complete mapping of VALD Groups to their parent Category (Sport)
# Based on Team Saudi VALD account structure
GROUP_TO_CATEGORY = {
    # Fencing (3 groups)
    'Epee': 'Fencing',
    'Epee ': 'Fencing',  # With trailing space (data quirk)
    'Foil': 'Fencing',
    'Sabre': 'Fencing',

    # Athletics (5 groups)
    'Athletics - Horizontal Jumps': 'Athletics',
    'Athletics - Middle distance': 'Athletics',
    'Short Sprints': 'Athletics',
    'Throwers': 'Athletics',
    'Decathlon': 'Athletics',

    # Wrestling (4 groups)
    'Freestyle': 'Wrestling',
    'Greco Roman': 'Wrestling',
    'GS': 'Wrestling',
    'RUS': 'Wrestling',

    # Taekwondo (4 groups)
    'TKD Junior Female': 'Taekwondo',
    'TKD Junior Male': 'Taekwondo',
    'TKD Senior Female': 'Taekwondo',
    'TKD Senior Male': 'Taekwondo',

    # Swimming (1 group)
    'SOTC Swimming': 'Swimming',

    # Para Sports (5 groups)
    'Para Swimming': 'Para Swimming',
    'Para Sprints': 'Para Athletics',
    'Para TKD': 'Para Taekwondo',
    'Para Cycling': 'Para Cycling',
    'Wheel Chair': 'Wheelchair Sports',

    # Individual Sports (group name = category name or direct mapping)
    'Karate': 'Karate',
    'Coastal': 'Rowing',
    'Pistol 10m': 'Shooting',
    'Snow Sports': 'Snow Sports',
    'Equestrian': 'Equestrian',
    'Judo': 'Judo',
    'Jiu-Jitsu': 'Jiu-Jitsu',
    'Weightlifting': 'Weightlifting',

    # Exclude from sport assignment (return None)
    'ARCHIVED': None,
    'Staff': None,
    'TBC': None,
    'All Athletes': None,
    'All athletes': None,
}

# Groups to skip when determining sport
SKIP_GROUPS = {
    'ARCHIVED', 'Staff', 'TBC', 'All Athletes', 'All athletes',
    'VALD HQ', 'Test Group', 'Performance Staff', 'Coaches',
    'Medical', 'Admin', 'SOTC Performance', 'Unknown'
}

# All known sports for UI dropdown (alphabetical)
ALL_SPORTS = [
    'Athletics',
    'Equestrian',
    'Fencing',
    'Jiu-Jitsu',
    'Judo',
    'Karate',
    'Para Athletics',
    'Para Cycling',
    'Para Swimming',
    'Para Taekwondo',
    'Rowing',
    'Shooting',
    'Snow Sports',
    'Swimming',
    'Taekwondo',
    'Weightlifting',
    'Wheelchair Sports',
    'Wrestling',
]


def get_sport_from_groups(group_names: list) -> str:
    """
    Given a list of VALD group names, return the athlete's sport (category).

    Priority:
    1. Direct mapping from GROUP_TO_CATEGORY
    2. If no match, use group name as sport (if not in SKIP_GROUPS)
    3. Return 'Unknown' if no valid sport found

    Args:
        group_names: List of VALD group names the athlete belongs to

    Returns:
        Sport name (category) or 'Unknown'
    """
    if not group_names:
        return 'Unknown'

    for group_name in group_names:
        # Skip administrative groups
        if group_name in SKIP_GROUPS:
            continue

        # Check direct mapping first
        if group_name in GROUP_TO_CATEGORY:
            category = GROUP_TO_CATEGORY[group_name]
            if category:  # Not None (excluded groups)
                return category

        # Use group name as sport if not in skip list
        if group_name and group_name not in SKIP_GROUPS:
            return group_name

    return 'Unknown'
