# VALD Category-to-Group Mapping Fix

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix athlete sport assignment by properly mapping VALD Groups to their parent Categories (sports).

**Architecture:** VALD uses a hierarchy: Categories (sports) → Groups (sub-groups) → Athletes. Current code tries to derive sport from group names, but the correct approach is to use the Categories API to get the parent category (the actual sport) for each group.

**Tech Stack:** Python, requests, pandas, VALD External API

---

## Problem Summary

Currently 298 of 1110 tests show "Unknown" sport because:
1. The bulk profiles endpoint does NOT return groupIds
2. Individual profile endpoint returns groupIds, but we map group names incorrectly
3. Group names like "Epee", "Foil", "Sabre" need to be mapped to their category "Fencing"

## VALD API Structure

From user's VALD account:

| Category (Sport) | Groups |
|-----------------|--------|
| Fencing | Epee, Foil, Sabre |
| Athletics | Athletics - Horizontal Jumps, Athletics - Middle distance, Short Sprints, Throwers, Decathlon |
| Wrestling | Freestyle, Greco Roman, GS, RUS |
| Taekwondo | TKD Junior Female, TKD Junior Male, TKD Senior Female, TKD Senior Male |
| Swimming | SOTC Swimming |
| Para Swimming | Para Swimming |
| Para Athletics | Para Sprints |
| Karate | Karate |
| Rowing | Coastal |
| Shooting | Pistol 10m |
| Snow Sports | Snow Sports |
| Equestrian | Equestrian |
| Judo | Judo |
| Jiu-Jitsu | Jiu-Jitsu |
| Weightlifting | Weightlifting |
| Para Taekwondo | Para TKD |
| Para Cycling | Para Cycling |
| Wheelchair Sports | Wheel Chair |
| ARCHIVED | ARCHIVED |
| STAFF | Staff |

---

## Task 1: Create Backups (Rollback Mechanism)

**Files:**
- Backup: `dashboard/world_class_vald_dashboard.py` → `dashboard/world_class_vald_dashboard.py.backup`
- Backup: `enrich_from_forcedecks.py` → `enrich_from_forcedecks.py.backup`
- Backup: `scripts/github_sync.py` → `scripts/github_sync.py.backup`

**Step 1: Create backup files**

```bash
cd "c:\Users\l.gallagher\OneDrive - Team Saudi\Documents\Performance Analysis\Vald"
cp dashboard/world_class_vald_dashboard.py dashboard/world_class_vald_dashboard.py.backup
cp enrich_from_forcedecks.py enrich_from_forcedecks.py.backup
cp scripts/github_sync.py scripts/github_sync.py.backup
```

**Step 2: Verify backups exist**

```bash
ls -la dashboard/*.backup enrich_from_forcedecks.py.backup scripts/*.backup
```

---

## Task 2: Create Centralized Group-to-Category Mapping

**Files:**
- Create: `config/vald_categories.py`

**Step 1: Create the category mapping file**

```python
"""
VALD Group to Category (Sport) Mapping

Categories are the actual sports. Groups are sub-divisions within sports.
This mapping is based on Team Saudi's VALD account structure.
"""

# Complete mapping of VALD Groups to their parent Category (Sport)
GROUP_TO_CATEGORY = {
    # Fencing
    'Epee': 'Fencing',
    'Epee ': 'Fencing',  # With trailing space (data quirk)
    'Foil': 'Fencing',
    'Sabre': 'Fencing',

    # Athletics
    'Athletics - Horizontal Jumps': 'Athletics',
    'Athletics - Middle distance': 'Athletics',
    'Short Sprints': 'Athletics',
    'Throwers': 'Athletics',
    'Decathlon': 'Athletics',

    # Wrestling
    'Freestyle': 'Wrestling',
    'Greco Roman': 'Wrestling',
    'GS': 'Wrestling',
    'RUS': 'Wrestling',

    # Taekwondo
    'TKD Junior Female': 'Taekwondo',
    'TKD Junior Male': 'Taekwondo',
    'TKD Senior Female': 'Taekwondo',
    'TKD Senior Male': 'Taekwondo',

    # Swimming
    'SOTC Swimming': 'Swimming',

    # Para Sports
    'Para Swimming': 'Para Swimming',
    'Para Sprints': 'Para Athletics',
    'Para TKD': 'Para Taekwondo',
    'Para Cycling': 'Para Cycling',
    'Wheel Chair': 'Wheelchair Sports',

    # Individual Sports (Group name = Category name)
    'Karate': 'Karate',
    'Coastal': 'Rowing',
    'Pistol 10m': 'Shooting',
    'Snow Sports': 'Snow Sports',
    'Equestrian': 'Equestrian',
    'Judo': 'Judo',
    'Jiu-Jitsu': 'Jiu-Jitsu',
    'Weightlifting': 'Weightlifting',

    # Exclude from sport assignment
    'ARCHIVED': None,
    'Staff': None,
    'TBC': None,
    'All Athletes': None,
}

# Groups to skip when determining sport
SKIP_GROUPS = {'ARCHIVED', 'Staff', 'TBC', 'All Athletes', 'All athletes',
               'VALD HQ', 'Test Group', 'Performance Staff', 'Coaches',
               'Medical', 'Admin', 'SOTC Performance'}

# All known sports for UI dropdown
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
    Given a list of VALD group names, return the athlete's sport.

    Priority:
    1. Direct mapping from GROUP_TO_CATEGORY
    2. If no match, use group name as sport (if not in SKIP_GROUPS)
    3. Return 'Unknown' if no valid sport found
    """
    for group_name in group_names:
        # Skip administrative groups
        if group_name in SKIP_GROUPS:
            continue

        # Check direct mapping
        if group_name in GROUP_TO_CATEGORY:
            category = GROUP_TO_CATEGORY[group_name]
            if category:  # Not None
                return category

        # Use group name as sport if not in skip list
        if group_name and group_name not in SKIP_GROUPS:
            return group_name

    return 'Unknown'
```

**Step 2: Verify file created**

```bash
cat config/vald_categories.py | head -30
```

---

## Task 3: Update enrich_from_forcedecks.py

**Files:**
- Modify: `enrich_from_forcedecks.py`

**Step 1: Import the centralized mapping**

Replace the `derive_sport_from_groups` function with import from centralized config:

```python
# At top of file, after other imports
try:
    from config.vald_categories import get_sport_from_groups, SKIP_GROUPS
except ImportError:
    # Inline fallback if config not available
    from vald_categories_inline import get_sport_from_groups, SKIP_GROUPS
```

**Step 2: Remove the old derive_sport_from_groups function**

Delete lines 111-172 (the old function).

**Step 3: Update build_athlete_lookup to use new function**

In `build_athlete_lookup()`, change line 274 from:
```python
sport = derive_sport_from_groups(group_names)
```
to:
```python
sport = get_sport_from_groups(group_names)
```

**Step 4: Test the script**

```bash
python enrich_from_forcedecks.py
```

Expected: Script runs, shows progress fetching individual profiles, enriches data.

---

## Task 4: Update scripts/github_sync.py

**Files:**
- Modify: `scripts/github_sync.py`

**Step 1: Add the same category mapping inline**

Add after imports (line 12):

```python
# Group to Category mapping (from VALD account structure)
GROUP_TO_CATEGORY = {
    'Epee': 'Fencing', 'Epee ': 'Fencing', 'Foil': 'Fencing', 'Sabre': 'Fencing',
    'Athletics - Horizontal Jumps': 'Athletics', 'Athletics - Middle distance': 'Athletics',
    'Short Sprints': 'Athletics', 'Throwers': 'Athletics', 'Decathlon': 'Athletics',
    'Freestyle': 'Wrestling', 'Greco Roman': 'Wrestling', 'GS': 'Wrestling', 'RUS': 'Wrestling',
    'TKD Junior Female': 'Taekwondo', 'TKD Junior Male': 'Taekwondo',
    'TKD Senior Female': 'Taekwondo', 'TKD Senior Male': 'Taekwondo',
    'SOTC Swimming': 'Swimming', 'Para Swimming': 'Para Swimming',
    'Para Sprints': 'Para Athletics', 'Para TKD': 'Para Taekwondo',
    'Para Cycling': 'Para Cycling', 'Wheel Chair': 'Wheelchair Sports',
    'Karate': 'Karate', 'Coastal': 'Rowing', 'Pistol 10m': 'Shooting',
    'Snow Sports': 'Snow Sports', 'Equestrian': 'Equestrian',
    'Judo': 'Judo', 'Jiu-Jitsu': 'Jiu-Jitsu', 'Weightlifting': 'Weightlifting',
}

SKIP_GROUPS = {'ARCHIVED', 'Staff', 'TBC', 'All Athletes', 'All athletes',
               'VALD HQ', 'Test Group', 'Performance Staff', 'Coaches', 'Medical', 'Admin'}


def get_sport_from_groups(group_names):
    """Get sport category from group names."""
    for name in group_names:
        if name in SKIP_GROUPS:
            continue
        if name in GROUP_TO_CATEGORY:
            return GROUP_TO_CATEGORY[name]
        if name and name not in SKIP_GROUPS:
            return name
    return 'Unknown'
```

**Step 2: Update enrich_with_groups function**

In `enrich_with_groups()`, replace lines 143-150 (the sport derivation logic):

From:
```python
sport = 'Unknown'
for name in group_names:
    if name not in generic_groups and name != 'Unknown':
        sport = name
        break
```

To:
```python
sport = get_sport_from_groups(group_names)
```

Do this in both places where sport is derived (around lines 145 and 175).

**Step 3: Test locally (if possible)**

The GitHub Action will run this script, but you can test locally:
```bash
python scripts/github_sync.py
```

---

## Task 5: Update Dashboard Sport Filter

**Files:**
- Modify: `dashboard/world_class_vald_dashboard.py`

**Step 1: Import ALL_SPORTS from config**

At top of dashboard file (after other imports):
```python
try:
    from config.vald_categories import ALL_SPORTS
except ImportError:
    ALL_SPORTS = ['Athletics', 'Fencing', 'Judo', 'Karate', 'Rowing',
                  'Shooting', 'Swimming', 'Taekwondo', 'Weightlifting', 'Wrestling']
```

**Step 2: Use ALL_SPORTS in sidebar dropdown**

Find the sport selection dropdown (around line 1840) and update to use ALL_SPORTS list.

---

## Task 6: Run Data Refresh

**Step 1: Run the enrichment script**

```bash
cd "c:\Users\l.gallagher\OneDrive - Team Saudi\Documents\Performance Analysis\Vald"
python enrich_from_forcedecks.py
```

**Step 2: Copy enriched files to vald-data**

```bash
cp data/master_files/*_with_athletes.csv "../vald-data/data/"
cd "../vald-data"
git add data/*.csv
git commit -m "Fix: Proper sport assignment via group-to-category mapping"
git push
```

**Step 3: Verify sport distribution improved**

```bash
python -c "import pandas as pd; df = pd.read_csv('../vald-data/data/forcedecks_allsports_with_athletes.csv'); print(df['athlete_sport'].value_counts())"
```

Expected: Fewer "Unknown" sports, more specific sports like "Fencing", "Taekwondo", etc.

---

## Task 7: Test Dashboard

**Step 1: Run dashboard locally**

```bash
cd dashboard
streamlit run world_class_vald_dashboard.py
```

**Step 2: Verify:**
- [ ] Sport dropdown shows all sports (Fencing, Athletics, Swimming, etc.)
- [ ] Selecting a sport shows correct athletes
- [ ] All charts and tabs work correctly
- [ ] No more "Unknown" sports (or significantly reduced)

---

## Task 8: Commit and Push

**Step 1: Commit changes to main repo**

```bash
cd "c:\Users\l.gallagher\OneDrive - Team Saudi\Documents\Performance Analysis\Vald"
git add config/vald_categories.py enrich_from_forcedecks.py scripts/github_sync.py
git commit -m "feat: Proper VALD group-to-category sport mapping

- Add centralized GROUP_TO_CATEGORY mapping in config/vald_categories.py
- Maps VALD groups (Epee, Foil, etc.) to categories (Fencing, Athletics, etc.)
- Update enrich_from_forcedecks.py to use new mapping
- Update github_sync.py for automated updates
- Fixes issue where 298 tests showed 'Unknown' sport"
git push origin main
git push origin main:master
```

---

## Rollback Procedure

If anything breaks:

**Option 1: Restore from backup files**
```bash
cp dashboard/world_class_vald_dashboard.py.backup dashboard/world_class_vald_dashboard.py
cp enrich_from_forcedecks.py.backup enrich_from_forcedecks.py
cp scripts/github_sync.py.backup scripts/github_sync.py
```

**Option 2: Git revert**
```bash
git checkout HEAD~1 -- enrich_from_forcedecks.py scripts/github_sync.py
```

**Option 3: Revert to previous data**
```bash
cd "../vald-data"
git checkout HEAD~1 -- data/forcedecks_allsports_with_athletes.csv
git push
```

---

## Summary

| File | Action |
|------|--------|
| `config/vald_categories.py` | CREATE - Centralized mapping |
| `enrich_from_forcedecks.py` | MODIFY - Use new mapping |
| `scripts/github_sync.py` | MODIFY - Use new mapping |
| `dashboard/world_class_vald_dashboard.py` | KEEP - No changes needed (already has ALL_KNOWN_SPORTS) |

**Key Fix:** Map VALD Groups (Epee, TKD Junior Male, etc.) to their parent Category (Fencing, Taekwondo, etc.) instead of using group names directly as sports.
