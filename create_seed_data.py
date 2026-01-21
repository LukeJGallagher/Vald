"""
Script to create seed data for Luke Gallagher by copying real test data
and changing the athlete name.
"""
import pandas as pd
import os
from datetime import datetime, timedelta
import random

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'dashboard', 'data')
FORCEDECKS_FILE = os.path.join(DATA_DIR, 'forcedecks_allsports_with_athletes.csv')
FORCEFRAME_FILE = os.path.join(DATA_DIR, 'forceframe_allsports.csv')
NORDBORD_FILE = os.path.join(DATA_DIR, 'nordbord_allsports.csv')

# Target athlete details
TARGET_NAME = "Luke Gallagher"
TARGET_SPORT = "Athletics - Throws"
TARGET_PROFILE_ID = "SEED_LUKE_001"

def create_forcedecks_seed():
    """Create ForceDecks seed data for Luke Gallagher."""
    print("Loading ForceDecks data...")
    df = pd.read_csv(FORCEDECKS_FILE, low_memory=False)

    # Test types we want to create data for
    test_types = ['CMJ', 'IMTP', 'RSHIP', 'DJ', 'SLIMTP', 'SLISOSQT', 'SLCMRJ', 'SJ', 'ISOSQT']

    seed_rows = []

    for test_type in test_types:
        # Get rows for this test type
        type_df = df[df['testType'] == test_type]
        if type_df.empty:
            print(f"  No data found for {test_type}")
            continue

        # Get 4-6 rows (different dates) from different athletes
        sample_size = min(5, len(type_df))
        sample_rows = type_df.sample(n=sample_size) if len(type_df) >= sample_size else type_df.copy()

        # Modify each row for Luke Gallagher
        for idx, row in sample_rows.iterrows():
            new_row = row.copy()
            new_row['full_name'] = TARGET_NAME
            new_row['athlete_sport'] = TARGET_SPORT
            new_row['athlete_sex'] = 'Male'
            new_row['profileId'] = TARGET_PROFILE_ID
            new_row['testId'] = f"SEED_{test_type}_{random.randint(1000,9999)}"

            # Adjust date to be within last 6 months
            days_ago = random.randint(1, 180)
            new_date = datetime.now() - timedelta(days=days_ago)
            new_row['recordedDateUtc'] = new_date.strftime('%Y-%m-%dT%H:%M:%S')
            new_row['modifiedDateUtc'] = new_date.strftime('%Y-%m-%dT%H:%M:%S')

            seed_rows.append(new_row)

        print(f"  Created {len(sample_rows)} entries for {test_type}")

    if seed_rows:
        # Convert to DataFrame
        seed_df = pd.DataFrame(seed_rows)

        # Append to existing file
        df_combined = pd.concat([df, seed_df], ignore_index=True)
        df_combined.to_csv(FORCEDECKS_FILE, index=False)
        print(f"Added {len(seed_rows)} ForceDecks entries for {TARGET_NAME}")
    else:
        print("No seed rows created for ForceDecks")

def create_forceframe_seed():
    """Create ForceFrame seed data for Luke Gallagher."""
    print("\nLoading ForceFrame data...")
    df = pd.read_csv(FORCEFRAME_FILE)

    # Test types we want
    test_types = ['Shoulder IR/ER', 'Hip AD/AB', 'Hip IR/ER', 'Knee Extension', 'Knee Flexion']

    seed_rows = []

    for test_type in test_types:
        type_df = df[df['testTypeName'].str.contains(test_type, case=False, na=False)]
        if type_df.empty:
            print(f"  No data found for {test_type}")
            continue

        # Get 3-5 sample rows
        sample_size = min(4, len(type_df))
        sample_rows = type_df.sample(n=sample_size) if len(type_df) >= sample_size else type_df.copy()

        for idx, row in sample_rows.iterrows():
            new_row = row.copy()
            new_row['athleteId'] = TARGET_PROFILE_ID
            # Add Name column if not exists
            new_row['Name'] = TARGET_NAME
            new_row['testId'] = f"SEED_FF_{test_type.replace(' ', '_')}_{random.randint(1000,9999)}"

            # Adjust date
            days_ago = random.randint(1, 180)
            new_date = datetime.now() - timedelta(days=days_ago)
            new_row['testDateUtc'] = new_date.strftime('%Y-%m-%dT%H:%M:%S')
            new_row['modifiedDateUtc'] = new_date.strftime('%Y-%m-%dT%H:%M:%S')

            seed_rows.append(new_row)

        print(f"  Created {len(sample_rows)} entries for {test_type}")

    if seed_rows:
        seed_df = pd.DataFrame(seed_rows)
        df_combined = pd.concat([df, seed_df], ignore_index=True)
        df_combined.to_csv(FORCEFRAME_FILE, index=False)
        print(f"Added {len(seed_rows)} ForceFrame entries for {TARGET_NAME}")

def create_nordbord_seed():
    """Create NordBord seed data for Luke Gallagher."""
    print("\nLoading NordBord data...")
    df = pd.read_csv(NORDBORD_FILE)

    if df.empty:
        print("  No NordBord data available")
        return

    # Get 5 sample rows
    sample_size = min(5, len(df))
    sample_rows = df.sample(n=sample_size) if len(df) >= sample_size else df.copy()

    seed_rows = []
    for idx, row in sample_rows.iterrows():
        new_row = row.copy()
        new_row['athleteId'] = TARGET_PROFILE_ID
        new_row['Name'] = TARGET_NAME
        new_row['full_name'] = TARGET_NAME

        # Adjust date
        days_ago = random.randint(1, 180)
        new_date = datetime.now() - timedelta(days=days_ago)
        if 'recordedDateUtc' in new_row:
            new_row['recordedDateUtc'] = new_date.strftime('%Y-%m-%dT%H:%M:%S')
        if 'modifiedDateUtc' in new_row:
            new_row['modifiedDateUtc'] = new_date.strftime('%Y-%m-%dT%H:%M:%S')

        seed_rows.append(new_row)

    if seed_rows:
        seed_df = pd.DataFrame(seed_rows)
        df_combined = pd.concat([df, seed_df], ignore_index=True)
        df_combined.to_csv(NORDBORD_FILE, index=False)
        print(f"Added {len(seed_rows)} NordBord entries for {TARGET_NAME}")

if __name__ == "__main__":
    print(f"Creating seed data for {TARGET_NAME}...\n")

    create_forcedecks_seed()
    create_forceframe_seed()
    create_nordbord_seed()

    print("\n✅ Seed data creation complete!")
    print("Note: You can delete these entries later by removing rows where full_name = 'Luke Gallagher'")
