"""
Athlete Data Integration Script
Team Saudi - VALD Performance Analysis

Quick script to run athlete integration once data pull completes.
Enriches ForceDecks, ForceFrame, and NordBord test data with athlete names, sports and biometrics.

Usage:
    python integrate_athletes.py
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from vald_production_system import UnifiedValdSystem, DEVICE_CONFIG
except ImportError:
    print("ERROR: vald_production_system.py not found!")
    print("Please ensure you're in the correct directory.")
    sys.exit(1)


def integrate_all_devices():
    """Run athlete integration for all devices (ForceDecks, ForceFrame, NordBord)"""
    print("\n" + "=" * 80)
    print("ATHLETE DATA INTEGRATION - ALL DEVICES")
    print("Team Saudi - VALD Performance Analysis")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Initialize system
    print("Initializing VALD system...")
    system = UnifiedValdSystem()

    # Fetch athletes first
    print("\n" + "-" * 80)
    print("STEP 1: Fetching athlete data from VALD API")
    print("-" * 80)

    athletes_df = system.athlete_manager.fetch_athletes()

    if athletes_df.empty:
        print("ERROR: No athlete data found from API")
        sys.exit(1)

    print(f"Loaded {len(athletes_df)} athletes")

    # Create mapping
    athlete_mapping = system.athlete_manager.create_athlete_mapping(athletes_df)

    # Process each device
    devices = ['forcedecks', 'forceframe', 'nordbord']
    results = {}

    for device in devices:
        print("\n" + "-" * 80)
        print(f"Processing {device.upper()}")
        print("-" * 80)

        master_file = os.path.join(
            system.config.MASTER_DIR,
            DEVICE_CONFIG[device]['master_file']
        )

        if not os.path.exists(master_file):
            print(f"  No {device} data found at {master_file}")
            results[device] = {'status': 'not_found'}
            continue

        try:
            # Load data
            df = pd.read_csv(master_file)
            print(f"  Loaded {len(df)} {device} records")

            # Enrich with athlete data
            enriched_df = system.athlete_manager.enrich_with_athlete_data(df, athlete_mapping)

            # Save enriched data
            output_file = master_file.replace('.csv', '_with_athletes.csv')
            enriched_df.to_csv(output_file, index=False)
            print(f"  Enriched data saved to: {output_file}")

            # Count matches
            matched = enriched_df['athlete_sport'].notna().sum() if 'athlete_sport' in enriched_df.columns else 0
            named = enriched_df['full_name'].notna().sum() if 'full_name' in enriched_df.columns else 0

            results[device] = {
                'status': 'success',
                'total_records': len(enriched_df),
                'matched_records': matched,
                'named_records': named,
                'output_file': output_file
            }

            print(f"  Matched {matched}/{len(enriched_df)} records with athlete data")
            print(f"  Named {named}/{len(enriched_df)} records with full names")

        except Exception as e:
            print(f"  ERROR processing {device}: {e}")
            results[device] = {'status': 'error', 'error': str(e)}

    # Summary
    print("\n" + "=" * 80)
    print("INTEGRATION COMPLETE!")
    print("=" * 80)

    print("\nResults Summary:")
    for device, result in results.items():
        if result['status'] == 'success':
            print(f"  {device}: {result['named_records']}/{result['total_records']} athletes named")
        elif result['status'] == 'not_found':
            print(f"  {device}: No data file found")
        else:
            print(f"  {device}: ERROR - {result.get('error', 'Unknown')}")

    # Next steps
    print("\n" + "-" * 80)
    print("NEXT STEPS")
    print("-" * 80)
    print("1. View data in dashboard:")
    print("   streamlit run dashboard/world_class_vald_dashboard.py")
    print("\n2. Use interactive trace selector:")
    print("   python interactive_trace_selector.py")

    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    return results


def main():
    """Run athlete integration"""
    try:
        integrate_all_devices()
    except Exception as e:
        print(f"\nERROR during integration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
