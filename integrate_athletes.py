"""
Athlete Data Integration Script
Team Saudi - VALD Performance Analysis

Quick script to run athlete integration once data pull completes.
Enriches ForceDecks test data with athlete sports and biometrics.

Usage:
    python integrate_athletes.py
"""

import sys
import os
from datetime import datetime

# Add to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from vald_production_system import UnifiedValdSystem
except ImportError:
    print("ERROR: vald_production_system.py not found!")
    print("Please ensure you're in the correct directory.")
    sys.exit(1)


def main():
    """Run athlete integration"""
    print("\n" + "=" * 80)
    print("ATHLETE DATA INTEGRATION")
    print("Team Saudi - VALD Performance Analysis")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Initialize system
    print("Initializing VALD system...")
    system = UnifiedValdSystem()

    # Check if ForceDecks data exists
    forcedecks_file = os.path.join(
        system.config.MASTER_DIR,
        'forcedecks_master.csv'
    )

    if not os.path.exists(forcedecks_file):
        print("\nERROR: No ForceDecks data found!")
        print("Please run: python vald_production_system.py --update-all")
        sys.exit(1)

    print(f"Found ForceDecks data: {forcedecks_file}")

    # Run integration
    print("\n" + "-" * 80)
    print("STEP 1: Fetching athlete data from VALD API")
    print("-" * 80)

    try:
        result = system.integrate_athlete_data('forcedecks')

        print("\n" + "=" * 80)
        print("INTEGRATION COMPLETE!")
        print("=" * 80)

        # Show summary
        print(f"\nAthletes fetched: {result.get('athletes_fetched', 0)}")
        print(f"Tests matched: {result.get('matched_tests', 0)}")
        print(f"Tests with sport data: {result.get('tests_with_sport', 0)}")

        if 'output_file' in result:
            print(f"\nEnriched data saved to:")
            print(f"  {result['output_file']}")

        # Show sport breakdown
        if 'sport_counts' in result:
            print("\nSport Distribution:")
            for sport, count in sorted(result['sport_counts'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {sport}: {count} tests")

        # Next steps
        print("\n" + "-" * 80)
        print("NEXT STEPS")
        print("-" * 80)
        print("1. View data in dashboard:")
        print("   streamlit run dashboard/world_class_vald_dashboard.py")
        print("\n2. Use interactive trace selector:")
        print("   python interactive_trace_selector.py")
        print("\n3. Generate athlete reports:")
        print("   python vald_production_system.py --action 4")

        print("\n" + "=" * 80)
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nERROR during integration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
