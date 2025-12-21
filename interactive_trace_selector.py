"""
Interactive Force Trace Selector
Team Saudi - VALD Performance Analysis

Shows athlete summary with test counts, then allows selective trace fetching.
Much more efficient than fetching all traces blindly.
"""

import pandas as pd
import os
from typing import List, Dict, Optional
from datetime import datetime
import sys

# Add to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from vald_production_system import UnifiedValdSystem
except ImportError:
    print("ERROR: vald_production_system.py not found!")
    sys.exit(1)


class InteractiveTraceSelector:
    """Interactive interface for selecting and fetching force traces"""

    def __init__(self):
        self.system = UnifiedValdSystem()
        self.master_file = os.path.join(
            self.system.config.MASTER_DIR,
            'forcedecks_allsports_with_athletes.csv'
        )

        if not os.path.exists(self.master_file):
            print("ERROR: No ForceDecks data found.")
            print("Run: python vald_production_system.py --update-all")
            sys.exit(1)

        self.df = pd.read_csv(self.master_file)
        print(f"Loaded {len(self.df)} total tests")

    def show_athlete_summary(self, sport: Optional[str] = None,
                            test_type: Optional[str] = None) -> pd.DataFrame:
        """
        Show summary of athletes with test counts

        Returns DataFrame with:
        - Athlete name
        - Sport
        - Total tests
        - Test types (counts)
        - Most recent test date
        - Average jump height (or key metric)
        """

        df = self.df.copy()

        # Apply filters
        if sport:
            df = df[df['athlete_sport'] == sport]
        if test_type:
            df = df[df['testType'] == test_type]

        # Group by athlete
        summary_data = []

        for athlete_name in df['Name'].unique():
            athlete_df = df[df['Name'] == athlete_name]

            # Test type breakdown
            test_counts = athlete_df['testType'].value_counts().to_dict()
            test_summary = ', '.join([f"{k}:{v}" for k, v in test_counts.items()])

            # Most recent test
            if 'recordedDateUtc' in athlete_df.columns:
                athlete_df['recordedDateUtc'] = pd.to_datetime(athlete_df['recordedDateUtc'])
                latest_test = athlete_df['recordedDateUtc'].max()
            else:
                latest_test = None

            # Key metric (jump height if available)
            key_metric = None
            if 'Jump Height (Flight Time) [cm]' in athlete_df.columns:
                key_metric = athlete_df['Jump Height (Flight Time) [cm]'].mean()
            elif 'Peak Force [N]' in athlete_df.columns:
                key_metric = athlete_df['Peak Force [N]'].mean()

            # Sport
            sport_name = athlete_df['athlete_sport'].mode()[0] if 'athlete_sport' in athlete_df.columns else 'Unknown'

            summary_data.append({
                'Athlete': athlete_name,
                'Sport': sport_name,
                'Total Tests': len(athlete_df),
                'Test Types': test_summary,
                'Latest Test': latest_test.strftime('%Y-%m-%d') if latest_test else 'Unknown',
                'Avg Metric': f"{key_metric:.1f}" if key_metric else 'N/A'
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Total Tests', ascending=False)

        return summary_df

    def get_athlete_tests(self, athlete_name: str) -> pd.DataFrame:
        """Get all tests for a specific athlete"""
        athlete_df = self.df[self.df['Name'] == athlete_name].copy()

        if 'recordedDateUtc' in athlete_df.columns:
            athlete_df['recordedDateUtc'] = pd.to_datetime(athlete_df['recordedDateUtc'])
            athlete_df = athlete_df.sort_values('recordedDateUtc', ascending=False)

        # Select relevant columns for display
        display_cols = ['testId', 'recordedDateUtc', 'testType', 'athlete_sport']

        # Add key metrics if available
        if 'Jump Height (Flight Time) [cm]' in athlete_df.columns:
            display_cols.append('Jump Height (Flight Time) [cm]')
        if 'Peak Force [N]' in athlete_df.columns:
            display_cols.append('Peak Force [N]')
        if 'Peak Power / BM_Trial' in athlete_df.columns:
            display_cols.append('Peak Power / BM_Trial')

        available_cols = [col for col in display_cols if col in athlete_df.columns]

        return athlete_df[available_cols]

    def fetch_selected_traces(self, test_ids: List[str]) -> Dict:
        """Fetch force traces for selected test IDs"""
        print(f"\nFetching force traces for {len(test_ids)} selected tests...")
        print("This may take a few minutes depending on number of trials per test.")

        trace_data = self.system.devices['forcedecks'].fetch_force_traces(test_ids)

        print(f"\nSUCCESS: Fetched {len(trace_data)} force traces")
        print(f"Saved to: data/force_traces/")

        return trace_data

    def interactive_selection(self):
        """Run interactive trace selection workflow"""
        print("\n" + "=" * 80)
        print("INTERACTIVE FORCE TRACE SELECTOR")
        print("Team Saudi - VALD Performance Analysis")
        print("=" * 80)

        # Step 1: Show available sports
        print("\nStep 1: Filter by Sport (optional)")
        print("-" * 40)

        sports = sorted([s for s in self.df['athlete_sport'].unique() if pd.notna(s)])
        print("\nAvailable sports:")
        for i, sport in enumerate(sports, 1):
            count = len(self.df[self.df['athlete_sport'] == sport])
            print(f"  {i}. {sport} ({count} tests)")
        print(f"  0. All sports")

        sport_choice = input("\nSelect sport number (or 0 for all): ").strip()

        selected_sport = None
        if sport_choice.isdigit() and int(sport_choice) > 0 and int(sport_choice) <= len(sports):
            selected_sport = sports[int(sport_choice) - 1]
            print(f"Selected: {selected_sport}")

        # Step 2: Show available test types
        print("\nStep 2: Filter by Test Type (optional)")
        print("-" * 40)

        filtered_df = self.df.copy()
        if selected_sport:
            filtered_df = filtered_df[filtered_df['athlete_sport'] == selected_sport]

        test_types = sorted(filtered_df['testType'].unique())
        print("\nAvailable test types:")
        for i, tt in enumerate(test_types, 1):
            count = len(filtered_df[filtered_df['testType'] == tt])
            print(f"  {i}. {tt} ({count} tests)")
        print(f"  0. All test types")

        tt_choice = input("\nSelect test type number (or 0 for all): ").strip()

        selected_test_type = None
        if tt_choice.isdigit() and int(tt_choice) > 0 and int(tt_choice) <= len(test_types):
            selected_test_type = test_types[int(tt_choice) - 1]
            print(f"Selected: {selected_test_type}")

        # Step 3: Show athlete summary
        print("\nStep 3: Athlete Summary")
        print("-" * 40)

        summary_df = self.show_athlete_summary(sport=selected_sport, test_type=selected_test_type)

        print(f"\nFound {len(summary_df)} athletes:")
        print(summary_df.to_string(index=False))

        # Step 4: Select athletes
        print("\nStep 4: Select Athletes")
        print("-" * 40)

        athletes_list = summary_df['Athlete'].tolist()

        print("\nOptions:")
        print("  1. Select specific athletes (enter numbers separated by commas)")
        print("  2. Select ALL athletes")
        print("  3. Select TOP N athletes (by test count)")

        selection_method = input("\nChoose method (1/2/3): ").strip()

        selected_athletes = []

        if selection_method == '1':
            # Manual selection
            print("\nAthletes:")
            for i, athlete in enumerate(athletes_list, 1):
                tests = summary_df[summary_df['Athlete'] == athlete]['Total Tests'].values[0]
                print(f"  {i}. {athlete} ({tests} tests)")

            athlete_nums = input("\nEnter athlete numbers (e.g., 1,3,5): ").strip()
            for num in athlete_nums.split(','):
                num = num.strip()
                if num.isdigit() and 1 <= int(num) <= len(athletes_list):
                    selected_athletes.append(athletes_list[int(num) - 1])

        elif selection_method == '2':
            # Select all
            selected_athletes = athletes_list

        elif selection_method == '3':
            # Top N
            n = input("Enter N (number of athletes): ").strip()
            if n.isdigit():
                selected_athletes = athletes_list[:int(n)]

        if not selected_athletes:
            print("No athletes selected. Exiting.")
            return

        print(f"\nSelected {len(selected_athletes)} athletes:")
        for athlete in selected_athletes:
            print(f"  - {athlete}")

        # Step 5: Show tests for selected athletes
        print("\nStep 5: Select Specific Tests")
        print("-" * 40)

        all_selected_tests = []

        for athlete in selected_athletes:
            athlete_tests = self.get_athlete_tests(athlete)

            print(f"\n{athlete} - {len(athlete_tests)} tests:")
            print(athlete_tests.to_string(index=False))

            print("\nOptions for this athlete:")
            print("  1. Fetch ALL tests")
            print("  2. Fetch LATEST N tests")
            print("  3. SELECT specific tests")
            print("  0. SKIP this athlete")

            choice = input("Choose option (0/1/2/3): ").strip()

            if choice == '0':
                continue
            elif choice == '1':
                # All tests
                all_selected_tests.extend(athlete_tests['testId'].tolist())
            elif choice == '2':
                # Latest N
                n = input("  How many latest tests? ").strip()
                if n.isdigit():
                    all_selected_tests.extend(athlete_tests['testId'].head(int(n)).tolist())
            elif choice == '3':
                # Manual selection
                print("\nTests:")
                for i, (idx, row) in enumerate(athlete_tests.iterrows(), 1):
                    date = row['recordedDateUtc'].strftime('%Y-%m-%d') if pd.notna(row['recordedDateUtc']) else 'Unknown'
                    print(f"  {i}. {row['testType']} - {date} - ID: {row['testId'][:8]}...")

                test_nums = input("  Enter test numbers (e.g., 1,2,3): ").strip()
                test_list = athlete_tests['testId'].tolist()
                for num in test_nums.split(','):
                    num = num.strip()
                    if num.isdigit() and 1 <= int(num) <= len(test_list):
                        all_selected_tests.append(test_list[int(num) - 1])

        # Step 6: Confirm and fetch
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total tests selected: {len(all_selected_tests)}")
        print(f"Athletes: {len(selected_athletes)}")
        print(f"\nEstimated download size: ~{len(all_selected_tests) * 2} MB")
        print(f"Estimated time: ~{len(all_selected_tests) * 5} seconds")

        confirm = input("\nProceed with trace fetching? (yes/no): ").strip().lower()

        if confirm in ['yes', 'y']:
            trace_data = self.fetch_selected_traces(all_selected_tests)

            print("\n" + "=" * 80)
            print("TRACE FETCHING COMPLETE!")
            print("=" * 80)
            print(f"Fetched {len(trace_data)} force traces")
            print(f"Saved to: data/force_traces/")
            print(f"\nYou can now use these traces in the dashboard for:")
            print("  - Force-time curve visualization")
            print("  - Phase analysis (eccentric, concentric, landing)")
            print("  - Athlete comparisons")
            print("  - Biomechanics analysis")

            return trace_data
        else:
            print("Cancelled.")
            return None


def main():
    """Run interactive trace selector"""
    selector = InteractiveTraceSelector()
    selector.interactive_selection()


if __name__ == "__main__":
    main()
