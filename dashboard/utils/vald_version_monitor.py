"""
VALD R Package Monitor
Checks for updates to the official valdr R package and suggests Python changes
"""

import requests
import re
from typing import Dict, List
from datetime import datetime


class VALDVersionMonitor:
    """Monitor VALD R package for API changes"""

    def __init__(self):
        self.r_package_url = "https://cran.r-project.org/web/packages/valdr/index.html"
        self.github_url = "https://github.com/cran/valdr"
        self.local_r_script = "VALD-R-ForceDecks/Get_ForceDecks_Data.r"

    def check_r_package_version(self) -> Dict:
        """
        Check CRAN for latest valdr package version
        """
        try:
            response = requests.get(self.r_package_url, timeout=10)
            if response.status_code == 200:
                # Extract version from HTML
                version_match = re.search(r'Version:\s*<\/td><td>([0-9.]+)', response.text)
                date_match = re.search(r'Published:\s*<\/td><td>([0-9-]+)', response.text)

                if version_match and date_match:
                    return {
                        'version': version_match.group(1),
                        'published_date': date_match.group(1),
                        'url': self.r_package_url,
                        'status': 'available'
                    }

            return {'status': 'error', 'message': 'Could not fetch version'}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def check_local_r_script(self, script_path: str = None) -> Dict:
        """
        Parse local R script to extract API endpoints and structure
        """
        if script_path is None:
            script_path = self.local_r_script

        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract API base URLs
            api_urls = re.findall(r'api_base_\w+\s*=\s*paste0\("([^"]+)"', content)

            # Extract function names
            functions = re.findall(r'^(\w+)\s*<-\s*function', content, re.MULTILINE)

            # Extract endpoint patterns
            endpoints = re.findall(r'(\/\w+\/\w+\/\w+)', content)

            return {
                'api_urls': list(set(api_urls)),
                'functions': functions,
                'endpoints': list(set(endpoints)),
                'last_modified': datetime.fromtimestamp(
                    __import__('os').path.getmtime(script_path)
                ).strftime('%Y-%m-%d')
            }

        except FileNotFoundError:
            return {'status': 'error', 'message': f'R script not found: {script_path}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def compare_with_python_implementation(self, r_info: Dict, python_info: Dict) -> List[Dict]:
        """
        Compare R implementation with Python and suggest updates
        """
        suggestions = []

        # Check API URLs
        if 'api_urls' in r_info:
            for r_url in r_info['api_urls']:
                if python_info.get('api_urls') and r_url not in python_info['api_urls']:
                    suggestions.append({
                        'type': 'api_endpoint',
                        'priority': 'high',
                        'description': f'New API URL in R package: {r_url}',
                        'action': f'Add "{r_url}" to Python API configuration'
                    })

        # Check endpoints
        if 'endpoints' in r_info:
            r_endpoints = set(r_info['endpoints'])
            py_endpoints = set(python_info.get('endpoints', []))

            new_endpoints = r_endpoints - py_endpoints
            for endpoint in new_endpoints:
                suggestions.append({
                    'type': 'endpoint',
                    'priority': 'medium',
                    'description': f'New endpoint in R: {endpoint}',
                    'action': f'Implement endpoint handler for: {endpoint}'
                })

        # Check functions
        if 'functions' in r_info:
            r_functions = set(r_info['functions'])
            py_functions = set(python_info.get('functions', []))

            new_functions = r_functions - py_functions
            for func in new_functions:
                suggestions.append({
                    'type': 'function',
                    'priority': 'low',
                    'description': f'New function in R: {func}',
                    'action': f'Consider implementing: {func}()'
                })

        return suggestions

    def get_python_implementation_info(self, script_path: str) -> Dict:
        """
        Extract information from Python implementation
        """
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract API URLs
            api_urls = re.findall(r'["\']https://prd-[^"\']+["\']', content)
            api_urls = [url.strip('"\'') for url in api_urls]

            # Extract class methods (functions)
            functions = re.findall(r'def\s+(\w+)\(', content)

            # Extract endpoint patterns
            endpoints = re.findall(r'["\'](\\/\w+\\/\w+\\/\w+)["\']', content)

            return {
                'api_urls': list(set(api_urls)),
                'functions': functions,
                'endpoints': list(set(endpoints))
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def generate_update_report(self) -> Dict:
        """
        Generate comprehensive update report
        """
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'r_package_version': self.check_r_package_version(),
            'r_script_info': self.check_local_r_script(),
            'suggestions': []
        }

        # Compare if both are available
        if report['r_script_info'].get('status') != 'error':
            # Try to get Python info
            python_script = 'vald_api_pulls-main/forcedecks/20250730_omplete_vald_system.py'
            python_info = self.get_python_implementation_info(python_script)

            if python_info.get('status') != 'error':
                report['python_implementation'] = python_info
                report['suggestions'] = self.compare_with_python_implementation(
                    report['r_script_info'],
                    python_info
                )

        return report


def check_for_vald_updates():
    """
    Main function to check for VALD updates
    """
    monitor = VALDVersionMonitor()
    report = monitor.generate_update_report()

    print("="*80)
    print("VALD API Update Check")
    print("="*80)
    print(f"Timestamp: {report['timestamp']}\n")

    # R Package Version
    if report['r_package_version'].get('status') == 'available':
        print("OK CRAN valdr Package:")
        print(f"  Version: {report['r_package_version']['version']}")
        print(f"  Published: {report['r_package_version']['published_date']}")
    else:
        print("WARNING: Could not fetch R package version")

    print()

    # R Script Info
    if report['r_script_info'].get('status') != 'error':
        print("OK Local R Script Analysis:")
        print(f"  API URLs: {len(report['r_script_info'].get('api_urls', []))}")
        print(f"  Functions: {len(report['r_script_info'].get('functions', []))}")
        print(f"  Endpoints: {len(report['r_script_info'].get('endpoints', []))}")
        print(f"  Last Modified: {report['r_script_info'].get('last_modified', 'Unknown')}")
    else:
        print(f"WARNING R Script Error: {report['r_script_info'].get('message')}")

    print()

    # Suggestions
    if report['suggestions']:
        print("Update Suggestions:")
        print(f"  Total: {len(report['suggestions'])}")
        print()

        high_priority = [s for s in report['suggestions'] if s['priority'] == 'high']
        medium_priority = [s for s in report['suggestions'] if s['priority'] == 'medium']
        low_priority = [s for s in report['suggestions'] if s['priority'] == 'low']

        if high_priority:
            print("  HIGH PRIORITY:")
            for s in high_priority:
                print(f"    - {s['description']}")
                print(f"      Action: {s['action']}")
            print()

        if medium_priority:
            print("  MEDIUM PRIORITY:")
            for s in medium_priority:
                print(f"    - {s['description']}")
                print(f"      Action: {s['action']}")
            print()

        if low_priority:
            print("  LOW PRIORITY:")
            for s in low_priority:
                print(f"    - {s['description']}")
            print()
    else:
        print("OK No updates needed - Python implementation is up to date")

    print("="*80)

    return report


def get_vald_api_documentation_links() -> Dict[str, str]:
    """
    Return links to official VALD documentation
    """
    return {
        'CRAN valdr': 'https://cran.r-project.org/web/packages/valdr/',
        'VALD Support': 'https://support.vald.com/hc/en-au/articles/48730811824281-A-guide-to-using-the-valdr-R-package',
        'VALD API Docs': 'https://valdperformance.com/news/introducing-valdr',
        'GitHub': 'https://github.com/cran/valdr'
    }


if __name__ == "__main__":
    # Run update check
    report = check_for_vald_updates()

    # Show documentation links
    print("\nOfficial Documentation:")
    for name, url in get_vald_api_documentation_links().items():
        print(f"  {name}: {url}")
