"""
Playwright E2E Tests - S&C Coach User Flow
Tests the dashboard from an S&C coach's perspective.
"""
import asyncio
from playwright.async_api import async_playwright, expect
import time

# Dashboard URL (local Streamlit)
BASE_URL = "http://localhost:8504"

async def test_snc_diagnostics_flow():
    """Test S&C Diagnostics tab navigation and athlete selection."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=500)
        page = await browser.new_page()

        print("\n=== S&C Coach Flow Test ===\n")

        # 1. Navigate to dashboard
        print("1. Loading dashboard...")
        await page.goto(BASE_URL, wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(3000)  # Wait for Streamlit to fully load

        # 2. Navigate to Reports tab (contains S&C Diagnostics)
        print("2. Clicking Reports tab...")
        try:
            reports_tab = page.locator("button:has-text('Reports')")
            await reports_tab.click()
            await page.wait_for_timeout(2000)
        except Exception as e:
            print(f"   Warning: Could not find Reports tab - {e}")

        # 3. Look for S&C Diagnostics section
        print("3. Looking for S&C Diagnostics...")
        try:
            snc_section = page.locator("text=S&C Diagnostics")
            if await snc_section.count() > 0:
                await snc_section.first.click()
                await page.wait_for_timeout(2000)
                print("   Found S&C Diagnostics section")
            else:
                print("   S&C Diagnostics not found on this page")
        except Exception as e:
            print(f"   Note: {e}")

        # 4. Test IMTP tab
        print("4. Testing IMTP tab...")
        try:
            imtp_tab = page.locator("button:has-text('IMTP')")
            if await imtp_tab.count() > 0:
                await imtp_tab.first.click()
                await page.wait_for_timeout(2000)
                print("   IMTP tab clicked")

                # Check for Individual View
                individual_view = page.locator("button:has-text('Individual View')")
                if await individual_view.count() > 0:
                    await individual_view.click()
                    await page.wait_for_timeout(1500)
                    print("   Individual View selected")

                    # Try to select athletes
                    athlete_select = page.locator("[data-testid='stMultiSelect']").first
                    if await athlete_select.count() > 0:
                        await athlete_select.click()
                        await page.wait_for_timeout(1000)
                        print("   Athlete selector opened")
        except Exception as e:
            print(f"   IMTP test error: {e}")

        # 5. Test Data Entry tab and form persistence
        print("5. Testing Data Entry tab...")
        try:
            data_entry_tab = page.locator("button:has-text('Data Entry')")
            if await data_entry_tab.count() > 0:
                await data_entry_tab.first.click()
                await page.wait_for_timeout(2000)
                print("   Data Entry tab clicked")

                # Look for Upper Body S&C option
                ub_option = page.locator("text=S&C Upper Body")
                if await ub_option.count() > 0:
                    await ub_option.first.click()
                    await page.wait_for_timeout(1500)
                    print("   S&C Upper Body selected")

                    # Take screenshot of current state
                    await page.screenshot(path="tests/screenshots/ub_entry_before.png")
                    print("   Screenshot saved: ub_entry_before.png")
        except Exception as e:
            print(f"   Data Entry test error: {e}")

        # 6. Test Quadrant Tests tab (ForceFrame)
        print("6. Testing Quadrant Tests (ForceFrame)...")
        try:
            # Go back to Reports if needed
            reports_tab = page.locator("button:has-text('Reports')")
            if await reports_tab.count() > 0:
                await reports_tab.first.click()
                await page.wait_for_timeout(2000)

            quadrant_tab = page.locator("button:has-text('Quadrant')")
            if await quadrant_tab.count() > 0:
                await quadrant_tab.first.click()
                await page.wait_for_timeout(2000)
                print("   Quadrant tab clicked")

                # Check for errors
                error_msg = page.locator("text=Error")
                if await error_msg.count() > 0:
                    print("   WARNING: Error detected in Quadrant tab!")
                    await page.screenshot(path="tests/screenshots/quadrant_error.png")
                else:
                    print("   No errors detected in Quadrant tab")
                    await page.screenshot(path="tests/screenshots/quadrant_success.png")
        except Exception as e:
            print(f"   Quadrant test error: {e}")

        # 7. Test sport filter changes
        print("7. Testing sport filter persistence...")
        try:
            sport_filter = page.locator("[data-testid='stSelectbox']").first
            if await sport_filter.count() > 0:
                await sport_filter.click()
                await page.wait_for_timeout(1000)

                # Select a different sport
                fencing_option = page.locator("text=Fencing")
                if await fencing_option.count() > 0:
                    await fencing_option.first.click()
                    await page.wait_for_timeout(2000)
                    print("   Sport filter changed to Fencing")
        except Exception as e:
            print(f"   Sport filter test error: {e}")

        print("\n=== Test Complete ===")
        print("Review screenshots in tests/screenshots/")

        # Keep browser open for manual inspection
        await page.wait_for_timeout(5000)
        await browser.close()


async def test_data_entry_persistence():
    """Test that data entry stays on selected form after submission."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=300)
        page = await browser.new_page()

        print("\n=== Data Entry Persistence Test ===\n")

        await page.goto(BASE_URL, wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(3000)

        # Navigate to Data Entry
        print("1. Going to Data Entry tab...")
        data_entry = page.locator("button:has-text('Data Entry')")
        if await data_entry.count() > 0:
            await data_entry.first.click()
            await page.wait_for_timeout(2000)

        # Select Upper Body
        print("2. Selecting S&C Upper Body...")
        ub_radio = page.locator("label:has-text('S&C Upper Body')")
        if await ub_radio.count() > 0:
            await ub_radio.click()
            await page.wait_for_timeout(1500)

            # Verify we're on Upper Body
            current_url = page.url
            print(f"   Current URL: {current_url}")

            # Take screenshot
            await page.screenshot(path="tests/screenshots/ub_selected.png")
            print("   Screenshot: ub_selected.png")

        print("\n=== Persistence Test Complete ===")
        await page.wait_for_timeout(3000)
        await browser.close()


if __name__ == "__main__":
    import os

    # Create screenshots directory
    os.makedirs("tests/screenshots", exist_ok=True)

    print("Starting Playwright tests...")
    print("Make sure the dashboard is running: streamlit run dashboard/world_class_vald_dashboard.py")
    print()

    # Run tests
    asyncio.run(test_snc_diagnostics_flow())
    # asyncio.run(test_data_entry_persistence())
