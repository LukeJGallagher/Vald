"""
Test if changing filters causes page reset to home.
"""
import asyncio
from playwright.async_api import async_playwright

BASE_URL = "http://localhost:8505"

async def test_filter_page_reset():
    """Test that changing filters doesn't reset to home page."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=500)
        page = await browser.new_page()

        print("\n=== Filter Navigation Test ===\n")

        # 1. Load dashboard
        print("1. Loading dashboard...")
        await page.goto(BASE_URL, wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(5000)

        # 2. Navigate to Reports tab
        print("2. Navigating to Reports tab...")
        reports_tab = page.locator("button[role='tab']:has-text('Reports')")
        if await reports_tab.count() > 0:
            await reports_tab.click()
            await page.wait_for_timeout(3000)
            print("   On Reports tab")

        # 3. Find sport filter and note current position
        print("3. Looking for sport filter...")
        current_url_before = page.url

        # Find all selectboxes
        selectboxes = page.locator("[data-testid='stSelectbox']")
        selectbox_count = await selectboxes.count()
        print(f"   Found {selectbox_count} selectboxes")

        if selectbox_count > 0:
            # Click the first visible selectbox
            for i in range(selectbox_count):
                sb = selectboxes.nth(i)
                if await sb.is_visible():
                    print(f"   Clicking selectbox {i}...")
                    await sb.click()
                    await page.wait_for_timeout(1000)

                    # Look for Fencing option
                    options = page.locator("[data-testid='stSelectboxVirtualDropdown'] >> text=Fencing")
                    if await options.count() > 0:
                        print("   Found Fencing option, selecting...")
                        await options.first.click()
                        await page.wait_for_timeout(3000)
                        break

        # 4. Check if we're still on the same tab
        print("4. Checking current position after filter change...")
        current_url_after = page.url

        # Check if Reports tab is still active
        reports_active = page.locator("button[role='tab'][aria-selected='true']:has-text('Reports')")
        is_still_on_reports = await reports_active.count() > 0

        if is_still_on_reports:
            print("   SUCCESS: Still on Reports tab after filter change")
        else:
            print("   FAIL: Page reset - no longer on Reports tab")

            # Check which tab is active
            active_tab = page.locator("button[role='tab'][aria-selected='true']")
            if await active_tab.count() > 0:
                tab_text = await active_tab.first.text_content()
                print(f"   Currently on: {tab_text}")

        # 5. Test athlete selection in Individual View
        print("\n5. Testing athlete selection persistence...")

        # Look for Individual View tab
        individual_view = page.locator("button:has-text('Individual View')")
        if await individual_view.count() > 0:
            await individual_view.first.click()
            await page.wait_for_timeout(2000)
            print("   Clicked Individual View")

            # Find multiselect
            multiselect = page.locator("[data-testid='stMultiSelect']")
            if await multiselect.count() > 0:
                await multiselect.first.click()
                await page.wait_for_timeout(1000)
                print("   Opened athlete selector")

                # Select an option
                option = page.locator("[data-testid='stMultiSelectDropdown'] >> div").first
                if await option.count() > 0:
                    await option.click()
                    await page.wait_for_timeout(2000)
                    print("   Selected an athlete")

                    # Now check if still on same view
                    still_on_individual = page.locator("button[aria-selected='true']:has-text('Individual View')")
                    if await still_on_individual.count() > 0:
                        print("   SUCCESS: Still on Individual View after athlete selection")
                    else:
                        print("   FAIL: View reset after athlete selection")

        print("\n=== Test Complete ===")
        await page.wait_for_timeout(3000)
        await browser.close()


if __name__ == "__main__":
    print("Testing filter navigation...")
    print("Dashboard should be running on port 8505")
    asyncio.run(test_filter_page_reset())
