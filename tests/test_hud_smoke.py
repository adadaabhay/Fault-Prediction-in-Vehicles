"""End-to-end smoke test for the docs/ HUD.

This is the only test that exercises the published artifacts in a real
browser.  It opens ``docs/index.html`` over ``file://`` (the same
source GitHub Pages serves) and asserts:

  1. The dynamic chip-flip hook in ``docs/dashboard.js`` reports the
     retrain chip as ``ok`` (both ``config.json`` and ``model.json``
     return 200 from the in-browser fetch).
  2. The 5 static "ok" chips in the REMEDIATION ROUND 1 bar are
     present and green.
  3. A part card with ``health_exclude`` parameters (NBC, Exhaust,
     Acoustics, Hydraulics) opens its modal and renders each
     excluded parameter with the ``display-only`` modifier and the
     amber "n/a" tag.

The test follows the same opt-in pattern as the rest of the suite
(``tests/test_hil_ingest.py``, ``tests/test_fdir_on_real_data.py``):
the import is wrapped in a try/except, the test class is
``@unittest.skipUnless(HAVE, ...)`` so a missing Playwright install
shows as a skip, not a failure.  CI installs Playwright before
running; local machines without it skip cleanly.

Install: ``pip install playwright && playwright install chromium``
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS_INDEX = REPO / "docs" / "index.html"

# The 6 health_exclude parameters added/marked in the audit-remediation
# round 1.  The HUD must render every one of them with the
# ``display-only`` modifier and the amber "n/a" tag when its part
# modal is opened.
HEALTH_EXCLUDE_PARAMS = (
    ("nbc", "nbc_overpressure"),
    ("nbc", "nbc_filter_dp"),
    ("exhaust", "exhaust_backpressure"),
    ("exhaust", "particulate_index"),
    ("acoustics", "ae_burst_energy"),
    ("hydraulics", "hyd_force"),
)

# Static REMEDIATION ROUND 1 chips, in declaration order.
STATIC_CHIPS = (
    "LSTM GRADIENT FIXED",
    "INPUT SCHEMA D=26",
    "COMBO LABELS FIXED",
    "DISPLAY-ONLY FLAGGED",
    "LEAKAGE GATES LIVE",
)

try:
    from playwright.sync_api import sync_playwright  # noqa: F401
    HAVE = DOCS_INDEX.exists()
except Exception:  # pragma: no cover - environment dependent
    HAVE = False


@unittest.skipUnless(HAVE, "playwright not installed or docs/index.html missing")
class TestHudSmoke(unittest.TestCase):
    """End-to-end checks against the published HUD."""

    @classmethod
    def setUpClass(cls):
        cls.url = DOCS_INDEX.as_uri()
        cls._pw = sync_playwright().start()
        cls._browser = cls._pw.chromium.launch(headless=True)
        cls._ctx = cls._browser.new_context()
        cls._page = cls._ctx.new_page()

    @classmethod
    def tearDownClass(cls):
        cls._ctx.close()
        cls._browser.close()
        cls._pw.stop()

    def test_page_loads_without_console_errors(self):
        errors: list[str] = []
        self._page.on("pageerror", lambda e: errors.append(str(e)))
        self._page.goto(self.url, wait_until="networkidle")
        # The init() function in dashboard.js awaits both config.json and
        # model.json, then runs the chip-flip hook.  Wait for the
        # retrain chip to leave its initial "pending" class.
        self._page.wait_for_function(
            "() => { const c = document.getElementById('chip_retrain'); "
            "return c && !c.classList.contains('pending'); }",
            timeout=10000,
        )
        self.assertEqual(errors, [],
                         f"page raised console errors: {errors}")

    def test_retrain_chip_is_ok(self):
        self._page.goto(self.url, wait_until="networkidle")
        self._page.wait_for_function(
            "() => { const c = document.getElementById('chip_retrain'); "
            "return c && !c.classList.contains('pending'); }",
            timeout=10000,
        )
        chip = self._page.locator("#chip_retrain")
        cls = chip.get_attribute("class") or ""
        self.assertIn("r-chip", cls)
        self.assertIn("ok", cls,
                      f"chip class is {cls!r}; expected 'ok' after artifact probe")
        self.assertEqual(chip.text_content().strip(),
                         "MODEL RETRAIN COMPLETE")

    def test_static_remediation_chips_present(self):
        self._page.goto(self.url, wait_until="networkidle")
        # The bar is in the DOM as soon as the page parses; no async.
        for label in STATIC_CHIPS:
            chip = self._page.locator(".r-chip", has_text=label).first
            self.assertTrue(chip.count() > 0,
                            f"missing static chip: {label}")
            cls = chip.get_attribute("class") or ""
            self.assertIn("ok", cls,
                          f"chip {label!r} class is {cls!r}; expected 'ok'")

    def test_health_exclude_params_render_as_display_only(self):
        self._page.goto(self.url, wait_until="networkidle")
        for pid, key in HEALTH_EXCLUDE_PARAMS:
            card = self._page.locator(f"#card_{pid}")
            self.assertTrue(card.count() > 0, f"missing card #{pid}")
            card.first.click()
            # The modal opens and the params list is built by openModule().
            # Wait for the spark row to appear, then assert the
            # display-only modifier and the n/a tag.
            sel = f"#mspark_{key}"
            self._page.wait_for_selector(sel, timeout=5000)
            item = self._page.locator(f".spark-item:has({sel})")
            cls = item.first.get_attribute("class") or ""
            self.assertIn("display-only", cls,
                          f"{pid}.{key} modal row class is {cls!r}; "
                          f"expected 'display-only'")
            tag = self._page.locator(
                f".spark-item:has({sel}) .display-only-tag"
            )
            self.assertTrue(tag.count() > 0,
                            f"{pid}.{key} modal row missing the n/a tag")
            # Close the modal before the next iteration.
            self._page.keyboard.press("Escape")
            # The close handler is a click on the close button; press
            # Escape as a best-effort fallback (no-op if not bound).


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
