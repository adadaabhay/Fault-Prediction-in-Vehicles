"""End-to-end smoke test for the docs/ HUD.

This is the only test that exercises the published artifacts in a real
browser.  It starts a local HTTP server rooted at ``docs/`` (the same
source GitHub Pages serves, just over ``http://localhost:PORT/``
instead of ``file://``) and asserts:

  1. The page loads without raising any uncaught JS exception **or**
     emitting a ``console.error`` (the ``init().catch`` path writes
     to ``console.error``, which ``pageerror`` would not catch).
  2. The dynamic chip-flip hook in ``docs/dashboard.js`` reports the
     retrain chip as ``ok`` (both ``config.json`` and ``model.json``
     return a 2xx with a parseable JSON body).
  3. The 5 static "ok" chips in the REMEDIATION ROUND 1 bar are
     present and green.
  4. Every parameter flagged ``health_exclude`` in ``ml/parts.py``
     (loaded dynamically) renders with the ``display-only`` modifier
     and the amber "n/a" tag when its part modal is opened.
  5. The ``error`` and ``pending`` terminal states of the chip are
     reachable when the artifact probes fail (using ``page.route``
     to inject 404 responses without disturbing the real files), and
     a 2xx response with a non-parseable body is correctly routed
     to ``error`` (a build defect, not a mid-deploy state).

Why HTTP and not ``file://``: the dashboard's ``init()`` does
``fetch('config.json')`` and ``fetch('model.json')`` for the
retrain-chip probe.  CORS disallows ``fetch`` from a ``file://``
origin, so the chip would always land in the ``error`` state and
every positive assertion would fail for an environmental reason
unrelated to the code under test.  Serving from a real origin
matches the production deployment (GitHub Pages over https) and
unifies the test path with the deploy path.

The test follows the same opt-in pattern as the rest of the suite
(``tests/test_hil_ingest.py``, ``tests/test_fdir_on_real_data.py``):
the import is wrapped in a try/except, the test class is
``@unittest.skipUnless(HAVE, ...)`` so a missing Playwright install
shows as a skip, not a failure.  The ``setUpClass`` hook additionally
probes the chromium executable and raises ``SkipTest`` if the browser
binary is not installed, so a CI runner with the Python package but
no ``playwright install chromium`` step skips cleanly instead of
erroring out the class.

Install: ``pip install playwright && playwright install chromium``
"""

from __future__ import annotations

import contextlib
import importlib
import http.server
import socket
import socketserver
import sys
import threading
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS_INDEX = REPO / "docs" / "index.html"
DOCS_ROOT = REPO / "docs"

# Static REMEDIATION ROUND 1 chips, in declaration order.  Sourced
# from ``docs/index.html`` lines 68-72; the test will fail with a
# helpful message if a future round rewords one of them.
STATIC_CHIPS = (
    "LSTM GRADIENT FIXED",
    "INPUT SCHEMA D=26",
    "COMBO LABELS FIXED",
    "DISPLAY-ONLY FLAGGED",
    "LEAKAGE GATES LIVE",
)

# Chip terminal states, in declaration order.  See
# ``docs/REMEDIATION.md`` "Dynamic retrain chip contract".
CHIP_OK = "ok"
CHIP_PENDING = "pending"
CHIP_ERROR = "error"

try:
    from playwright.sync_api import sync_playwright  # noqa: F401
    HAVE = DOCS_INDEX.exists()
except Exception:  # pragma: no cover - environment dependent
    HAVE = False


def _load_health_exclude_params() -> tuple[tuple[str, str], ...]:
    """Return every ``(part_id, param_key)`` flagged ``health_exclude``.

    Loaded from ``ml.parts.PARTS`` at import time so a future round
    that adds a new display-only parameter does not silently skip it.
    Skipped (and reported as such) when the package is not importable
    in the current Python environment.
    """
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        parts_mod = importlib.import_module("ml.parts")
    except Exception as e:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(
            f"ml.parts not importable in this env: {e!r}"
        ) from e
    out: list[tuple[str, str]] = []
    for pid, p in parts_mod.PARTS.items():
        for param in p.get("params", []):
            if param.get("health_exclude"):
                out.append((pid, param["key"]))
    return tuple(out)


def _chromium_is_launchable(pw) -> bool:
    """Return True if the chromium binary is on disk and launchable.

    Catches the case where the Playwright Python package is installed
    but ``playwright install chromium`` was never run.  This is
    environment-dependent and cannot be unit-tested itself.
    """
    try:
        browser = pw.chromium.launch(headless=True)
        browser.close()
        return True
    except Exception:
        return False


def _start_docs_server(root: Path) -> tuple[socketserver.TCPServer, str]:
    """Serve ``root`` over HTTP on a free localhost port; return
    ``(server, base_url)``.

    Why we need this (and why ``file://`` won't work): the dashboard's
    ``init()`` does ``fetch('config.json')`` and ``fetch('model.json')``
    to drive the retrain chip.  CORS blocks ``fetch`` from a
    ``file://`` origin, so the chip would always land in the
    ``error`` state under the old ``file://`` test and every
    positive assertion would fail for an environmental reason.  A
    localhost server matches the production deploy (GitHub Pages
    over https) closely enough that the test path and the deploy
    path share the same fetch + CORS semantics.

    The port is allocated by binding to port 0 and reading back the
    assigned number, so a parallel test run does not race.  The
    server thread is a daemon so it dies with the interpreter if
    ``shutdown()`` is somehow skipped.
    """

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, *_args, **_kwargs):
            # Silence the per-request access log during tests;
            # failures are still visible via the pageerror / console
            # listeners in the test class.
            return

    class _ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    # Pick a port the OS finds for us; using ThreadingMixIn so a
    # page.route() handler that blocks the event loop does not
    # deadlock the test client.
    httpd = _ThreadingServer(("127.0.0.1", 0), _Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, f"http://127.0.0.1:{port}/index.html"


@unittest.skipUnless(HAVE, "playwright not installed or docs/index.html missing")
class TestHudSmoke(unittest.TestCase):
    """End-to-end checks against the published HUD."""

    health_exclude_params: tuple[tuple[str, str], ...] = ()

    @classmethod
    def setUpClass(cls):
        # Defer the health_exclude import to setUpClass so a missing
        # ml package shows as a single, actionable skip rather than
        # blocking the whole module from importing.
        cls.health_exclude_params = _load_health_exclude_params()
        # Spin up a single-shot HTTP server rooted at docs/ so the
        # dashboard's fetch() calls for config.json / model.json are
        # allowed by CORS.  See module docstring for why file:// is
        # not viable here.
        cls._httpd, cls.url = _start_docs_server(DOCS_ROOT)
        cls._pw = sync_playwright().start()
        if not _chromium_is_launchable(cls._pw):
            cls._pw.stop()
            cls._httpd.shutdown()
            raise unittest.SkipTest(
                "playwright python is installed but the chromium binary "
                "is not; run `playwright install chromium`"
            )
        cls._browser = cls._pw.chromium.launch(headless=True)
        cls._ctx = cls._browser.new_context()
        cls._page = cls._ctx.new_page()
        # Register a single ``console`` + ``pageerror`` pair for the
        # whole class.  ``init().catch`` writes to ``console.error``,
        # which ``pageerror`` would not catch -- so we have to wire
        # both or the test will silently pass on a caught rejection.
        cls._console_errors: list[str] = []
        cls._page.on("console",
                     lambda m: cls._console_errors.append(m.text)
                     if m.type == "error" else None)
        cls._page.on("pageerror", lambda e: cls._console_errors.append(str(e)))

    @classmethod
    def tearDownClass(cls):
        cls._ctx.close()
        cls._browser.close()
        cls._pw.stop()
        # ThreadingHTTPServer.shutdown() blocks until serve_forever()
        # returns; the daemon thread joins on its own.  Wrapped in a
        # try/except so a tearDown on a half-initialised class
        # (e.g. chromium missing) does not raise.
        try:
            cls._httpd.shutdown()
        except Exception:  # pragma: no cover - teardown-only
            pass
        cls._httpd.server_close()

    def setUp(self):
        # Per-test reset so the assertion in test_01 can read the
        # errors raised by the navigation in this test, not the
        # previous one.
        self._console_errors.clear()

    def _goto(self):
        # ``wait_until="load"`` is enough; the sentinel poll below
        # covers the async init() completion.  ``networkidle`` interacts
        # badly with the long-lived LiveSocket WebSocket and is
        # environment-dependent.
        self._page.goto(self.url, wait_until="load")
        # Wait on the ``data-probed`` sentinel that verifyArtifacts() (or
        # the init().catch handler) sets once it has decided the chip's
        # terminal state.  Waiting on the negation of the ``pending``
        # class is a bug: the chip starts in ``pending`` AND the
        # pending terminal state is also ``pending``, so a test that
        # drives the chip into the pending branch via ``page.route``
        # will time out instead of asserting.  The sentinel is set
        # exactly once and is independent of the terminal class.
        self._page.wait_for_function(
            "() => { const c = document.getElementById('chip_retrain'); "
            "return c && c.dataset.probed === '1'; }",
            timeout=10000,
        )

    def test_01_page_loads_without_console_errors(self):
        self._goto()
        self.assertEqual(self._console_errors, [],
                         f"page raised console errors: {self._console_errors}")

    def test_02_retrain_chip_is_ok(self):
        self._goto()
        chip = self._page.locator("#chip_retrain")
        chip_class = chip.get_attribute("class") or ""
        # Tokenize so a future class like ``r-chip ok-pending`` does
        # not pass the assertion by substring.
        self.assertEqual((chip_class.split() or [""])[0], "r-chip")
        self.assertIn(CHIP_OK, chip_class.split(),
                      f"chip class is {chip_class!r}; expected 'ok' "
                      f"after the artifact probe")
        self.assertEqual(chip.text_content().strip(),
                         "MODEL RETRAIN COMPLETE")

    def test_03_static_remediation_chips_present(self):
        self._goto()
        for label in STATIC_CHIPS:
            chip = self._page.locator(".r-chip", has_text=label).first
            self.assertTrue(chip.count() > 0,
                            f"missing static chip: {label}")
            chip_class = chip.get_attribute("class") or ""
            self.assertIn(CHIP_OK, chip_class.split(),
                          f"chip {label!r} class is {chip_class!r}; "
                          f"expected 'ok'")

    def test_04_health_exclude_params_render_as_display_only(self):
        self._goto()
        # Guard: at least one health_exclude param must exist, or the
        # test would pass vacuously.  Loaded from ml.parts at class
        # setup so a future round that adds display-only params
        # automatically extends the coverage.
        self.assertGreater(len(self.health_exclude_params), 0,
                           "no health_exclude parameters found in ml/parts.py")
        for pid, key in self.health_exclude_params:
            card = self._page.locator(f"#card_{pid}")
            self.assertTrue(card.count() > 0, f"missing card #{pid}")
            card.first.click()
            sel = f"#mspark_{key}"
            self._page.wait_for_selector(sel, timeout=5000)
            item = self._page.locator(f".spark-item:has({sel})")
            item_class = item.first.get_attribute("class") or ""
            self.assertIn("display-only", item_class.split(),
                          f"{pid}.{key} modal row class is "
                          f"{item_class!r}; expected 'display-only'")
            tag = self._page.locator(
                f".spark-item:has({sel}) .display-only-tag"
            )
            # Exactly one tag per spark item; ``> 0`` would let a
            # future regression that appends two tags slip through.
            self.assertEqual(tag.count(), 1,
                             f"{pid}.{key} modal row should have exactly "
                             f"one n/a tag, found {tag.count()}")
            # closeModule() is bound to Escape in init() at
            # dashboard.js:876 -- no comment about a no-op fallback.
            self._page.keyboard.press("Escape")

    def test_05_retrain_chip_is_error_when_both_probes_404(self):
        # Inject 404s for both artifacts without touching the real
        # files.  All other requests pass through.
        def _route_404(route):
            if route.request.url.endswith("config.json") or \
               route.request.url.endswith("model.json"):
                route.fulfill(status=404, body="not found")
            else:
                route.continue_()
        self._page.route("**/*", _route_404)
        try:
            self._goto()
        finally:
            self._page.unroute("**/*", _route_404)
        chip_class = self._page.locator(
            "#chip_retrain"
        ).get_attribute("class") or ""
        self.assertIn(CHIP_ERROR, chip_class.split(),
                      f"chip class is {chip_class!r}; expected 'error' "
                      f"when both probes 404")
        self.assertEqual(
            self._page.locator("#chip_retrain").text_content().strip(),
            "ARTIFACTS UNREACHABLE",
        )

    def test_06_retrain_chip_is_pending_when_one_probe_404s(self):
        def _route_one_404(route):
            if route.request.url.endswith("config.json"):
                route.fulfill(status=404, body="not found")
            else:
                route.continue_()
        self._page.route("**/*", _route_one_404)
        try:
            self._goto()
        finally:
            self._page.unroute("**/*", _route_one_404)
        chip_class = self._page.locator(
            "#chip_retrain"
        ).get_attribute("class") or ""
        self.assertIn(CHIP_PENDING, chip_class.split(),
                      f"chip class is {chip_class!r}; expected 'pending' "
                      f"when one probe 404s")
        self.assertEqual(
            self._page.locator("#chip_retrain").text_content().strip(),
            "MODEL RETRAIN PENDING",
        )

    def test_07_retrain_chip_is_error_when_one_body_fails_to_parse(self):
        # A 2xx with a non-parseable body is a build defect, not a
        # mid-deploy state.  Per the chip contract (see
        # docs/REMEDIATION.md "Dynamic retrain chip contract"), any
        # parse failure on either artifact must route to ``error``,
        # never ``pending`` -- a green chip on a broken body is the
        # exact lie this gate is meant to prevent.  test_06 covers
        # the HTTP-failure-pending branch; this test covers the
        # 2xx+garbage-error branch on the opposite side of the
        # symmetry.
        def _route_garbage_model(route):
            if route.request.url.endswith("model.json"):
                route.fulfill(status=200,
                              content_type="application/json",
                              body="<html>oops</html>")
            else:
                route.continue_()
        self._page.route("**/*", _route_garbage_model)
        try:
            self._goto()
        finally:
            self._page.unroute("**/*", _route_garbage_model)
        chip_class = self._page.locator(
            "#chip_retrain"
        ).get_attribute("class") or ""
        self.assertIn(CHIP_ERROR, chip_class.split(),
                      f"chip class is {chip_class!r}; expected 'error' "
                      f"when one body is 2xx+non-JSON")
        self.assertEqual(
            self._page.locator("#chip_retrain").text_content().strip(),
            "ARTIFACTS UNREACHABLE",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
