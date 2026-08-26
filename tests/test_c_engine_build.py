"""The C edge runtime must actually compile.

The previous suite only text-scraped the source for `malloc(` and computed
struct sizes arithmetically, so nothing established that the code built -- let
alone that it computed correctly.
"""

import subprocess
import unittest

from c_engine.build import BUILD_DIR, build_engine, have_toolchain, missing_tools

_SKIP_REASON = "no C toolchain: missing " + ", ".join(missing_tools() or ["-"])


@unittest.skipUnless(have_toolchain(), _SKIP_REASON)
class TestCEngineCompiles(unittest.TestCase):
    def test_builds_shared_library(self):
        lib = build_engine(force=True)
        self.assertIsNotNone(lib, "build_engine returned None with a toolchain present")
        self.assertTrue(lib.exists(), f"{lib} was not produced")
        self.assertGreater(lib.stat().st_size, 0)

    def test_self_test_binary_runs_clean(self):
        build_engine()
        base = BUILD_DIR / "tank_pdm_selftest"
        candidates = (base,
                      base.with_suffix(".exe"),
                      BUILD_DIR / "Release" / "tank_pdm_selftest.exe")
        for candidate in candidates:
            if candidate.exists():
                proc = subprocess.run([str(candidate)], capture_output=True,
                                      text=True, cwd=str(BUILD_DIR))
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                return
        self.skipTest("self-test binary not produced by this generator")


class TestToolchainReporting(unittest.TestCase):
    """These run everywhere: a skip must be able to explain itself."""

    def test_missing_tools_is_empty_exactly_when_toolchain_present(self):
        self.assertEqual(have_toolchain(), not missing_tools())

    def test_build_returns_none_without_a_toolchain(self):
        if have_toolchain():
            self.skipTest("toolchain present")
        self.assertIsNone(build_engine())
