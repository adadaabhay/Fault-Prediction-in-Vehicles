"""Tests for per-subsystem generator CSV unit handling.

The Pa -> bar scale fix in :func:`sim.generators._base.run_subsystem`
means the CSV column ``oil_pressure`` (declared as bar with
``scale=1e-5`` in ``engine.py``) should now contain values in the bar
range, not the raw Pa range the simulator produces.  This test pins
the post-fix behaviour so a regression that drops the ``c.scale``
multiplication will fail loudly, and it smoke-tests the new NBC
generator (wired to ``CVRDEAuxiliaryNBC``) so a wiring regression
fails too.
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.generators import SUBSYSTEMS


class TestGeneratorUnits(unittest.TestCase):
    """Generator CSV writes published units, not raw SI."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _column(self, csv_path: Path, name: str) -> list[float]:
        with csv_path.open(newline="", encoding="utf-8") as f:
            return [float(r[name]) for r in csv.DictReader(f)]

    # ------------------------------------------------------------------
    # Pa -> bar fix: the upper bound is the meaningful check (the Pa
    # range is ~1e5x the bar range, so a regression that drops the
    # ``c.scale`` multiplication will blow through these ceilings).
    # ------------------------------------------------------------------
    def test_engine_oil_pressure_in_bar(self):
        out = self.tmp / "engine.csv"
        SUBSYSTEMS["engine"].generate(str(out), steps=200, seed=42, dt=0.05)
        values = self._column(out, "oil_pressure")
        # Nominal gallery is ~5 bar.  2-8 bar covers healthy transients
        # without admitting the Pa-range regression.
        self.assertGreater(min(values), 2.0,
            f"oil_pressure min={min(values):.1f} (expected bar range)")
        self.assertLess(max(values), 8.0,
            f"oil_pressure max={max(values):.1f} (expected bar range)")

    def test_engine_exhaust_pressure_in_bar(self):
        out = self.tmp / "engine.csv"
        SUBSYSTEMS["engine"].generate(str(out), steps=200, seed=42, dt=0.05)
        values = self._column(out, "exhaust_pressure")
        # Nominal post-turbine is ~1.3 bar.
        self.assertGreater(min(values), 0.3,
            f"exhaust_pressure min={min(values):.1f} (expected bar range)")
        self.assertLess(max(values), 3.0,
            f"exhaust_pressure max={max(values):.1f} (expected bar range)")

    def test_hydraulics_hyd_pressure_in_bar(self):
        out = self.tmp / "hydraulics.csv"
        SUBSYSTEMS["hydraulics"].generate(str(out), steps=200, seed=42, dt=0.05)
        values = self._column(out, "hyd_pressure")
        # 21 MPa circuit -> ~210 bar nominal.
        self.assertGreater(min(values), 100.0,
            f"hyd_pressure min={min(values):.1f} (expected bar range)")
        self.assertLess(max(values), 300.0,
            f"hyd_pressure max={max(values):.1f} (expected bar range)")

    # ------------------------------------------------------------------
    # NBC wiring: the new generator drives CVRDEAuxiliaryNBC and should
    # emit real (non-NaN) cabin overpressure + filter dp values, plus
    # a sidecar manifest whose ``part_key`` and channel count match
    # ``ml.parts.PARTS['nbc']``.
    # ------------------------------------------------------------------
    def test_nbc_emits_real_overpressure_and_filter_dp(self):
        out = self.tmp / "nbc.csv"
        rows = SUBSYSTEMS["nbc"].generate(str(out), steps=100, seed=42, dt=0.05)
        self.assertEqual(rows, 100)

        with out.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or [])
            self.assertEqual(
                columns,
                {"time", "step", "nbc_overpressure", "nbc_filter_dp"},
            )
            first = next(reader)

        # Healthy steady-state: 500 Pa cabin overpressure, 120 Pa
        # filter dp.  Allow the warmup band so a model that starts
        # the cabin pressure at 0 doesn't fail.
        overpressure = float(first["nbc_overpressure"])
        filter_dp = float(first["nbc_filter_dp"])
        self.assertGreater(overpressure, 100.0,
            f"nbc_overpressure={overpressure:.1f} Pa (expected APU on)")
        self.assertLess(overpressure, 700.0,
            f"nbc_overpressure={overpressure:.1f} Pa (expected healthy)")
        self.assertGreater(filter_dp, 80.0,
            f"nbc_filter_dp={filter_dp:.1f} Pa (expected clean filter)")
        self.assertLess(filter_dp, 200.0,
            f"nbc_filter_dp={filter_dp:.1f} Pa (expected clean filter)")

    def test_nbc_manifest_matches_part_schema(self):
        out = self.tmp / "nbc.csv"
        SUBSYSTEMS["nbc"].generate(str(out), steps=50, seed=42, dt=0.05)

        manifest_path = out.with_name(out.name + ".manifest.json")
        self.assertTrue(manifest_path.exists(),
            f"missing manifest sidecar at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["subsystem"], "nbc")
        self.assertEqual(manifest["part_key"], "nbc")
        self.assertEqual(len(manifest["channels"]), 4)
        channel_keys = {c["key"] for c in manifest["channels"]}
        self.assertEqual(
            channel_keys,
            {"time", "step", "nbc_overpressure", "nbc_filter_dp"},
        )


if __name__ == "__main__":
    unittest.main()
