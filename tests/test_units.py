"""The SI <-> engineering unit contract must hold, and must cover every
channel the FDIR catalog scores.

Two conversion failures shipped before `telemetry_gateway/units.py` existed:

* Loud -- the three pressure channels were emitted in Pa and scored against a
  bar envelope, so every frame raised a spurious SHORT_CIRCUIT and got clamped
  (oil_pressure 527705 Pa -> 15.0 bar), driving lubrication and hydraulics
  health to 0.0 on a healthy mission.
* Silent -- `fuel_level` 0.89 (a fraction) sits *inside* the [0, 100] percent
  envelope, so nothing fired. It was simply read as 0.89 % full, all the way
  through health scoring and DTC generation.

The second is why `test_no_channel_lands_outside_its_envelope` matters more
than the round-trip test: a range check cannot catch a conversion error that
lands in range. The only defence is asserting that real telemetry, converted,
sits near the *nominal* the catalog declares.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Fault-Prediction-in-Vehicles"))

from telemetry_gateway.sensor_plausibility import SENSOR_LIMITS_CATALOG
from telemetry_gateway.units import (CANONICAL_UNITS, NATIVE_TO_CANONICAL,
                                     looks_canonical, to_canonical, to_native)

_DEMO = (ROOT / "Fault-Prediction-in-Vehicles" / "docs" / "live_stream.json")


def _demo_records():
    if not _DEMO.exists():
        return []
    return json.loads(_DEMO.read_text(encoding="utf-8"))["records"]


class TestRoundTrip(unittest.TestCase):

    SAMPLE = {"oil_pressure": 5.27e5, "hyd_pressure": 2.1e7,
              "exhaust_pressure": 1.29e5, "fuel_level": 0.89,
              "oil_level": 0.94, "coolant_level": 0.95,
              "coolant_temp": 92.0, "rpm": 1500.0}

    def test_to_canonical_then_to_native_is_identity(self):
        back = to_native(to_canonical(self.SAMPLE))
        for k, v in self.SAMPLE.items():
            self.assertAlmostEqual(back[k], v, places=6, msg=k)

    def test_conversion_actually_converts(self):
        c = to_canonical(self.SAMPLE)
        self.assertAlmostEqual(c["oil_pressure"], 5.27, places=4)
        self.assertAlmostEqual(c["hyd_pressure"], 210.0, places=4)
        self.assertAlmostEqual(c["fuel_level"], 89.0, places=4)

    def test_unconverted_channels_pass_through_untouched(self):
        c = to_canonical(self.SAMPLE)
        self.assertEqual(c["coolant_temp"], 92.0)
        self.assertEqual(c["rpm"], 1500.0)

    def test_non_numeric_values_are_left_alone(self):
        c = to_canonical({"oil_pressure": None, "source": "udp", "ok": True})
        self.assertIsNone(c["oil_pressure"])
        self.assertEqual(c["source"], "udp")
        self.assertIs(c["ok"], True)

    def test_missing_channels_are_not_invented(self):
        c = to_canonical({"rpm": 900.0})
        self.assertNotIn("oil_pressure", c)


class TestIdempotenceGuard(unittest.TestCase):
    """A client already speaking bar/percent must not be converted twice."""

    def test_si_frame_is_detected_as_native(self):
        self.assertFalse(looks_canonical({"oil_pressure": 5.27e5}))

    def test_engineering_frame_is_detected_as_canonical(self):
        self.assertTrue(looks_canonical({"oil_pressure": 5.27}))

    def test_hyd_pressure_is_used_when_oil_pressure_is_absent(self):
        self.assertFalse(looks_canonical({"hyd_pressure": 2.1e7}))
        self.assertTrue(looks_canonical({"hyd_pressure": 210.0}))


class TestCatalogAgreement(unittest.TestCase):

    def test_declared_units_match_the_fdir_catalog(self):
        """units.py and the catalog must not drift apart."""
        for channel, unit in CANONICAL_UNITS.items():
            lim = SENSOR_LIMITS_CATALOG.get(channel)
            if lim is None:
                continue
            self.assertEqual(lim.unit, unit,
                             f"{channel}: units.py says {unit!r}, "
                             f"catalog says {lim.unit!r}")

    @unittest.skipUnless(_DEMO.exists(), "docs/live_stream.json not built")
    def test_no_channel_lands_outside_its_envelope(self):
        """Real simulator telemetry, converted, must sit inside the envelope."""
        records = _demo_records()[::40]
        offenders = []
        for rec in records:
            conv = to_canonical(rec)
            for channel, lim in SENSOR_LIMITS_CATALOG.items():
                v = conv.get(channel)
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                if not (lim.min_physical <= v <= lim.max_physical):
                    offenders.append(f"{channel}={v:.4g} outside "
                                     f"[{lim.min_physical}, {lim.max_physical}]")
        self.assertEqual(sorted(set(offenders)), [], str(sorted(set(offenders))))

    @unittest.skipUnless(_DEMO.exists(), "docs/live_stream.json not built")
    def test_converted_values_are_near_the_declared_nominal(self):
        """The guard against a silent conversion error.

        `fuel_level` at 0.89 was inside [0, 100] and therefore invisible to
        every range check -- but it is nowhere near the declared nominal of a
        healthy tank. Order-of-magnitude agreement with `healthy_nominal` is
        what actually catches a wrong factor.
        """
        records = _demo_records()[::40]
        if not records:
            self.skipTest("no records")
        offenders = []
        for channel in NATIVE_TO_CANONICAL:
            lim = SENSOR_LIMITS_CATALOG.get(channel)
            if lim is None or not lim.healthy_nominal:
                continue
            vals = [to_canonical(r).get(channel) for r in records]
            vals = [v for v in vals
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if not vals:
                continue
            median = sorted(vals)[len(vals) // 2]
            nominal = abs(lim.healthy_nominal)
            if nominal <= 0:
                continue
            ratio = abs(median) / nominal
            # An order of magnitude either way. A unit error is 100x or 1e5x.
            if not (0.1 <= ratio <= 10.0):
                offenders.append(
                    f"{channel}: median {median:.4g} vs nominal "
                    f"{lim.healthy_nominal:.4g} (ratio {ratio:.3g})")
        self.assertEqual(offenders, [], str(offenders))


if __name__ == "__main__":
    unittest.main()
