"""The PHM block chain must actually run, in order, on every frame.

This is the regression guard for the defect that made the whole gateway
misleading: `PROJECT.md` documented

    ingest -> FDIR gate -> neural inference -> DTC engine -> WebSocket

while `server.py` imported none of those modules and passed the raw broker
frame straight through. `sensor_plausibility.py` (1,401 LOC) and
`dtc_engine.py` (1,247 LOC) were reachable only from their own unit tests.
Those unit tests all passed, which is exactly why the gap survived: testing a
component proves it works, not that anything calls it.

So the assertions here are about *wiring*, not about the components:
health must be computed rather than echoed from the input, a fault injected at
the ingest boundary must reach the DTC engine, and a frame the gate rejects
must not silently become a health number.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Fault-Prediction-in-Vehicles"))

from telemetry_gateway.pipeline import PHMPipeline


def _frame(**over):
    """A plausible SI-unit frame, as sim emits it."""
    base = {
        "step": 0, "time": 0.0, "rpm": 1500.0, "load": 0.45, "terrain": 0.2,
        "coolant_temp": 92.0, "oil_temp": 92.0, "exhaust_temp": 520.0,
        "oil_pressure": 5.3e5, "oil_viscosity": 0.0133,
        "debris_rate": 1.0, "debris_cumulative": 300.0,
        "shaft_torque": 1750.0, "lambda": 3.0, "lambda_residual": 1.0,
        "hyd_pressure": 2.1e7, "hyd_leak_flow": 0.0,
        "susp_load_kN": 53.0, "susp_strain_ue": 62.0, "susp_compliance": 1.25,
        "torsion_twist_deg": 1.8, "torsion_cumulative_twist": 300.0,
        "shock_a_rms_g": 1.6, "spl_db": 110.0,
        "ae_event_rate": 2.0, "ae_energy": 0.5,
        "vib_rms": 0.46, "vib_kurtosis": 2.0, "vib_dom_amp": 33.0,
        "driveline_efficiency": 1.0, "coolant_level": 0.95,
        "fuel_level": 0.9, "exhaust_pressure": 1.29e5,
    }
    base.update(over)
    return base


class TestBlockChainRuns(unittest.TestCase):

    def setUp(self):
        self.p = PHMPipeline(flash_file=str(ROOT / "results" / "_test_flash.jsonl"))

    def test_every_block_produces_its_output(self):
        out = self.p.process(_frame())
        for key in ("clean_telemetry", "sensor_faults", "subsystem_health",
                    "prognosis", "dtcs_active", "dm1_hex", "dm2_hex"):
            self.assertIn(key, out, key)

    def test_health_is_computed_not_echoed_from_the_input(self):
        """The HUD used to receive `latest.get("composite_chi", 95.0)` -- i.e.
        whatever the client claimed its own health was."""
        out = self.p.process(_frame(composite_chi=12.0))
        health = out["subsystem_health"]
        self.assertIsNotNone(health)
        self.assertNotAlmostEqual(health["overall"], 12.0, places=3)
        self.assertGreater(health["overall"], 50.0,
                           "a nominal frame must not score as unhealthy")

    def test_no_subsystem_health_value_is_a_hardcoded_literal(self):
        """The broadcast vector was `[chi, chi, chi, 98, 99, 95, 96, 99]`.

        Driving one subsystem's inputs to a fault must move that subsystem and
        leave the unrelated ones alone -- impossible if any are constants.
        """
        nominal = self.p.process(_frame())["subsystem_health"]
        hot = PHMPipeline(flash_file=str(ROOT / "results" / "_test_flash.jsonl"))
        hot_health = hot.process(_frame(coolant_temp=125.0))["subsystem_health"]
        self.assertLess(hot_health["cooling"], nominal["cooling"] - 5.0,
                        "cooling health did not respond to a coolant excursion")
        self.assertAlmostEqual(hot_health["suspension"], nominal["suspension"],
                               delta=1.0,
                               msg="an unrelated subsystem moved")

    def test_a_sensor_fault_reaches_the_dtc_engine(self):
        """End of the chain: gate detection -> SPN/FMI -> active DM1."""
        self.p.process(_frame())
        out = self.p.process(_frame(step=1, oil_pressure=float("nan")))
        self.assertTrue(out["sensor_faults"],
                        "gate did not flag a non-finite oil pressure")
        self.assertTrue(out["dtcs_active"],
                        "gate fault never became a DTC")
        self.assertGreater(len(out["dm1_hex"]), 0)

    def test_gate_sanitises_before_health_is_scored(self):
        """Order matters: health must see clean values, not raw ones."""
        out = self.p.process(_frame(coolant_temp=float("inf")))
        self.assertTrue(all(v == v for v in out["clean_telemetry"].values()
                            if isinstance(v, float)),
                        "NaN/Inf survived the gate into clean_telemetry")
        self.assertIsNotNone(out["subsystem_health"])

    def test_units_are_normalised_so_pressures_do_not_trip_the_envelope(self):
        """SI input must not read as an electrical fault.

        oil_pressure 5.3e5 Pa against a [0, 15] bar envelope raised a
        SHORT_CIRCUIT on every frame and clamped to 15.0.
        """
        out = self.p.process(_frame())
        kinds = {f.get("fault_type") for f in out["sensor_faults"]}
        self.assertNotIn("SHORT_CIRCUIT", kinds,
                         f"nominal frame raised {kinds}")
        self.assertLess(out["clean_telemetry"]["oil_pressure"], 20.0,
                        "oil_pressure was not converted to bar")

    def test_inference_reports_its_own_availability(self):
        """A missing model must not be papered over with a fabricated number."""
        out = self.p.process(_frame())
        self.assertIn("inference_available", out)
        if not out["inference_available"]:
            self.assertIsNone(out["prognosis"])

    def test_prognosis_appears_once_the_window_fills(self):
        if not self.p.inference.available:
            self.skipTest("trained artifacts not present")
        window = self.p.inference.window
        out = None
        for i in range(window + 2):
            out = self.p.process(_frame(step=i, time=i * 0.05))
        self.assertIsNotNone(out["prognosis"],
                             "no prognosis after the window filled")
        probs = out["prognosis"]["fault_probs"]
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
        self.assertEqual(len(out["prognosis"]["rul_fraction"]), 8)

    def tearDown(self):
        f = ROOT / "results" / "_test_flash.jsonl"
        if f.exists():
            f.unlink()


if __name__ == "__main__":
    unittest.main()
