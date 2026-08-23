"""Signal- and label-integrity regression tests for the simulator.

Every test here guards a defect that previously shipped while the suite stayed
green: channels that carried no information, health curves that tracked elapsed
time instead of condition, and training labels that were constant.
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ml.parts import (FAIL_HEALTH, INPUT_FEATURES, PART_ORDER,
                      overall_health_index, part_health_index)
from ml.scenarios import (Scenario, build_scaler, feature_matrix, normalise,
                          part_health_series, rul_labels, run_scenario)
from tank_sim.config import TankConfig
from tank_sim.faults import FaultManager
from tank_sim.physics import AcousticSensor, TorqueSensor, VibrationSensor
from tank_sim.physics.vibration import characteristic_frequencies
from tank_sim.tank import TankSimulator


class TestVibrationCarriesSeverity(unittest.TestCase):
    """vib_rms used to be pinned to ~0.36-0.40 across the whole severity range
    by a renormalisation, making its warn/crit thresholds unreachable."""

    def setUp(self):
        self.cfg = TankConfig()
        self.sensor = VibrationSensor(self.cfg, np.random.default_rng(1))

    def test_rms_grows_monotonically_with_severity(self):
        vals = [self.sensor.features(1800.0, s, "bearing_outer")["vib_rms"]
                for s in (0.25, 0.5, 1.0, 2.0)]
        for a, b in zip(vals, vals[1:]):
            self.assertGreater(b, a)

    def test_rms_can_reach_the_configured_critical_threshold(self):
        crit = next(p["crit_hi"] for p in
                    __import__("ml.parts", fromlist=["PARTS"]).PARTS["powertrain"]["params"]
                    if p["key"] == "vib_rms")
        worst = self.sensor.features(1800.0, 2.0, "bearing_outer")["vib_rms"]
        self.assertGreater(worst, crit)

    def test_gear_wear_also_reaches_warning(self):
        """Gear wear peaked at 0.72 against a 0.75 warn threshold, so it could
        never raise an alarm regardless of severity."""
        v = self.sensor.features(1800.0, 1.0, "gear_wear")["vib_rms"]
        self.assertGreater(v, 0.75)

    def test_defect_signature_appears_at_its_characteristic_frequency(self):
        """The burst rate must clear Nyquist for the gear-mesh tone, otherwise
        the signature folds back and dom_freq just reports shaft rate."""
        f = characteristic_frequencies(self.cfg, 1800.0)
        self.assertGreater(self.cfg.sample_rate / 2.0, f["f_gmf"])

        bearing = self.sensor.features(1800.0, 1.5, "bearing_outer")["vib_dom_freq"]
        self.assertAlmostEqual(bearing, f["BPFO"], delta=8.0)

        gear = self.sensor.features(1800.0, 1.5, "gear_wear")["vib_dom_freq"]
        self.assertAlmostEqual(gear, f["f_gmf"], delta=8.0)


class TestAcousticsCarriesInformation(unittest.TestCase):
    """acoustic_dom_freq was a hardcoded 120 Hz tone: one unique value across
    an entire run, and SPL never responded to any fault."""

    def setUp(self):
        self.cfg = TankConfig()
        self.sensor = AcousticSensor(self.cfg, np.random.default_rng(0))

    def test_dominant_frequency_tracks_engine_speed(self):
        seen = set()
        for rpm in (800.0, 1500.0, 2600.0):
            f = self.sensor.features(rpm)["acoustic_dom_freq"]
            expected = rpm / 60.0 * self.cfg.cylinders / 2.0
            self.assertAlmostEqual(f, expected, delta=max(8.0, expected * 0.05))
            seen.add(round(f))
        self.assertEqual(len(seen), 3, "dom_freq must vary with rpm")

    def test_spl_responds_to_mechanical_wear(self):
        quiet = self.sensor.features(1800.0, mech_severity=0.0)["spl_db"]
        loud = self.sensor.features(1800.0, mech_severity=2.0)["spl_db"]
        self.assertGreater(loud, quiet + 5.0)


class TestDrivetrainDirection(unittest.TestCase):
    """A reduction gearbox multiplies torque; efficiency loss must reduce what
    reaches the sprocket, not raise it."""

    def test_efficiency_loss_reduces_delivered_torque(self):
        cfg = TankConfig()
        s = TorqueSensor(cfg, np.random.default_rng(0))
        good = s.read(2000.0, 800_000.0, efficiency=1.0)
        bad = s.read(2000.0, 800_000.0, efficiency=0.75)
        self.assertLess(bad["sprocket_torque"], good["sprocket_torque"])
        self.assertLess(bad["delivered_power"], good["delivered_power"])

    def test_gearbox_multiplies_torque_downstream(self):
        cfg = TankConfig()
        s = TorqueSensor(cfg, np.random.default_rng(0))
        r = s.read(2000.0, 800_000.0, efficiency=1.0)
        self.assertGreater(r["sprocket_torque"], r["shaft_torque"])

    def test_peak_shaft_power_is_physically_plausible(self):
        """Fuel heat release was previously reported as shaft power, giving
        ~2.4 MW (~3,200 hp) for a 58 t MBT."""
        cfg = TankConfig()
        peak_w = cfg.max_fuel_energy_rate * cfg.brake_thermal_efficiency
        self.assertLess(peak_w / 745.7, 1800.0)
        self.assertGreater(peak_w / 745.7, 900.0)


class TestSimulatorIsIdempotent(unittest.TestCase):
    def test_two_runs_produce_identical_records(self):
        cfg = TankConfig()
        cfg.window_samples = 32
        sim = TankSimulator(cfg, faults=FaultManager(np.random.default_rng(0)), seed=0)
        a, b = sim.run(), sim.run()
        self.assertEqual(len(a), len(b))
        for key in ("coolant_temp", "fuel_level", "debris_cumulative",
                    "torsion_cumulative_twist"):
            self.assertAlmostEqual(a[0][key], b[0][key], places=9, msg=key)
            self.assertAlmostEqual(a[-1][key], b[-1][key], places=9, msg=key)


class TestHealthReflectsConditionNotClock(unittest.TestCase):
    def setUp(self):
        self.records, _ = run_scenario(
            Scenario("healthy", [], seed=100, steps=1200), 64, 4000.0)
        self.health = part_health_series(self.records)

    def test_healthy_run_starts_near_full_health(self):
        for part in PART_ORDER:
            self.assertGreater(self.health[part][0], 70.0, part)

    def test_healthy_run_never_crosses_the_failure_threshold(self):
        """Cumulative counters and fuel burn used to drag health down over
        time, so a fault-free mission still 'failed'."""
        for part in PART_ORDER:
            self.assertGreater(self.health[part].min(), FAIL_HEALTH, part)

    def test_healthy_run_has_no_failure_rul_targets(self):
        Y = rul_labels(self.health)
        self.assertTrue(np.allclose(Y, 1.0),
                        "a fault-free scenario must carry full RUL everywhere")

    def test_deviation_is_one_sided(self):
        """Being better than the healthy reference must not be penalised."""
        good = part_health_index("lubrication", {"oil_pressure": 6.0e5,
                                                 "oil_temp": 50.0,
                                                 "debris_rate": 1.0})
        self.assertAlmostEqual(good, 100.0, places=6)

    def test_overall_is_fused_from_subsystems(self):
        self.assertLess(overall_health_index({"engine": 10.0, "cooling": 100.0}), 60.0)
        self.assertAlmostEqual(overall_health_index({"engine": 100.0,
                                                     "cooling": 100.0}), 100.0)


class TestFaultsDriveHealthDown(unittest.TestCase):
    CASES = [
        ("bearing_wear", ("powertrain", "lubrication")),
        ("cooling_failure", ("cooling", "engine")),
        ("structural_crack", ("structure",)),
        ("torsion_fatigue", ("structure",)),
    ]

    def test_each_fault_degrades_its_subsystem(self):
        for fault, parts in self.CASES:
            records, _ = run_scenario(
                Scenario(fault, [(fault, 0.30)], seed=100, steps=2400), 64, 4000.0)
            health = part_health_series(records)
            worst = min(health[p].min() for p in parts)
            self.assertLess(worst, FAIL_HEALTH,
                            f"{fault} should drive {parts} below {FAIL_HEALTH}")


class TestScalerSpansTheSuite(unittest.TestCase):
    """The scaler was fitted on the first (healthy) scenario alone, leaving
    debris_rate and ae_event_rate with min == max so every fault value
    saturated to a constant 1.0."""

    # One representative scenario per fault family, so every input channel is
    # exercised by something.  A channel only moves under the fault that drives
    # it -- ae_event_rate needs an AE fault, driveline_efficiency needs a
    # drivetrain fault -- so a narrower set would prove nothing.
    COVERING_SUITE = (
        ("healthy", []),
        ("bearing", [("bearing_wear", 0.2)]),
        ("crack", [("structural_crack", 0.2)]),
        ("torsion", [("torsion_fatigue", 0.2)]),
        ("driveline", [("drivetrain_efficiency_loss", 0.2)]),
        ("cooling", [("cooling_failure", 0.2)]),
        ("hydraulic", [("hydraulic_valve_fault", 0.2)]),
    )

    def _suite_features(self):
        mats = {}
        for name, faults in self.COVERING_SUITE:
            records, _ = run_scenario(
                Scenario(name, faults, seed=100, steps=600), 64, 4000.0)
            mats[name] = feature_matrix(records)
        return mats

    def test_no_channel_is_degenerate_across_the_suite(self):
        mats = self._suite_features()
        scaler = build_scaler(np.concatenate(list(mats.values())))
        degenerate = [k for k in INPUT_FEATURES
                      if scaler[k]["max"] - scaler[k]["min"] < 1e-9]
        self.assertEqual(degenerate, [], f"degenerate channels: {degenerate}")

    def test_fault_values_are_not_all_clipped(self):
        """Each diagnostic channel must retain dynamic range under the fault
        that actually drives it."""
        mats = self._suite_features()
        scaler = build_scaler(np.concatenate(list(mats.values())))
        for key, scenario in (("debris_rate", "bearing"),
                              ("ae_event_rate", "crack"),
                              ("vib_rms", "bearing"),
                              ("driveline_efficiency", "driveline")):
            j = INPUT_FEATURES.index(key)
            norm = normalise(mats[scenario], scaler)
            saturated = float(np.mean((norm[:, j] <= 0.0) | (norm[:, j] >= 1.0)))
            self.assertLess(saturated, 0.99,
                            f"{key} is constant under {scenario} after normalisation")


if __name__ == "__main__":
    unittest.main()
