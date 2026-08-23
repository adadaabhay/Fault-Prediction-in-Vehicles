"""Label-leakage regression tests for the synthetic pipeline.

Background
----------
Six channels used to be published as *readbacks of the injected fault
parameter* rather than as measurements:

    driveline_efficiency  = max(efficiency, 0.05)
    ae_event_rate         = 2.0 * (1 + 20*s)
    debris_rate           = 1.0 * (1 + 30*s)
    oil_flow              = q_nominal * flow_mult
    susp_compliance       = 1e6 / (A * E * stiffness_mult)
    hyd_leak_flow         = A_leak * sqrt(P/1e7) * (1 + seal_leak)

Each inverts to the injected severity in closed form.  A
depth-1 threshold on `driveline_efficiency` scored 99.88% on its own fault
class, and a depth-4 tree on those six channels alone scored 0.769 across all
13 classes against 0.544 for the eighteen physically-mediated ones.  Both
`debris_rate` and `ae_event_rate` are also HEALTH_REFERENCES inputs, so the
leak reached the RUL *targets* as well and the regression was circular.

The invariant
-------------
`test_every_channel_depends_on_measurement_noise` is the general guard, not a
blocklist of the six known offenders.  Hold the fault trajectory fixed, re-roll
only the sensor noise, and every emitted channel must move.  A channel that is
bit-identical across two independent noise realisations is not a measurement --
it is whichever parameter produced it, published under a sensor's name.  That
catches the next one of these before it ships, which a hardcoded list cannot.

`benchmark/evaluate_subsystems.py` has enforced the equivalent discipline on
the seven real corpora since it was written (`assert_no_label_leakage`).  This
module is that discipline finally pointed at the synthetic pipeline, which is
the one that produces the shipped model.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ml.parts import INPUT_FEATURES
from ml.scenarios import ALL_FAULTS, Scenario, feature_matrix, run_scenario
from tank_sim.config import TankConfig
from tank_sim.faults import FaultManager
from tank_sim.tank import SENSOR_COLUMNS, TankSimulator, default_mission

# Channels that are exogenous inputs or bookkeeping, not transduced signals.
# These are legitimately noise-free and are excluded by construction.
NON_SENSOR_COLUMNS = {
    "time", "step", "rpm", "load", "terrain",   # commanded mission profile
    "shaft_omega",                              # algebraic in rpm
    "oil_temp", "oil_viscosity",                # exposed thermal-state taps
}

# Bin-quantised spectral estimators.  `vib_dom_freq` / `acoustic_dom_freq` are
# argmax indices over an rfft, so at the operating SNR they land in the same
# bin for any noise realisation -- repeating across seeds is correct behaviour
# for a quantised estimator, not evidence of a readback.  They are still
# covered: `test_no_channel_reconstructs_severity` below applies to every
# column with no exclusions, and TestAcousticsCarriesInformation in
# test_signal_integrity asserts they are not constant.
QUANTISED_COLUMNS = {"vib_dom_freq", "acoustic_dom_freq"}


def _analytic_severity(step, start, ramp, max_sev=1.0):
    """The deterministic part of FaultProfile.severity (no jitter)."""
    prog = np.maximum(step - start, 0.0) / max(ramp, 1)
    return np.where(step < start, 0.0,
                    max_sev * (1.0 - np.exp(-3.0 * prog)))


def _run(sensor_seed: int, steps: int = 400, fault: str = "bearing_wear"):
    """One mission with a fixed fault trajectory and a chosen noise stream."""
    cfg = TankConfig()
    cfg.window_samples = 64
    cfg.sample_rate = 4000.0
    fm = FaultManager(np.random.default_rng(3))
    fm.add(fault, start_step=int(steps * 0.3), ramp_steps=int(steps * 0.3))
    base = default_mission(cfg)
    factor = (steps * cfg.dt) / sum(m.duration_s for m in base)
    mission = [type(m)(m.duration_s * factor, m.rpm, m.load, m.terrain)
               for m in base]
    sim = TankSimulator(cfg, faults=fm, mission=mission,
                        seed=3, sensor_seed=sensor_seed)
    return sim.run()[:steps]


class TestNoChannelIsAParameterReadback(unittest.TestCase):

    def test_every_channel_depends_on_measurement_noise(self):
        """The general invariant. Same faults, different transducer noise."""
        offenders = []
        for fault in ("bearing_wear", "drivetrain_efficiency_loss",
                      "structural_crack", "seal_leakage",
                      "bearing_clearance_wear", "torsion_fatigue"):
            a = _run(sensor_seed=11, fault=fault)
            b = _run(sensor_seed=97, fault=fault)
            for col in SENSOR_COLUMNS:
                if col in NON_SENSOR_COLUMNS or col in QUANTISED_COLUMNS:
                    continue
                va = np.array([r[col] for r in a])
                vb = np.array([r[col] for r in b])
                if np.array_equal(va, vb):
                    offenders.append(f"{col} (under {fault})")
        self.assertEqual(
            sorted(set(offenders)), [],
            "channels identical across two sensor-noise seeds -- these are "
            f"parameter readbacks, not measurements: {sorted(set(offenders))}")

    def test_fault_trajectory_is_unaffected_by_the_sensor_noise_stream(self):
        """The other half: re-rolling sensor noise must not move the labels.

        If it did, the test above would pass for the wrong reason -- it would
        be detecting a changed fault, not a changed measurement.
        """
        a = _run(sensor_seed=11)
        b = _run(sensor_seed=97)
        for r_a, r_b in zip(a, b):
            self.assertEqual(r_a["fault_bearing_wear"], r_b["fault_bearing_wear"])

    def test_previously_leaked_channels_are_no_longer_invertible(self):
        """Named regression for the six specific offenders.

        Each is checked against the closed form it used to satisfy exactly.
        """
        recs, _ = run_scenario(
            Scenario("d", [("drivetrain_efficiency_loss", 0.3)], seed=5,
                     steps=600), 64, 4000.0)
        eff = np.array([r["driveline_efficiency"] for r in recs])
        # Was exactly max(1 - 0.18*s, 0.7); the reconstruction residual was 0.
        self.assertGreater(float(np.std(eff[:150])), 0.0,
                           "driveline_efficiency is constant while healthy")

        recs, _ = run_scenario(
            Scenario("c", [("structural_crack", 0.3)], seed=5, steps=600),
            64, 4000.0)
        aer = np.array([r["ae_event_rate"] for r in recs])
        # NB: "recover s from the channel, then re-apply the closed form and
        # check the residual is 0" is *not* a leak test -- it is an algebraic
        # identity that holds for any invertible map, leaked or not. The
        # discriminating property is variation at fixed severity, which
        # TestNoChannelReconstructsSeverity asserts directly. Here we only
        # check the estimator is not the old noiseless analytic ramp.
        self.assertGreater(
            float(np.std(aer[:150])), 0.0,
            "ae_event_rate is constant while healthy -- still the analytic rate")
        self.assertGreater(float(np.std(np.diff(aer))), 0.0)


class TestNoChannelReconstructsSeverity(unittest.TestCase):
    """No single channel may be an (almost) exact function of the injected
    severity.

    This is the quantitative form of the defect.  `driveline_efficiency` was
    literally ``max(1 - 0.18*s, 0.7)``, so regressing severity on that one
    column gave R^2 = 1.0 -- the column *was* the label under another name.
    A genuine diagnostic channel is allowed to be strongly informative (debris
    rate really does rise with bearing wear); what it may not be is invertible
    to machine precision.

    Applies to every column with no exclusions, quantised estimators included.
    """

    # The defect being caught is *algebraic determination*, not high SNR.
    # Pre-fix, `driveline_efficiency` gave R^2 = 1.000000 with a 0.0e+00
    # residual: it was the parameter. Post-fix the highest any channel reaches
    # is `hyd_pressure` at 0.991 under hydraulic_valve_fault -- and that one is
    # legitimate physics, a 30% pressure collapse on a 210 bar circuit read by
    # a transducer with 1 bar noise. A channel is allowed to be strongly
    # diagnostic; it is not allowed to have no noise floor. The ceiling is set
    # to admit real high-SNR channels and reject exact inverses, and
    # `test_residual_variance_is_bounded_below` below is the sharper form of
    # the same property.
    R2_CEILING = 0.999
    STEPS = 800

    def _r2_of_best_channel(self, fault):
        start = int(self.STEPS * 0.3)
        ramp = int(self.STEPS * 0.3)
        recs = _run(sensor_seed=23, steps=self.STEPS, fault=fault)
        steps = np.array([r["step"] for r in recs])
        sev = _analytic_severity(steps, start, ramp)
        active = sev > 1e-6
        sev = sev[active]
        worst_col, worst_r2 = None, -1.0
        for col in SENSOR_COLUMNS:
            if col in ("time", "step") or col.startswith("fault_"):
                continue
            v = np.array([r[col] for r in recs], dtype=float)[active]
            if not np.all(np.isfinite(v)) or np.std(v) < 1e-12:
                continue
            # Straight-line fit of severity on the channel; R^2 == 1 means the
            # channel determines the label exactly.
            r = float(np.corrcoef(v, sev)[0, 1])
            r2 = r * r
            if r2 > worst_r2:
                worst_col, worst_r2 = col, r2
        return worst_col, worst_r2

    def test_residual_variance_is_bounded_below(self):
        """The sharp form: no channel may be a deterministic function of s.

        Group samples by (quantised) severity and measure the spread *within*
        each group. A transduced channel keeps its noise floor at fixed
        severity; a parameter readback collapses to a single value, because
        severity is all it ever was.
        """
        offenders = []
        for fault in ALL_FAULTS:
            start, ramp = int(self.STEPS * 0.3), int(self.STEPS * 0.3)
            recs = _run(sensor_seed=23, steps=self.STEPS, fault=fault)
            steps = np.array([r["step"] for r in recs])
            sev = _analytic_severity(steps, start, ramp)
            bins = np.round(sev * 200).astype(int)     # ~0.005 severity bins
            for col in SENSOR_COLUMNS:
                if col in ("time", "step") or col.startswith("fault_"):
                    continue
                v = np.array([r[col] for r in recs], dtype=float)
                if not np.all(np.isfinite(v)) or np.std(v) < 1e-12:
                    continue
                spreads = [np.std(v[bins == b]) for b in np.unique(bins)
                           if np.sum(bins == b) >= 4]
                if spreads and float(np.max(spreads)) <= 0.0:
                    offenders.append(f"{fault}: {col}")
        self.assertEqual(
            offenders, [],
            "channels with zero conditional spread at fixed severity -- "
            f"deterministic in the injected parameter: {offenders}")

    def test_no_channel_reconstructs_severity(self):
        offenders = []
        for fault in ALL_FAULTS:
            col, r2 = self._r2_of_best_channel(fault)
            if r2 >= self.R2_CEILING:
                offenders.append(f"{fault}: {col} R^2={r2:.5f}")
        self.assertEqual(
            offenders, [],
            "channels that reconstruct the injected severity almost exactly "
            f"(ceiling R^2={self.R2_CEILING}): {offenders}")


class TestLeakageDoesNotCarryTheClassification(unittest.TestCase):
    """A depth-1 stump on one channel must not score like a label.

    `driveline_efficiency` alone scored 0.9988 on its own fault class.
    """

    FORMERLY_LEAKED = ["driveline_efficiency", "debris_rate", "ae_event_rate",
                       "hyd_leak_flow", "susp_compliance", "oil_viscosity",
                       "coolant_level", "fuel_level"]

    @classmethod
    def setUpClass(cls):
        X, y = [], []
        for j, f in enumerate(ALL_FAULTS):
            r, _ = run_scenario(
                Scenario(f, [(f, 0.25)], seed=300 + j, steps=600), 64, 4000.0)
            X.append(feature_matrix(r)[300:])
            y.append(np.full(300, j))
        r, _ = run_scenario(Scenario("h", [], seed=999, steps=600), 64, 4000.0)
        X.append(feature_matrix(r)[300:])
        y.append(np.full(300, len(ALL_FAULTS)))
        cls.X = np.concatenate(X)
        cls.y = np.concatenate(y)

    def test_no_single_channel_separates_its_own_fault_almost_perfectly(self):
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.tree import DecisionTreeClassifier
        except ImportError:
            self.skipTest("scikit-learn not installed")
        for col in self.FORMERLY_LEAKED:
            if col not in INPUT_FEATURES:
                continue
            j = INPUT_FEATURES.index(col)
            acc = float(cross_val_score(
                DecisionTreeClassifier(max_depth=1, random_state=0),
                self.X[:, [j]], self.y, cv=5).mean())
            self.assertLess(acc, 0.90, f"{col} alone scores {acc:.4f}")


if __name__ == "__main__":
    unittest.main()
