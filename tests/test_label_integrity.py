"""Label-integrity regression tests.

Each test here guards a defect that previously shipped while the existing test
suite stayed green.  They assert provenance and independence of labels, not just
the presence of columns.
"""

import unittest

import numpy as np
import pandas as pd

from benchmark.evaluate_subsystems import assert_no_label_leakage
from pipelines.apu_metropt import (METROPT3_FAILURES, failure_windows,
                                   label_from_reports, load_metropt_episodes,
                                   feature_columns as metropt_features)
from pipelines.engine_deutz import (LABEL_DEFINING_COLUMNS, LABEL_PROVENANCE,
                                    LAMBDA_PHYSICAL_MAX, load_deutz_nrtc_data,
                                    load_deutz_residuals)
from pipelines.hydraulics_zema import (load_zema_hydraulic_data,
                                       stratified_cycle_sample,
                                       degradation_ordered_sample)
from pipelines.naval_gasturbine import (TARGET_NAMES as NAVAL_TARGETS,
                                        feature_columns as naval_features,
                                        health_from_decay,
                                        load_naval_propulsion_data)


class TestMetroPTLabelsComeFromReports(unittest.TestCase):
    """Labels must derive from the company failure reports, not from a
    percentile of the motor-current channel they are predicted from."""

    def test_failure_windows_match_data_description(self):
        w = failure_windows()
        self.assertEqual(len(w), 4)
        self.assertEqual(w.loc[0, "start"], pd.Timestamp("2020-04-18 00:00:00"))
        self.assertEqual(w.loc[3, "end"], pd.Timestamp("2020-07-15 19:00:00"))
        self.assertTrue((w["mode"] == "air_leak").all())

    def test_labels_are_a_pure_function_of_timestamps(self):
        """Same timestamps -> same labels regardless of any sensor value."""
        ts = pd.Series(pd.date_range("2020-04-17 12:00", periods=500, freq="10s"))
        a = label_from_reports(ts)
        b = label_from_reports(ts)
        pd.testing.assert_frame_equal(a, b)
        # A timestamp inside window #1 is positive; one a week earlier is not.
        inside = label_from_reports(pd.Series([pd.Timestamp("2020-04-18 12:00")]))
        outside = label_from_reports(pd.Series([pd.Timestamp("2020-04-11 12:00")]))
        self.assertEqual(int(inside["apu_failure"].iloc[0]), 1)
        self.assertEqual(int(outside["apu_failure"].iloc[0]), 0)

    def test_positive_rate_is_not_a_quantile_artifact(self):
        """The old label was Motor_current > q95, which pins positives at
        exactly 5.00%. A report-derived rate must not land on that value."""
        df = load_metropt_episodes()
        rate = df["apu_failure"].mean()
        self.assertNotAlmostEqual(rate, 0.05, places=4)
        self.assertGreater(df["apu_failure"].sum(), 0)

    def test_all_four_episodes_present_and_contiguous(self):
        df = load_metropt_episodes()
        self.assertEqual(sorted(df["episode"].unique()),
                         [w["id"] for w in METROPT3_FAILURES])
        for ep, g in df.groupby("episode"):
            self.assertTrue(g["timestamp"].is_monotonic_increasing,
                            f"episode {ep} is not contiguous in time")

    def test_feature_columns_exclude_every_label(self):
        df = load_metropt_episodes()
        feats = metropt_features(df)
        for banned in ("apu_failure", "apu_prefailure", "apu_system_fault",
                       "failure_id", "episode", "timestamp"):
            self.assertNotIn(banned, feats)


class TestDeutzDeclaresHeuristicProvenance(unittest.TestCase):
    """Deutz ships no fault labels; the derived indicator must say so."""

    def test_provenance_is_declared_heuristic(self):
        self.assertEqual(LABEL_PROVENANCE, "heuristic_derived")
        df = load_deutz_nrtc_data()
        self.assertEqual(df.attrs.get("label_provenance"), "heuristic_derived")

    def test_label_defining_columns_are_named(self):
        """The circular channels must be enumerated so consumers can drop them."""
        self.assertIn("lambda", LABEL_DEFINING_COLUMNS)
        self.assertIn("engine_power_kw", LABEL_DEFINING_COLUMNS)

    def test_leakage_guard_rejects_label_defining_features(self):
        df = load_deutz_nrtc_data()
        all_numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        with self.assertRaises(ValueError):
            assert_no_label_leakage(all_numeric, LABEL_DEFINING_COLUMNS, context="test")

    def test_lambda_is_masked_to_physical_range(self):
        """Raw lambda diverges to ~1.2e8 at zero fuelling; the masked copy must not."""
        df = load_deutz_nrtc_data()
        self.assertGreater(df["lambda"].max(), 1e6)  # the raw pathology still visible
        self.assertLessEqual(df["lambda_phys"].max(), LAMBDA_PHYSICAL_MAX)
        self.assertEqual(int(df.loc[df["lambda"] > LAMBDA_PHYSICAL_MAX, "lambda_valid"].sum()), 0)

    def test_residuals_align_bench_to_cfd(self):
        r = load_deutz_residuals()
        self.assertGreater(len(r.attrs["shared_channels"]), 20)
        self.assertEqual(r.attrs["label_provenance"], "model_residual_unsupervised")
        for c in r.attrs["shared_channels"][:5]:
            self.assertIn(f"{c}_resid", r.columns)
            np.testing.assert_allclose(
                r[f"{c}_resid"].values,
                r[f"{c}_meas"].values - r[f"{c}_sim"].values, rtol=1e-9)


class TestZeMASamplingIsNotHeadSliced(unittest.TestCase):
    """The cycles are ordered by experimental block, so a head slice silently
    selects a single condition and removes the degradation being demonstrated."""

    def setUp(self):
        self.df = load_zema_hydraulic_data()

    def test_head_slice_is_degenerate(self):
        """Documents why stratified sampling exists; if this ever stops being
        true the guard below can be relaxed."""
        head = self.df.iloc[:200]
        self.assertEqual(len(head["valve_condition"].unique()), 1)

    def test_stratified_sample_spans_all_condition_levels(self):
        s = stratified_cycle_sample(self.df, 200, "valve_condition")
        self.assertEqual(len(s), 200)
        self.assertEqual(sorted(s["valve_condition"].unique()),
                         sorted(self.df["valve_condition"].unique()))

    def test_degradation_ordered_sample_is_monotonic(self):
        s = degradation_ordered_sample(self.df, 200, "valve_condition")
        vals = s["valve_condition"].values
        self.assertTrue(np.all(np.diff(vals) <= 0), "sequence must degrade monotonically")
        self.assertGreater(len(np.unique(vals)), 1)

    def test_stratified_sample_is_reproducible(self):
        a = stratified_cycle_sample(self.df, 120, "valve_condition", random_state=7)
        b = stratified_cycle_sample(self.df, 120, "valve_condition", random_state=7)
        pd.testing.assert_frame_equal(a, b)


class TestNavalGasTurbineTargets(unittest.TestCase):
    """Decay coefficients are the experiment's independent variable, so they
    cannot be circular with the measured channels."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_naval_propulsion_data()

    def test_targets_are_continuous_and_multi_level(self):
        for t in NAVAL_TARGETS:
            self.assertIn(t, self.df.columns)
            self.assertGreater(self.df[t].nunique(), 20, t)
            self.assertLessEqual(self.df[t].max(), 1.0)

    def test_features_exclude_every_target(self):
        feats = naval_features(self.df)
        for t in NAVAL_TARGETS:
            self.assertNotIn(t, feats)
        self.assertNotIn("compressor_degraded", feats)
        self.assertNotIn("turbine_degraded", feats)

    def test_leakage_guard_rejects_targets(self):
        with self.assertRaises(ValueError):
            assert_no_label_leakage(list(self.df.columns), NAVAL_TARGETS,
                                    context="naval")

    def test_health_maps_decay_onto_zero_to_hundred(self):
        h = health_from_decay(self.df, "gt_compressor_decay")
        self.assertGreaterEqual(h.min(), 0.0)
        self.assertLessEqual(h.max(), 100.0)
        # A pristine machine (coefficient 1.0) must read full health.
        fresh = self.df["gt_compressor_decay"].max()
        self.assertAlmostEqual(
            float(h[self.df["gt_compressor_decay"].values == fresh][0]), 100.0, places=6)


class TestAegisCanIsDeclaredUnlabelled(unittest.TestCase):
    """The trace has diagnostic lamp channels; whether any fires decides
    whether it is a fault corpus or only a duty-cycle source."""

    def setUp(self):
        try:
            import h5py  # noqa: F401
        except ImportError:
            self.skipTest("h5py not installed")
        from pipelines.can_aegis import find_trips
        if not find_trips():
            self.skipTest("no AEGIS trips procured")

    def test_lamp_channels_are_checked_not_assumed(self):
        from pipelines.can_aegis import lamp_activity
        la = lamp_activity()
        self.assertGreater(len(la), 0, "no lamp channels inspected")
        self.assertIn("active_samples", la.columns)
        # Provenance must agree with what the data actually shows.
        self.assertEqual(bool(la.attrs["has_faults"]),
                         bool(la["active_samples"].sum() > 0))

    def test_signal_values_are_physical_not_a_timebase(self):
        """Datasets are (N, 2) with value in column 0 and time in column 1.
        Reading them the other way round yields a monotonic ramp for every
        channel, which is silently plausible-looking nonsense."""
        from pipelines.can_aegis import load_can_trip
        df = load_can_trip()
        rpm = df["rpm"].dropna()
        self.assertGreater(rpm.min(), 100.0)
        self.assertLess(rpm.max(), 9000.0)
        self.assertFalse(rpm.is_monotonic_increasing,
                         "rpm looks like a timebase, not a measurement")
        coolant = df["coolant_temp"].dropna()
        self.assertTrue(20.0 < coolant.median() < 130.0)


class TestLeakageGuard(unittest.TestCase):
    def test_passes_when_clean(self):
        assert_no_label_leakage(["a", "b"], ("target",), context="clean")

    def test_raises_and_names_the_leaked_columns(self):
        with self.assertRaises(ValueError) as ctx:
            assert_no_label_leakage(["a", "target"], ("target",), context="dirty")
        self.assertIn("target", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
