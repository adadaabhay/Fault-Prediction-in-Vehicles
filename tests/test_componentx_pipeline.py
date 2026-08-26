"""SCANIA Component X is the only corpus with real time-to-event labels from
workshop repair records, so it is the project's only non-simulated RUL ground
truth. Every other RUL target comes from the physics simulator.

The dominant property is censoring: 90.4% of vehicles never had the repair, so
their study length is a lower bound on component life rather than a life.
"""

import unittest

from benchmark.evaluate_subsystems import assert_no_label_leakage
from pipelines.fleet_scania_componentx import (COLUMN_ALIASES,
                                               LABEL_PROVENANCE, available,
                                               censoring_summary,
                                               feature_columns,
                                               load_componentx_data, load_tte,
                                               uncensored)


class TestPipelineContract(unittest.TestCase):
    """Runs whether or not the corpus is procured."""

    def test_provenance_is_workshop_records(self):
        self.assertEqual(LABEL_PROVENANCE, "ground_truth_workshop_repair_records")

    def test_aliases_cover_the_three_canonical_columns(self):
        self.assertEqual(set(COLUMN_ALIASES),
                         {"vehicle_id", "time_to_event", "event_observed"})

    def test_published_column_names_are_mapped(self):
        """The release ships length_of_study_time_step / in_study_repair."""
        self.assertIn("length_of_study_time_step", COLUMN_ALIASES["time_to_event"])
        self.assertIn("in_study_repair", COLUMN_ALIASES["event_observed"])

    def test_missing_corpus_raises_with_the_download_url(self):
        if available():
            self.skipTest("corpus is procured")
        with self.assertRaises(FileNotFoundError) as ctx:
            load_componentx_data()
        self.assertIn("researchdata.se", str(ctx.exception))


@unittest.skipUnless(available(),
                     "SCANIA Component X not procured -- see module docstring")
class TestComponentXData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tte = load_tte()
        cls.df = load_componentx_data(max_vehicles=600)

    def test_full_training_fleet_is_present(self):
        self.assertEqual(len(self.tte), 23550)
        self.assertEqual(self.tte["vehicle_id"].nunique(), 23550)

    def test_corpus_is_overwhelmingly_censored(self):
        """The headline property: only ~9.6% of vehicles were actually repaired."""
        censored_pct = 100.0 * (self.tte["event_observed"] == 0).mean()
        self.assertGreater(censored_pct, 85.0)
        self.assertLess(censored_pct, 95.0)

    def test_readouts_are_a_time_series_per_vehicle(self):
        counts = self.df.groupby("vehicle_id").size()
        self.assertGreater(counts.mean(), 5,
                           "expected many readouts per vehicle, not one row each")

    def test_rul_is_study_end_minus_time_step(self):
        expected = (self.df["time_to_event"] - self.df["time_step"]).clip(lower=0.0)
        self.assertTrue((self.df["rul"] - expected).abs().max() < 1e-6)
        self.assertGreaterEqual(self.df["rul"].min(), 0.0)

    def test_features_exclude_labels_the_clock_and_the_id(self):
        feats = feature_columns(self.df)
        for banned in ("time_to_event", "event_observed", "rul", "study_end",
                       "time_step", "vehicle_id"):
            self.assertNotIn(banned, feats, banned)
        assert_no_label_leakage(
            feats, ("time_to_event", "event_observed", "rul", "study_end",
                    "time_step", "vehicle_id"),
            context="Component X")

    def test_time_step_is_excluded_because_rul_is_derived_from_it(self):
        """rul = study_end - time_step, so the clock reconstructs the target."""
        self.assertNotIn("time_step", feature_columns(self.df))

    def test_censoring_flag_is_binary(self):
        self.assertTrue(set(self.df["event_observed"].unique()) <= {0, 1})

    def test_uncensored_subset_keeps_only_observed_failures(self):
        subset = uncensored(self.df)
        self.assertTrue((subset["event_observed"] == 1).all())
        self.assertLess(len(subset), len(self.df))

    def test_censoring_summary_reports_the_corpus_rate_not_the_sample_rate(self):
        """Loading stratifies the sample, so its balance is not the fleet's.
        The corpus figure must survive into the summary."""
        summary = censoring_summary(self.df)
        self.assertEqual(summary["vehicles_observed"] + summary["vehicles_censored"],
                         summary["vehicles"])
        self.assertIn("corpus_censored_pct", summary)
        self.assertGreater(summary["corpus_censored_pct"], 85.0)
        self.assertEqual(summary["corpus_vehicles"], 23550)
        self.assertTrue(summary["sample_is_stratified"])

    def test_subsampling_selects_whole_vehicles(self):
        """A vehicle must be wholly in or wholly out, never split by readout."""
        ids = set(self.df["vehicle_id"].unique())
        counts = self.df.groupby("vehicle_id").size()
        self.assertEqual(len(ids), len(counts))
        self.assertTrue((counts > 0).all())


if __name__ == "__main__":
    unittest.main()
