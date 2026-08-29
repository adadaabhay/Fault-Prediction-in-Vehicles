"""Shipped artifacts must stay consistent with the code that generates them.

The dashboard once shipped a 22-feature model against a 24-feature schema, and
the edge runtime still declares dimensions that no longer match either. These
checks fail the build rather than letting a mismatched artifact deploy.
"""

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


class TestArtifactConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads((DOCS / "config.json").read_text(encoding="utf-8"))
        cls.model = json.loads((DOCS / "model.json").read_text(encoding="utf-8"))
        cls.streams = json.loads(
            (DOCS / "live_multi_streams.json").read_text(encoding="utf-8"))

    def test_model_dimensions_match_config(self):
        self.assertEqual(self.model["D"], len(self.cfg["input_features"]))
        self.assertEqual(self.model["R"], len(self.cfg["part_order"]))
        self.assertEqual(self.model["C"], len(self.cfg["class_names"]))

    def test_scaler_covers_every_input_feature(self):
        for key in self.cfg["input_features"]:
            self.assertIn(key, self.cfg["scaler"], key)

    def test_no_degenerate_scaler_channel(self):
        degenerate = [k for k, v in self.cfg["scaler"].items()
                      if v["max"] - v["min"] < 1e-9]
        self.assertEqual(degenerate, [], f"degenerate channels: {degenerate}")

    def test_every_dashboard_button_has_a_stream(self):
        html = (DOCS / "index.html").read_text(encoding="utf-8")
        for button in re.findall(r'data-stream="([^"]+)"', html):
            self.assertIn(button, self.streams["streams"], f"dead button: {button}")

    def test_measured_streams_declare_provenance(self):
        for sid, stream in self.streams["streams"].items():
            if not sid.startswith("real_"):
                continue
            meta = stream["meta"]
            self.assertIn("channels_measured", meta, sid)
            self.assertIn("channels_synthetic", meta, sid)
            self.assertIn("health_provenance", meta, sid)

    def test_health_arrays_match_record_counts(self):
        for sid, stream in self.streams["streams"].items():
            n = len(stream["records"])
            for part in self.cfg["part_order"]:
                self.assertIn(part, stream["health"], f"{sid}:{part}")
                self.assertEqual(len(stream["health"][part]), n, f"{sid}:{part}")

    def test_requirements_are_pinned(self):
        # A line counts as "pinned" if it declares a version bound --
        # either a hard pin (==X.Y.Z) or a bounded range with both
        # floor and ceiling (>=A,<B).  Bare ``>=A`` with no ceiling
        # is rejected because it lets any future major in.
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        missing_bounds: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            has_floor = ">=" in line or "==" in line
            has_ceiling = "<" in line or "==" in line
            if not (has_floor and has_ceiling):
                missing_bounds.append(line)
        self.assertEqual(missing_bounds, [],
                         f"dependencies without floor+ceiling: {missing_bounds}")

    def test_provenance_document_lists_every_corpus(self):
        doc = (ROOT / "docs" / "PROVENANCE.md").read_text(encoding="utf-8")
        for corpus in ("ZeMA", "MetroPT", "Scania", "Naval", "Deutz", "AEGIS"):
            self.assertIn(corpus, doc, f"{corpus} missing from PROVENANCE.md")

    def test_benchmark_results_declare_provenance_for_every_scored_row(self):
        results = json.loads(
            (ROOT / "results" / "subsystems_benchmark.json").read_text(encoding="utf-8"))
        for name, block in results["subsystems"].items():
            provenance = block["metrics"].get("label_provenance", "")
            self.assertTrue(provenance.startswith("ground_truth"),
                            f"{name} is scored without ground-truth provenance")
        for name, block in results["excluded_from_scoring"].items():
            self.assertTrue(block.get("reason"), f"{name} excluded without a reason")


if __name__ == "__main__":
    unittest.main()
