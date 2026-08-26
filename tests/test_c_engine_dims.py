"""D/H/R/C are declared in three places (C header, ctypes binding, trained
model). They had drifted to D=58 against a model with D=24 -- the edge runtime
could not run the shipped model at all. All three must derive from config.json.
"""

import json
import re
import unittest
from pathlib import Path

from c_engine.gen_dims import MODEL_PATH, model_dims, write_dims_header

ROOT = Path(__file__).resolve().parent.parent


class TestDimensionsAreSingleSourced(unittest.TestCase):
    def test_dims_match_the_trained_model(self):
        dims = model_dims()
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(dims["D"], model["D"])
        self.assertEqual(dims["H"], model["H"])
        self.assertEqual(dims["R"], model["R"])
        self.assertEqual(dims["C"], model["C"])

    def test_generated_header_matches_config(self):
        text = write_dims_header().read_text(encoding="utf-8")
        dims = model_dims()
        pairs = (("TANK_INFER_D_FEATURES", "D"),
                 ("TANK_INFER_H_HIDDEN", "H"),
                 ("TANK_INFER_R_PARTS", "R"),
                 ("TANK_INFER_C_CLASSES", "C"))
        for macro, key in pairs:
            match = re.search(r"#define\s+" + macro + r"\s+(\d+)", text)
            self.assertIsNotNone(match, macro)
            self.assertEqual(int(match.group(1)), dims[key], macro)

    def test_infer_header_does_not_hardcode_dims(self):
        text = (ROOT / "c_engine" / "tank_pdm_infer.h").read_text(encoding="utf-8")
        self.assertIn("tank_pdm_dims.h", text)
        self.assertIsNone(
            re.search(r"#define\s+TANK_INFER_D_FEATURES\s+\d+", text),
            "dimensions must come from the generated header, not be hardcoded")

    def test_binding_matches_config(self):
        from c_engine import binding
        dims = model_dims()
        self.assertEqual(binding.D_FEATURES, dims["D"])
        self.assertEqual(binding.H_HIDDEN, dims["H"])
        self.assertEqual(binding.R_PARTS, dims["R"])
        self.assertEqual(binding.C_CLASSES, dims["C"])

    def test_generator_rejects_a_config_model_mismatch(self):
        """The drift this task removes must be detected, not silently accepted."""
        import c_engine.gen_dims as gd
        original = gd.CONFIG_PATH
        bogus = ROOT / "c_engine" / "_bogus_config.json"
        try:
            cfg = json.loads(original.read_text(encoding="utf-8"))
            cfg["input_features"] = list(cfg["input_features"]) + ["phantom_channel"]
            bogus.write_text(json.dumps(cfg), encoding="utf-8")
            gd.CONFIG_PATH = bogus
            with self.assertRaises(ValueError):
                gd.model_dims()
        finally:
            gd.CONFIG_PATH = original
            bogus.unlink(missing_ok=True)
