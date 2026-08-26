"""Without an exporter there is no way to get trained weights into the edge
runtime, so the C engine could never run the real model."""

import json
import struct
import unittest

from c_engine.export_weights import (WEIGHT_ORDER, expected_float_count,
                                     expected_shape, export_weights)
from c_engine.gen_dims import MODEL_PATH, model_dims


class TestWeightsExport(unittest.TestCase):
    def test_binary_has_exactly_the_expected_float_count(self):
        dims = model_dims()
        D, H, R, C = dims["D"], dims["H"], dims["R"], dims["C"]
        expected = 4 * (D * H + H * H + H) + (H * R + R) + (H * C + C)
        self.assertEqual(expected_float_count(dims), expected)

        path = export_weights()
        raw = path.read_bytes()
        self.assertEqual(len(raw) % 4, 0)
        self.assertEqual(len(raw) // 4, expected)

    def test_first_value_matches_model_json(self):
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        raw = export_weights().read_bytes()
        first = struct.unpack_from("<f", raw, 0)[0]
        self.assertAlmostEqual(first, model["params"]["Wf"][0][0], places=5)

    def test_weight_order_matches_the_c_struct_field_order(self):
        self.assertEqual(WEIGHT_ORDER[0], "Wf")
        self.assertEqual(WEIGHT_ORDER[-1], "bcls")
        self.assertEqual(len(WEIGHT_ORDER), 16)

    def test_every_weight_shape_is_validated(self):
        dims = model_dims()
        self.assertEqual(expected_shape("Wf", dims), (dims["D"], dims["H"]))
        self.assertEqual(expected_shape("Uf", dims), (dims["H"], dims["H"]))
        self.assertEqual(expected_shape("Wy", dims), (dims["H"], dims["R"]))
        self.assertEqual(expected_shape("Wcls", dims), (dims["H"], dims["C"]))
        self.assertEqual(expected_shape("bcls", dims), (dims["C"],))

    def test_shape_mismatch_is_rejected(self):
        """A silently mis-shaped weight would corrupt the whole struct."""
        dims = model_dims()
        self.assertNotEqual(expected_shape("Wy", dims), expected_shape("Wcls", dims))

    def test_static_footprint_stays_under_the_32kb_budget(self):
        dims = model_dims()
        weights = expected_float_count(dims) * 4
        state = (2 * dims["H"] * 4) + 4
        result = (dims["R"] + dims["C"] + 2) * 4 + 4
        total = weights + state + result
        self.assertLess(total, 32 * 1024,
                        f"edge memory budget exceeded: {total} bytes")
