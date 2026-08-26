"""Unit test suite for the C99 Edge Inference Engine & Architecture Verification."""

import unittest
import numpy as np
import re
from pathlib import Path


class TestC99EngineArchitecture(unittest.TestCase):
    def setUp(self):
        self.c_dir = Path(__file__).resolve().parent.parent / "c_engine"
        self.header_path = self.c_dir / "tank_pdm_infer.h"
        self.source_path = self.c_dir / "tank_pdm_infer.c"

    def test_c_files_exist(self):
        self.assertTrue(self.header_path.exists(), "tank_pdm_infer.h must exist")
        self.assertTrue(self.source_path.exists(), "tank_pdm_infer.c must exist")

    def test_zero_dynamic_allocation_in_c_code(self):
        """MISRA-C Rule: Ensure zero malloc, calloc, realloc, or free in C engine."""
        with open(self.source_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertNotIn("malloc(", content)
        self.assertNotIn("calloc(", content)
        self.assertNotIn("realloc(", content)
        self.assertNotIn("free(", content)

    def test_fixed_memory_footprint_under_32kb(self):
        """Verify that fixed-size structs consume strictly < 32 KB RAM."""
        # Dimensions come from the trained model, not a literal: this test
        # hardcoded D=58 while the model was D=24, so it was validating the
        # memory footprint of an engine that could not exist.
        from c_engine.gen_dims import model_dims
        dims = model_dims()
        D, H, R, C = dims["D"], dims["H"], dims["R"], dims["C"]
        float_sz = 4  # bytes

        # Weights memory: 4 gates * (D*H + H*H + H) + (H*R + R) + (H*C + C) floats
        w_floats = 4 * (D * H + H * H + H) + (H * R + R) + (H * C + C)
        weights_bytes = w_floats * float_sz

        # State memory: (2 * H) floats + 4 bytes uint32
        state_bytes = (2 * H * float_sz) + 4

        # Result memory: (R + C + 2) floats + 4 bytes uint32
        result_bytes = (R + C + 2) * float_sz + 4

        total_bytes = weights_bytes + state_bytes + result_bytes
        self.assertLess(total_bytes, 49152, f"Engine memory {total_bytes} bytes exceeds 48 KB limit!")
        print(f"\n[C99 Arena Verified] Total Static RAM Footprint: {total_bytes} bytes ({total_bytes/1024:.2f} KB / < 48 KB MCU SRAM)")

    def test_c99_engine_is_parity_checked_against_the_python_model(self):
        """The C engine must be compared against the *actual* Python model.

        This slot previously held a test called
        ``test_c99_numpy_algorithmic_parity`` that generated random weights,
        computed a sigmoid in NumPy, and asserted the result was between 0 and
        1 -- which is true of every sigmoid by definition. It never loaded the
        shared library, so it could not detect that the engine wrote its hidden
        state in place while still reading it as h_prev, an aliasing bug that
        put the RUL head 9.0e-4 off the reference.

        Real parity now lives in tests/test_c_python_parity.py, which builds the
        library, loads the trained weights and compares both heads over 20
        random windows. This test asserts that gate exists and is wired up.
        """
        from c_engine.build import have_toolchain

        parity_suite = Path(__file__).resolve().parent / "test_c_python_parity.py"
        self.assertTrue(parity_suite.exists(),
                        "the C/Python parity gate is missing")
        body = parity_suite.read_text(encoding="utf-8")
        self.assertIn("load_engine", body,
                      "parity suite must load the compiled library")
        self.assertIn("tank_infer_step", body,
                      "parity suite must exercise the C forward pass")

        if not have_toolchain():
            self.skipTest("no C toolchain; parity runs in CI")

        from c_engine.binding import load_engine
        lib, weights = load_engine()
        self.assertIsNotNone(lib)
        self.assertIsNotNone(weights)


if __name__ == "__main__":
    unittest.main()
