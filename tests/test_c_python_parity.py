"""The edge runtime and the Python model must agree numerically.

Without this gate a deployed C engine could silently disagree with everything
the project validated in Python -- and until now the C engine could not even
consume the shipped weights, because it declared D=58 against a D=24 model.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

from c_engine.build import have_toolchain, missing_tools
from c_engine.gen_dims import MODEL_PATH, model_dims

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# float32 C against float64 Python, accumulated over a 40-step window.
TOLERANCE = 1e-4
WINDOW = 40

_SKIP_REASON = "no C toolchain: missing " + ", ".join(missing_tools() or ["-"])


@unittest.skipUnless(have_toolchain(), _SKIP_REASON)
class TestCPythonParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from c_engine.binding import load_engine
        from ml.lstm import LSTMModel

        cls.lib, cls.weights = load_engine()
        cls.dims = model_dims()

        model_json = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        cls.py = LSTMModel(D=cls.dims["D"], H=cls.dims["H"],
                           R=cls.dims["R"], C=cls.dims["C"], seed=0)
        for name, value in model_json["params"].items():
            cls.py.p[name] = np.asarray(value, dtype=np.float64)

    def _run_c(self, window):
        import ctypes

        from c_engine.binding import TankInferenceResult, TankInferenceState

        state = TankInferenceState()
        result = TankInferenceResult()
        self.lib.tank_infer_reset(ctypes.byref(state))
        arr_t = ctypes.c_float * self.dims["D"]
        for row in window:
            x = arr_t(*[float(v) for v in row])
            status = self.lib.tank_infer_step(ctypes.byref(self.weights),
                                              ctypes.byref(state), x,
                                              ctypes.byref(result))
            self.assertEqual(status, 0, f"inference status {status}")
        return (np.array(result.ruls[:], dtype=np.float64),
                np.array(result.fault_probs[:], dtype=np.float64),
                float(result.mean_rul_pct))

    def _window(self, seed):
        rng = np.random.default_rng(seed)
        return rng.random((WINDOW, self.dims["D"]), dtype=np.float64)

    def test_rul_head_matches(self):
        window = self._window(0)
        c_rul, _, _ = self._run_c(window)
        np.testing.assert_allclose(c_rul, self.py.forward(window)["reg"],
                                   atol=TOLERANCE)

    def test_classifier_head_matches(self):
        window = self._window(1)
        _, c_cls, _ = self._run_c(window)
        np.testing.assert_allclose(c_cls, self.py.forward(window)["cls"],
                                   atol=TOLERANCE)
        self.assertAlmostEqual(float(c_cls.sum()), 1.0, places=4)

    def test_agrees_across_many_random_windows(self):
        worst = 0.0
        for seed in range(20):
            window = self._window(100 + seed)
            c_rul, _, _ = self._run_c(window)
            worst = max(worst, float(np.max(np.abs(
                c_rul - self.py.forward(window)["reg"]))))
        self.assertLess(worst, TOLERANCE, f"worst RUL divergence {worst:.3e}")

    def test_top_fault_class_agrees(self):
        for seed in range(10):
            window = self._window(200 + seed)
            _, c_cls, _ = self._run_c(window)
            self.assertEqual(int(np.argmax(c_cls)),
                             int(np.argmax(self.py.forward(window)["cls"])),
                             f"top class disagrees on seed {seed}")

    def test_mean_rul_pct_is_the_mean_rul_percentage(self):
        window = self._window(5)
        c_rul, _, chi = self._run_c(window)
        self.assertAlmostEqual(chi, float(np.mean(c_rul)) * 100.0, places=3)

    def test_reset_clears_recurrent_state(self):
        window = self._window(3)
        first, _, _ = self._run_c(window)
        second, _, _ = self._run_c(window)
        np.testing.assert_allclose(first, second, atol=1e-6)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(have_toolchain(), _SKIP_REASON)
class TestEdgeRuntimeFailsSafe(unittest.TestCase):
    """A corrupted frame must not read as "healthy".

    Before this, a single NaN propagated through the softmax; every
    `fault_probs[c] > top_prob` comparison against NaN is false, so the loop
    never advanced past index 0 and the engine reported top_fault_id = 0 --
    class "healthy" -- with top_fault_prob = NaN. Fail-to-nominal on a
    safety-relevant box.
    """

    @classmethod
    def setUpClass(cls):
        from c_engine.binding import load_engine
        cls.lib, cls.weights = load_engine()
        cls.dims = model_dims()

    def _step(self, values):
        import ctypes
        from c_engine.binding import TankInferenceResult, TankInferenceState
        state = TankInferenceState()
        result = TankInferenceResult()
        self.lib.tank_infer_reset(ctypes.byref(state))
        arr_t = ctypes.c_float * self.dims["D"]
        status = self.lib.tank_infer_step(
            ctypes.byref(self.weights), ctypes.byref(state),
            arr_t(*values), ctypes.byref(result))
        return status, result

    def test_nan_input_is_rejected_not_reported_as_healthy(self):
        from c_engine.binding import (TANK_INFER_ERR_INPUT,
                                      TANK_INFER_FAULT_UNKNOWN)
        vals = [0.5] * self.dims["D"]
        vals[3] = float("nan")
        status, res = self._step(vals)
        self.assertEqual(status, TANK_INFER_ERR_INPUT)
        self.assertEqual(res.top_fault_id, TANK_INFER_FAULT_UNKNOWN)
        self.assertNotEqual(res.top_fault_id, 0, "NaN must not read as healthy")
        self.assertEqual(res.mean_rul_pct, 0.0)

    def test_inf_input_is_rejected(self):
        from c_engine.binding import TANK_INFER_ERR_INPUT
        vals = [0.5] * self.dims["D"]
        vals[0] = float("inf")
        status, res = self._step(vals)
        self.assertEqual(status, TANK_INFER_ERR_INPUT)
        self.assertEqual(res.top_fault_prob, 0.0)

    def test_clean_input_still_succeeds(self):
        from c_engine.binding import TANK_INFER_OK
        status, res = self._step([0.5] * self.dims["D"])
        self.assertEqual(status, TANK_INFER_OK)
        self.assertLess(res.top_fault_id, self.dims["C"])
        self.assertGreater(res.top_fault_prob, 0.0)
