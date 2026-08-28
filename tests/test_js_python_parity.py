"""The browser inference engine must agree numerically with the Python model.

`docs/lstm.js` runs the shipped weights in front of every user of the
dashboard, and it was the only one of the three implementations of this
forward pass with no behavioural test at all. CI ran `node --check` on it,
which verifies that the file parses -- not that it computes anything correct.
The C runtime had `tests/test_c_python_parity.py`; the JS runtime had nothing,
despite being the implementation that actually ships to users.

Weights are additionally rounded to 5 decimals by `LSTMModel.to_json`, so this
also pins the error that rounding introduces.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT))

# float64 both sides; the only divergence should be summation order.
TOLERANCE = 1e-9
WINDOW = 40


def _have_node() -> bool:
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


# Plain `require` of the real shipped file -- no eval, no copy, no shim. The
# module.exports footer added to docs/lstm.js is what makes that possible.
_DRIVER = r"""
const fs = require('fs');
const { LSTM } = require(process.argv[2]);
const model = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const X = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));
const out = LSTM.forward(model, X);
process.stdout.write(JSON.stringify({reg: out.reg, cls: out.cls}));
"""


@unittest.skipUnless(_have_node(), "node not available")
@unittest.skipUnless((DOCS / "model.json").exists(), "docs/model.json not built")
class TestJSPythonParity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ml.lstm import LSTMModel

        cls.model_json = json.loads((DOCS / "model.json").read_text(encoding="utf-8"))
        cls.py = LSTMModel(D=cls.model_json["D"], H=cls.model_json["H"],
                           R=cls.model_json["R"], C=cls.model_json["C"], seed=0)
        for name, value in cls.model_json["params"].items():
            cls.py.p[name] = np.asarray(value, dtype=np.float64)
        cls.tmp = tempfile.mkdtemp()
        cls.driver = Path(cls.tmp) / "driver.js"
        cls.driver.write_text(_DRIVER, encoding="utf-8")

    def _run_js(self, X):
        in_path = Path(self.tmp) / "x.json"
        in_path.write_text(json.dumps(X.tolist()), encoding="utf-8")
        proc = subprocess.run(
            ["node", str(self.driver), str((DOCS / "lstm.js").as_posix()),
             str(DOCS / "model.json"), str(in_path)],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        return np.asarray(out["reg"]), np.asarray(out["cls"])

    def _cases(self):
        D = self.model_json["D"]
        rng = np.random.default_rng(4)
        yield "uniform", rng.random((WINDOW, D))
        yield "zeros", np.zeros((WINDOW, D))
        yield "ones", np.ones((WINDOW, D))
        yield "ramp", np.tile(np.linspace(0, 1, WINDOW)[:, None], (1, D))

    def test_forward_pass_matches_python(self):
        for name, X in self._cases():
            with self.subTest(case=name):
                js_reg, js_cls = self._run_js(X)
                cache = self.py.forward(X)
                np.testing.assert_allclose(js_reg, cache["reg"], atol=TOLERANCE,
                                           err_msg=f"RUL head diverged ({name})")
                np.testing.assert_allclose(js_cls, cache["cls"], atol=TOLERANCE,
                                           err_msg=f"class head diverged ({name})")

    def test_class_probabilities_are_a_distribution(self):
        _, js_cls = self._run_js(next(self._cases())[1])
        self.assertAlmostEqual(float(js_cls.sum()), 1.0, places=9)
        self.assertTrue(np.all(js_cls >= 0.0))

    def test_argmax_class_agrees(self):
        """The number the HUD actually displays."""
        for name, X in self._cases():
            with self.subTest(case=name):
                js_reg, js_cls = self._run_js(X)
                cache = self.py.forward(X)
                self.assertEqual(int(np.argmax(js_cls)),
                                 int(np.argmax(cache["cls"])), name)

    def test_exported_weights_survive_json_rounding(self):
        """`to_json` rounds to 5 decimals; that must not move a prediction."""
        from ml.lstm import LSTMModel
        exact = LSTMModel(D=self.model_json["D"], H=self.model_json["H"],
                          R=self.model_json["R"], C=self.model_json["C"], seed=0)
        for name, value in self.model_json["params"].items():
            exact.p[name] = np.asarray(value, dtype=np.float64)
        rounded = json.loads(json.dumps(exact.to_json()))
        approx = LSTMModel(D=exact.D, H=exact.H, R=exact.R, C=exact.C, seed=0)
        for name, value in rounded["params"].items():
            approx.p[name] = np.asarray(value, dtype=np.float64)
        X = next(self._cases())[1]
        a = exact.forward(X)
        b = approx.forward(X)
        np.testing.assert_allclose(a["reg"], b["reg"], atol=1e-5)
        self.assertEqual(int(np.argmax(a["cls"])), int(np.argmax(b["cls"])))


if __name__ == "__main__":
    unittest.main()
