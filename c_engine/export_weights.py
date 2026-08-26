"""Serialise trained LSTM weights for the C edge runtime.

Layout: little-endian float32, row-major, in WEIGHT_ORDER -- which is exactly
the field order of TankModelWeights, so the C side fills the struct with a
single fread and performs no allocation.

Without this the edge engine had no path to the trained model at all.
"""

import json
from pathlib import Path

import numpy as np

from c_engine.gen_dims import MODEL_PATH, model_dims

C_DIR = Path(__file__).resolve().parent
WEIGHTS_BIN = C_DIR / "tank_pdm_weights.bin"

WEIGHT_ORDER = ("Wf", "Uf", "bf", "Wi", "Ui", "bi", "Wc", "Uc", "bc",
                "Wo", "Uo", "bo", "Wy", "by", "Wcls", "bcls")


def expected_shape(name: str, dims: dict) -> tuple:
    D, H, R, C = dims["D"], dims["H"], dims["R"], dims["C"]
    if name in ("Wf", "Wi", "Wc", "Wo"):
        return (D, H)
    if name in ("Uf", "Ui", "Uc", "Uo"):
        return (H, H)
    if name == "Wy":
        return (H, R)
    if name == "Wcls":
        return (H, C)
    if name == "by":
        return (R,)
    if name == "bcls":
        return (C,)
    return (H,)


def expected_float_count(dims: dict) -> int:
    return sum(int(np.prod(expected_shape(n, dims))) for n in WEIGHT_ORDER)


def export_weights(out=None) -> Path:
    out = Path(out) if out is not None else WEIGHTS_BIN
    dims = model_dims()
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    params = model["params"]

    with open(out, "wb") as fh:
        for name in WEIGHT_ORDER:
            arr = np.asarray(params[name], dtype=np.float32)
            want = expected_shape(name, dims)
            if arr.shape != want:
                raise ValueError(f"{name}: expected {want}, got {arr.shape}")
            fh.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())
    return out


if __name__ == "__main__":
    path = export_weights()
    print(f"wrote {path} ({path.stat().st_size} bytes, "
          f"{expected_float_count(model_dims())} floats)")
