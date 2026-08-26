"""Python ctypes bindings for the C99 Tank PDM Inference Engine."""

import ctypes
import os
from pathlib import Path
import numpy as np

from c_engine.gen_dims import model_dims

# Read from the exported model rather than hardcoding: these had drifted to
# D=58 while the trained model was D=24.
_DIMS = model_dims()
D_FEATURES = _DIMS["D"]
H_HIDDEN = _DIMS["H"]
R_PARTS = _DIMS["R"]
C_CLASSES = _DIMS["C"]


class TankModelWeights(ctypes.Structure):
    _fields_ = [
        ("Wf", (ctypes.c_float * H_HIDDEN) * D_FEATURES),
        ("Uf", (ctypes.c_float * H_HIDDEN) * H_HIDDEN),
        ("bf", ctypes.c_float * H_HIDDEN),
        ("Wi", (ctypes.c_float * H_HIDDEN) * D_FEATURES),
        ("Ui", (ctypes.c_float * H_HIDDEN) * H_HIDDEN),
        ("bi", ctypes.c_float * H_HIDDEN),
        ("Wc", (ctypes.c_float * H_HIDDEN) * D_FEATURES),
        ("Uc", (ctypes.c_float * H_HIDDEN) * H_HIDDEN),
        ("bc", ctypes.c_float * H_HIDDEN),
        ("Wo", (ctypes.c_float * H_HIDDEN) * D_FEATURES),
        ("Uo", (ctypes.c_float * H_HIDDEN) * H_HIDDEN),
        ("bo", ctypes.c_float * H_HIDDEN),
        ("Wy", (ctypes.c_float * R_PARTS) * H_HIDDEN),
        ("by", ctypes.c_float * R_PARTS),
        ("Wcls", (ctypes.c_float * C_CLASSES) * H_HIDDEN),
        ("bcls", ctypes.c_float * C_CLASSES),
    ]


class TankInferenceState(ctypes.Structure):
    _fields_ = [
        ("h", ctypes.c_float * H_HIDDEN),
        ("c", ctypes.c_float * H_HIDDEN),
        ("step_count", ctypes.c_uint32),
    ]


class TankInferenceResult(ctypes.Structure):
    _fields_ = [
        ("ruls", ctypes.c_float * R_PARTS),
        ("fault_probs", ctypes.c_float * C_CLASSES),
        ("top_fault_id", ctypes.c_uint32),
        ("top_fault_prob", ctypes.c_float),
        # Renamed from `composite_chi`: it is the mean RUL fraction across
        # parts scaled to 0-100, not a health index. Must stay in step with
        # TankInferenceResult in tank_pdm_infer.h.
        ("mean_rul_pct", ctypes.c_float),
        ("status", ctypes.c_int),
    ]


# Mirrors TankInferStatus in tank_pdm_infer.h.
TANK_INFER_OK = 0
TANK_INFER_ERR_NULL = 1
TANK_INFER_ERR_INPUT = 2
TANK_INFER_ERR_STATE = 3
TANK_INFER_FAULT_UNKNOWN = 0xFFFFFFFF


def load_c_engine_lib():
    dll_path = Path(__file__).resolve().parent / "libtank_infer.dll"
    if not dll_path.exists():
        raise FileNotFoundError(f"{dll_path} not found. Please compile it first.")
    
    lib = ctypes.CDLL(str(dll_path))
    
    lib.tank_infer_reset.argtypes = [ctypes.POINTER(TankInferenceState)]
    lib.tank_infer_reset.restype = None
    
    lib.tank_infer_step.argtypes = [
        ctypes.POINTER(TankModelWeights),
        ctypes.POINTER(TankInferenceState),
        ctypes.POINTER(ctypes.c_float * D_FEATURES),
        ctypes.POINTER(TankInferenceResult),
    ]
    lib.tank_infer_step.restype = ctypes.c_int
    
    return lib


# ---------------------------------------------------------------------------
# Engine loading
# ---------------------------------------------------------------------------
from c_engine.build import build_engine, missing_tools  # noqa: E402
from c_engine.export_weights import WEIGHTS_BIN, export_weights  # noqa: E402


_ENGINE_CACHE = None


def load_engine():
    """Build if needed, load the shared library, and populate weights.

    Returns ``(lib, weights)``. Raises RuntimeError when no toolchain is
    available rather than silently returning a half-initialised engine.
    """
    # Cached: two test classes each calling load_engine() rebuilt and
    # re-opened the DLL, and on Windows the second build fails while the first
    # handle is still open. That surfaced as a setUpClass error only in the
    # full-suite run, never when the file was run alone.
    global _ENGINE_CACHE
    if _ENGINE_CACHE is not None:
        return _ENGINE_CACHE

    lib_path = build_engine()
    if lib_path is None:
        raise RuntimeError(
            "cannot build the edge runtime; missing: " + ", ".join(missing_tools()))

    lib = ctypes.CDLL(str(lib_path))

    lib.tank_infer_reset.argtypes = [ctypes.POINTER(TankInferenceState)]
    lib.tank_infer_reset.restype = None

    lib.tank_infer_step.argtypes = [
        ctypes.POINTER(TankModelWeights),
        ctypes.POINTER(TankInferenceState),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(TankInferenceResult),
    ]
    lib.tank_infer_step.restype = ctypes.c_int

    lib.tank_weights_load.argtypes = [ctypes.c_char_p,
                                      ctypes.POINTER(TankModelWeights)]
    lib.tank_weights_load.restype = ctypes.c_int

    if not WEIGHTS_BIN.exists():
        export_weights()

    weights = TankModelWeights()
    rc = lib.tank_weights_load(str(WEIGHTS_BIN).encode("utf-8"),
                               ctypes.byref(weights))
    if rc != 0:
        raise RuntimeError("tank_weights_load failed with code " + str(rc))
    _ENGINE_CACHE = (lib, weights)
    return _ENGINE_CACHE
