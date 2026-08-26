"""PHM processing pipeline: the block chain between ingest and broadcast.

Why this module exists
----------------------
`PROJECT.md` documented the flow as

    ingest -> FDIR plausibility gate -> neural inference -> DTC engine -> WS

but nothing implemented it. `server.py` read the raw broker frame and shipped
it straight to the WebSocket. `grep 'plausib|dtc|infer|lstm' server.py`
returned nothing. `sensor_plausibility.py` (1,401 LOC) and `dtc_engine.py`
(1,247 LOC) were imported only by `__init__.py` and by their unit tests --
2,648 lines of the system's headline functionality, never invoked at runtime.
The health index broadcast to the HUD was `latest.get("composite_chi", 95.0)`,
i.e. whatever the client posted, and four of the eight subsystem health values
were hardcoded literals (`[chi, chi, chi, 98, 99, 95, 96, 99]`).

In ISO 13374 / OSA-CBM terms the deployed system had blocks 1 (Data
Acquisition) and 6 (presentation) and nothing in between. This module is
blocks 2-5: Data Manipulation, State Detection, Health Assessment and
Prognostic Assessment, in that order, each with an explicit input and output.

Degradation policy
------------------
Inference is optional: if the trained artifacts are absent the pipeline still
runs the gate and the DTC engine, and reports `inference_available=False`
rather than fabricating a health number. Nothing downstream is permitted to
invent a value it did not compute -- that is the defect this module replaces.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .dtc_engine import DTCEngine
from .sensor_plausibility import SensorPlausibilityGate
from .units import looks_canonical, to_canonical, to_native

logger = logging.getLogger("phm_pipeline")

_DOCS = Path(__file__).resolve().parents[1] / "Fault-Prediction-in-Vehicles" / "docs"
_ML_ROOT = Path(__file__).resolve().parents[1] / "Fault-Prediction-in-Vehicles"


class _Inference:
    """Rolling-window LSTM forward pass over the shipped artifacts.

    Mirrors `ml/lstm.py::LSTMModel.forward` and `docs/lstm.js`; the C runtime in
    `c_engine/` is the third implementation of the same arithmetic and is
    covered by `tests/test_c_python_parity.py`.
    """

    def __init__(self, docs: Path = _DOCS):
        self.available = False
        self.window: int = 40
        self._buf: Deque[List[float]] = deque()
        try:
            import sys

            import numpy as np  # noqa: F401

            if str(_ML_ROOT) not in sys.path:
                sys.path.insert(0, str(_ML_ROOT))
            from ml.lstm import LSTMModel

            cfg = json.loads((docs / "config.json").read_text(encoding="utf-8"))
            mdl = json.loads((docs / "model.json").read_text(encoding="utf-8"))
        except Exception as exc:                       # pragma: no cover
            logger.warning("inference unavailable: %s", exc)
            return

        import numpy as np

        self.features: List[str] = cfg["input_features"]
        self.scaler: Dict[str, Any] = cfg["scaler"]
        self.class_names: List[str] = cfg["class_names"]
        self.part_order: List[str] = cfg["part_order"]
        self.window = int(cfg.get("window", 40))
        self._buf = deque(maxlen=self.window)

        self.model = LSTMModel(D=mdl["D"], H=mdl["H"], R=mdl["R"], C=mdl["C"],
                               seed=0)
        for name, value in mdl["params"].items():
            self.model.p[name] = np.asarray(value, dtype=np.float64)
        self._np = np
        self.available = True

    def _vector(self, clean: Dict[str, float]) -> Optional[List[float]]:
        out = []
        for key in self.features:
            if key not in clean:
                return None
            sc = self.scaler[key]
            v = float(clean[key])
            if sc.get("log"):
                v = self._np.log1p(max(v, 0.0))
            span = max(sc["max"] - sc["min"], 1e-9)
            out.append(float(min(max((v - sc["min"]) / span, 0.0), 1.0)))
        return out

    def step(self, clean: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Returns None until the window has filled, or if channels are missing."""
        if not self.available:
            return None
        vec = self._vector(clean)
        if vec is None:
            return None
        self._buf.append(vec)
        if len(self._buf) < self.window:
            return None
        X = self._np.asarray(self._buf, dtype=float)
        cache = self.model.forward(X)
        return {
            "rul_fraction": {p: float(v) for p, v
                             in zip(self.part_order, cache["reg"])},
            "fault_probs": {n: float(p) for n, p
                            in zip(self.class_names, cache["cls"])},
        }


class PHMPipeline:
    """Blocks 2-5 of the condition-monitoring chain, composed and ordered."""

    def __init__(self, sample_rate_hz: float = 20.0,
                 flash_file: str = "results/dtc_flash_log.jsonl"):
        self._lock = threading.RLock()
        self.gate = SensorPlausibilityGate(sample_rate_hz=sample_rate_hz)
        self.dtc = DTCEngine(flash_file=flash_file, auto_recover=False)
        self.inference = _Inference()
        self._health_fn = self._load_health_scorer()

    @staticmethod
    def _load_health_scorer():
        try:
            import sys
            if str(_ML_ROOT) not in sys.path:
                sys.path.insert(0, str(_ML_ROOT))
            from ml.parts import (PART_ORDER, overall_health_index,
                                  part_health_index)
            return PART_ORDER, part_health_index, overall_health_index
        except Exception as exc:                       # pragma: no cover
            logger.warning("health scoring unavailable: %s", exc)
            return None

    def subsystem_health(self, clean: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Score every subsystem from the *sanitised* telemetry.

        Previously the eight values broadcast to the HUD were
        `[chi, chi, chi, 98, 99, 95, 96, 99]` -- three copies of a
        client-supplied number followed by five literals.
        """
        if self._health_fn is None:
            return None
        part_order, part_health, overall_health = self._health_fn
        subs = [p for p in part_order if p != "overall"]
        health = {p: float(part_health(p, clean)) for p in subs}
        if "overall" in part_order:
            health["overall"] = float(overall_health(health))
        return health

    def process(self, raw_frame: Dict[str, Any]) -> Dict[str, Any]:
        """Run one frame through the full chain.

        Returns a envelope carrying, in order: the sanitised signals, the
        electrical/plausibility faults, the subsystem health vector, the
        prognostic outputs, and the resulting active DTCs.
        """
        with self._lock:
            # --- Block 1b: unit normalisation -------------------------------
            # The FDIR catalog, the J1939 SPN scalings and the HUD are all
            # engineering-native (bar, %); tank_sim is SI-native (Pa,
            # fraction). Without this the pressures tripped a spurious
            # SHORT_CIRCUIT every frame and levels were silently read as
            # ~0.9% full. See units.py.
            frame = (dict(raw_frame) if looks_canonical(raw_frame)
                     else to_canonical(raw_frame))

            # --- Block 2: Data Manipulation (FDIR sanitisation) -------------
            result = self.gate.filter_frame(frame)
            clean = result.clean_telemetry

            # --- Block 3: State Detection (electrical / range faults) ------
            fdir_dtcs = self.dtc.process_fdir_faults(result.faults_detected)

            # --- Block 4: Health Assessment --------------------------------
            # ml/parts.py thresholds carry their own `scale` factors and so
            # expect the SI form; convert back rather than double-scaling.
            health = self.subsystem_health(to_native(clean))

            # --- Block 5: Prognostic Assessment ----------------------------
            # The scaler was fitted on SI-unit features, so the model sees
            # the same view it was trained on.
            prognosis = self.inference.step(to_native(clean))

            # --- Advisory: neural findings become DTCs ---------------------
            neural_dtcs: List[Any] = []
            if health is not None or prognosis is not None:
                neural_dtcs = self.dtc.process_neural_predictions(
                    subsystem_health=health,
                    fault_probs=(prognosis or {}).get("fault_probs"),
                )

            active = self.dtc.get_active_dtcs()
            return {
                "clean_telemetry": clean,
                "is_valid": bool(result.is_valid),
                "gate_ms": float(result.processing_time_ms),
                "sensor_faults": [self._fault_dict(f)
                                  for f in result.faults_detected],
                "subsystem_health": health,
                "health_available": health is not None,
                "prognosis": prognosis,
                "inference_available": bool(self.inference.available),
                "dtcs_new": [d.to_dict() for d in (fdir_dtcs + neural_dtcs)],
                "dtcs_active": [d.to_dict() for d in active],
                "dm1_hex": self.dtc.encode_dm1_packet().hex().upper(),
                "dm2_hex": self.dtc.encode_dm2_packet().hex().upper(),
            }

    @staticmethod
    def _fault_dict(f: Any) -> Dict[str, Any]:
        if hasattr(f, "__dict__"):
            return {k: v for k, v in vars(f).items() if not k.startswith("_")}
        return dict(f) if isinstance(f, dict) else {"fault": str(f)}


_pipeline: Optional[PHMPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> PHMPipeline:
    """Process-wide singleton, built lazily so import stays cheap."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = PHMPipeline()
        return _pipeline


__all__ = ["PHMPipeline", "get_pipeline"]
