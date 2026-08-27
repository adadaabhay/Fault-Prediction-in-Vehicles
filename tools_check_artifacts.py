#!/usr/bin/env python3
"""Verify the published dashboard artifacts are mutually consistent.

The dashboard runs the exported model in the browser. Publishing a `docs/`
whose `model.json` and `config.json` disagree, or whose `live_stream.json`
lacks channels the model expects, produces a page that loads cleanly and then
quietly renders nothing -- which is exactly what happened: `save_demo` omitted
`susp_compliance` and `driveline_efficiency`, so the in-browser per-module RUL
could never run on the demo stream, and nothing reported it.

Run locally with `python tools_check_artifacts.py`; runs in CI before deploy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# This script is run from the repo root (`python tools_check_artifacts.py`)
# but it imports from the ``ml`` package, which only resolves when the repo
# root is on ``sys.path``.  Do that first, before any other import that
# would pull in ``ml`` transitively, so we don't need a ``# noqa: E402``
# on every line.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.parts import INPUT_FEATURES, PART_ORDER
from ml.scenarios import ALL_FAULTS

# Authoritative targets.  These come from the Python schema
# (``ml.parts.INPUT_FEATURES``, ``ml.parts.PART_ORDER``) and the declared
# fault taxonomy (``ml.scenarios.ALL_FAULTS``).  Read them here so a future
# retrain that reverts the schema cannot silently re-ship D=24.
EXPECTED_D = len(INPUT_FEATURES)
EXPECTED_R = len(PART_ORDER)
EXPECTED_C = 1 + len(ALL_FAULTS)   # healthy + 12 declared fault classes
EXPECTED_H = 24                     # LSTM hidden size (training-time choice)

DOCS = _REPO_ROOT / "docs"


def main() -> int:
    cfg = json.loads((DOCS / "config.json").read_text(encoding="utf-8"))
    mdl = json.loads((DOCS / "model.json").read_text(encoding="utf-8"))
    stream = json.loads((DOCS / "live_stream.json").read_text(encoding="utf-8"))

    feats = cfg["input_features"]
    errs: list[str] = []

    # --- C-engine / model / schema agreement -----------------------------
    if mdl["D"] != EXPECTED_D:
        errs.append(f"model D={mdl['D']} != schema D={EXPECTED_D}; retrain "
                    f"(`python -m ml.train`) to reconcile")
    if mdl["R"] != EXPECTED_R:
        errs.append(f"model R={mdl['R']} != schema R={EXPECTED_R}")
    if mdl["C"] != EXPECTED_C:
        errs.append(f"model C={mdl['C']} != schema C={EXPECTED_C} "
                    f"(healthy + {len(ALL_FAULTS)} declared faults)")
    if mdl["H"] != EXPECTED_H:
        errs.append(f"model H={mdl['H']} != expected H={EXPECTED_H}")

    if mdl["D"] != len(feats):
        errs.append(f"model D={mdl['D']} but config lists {len(feats)} features")
    if mdl["C"] != len(cfg["class_names"]):
        errs.append(f"model C={mdl['C']} but {len(cfg['class_names'])} class names")
    if mdl["R"] != len(cfg["part_order"]):
        errs.append(f"model R={mdl['R']} but {len(cfg['part_order'])} parts")

    # Weight matrices must have the leading dim the header advertises.
    for gate in ("Wf", "Wi", "Wc", "Wo"):
        rows = len(mdl["params"][gate])
        if rows != EXPECTED_D:
            errs.append(f"weight matrix {gate} has {rows} input rows but "
                        f"D={EXPECTED_D} is required")

    # --- stream + scaler coverage ----------------------------------------
    records = stream.get("records") or []
    if not records:
        errs.append("live_stream.json has no records")
    else:
        missing = [f for f in feats if f not in records[0]]
        if missing:
            errs.append(f"live_stream.json missing model inputs: {missing}")

    missing_scaler = [f for f in feats if f not in cfg["scaler"]]
    if missing_scaler:
        errs.append(f"scaler has no entry for: {missing_scaler}")

    for part in cfg["part_order"]:
        if part not in stream.get("health", {}):
            errs.append(f"stream health series missing part '{part}'")

    if errs:
        for e in errs:
            print(f"::error::{e}")
        return 1

    print(f"artifacts consistent: D={mdl['D']} H={mdl['H']} "
          f"R={mdl['R']} C={mdl['C']}, {len(records)} demo steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
