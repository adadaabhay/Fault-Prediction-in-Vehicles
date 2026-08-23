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

DOCS = Path(__file__).resolve().parent / "docs"


def main() -> int:
    cfg = json.loads((DOCS / "config.json").read_text(encoding="utf-8"))
    mdl = json.loads((DOCS / "model.json").read_text(encoding="utf-8"))
    stream = json.loads((DOCS / "live_stream.json").read_text(encoding="utf-8"))

    feats = cfg["input_features"]
    errs: list[str] = []

    if mdl["D"] != len(feats):
        errs.append(f"model D={mdl['D']} but config lists {len(feats)} features")
    if mdl["C"] != len(cfg["class_names"]):
        errs.append(f"model C={mdl['C']} but {len(cfg['class_names'])} class names")
    if mdl["R"] != len(cfg["part_order"]):
        errs.append(f"model R={mdl['R']} but {len(cfg['part_order'])} parts")

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
