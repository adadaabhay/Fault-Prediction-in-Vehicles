"""Verify the published dashboard artifacts are mutually consistent.

The dashboard runs the exported model in the browser.  Publishing a
``docs/`` whose ``model.json`` and ``config.json`` disagree, or whose
``live_stream.json`` lacks channels the model expects, produces a page
that loads cleanly and then quietly renders nothing -- which is exactly
what happened before: ``save_demo`` omitted ``susp_compliance`` and
``driveline_efficiency``, so the in-browser per-module RUL could never
run on the demo stream, and nothing reported it.

This module is the CI gate that catches the mistake before a release.
It is also runnable locally:

    python -m tools.check_artifacts
    # or, after `pip install -e .`:
    phm-check-artifacts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Locate the repo root.

    Looks for ``ml/parts.py`` upward from this file's parent so the
    module works whether it is imported as ``tools.check_artifacts``
    (after install) or run as a script with ``python tools/check_artifacts.py``.
    """
    here = Path(__file__).resolve().parent
    for ancestor in (here, *here.parents):
        candidate = ancestor / "ml" / "parts.py"
        if candidate.exists():
            return ancestor
    # Last-resort: current working directory.  Keeps the error message
    # informative if the layout is unexpectedly different.
    return Path.cwd()


# This module is run from the repo root but it imports from ``ml``, which
# only resolves when the repo root is on ``sys.path``.  Insert it before any
# other import that would pull in ``ml`` transitively, so we don't need a
# ``# noqa: E402`` on every line.
_REPO_ROOT = _repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.parts import INPUT_FEATURES, PART_ORDER  # noqa: E402
from ml.scenarios import ALL_FAULTS  # noqa: E402

# Authoritative targets.  These come from the Python schema
# (``ml.parts.INPUT_FEATURES``, ``ml.parts.PART_ORDER``) and the declared
# fault taxonomy (``ml.scenarios.ALL_FAULTS``).  Reading them here keeps a
# future retrain from silently re-shipping a stale ``D`` value.
# ``H`` is a training-time hyperparameter, not a schema constant --
# any value the model and config agree on is acceptable, so we read the
# published value out of ``config.json`` and assert self-consistency
# in :func:`main` rather than pinning a constant here.
EXPECTED_D = len(INPUT_FEATURES)
EXPECTED_R = len(PART_ORDER)
EXPECTED_C = 1 + len(ALL_FAULTS)   # healthy + declared fault classes

DOCS = _REPO_ROOT / "docs"


def main() -> int:
    """Run the consistency check.  Returns 0 on success, 1 on any error."""
    if not DOCS.is_dir():
        print(f"::error::docs/ directory not found at {DOCS}")
        return 1

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
    # ``H`` self-consistency: a legitimate retrain to any H is fine as
    # long as model.json and config.json agree.  If config.json is
    # missing the field, surface that instead of crashing.
    published_h = (cfg.get("model") or {}).get("hidden")
    if published_h is None:
        errs.append(
            f"config.json missing model.hidden; "
            f"cannot verify H={mdl.get('H')!r}"
        )
    elif mdl["H"] != published_h:
        errs.append(
            f"model.json H={mdl['H']} != config.json "
            f"model.hidden={published_h}; "
            f"rerun `python -m ml.train` to reconcile"
        )

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
