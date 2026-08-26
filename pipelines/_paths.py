"""Dataset root resolution for all pipeline loaders.

The datasets/ directory sits one level above Fault-Prediction-in-Vehicles/
when the project is laid out as the canonical structure dictates:

    Vnest/
    ├── datasets/          ← real data, gitignored
    └── Fault-Prediction-in-Vehicles/   ← this repo
        └── pipelines/

The loaders accept an explicit data_dir so test fixtures can override it.
When data_dir is None the resolver is invoked and tries, in order:

    1. datasets/<subdir>                  (running from Vnest/ or any parent)
    2. ../datasets/<subdir>               (running from Fault-Prediction-in-Vehicles/)
    3. <repo_root>/../datasets/<subdir>   (repo is a sub-directory deeper)
"""

from __future__ import annotations

import os
from pathlib import Path

# Absolute path of the Fault-Prediction-in-Vehicles/ directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Candidate dataset roots, evaluated in priority order.
_CANDIDATES = (
    _REPO_ROOT.parent / "datasets",    # ../datasets  (from FPiV/)
    Path("datasets"),                   # CWD/datasets (from Vnest/ or CI)
    _REPO_ROOT / "datasets",            # inside the repo (future)
)


def dataset_root() -> Path:
    """Return the first existing datasets/ directory."""
    for c in _CANDIDATES:
        if c.is_dir():
            return c.resolve()
    return _REPO_ROOT.parent / "datasets"  # best-guess even if absent


def resolve(subdir: str) -> str:
    """Return absolute path to a dataset sub-directory.

    ``subdir`` should use forward slashes and must be relative to datasets/,
    e.g. ``"procured/metropt3_apu"``.
    """
    return str(dataset_root() / Path(subdir))
