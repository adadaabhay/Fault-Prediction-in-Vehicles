"""Dataset root resolution for all pipeline loaders.

The pipelines/ loaders expect a ``datasets/`` directory somewhere on disk
holding the procured public corpora (METROPT, Zema, Deutz NRTC, ...).
The corpora are gitignored -- they ship from the procurement step, not
from the repo -- so the loader has to *find* the directory at runtime.

Canonical layout (this repo, cloned to any path):

    <repo>/                 ← ``pipelines/_paths.py`` lives under here
    ├── pipelines/
    ├── datasets/           ← gitignored, populated by the data procurement step
    └── ...

Optional workspace-parent layout (opt-in; set ``PHM_DATASETS_PARENT=1`` so
existing procurement copies that live one level above the repo keep
working without re-procurement):

    <workspace>/
    ├── datasets/           ← one level above the repo
    └── Fault-Prediction-in-Vehicles/

Resolution order, first hit wins:

    1. ``$PHM_DATASETS_DIR`` if set (explicit override; useful in CI)
    2. ``<repo>/datasets/``           (canonical)
    3. ``<repo>/../datasets/``        (only if ``PHM_DATASETS_PARENT=1``)
    4. ``CWD/datasets/``              (developer convenience; e.g. running
                                      a notebook from the procurement root)

The loaders accept an explicit ``data_dir=`` so test fixtures can override
the resolver without touching the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

# Absolute path of the repo root (the directory that contains this file's
# parent's parent -- ``pipelines/_paths.py`` is at ``<repo>/pipelines/``).
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _candidate_dirs() -> tuple[Path, ...]:
    """Build the resolution candidate list, honouring opt-in env flags.

    The workspace-parent candidate (``<repo>/../datasets/``) is opt-in
    via ``PHM_DATASETS_PARENT=1`` so a clean clone of the target does
    not silently walk up to a sibling workspace it knows nothing about.
    """
    candidates: list[Path] = [
        _REPO_ROOT / "datasets",           # <repo>/datasets/   (canonical)
    ]
    if os.environ.get("PHM_DATASETS_PARENT") == "1":
        candidates.append(_REPO_ROOT.parent / "datasets")
    candidates.append(Path("datasets"))    # CWD/datasets/       (dev convenience)
    return tuple(candidates)


def dataset_root() -> Path:
    """Return the first existing datasets/ directory.

    Honours ``$PHM_DATASETS_DIR`` as an explicit override (so CI can pin
    a known path).  Falls back to the repo-local ``<repo>/datasets/``
    even if the directory does not yet exist -- this keeps error messages
    from downstream loaders informative (``"no such file: <repo>/datasets/..."``)
    rather than masking the canonical location.
    """
    override = os.environ.get("PHM_DATASETS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    for c in _candidate_dirs():
        if c.is_dir():
            return c.resolve()
    return _REPO_ROOT / "datasets"  # canonical best-guess, even if absent


def resolve(subdir: str) -> str:
    """Return absolute path to a dataset sub-directory.

    ``subdir`` should use forward slashes and must be relative to datasets/,
    e.g. ``"procured/metropt3_apu"``.
    """
    return str(dataset_root() / Path(subdir))
