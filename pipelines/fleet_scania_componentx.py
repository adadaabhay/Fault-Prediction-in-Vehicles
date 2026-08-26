"""SCANIA Component X Fleet Run-to-Failure Pipeline (researchdata.se 2024-34).

Operational readouts from a fleet of heavy trucks joined to workshop repair
records, which supply genuine time-to-event labels. This is the project's only
non-simulated RUL ground truth: every other RUL target in the codebase is
produced by the physics simulator.

Shape of the corpus (training split)
------------------------------------
- ``train_operational_readouts.csv``  1,122,452 rows x 107 cols (1.2 GB).
  ``vehicle_id``, ``time_step``, and 105 anonymised counter/histogram channels.
  Roughly 48 readouts per vehicle -- this is a time series *per vehicle*, not
  one row per vehicle.
- ``train_tte.csv``  23,550 vehicles: ``length_of_study_time_step`` (how long
  the vehicle was observed) and ``in_study_repair`` (whether the component was
  actually repaired inside that window).
- ``train_specifications.csv``  8 categorical spec columns per vehicle.

Right-censoring dominates
-------------------------
**90.4% of vehicles are censored** (21,278 of 23,550 never had the repair).
For those, ``length_of_study_time_step`` is a *lower bound* on component life,
not the life. Regressing on it directly teaches the model that healthy trucks
fail at the moment observation stopped, which biases every estimate downwards.
:func:`uncensored` exists for that reason and the benchmark calls it explicitly.

Grouping
--------
Readouts from one vehicle are highly correlated, so folds must be split by
``vehicle_id`` (``GroupKFold``). A random row split leaks the same truck into
train and test -- the same error that made MetroPT-3 look like a 0.9999 problem.
"""

import os

import numpy as np
import pandas as pd

LABEL_PROVENANCE = "ground_truth_workshop_repair_records"

from pipelines._paths import dataset_root as _dataset_root

def _scania_data_dir() -> str:
    base = _dataset_root()
    return str(base / "scania" / "2024-34-2" / "data")

DATA_DIR = os.path.join("datasets", "scania", "2024-34-2", "data")

# Fallback locations — searched in priority order.
_CANDIDATE_DIRS = (
    _scania_data_dir(),
    DATA_DIR,
    os.path.join("datasets", "scania"),
    os.path.join("datasets", "procured", "scania_componentx"),
)

# Source column -> canonical name.
COLUMN_ALIASES = {
    "vehicle_id": ("vehicle_id", "veh_id", "id", "vehicle"),
    "time_to_event": ("time_to_event", "length_of_study_time_step", "tte"),
    "event_observed": ("event_observed", "in_study_repair", "event"),
}

# Never model inputs: identifiers, the clock, and the labels themselves.
NON_FEATURE_COLUMNS = {"vehicle_id", "time_to_event", "event_observed",
                       "rul", "study_end"}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    for canonical, candidates in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for candidate in candidates:
            if candidate in df.columns:
                df = df.rename(columns={candidate: canonical})
                break
    return df


def _resolve_dir(data_dir=None) -> str:
    if data_dir is not None:
        return data_dir
    for candidate in _CANDIDATE_DIRS:
        if os.path.exists(os.path.join(candidate, "train_tte.csv")):
            return candidate
        nested = os.path.join(candidate, "2024-34-2", "data")
        if os.path.exists(os.path.join(nested, "train_tte.csv")):
            return nested
    return DATA_DIR


def available(data_dir=None) -> bool:
    """True when the corpus has been procured."""
    resolved = _resolve_dir(data_dir)
    return os.path.exists(os.path.join(resolved, "train_tte.csv"))


def load_tte(data_dir=None, split: str = "train") -> pd.DataFrame:
    """Per-vehicle time-to-event labels."""
    resolved = _resolve_dir(data_dir)
    path = os.path.join(resolved, f"{split}_tte.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. SCANIA Component X is not redistributable; "
            "download it from https://researchdata.se/en/catalogue/dataset/2024-34")
    df = _normalise_columns(pd.read_csv(path))
    df["event_observed"] = df["event_observed"].fillna(0).astype(int).clip(0, 1)
    return df


def load_componentx_data(data_dir=None, split: str = "train",
                         max_vehicles: int | None = 4000,
                         random_state: int = 42,
                         with_specifications: bool = True) -> pd.DataFrame:
    """Join operational readouts to their time-to-event labels.

    Adds a per-readout ``rul`` column: the study end minus the current
    ``time_step``. For an observed failure that is the true remaining life; for
    a censored vehicle it is a lower bound, which is why ``event_observed`` is
    carried alongside rather than discarded.

    ``max_vehicles`` subsamples *whole vehicles* (never individual readouts) to
    keep memory bounded -- the full training readouts file is 1.2 GB. Pass
    ``None`` to load every vehicle.
    """
    resolved = _resolve_dir(data_dir)
    readouts_path = os.path.join(resolved, f"{split}_operational_readouts.csv")
    if not os.path.exists(readouts_path):
        raise FileNotFoundError(
            f"{readouts_path} not found. SCANIA Component X is not "
            "redistributable; download it from "
            "https://researchdata.se/en/catalogue/dataset/2024-34")

    tte = load_tte(resolved, split)

    keep_ids = None
    if max_vehicles is not None and max_vehicles < len(tte):
        # Stratify the subsample by outcome so the ~9.6% observed failures are
        # not lost to chance.
        rng = np.random.default_rng(random_state)
        observed = tte.loc[tte["event_observed"] == 1, "vehicle_id"].to_numpy()
        censored = tte.loc[tte["event_observed"] == 0, "vehicle_id"].to_numpy()
        n_obs = min(len(observed), max(1, max_vehicles // 2))
        n_cen = min(len(censored), max_vehicles - n_obs)
        keep_ids = set(np.concatenate([
            rng.choice(observed, size=n_obs, replace=False),
            rng.choice(censored, size=n_cen, replace=False)]).tolist())

    frames = []
    for chunk in pd.read_csv(readouts_path, chunksize=200_000):
        chunk = _normalise_columns(chunk)
        if keep_ids is not None:
            chunk = chunk[chunk["vehicle_id"].isin(keep_ids)]
        if len(chunk):
            frames.append(chunk)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    df = df.merge(tte, on="vehicle_id", how="inner")

    if with_specifications:
        specs_path = os.path.join(resolved, f"{split}_specifications.csv")
        if os.path.exists(specs_path):
            specs = _normalise_columns(pd.read_csv(specs_path))
            # Spec columns are categorical strings; encode as integer codes so
            # they can be used without inventing an ordering in the loader.
            for col in specs.columns:
                if col != "vehicle_id" and specs[col].dtype == object:
                    specs[col] = specs[col].astype("category").cat.codes
            df = df.merge(specs, on="vehicle_id", how="left")

    # Remaining life at this readout. Censored rows carry a lower bound.
    derived = pd.DataFrame({
        "study_end": df["time_to_event"].to_numpy(),
        "rul": np.clip((df["time_to_event"] - df["time_step"]).to_numpy(), 0.0, None),
    }, index=df.index)
    df = pd.concat([df, derived], axis=1)

    df.attrs["label_provenance"] = LABEL_PROVENANCE
    df.attrs["split"] = split
    # The corpus censoring rate, recorded before any subsampling, so a
    # stratified slice can never be mistaken for the fleet's real balance.
    df.attrs["corpus_censored_pct"] = float(
        100.0 * (tte["event_observed"] == 0).mean())
    df.attrs["corpus_vehicles"] = int(len(tte))
    df.attrs["subsampled"] = keep_ids is not None
    return df


def feature_columns(df: pd.DataFrame) -> list:
    """Numeric model inputs.

    Excludes the identifier, the labels and the derived RUL. ``time_step`` is
    also excluded: ``rul`` is defined as ``study_end - time_step``, so leaving
    the clock in lets the model reconstruct the target from it.
    """
    excluded = set(NON_FEATURE_COLUMNS) | {"time_step"}
    return [c for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def uncensored(df: pd.DataFrame) -> pd.DataFrame:
    """Only vehicles whose component repair was actually observed.

    90.4% of the fleet is censored, so this is a small slice -- but a censored
    ``time_to_event`` is a lower bound on life, not a life, and must not be fed
    to a plain regression target.
    """
    if "event_observed" not in df.columns:
        return df
    return df[df["event_observed"] == 1].copy()


def censoring_summary(df: pd.DataFrame) -> dict:
    """How much of the corpus is censored -- required context for any metric."""
    if "event_observed" not in df.columns:
        return {"observed": len(df), "censored": 0, "censored_pct": 0.0}
    per_vehicle = df.drop_duplicates("vehicle_id") if "vehicle_id" in df else df
    observed = int((per_vehicle["event_observed"] == 1).sum())
    censored = int(len(per_vehicle) - observed)
    summary = {
        "rows": int(len(df)),
        "vehicles": int(len(per_vehicle)),
        "vehicles_observed": observed,
        "vehicles_censored": censored,
        "censored_pct": float(100.0 * censored / max(len(per_vehicle), 1)),
    }
    # Loading may stratify the sample, so the slice's balance is not the
    # fleet's. Carry the corpus figure through so it cannot be misread.
    if "corpus_censored_pct" in df.attrs:
        summary["corpus_censored_pct"] = float(df.attrs["corpus_censored_pct"])
        summary["corpus_vehicles"] = int(df.attrs["corpus_vehicles"])
        summary["sample_is_stratified"] = bool(df.attrs.get("subsampled"))
    return summary
