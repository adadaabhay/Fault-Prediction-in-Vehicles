"""ZeMA Hydraulic Condition Monitoring Pipeline (UCI 447).

Extracts statistical time-domain features from 20 raw sensor streams across
2,205 repeating hydraulic machine cycles.

Ground truth
------------
Condition labels come from ``profile.txt`` -- physically annotated rig state,
recorded independently of the sensor channels.  This is genuine ground truth
and the strongest label provenance in the project.

Sampling warning
----------------
The cycles are ordered by experimental block, NOT shuffled.  Taking a head
slice (``df.iloc[:n]``) therefore lands inside a single condition block: the
first 200 cycles contain only ``valve_condition == 100`` (optimal) and only
``cooler_condition == 3``, i.e. no valve fault variation at all.  Always draw
subsets with :func:`stratified_cycle_sample`.
"""

import os
import numpy as np
import pandas as pd
from pipelines._paths import resolve as _resolve


def load_zema_hydraulic_data(data_dir: str = None) -> pd.DataFrame:
    """Loads and extracts statistical features from the ZeMA hydraulic dataset."""
    if data_dir is None:
        data_dir = _resolve("condition+monitoring+of+hydraulic+systems")
    profile_path = os.path.join(data_dir, "profile.txt")
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"profile.txt not found in {data_dir}")

    # Load targets from profile.txt
    # Col 0: Cooler condition (3: close to fail, 20: reduced, 100: full)
    # Col 1: Valve condition (100: optimal, 90: small lag, 80: severe lag, 73: close to failure)
    # Col 2: Internal pump leakage (0: none, 1: weak, 2: severe)
    # Col 3: Hydraulic accumulator (130: optimal, 115: slightly reduced, 100: severely reduced, 90: close to fail)
    # Col 4: Stable flag (0: conditions were stable, 1: static conditions not reached)
    profiles = np.loadtxt(profile_path)
    n_cycles = profiles.shape[0]

    features = {}
    features["cycle_id"] = np.arange(n_cycles)
    features["cooler_condition"] = profiles[:, 0]
    features["valve_condition"] = profiles[:, 1]
    features["pump_leakage"] = profiles[:, 2]
    features["accumulator_bar"] = profiles[:, 3]
    features["stable_flag"] = profiles[:, 4]

    # Binary failure targets for ML classification
    features["cooler_fault"] = (profiles[:, 0] < 100).astype(int)
    features["valve_fault"] = (profiles[:, 1] < 100).astype(int)
    features["pump_fault"] = (profiles[:, 2] > 0).astype(int)
    features["accumulator_fault"] = (profiles[:, 3] < 130).astype(int)

    # Core sensors to extract: Pressures (PS1, PS2), Motor Power (EPS1), Flow (FS1), Temperatures (TS1, TS2)
    sensor_files = {
        "PS1": "PS1.txt",   # Main pressure (bar)
        "PS2": "PS2.txt",   # Secondary pressure
        "EPS1": "EPS1.txt", # Motor electric power (W)
        "FS1": "FS1.txt",   # Volume flow rate (l/min)
        "TS1": "TS1.txt",   # Temperature 1 (°C)
        "TS2": "TS2.txt",   # Temperature 2 (°C)
    }

    for sensor_name, fname in sensor_files.items():
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            data = np.loadtxt(fpath)
            features[f"{sensor_name}_mean"] = np.mean(data, axis=1)
            features[f"{sensor_name}_std"] = np.std(data, axis=1)
            features[f"{sensor_name}_max"] = np.max(data, axis=1)
            features[f"{sensor_name}_min"] = np.min(data, axis=1)
            features[f"{sensor_name}_p2p"] = features[f"{sensor_name}_max"] - features[f"{sensor_name}_min"]
            features[f"{sensor_name}_rms"] = np.sqrt(np.mean(data**2, axis=1))

    return pd.DataFrame(features)


LABEL_PROVENANCE = "ground_truth_rig_profile"

TARGET_COLUMNS = ("cooler_condition", "valve_condition",
                  "pump_leakage", "accumulator_bar")

BINARY_TARGETS = ("cooler_fault", "valve_fault", "pump_fault", "accumulator_fault")


def feature_columns(df: pd.DataFrame) -> list:
    """Sensor-derived model inputs, excluding every condition/label column."""
    return [c for c in df.columns
            if c.endswith(("_mean", "_std", "_max", "_min", "_p2p", "_rms"))]


def stratified_cycle_sample(df: pd.DataFrame, n: int,
                            stratify_by: str | list = "valve_condition",
                            random_state: int = 42) -> pd.DataFrame:
    """Draw ``n`` cycles spanning every level of ``stratify_by``.

    Head-slicing this dataset silently selects a single experimental block (see
    module docstring), which removes the degradation the sample is meant to
    demonstrate.  This allocates the budget evenly across the distinct label
    levels and returns the result in cycle order so the series stays readable
    as a progression.
    """
    keys = [stratify_by] if isinstance(stratify_by, str) else list(stratify_by)
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(f"stratify_by column(s) not in frame: {missing}")

    groups = list(df.groupby(keys, sort=True))
    if not groups:
        return df.head(n).copy()

    rng = np.random.default_rng(random_state)
    per = max(1, n // len(groups))
    picks = []
    for _, g in groups:
        take = min(per, len(g))
        picks.append(g.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))

    out = pd.concat(picks)
    if len(out) < n:  # top up from whatever is left, without replacement
        rest = df.drop(index=out.index)
        if len(rest):
            out = pd.concat([out, rest.sample(n=min(n - len(out), len(rest)),
                                              random_state=random_state)])
    return out.sort_index().head(n).copy()


def degradation_ordered_sample(df: pd.DataFrame, n: int,
                               target: str = "valve_condition") -> pd.DataFrame:
    """``n`` cycles ordered healthy -> failed for the given condition target.

    Produces a monotonically degrading sequence suitable for a dashboard replay
    or an RUL demonstration, using real annotated condition levels rather than
    an injected ramp.
    """
    if target not in df.columns:
        raise KeyError(f"{target!r} not in frame")
    ordered = df.sort_values(target, ascending=(target == "pump_leakage"))
    sub = stratified_cycle_sample(ordered, n, stratify_by=target)
    return sub.sort_values(target, ascending=(target == "pump_leakage")).reset_index(drop=True)
