"""MetroPT-3 Train Air Production Unit Pipeline (UCI 791).

Ingests compressor pressure, temperature, motor current and air-intake valve
telemetry from an operating metro train APU (Metro do Porto, Feb-Aug 2020).

Ground truth
------------
The published dataset is *unlabeled*.  The only ground truth is the set of
four air-leak failure reports supplied by the operating company, transcribed
below from ``Data Description_Metro.pdf`` which ships with the dataset.
Labels are derived exclusively from those reports -- never from thresholds on
the sensor channels themselves, which would make the target circular with its
own features.

Reference: Veloso, B., Ribeiro, R.P., Pereira, P.M., Gama, J.
"The MetroPT dataset for predictive maintenance." Scientific Data 9(1):764, 2022.
"""

import os

import numpy as np
import pandas as pd
from pipelines._paths import resolve as _resolve

# Company failure reports (authoritative ground truth; see module docstring).
# The source PDF numbers these #1, #1, #3, #4 -- the second is a typo for #2.
METROPT3_FAILURES = (
    {"id": 1, "start": "2020-04-18 00:00:00", "end": "2020-04-18 23:59:00",
     "mode": "air_leak", "severity": "high_stress", "report": ""},
    {"id": 2, "start": "2020-05-29 23:30:00", "end": "2020-05-30 06:00:00",
     "mode": "air_leak", "severity": "high_stress", "report": "maintenance 30 May 12:00"},
    {"id": 3, "start": "2020-06-05 10:00:00", "end": "2020-06-07 14:30:00",
     "mode": "air_leak", "severity": "high_stress", "report": "maintenance 8 Jun 16:00"},
    {"id": 4, "start": "2020-07-15 14:30:00", "end": "2020-07-15 19:00:00",
     "mode": "air_leak", "severity": "high_stress", "report": "maintenance 16 Jul 00:00"},
)

# Documented physical operating levels of the three-phase motor current (A),
# from the dataset description: ~0 off, ~4 offloaded, ~7 under load, ~9 starting.
# Used as fixed physical thresholds -- NOT data-derived quantiles, which would
# leak global statistics across cross-validation folds.
MOTOR_CURRENT_OFF_A = 0.5
MOTOR_CURRENT_OFFLOAD_A = 4.0
MOTOR_CURRENT_LOAD_A = 7.0
MOTOR_CURRENT_START_A = 9.0

# Sensor channels safe to use as model inputs (7 analogue + 8 digital).
ANALOGUE_CHANNELS = ("TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
                     "Motor_current", "Oil_temperature")
DIGITAL_CHANNELS = ("COMP", "DV_eletric", "Towers", "MPG", "LPS",
                    "Pressure_switch", "Oil_level", "Caudal_impulses")


def failure_windows() -> pd.DataFrame:
    """The four reported air-leak failures as a typed DataFrame."""
    df = pd.DataFrame(list(METROPT3_FAILURES))
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def label_from_reports(timestamps: pd.Series, prefail_horizon_h: float = 24.0) -> pd.DataFrame:
    """Derive ground-truth labels for a timestamp series from the failure reports.

    Returns three columns:
      ``apu_failure``     1 while a reported failure is in progress
      ``apu_prefailure``  1 in the ``prefail_horizon_h`` hours preceding onset
                          (the operationally useful predictive-maintenance target)
      ``failure_id``      episode id, or 0 outside any episode; use as the CV
                          grouping key so a single episode cannot straddle folds
    """
    ts = pd.to_datetime(timestamps)
    n = len(ts)
    failure = np.zeros(n, dtype=int)
    prefail = np.zeros(n, dtype=int)
    fid = np.zeros(n, dtype=int)

    horizon = pd.Timedelta(hours=prefail_horizon_h)
    for w in METROPT3_FAILURES:
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        in_fail = (ts >= start) & (ts <= end)
        in_pre = (ts >= start - horizon) & (ts < start)
        failure[in_fail.to_numpy()] = 1
        prefail[in_pre.to_numpy()] = 1
        fid[(in_fail | in_pre).to_numpy()] = w["id"]

    return pd.DataFrame({"apu_failure": failure,
                         "apu_prefailure": prefail,
                         "failure_id": fid}, index=ts.index)


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features that do not encode the label."""
    if "TP2" in df.columns and "TP3" in df.columns:
        df["pressure_differential_bar"] = df["TP2"] - df["TP3"]
    else:
        df["pressure_differential_bar"] = 0.0

    if "Reservoirs" in df.columns and "TP3" in df.columns:
        # Reservoirs should track pneumatic-panel pressure; divergence indicates
        # a downstream leak or restriction.
        df["reservoir_tracking_error_bar"] = df["Reservoirs"] - df["TP3"]
    else:
        df["reservoir_tracking_error_bar"] = 0.0

    if "Motor_current" in df.columns:
        cur = df["Motor_current"]
        df["motor_state_loaded"] = (cur >= MOTOR_CURRENT_LOAD_A).astype(int)
        df["motor_state_starting"] = (cur >= MOTOR_CURRENT_START_A).astype(int)
        df["motor_state_off"] = (cur < MOTOR_CURRENT_OFF_A).astype(int)
    else:
        df["motor_state_loaded"] = 0
        df["motor_state_starting"] = 0
        df["motor_state_off"] = 0

    return df


def load_metropt_apu_data(data_dir: str = None,
                          max_rows: int = 100000,
                          prefail_horizon_h: float = 24.0,
                          usecols: list | None = None) -> pd.DataFrame:
    """Load MetroPT-3 APU telemetry with ground-truth labels from failure reports.

    ``max_rows`` reads only the head of the file and is intended for smoke tests;
    the head predates every reported failure, so use :func:`load_metropt_episodes`
    for anything that needs positive samples.
    """
    if data_dir is None:
        data_dir = _resolve("procured/metropt3_apu")
    csv_path = os.path.join(data_dir, "MetroPT3(AirCompressor).csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"MetroPT3(AirCompressor).csv not found in {data_dir}")

    df = pd.read_csv(csv_path, nrows=max_rows, usecols=usecols)
    df = df.rename(columns={"Unnamed: 0": "row_idx"})

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = pd.concat([df, label_from_reports(df["timestamp"], prefail_horizon_h)], axis=1)
    else:
        df["apu_failure"] = 0
        df["apu_prefailure"] = 0
        df["failure_id"] = 0

    df = _engineer(df)

    # Retained for backward compatibility; now backed by the company reports
    # rather than a percentile of the motor-current channel it predicts from.
    df["apu_system_fault"] = df["apu_failure"]
    return df


def load_metropt_episodes(data_dir: str = None,
                          prefail_horizon_h: float = 24.0,
                          context_h: float = 48.0,
                          target: str = "apu_failure") -> pd.DataFrame:
    """Load only the reported failure episodes plus surrounding healthy context.

    Reads the full 7-month file, then keeps, for each reported failure, the
    window spanning ``context_h`` hours before onset through ``context_h`` hours
    after recovery.  This yields a set that actually contains positives while
    keeping each episode contiguous, so it can be split by ``failure_id`` with
    ``GroupKFold`` instead of being shuffled across folds.
    """
    if data_dir is None:
        data_dir = _resolve("procured/metropt3_apu")
    csv_path = os.path.join(data_dir, "MetroPT3(AirCompressor).csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"MetroPT3(AirCompressor).csv not found in {data_dir}")

    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Unnamed: 0": "row_idx"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    context = pd.Timedelta(hours=context_h)
    horizon = pd.Timedelta(hours=prefail_horizon_h)
    keep = np.zeros(len(df), dtype=bool)
    episode = np.zeros(len(df), dtype=int)
    ts = df["timestamp"]

    for w in METROPT3_FAILURES:
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        lo = start - context - (horizon if target == "apu_prefailure" else pd.Timedelta(0))
        sel = ((ts >= lo) & (ts <= end + context)).to_numpy()
        keep |= sel
        episode[sel] = w["id"]

    out = df.loc[keep].copy()
    out = pd.concat([out, label_from_reports(out["timestamp"], prefail_horizon_h)], axis=1)
    out["episode"] = episode[keep]
    out = _engineer(out)
    out["apu_system_fault"] = out["apu_failure"]
    return out.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list:
    """Model input columns: raw sensor channels plus engineered features.

    Excludes timestamps, row indices, episode keys and every label column, so a
    caller cannot accidentally train on the target.
    """
    excluded = {"timestamp", "row_idx", "episode", "failure_id",
                "apu_failure", "apu_prefailure", "apu_system_fault"}
    return [c for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
