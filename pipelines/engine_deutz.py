"""Deutz TCD 12.0 V6 Diesel Engine Air Path & Combustion Pipeline.

Ingests testbench Non-Road Transient Cycle (NRTC) measurements paired with
matched CFD/GT-Power simulation output (Zenodo 5766940).

Label provenance -- read before reporting any metric
----------------------------------------------------
**This dataset ships no fault labels.** The engine is healthy throughout; the
value of the corpus is the transient duty cycle (rapid load and speed
excursions with live EGR, wastegate and throttle actuation), not degradation.

Two consequences:

1. ``combustion_anomaly`` is a *heuristic indicator*, not ground truth.  It is
   a threshold rule over ``lambda`` and ``engine_power_kw``.  Training a
   classifier on it while those two channels remain in the feature matrix
   simply re-derives the rule -- it yields ROC-AUC 1.0 and means nothing.
   ``LABEL_PROVENANCE`` is set accordingly and consumers must honour it.
2. The legitimate, non-circular signal here is the **model-vs-measurement
   residual**: the physical bench and the CFD model cover the same NRTC window
   (700-900 s), so measured minus simulated per channel is a model-based FDIR
   residual.  See :func:`load_deutz_residuals`.

Use this corpus as a regime supplier and a residual baseline.  Do not use it as
a supervised fault-classification benchmark.
"""

import os

import numpy as np
import pandas as pd

# Declares that the shipped indicator is derived, not observed.  Consumers
# (e.g. benchmark/evaluate_subsystems.py) must surface this next to any metric.
LABEL_PROVENANCE = "heuristic_derived"

# Channels that define `combustion_anomaly`. Any supervised evaluation of that
# indicator MUST exclude these or the target becomes circular with its features.
LABEL_DEFINING_COLUMNS = ("lambda", "lambda_phys", "lambda_valid", "engine_power_kw")

# Physically meaningful bounds for diesel air-fuel equivalence ratio. Outside
# these the channel is a zero-fuelling artefact, not a measurement.
LAMBDA_PHYSICAL_MIN = 0.5
LAMBDA_PHYSICAL_MAX = 20.0


def load_deutz_nrtc_data(data_dir: str = "datasets/procured/deutz_engine",
                         source: str = "testbench") -> pd.DataFrame:
    """Load the Deutz NRTC cycle and compute thermodynamic/turbomachinery features.

    ``source`` selects ``"testbench"`` (physical bench, 10 Hz) or ``"cfd"``
    (GT-Power simulation, 100 Hz) over the same 700-900 s cycle window.
    """
    names = {"testbench": ("testbench_nrtc.csv", "tb_nrtc.csv"),
             "cfd": ("gt_nrtc.csv",),
             "doe": ("gt_doe.csv",)}
    if source not in names:
        raise ValueError(f"source must be one of {sorted(names)}, got {source!r}")

    if not os.path.exists(data_dir) and os.path.exists(os.path.join("..", data_dir)):
        data_dir = os.path.join("..", data_dir)

    df = None
    for fname in names[source]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    if df is None:
        raise FileNotFoundError(
            f"None of {names[source]} found in {data_dir} for source={source!r}")

    df = df.rename(columns={"Unnamed: 0": "sample_idx"})

    # Mechanical and thermodynamic feature engineering
    if "n_eng" in df.columns and "M_l" in df.columns:
        df["engine_power_kw"] = (2.0 * np.pi * df["n_eng"] * df["M_l"]) / 60000.0
    else:
        df["engine_power_kw"] = 0.0

    if "p_22" in df.columns and "p_0" in df.columns:
        df["boost_pressure_ratio"] = df["p_22"] / np.clip(df["p_0"], 80000.0, 120000.0)
    else:
        df["boost_pressure_ratio"] = 1.0

    if "T_20" in df.columns and "T_22" in df.columns:
        df["intercooler_dT"] = df["T_20"] - df["T_22"]
    else:
        df["intercooler_dT"] = 0.0

    if "p_3" in df.columns and "p_40" in df.columns:
        df["turbine_expansion_ratio"] = df["p_3"] / np.clip(df["p_40"], 80000.0, 150000.0)
    else:
        df["turbine_expansion_ratio"] = 1.0

    # Exhaust-gas temperature drop across the turbine: a documented indicator of
    # compressor fouling / turbine wheel erosion.
    if "T_3" in df.columns and "T_40" in df.columns:
        df["turbine_dT"] = df["T_3"] - df["T_40"]
    else:
        df["turbine_dT"] = 0.0

    # Air-fuel equivalence ratio is computed against fuel flow, so it diverges
    # towards infinity during motoring / overrun when fuelling goes to zero.
    # The raw channel reaches ~1.2e8 in this cycle, so any threshold rule on it
    # must operate on a validity-masked copy rather than the raw column.
    if "lambda" in df.columns:
        lam = df["lambda"]
        df["lambda_valid"] = ((lam > 0.0) & (lam <= LAMBDA_PHYSICAL_MAX)).astype(int)
        df["lambda_phys"] = lam.where(df["lambda_valid"] == 1).clip(
            LAMBDA_PHYSICAL_MIN, LAMBDA_PHYSICAL_MAX)
    else:
        df["lambda_valid"] = 0
        df["lambda_phys"] = np.nan

    # HEURISTIC indicator -- see module docstring. Not ground truth.
    if "lambda" in df.columns:
        df["combustion_anomaly"] = (
            (df["lambda_valid"] == 1)
            & (df["lambda_phys"] < 1.15)
            & (df["engine_power_kw"] > 150.0)).astype(int)
    else:
        df["combustion_anomaly"] = 0
    df.attrs["label_provenance"] = LABEL_PROVENANCE

    return df


def load_deutz_residuals(data_dir: str = "datasets/procured/deutz_engine",
                         channels: list | None = None) -> pd.DataFrame:
    """Model-vs-measurement residuals over the shared NRTC window.

    Interpolates the CFD run onto the physical bench timebase and returns, for
    every shared channel, ``<channel>_resid = measured - simulated`` alongside
    both source values.  A sustained residual excursion is a model-based FDIR
    detection -- unlike :func:`load_deutz_nrtc_data`'s heuristic indicator, it
    does not derive the signal from the channel it is evaluated against.
    """
    tb = load_deutz_nrtc_data(data_dir, source="testbench")
    cfd = load_deutz_nrtc_data(data_dir, source="cfd")
    if "Time" not in tb.columns or "Time" not in cfd.columns:
        raise ValueError("Both NRTC sources must carry a 'Time' column to align.")

    shared = [c for c in tb.columns
              if c in cfd.columns and c != "Time"
              and c != "lambda"  # unbounded at zero fuelling; use lambda_phys
              and pd.api.types.is_numeric_dtype(tb[c])
              and pd.api.types.is_numeric_dtype(cfd[c])]
    if channels is not None:
        shared = [c for c in shared if c in channels]

    cfd_sorted = cfd.sort_values("Time")
    cols = {"Time": tb["Time"].values}
    for c in shared:
        sim = np.interp(tb["Time"].values,
                        cfd_sorted["Time"].values, cfd_sorted[c].values)
        cols[f"{c}_meas"] = tb[c].values
        cols[f"{c}_sim"] = sim
        cols[f"{c}_resid"] = tb[c].values - sim
    out = pd.DataFrame(cols)

    out.attrs["shared_channels"] = shared
    out.attrs["label_provenance"] = "model_residual_unsupervised"
    return out


def regime_summary(df: pd.DataFrame) -> dict:
    """Characterise the duty cycle this corpus supplies (its actual value)."""
    def rng(col):
        return (float(df[col].min()), float(df[col].max())) if col in df.columns else None
    return {
        "samples": int(len(df)),
        "duration_s": float(df["Time"].max() - df["Time"].min()) if "Time" in df.columns else None,
        "engine_speed_rpm": rng("n_eng"),
        "load_torque_Nm": rng("M_l"),
        "power_kw": rng("engine_power_kw"),
        "boost_ratio": rng("boost_pressure_ratio"),
        "lambda": rng("lambda"),
        "label_provenance": LABEL_PROVENANCE,
    }
