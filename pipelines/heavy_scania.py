"""Scania APS Component Failure Pipeline (UCI 421).
Ingests high-dimensional heavy commercial vehicle operational records.
Implements missing value handling and cost matrix target evaluation.
"""

import os
import numpy as np
import pandas as pd
from pipelines._paths import resolve as _resolve


LABEL_PROVENANCE = "ground_truth_workshop_records"


def load_scania_aps_data(data_dir: str = None,
                         max_rows: int = 15000,
                         impute: bool = True) -> pd.DataFrame:
    """Loads and preprocesses the Scania APS tabular failure dataset.

    ``impute`` fills missing values with the column median computed over the
    whole frame.  That is a global statistic and therefore leaks a little
    information across cross-validation folds; pass ``impute=False`` to keep
    NaNs and let a NaN-aware learner (LightGBM, HistGradientBoosting) handle
    them inside each fold instead.
    """
    if data_dir is None:
        data_dir = _resolve("procured/scania_aps")
    train_path = os.path.join(data_dir, "aps_failure_training_set.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"aps_failure_training_set.csv not found in {data_dir}")

    # The CSV has metadata in first 20 rows, actual header on row 21
    df = pd.read_csv(train_path, skiprows=20, nrows=max_rows, na_values="na")
    df = df.copy()

    # Encode binary target: 'pos' -> 1, 'neg' -> 0
    if "class" in df.columns:
        df["aps_failure"] = (df["class"] == "pos").astype(int)
        feature_cols = [c for c in df.columns if c not in ("class", "aps_failure")]
    else:
        feature_cols = list(df.columns)

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if impute:
        for col in feature_cols:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

    df.attrs["label_provenance"] = LABEL_PROVENANCE
    df.attrs["imputed"] = bool(impute)
    return df
