"""Naval Gas Turbine Propulsion Plant Pipeline (UCI 316, CBM dataset).

Ingests a full-scale simulator of a naval vessel's COGAG propulsion plant,
covering the gas turbine, gas generator, compressor and controllable-pitch
propellers over a sweep of operating profiles and degradation states.

Why this corpus is here
-----------------------
It is the only procured dataset with a **continuous degradation target**:
``GT Compressor decay state coefficient`` and ``GT Turbine decay state
coefficient`` are the experiment's independent design variables, and the 16
sensor channels are the measured consequence.  The labels therefore cannot be
circular with the features -- unlike a threshold rule applied to a channel that
also sits in the feature matrix.

For an MBT programme this covers the gas-turbine powerplant path (AGT1500-class
platforms such as the M1 Abrams), which none of the other procured corpora
reach -- Deutz and MetroPT are both reciprocating machines.

Reference: Coraddu, Oneto, Ghio, Savio, Anguita, Figari, "Machine learning
approaches for improving condition-based maintenance of naval propulsion
plants", Journal of Engineering for the Maritime Environment, 2016.
"""

import os

import numpy as np
import pandas as pd

LABEL_PROVENANCE = "ground_truth_experimental_design"

# Column order is fixed by Features.txt shipped with the dataset.
FEATURE_NAMES = (
    "lever_position",
    "ship_speed_kn",
    "gt_shaft_torque_kNm",
    "gt_rev_rpm",
    "gas_generator_rev_rpm",
    "starboard_propeller_torque_kN",
    "port_propeller_torque_kN",
    "hp_turbine_exit_temp_C",
    "compressor_inlet_temp_C",
    "compressor_outlet_temp_C",
    "hp_turbine_exit_pressure_bar",
    "compressor_inlet_pressure_bar",
    "compressor_outlet_pressure_bar",
    "exhaust_pressure_bar",
    "turbine_injection_control_pct",
    "fuel_flow_kg_s",
)

TARGET_NAMES = ("gt_compressor_decay", "gt_turbine_decay")

# Healthy state is 1.0; the sweep runs down to these values.
COMPRESSOR_DECAY_MIN = 0.95
TURBINE_DECAY_MIN = 0.975


def _find_data_file(data_dir: str) -> str:
    if not os.path.exists(data_dir) and os.path.exists(os.path.join("..", data_dir)):
        data_dir = os.path.join("..", data_dir)
    for root, _dirs, files in os.walk(data_dir):
        if "__MACOSX" in root:
            continue
        for name in files:
            if name.lower() == "data.txt":
                return os.path.join(root, name)
    raise FileNotFoundError(f"data.txt not found under {data_dir}")


def load_naval_propulsion_data(
        data_dir: str = "datasets/procured/naval_propulsion") -> pd.DataFrame:
    """Load the CBM propulsion-plant sweep with engineered turbomachinery features."""
    path = _find_data_file(data_dir)
    raw = np.loadtxt(path)
    if raw.shape[1] != len(FEATURE_NAMES) + len(TARGET_NAMES):
        raise ValueError(
            f"expected {len(FEATURE_NAMES) + len(TARGET_NAMES)} columns, got {raw.shape[1]}")

    df = pd.DataFrame(raw, columns=list(FEATURE_NAMES) + list(TARGET_NAMES))

    # --- turbomachinery features (mirroring the Deutz air-path treatment) ---
    df["compressor_pressure_ratio"] = (df["compressor_outlet_pressure_bar"]
                                       / df["compressor_inlet_pressure_bar"].clip(lower=1e-6))
    df["compressor_temp_rise_C"] = (df["compressor_outlet_temp_C"]
                                    - df["compressor_inlet_temp_C"])
    df["turbine_expansion_ratio"] = (df["hp_turbine_exit_pressure_bar"]
                                     / df["exhaust_pressure_bar"].clip(lower=1e-6))
    # Isentropic-ish efficiency proxy: actual rise vs ideal rise for the ratio.
    ideal_rise = (df["compressor_inlet_temp_C"] + 273.15) * (
        df["compressor_pressure_ratio"] ** (0.4 / 1.4) - 1.0)
    df["compressor_efficiency_proxy"] = ideal_rise / df["compressor_temp_rise_C"].clip(lower=1e-6)

    omega = 2.0 * np.pi * df["gt_rev_rpm"] / 60.0
    df["gt_shaft_power_kW"] = df["gt_shaft_torque_kNm"] * omega
    df["specific_fuel_consumption"] = (df["fuel_flow_kg_s"]
                                       / df["gt_shaft_power_kW"].clip(lower=1e-6))
    df["propeller_torque_imbalance_kN"] = (df["starboard_propeller_torque_kN"]
                                           - df["port_propeller_torque_kN"])

    # Binary views for classification benchmarks; the continuous coefficients
    # remain available and are the preferred regression targets.
    df["compressor_degraded"] = (df["gt_compressor_decay"] < 1.0).astype(int)
    df["turbine_degraded"] = (df["gt_turbine_decay"] < 1.0).astype(int)

    df.attrs["label_provenance"] = LABEL_PROVENANCE
    return df


def feature_columns(df: pd.DataFrame) -> list:
    """Model inputs: sensor channels plus engineered features, never the targets."""
    excluded = set(TARGET_NAMES) | {"compressor_degraded", "turbine_degraded"}
    return [c for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def degradation_ordered_sample(df: pd.DataFrame, n: int,
                               target: str = "gt_compressor_decay") -> pd.DataFrame:
    """``n`` rows ordered healthy -> most degraded for a dashboard replay.

    Holds the operating point roughly constant (a mid-range lever position) so
    the sequence shows degradation rather than throttle movement.
    """
    if target not in df.columns:
        raise KeyError(target)
    lever = df["lever_position"]
    band = df[(lever >= lever.quantile(0.45)) & (lever <= lever.quantile(0.65))]
    if len(band) < n:
        band = df
    out = band.sort_values(target, ascending=False)
    step = max(1, len(out) // n)
    return out.iloc[::step].head(n).reset_index(drop=True)


def health_from_decay(df: pd.DataFrame, target: str = "gt_compressor_decay") -> np.ndarray:
    """Map the decay coefficient onto a 0-100 health scale.

    1.0 is a fresh machine; the sweep floor is the fully degraded end of the
    experiment.  This is a direct rescaling of ground truth, not an inferred
    or injected curve.
    """
    floor = COMPRESSOR_DECAY_MIN if target == "gt_compressor_decay" else TURBINE_DECAY_MIN
    frac = (df[target].values - floor) / max(1.0 - floor, 1e-9)
    return np.clip(frac, 0.0, 1.0) * 100.0
