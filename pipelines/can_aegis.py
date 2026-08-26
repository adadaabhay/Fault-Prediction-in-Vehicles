"""AEGIS Real-Vehicle CAN Bus Trace Pipeline.

Ingests instrumented on-road driving traces recorded from a production
vehicle's CAN bus (115 signals, ~34 Hz, ~71 minutes per trip).

Label provenance -- read before reporting any metric
----------------------------------------------------
**This corpus carries no fault labels.**  Every diagnostic lamp and warning
channel in the trace (``ENG_OBD2_Lamp``, ``ENG_Systemlamp``, ``ENG_Hotlamp``,
``WIV_Oilpr_Warn_Engine``, ``SCS_Error`` and the rest) reads zero for the whole
trip, i.e. the vehicle was healthy throughout.  :func:`lamp_activity` verifies
this rather than assuming it.

Its value is therefore twofold, and neither involves supervised fault metrics:

1. **Real CAN framing and real signal values** to exercise the J1939 gateway
   (``telemetry_gateway``) end to end, instead of synthetic frames.
2. **A real road duty cycle** -- engine speed, pedal, torque demand, boost and
   coolant temperature under genuine driver behaviour -- to use as an
   excitation profile, the same role Deutz plays for the transient bench cycle.

If a trip *is* found with lamp activity, those channels become genuine
on-vehicle fault ground truth; :func:`lamp_activity` is how you check.
"""

import os
import glob

import numpy as np
import pandas as pd

LABEL_PROVENANCE = "no_labels_healthy_trip"

# Signals worth surfacing for PHM work, mapped onto the project's naming.
CAN_SIGNAL_MAP = {
    "EngineSpeed_CAN": "rpm",
    "EngineTemperature": "coolant_temp",
    "BoostPressure": "boost_pressure_bar",
    "AirIntakeTemperature": "air_intake_temp_C",
    "AmbientTemperature": "ambient_temp_C",
    "AccPedal": "accel_pedal_pct",
    "ENG_Trq_m_ex": "engine_torque_Nm",
    "ENG_Trq_DMD": "torque_demand_Nm",
    "ENG_TorqueIntegral": "torque_integral",
    "FuelConsumption": "fuel_consumption",
}

# Channels whose non-zero value would indicate a real on-vehicle fault.
LAMP_CHANNELS = (
    "ENG_OBD2_Lamp", "ENG_Systemlamp", "ENG_Hotlamp", "ENG_Particle_Lamp",
    "ENG_Avus_Engineprotect", "ENG_FuelTankCap_Lamp", "ENG_Text_Error_Fuelsys",
    "ENG_Text_MSG_Service", "WIV_Oilpr_Warn_Engine", "WIV_Oilmin_Warn",
    "WIV_Sensorerror", "WIV_Underfill_Warn", "WIV_Overfill_Warn",
    "SCS_Error", "SCS_Tiptronic_Error", "GBX_Kickdown_Error",
)


def _require_h5py():
    try:
        import h5py  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "h5py is required to read AEGIS CAN traces: pip install h5py") from exc
    import h5py
    return h5py


def find_trips(data_dir: str = "datasets/procured/aegis_can_bus") -> list:
    """Every .hdf trip file under ``data_dir``."""
    return sorted(glob.glob(os.path.join(data_dir, "*.hdf")))


def load_can_trip(path: str | None = None,
                  data_dir: str = "datasets/procured/aegis_can_bus",
                  signals: dict | None = None,
                  resample_hz: float | None = 10.0) -> pd.DataFrame:
    """Load one CAN trip into a time-indexed frame.

    Each HDF dataset is stored as ``(N, 2)`` with **column 0 the value and
    column 1 the timestamp** -- note the ordering, which is the reverse of the
    usual convention and silently yields a timebase in place of every signal if
    assumed the other way round.
    """
    h5py = _require_h5py()
    if path is None:
        trips = find_trips(data_dir)
        if not trips:
            raise FileNotFoundError(f"no .hdf trips found in {data_dir}")
        path = trips[0]

    signals = signals or CAN_SIGNAL_MAP
    with h5py.File(path, "r") as f:
        if "CAN" not in f:
            raise ValueError(f"{path} has no CAN group")
        can = f["CAN"]
        cols, time_axis = {}, None
        for raw_name, out_name in signals.items():
            if raw_name not in can:
                continue
            arr = np.asarray(can[raw_name])
            cols[out_name] = arr[:, 0].astype(float)
            if time_axis is None:
                time_axis = arr[:, 1].astype(float)

    if not cols:
        raise ValueError(f"none of the requested signals present in {path}")

    df = pd.DataFrame(cols)
    df.insert(0, "time_s", time_axis)
    df["trip"] = os.path.splitext(os.path.basename(path))[0]

    if resample_hz:
        step = max(1, int(round((1.0 / resample_hz) / max(np.median(np.diff(time_axis)), 1e-9))))
        df = df.iloc[::step].reset_index(drop=True)

    df.attrs["label_provenance"] = LABEL_PROVENANCE
    df.attrs["source_file"] = path
    return df


def lamp_activity(path: str | None = None,
                  data_dir: str = "datasets/procured/aegis_can_bus") -> pd.DataFrame:
    """Per-lamp activation counts for a trip.

    An all-zero result means the trip is fault-free, so the trace may be used
    as a duty-cycle/regime source but not as a supervised fault corpus.
    """
    h5py = _require_h5py()
    if path is None:
        trips = find_trips(data_dir)
        if not trips:
            raise FileNotFoundError(f"no .hdf trips found in {data_dir}")
        path = trips[0]

    rows = []
    with h5py.File(path, "r") as f:
        can = f.get("CAN")
        for name in LAMP_CHANNELS:
            if can is None or name not in can:
                continue
            v = np.asarray(can[name])[:, 0].astype(float)
            rows.append({"channel": name,
                         "active_samples": int(np.nansum(v != 0)),
                         "max_value": float(np.nanmax(v)) if len(v) else 0.0})
    out = pd.DataFrame(rows)
    out.attrs["has_faults"] = bool(len(out) and out["active_samples"].sum() > 0)
    return out


def duty_cycle_summary(df: pd.DataFrame) -> dict:
    """Characterise the driving regime this trip supplies (its actual value)."""
    def rng(col):
        if col not in df.columns:
            return None
        v = df[col].to_numpy()
        v = v[~np.isnan(v)]
        if not len(v):
            return None
        return {"min": float(v.min()), "median": float(np.median(v)),
                "max": float(v.max())}

    dur = float(df["time_s"].max() - df["time_s"].min()) if "time_s" in df else None
    return {
        "trip": str(df["trip"].iloc[0]) if "trip" in df.columns and len(df) else None,
        "samples": int(len(df)),
        "duration_s": dur,
        "duration_min": (dur / 60.0) if dur else None,
        "rpm": rng("rpm"),
        "coolant_temp": rng("coolant_temp"),
        "boost_pressure_bar": rng("boost_pressure_bar"),
        "engine_torque_Nm": rng("engine_torque_Nm"),
        "accel_pedal_pct": rng("accel_pedal_pct"),
        "label_provenance": LABEL_PROVENANCE,
    }


def to_telemetry_frames(df: pd.DataFrame, limit: int | None = None) -> list:
    """Convert CAN rows into gateway telemetry dicts.

    Produces the channel names the plausibility gate and DTC engine expect, so
    a real recorded trip can be pushed through the ingest -> FDIR -> J1939 path
    instead of synthetic frames.
    """
    sub = df if limit is None else df.head(limit)
    frames = []
    for row in sub.itertuples(index=False):
        d = row._asdict()
        frame = {"timestamp_s": float(d.get("time_s", 0.0))}
        if "rpm" in d and not np.isnan(d["rpm"]):
            frame["rpm"] = float(d["rpm"])
        if "coolant_temp" in d and not np.isnan(d["coolant_temp"]):
            frame["coolant_temp"] = float(d["coolant_temp"])
        if "boost_pressure_bar" in d and not np.isnan(d["boost_pressure_bar"]):
            frame["boost_pressure"] = float(d["boost_pressure_bar"]) * 1e5
        if "air_intake_temp_C" in d and not np.isnan(d["air_intake_temp_C"]):
            frame["air_intake_temp"] = float(d["air_intake_temp_C"])
        if "engine_torque_Nm" in d and not np.isnan(d["engine_torque_Nm"]):
            frame["shaft_torque"] = float(d["engine_torque_Nm"])
        if "accel_pedal_pct" in d and not np.isnan(d["accel_pedal_pct"]):
            frame["load"] = float(d["accel_pedal_pct"]) / 100.0
        frames.append(frame)
    return frames
