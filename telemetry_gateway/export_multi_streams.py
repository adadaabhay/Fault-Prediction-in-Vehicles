"""Multi-Subsystem Telemetry Stream Exporter.

Packages real hardware testbed recordings (Deutz engine bench, ZeMA hydraulic
rig, MetroPT-3 metro APU) alongside the 58t MBT multi-physics simulation into
the record format the dashboard consumes.

Honesty contract
----------------
Every stream declares, per record set:

``channels_measured``  channels carrying values read from the source recording
``channels_synthetic`` channels present only to satisfy the model input schema
``health_provenance``  where the health curve comes from

Two rules follow, and the dashboard depends on them:

1. A channel is written from the source file whenever the source has it.  The
   Deutz bench ships real ``T_clnt``, ``T_oil``, ``T_3``/``T_40`` and ``p_rail``;
   synthesising coolant/oil/exhaust temperature from a load ramp while those
   columns sit unread is not acceptable.
2. Health curves are derived from ground truth (rig condition annotations,
   company failure reports) or from a model residual -- never from an invented
   severity ramp, and never from a threshold on the channel being displayed.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.engine_deutz import load_deutz_nrtc_data, load_deutz_residuals
from pipelines.hydraulics_zema import load_zema_hydraulic_data, degradation_ordered_sample
from pipelines.apu_metropt import load_metropt_episodes, METROPT3_FAILURES

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "Fault-Prediction-in-Vehicles" / "docs"


def _load_config():
    with open(DOCS / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _blank_record(cfg):
    """Schema-complete record with synthetic placeholders.

    The model input vector is fixed-width, so absent channels need *a* value.
    They are enumerated in ``channels_synthetic`` so the dashboard can mark them
    unavailable rather than presenting a midpoint as a measurement.
    """
    rec = {}
    for k in cfg["input_features"]:
        s = cfg["scaler"][k]
        rec[k] = (s["min"] + s["max"]) * 0.5
    return rec


def _finish(records, measured, cfg):
    """Split channels into measured vs synthetic for the stream metadata.

    Returns three lists: schema channels backed by the source recording,
    schema channels that are placeholders, and real source channels carried
    outside the fixed model-input schema (which the dashboard can display even
    though the LSTM does not consume them).
    """
    schema = set(cfg["input_features"])
    measured = sorted(set(measured))
    synthetic = sorted(schema - set(measured))
    emitted = set()
    for r in records:
        emitted |= set(r.keys())
    extra = sorted(emitted - schema - {"time", "step", "rpm", "load", "terrain"})
    return measured, synthetic, extra


def _health_block(part_order, per_part):
    """Expand a {part: array} mapping into the full 8-part health dict."""
    out = {}
    for p in part_order:
        out[p] = [round(float(v), 1) for v in per_part[p]]
    return out


# ---------------------------------------------------------------------------
# 1. Deutz TCD 12.0 V6 -- real bench measurements, healthy transient cycle
# ---------------------------------------------------------------------------
def build_deutz_stream(cfg, n_max=500):
    df = load_deutz_nrtc_data(source="testbench")
    resid = load_deutz_residuals()
    n = min(len(df), n_max)
    df = df.iloc[:n].reset_index(drop=True)
    resid = resid.iloc[:n].reset_index(drop=True)

    # Model-vs-measurement deviation index over the boost/turbine channels.
    # This is a genuine model-based FDIR residual, not an injected ramp.
    resid_cols = [c for c in ("p_22_resid", "p_3_resid", "n_tc_resid", "T_22_resid")
                  if c in resid.columns]
    if resid_cols:
        z = np.zeros(n)
        for c in resid_cols:
            v = resid[c].abs().values
            scale = np.percentile(v, 95) or 1.0
            z += np.clip(v / scale, 0.0, 1.5)
        deviation = z / len(resid_cols)
    else:
        deviation = np.zeros(n)
    engine_health = np.clip(100.0 - deviation * 45.0, 30.0, 100.0)

    records, measured = [], set()
    for i in range(n):
        row = df.iloc[i]
        rec = _blank_record(cfg)

        def put(key, col, transform=lambda v: v):
            if col in df.columns and pd.notna(row[col]):
                rec[key] = float(transform(row[col]))
                measured.add(key)

        put("coolant_temp", "T_clnt")          # real coolant temperature
        put("oil_temp", "T_oil")               # real oil temperature
        put("exhaust_temp", "T_3")             # real pre-turbine EGT
        put("lambda", "lambda_phys")           # real, physically masked
        put("shaft_torque", "M_l")             # real load torque
        rec["rpm"] = float(row.get("n_eng", 0.0))
        rec["load"] = float(np.clip(row.get("engine_power_kw", 0.0) / 400.0, 0.0, 1.0))
        rec["terrain"] = 0.0                   # engine dyno: no terrain
        rec["time"] = float(row.get("Time", i * 0.1))
        rec["step"] = float(i)
        if "p_rail" in df.columns:
            rec["rail_pressure"] = float(row["p_rail"])
        if "p_22" in df.columns:
            rec["boost_pressure"] = float(row["p_22"])
        if "T_40" in df.columns:
            rec["egt_post_turbine"] = float(row["T_40"])
        records.append(rec)

    meas, synth, extra = _finish(records, measured, cfg)
    part_order = [p for p in cfg["part_order"]]
    per_part = {p: (engine_health if p in ("engine", "cooling", "powertrain", "overall")
                    else np.full(n, 100.0)) for p in part_order}

    return {
        "meta": {
            "name": "Deutz TCD 12.0 V6 - NRTC Transient Bench Cycle",
            "description": ("Real engine test-bench measurements over the Non-Road "
                            "Transient Cycle. The engine is HEALTHY throughout: this "
                            "stream supplies a transient duty cycle, not a fault. The "
                            "health curve is a CFD-vs-bench model residual, not a "
                            "degradation label."),
            "faults": [],
            "source": "Zenodo 5766940 (physical test bench)",
            "channels_measured": meas,
            "channels_synthetic": synth,
            "channels_measured_extra": extra,
            "health_provenance": "model_residual_cfd_vs_bench",
            "sampling_hz": 10.0,
        },
        "records": records,
        "health": _health_block(part_order, per_part),
    }


# ---------------------------------------------------------------------------
# 2. ZeMA hydraulic rig -- real annotated condition progression
# ---------------------------------------------------------------------------
def build_zema_stream(cfg, n_max=400):
    df = load_zema_hydraulic_data()
    sub = degradation_ordered_sample(df, min(n_max, len(df)), target="valve_condition")
    n = len(sub)

    records, measured = [], set()
    for i in range(n):
        row = sub.iloc[i]
        rec = _blank_record(cfg)

        # PS1 is the real main-circuit pressure in bar.
        if "PS1_mean" in sub.columns:
            rec["hyd_pressure"] = float(row["PS1_mean"]) * 1e5
            measured.add("hyd_pressure")
        if "TS1_mean" in sub.columns:
            rec["oil_temp"] = float(row["TS1_mean"])
            measured.add("oil_temp")
        if "TS2_mean" in sub.columns:
            rec["coolant_temp"] = float(row["TS2_mean"])
            measured.add("coolant_temp")
        if "FS1_mean" in sub.columns:
            rec["hyd_flow"] = float(row["FS1_mean"])
            measured.add("hyd_flow")
        if "EPS1_mean" in sub.columns:
            rec["mech_power"] = float(row["EPS1_mean"])
        rec["rpm"] = 1500.0
        rec["load"] = 0.65
        rec["terrain"] = 0.0
        rec["step"] = float(i)
        rec["time"] = float(i * 60.0)          # one 60 s working cycle per record
        # Real annotated rig condition, carried through for display.
        rec["zema_valve_condition"] = float(row["valve_condition"])
        rec["zema_cooler_condition"] = float(row["cooler_condition"])
        rec["zema_pump_leakage"] = float(row["pump_leakage"])
        records.append(rec)

    # Health straight from the rig annotations: valve 100->73, cooler 100->3.
    valve = sub["valve_condition"].values.astype(float)
    cooler = sub["cooler_condition"].values.astype(float)
    pump = sub["pump_leakage"].values.astype(float)
    valve_h = (valve - 73.0) / (100.0 - 73.0) * 100.0
    cooler_h = (cooler - 3.0) / (100.0 - 3.0) * 100.0
    pump_h = (2.0 - pump) / 2.0 * 100.0
    hyd_h = np.clip(0.5 * valve_h + 0.3 * pump_h + 0.2 * cooler_h, 0.0, 100.0)

    part_order = [p for p in cfg["part_order"]]
    per_part = {}
    for p in part_order:
        if p == "hydraulics":
            per_part[p] = hyd_h
        elif p == "cooling":
            per_part[p] = np.clip(cooler_h, 0.0, 100.0)
        elif p == "lubrication":
            per_part[p] = np.clip(pump_h, 0.0, 100.0)
        elif p == "overall":
            per_part[p] = np.clip(0.6 * hyd_h + 0.4 * cooler_h, 0.0, 100.0)
        else:
            per_part[p] = np.full(n, 100.0)

    meas, synth, extra = _finish(records, measured, cfg)
    return {
        "meta": {
            "name": "ZeMA Hydraulic Rig - Annotated Valve & Cooler Degradation",
            "description": ("Real 2,205-cycle hydraulic test rig. Cycles are sampled "
                            "across every annotated condition level and ordered "
                            "healthy->failed, so the valve progression 100 -> 90 -> 80 "
                            "-> 73 shown here is measured ground truth from "
                            "profile.txt, not an injected ramp."),
            "faults": ["valve_hysteresis_lag", "cooler_efficiency_loss", "pump_leakage"],
            "source": "UCI 447 (ZeMA physical rig)",
            "channels_measured": meas,
            "channels_synthetic": synth,
            "channels_measured_extra": extra,
            "health_provenance": "ground_truth_rig_profile_annotations",
            "sampling_hz": 1.0 / 60.0,
        },
        "records": records,
        "health": _health_block(part_order, per_part),
    }


# ---------------------------------------------------------------------------
# 3. MetroPT-3 APU -- real reported air-leak failure episode
# ---------------------------------------------------------------------------
def build_metropt_stream(cfg, episode=3, n_max=600):
    df = load_metropt_episodes()
    ep = df[df["episode"] == episode].reset_index(drop=True)
    if len(ep) > n_max:                       # even decimation preserves the shape
        ep = ep.iloc[:: max(1, len(ep) // n_max)].head(n_max).reset_index(drop=True)
    n = len(ep)

    records, measured = [], set()
    for i in range(n):
        row = ep.iloc[i]
        rec = _blank_record(cfg)
        # Real pneumatic channels, kept in their own units and named honestly.
        for key, col in (("apu_tp2_bar", "TP2"), ("apu_tp3_bar", "TP3"),
                         ("apu_reservoir_bar", "Reservoirs"),
                         ("apu_dv_pressure_bar", "DV_pressure"),
                         ("apu_h1_bar", "H1")):
            if col in ep.columns:
                rec[key] = float(row[col])
        if "Oil_temperature" in ep.columns:
            rec["oil_temp"] = float(row["Oil_temperature"])
            measured.add("oil_temp")
        if "Motor_current" in ep.columns:
            rec["apu_motor_current_a"] = float(row["Motor_current"])
        rec["rpm"] = 0.0                       # stationary compressor, not an engine
        rec["load"] = float(row.get("motor_state_loaded", 0.0))
        rec["terrain"] = 0.0
        rec["step"] = float(i)
        rec["time"] = float(i * 10.0)          # 10 s sampling interval
        rec["apu_failure_active"] = float(row["apu_failure"])
        records.append(rec)

    # Health from the company failure report window: healthy outside, degraded
    # inside the reported air-leak interval.
    fail = ep["apu_failure"].values.astype(float)
    apu_h = np.where(fail > 0, 35.0, 97.0)

    part_order = [p for p in cfg["part_order"]]
    per_part = {}
    for p in part_order:
        if p in ("hydraulics", "overall"):
            per_part[p] = apu_h
        else:
            per_part[p] = np.full(n, 100.0)

    w = next(x for x in METROPT3_FAILURES if x["id"] == episode)
    meas, synth, extra = _finish(records, measured, cfg)
    return {
        "meta": {
            "name": f"MetroPT-3 APU - Reported Air Leak #{episode}",
            "description": (f"Real metro-train Air Production Unit telemetry spanning "
                            f"reported air-leak failure #{episode} "
                            f"({w['start']} to {w['end']}, {w['severity']}). The health "
                            f"curve marks the company-reported failure window; it is "
                            f"not a threshold on any displayed channel."),
            "faults": ["air_leak"],
            "source": "UCI 791 / Metro do Porto (in-service vehicle)",
            "channels_measured": meas,
            "channels_synthetic": synth,
            "channels_measured_extra": extra,
            "health_provenance": "ground_truth_company_failure_reports",
            "failure_window": {"start": w["start"], "end": w["end"],
                               "severity": w["severity"]},
            "sampling_hz": 0.1,
        },
        "records": records,
        "health": _health_block(part_order, per_part),
    }


def export_all_streams(out_path: str | None = None):
    out_path = out_path or str(DOCS / "live_multi_streams.json")
    print(f"Exporting multi-subsystem telemetry streams to {out_path}...")
    cfg = _load_config()

    # Streams produced elsewhere (e.g. tank_sim.cvrde.cvrde_generator) must
    # survive a re-export; this exporter owns only the ids it builds below.
    preserved, preserved_meta = {}, []
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                prior = json.load(f)
            owned = {"sim_mbt", "real_deutz", "real_zema", "real_metropt"}
            preserved = {k: v for k, v in prior.get("streams", {}).items()
                         if k not in owned}
            preserved_meta = [m for m in prior.get("metadata", {}).get("streams_available", [])
                              if m.get("id") in preserved]
        except (json.JSONDecodeError, OSError):
            preserved, preserved_meta = {}, []

    multi = {
        "metadata": {
            "version": "3.0",
            "provenance_policy": (
                "Each stream lists channels_measured (read from the source "
                "recording) and channels_synthetic (schema placeholders, not "
                "measurements). Health curves cite their provenance."),
            "streams_available": [],
        },
        "streams": {},
    }

    sim_path = DOCS / "live_stream.json"
    if sim_path.exists():
        with open(sim_path, "r", encoding="utf-8") as f:
            multi["streams"]["sim_mbt"] = json.load(f)
        multi["metadata"]["streams_available"].append(
            {"id": "sim_mbt", "label": "58-Ton MBT Multi-Physics (Sim)",
             "origin": "First-principles physics twin", "kind": "simulation"})
        print("  -> loaded 58t MBT multi-physics simulation stream")

    builders = [
        ("real_deutz", "Deutz TCD 12.0 V6 (Bench)", "Zenodo 5766940 engine test bench",
         build_deutz_stream),
        ("real_zema", "ZeMA Hydraulics (Rig)", "UCI 447 ZeMA hydraulic test rig",
         build_zema_stream),
        ("real_metropt", "MetroPT-3 APU (In-Service)", "UCI 791 Metro do Porto APU",
         build_metropt_stream),
    ]
    for sid, label, origin, fn in builders:
        stream = fn(cfg)
        multi["streams"][sid] = stream
        multi["metadata"]["streams_available"].append(
            {"id": sid, "label": label, "origin": origin, "kind": "measured"})
        m = stream["meta"]
        print(f"  -> {sid}: {len(stream['records'])} records, "
              f"{len(m['channels_measured'])}+{len(m['channels_measured_extra'])} measured / "
              f"{len(m['channels_synthetic'])} synthetic channels, "
              f"health={m['health_provenance']}")

    for sid, stream in preserved.items():
        multi["streams"][sid] = stream
        print(f"  -> preserved externally-generated stream: {sid}")
    multi["metadata"]["streams_available"].extend(preserved_meta)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(multi, f)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.2f} MB)")
    return multi


if __name__ == "__main__":
    export_all_streams()
