"""Subsystem (part) definitions, parameter thresholds and LSTM input
feature schema shared between the Python pipeline and the web dashboard.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Per-part parameters: used for gauges, digital displays, thresholds and
# per-part health scoring.  Each parameter carries display metadata and
# safety thresholds (warn_lo/warn_hi = yellow zone, crit_lo/crit_hi = red
# zone / SOS).
# ---------------------------------------------------------------------------

PARTS = {
    "engine": {
        "label": "Engine",
        "params": [
            {"key": "coolant_temp", "label": "Coolant Temp", "unit": "°C",
             "min": 0, "max": 150, "warn_hi": 105, "crit_hi": 120,
             "healthy": 92},
            {"key": "oil_temp", "label": "Oil Temp", "unit": "°C",
             "min": 0, "max": 160, "warn_hi": 115, "crit_hi": 135,
             "healthy": 92},
            {"key": "exhaust_temp", "label": "Exhaust Temp", "unit": "°C",
             "min": 0, "max": 800, "warn_hi": 680, "crit_hi": 750,
             "healthy": 560},
            # Load-normalised. Raw lambda is display-only (below): on a
            # quality-governed diesel it ranges 1.4-5.0 with duty alone, so a
            # fixed reference scores idling as a fault. Over-fuelling drives
            # the residual down; an air restriction drives it up.
            {"key": "lambda_residual", "label": "λ Residual", "unit": "",
             "min": 0.3, "max": 2.0, "warn_lo": 0.82, "warn_hi": 1.25,
             "crit_lo": 0.70, "crit_hi": 1.45, "healthy": 1.0,
             "decimals": 3},
            {"key": "lambda", "label": "Air-Fuel Ratio λ", "unit": "",
             "min": 1.0, "max": 6.5, "healthy": 3.0, "decimals": 2,
             "health_exclude": True},
        ],
        "gauge": "coolant_temp",
        "alarm_key": "coolant_temp",
    },
    "powertrain": {
        "label": "Powertrain",
        "params": [
            {"key": "shaft_torque", "label": "Shaft Torque", "unit": "kN·m",
             "min": 0, "max": 5.0, "warn_hi": 3.6, "crit_hi": 4.2,
             "healthy": 1.6, "scale": 0.001, "decimals": 2,
             "health_exclude": True},
            {"key": "driveline_efficiency", "label": "Driveline Eff.", "unit": "%",
             "min": 60, "max": 100, "warn_lo": 88, "crit_lo": 80,
             "healthy": 100, "scale": 100, "decimals": 1},
            {"key": "vib_rms", "label": "Vibration RMS", "unit": "m/s²",
             "min": 0, "max": 6.0, "warn_hi": 0.75, "crit_hi": 1.2,
             "healthy": 0.46},
            {"key": "vib_kurtosis", "label": "Vibration Kurtosis", "unit": "",
             "min": 0, "max": 30, "warn_hi": 4.5, "crit_hi": 8,
             "healthy": 2.0},
            {"key": "vib_dom_amp", "label": "Dominant FFT Amp", "unit": "",
             "min": 0, "max": 300, "warn_hi": 90, "crit_hi": 130,
             "healthy": 33},
        ],
        "gauge": "vib_rms",
        "alarm_key": "vib_kurtosis",
    },
    "lubrication": {
        "label": "Lubrication",
        "params": [
            {"key": "oil_pressure", "label": "Oil Pressure", "unit": "bar",
             "min": 0, "max": 7, "warn_lo": 3.2, "crit_lo": 2.0,
             "healthy": 5.0, "scale": 1e-5},
            {"key": "oil_temp", "label": "Oil Temp", "unit": "°C",
             "min": 0, "max": 160, "warn_hi": 115, "crit_hi": 135,
             "healthy": 92},
            {"key": "debris_rate", "label": "Debris Rate", "unit": "/s",
             "min": 0, "max": 40, "warn_hi": 8, "crit_hi": 15,
             "healthy": 1.0},
            {"key": "debris_cumulative", "label": "Total Debris", "unit": "pts",
             "min": 0, "max": 4000, "warn_hi": 1200, "crit_hi": 2000,
             "healthy": 200, "health_exclude": True},
        ],
        "gauge": "oil_pressure",
        "alarm_key": "oil_pressure",
    },
    "cooling": {
        "label": "Cooling System",
        "params": [
            {"key": "coolant_temp", "label": "Coolant Temp", "unit": "°C",
             "min": 0, "max": 150, "warn_hi": 105, "crit_hi": 120,
             "healthy": 92},
            {"key": "coolant_level", "label": "Coolant Level", "unit": "%",
             "min": 0, "max": 100, "warn_lo": 55, "crit_lo": 35,
             "healthy": 95, "scale": 100},
            {"key": "exhaust_pressure", "label": "Exhaust Press.", "unit": "bar",
             "min": 0, "max": 3, "warn_hi": 2.2, "crit_hi": 2.6,
             "healthy": 1.29, "scale": 1e-5},
        ],
        "gauge": "coolant_temp",
        "alarm_key": "coolant_temp",
    },
    "hydraulics": {
        "label": "Hydraulics",
        "params": [
            {"key": "hyd_pressure", "label": "Circuit Pressure", "unit": "bar",
             "min": 0, "max": 300, "warn_lo": 140, "crit_lo": 90,
             "healthy": 210, "scale": 1e-5},
            {"key": "hyd_force", "label": "Actuator Force", "unit": "kN",
             "min": 0, "max": 6, "warn_lo": 2.9, "crit_lo": 2.0,
             "healthy": 4.2, "scale": 1e-3},
            {"key": "hyd_leak_flow", "label": "Seal Leak Flow", "unit": "L/s",
             "min": 0, "max": 0.08, "warn_hi": 0.025, "crit_hi": 0.045,
             "healthy": 0.0087, "scale": 1000, "decimals": 4},
        ],
        "gauge": "hyd_pressure",
        "alarm_key": "hyd_pressure",
    },
    "suspension": {
        "label": "Suspension",
        "params": [
            {"key": "susp_load_kN", "label": "Road-Wheel Load", "unit": "kN",
             "min": 0, "max": 130, "warn_hi": 88, "crit_hi": 100,
             "healthy": 50, "health_exclude": True},
            {"key": "susp_compliance", "label": "Compliance", "unit": "ue/kN",
             "min": 0, "max": 4.0, "warn_hi": 1.8, "crit_hi": 2.3,
             "healthy": 1.25, "decimals": 2},
            {"key": "susp_strain_ue", "label": "Strain", "unit": "με",
             "min": 0, "max": 200, "warn_hi": 110, "crit_hi": 125,
             "healthy": 62, "health_exclude": True},
            {"key": "shock_a_rms_g", "label": "Shock RMS", "unit": "g",
             "min": 0, "max": 8, "warn_hi": 3.2, "crit_hi": 4.5,
             "healthy": 1.8, "health_exclude": True},
        ],
        "gauge": "susp_load_kN",
        "alarm_key": "susp_strain_ue",
    },
    "structure": {
        "label": "Structure / Torsion",
        "params": [
            {"key": "torsion_twist_deg", "label": "Torsion Twist", "unit": "°",
             "min": 0, "max": 10.0, "warn_hi": 5.0, "crit_hi": 6.0,
             "healthy": 1.8, "decimals": 2},
            {"key": "torsion_cumulative_twist", "label": "Cum. Twist", "unit": "rad",
             "min": 0, "max": 4000, "warn_hi": 1200, "crit_hi": 2000,
             "healthy": 300, "health_exclude": True},
            {"key": "ae_event_rate", "label": "AE Event Rate", "unit": "/s",
             "min": 0, "max": 40, "warn_hi": 7, "crit_hi": 12,
             "healthy": 2.0},
            {"key": "ae_energy", "label": "AE Energy", "unit": "",
             "min": 0, "max": 40, "warn_hi": 8, "crit_hi": 15,
             "healthy": 0.5},
        ],
        "gauge": "torsion_twist_deg",
        "alarm_key": "ae_event_rate",
    },
    "overall": {
        "label": "Overall Tank",
        "params": [
            {"key": "health_index", "label": "Fused Health Index", "unit": "",
             "min": 0, "max": 100, "warn_lo": 40, "crit_lo": 25,
             "healthy": 90},
            {"key": "spl_db", "label": "Noise SPL", "unit": "dB",
             "min": 60, "max": 150, "warn_hi": 122, "crit_hi": 132,
             "healthy": 107},
            {"key": "fuel_level", "label": "Fuel Level", "unit": "%",
             "min": 0, "max": 100, "warn_lo": 20, "crit_lo": 10,
             "healthy": 90, "scale": 100, "health_exclude": True},
        ],
        "gauge": "health_index",
        "alarm_key": "health_index",
    },
}

PART_ORDER = ["engine", "powertrain", "lubrication", "cooling",
              "hydraulics", "suspension", "structure", "overall"]

# LSTM input features (normalised 0-1) spanning all subsystems.
INPUT_FEATURES = [
    "coolant_temp", "oil_temp", "exhaust_temp", "oil_pressure",
    "oil_viscosity", "debris_rate", "debris_cumulative", "shaft_torque",
    "lambda_residual", "hyd_pressure", "hyd_leak_flow", "susp_load_kN",
    "susp_strain_ue", "torsion_twist_deg", "torsion_cumulative_twist",
    "shock_a_rms_g", "spl_db", "ae_event_rate", "ae_energy", "vib_rms",
    "vib_kurtosis", "vib_dom_amp", "susp_compliance", "driveline_efficiency",
]

# Failure threshold for per-part health index (RUL = time to cross it).
FAIL_HEALTH = 25.0
# RUL cap (steps) for piecewise-linear RUL targets.
RUL_CAP_STEPS = 2400


def part_features(part: str) -> list[str]:
    return [p["key"] for p in PARTS[part]["params"]]


def healthy_reference(part: str) -> dict[str, float]:
    return {p["key"]: p["healthy"] for p in PARTS[part]["params"]}


def threshold_span(part: str) -> dict[str, float]:
    """Distance from the healthy value to the critical threshold, per side.

    Returns ``{key: (span_lo, span_hi)}``.  ``span_hi`` is the headroom above
    the healthy value before ``crit_hi``; ``span_lo`` the headroom below before
    ``crit_lo``.  A side with no critical threshold defined is ``None``, meaning
    deviation in that direction is not a fault.
    """
    span: dict[str, tuple] = {}
    for p in PARTS[part]["params"]:
        healthy = p["healthy"]
        hi = (p["crit_hi"] - healthy) if "crit_hi" in p else None
        lo = (healthy - p["crit_lo"]) if "crit_lo" in p else None
        span[p["key"]] = (lo if (lo or 0) > 0 else None,
                          hi if (hi or 0) > 0 else None)
    return span


def _param_deviation(p: dict, v_disp: float) -> float:
    """One-sided deviation of a parameter, in units of "fraction of the way
    from healthy to critical".

    0.0 means at (or better than) the healthy reference; 1.0 means exactly at
    the critical threshold.  Deviating in the *safe* direction scores 0 -- the
    previous implementation used ``abs()``, so oil pressure above its healthy
    value was penalised exactly as hard as oil pressure below it.
    """
    healthy = p["healthy"]
    hi = (p["crit_hi"] - healthy) if "crit_hi" in p else None
    lo = (healthy - p["crit_lo"]) if "crit_lo" in p else None
    dev = 0.0
    if v_disp > healthy and hi and hi > 0:
        dev = (v_disp - healthy) / hi
    elif v_disp < healthy and lo and lo > 0:
        dev = (healthy - v_disp) / lo
    return max(dev, 0.0)


def part_health_index(part: str, values: dict[str, float]) -> float:
    """0-100 health score for one part from its parameter values.

    Anchored to the documented thresholds: a part whose worst parameter sits
    exactly at its critical threshold scores ``FAIL_HEALTH`` by construction, so
    the health curve and the threshold alarms can never disagree.

    Parameters flagged ``health_exclude`` are omitted.  Those are monotonically
    accumulating counters (total debris, cumulative twist) and consumables
    (fuel level): including them made health a proxy for elapsed mission time,
    so a fault-free run still decayed steadily towards failure.
    """
    devs = []
    for p in PARTS[part]["params"]:
        if p.get("health_exclude"):
            continue
        v = values.get(p["key"])
        if v is None:
            continue
        v_disp = v * p.get("scale", 1.0)
        devs.append(min(_param_deviation(p, v_disp), 3.0))
    if not devs:
        return 100.0
    # Weight the worst channel heavily: one subsystem parameter at critical is
    # a failure, not something to be averaged away by healthy siblings.
    agg = 0.7 * max(devs) + 0.3 * (sum(devs) / len(devs))
    scale = (100.0 - FAIL_HEALTH) / 100.0        # dev == 1.0 -> FAIL_HEALTH
    return float(np_clip(100.0 * (1.0 - scale * agg), 0.0, 100.0))


def overall_health_index(part_healths: dict[str, float]) -> float:
    """Composite vehicle health from the subsystem healths.

    ``overall`` used to be scored from its own ``health_index`` parameter, which
    is never present in a raw sensor record -- so it silently fell back to SPL
    and fuel level and never responded to any fault.  It is now a fusion of the
    subsystems, which is what a composite health index is.
    """
    vals = [v for k, v in part_healths.items()
            if k != "overall" and v is not None]
    if not vals:
        return 100.0
    return float(np_clip(0.6 * min(vals) + 0.4 * (sum(vals) / len(vals)),
                         0.0, 100.0))


def np_clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x