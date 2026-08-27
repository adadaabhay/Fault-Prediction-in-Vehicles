"""Scenario generation: create labelled training data from the physics
simulator, with per-part health indices, piecewise-linear RUL targets
and fault-class labels, plus the demo live stream for the dashboard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from sim.config import TankConfig
from sim.faults import FaultManager
from sim.tank import TankSimulator, MissionStep, default_mission

from .parts import (
    FAIL_HEALTH,
    INPUT_FEATURES,
    PART_ORDER,
    RUL_CAP_STEPS,
    overall_health_index,
    part_health_index,
)

ALL_FAULTS = sorted(FaultManager.FAULT_MAP.keys())


# Duty-cycle profiles. The suite previously ran every scenario through one
# mission (`default_mission`), so "held-out" meant a different fault onset time
# on an otherwise identical run -- and each held-out scenario had four
# near-identical siblings sitting in the training set. That measures
# memorisation of onset timing, not generalisation. Holding out a whole
# (fault x duty profile) cell gives the evaluation something the model has
# genuinely not seen.
#
# Note the honest limit of this: it demonstrates generalisation across duty
# cycle and onset within one simulator and one vehicle. It says nothing about
# transfer to real hardware. The real-corpus evidence lives in
# benchmark/evaluate_subsystems.py and is the only sim-to-real claim this
# project can make.
MISSION_PROFILES: dict[str, list[tuple]] = {
    # (duration_s, rpm, load, terrain)
    "road_march": [(60, 800, 0.10, 0.10), (300, 1500, 0.40, 0.15),
                   (240, 1800, 0.50, 0.20), (120, 900, 0.15, 0.10)],
    "cross_country": [(60, 900, 0.20, 0.30), (180, 1700, 0.55, 0.80),
                      (180, 2100, 0.75, 0.90), (120, 1400, 0.40, 0.60),
                      (60, 800, 0.10, 0.20)],
    "assault": [(45, 1000, 0.25, 0.20), (120, 2200, 0.80, 0.35),
                (150, 2600, 0.95, 0.55), (105, 1900, 0.65, 0.45),
                (60, 850, 0.12, 0.15)],
    "convoy_idle": [(240, 800, 0.08, 0.05), (120, 1200, 0.25, 0.10),
                    (180, 850, 0.10, 0.05), (60, 1600, 0.45, 0.15)],
    "recovery_tow": [(90, 1100, 0.35, 0.40), (240, 1400, 0.85, 0.70),
                     (150, 1250, 0.90, 0.75), (120, 900, 0.20, 0.30)],
}


@dataclass
class Scenario:
    name: str
    faults: list[tuple[str, float]]  # (fault name, start fraction)
    seed: int
    steps: int = 2400
    profile: str = "default"


PROFILE_NAMES = tuple(MISSION_PROFILES)


def scenario_suite() -> list[Scenario]:
    """Healthy runs, per-fault runs across onsets and duty profiles, and
    multi-fault combinations."""
    suite: list[Scenario] = []
    for i, prof in enumerate(PROFILE_NAMES):
        for k in range(1):
            suite.append(Scenario(f"healthy_{prof}_{k}", [],
                                  seed=100 + 10 * i + k, profile=prof))
    for j, fault in enumerate(ALL_FAULTS):
        for pi, prof in enumerate(PROFILE_NAMES):
            for k, frac in enumerate((0.32, 0.58)):
                suite.append(Scenario(
                    f"single_{fault}_{prof}_{k}", [(fault, frac)],
                    seed=200 + 100 * j + 10 * pi + k, profile=prof))
    combos = [("combo_wear_overheat", [("bearing_wear", 0.38),
                                       ("cooling_failure", 0.55)]),
              ("combo_wear_hyd", [("bearing_wear", 0.40),
                                  ("hydraulic_valve_fault", 0.58)]),
              ("combo_lube_fatigue", [("oil_pump_degradation", 0.42),
                                      ("torsion_fatigue", 0.60)]),
              ("combo_lube_seal", [("seal_leakage", 0.45),
                                   ("oil_pump_degradation", 0.60)])]
    for ci, (name, faults) in enumerate(combos):
        prof = PROFILE_NAMES[ci % len(PROFILE_NAMES)]
        suite.append(Scenario(f"{name}_{prof}", faults,
                              seed=900 + ci, profile=prof))
    return suite


def scenario_group(sc: Scenario) -> str:
    """Grouping key for the split: one (fault-family, duty-profile) cell.

    Every scenario sharing a key lands in the same split, so a held-out
    scenario has no near-duplicate sibling in training.
    """
    family = sc.faults[0][0] if sc.faults else "healthy"
    if len(sc.faults) > 1:
        family = "combo"
    return f"{family}|{sc.profile}"


def split_suite(suite: list[Scenario], seed: int = 11
                ) -> dict[str, list[int]]:
    """Three-way group split: train / val / test.

    There were only two splits before, and `train.py` selected `best_params` by
    val loss and then reported the same set as "Evaluation (held-out
    scenarios)". Selecting on a set and reporting on it makes the reported
    number optimistically biased -- it is a training statistic, not an estimate
    of generalisation. `test` below is touched exactly once, after selection.

    Every fault class is kept present in `train`: holding out a whole class
    would make its classification accuracy vacuously zero rather than
    informative. What is held out is the (class, duty profile) cell.
    """
    by_family: dict[str, list[str]] = {}
    for sc in suite:
        fam = scenario_group(sc).split("|")[0]
        key = scenario_group(sc)
        by_family.setdefault(fam, [])
        if key not in by_family[fam]:
            by_family[fam].append(key)

    rng = np.random.default_rng(seed)
    assign: dict[str, str] = {}
    for fam, keys in by_family.items():
        keys = sorted(keys)
        if len(keys) >= 4:
            # One cell to val, one to test, the rest to train (~60/20/20).
            order = list(rng.permutation(len(keys)))
            assign[keys[order[0]]] = "val"
            assign[keys[order[1]]] = "test"
            for i in order[2:]:
                assign[keys[i]] = "train"
        elif len(keys) >= 2:
            order = list(rng.permutation(len(keys)))
            assign[keys[order[0]]] = "test"
            for i in order[1:]:
                assign[keys[i]] = "train"
        else:                       # too few cells to hold one out
            for k in keys:
                assign[k] = "train"

    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, sc in enumerate(suite):
        out[assign.get(scenario_group(sc), "train")].append(i)
    return out


def _mission_for_steps(steps: int, dt: float,
                       profile: str = "default") -> list[MissionStep]:
    if profile in MISSION_PROFILES:
        base = [MissionStep(*row) for row in MISSION_PROFILES[profile]]
    else:
        base = default_mission(TankConfig())
    total_duration = steps * dt
    base_duration = sum(m.duration_s for m in base)
    factor = total_duration / base_duration
    return [MissionStep(m.duration_s * factor, m.rpm, m.load, m.terrain)
            for m in base]


def run_scenario(scenario: Scenario, window_samples: int = 512,
                 sample_rate: float = 500.0) -> tuple[list[dict], np.ndarray]:
    """Simulate one scenario; return (records, fault_label_per_step)."""
    cfg = TankConfig()
    cfg.window_samples = window_samples
    cfg.sample_rate = sample_rate
    rng = np.random.default_rng(scenario.seed)
    fm = FaultManager(rng)
    for name, start_frac in scenario.faults:
        fm.add(name, start_step=int(scenario.steps * start_frac),
               ramp_steps=max(int(scenario.steps * 0.22), 50))
    sim = TankSimulator(cfg, faults=fm,
                        mission=_mission_for_steps(scenario.steps, cfg.dt,
                                                   scenario.profile),
                        seed=scenario.seed)
    records = sim.run()[: scenario.steps]
    # A label is only asserted once the fault is actually *observable*.
    # `active_faults` flips at start_step, where the sigmoid severity is ~0 and
    # the signature is far below the noise floor, so thousands of samples were
    # labelled with a fault that had not yet manifested. `observable_faults`
    # additionally requires the severity to clear DETECTABLE_SEVERITY.
    #
    # `observable_faults` returns a list that may carry more than one fault for
    # combo scenarios. The classifier is single-label (C is fixed by the
    # declared fault taxonomy, not by what is co-firing in a window), so we
    # pick the alphabetically first observable fault to keep the assignment
    # stable across RNG state. Combo scenarios are tracked separately in
    # `per_scenario` metadata so the HUD can surface the co-fault without
    # distorting the classification head.
    labels = []
    for i in range(len(records)):
        obs = sim.faults.observable_faults(i)
        labels.append(sorted(obs)[0] if obs else "healthy")
    return records, np.array(labels)


def part_health_series(records: list[dict]) -> dict[str, np.ndarray]:
    """Per-part health over a run; ``overall`` is fused from the subsystems."""
    out: dict[str, np.ndarray] = {}
    for part in PART_ORDER:
        if part == "overall":
            continue
        out[part] = np.array([part_health_index(part, r) for r in records])
    if "overall" in PART_ORDER:
        subs = [p for p in PART_ORDER if p != "overall"]
        out["overall"] = np.array([
            overall_health_index({p: out[p][i] for p in subs})
            for i in range(len(records))])
    return out


def rul_labels(health_series: dict[str, np.ndarray]) -> np.ndarray:
    """Piecewise-linear RUL (steps) per part, capped, normalised to [0,1].

    A part that never crosses the failure threshold keeps full RUL (1.0);
    otherwise RUL ramps down to 0 as the failure point approaches.
    """
    n = len(health_series[PART_ORDER[0]])
    out = np.zeros((n, len(PART_ORDER)))
    for j, part in enumerate(PART_ORDER):
        h = health_series[part]
        fail_idx = np.where(h < FAIL_HEALTH)[0]
        if fail_idx.size == 0:
            out[:, j] = 1.0
            continue
        first_fail = int(fail_idx[0])
        rul = np.clip(first_fail - np.arange(n), 0, RUL_CAP_STEPS) / RUL_CAP_STEPS
        out[:, j] = rul
    return out


def feature_matrix(records: list[dict]) -> np.ndarray:
    return np.array([[r[k] for k in INPUT_FEATURES] for r in records], dtype=float)


# Count-like channels sit at their floor for most of a mission and spike only
# during a fault, so plain min-max puts ~80-90% of their mass at exactly 0 and
# throws away the dynamic range that matters.  These are log-compressed first.
LOG_SCALED_FEATURES = {
    "debris_rate", "debris_cumulative", "ae_event_rate", "ae_energy",
    "vib_dom_amp", "vib_rms",
}


def _forward(values: np.ndarray, key: str) -> np.ndarray:
    if key in LOG_SCALED_FEATURES:
        return np.log1p(np.clip(values, 0.0, None))
    return values


def build_scaler(features_all: np.ndarray) -> dict:
    """Percentile-based scaler fitted over the whole scenario suite.

    Two departures from plain min-max, both of which were causing the model to
    see constants where the physics had signal:

    * fitted across every scenario, not just the first (a healthy one, which
      left several channels with ``min == max``);
    * 1st/99th percentile bounds rather than absolute extrema, so a single
      impulse cannot compress the rest of the channel into a sliver.
    """
    scaler: dict[str, dict] = {}
    for j, k in enumerate(INPUT_FEATURES):
        col = _forward(features_all[:, j], k)
        lo = float(np.percentile(col, 1.0))
        hi = float(np.percentile(col, 99.0))
        if hi - lo < 1e-9:                      # fall back to full extent
            lo, hi = float(np.min(col)), float(np.max(col))
        scaler[k] = {"min": lo, "max": hi,
                     "log": bool(k in LOG_SCALED_FEATURES)}
    return scaler


def normalise(features: np.ndarray, scaler: dict) -> np.ndarray:
    out = np.zeros_like(features)
    for j, k in enumerate(INPUT_FEATURES):
        mn, mx = scaler[k]["min"], scaler[k]["max"]
        col = _forward(features[:, j], k) if scaler[k].get("log") else features[:, j]
        out[:, j] = np.clip((col - mn) / max(mx - mn, 1e-9), 0, 1)
    return out


def make_windows(features: np.ndarray, rul: np.ndarray, labels: np.ndarray,
                 window: int, stride: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y, L = [], [], []
    for i in range(0, len(features) - window + 1, stride):
        X.append(features[i:i + window])
        Y.append(rul[i + window - 1])
        L.append(labels[i + window - 1])
    return np.array(X), np.array(Y), np.array(L)


def build_dataset(window_samples: int = 512, sample_rate: float = 500.0,
                  window: int = 40, stride: int = 6,
                  demo_steps: int = 3000, demo_faults: list[tuple] | None = None,
                  demo_seed: int = 7) -> dict:
    """Generate everything: training arrays, scaler, demo stream."""
    suite = scenario_suite()
    raw_feats, all_rul, all_labels, all_records = [], [], [], []
    all_feats, all_win_rul, all_win_labels = [], [], []
    scaler = None

    for sc in suite:
        records, labels = run_scenario(sc, window_samples, sample_rate)
        feats = feature_matrix(records)
        health = part_health_series(records)
        rul = rul_labels(health)
        raw_feats.append(feats)
        all_rul.append(rul)
        all_labels.append(labels)
        all_records.append(records)

    splits = split_suite(suite)

    # The scaler is fitted on the TRAINING scenarios only.
    #
    # It used to be fitted on `np.concatenate(raw_feats)` -- the whole suite,
    # before splitting -- so the 1st/99th percentile bounds of every channel
    # carried information from the validation and test scenarios into the
    # normalisation the model trained under. `benchmark/evaluate_subsystems.py`
    # already got this right for the real corpora (StandardScaler fitted inside
    # each fold); the synthetic pipeline did not.
    #
    # It must still span more than one scenario: fitting on a single healthy
    # run left `debris_rate` and `ae_event_rate` with min == max, so every
    # fault value saturated to a constant 1.0.
    scaler = build_scaler(np.concatenate([raw_feats[i] for i in splits["train"]]))
    for feats, rul, labels in zip(raw_feats, all_rul, all_labels):
        feats_n = normalise(feats, scaler)
        X, Y, L = make_windows(feats_n, rul, labels, window, stride)
        all_feats.append(X)
        all_win_rul.append(Y)
        all_win_labels.append(L)

    Y = np.concatenate(all_win_rul)
    class_names = ["healthy"] + ALL_FAULTS
    class_index = {name: i for i, name in enumerate(class_names)}
    L = np.array([class_index[lbl] for lbl in np.concatenate(all_win_labels)])

    per_scenario = [
        {"name": sc.name, "profile": sc.profile,
         "group": scenario_group(sc), "X": x, "Y": y,
         "L": np.array([class_index[lbl] for lbl in l]),
         # The classifier is single-label (C is the declared fault taxonomy);
         # co-firing faults in combo scenarios are surfaced here so the HUD
         # can render them without distorting the classification head.
         "combo": tuple(sorted({f for f, _ in sc.faults}))}
        for sc, x, y, l in zip(suite, all_feats, all_win_rul, all_win_labels)
    ]

    # Demo live stream: healthy -> staged fault progression.
    if demo_faults is None:
        demo_faults = [("bearing_wear", 0.25), ("cooling_failure", 0.48),
                       ("hydraulic_valve_fault", 0.68), ("torsion_fatigue", 0.85)]
    demo = Scenario("demo_live", demo_faults, seed=demo_seed,
                    steps=demo_steps, profile="cross_country")
    demo_records, _ = run_scenario(demo, window_samples, sample_rate)
    demo_health = part_health_series(demo_records)

    stream = {
        "records": demo_records,
        "health": {p: demo_health[p].tolist() for p in PART_ORDER},
        "meta": {"name": demo.name, "steps": len(demo_records),
                 "dt": TankConfig().dt,
                 "faults": [f for f, _ in demo_faults]},
    }
    # `X` used to be the loop variable from the make_windows call above, i.e.
    # only the *last* scenario's windows, while `Y`/`labels` were concatenated
    # across all of them -- mismatched shapes waiting for the first caller that
    # used data["X"] instead of data["per_scenario"].
    X_all = np.concatenate(all_feats)
    return {
        "X": X_all, "Y": Y, "labels": L, "scaler": scaler,
        "splits": splits,
        "part_order": PART_ORDER, "stream": stream,
        "class_names": class_names,
        "input_features": INPUT_FEATURES,
        "window": window,
        "per_scenario": per_scenario,
    }


def save_demo(stream: dict, path: str) -> None:
    """Trim the demo records to the fields the dashboard displays."""
    # Every LSTM input feature, plus the extra channels the HUD displays.
    # `susp_compliance` and `driveline_efficiency` were absent, so the exported
    # stream could not satisfy INPUT_FEATURES -- the in-browser per-module RUL
    # (docs/lstm.js) and the gateway's inference block both silently produced
    # nothing on the demo stream.
    needed = set(INPUT_FEATURES) | {
        "rpm", "load", "terrain", "step", "time",
        "coolant_temp", "oil_temp", "exhaust_temp", "lambda", "shaft_torque",
        "vib_rms", "vib_kurtosis", "vib_dom_amp", "oil_pressure",
        "oil_viscosity", "debris_rate", "debris_cumulative", "coolant_level",
        "exhaust_pressure", "hyd_pressure", "hyd_leak_flow", "hyd_flow",
        "susp_load_kN", "susp_strain_ue", "shock_a_rms_g",
        "torsion_twist_deg", "torsion_cumulative_twist", "ae_event_rate",
        "ae_energy", "health_index", "spl_db", "fuel_level",
    }
    trimmed = [{k: round(r[k], 4) if isinstance(r[k], float) else r[k]
                for k in needed if k in r} for r in stream["records"]]
    health = {p: [round(v, 2) for v in arr] for p, arr in stream["health"].items()}
    out = {**stream, "records": trimmed, "health": health}
    with open(path, "w") as fh:
        json.dump(out, fh)


def split_indices(n_seq: int, val_frac: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_seq)
    n_val = int(n_seq * val_frac)
    return idx[n_val:], idx[:n_val]