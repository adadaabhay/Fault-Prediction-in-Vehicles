"""Multi-Subsystem Ground Truth Machine Learning Benchmark.

Evaluates leak-free tree ensembles across the real subsystem corpora and writes
verified metrics to results/subsystems_benchmark.json.

Evaluation rules enforced here
------------------------------
1. **Label provenance is reported next to every metric.**  A corpus whose
   labels are derived from thresholds on its own channels is not a supervised
   benchmark, and is excluded from the scored table rather than quoted.
2. **Leakage guard.**  Any column that participates in a label definition is
   asserted out of the feature matrix before fitting.
3. **Grouped CV where episodes exist.**  MetroPT-3 is a 10-second-interval time
   series; a random shuffle puts adjacent samples in train and test.  Folds are
   split by failure episode instead, so no episode straddles the split.
4. **Degenerate folds are recorded, not papered over.**  A single-class fold
   contributes NaN, never a free 1.0.
5. **Positive counts are reported.**  An AUC over five positives is noise and
   the reader must be able to see that.
"""

import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold

# Contiguous ZeMA cycles per group. The rig holds a fault condition for a run
# of cycles, so neighbouring cycles are near-duplicates; 60 spans several
# condition changes without making the groups so large that folds unbalance.
ZEMA_BLOCK_CYCLES = 60
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, brier_score_loss, average_precision_score,
                             r2_score, mean_absolute_error)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.engine_deutz import (load_deutz_nrtc_data, load_deutz_residuals,
                                    regime_summary, LABEL_PROVENANCE as DEUTZ_PROV,
                                    LABEL_DEFINING_COLUMNS as DEUTZ_LABEL_COLS)
from pipelines.hydraulics_zema import (load_zema_hydraulic_data, feature_columns as zema_features,
                                       BINARY_TARGETS as ZEMA_TARGETS,
                                       LABEL_PROVENANCE as ZEMA_PROV)
from pipelines.apu_metropt import load_metropt_episodes, feature_columns as metropt_features
from pipelines.heavy_scania import load_scania_aps_data
from pipelines.fleet_scania_componentx import (
    LABEL_PROVENANCE as CX_PROV, available as cx_available,
    censoring_summary as cx_censoring, feature_columns as cx_features,
    load_componentx_data, uncensored as cx_uncensored)
from pipelines.naval_gasturbine import (load_naval_propulsion_data,
                                        feature_columns as naval_features,
                                        TARGET_NAMES as NAVAL_TARGETS,
                                        LABEL_PROVENANCE as NAVAL_PROV)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def _make_model(random_state: int = 42):
    if HAS_LGBM:
        return LGBMClassifier(n_estimators=100, learning_rate=0.05,
                              random_state=random_state, verbose=-1)
    return RandomForestClassifier(n_estimators=100, max_depth=8,
                                  random_state=random_state, n_jobs=-1)


def assert_no_label_leakage(feature_cols, label_defining_cols, context: str = "") -> None:
    """Fail loudly if a label-defining channel survived into the feature matrix."""
    leaked = sorted(set(feature_cols) & set(label_defining_cols))
    if leaked:
        raise ValueError(
            f"Label leakage in {context}: {leaked} define the target and must be "
            f"excluded from X, otherwise the model re-derives the labelling rule.")


def evaluate_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
                n_splits: int = 5, random_state: int = 42) -> dict:
    """Leak-free cross validation.

    Scaling is fit strictly on the training fold.  When ``groups`` is supplied
    the split is grouped (no group spans folds); otherwise it is stratified.
    """
    if groups is not None:
        n_groups = len(np.unique(groups))
        splitter = GroupKFold(n_splits=min(n_splits, n_groups))
        splits = list(splitter.split(X, y, groups))
        strategy = (f"GroupKFold by group ({min(n_splits, n_groups)} folds, "
                    f"{n_groups} groups)")
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(X, y))
        strategy = f"StratifiedKFold ({n_splits} folds, shuffled)"

    aucs, prs, f1s, precisions, recalls, briers = [], [], [], [], [], []
    degenerate = 0

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = _make_model(random_state)
        clf.fit(X_train_scaled, y_train)
        probs = clf.predict_proba(X_test_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)

        if len(np.unique(y_test)) > 1:
            aucs.append(float(roc_auc_score(y_test, probs)))
            prs.append(float(average_precision_score(y_test, probs)))
        else:
            degenerate += 1  # recorded, never scored as a free 1.0

        f1s.append(float(f1_score(y_test, preds, zero_division=0)))
        precisions.append(float(precision_score(y_test, preds, zero_division=0)))
        recalls.append(float(recall_score(y_test, preds, zero_division=0)))
        briers.append(float(brier_score_loss(y_test, probs)))

    def _mean(v):
        return float(np.mean(v)) if v else None

    def _std(v):
        return float(np.std(v)) if v else None

    return {
        "cv_strategy": strategy,
        "roc_auc_mean": _mean(aucs), "roc_auc_std": _std(aucs),
        "pr_auc_mean": _mean(prs),
        "f1_mean": _mean(f1s),
        "precision_mean": _mean(precisions),
        "recall_mean": _mean(recalls),
        "brier_score": _mean(briers),
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()),
        "folds_single_class": degenerate,
    }


def evaluate_regression_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                           random_state: int = 42,
                           groups: np.ndarray | None = None) -> dict:
    """Leak-free K-fold regression for continuous degradation targets.

    When ``groups`` is supplied the split is grouped, so correlated rows from
    the same unit (a vehicle's readout series, a failure episode) cannot appear
    in both train and test.
    """
    from sklearn.model_selection import GroupKFold, KFold
    try:
        from lightgbm import LGBMRegressor
        make = lambda: LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                     random_state=random_state, verbose=-1)
    except ImportError:
        make = lambda: RandomForestRegressor(n_estimators=200,
                                             random_state=random_state, n_jobs=-1)

    if groups is not None:
        n_groups = len(np.unique(groups))
        splitter = GroupKFold(n_splits=min(n_splits, n_groups))
        splits = list(splitter.split(X, y, groups))
        strategy = f"GroupKFold ({min(n_splits, n_groups)} folds, grouped)"
    else:
        splits = list(KFold(n_splits=n_splits, shuffle=True,
                            random_state=random_state).split(X))
        strategy = f"KFold ({n_splits} folds, shuffled)"

    r2s, maes = [], []
    for train_idx, test_idx in splits:
        scaler = StandardScaler().fit(X[train_idx])
        model = make().fit(scaler.transform(X[train_idx]), y[train_idx])
        pred = model.predict(scaler.transform(X[test_idx]))
        r2s.append(float(r2_score(y[test_idx], pred)))
        maes.append(float(mean_absolute_error(y[test_idx], pred)))
    return {
        "cv_strategy": strategy,
        "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
        "mae_mean": float(np.mean(maes)),
        "target_span": float(np.max(y) - np.min(y)),
        "mae_pct_of_span": float(np.mean(maes) / max(np.max(y) - np.min(y), 1e-9) * 100.0),
        "n_samples": int(len(y)),
        "n_distinct_levels": int(len(np.unique(y))),
    }


def run_all_benchmarks():
    print("=" * 80)
    print("MULTI-SUBSYSTEM GROUND TRUTH ML BENCHMARK")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "LightGBM" if HAS_LGBM else "RandomForest",
            "scoring_policy": (
                "Only corpora with independent ground-truth labels are scored. "
                "Label-defining channels are asserted out of X. Grouped splits "
                "are used wherever a temporal key exists (ZeMA cycle blocks, "
                "MetroPT episodes, Component X vehicles); the per-entry "
                "cv_strategy field is authoritative. Single-class folds are "
                "counted, not scored."),
            "scoring_policy_history": (
                "This field previously asserted 'time series are split by "
                "episode, never shuffled' while four ZeMA targets, Scania and "
                "both naval targets in fact ran shuffled K-fold. ZeMA is now "
                "grouped by contiguous cycle block. Scania APS rows are "
                "independent per-vehicle snapshots with no temporal ordering, "
                "so stratified shuffling is correct there and is stated as "
                "such."),
        },
        "subsystems": {},
        "excluded_from_scoring": {},
    }

    # ---- 1. Turret / actuation hydraulics -- ZeMA rig (real annotated labels) ----
    print("\n[1/4] ZeMA Hydraulic Rig (UCI 447) -- rig-annotated ground truth")
    df_zema = load_zema_hydraulic_data()
    feat_z = zema_features(df_zema)
    assert_no_label_leakage(feat_z, ZEMA_TARGETS + ("cooler_condition", "valve_condition",
                                                    "pump_leakage", "accumulator_bar"),
                            context="ZeMA hydraulics")
    X_z = df_zema[feat_z].values
    # ZeMA cycles are recorded sequentially and the rig's fault conditions are
    # varied in contiguous blocks, so a shuffled StratifiedKFold puts cycles
    # that are neighbours in time on both sides of the split. That is exactly
    # the leakage the metadata policy claims to avoid ("time series are split
    # by episode, never shuffled") -- and it was the reason all four ZeMA
    # targets reported ROC-AUC 0.9991-0.9999.
    #
    # Grouping by contiguous cycle blocks keeps temporal neighbours together.
    zema_groups = (df_zema["cycle_id"].values // ZEMA_BLOCK_CYCLES)
    for target in ZEMA_TARGETS:
        y_z = df_zema[target].values
        res = evaluate_cv(X_z, y_z, groups=zema_groups)
        res["label_provenance"] = ZEMA_PROV
        results["subsystems"][f"hydraulics_{target}"] = {
            "dataset": "ZeMA Hydraulic Rig (UCI 447)",
            "subsystem": "Turret traverse / stabiliser hydraulics",
            "features_count": len(feat_z),
            "metrics": res,
        }
        print(f"   {target:22s} AUC={res['roc_auc_mean']:.4f}  F1={res['f1_mean']:.4f}  "
              f"n_pos={res['n_positive']}")

    # ---- 2. APU / pneumatics -- MetroPT-3 (company failure reports) ----
    print("\n[2/4] MetroPT-3 APU (UCI 791) -- company failure reports, episode-grouped")
    for target, horizon, label in (("apu_failure", 24.0, "leak detection (in progress)"),
                                   ("apu_prefailure", 24.0, "24h-ahead prediction")):
        df_m = load_metropt_episodes(prefail_horizon_h=horizon, target=target)
        feat_m = metropt_features(df_m)
        assert_no_label_leakage(feat_m, ("apu_failure", "apu_prefailure",
                                         "apu_system_fault", "failure_id", "episode"),
                                context=f"MetroPT {target}")
        X_m = df_m[feat_m].values
        y_m = df_m[target].values
        res = evaluate_cv(X_m, y_m, groups=df_m["episode"].values, n_splits=4)
        res["label_provenance"] = "ground_truth_company_failure_reports"
        results["subsystems"][f"apu_metropt_{target}"] = {
            "dataset": "MetroPT-3 Train APU (UCI 791)",
            "subsystem": "Auxiliary power unit / pneumatics",
            "task": label,
            "features_count": len(feat_m),
            "metrics": res,
        }
        print(f"   {target:22s} AUC={res['roc_auc_mean']:.4f} +/- {res['roc_auc_std']:.4f}  "
              f"F1={res['f1_mean']:.4f}  recall={res['recall_mean']:.4f}  n_pos={res['n_positive']}")

    # ---- 3. Heavy fleet -- Scania APS (real workshop labels) ----
    print("\n[3/4] Scania APS (UCI 421) -- real maintenance labels")
    # LightGBM handles NaN natively, so skip the global-median impute and
    # avoid leaking a whole-dataset statistic across folds.
    df_s = load_scania_aps_data(max_rows=10000, impute=not HAS_LGBM)
    feat_s = [c for c in df_s.columns if c not in ("class", "aps_failure")]
    assert_no_label_leakage(feat_s, ("class", "aps_failure"), context="Scania APS")
    X_s = df_s[feat_s].values
    y_s = df_s["aps_failure"].values
    res = evaluate_cv(X_s, y_s)
    res["label_provenance"] = "ground_truth_workshop_records"
    res["imputation"] = "global_median" if df_s.attrs.get("imputed") else "none (NaN-aware learner)"
    results["subsystems"]["heavy_fleet_scania"] = {
        "dataset": "Scania APS Trucks (UCI 421)",
        "subsystem": "Air pressure system / heavy fleet",
        "features_count": len(feat_s),
        "metrics": res,
    }
    print(f"   aps_failure            AUC={res['roc_auc_mean']:.4f}  F1={res['f1_mean']:.4f}  "
          f"n_pos={res['n_positive']}")

    # ---- 4. Gas-turbine powerplant -- naval CBM (continuous decay targets) ----
    print("\n[4/4] Naval GT Propulsion (UCI 316) -- experimental decay coefficients")
    df_n = load_naval_propulsion_data()
    feat_n = naval_features(df_n)
    assert_no_label_leakage(feat_n, NAVAL_TARGETS + ("compressor_degraded",
                                                     "turbine_degraded"),
                            context="Naval gas turbine")
    X_n = df_n[feat_n].values
    for target in NAVAL_TARGETS:
        res = evaluate_regression_cv(X_n, df_n[target].values)
        res["label_provenance"] = NAVAL_PROV
        results["subsystems"][f"gasturbine_{target}"] = {
            "dataset": "Naval GT Propulsion Plant (UCI 316)",
            "subsystem": "Gas-turbine powerplant (AGT1500-class)",
            "task": "continuous degradation regression",
            "features_count": len(feat_n),
            "metrics": res,
        }
        print(f"   {target:22s} R2={res['r2_mean']:.4f}  MAE={res['mae_mean']:.6f}  "
              f"({res['mae_pct_of_span']:.2f}% of span)")

    # ---- 5. Fleet component RUL -- SCANIA Component X (time-to-event) ----
    print("\n[5/5] SCANIA Component X -- workshop repair records")
    if cx_available():
        df_cx = load_componentx_data()
        censoring = cx_censoring(df_cx)
        df_obs = cx_uncensored(df_cx)
        feat_cx = cx_features(df_obs)
        assert_no_label_leakage(
            feat_cx, ("time_to_event", "event_observed", "rul", "study_end",
                      "time_step", "vehicle_id"),
            context="Component X")
        res = evaluate_regression_cv(
            df_obs[feat_cx].fillna(0.0).to_numpy(dtype=float),
            df_obs["rul"].to_numpy(dtype=float),
            groups=df_obs["vehicle_id"].to_numpy())
        res["label_provenance"] = CX_PROV
        res["censoring"] = censoring
        res["target"] = "rul = study_end - time_step, uncensored vehicles only"
        res["note"] = ("A censored vehicle's study length is a lower bound on "
                       "component life, not a life; regressing on it biases "
                       "estimates downwards. Folds are grouped by vehicle_id "
                       "because a truck's readouts are highly correlated.")
        results["subsystems"]["fleet_rul_componentx"] = {
            "dataset": "SCANIA Component X (researchdata.se 2024-34)",
            "subsystem": "Fleet component RUL",
            "task": "time-to-event regression",
            "features_count": len(feat_cx),
            "metrics": res,
        }
        print(f"   rul (uncensored)       R2={res['r2_mean']:.4f} +/- {res['r2_std']:.4f}  "
              f"MAE={res['mae_mean']:.2f}")
        print(f"                          {censoring['vehicles_observed']} observed / "
              f"{censoring['vehicles']} sampled vehicles; "
              f"corpus is {censoring['corpus_censored_pct']:.1f}% censored")
    else:
        results["excluded_from_scoring"]["fleet_rul_componentx"] = {
            "dataset": "SCANIA Component X (researchdata.se 2024-34)",
            "subsystem": "Fleet component RUL",
            "reason": ("Not procured. This is the only corpus with real "
                       "time-to-event labels; every other RUL target in the "
                       "project is simulator-derived."),
            "label_provenance": CX_PROV,
            "procurement_url": "https://researchdata.se/en/catalogue/dataset/2024-34",
        }
        print("   NOT PROCURED -- recorded under excluded_from_scoring")

    # ---- Excluded: Deutz carries no fault labels ----
    print("\n[--] Deutz TCD 12.0 V6 (Zenodo 5766940) -- EXCLUDED from scoring")
    df_d = load_deutz_nrtc_data()
    resid = load_deutz_residuals()
    resid_cols = [c for c in resid.columns if c.endswith("_resid")]
    results["excluded_from_scoring"]["engine_core_deutz"] = {
        "dataset": "Deutz TCD 12.0 V6 (Zenodo 5766940)",
        "subsystem": "Engine core / air path",
        "reason": (
            "Source ships no fault labels. The 'combustion_anomaly' indicator is a "
            "threshold rule over lambda and engine_power_kw; scoring a classifier "
            "against it while those channels remain in X re-derives the rule "
            "(ROC-AUC 1.0 over 5 positives) and is not evidence of anything."),
        "label_provenance": DEUTZ_PROV,
        "label_defining_columns": list(DEUTZ_LABEL_COLS),
        "heuristic_positive_count": int(df_d["combustion_anomaly"].sum()),
        "role": "regime supplier + model-vs-measurement residual baseline",
        "regime": regime_summary(df_d),
        "residual_baseline": {
            "shared_channels": len(resid.attrs["shared_channels"]),
            "samples": int(len(resid)),
            "mean_abs_residual_top5": {
                c: float(resid[c].abs().mean())
                for c in resid[resid_cols].abs().mean().sort_values(ascending=False).head(5).index
            },
        },
    }
    print(f"   reason: no ground-truth labels in source "
          f"(heuristic indicator has {int(df_d['combustion_anomaly'].sum())} positives)")
    print(f"   retained as: regime supplier + {len(resid.attrs['shared_channels'])}-channel "
          f"CFD-vs-bench residual baseline")

    # ---- Excluded: AEGIS CAN is a healthy trip ----
    try:
        from pipelines.can_aegis import (load_can_trip, lamp_activity,
                                         duty_cycle_summary,
                                         LABEL_PROVENANCE as AEGIS_PROV)
        la = lamp_activity()
        trip = load_can_trip()
        results["excluded_from_scoring"]["can_bus_aegis"] = {
            "dataset": "AEGIS instrumented vehicle CAN trace",
            "subsystem": "Vehicle data bus / powertrain duty cycle",
            "reason": ("Every diagnostic lamp and warning channel reads zero for "
                       "the whole trip, so there is no fault to predict. Verified "
                       "rather than assumed -- see lamp_channels_active."),
            "label_provenance": AEGIS_PROV,
            "lamp_channels_checked": int(len(la)),
            "lamp_channels_active": int((la["active_samples"] > 0).sum()) if len(la) else 0,
            "role": "real CAN framing for the J1939 gateway + road duty cycle",
            "duty_cycle": duty_cycle_summary(trip),
        }
        print("\n[--] AEGIS CAN trace -- EXCLUDED from scoring")
        print(f"   reason: all {len(la)} lamp/warning channels inactive (healthy trip)")
        print("   retained as: real CAN signal source for the J1939 gateway")
    except (ImportError, FileNotFoundError, ValueError) as exc:
        print(f"\n[--] AEGIS CAN trace unavailable: {exc}")

    os.makedirs("results", exist_ok=True)
    out_path = "results/subsystems_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE -> {out_path}")
    print(f"  scored subsystems : {len(results['subsystems'])}")
    print(f"  excluded          : {len(results['excluded_from_scoring'])}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_all_benchmarks()
