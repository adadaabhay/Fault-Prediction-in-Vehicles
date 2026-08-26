"""Train the LSTM RUL/fault model on physics-simulated tank scenarios and
export everything the GitHub-Pages dashboard needs:

    docs/model.json        LSTM weights (browser forward pass)
    docs/config.json       parts, thresholds, scaler, meta
    docs/live_stream.json  demo telemetry replay
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .lstm import LSTMModel, predict_rul, train
from .constants import DEFAULT_W_CLS, DEFAULT_W_REG
from .parts import (INPUT_FEATURES, PART_ORDER, RUL_CAP_STEPS, PARTS,
                    FAIL_HEALTH)
from .scenarios import build_dataset, save_demo
from sim.config import TankConfig

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _report(model, sets, verbose: bool = False) -> dict:
    """Score a list of scenarios. Returns pooled RUL MAE (steps) and accuracy."""
    abs_err, correct, total = [], 0, 0
    for s in sets:
        preds = [predict_rul(model, s["X"][i]) for i in range(len(s["X"]))]
        rul_pred = np.stack([p[0] for p in preds])
        cls_pred = np.argmax(np.stack([p[1] for p in preds]), axis=1)
        err = np.abs(rul_pred - s["Y"])
        abs_err.append(err)
        correct += int(np.sum(cls_pred == s["L"]))
        total += len(s["L"])
        if verbose:
            per_part = np.mean(err, axis=0) * RUL_CAP_STEPS
            parts_str = ", ".join(f"{p}={v:.0f}"
                                  for p, v in zip(PART_ORDER, per_part))
            print(f"  {s['name']:<34} RUL MAE={np.mean(err)*RUL_CAP_STEPS:6.1f} "
                  f"({parts_str})  acc={np.mean(cls_pred == s['L']):.2f}")
    pooled = np.concatenate(abs_err) if abs_err else np.zeros((1, 1))
    return {"rul_mae": float(np.mean(pooled) * RUL_CAP_STEPS),
            "cls_acc": float(correct / max(total, 1)),
            "n_windows": int(total)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--demo-steps", type=int, default=2000)
    ap.add_argument("--w-reg", type=float, default=2.0)
    ap.add_argument("--w-cls", type=float, default=DEFAULT_W_CLS,
                    help="classification loss weight")
    ap.add_argument("--quick", action="store_true",
                    help="small dataset for smoke testing")
    args = ap.parse_args()

    print("Generating physics-simulated scenarios ...")
    data = build_dataset(window_samples=256 if args.quick else 512,
                         sample_rate=4000.0,
                         window=args.window, stride=args.stride,
                         demo_steps=args.demo_steps)
    if args.quick:
        # Instead of breaking splits by just slicing per_scenario, regenerate trivial splits
        # for the sliced 4-element list so indices remain valid.
        data["per_scenario"] = data["per_scenario"][:4]
        data["splits"] = {"train": [0, 1], "val": [2], "test": [3]}

    n_classes = len(data["class_names"])
    model = LSTMModel(D=len(INPUT_FEATURES), H=args.hidden,
                      R=len(PART_ORDER), C=n_classes, seed=0)

    per = data["per_scenario"]
    splits = data["splits"]
    train_sets = [per[i] for i in splits["train"]]
    val_sets = [per[i] for i in splits["val"]]
    test_sets = [per[i] for i in splits["test"]]
    if not val_sets or not test_sets:
        raise SystemExit("split produced an empty val or test set")
    print(f"scenarios: train={len(train_sets)} val={len(val_sets)} "
          f"test={len(test_sets)}")
    print(f"windows:   train={sum(len(s['X']) for s in train_sets)} "
          f"val={sum(len(s['X']) for s in val_sets)} "
          f"test={sum(len(s['X']) for s in test_sets)}")

    print("Training LSTM (RUL regression + fault classification) ...")
    train(model, train_sets, val_sets, epochs=args.epochs,
          w_reg=args.w_reg, w_cls=args.w_cls)

    # ---- evaluation --------------------------------------------------------
    # `val_sets` selected the checkpoint, so numbers on it are a training
    # statistic, not an estimate of generalisation. `test_sets` is scored
    # exactly once, here, after selection has finished. Only the test figure
    # may be quoted. Previously there was no test set: the same `val_sets` was
    # used to pick `best_params` and then printed under the heading
    # "Evaluation (held-out scenarios)".
    print("\nVALIDATION (drove checkpoint selection -- NOT generalisation):")
    val_summary = _report(model, val_sets)
    print(f"  val  RUL MAE={val_summary['rul_mae']:.1f} steps  "
          f"cls_acc={val_summary['cls_acc']:.3f}")

    print("\nTEST (held-out (fault x duty-profile) cells, scored once):")
    test_summary = _report(model, test_sets, verbose=True)
    print(f"  test RUL MAE={test_summary['rul_mae']:.1f} steps  "
          f"cls_acc={test_summary['cls_acc']:.3f}")

    # ---- export ----------------------------------------------------------
    DOCS.mkdir(exist_ok=True)
    model.save(str(DOCS / "model.json"))
    save_demo(data["stream"], str(DOCS / "live_stream.json"))
    config = {
        "parts": PARTS,
        "part_order": PART_ORDER,
        "input_features": INPUT_FEATURES,
        "scaler": data["scaler"],
        "class_names": data["class_names"],
        "rul_cap_steps": RUL_CAP_STEPS,
        "fail_health": FAIL_HEALTH,
        "window": args.window,
        "dt": TankConfig().dt,
        "model": {"hidden": args.hidden},
        # Recorded so the dashboard can state what the model actually scored
        # and on what, rather than implying the validation number is a
        # generalisation estimate.
        "evaluation": {
            "val": val_summary,
            "test": test_summary,
            "val_groups": sorted({s["group"] for s in val_sets}),
            "test_groups": sorted({s["group"] for s in test_sets}),
            "note": "val drove checkpoint selection; test was scored once "
                    "after selection. Quote test only.",
        },
    }
    with open(DOCS / "config.json", "w") as fh:
        json.dump(config, fh)
    print(f"\nExported docs/model.json, docs/config.json, "
          f"docs/live_stream.json ({len(data['stream']['records'])} steps)")


if __name__ == "__main__":
    main()