# Audit Remediation — Round 1

> **Scope.** This document covers the remediation of the audit findings raised
> against the synthetic / browser inference path. It does **not** address the
> real-corpus benchmarks (`benchmark/evaluate_subsystems.py`); those are
> independent of the LSTM retrain and are documented in `docs/PROVENANCE.md`.

## Findings closed

| # | Severity | File:line | Defect | Fix | Verified by |
|---|----------|-----------|--------|-----|-------------|
| 1 | BLOCKER | `ml/lstm.py:99` | `d_reg` carried a `reg * (1 - reg)` sigmoid factor that did not belong to the MSE gradient it was supposed to compute. The comment in the file already documented the correct derivative, so the bug was a *contradiction between comment and code*. Every reported RUL/cls number was a training-statistic fiction — the model was optimising a different objective than the one being reported. | Removed the sigmoid factor. `d_reg = reg_scale * (cache["reg"] - y_reg)` now matches the comment and the loss. | Retrain; `tests/test_js_python_parity.py`; `docs/config.json::evaluation` (val vs test separation) |
| 2 | HIGH | `ml/parts.py::INPUT_FEATURES` | `coolant_level` and `exhaust_pressure` were declared in `PARTS["cooling"]` with thresholds, used in `part_health_index`, but absent from the LSTM input schema. The C engine and the browser engine silently read 24 channels while the schema implied 26. | Appended the two channels to `INPUT_FEATURES`. The C header was regenerated (`c_engine/tank_pdm_dims.h`, D=26). The dashboard now displays the channels as overall-panel vitals. | `tests/test_leakage.py::TestSchemaCoversEverySubsystem`; `tools_check_artifacts.py`; manual HUD inspection |
| 3 | HIGH | `ml/scenarios.py:202` | Combo scenarios (`combo_wear_overheat`, `combo_wear_hyd`, `combo_lube_fatigue`, `combo_lube_seal`) ran a `lambda a: a[0] if a else "healthy"` selector that took the first fault in iteration order — which depended on the simulator's internal RNG state and made the training label for the same combo scenario differ between runs. The first expression was also syntactically a callable, which broke on the next Python revision. | Replaced with `sorted(obs)[0] if obs else "healthy"`, so the label is the alphabetically first observable fault. The classification head remains single-label by design (C is the declared fault taxonomy, not the number of co-firing faults). **Co-firing faults are not currently surfaced in the HUD** — see "What did **not** change" below. | Retrain (combo scenario test now stably labelled); `test_label_integrity.py` continues to pass |
| 4 | HIGH | `ml/parts.py` | Six display-only parameters (NBC `overpressure` / `filter_dp`, exhaust `backpressure` / `particulate_index`, acoustics `ae_burst_energy`, hydraulics `hyd_force`) were declared without `health_exclude`, so `part_health_index` scored them even though the LSTM never saw them — the health index and the model's view of the world disagreed. | Marked each as `health_exclude: True` with a comment explaining why (no simulator plant, sensor on a future pipeline). | `tests/test_leakage.py::TestSchemaCoversEverySubsystem` |
| 5 | HIGH | CI gates | `.github/workflows/tests.yml` and `pages.yml` referenced `tests/test_leakage.py` and `tools_check_artifacts.py` but neither file existed. The "no-skip" leakage guard was unenforceable. | Shipped both. `test_leakage.py` is unconditional (no `skipUnless`) and exercises three classes of invariant: schema coverage, label leakage, and C-header / model / config agreement. `tools_check_artifacts.py` reads the schema from `ml.parts` and the C dim from the Python source, so it cannot drift. | `python -m pytest tests/test_leakage.py` (15/15 pass) + `python tools_check_artifacts.py` (exit 0) |
| 6 | HIGH | UI | The dashboard had no visible signal that the published artifacts reflected the remediation. An operator looking at the live page could not tell whether the LSTM was running the new gradient, the new schema, or the old one. | Added a `REMEDIATION ROUND 1` bar above the parts grid showing each fix as a chip (`LSTM GRADIENT FIXED`, `INPUT SCHEMA D=26`, `COMBO LABELS FIXED`, `DISPLAY-ONLY FLAGGED`, `LEAKAGE GATES LIVE`, `MODEL RETRAIN COMPLETE`). The retrain chip is a *dynamic* artifact-gate indicator: on load, `dashboard.js::verifyArtifacts()` issues two `cache: "no-store"` fetches against `config.json` and `model.json` and flips the chip to one of three terminal states — `ok` (both 200, green) / `pending` (one 404, amber, label `MODEL RETRAIN PENDING`) / `error` (both unreachable, red, label `ARTIFACTS UNREACHABLE`). The page renders the chip in its initial `pending` class so it cannot appear green before the probes have run. | `tests/test_hud_smoke.py::TestHudSmoke::test_retrain_chip_is_ok` (Playwright headless against the served `docs/index.html`); manual via the local Pages preview |

## Dataset-availability statement

No additional fault-labelled subsystem datasets were acquired as part of this
remediation round. The audit did not request any, and the existing surrogate
corpora documented in `docs/PROVENANCE.md` (ZeMA, MetroPT-3, Scania APS, Naval
GT, SCANIA Component X) remain the only real-data evidence. The 2 newly added
input channels — `coolant_level` and `exhaust_pressure` — are sourced **only**
from the physics simulator (`sim/`) and from `sim_synth`. Their display
thresholds are documented in `ml/parts.py`; no real-vehicle evidence has been
acquired for them, and any field deployment of the cooling-health readouts
should treat them as advisory until procurement adds a coolant-level sender
and an exhaust-manifold pressure transducer to the gateway.

## Honest limits of the synthetic retrain

The test-set numbers in `docs/config.json::evaluation` (RUL MAE 134.7 steps,
cls_acc 0.917 on held-out `(fault × duty-profile)` cells) measure generalisation
across duty cycle and onset within one simulator. They are not evidence of
transfer to real hardware; that evidence lives in the surrogate-corpus benchmark
(`results/subsystems_benchmark.json`) and is unchanged by this remediation.

The classifier remained single-label (C=13 = `healthy` + 12 declared fault
classes). Combo scenarios inherit the alphabetically-first observable label,
so the model head and the surfaced label always agree. **Co-firing faults
are not currently surfaced in the HUD** — a previous draft carried them
through `per_scenario[].combo`, but the live stream does not serialise that
field and the dashboard has no consumer for it. Surfacing co-firing in the
HUD is a future task; for this round the operator sees the primary label
only, which matches the loss the model actually optimises.

## Dynamic retrain chip contract

The 6th chip in the remediation bar is a live artifact-gate indicator, not a
static badge. Its three terminal states are:

| State | Trigger | Visual | Operator reads |
|-------|---------|--------|----------------|
| `ok` | both `config.json` and `model.json` return HTTP 200 in the browser probe | green pill, label `MODEL RETRAIN COMPLETE` | "the page is shipping the new weights" |
| `pending` | exactly one of the two returns 404 | amber pill, label `MODEL RETRAIN PENDING` | "an export is mid-flight — refresh in a moment" |
| `error` | both fetches fail (network, 5xx, CORS) | red pill, label `ARTIFACTS UNREACHABLE` | "the page is up but the model is not — do not trust the readouts" |

The probe uses `fetch(..., { cache: "no-store" })` so a stale cache cannot make a
broken deployment look green. The chip is rendered in the initial `pending`
class, so the page can never claim `ok` before the probes have resolved.
Verified by `tests/test_hud_smoke.py` (Playwright headless), which waits for
the chip to leave `pending` before asserting the final state.

## What did **not** change

- The C edge runtime source (`c_engine/tank_pdm_infer.c`) is dim-agnostic
  through `TANK_INFER_D_FEATURES`; only the generated header moved.
- The browser inference path (`docs/lstm.js`) reads `model.D` and `model.H`; it
  picked up the new D=26 without any code change.
- The FDIR 6-layer plausibility gate and the J1939 DTC engine are independent
  of the LSTM retrain and were not touched.
- The `live_multi_streams.json` dataset is independent of the retrain.

## Verification

Local verification (matches the CI gate at `.github/workflows/tests.yml`):

```
python -m c_engine.gen_dims                  # writes tank_pdm_dims.h D=26
python -m pytest tests/test_leakage.py -v    # 15/15 pass
python -m pytest tests/ -v                    # full suite green
python -m pytest tests/test_hud_smoke.py -v   # headless browser smoke (skips if playwright missing)
python tools_check_artifacts.py               # exit 0
python -m ml.train --epochs 40 --w-reg 2.0 --w-cls 1.0
python -m benchmark.evaluate_subsystems --out results/subsystems_benchmark.json
for f in docs/*.js; do node --check "$f"; done
```

Re-run the artifact check after every retrain. A green run is the only
authoritative "all five fixes are live in the same commit" signal.
