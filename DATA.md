# DATA.md — Dataset Manifest

This document enumerates every external dataset the **phm-vehicle** project
and its sibling benchmark pipelines touch.  It is the authoritative
answer to "what data exists, where did it come from, and what is it
used for."

The simulator inside `sim/` does **not** require any of these datasets to
run -- every per-subsystem sensor generator in `sim/generators/` produces
its CSV from `sim.tank.TankSimulator` alone.  These external corpora are
consumed by the benchmark pipelines in this repo's `pipelines/`
directory, which validate the same models on real-world data so a
deployment-grade claim can be made.

---

## Layout

| Category                    | Where it lives in the repo             | Used by                                |
|-----------------------------|----------------------------------------|----------------------------------------|
| Synthetic (in-repo)         | `sim.generators.*` → `data/simulated/` | training, dashboard demo               |
| Public corpora (procured)   | `datasets/procured/<name>/`            | `pipelines/<name>.py`                  |
| Public corpora (raw, vendored) | `datasets/<name>/` (gitignored)        | reference, future re-runs              |
| Generated telemetry streams | `docs/live_stream.json`, `live_multi_streams.json` | dashboard replay             |

Synthetic data is committed **only** as the recorded JSON replay streams
shipped with the dashboard (so the page can demo without the
simulator).  Generated CSVs are gitignored.  Public corpora are
gitignored -- only their **SHA-256 fingerprints** are committed, in
`results/subsystems_benchmark.json` and in this file.

---

## Synthetic (in-repo)

| Name                              | Generator                          | Used as                                            | Determinism  |
|-----------------------------------|------------------------------------|----------------------------------------------------|--------------|
| Per-subsystem sensor CSVs         | `sim/generators/<name>.py`         | training, regression, FDIR fuzzing                 | seeded       |
| `live_stream.json` (6000 steps)   | `ml/train.py` → `save_demo`        | dashboard replay / single-stream                   | seeded (42)  |
| `live_multi_streams.json`         | `sim/cvrde/cvrde_generator.py` + 6 others | dashboard multi-stream view                | seeded (42)  |
| Thar Desert mission (60 s)        | `sim.cvrde.cvrde_generator`        | combat-scenario demo, FDIR gate validation         | seeded (42)  |

The per-subsystem generators write both a CSV and a `.csv.manifest.json`
sidecar documenting schema, units, fault profiles, sample rate, and the
SHA-256 of the CSV bytes.  A top-level `MANIFEST.json` aggregates every
CSV in a run.

## Public corpora (procured, gitignored)

The `pipelines/` directory in this repo contains adapter scripts that
load each public corpus and apply the same preprocessing the synthetic
pipeline uses, so the model trained on synthetic data can be benchmarked
on real-world data without re-engineering feature extraction.

| Corpus                                  | Pipeline                       | What it is                                              | License    |
|-----------------------------------------|--------------------------------|---------------------------------------------------------|------------|
| NASA C-MAPSS Turbofan                   | `pipelines/_paths.py` baseline | Turbofan engine run-to-failure, 4 sub-datasets, 26 channels | NASA Open  |
| MetroPT-3                               | `pipelines/apu_metropt.py`     | APU + HVAC telemetry, real-world industrial failure     | Research   |
| Zema Hydraulics                         | `pipelines/hydraulics_zema.py` | Condition monitoring of hydraulic systems (UCI 447)     | CC-BY-4.0  |
| Scania ComponentX (APS)                 | `pipelines/heavy_scania.py`    | Scania heavy-truck APS failure, air-pressure system      | Research   |
| Scania AEGIS CAN bus                    | `pipelines/can_aegis.py`       | Scania AEGIS onboard CAN bus, real fleet                | Research   |
| Scania 2024-34-2                        | `pipelines/fleet_scania_componentx.py` | Scania fleet failure data 2024-34-2            | Research   |
| Deutz Engine Fault                      | `pipelines/engine_deutz.py`    | Deutz engine fault dataset, multi-channel vibration      | Research   |
| Naval Propulsion (gasturbine)           | `pipelines/naval_gasturbine.py`| Naval gasturbine propulsion, UCI 00312                  | CC-BY-4.0  |
| AI4I 2020 Predictive Maintenance        | (not yet wired)                 | UCI 601 -- synthetic but realistic industrial PM data   | CC-BY-4.0  |
| Synthetic Tank Telemetry (pre-existing) | `sim/tank.py` bootstrap         | Earlier project-state synthetic runs (legacy)           | n/a        |

The synthetic-tank dataset (`datasets/synthetic_tank_telemetry/`) is
preserved as the **first** generation of the simulator -- it lacks the
display-channel discipline introduced in the M5 audit-remediation
pass and should be treated as a historical reference, not a current
source of training data.

### Dataset fingerprints

The integrity of each procured copy is pinned by SHA-256, computed at
the time of procurement and stored in
`results/subsystems_benchmark.json`.  Re-derive before use:

```bash
# Verify the MetroPT-3 procurement copy
python - <<'PY'
import hashlib, pathlib
p = pathlib.Path("datasets/procured/metropt3_apu")
h = hashlib.sha256()
for f in sorted(p.rglob("*")):
    if f.is_file():
        h.update(f.read_bytes())
print(h.hexdigest())
PY
# Compare against results/subsystems_benchmark.json["dataset_sha256"]["metropt3_apu"]
```

If a hash mismatches, the upstream source has republished the dataset;
update the pin in `results/subsystems_benchmark.json` after a
side-by-side diff and re-run `python benchmark/evaluate_subsystems.py`
to confirm the benchmark numbers are unchanged.

## Generated replay streams (committed, large)

Two pre-recorded streams are committed to `docs/` because the
dashboard's REPLAY mode needs them to be loadable without a running
simulator:

| File                            | Size   | Used by                                       |
|---------------------------------|--------|-----------------------------------------------|
| `docs/live_stream.json`         | ~1.7 MB | dashboard main view (LIVE HARDWARE STREAM badge shows REPLAY) |
| `docs/live_multi_streams.json`  | ~4.2 MB | dashboard multi-stream view (six scenarios)  |

Both are deterministic (seed 42) and generated by
`python -m ml.train` / `python -m sim.cvrde.cvrde_generator`.  They are
**not** datasets -- they are runtime artefacts that the dashboard
loads over HTTP.  Regenerate with the commands above after a model
retrain.

---

## What is *not* in this repo

For completeness, here is what is deliberately excluded:

* **Raw public corpora** -- several hundred MB total.  Re-download from
  the source URLs in the pipeline docstrings.
* **Trained model weights** (`.joblib`, `.pkl`, `.bin`) -- the
  in-browser ES6 LSTM and the C edge runtime both read their weights
  from `docs/model.json` and `c_engine/tank_pdm_weights.bin`
  respectively; both are committed because they are the deployment
  artefact, not the training artefact.
* **Training-set pickles** (`results/models/*.joblib`) -- not
  diff-able, not load-able across scikit-learn versions, and
  re-derivable from `python benchmark/evaluate_subsystems.py`.
* **Virtual environments** (`.venv/`, `venv/`) -- per Python tooling
  norms.
* **DTC flash-ring log** (`results/dtc_flash_log.jsonl`) -- runtime
  artefact, rewritten on every gateway run.

If you are reviewing this project for compliance (export-control, ITAR,
internal classification), the **synthetic data is the only thing that
ships**; the public corpora are procured copies of publicly available
research data and are not redistributed by this project.
