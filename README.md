# phm-vehicle — Defense-Grade PHM/CBM+ for Military AFVs

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](#requirements)
[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](#verification)
[![MISRA-C](https://img.shields.io/badge/edge-MISRA--C99-blueviolet.svg)](c_engine/CODING_STANDARD.md)

Prognostics & Health Management (PHM/CBM+) ecosystem for military armored
fighting vehicles — CVRDE Arjun Mk-1A, T-90S Bhishma, Zorawar Light Tank.
Connects physical / simulated sensors through a FastAPI telemetry gateway,
a pre-inference FDIR plausibility gate, dual-tier neural inference
(in-browser ES6 LSTM and embedded MISRA-C99 edge runtime), and a real-time
Web command HUD with SAE J1939-73 DM1/DM2 diagnostic logging.

The project ships as an installable Python package with a
deployment-grade layout, a generated-data pipeline for every subsystem,
and full documentation of the data contract.

---

## Table of contents

1. [What this is](#what-this-is)
2. [Sensor families and governing physics](#sensor-families-and-governing-physics)
3. [Repository layout](#repository-layout)
4. [Quickstart](#quickstart)
5. [Requirements](#requirements)
6. [Generate sensor data](#generate-sensor-data)
7. [Programmatic use](#programmatic-use)
8. [Run the gateway + HUD](#run-the-gateway--hud)
9. [Build the C edge runtime](#build-the-c-edge-runtime)
10. [Regenerate model + stream](#regenerate-model--stream)
11. [Verification](#verification)
12. [Documentation](#documentation)
13. [Datasets](#datasets)
14. [License](#license)

---

## What this is

The platform has three layers:

```
                    ┌────────────────────────────────────┐
   Physical /       │   Telemetry Gateway                │
   Simulated        │   (FastAPI, UDP 9000, Serial COM)  │
   sensors   ────►  │   Pre-inference FDIR plausibility  │
                    │   SAE J1939-73 DM1/DM2 logger      │
                    └─────────────┬──────────────────────┘
                                  │ clean telemetry + DTCs
                                  ▼
                    ┌────────────────────────────────────┐
   Dual-tier        │   In-browser ES6 LSTM              │
   inference        │   Embedded MISRA-C99 LSTM          │
                    │   (sizes D=26, H=24, 12 classes)   │
                    └─────────────┬──────────────────────┘
                                  │ fused health + RUL
                                  ▼
                    ┌────────────────────────────────────┐
   Command HUD      │   docs/                            │
                    │   Live 270° gauges, J1939 chips,   │
                    │   2 s watchdog fallback,           │
                    │   per-module RUL bars.             │
                    └────────────────────────────────────┘
```

The project targets defense-grade deployment: provenance-pinned weights,
deterministic sensor noise, a label-leakage test suite, parity tests
across the Python / JavaScript / C implementations, and a JSON manifest
sidecar on every generated dataset.

## Sensor families and governing physics

Every sensor below is simulated from a closed-form physics equation
(documented in `Physics_Based_Sensor_Equations_Vehicle_Preventive_Maintenance.docx`)
and coupled through an evolving shared vehicle state, so that an
injected fault degrades the affected readings in a physically
consistent way.  The resulting labelled dataset feeds the AI
pipeline for anomaly detection, fault diagnosis and RUL prediction.

```
Sensors -> Physics-Based Features -> Vehicle State -> AI -> Anomaly / Fault / RUL
```

| #  | Sensor                | Physics                                                                 | Key features                            | Associated fault                |
|----|-----------------------|-------------------------------------------------------------------------|-----------------------------------------|---------------------------------|
| 1  | Vibration (accelerometer) | `a = d²x/dt²`, `x = A·sin(2πft+φ)`                                   | RMS, kurtosis, FFT peaks, BPFO/BPFI     | Bearing / gear wear              |
| 2  | Engine / exhaust temperature | `m·cₚ·dT/dt = Q̇_gen − Q̇_cool − Q̇_exh`, `R = R₀(1+αΔT)`      | Temperature trend                       | Overheating / cooling failure    |
| 3  | Oil pressure          | `ΔP = 8μLQ/(πr⁴)`, `μ(T) = A·e^{E/RT}`                                  | Pressure deviation vs. speed/load/temp  | Pump / leakage / blockage        |
| 4  | Oil debris            | Inductive `L ≈ μN²A/l`, `ṅₚ = dNₚ/dt`                                   | Particle count & rate                   | Accelerated wear                 |
| 5  | Torque                | `P = τω`, `τ = Tr/J`, `ω = 2πN/60`                                      | Torque / power / shear                  | Load / efficiency degradation    |
| 6  | Exhaust               | `PV = nRT`, `ṁ = ρAv`, `λ`                                              | EGT, pressure, λ, O₂                    | Combustion abnormality           |
| 7  | Fluid levels          | `C = εA/d`, `V = πr²h`                                                   | Level / volume / capacitance            | Leakage / depletion              |
| 8  | Hydraulics            | `P = F/A`, `P_hyd = PQ`                                                  | Pressure–flow relation                   | Pump / valve / seal fault        |
| 9  | Suspension strain     | `σ = Eε`, `ΔR/R = GF·ε`                                                  | Strain / load                           | Fatigue / overload               |
| 10 | Torsion bar           | `θ = TL/JG`, `T = JGθ/L`                                                 | Twist / torque history                  | Torsion-bar fatigue              |
| 11 | Shock                 | `F = ma`, `a_RMS`                                                        | RMS / peak g                            | Terrain / component loading      |
| 12 | Acoustic              | `p(t) = P₀ + A·sin(2πft)`, `SPL = 20·log₁₀(p/p_ref)`                     | SPL, FFT signatures                     | Gear / bearing noise             |
| 13 | Acoustic emission     | Wave equation, event features                                            | Event rate / energy                     | Crack / impact / friction        |

## Repository layout

```
Fault-Prediction-in-Vehicles/
├── c_engine/             MISRA-C99 edge inference runtime + Python binding
├── ml/                   Python LSTM + scenario/label generation
├── sim/                  Physics-informed tank simulator
│   ├── cvrde/            CVRDE-specific subsystems (powerpack, HSU, GCS, NBC)
│   ├── physics/          Engine, exhaust, hydraulics, vibration, ... physics
│   └── generators/       Per-subsystem sensor data generators + master CLI
├── telemetry_gateway/    FastAPI ingest + WebSocket broadcast + J1939
├── tools/                Developer / CI entry points
├── pipelines/            Public-corpora ingesters (MetroPT, ZeMA, Scania, …)
├── benchmark/            Multi-subsystem benchmark harness
├── docs/                 Web command HUD (HTML/JS/CSS + model + streams)
├── tests/                Unit, integration, parity, adversarial, HUD-smoke
├── data/                 (gitignored) generated CSVs and manifests
├── results/              (gitignored) runtime artefacts
├── .github/              CI workflows + Dependabot config
├── pyproject.toml        Installable package + lint/type/test configuration
├── requirements.txt      Pinned runtime dependencies
├── requirements-dev.txt  Pinned dev / lint / test dependencies
├── Dockerfile            Multi-stage, distroless final, non-root
├── Makefile              Common commands (`make help` for the list)
├── LICENSE               Apache-2.0
├── README.md             (this file)
├── DATA.md               Datasets manifest (sources, licences, SHA-256)
├── ARCHITECTURE.md       Subsystem ↔ module map, deployment topology
├── CHANGELOG.md          Release history (keep-a-changelog)
├── SECURITY.md           Threat model + vulnerability reporting
├── CODE_OF_CONDUCT.md    Contributor Covenant v2.1
├── CONTRIBUTING.md       Dev setup, PR invariants, review agents
└── VERIFICATION.md       End-to-end acceptance sign-off
```

## Quickstart

```bash
# 1. Clone
git clone https://github.com/adadaabhay/Fault-Prediction-in-Vehicles.git
cd Fault-Prediction-in-Vehicles

# 2. Install (editable + dev extras)
python -m venv .venv
. .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e ".[dev,bench]"

# 3. Verify
make verify                    # lint + fast tests

# 4. Generate one CSV per subsystem
make data                      # writes data/simulated/<subsystem>.csv + manifests

# 5. Open the dashboard
make serve-hud                 # http://localhost:8080
```

A first successful run produces:

```
data/simulated/
├── engine.csv              + engine.csv.manifest.json
├── powertrain.csv          + powertrain.csv.manifest.json
├── hydraulics.csv          + hydraulics.csv.manifest.json
├── suspension.csv          + suspension.csv.manifest.json
├── structure.csv           + structure.csv.manifest.json
├── gun_control.csv         + gun_control.csv.manifest.json
├── nbc.csv                 + nbc.csv.manifest.json
├── exhaust.csv             + exhaust.csv.manifest.json
├── acoustics.csv           + acoustics.csv.manifest.json
├── fuel_levels.csv         + fuel_levels.csv.manifest.json
└── MANIFEST.json           aggregate manifest (sha256 of every CSV)
```

## Requirements

Runtime: **Python 3.10+** (3.11 / 3.12 supported; see `pyproject.toml`
`classifiers`).  The full set of runtime + optional dependency ranges is
declared in [`pyproject.toml`](pyproject.toml) under `[project.dependencies]`
and `[project.optional-dependencies]`.  `pip install -e ".[dev,bench]"` is
the canonical install.

A pinned snapshot for environments that prefer pip-tools lives in
[`requirements.txt`](requirements.txt) (runtime) and
[`requirements-dev.txt`](requirements-dev.txt) (dev / lint / test).  These
are kept in lock-step with `pyproject.toml`; do not hand-edit one
without the other.

External datasets are **not** required to run the simulator — every
generator here emits synthetic data from `sim.tank.TankSimulator`.  See
[DATA.md](DATA.md) for the list of public corpora the benchmark
pipelines in this repo's `pipelines/` consume.

## Generate sensor data

Two entry points are available — the master `main.py` for a single
flat CSV, and `tools.generate_sensor_data` for the per-subsystem
layout that downstream training and the dashboard expect.

```bash
# Single flat CSV with raw + physics-derived features + labels
python main.py --steps 12000 \
    --faults bearing_wear,cooling_failure,oil_pump_degradation \
    --out data/sim_dataset.csv
```

The CSV contains raw sensor readings, physics-derived features
(`vib_rms`, `vib_kurtosis`, `spl_db`, `debris_rate`, …), a fused
`health_index` (0–100), an extrapolated `rul_steps` estimate, an
`anomaly_score`, and one-hot `fault_*` label columns.

For the per-subsystem layout used by training and the HUD:

```bash
# One CSV per subsystem, 500 steps, fault-free
python -m tools.generate_sensor_data --subsystem all --steps 500

# Just the engine view, with two faults injected
python -m tools.generate_sensor_data --subsystem engine --steps 2000 \
    --fault cooling_failure oil_pump_degradation

# List every registered subsystem + its channels and fault profiles
python -m tools.generate_sensor_data --list

# Custom output directory
python -m tools.generate_sensor_data --subsystem all --out-dir data/sim
```

Every CSV is paired with a JSON manifest sidecar that documents:

* the column schema (key, unit, healthy / warn / crit thresholds, scale)
* the fault profiles that may be injected
* the simulation time step and sample rate
* the SHA-256 of the CSV, computed on the written bytes

A top-level `MANIFEST.json` aggregates every CSV written in the run so
a downstream training step, a release pipeline or a data-quality check
can verify the dataset without re-running the simulator.

Subsystem aliases: `lubrication` and `cooling` resolve to the engine
generator (the lubrication and cooling sensors are co-located on the
powerpack); they run the same simulation and publish a different
channel view.

### Available fault profiles

The simulator ships with 12 fault profiles, registered in
`sim.faults.FaultManager.FAULT_MAP`:

```
bearing_wear                  gear_wear
cooling_failure               oil_pump_degradation
bearing_clearance_wear        seal_leakage
fuel_injector_fault           exhaust_restriction
torsion_fatigue               hydraulic_valve_fault
structural_crack              drivetrain_efficiency_loss
```

Each profile implements a deterministic ramp from `start_step` to
`start_step + ramp_steps`, parameterised so a healthy machine stays
green for the first portion of the run and a labelled fault window
appears for the second.

## Programmatic use

The simulator is a flat Python package — `sim.config`, `sim.faults`,
`sim.tank`, `sim.dataset` — so a downstream tool can import any of
its primitives without going through the CLI:

```python
import numpy as np
from sim.config import TankConfig
from sim.faults import FaultManager
from sim.tank import TankSimulator
from sim.dataset import write_dataset

fm = FaultManager(np.random.default_rng(0))
fm.add("bearing_wear", start_step=3000, ramp_steps=1500)
fm.add("cooling_failure", start_step=6000, ramp_steps=1500)

sim = TankSimulator(TankConfig(), faults=fm, seed=0)
records = sim.run()                  # list of dict records
write_dataset(sim, "data/run.csv")   # CSV with health features + labels
```

The fused `health_index` is computed in `sim.features` from the
vibration RMS/kurtosis, oil temperature/pressure, debris rate, AE
activity, structural stress and lambda channels, per §15 of the
physics document.  `rul_steps` is obtained by linear extrapolation
of the health-index trajectory to the failure threshold, per §16.

## Run the gateway + HUD

```bash
# Boot the FastAPI ingest + WebSocket broadcast
make serve                     # http://localhost:8000
#   GET  /api/telemetry/status
#   POST /api/telemetry/push
#   WS   /ws/telemetry

# In another terminal, serve the dashboard
make serve-hud                 # http://localhost:8080
```

The dashboard reads its model from `docs/model.json` and its
replay stream from `docs/live_stream.json`.  The CI gate
`phm-check-artifacts` (alias for `python -m tools.check_artifacts`)
verifies the two are mutually consistent before any consumer
(local serve, downstream release, or partner integration) loads
them.

## Regenerate model + stream

After changing the simulator or training configuration, regenerate
the model weights and the live replay stream:

```bash
python -m ml.train --epochs 28 --stride 8 --demo-steps 3000
```

The training script writes the regenerated `docs/model.json` and
the per-subsystem replay CSVs consumed by the dashboard.  The
`phm-check-artifacts` gate then verifies the artifacts are
self-consistent (D, R, C dimensions match the schema, every input
feature has a scaler entry, every dashboard button has a stream)
before the dashboard consumes the new artifacts.

## Build the C edge runtime

```bash
make c-build                   # CMake build, Release config
make c-test                    # Python ↔ C parity suite
```

The edge runtime is a MISRA-C99 LSTM core designed to drop onto a
microcontroller without an OS.  It reads weights from a binary blob
emitted by `ml.train` and consumes a 26-element feature vector per
frame.  See [c_engine/CODING_STANDARD.md](c_engine/CODING_STANDARD.md) for
the coding rules the implementation follows.

## Verification

The full suite runs under either pytest or the stdlib
`unittest` discover entry — the latter is provided for
environments where the dev extra is not installed:

```bash
# pytest (preferred; honours markers and the strict CI gate)
make test                      # full pytest run
make test-fast                 # skip slow / hud / hil markers
make test-cov                  # coverage report
make test-hud                  # browser-driven smoke tests (requires playwright)
make verify                    # lint + fast tests (CI-style local gate)
make typecheck                 # mypy (advisory)

# stdlib fallback (no dev extras required)
python -m unittest discover -s tests -v
```

The suite is organised by markers so a partial gate (e.g. a CI job that
lacks the hardware rig) can skip what it cannot run:

The suite is organised by markers so a partial gate (e.g. a CI job that
lacks the hardware rig) can skip what it cannot run:

| Marker        | Purpose                                       |
|---------------|-----------------------------------------------|
| `adversarial` | FDIR, DTC encoders, J1939 conformance         |
| `hud`         | Browser-driven dashboard smoke tests          |
| `hil`         | Hardware-in-the-loop (skipped without rig)    |
| `parity`      | Cross-implementation parity (JS↔Py, C↔Py)     |
| `slow`        | Tests that take >5s                           |

## Documentation

* [ARCHITECTURE.md](ARCHITECTURE.md) — high-level architecture, subsystem
  ↔ module map, deployment topology.
* [DATA.md](DATA.md) — every external dataset the project touches:
  source, license, version pinned, what it is used for, SHA-256 of
  procured copies.
* [CHANGELOG.md](CHANGELOG.md) — release history in keep-a-changelog
  format.
* [PROJECT.md](PROJECT.md) — milestone tracking, feature inventory,
  interface contracts.
* [docs/PROVENANCE.md](docs/PROVENANCE.md) — model + dataset provenance.
* [docs/REMEDIATION.md](docs/REMEDIATION.md) — audit-remediation history.
* [docs/openapi.json](docs/openapi.json) — gateway HTTP API spec.
* [docs/assets/hud_nbc_modal.png](docs/assets/hud_nbc_modal.png) — the
  NBC subsystem detail modal the HUD renders when an operator opens a
  module from the live feed.

## Datasets

This repo does **not** ship any dataset.  The simulator in `sim/`
generates synthetic data from a physics-based tank model; the
generated CSVs land in `data/simulated/` (gitignored).

The benchmark pipelines in this repo's `pipelines/` consume the public
corpora listed in [DATA.md](DATA.md).  They are not required to use,
run, test, or extend this package.

## License

Apache-2.0.  See [LICENSE](LICENSE).  Patent grant included.
