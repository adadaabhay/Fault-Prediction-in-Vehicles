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
2. [Repository layout](#repository-layout)
3. [Quickstart](#quickstart)
4. [Requirements](#requirements)
5. [Generate sensor data](#generate-sensor-data)
6. [Run the gateway + HUD](#run-the-gateway--hud)
7. [Build the C edge runtime](#build-the-c-edge-runtime)
8. [Verification](#verification)
9. [Documentation](#documentation)
10. [Datasets](#datasets)
11. [License](#license)

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

## Repository layout

```
Fault-Prediction-in-Vehicles/
├── c_engine/             MISRA-C99 edge inference runtime + Python binding
├── ml/                   Python LSTM + scenario/label generation
├── sim/                  Physics-informed tank simulator
│   ├── cvrde/            CVRDE-specific subsystems (powerpack, HSU, GCS, NBC)
│   ├── physics/          Engine, exhaust, hydraulics, vibration, ... physics
│   ├── scripts/          Legacy one-off generators (preserved for back-compat)
│   └── generators/       Per-subsystem sensor data generators + master CLI
├── telemetry_gateway/    FastAPI ingest + WebSocket broadcast + J1939
├── tools/                Developer / CI entry points
├── docs/                 Web command HUD (HTML/JS/CSS + model + streams)
├── tests/                Unit, integration, parity, adversarial, HUD-smoke
├── data/                 (gitignored) generated CSVs and manifests
├── results/              (gitignored) runtime artefacts
├── pyproject.toml        Installable package + lint/type/test configuration
├── requirements.txt      Pinned runtime dependencies
├── requirements-dev.txt  Pinned dev / lint / test dependencies
├── Makefile              Common commands (`make help` for the list)
├── LICENSE               Apache-2.0
├── README.md             (this file)
├── DATA.md               Datasets manifest (sources, licences, SHA-256)
├── ARCHITECTURE.md       Subsystem ↔ module map, deployment topology
└── CHANGELOG.md          Release history (keep-a-changelog)
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

Runtime: **Python 3.10+**, NumPy 1.26+, FastAPI 0.110+, scikit-learn 1.4+.

The full list, pinned for reproducible numerics, lives in
[`requirements.txt`](requirements.txt).  Dev / lint / test dependencies are
in [`requirements-dev.txt`](requirements-dev.txt).  Both lists are also
embedded in [`pyproject.toml`](pyproject.toml) so `pip install -e .` is
sufficient on its own.

External datasets are **not** required to run the simulator — every
generator here emits synthetic data from `sim.tank.TankSimulator`.  See
[DATA.md](DATA.md) for the list of public corpora the benchmark
pipelines in the host monorepo consume.

## Generate sensor data

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
verifies the two are mutually consistent before deploy.

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

```bash
make test                      # full pytest run
make test-fast                 # skip slow / hud / hil markers
make test-cov                  # coverage report
make test-hud                  # browser-driven smoke tests (requires playwright)
make verify                    # lint + fast tests (CI-style local gate)
make typecheck                 # mypy (advisory)
```

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

## Datasets

This repo does **not** ship any dataset.  The simulator in `sim/`
generates synthetic data from a physics-based tank model; the
generated CSVs land in `data/simulated/` (gitignored).

The benchmark pipelines in the host monorepo (separate from this
repo) consume the public corpora listed in [DATA.md](DATA.md).  They
are not required to use, run, test, or extend this package.

## License

Apache-2.0.  See [LICENSE](LICENSE).  Patent grant included.
