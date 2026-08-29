# ARCHITECTURE.md

This document is the engineer-on-call's mental model of the **phm-vehicle**
deployment.  It covers the data flow, the subsystem ↔ module map, the
dual-tier inference contract, the package layout, and the deployment
topology.

For the *why* behind specific design decisions, see:

* [`PROJECT.md`](PROJECT.md) — milestones, feature inventory, interface contracts
* [`docs/PROVENANCE.md`](docs/PROVENANCE.md) — model and dataset provenance
* [`docs/REMEDIATION.md`](docs/REMEDIATION.md) — the M5 audit-remediation history
* [`c_engine/CODING_STANDARD.md`](c_engine/CODING_STANDARD.md) — MISRA-C99 coding rules

---

## 1. Data flow

```
   physical / simulated          ┌────────────────────────┐
   sensors                       │  Telemetry gateway     │
   (RS-485 / CAN / UDP 9000 ───► │  FastAPI on :8000      │
    / serial COM 115200)         │  + TelemetryBroker     │
                                 │  + pre-inference FDIR  │
                                 │  + SAE J1939-73 DTC    │
                                 │  + WebSocket /ws/...   │
                                 └───────────┬────────────┘
                                             │ clean telemetry + DTCs
                                             ▼
                                 ┌────────────────────────┐
                                 │  Dual-tier inference   │
                                 │  ┌───────────────────┐ │
                                 │  │ docs/lstm.js      │ │  in-browser ES6
                                 │  │ (worker thread)   │ │  zero-install
                                 │  └───────────────────┘ │
                                 │  ┌───────────────────┐ │
                                 │  │ c_engine          │ │  embedded MISRA-C99
                                 │  │ (MISRA-C99, no OS)│ │  bare-metal target
                                 │  └───────────────────┘ │
                                 │  ┌───────────────────┐ │
                                 │  │ ml/lstm.py        │ │  Python (training,
                                 │  │ (training/eval)   │ │  parity tests, batch)
                                 │  └───────────────────┘ │
                                 └───────────┬────────────┘
                                             │ fused health + RUL
                                             ▼
                                 ┌────────────────────────┐
                                 │  Command HUD           │
                                 │  docs/ (HTML/JS/CSS)   │
                                 │  270° gauges, J1939    │
                                 │  chips, 2s watchdog    │
                                 │  fallback, per-module  │
                                 │  RUL bars.             │
                                 └────────────────────────┘
```

The gateway never blocks on inference.  The pipeline is per-frame:

```
   raw frame ─► FDIR plausibility gate ─► health assessment ─► LSTM inference ─► DTC encoder
```

Failures at any stage are reported back as J1939 DM1 entries; the
inference block is wrapped so a JS-side failure (e.g. WebAssembly
disabled) does not stop the live data feed.

## 2. Subsystem ↔ module map

The ten physical subsystems in `ml.parts.PARTS` map to one of three
locations in the codebase:

| Part (ml.parts)         | Physics (sim.physics)        | CVRDE-specific (sim.cvrde)    | Generator (sim.generators)    | LSTM inputs |
|-------------------------|------------------------------|-------------------------------|-------------------------------|-------------|
| `engine`                | `temperature.py`, `oil.py`, `exhaust.py` | `powerpack.py`                | `engine.py`                   | coolant_temp, oil_temp, exhaust_temp, lambda_residual |
| `powertrain`            | `torque.py`, `vibration.py`  | (covered by `sim.tank`)       | `powertrain.py`               | shaft_torque, driveline_efficiency, vib_rms, vib_kurtosis, vib_dom_amp |
| `lubrication`           | `oil.py`                     | `powerpack.py`                | (alias of `engine.py`)        | oil_pressure, oil_temp, debris_rate |
| `cooling`               | `temperature.py`, `level.py` | `powerpack.py`                | (alias of `engine.py`)        | coolant_temp, coolant_level, exhaust_pressure |
| `hydraulics`            | `hydraulics.py`              | `gun_control.py`              | `hydraulics.py`               | hyd_pressure, hyd_leak_flow, hyd_force* |
| `suspension`            | `suspension.py`              | `hydrogas_suspension.py`      | `suspension.py`               | susp_load_kN, susp_compliance, susp_strain_ue, shock_a_rms_g |
| `structure`             | `torsion.py`                 | (gun mount in `gun_control.py`) | `structure.py`              | torsion_twist_deg, torsion_cumulative_twist, ae_event_rate, ae_energy |
| `nbc`                   | (no plant model)             | `auxiliary_nbc.py`            | `nbc.py`                      | (display only, health_exclude) |
| `exhaust`               | `exhaust.py`                 | `powerpack.py`                | `exhaust.py`                  | (display only; LSTM uses engine exhaust_temp) |
| `acoustics`             | `acoustics.py`               | (gun mount in `gun_control.py`) | `acoustics.py`              | ae_event_rate, spl_db |
| `overall`               | (composite)                  | (composite)                   | `fuel_levels.py`              | (composite health) |

\* `hyd_force` is flagged `health_exclude=True` (readback of joystick
command, not a measurement).

**Note on the CVRDE-specific modules:** `sim.cvrde.*` provides the
high-rate (200 Hz) physics for the CVRDE Arjun Mk-1A platform and is
used by the dashboard's Thar Desert demo mission.  The lower-rate
`sim.tank.TankSimulator` (20 Hz by default) is what the per-subsystem
generators in `sim/generators/` use -- the two share the underlying
state variables but the CVRDE version is the authoritative
"looks-like-the-fielded-tank" reference.

## 3. Dual-tier inference contract

Both inference engines must consume the **same 26-element feature
vector** in the **same order** (`ml.parts.INPUT_FEATURES`).  The
parity tests under `tests/test_js_python_parity.py` and
`tests/test_c_python_parity.py` enforce this on every CI run; the
gate fails if the C output diverges from the Python output by more
than `1e-9` on a fixed-weight sample.

```
   feature vector (D=26)
   │
   ├──► ml/lstm.py  (training, evaluation, batch inference)
   ├──► docs/lstm.js  (in-browser ES6; same D=26, same weights, same math)
   └──► c_engine/tank_pdm_infer.c  (embedded; same D=26, same weights, same math)
```

The exported weights live in three places:

* `ml/scenarios.py` (the Python training script re-emits them on retrain)
* `docs/model.json` (the browser reads this)
* `c_engine/tank_pdm_weights.bin` (the C runtime reads this)

`tools/check_artifacts.py` verifies the three agree on `D`, `H`, `R`,
`C` (hidden, recurrence, output class count) and that the per-record
feature set in `docs/live_stream.json` is a subset of the schema.

## 4. Package layout

The package uses a **flat layout** (top-level `telemetry_gateway/`,
`ml/`, `sim/`, `c_engine/`) rather than the more conventional
`src/phm_vehicle/` layout.  Why:

* `docs/` is the dashboard root.  The in-browser JS reads
  `docs/model.json`, `docs/config.json`, and `docs/live_*.json` by
  relative path; moving the Python packages into `src/` would not
  change that but would force a re-pathing of every import.
* `c_engine/` needs to be buildable in place by CMake; moving it
  under `src/` would require a `find_package` rewrite and a
  regenerated `tank_pdm_weights.bin` location.
* The single CLI surface (`phm-generate`, `phm-check-artifacts`,
  `phm-check-links`, `phm-server`) is declared in
  `[project.scripts]` of `pyproject.toml`; the entry-point paths
  are the module paths themselves, with no `phm_vehicle.*` shim.

The trade-off is that the `phm_vehicle` namespace does not exist; the
package installs as `phm-vehicle` (distribution name) but the import
paths remain `from telemetry_gateway import ...`.  This is documented
in `pyproject.toml` so a future maintainer can migrate if they need
to.

## 5. Deployment topology

```
   fielded vehicle                command post / depot
   ───────────────                ────────────────────
   ┌──────────────┐               ┌──────────────────┐
   │ c_engine     │   (optional)  │  docs/           │
   │ (bare-metal) │─── radio ───► │  HUD on tablet    │
   │              │               │  (in-browser LSTM)│
   │ sensors ──┐  │               │                  │
   └───────────┼──┘               └──────────────────┘
               │ UDP 9000 / serial
               ▼
   ┌──────────────────┐
   │ telemetry_gateway│  ◄── replay stream from docs/live_stream.json
   │ (FastAPI :8000)  │      when hardware is offline
   └──────────────────┘
```

The bare-metal C engine is optional -- the command-post deployment
runs the full ML pipeline in the browser.  The C engine exists for
the on-board ECU use case (display the fused health on the driver's
panel without needing a network round-trip to command).

## 6. Threading and back-pressure

* `TelemetryBroker` is a thread-safe singleton with a single mutex
  on its latest-frame slot; older frames are dropped, not queued.
* The FDIR gate runs on the ingest thread, the LSTM on a dedicated
  worker, the DTC encoder on the broadcast thread.  They communicate
  via thread-safe queues with a bounded depth (default 256 frames)
  so a slow LSTM can never block ingest.
* The WebSocket broadcast uses `asyncio` fan-out; slow clients are
  disconnected after a 2 s watchdog, after which the in-browser
  `LiveSocket` falls back to the recorded mission.

## 7. Logging and observability

* The DTC flash ring buffer (`results/dtc_flash_log.jsonl`) is
  rewritten on every gateway run; it is the on-disk ring buffer the
  spec promised (PGN 65226 DM1 + PGN 65227 DM2).
* The decision audit log (`results/decision_audit_log.jsonl`) records
  every fault-diagnosis call with its feature vector, prediction, and
  confidence.
* The model benchmark (`results/subsystems_benchmark.json`) records
  per-subsystem metrics on the public corpora; it is regenerated by
  `python -m benchmark.evaluate_subsystems` and pinned in CI.
* No PII, no telemetry of operator behaviour.  Logs are stripped of
  source identifiers in production (configurable).
