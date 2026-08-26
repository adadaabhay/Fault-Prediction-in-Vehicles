# Project: Defense-Grade PHM/CBM+ Ecosystem for Military AFVs

## Architecture
The Prognostics & Health Management (PHM/CBM+) ecosystem connects physical/simulated sensors from military armored fighting vehicles (CVRDE Arjun Mk-1A, T-90S Bhishma, Zorawar Light Tank) to dual-tier neural inference engines (in-browser ES6 LSTM and embedded MISRA-C99 edge runtime) with pre-inference FDIR plausibility gating and SAE J1939-73 military diagnostic logging.

```
[Physical Sensors / Microcontroller / UDP 9000 / Serial COM / CAN]
                           │
                           ▼
     ┌──────────────────────────────────────────────┐
     │  telemetry_gateway/live_sensor_ingest.py     │
     │  - UDP Port 9000 Async Datagram Listener     │
     │  - Serial COM (115200 baud) Ingest Engine    │
     │  - REST /api/telemetry/push Endpoint         │
     │  - Centralized TelemetryBroker               │
     └──────────────────────┬───────────────────────┘
                            │ Raw Telemetry Frame
                            ▼
     ┌──────────────────────────────────────────────┐
     │  telemetry_gateway/sensor_plausibility.py    │
     │  (Pre-Inference FDIR Plausibility Gate)      │
     │  - Open / Short Circuit Clamp & Flag         │
     │  - Slew Rate & Outlier / EMI Hampel Filter   │
     │  - Stuck-At & Dual-Sensor Consistency Gate   │
     └──────────────┬───────────────────────────────┘
                    │
       ┌────────────┴───────────────────────────┐
       │ Clean Clamped Signals                  │ Electrical Fault Events
       ▼                                        ▼
┌─────────────────────────────┐   ┌────────────────────────────────────────┐
│ Neural Inference Engines    │   │  telemetry_gateway/dtc_engine.py       │
│ - JS LSTM (docs/lstm.js)    │   │  (SAE J1939-73 DTC Logger)             │
│ - C99 Edge (tank_pdm_infer) │   │  - SPN / FMI DTC Translation Engine    │
│ - Py Model (ml/lstm.py)     │   │  - PGN 65226 DM1 (Active Malfunctions) │
└──────────────┬──────────────┘   │  - PGN 65227 DM2 (Historic Malfunction)│
               │ Fault Probs      │  - Flash Ring Buffer Persistence       │
               └─────────────────►└────────────────────┬───────────────────┘
                                                       │
   Composed and ordered by telemetry_gateway/          │
   pipeline.py (PHMPipeline), invoked per frame.       │
   Guarded by tests/test_pipeline.py.                  │
                                                       ▼
     ┌─────────────────────────────────────────────────────────────────────┐
     │  telemetry_gateway/server.py (FastAPI / WebSocket /ws/telemetry)    │
     │  - Broadcasts Clean Telemetry, J1939 Frames, DTCs, & Health Vector  │
     └─────────────────────────────────┬───────────────────────────────────┘
                                       │ Real-time Stream (< 20ms)
                                       ▼
     ┌─────────────────────────────────────────────────────────────────────┐
     │  docs/ (Frontend Command HUD)                                       │
     │  - [LIVE HARDWARE STREAM] / [SIMULATION] Status Badge               │
     │  - Real-time 270° Canvas Gauges & Multi-Subsystem Health Cards      │
     │  - Active SAE J1939 DM1/DM2 Diagnostic Trouble Code Chips           │
     │  - Automatic 2.0s Watchdog Fallback & Zero Console Error Guarantee │
     └─────────────────────────────────────────────────────────────────────┘
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | UDP 9000 Telemetry Listener | Asynchronous UDP socket listener on port 9000 receiving JSON/binary datagrams | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Serial COM Ingest Bridge | 115200 baud serial listener with auto-reconnect and mock serial fallback | M1 | ORIGINAL_REQUEST §R1 |
| 3 | REST Telemetry Push API | `/api/telemetry/push` and `/api/telemetry/status` HTTP endpoints on FastAPI gateway | M1 | ORIGINAL_REQUEST §R1 |
| 4 | Central Telemetry Broker | Thread-safe singleton broker coordinating ingest channels and WebSocket distribution | M1 | ORIGINAL_REQUEST §R1 |
| 5 | Physical Range & NaN Sanitizer | Sanitizes NaN/inf and clamps signals to physical operating envelopes | M2 | ORIGINAL_REQUEST §R2 |
| 6 | Open / Short Circuit Detection | Flags open-circuit (e.g. P_oil=-999) and short-circuit conditions as electrical faults | M2 | ORIGINAL_REQUEST §R2 |
| 7 | Slew Rate & Outlier / EMI Filter | Limits unphysical step changes and rejects transient EMI spikes | M2 | ORIGINAL_REQUEST §R2 |
| 8 | Stuck-At & Dual-Sensor Gate | Identifies deadlined/frozen sensors and cross-checks redundant sensor pairs | M2 | ORIGINAL_REQUEST §R2 |
| 9 | SAE J1939 SPN/FMI Mapper | Maps electrical sensor faults and neural subsystem anomalies to standard SPN/FMI codes | M3 | ORIGINAL_REQUEST §R3 |
| 10 | J1939 DM1 Active Malfunctions | Encodes 8-byte PGN 65226 DM1 packets with lamp status (Amber/Red/Protect) | M3 | ORIGINAL_REQUEST §R3 |
| 11 | J1939 DM2 Historic Log Engine | Tracks previously active DTCs and broadcasts PGN 65227 DM2 frames | M3 | ORIGINAL_REQUEST §R3 |
| 12 | On-Disk Flash Ring Buffer | Implements a thread-safe circular flash ring buffer (`dtc_flash_log.jsonl`) | M3 | ORIGINAL_REQUEST §R3 |
| 13 | HUD Live Hardware Stream Badge | `LiveSocket` in `live.js` + `#sourceBadge`: LIVE HARDWARE STREAM / CONNECTING / STALLED / REPLAY, with a frame counter and FDIR latency | M4 | ORIGINAL_REQUEST §R1 & AC6 |
| 14 | Automatic Watchdog Fallback | 2.0 s watchdog in `LiveSocket` plus exponential-backoff reconnect; falls back to the recorded mission and logs the transition | M4 | ORIGINAL_REQUEST §R1 |
| 15 | HUD J1939 Diagnostic Display | `renderDTCs()` renders active DM1 SPN/FMI chips coloured by lamp status | M4 | ORIGINAL_REQUEST §R3 |
| 16 | Defensive Input Clamping | Present in `dashboard.js`; DTC text uses `textContent`, never `innerHTML`. **Not covered by a browser test** — "zero console errors" is asserted, not measured. | M4 | ORIGINAL_REQUEST §AC2, AC6 |
| 17 | Verification Suite | 34 modules under `tests/` passing across unit, integration, and parity gates | M4 | ORIGINAL_REQUEST §AC4, AC5 |
| 18 | PHM Pipeline Composition | `telemetry_gateway/pipeline.py` — FDIR gate → health assessment → LSTM prognostics → DTC generation, per frame | M5 | audit remediation |
| 19 | Ingest Access Control | API key, per-client token bucket, payload cap, WebSocket origin check | M5 | audit remediation |
| 20 | Unit Contract | `telemetry_gateway/units.py` — SI ↔ engineering, round-trip and envelope tested | M5 | audit remediation |
| 21 | Label-Leakage Guards | `tests/test_leakage.py` — no channel may be a readback of an injected fault parameter | M5 | audit remediation |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Live Sensor Hardware Ingestion Bridge | `telemetry_gateway/live_sensor_ingest.py`, server endpoints `/api/telemetry/push`, `TelemetryBroker`, UDP 9000 & Serial 115200 ingestion | None | DONE |
| M2 | Adversarial Sensor Fuzzing & Plausibility Gate (FDIR) | `telemetry_gateway/sensor_plausibility.py`, 6-layer filter, open/short clamping, rate limiters, dual-sensor checks | None | DONE |
| M3 | SAE J1939-73 DM1 & DM2 Diagnostic Trouble Code Engine | `telemetry_gateway/dtc_engine.py`, SPN/FMI mapping, DM1/DM2 encoders, flash ring buffer storage | M2 | DONE |
| M4 | Frontend HUD Live Integration & Verification Suite | `Fault-Prediction-in-Vehicles/docs/` (`live.js`, `dashboard.js`, `index.html`, `style.css`) and the suites under `tests/` | M1, M2, M3 | DONE |
| M5 | Audit Remediation | Label-leakage removal, pipeline composition, ingest access control, unit contract, J1939-71 conformance, EGT / thermostat / diesel-λ physics, grouped three-way ML split, edge-runtime fail-safe | M1–M4 | DONE |

## Interface Contracts

### 1. `telemetry_gateway/live_sensor_ingest.py`
- Class `TelemetryBroker`:
  - `push_telemetry(data: dict, source: str = "http") -> dict`
  - `get_latest_telemetry() -> dict`
  - `get_status() -> dict`
  - `register_listener(callback: Callable[[dict], None])`
- Class `UDPSensorListener`:
  - `start(host: str = "0.0.0.0", port: int = 9000)`
  - `stop()`
- Class `SerialSensorListener`:
  - `start(port: str = "COM3", baudrate: int = 115200, mock_fallback: bool = True)`
  - `stop()`

### 2. `telemetry_gateway/sensor_plausibility.py`
- Class `SensorPlausibilityGate`:
  - `filter_frame(raw_telemetry: dict) -> PlausibilityResult`
- Class `PlausibilityResult`:
  - `clean_telemetry: dict` (all values sanitized, finite, clamped within physical limits)
  - `faults_detected: list[SensorFaultEvent]`
  - `is_valid: bool`
- Class `SensorFaultEvent`:
  - `channel: str`
  - `fault_type: str` ("OPEN_CIRCUIT", "SHORT_CIRCUIT", "RATE_OF_CHANGE_EXCEEDED", "STUCK_AT", "OUTLIER_EMI", "DUAL_SENSOR_MISMATCH")
  - `raw_value: float`
  - `clamped_value: float`
  - `spn: int`
  - `fmi: int`

### 3. `telemetry_gateway/dtc_engine.py`
- Class `DTCEngine`:
  - `process_fdir_faults(faults: list[SensorFaultEvent]) -> list[DTCRecord]`
  - `process_neural_predictions(subsystem_health: dict, fault_probs: dict) -> list[DTCRecord]`
  - `get_active_dtcs() -> list[DTCRecord]`
  - `get_historic_dtcs() -> list[DTCRecord]`
  - `encode_dm1_packet() -> bytes` (PGN 65226, 8+ bytes)
  - `encode_dm2_packet() -> bytes` (PGN 65227, 8+ bytes)
  - `persist_to_flash(record: DTCRecord)`
- Class `DTCRecord`:
  - `spn: int` (19-bit Suspect Parameter Number)
  - `fmi: int` (5-bit Failure Mode Identifier)
  - `oc: int` (7-bit Occurrence Count)
  - `cm: int` (1-bit Conversion Method)
  - `description: str`
  - `lamp_status: str` ("MIL", "RED_STOP", "AMBER_WARNING", "PROTECT")
  - `timestamp: str`

## Code Layout
```
Vnest/
├── .agents/                                    # Orchestrator & subagent workspaces (metadata only)
├── telemetry_gateway/
│   ├── __init__.py
│   ├── server.py                              # FastAPI backend with /ws/telemetry & /api/telemetry/push
│   ├── live_sensor_ingest.py                  # R1: UDP 9000, Serial COM, TelemetryBroker
│   ├── sensor_plausibility.py                 # R2: Pre-inference FDIR Plausibility Gate
│   ├── dtc_engine.py                          # R3: SAE J1939-73 DM1/DM2 DTC Engine & Flash Log
│   ├── pipeline.py                            # PHMPipeline: FDIR -> health -> LSTM -> DTC
│   ├── units.py                               # SI <-> engineering unit contract
│   ├── j1939_can_parser.py                    # J1939-71 EEC1 / ET1 / EFL_P1 encode+decode
│   ├── tactical_burst.py                      # 32-byte EMCON tactical burst encoder
│   └── export_multi_streams.py                # Telemetry streams generator
├── Fault-Prediction-in-Vehicles/
│   ├── docs/                                  # Frontend Web Command HUD
│   │   ├── index.html                         # Command HUD markup & indicators
│   │   ├── dashboard.js                       # Real-time HUD controller & gauge updater
│   │   ├── live.js                            # WebSocket live client with watchdog fallback
│   │   ├── lstm.js                            # In-browser JS neural inference engine
│   │   ├── gauges.js                          # 270° radial canvas gauges
│   │   └── style.css                          # Military tactical HUD styling
│   ├── ml/                                    # Python ML models & parts definition
│   │   ├── lstm.py
│   │   ├── parts.py
│   │   ├── scenarios.py                       # suite, duty profiles, grouped 3-way split
│   │   └── train.py
│   └── sim/                              # CVRDE Arjun Mk-1A & MBT vehicle models
├── c_engine/                                  # MISRA-C99 Edge Inference Engine
│   ├── tank_pdm_infer.h
│   └── tank_pdm_infer.c
├── tests/                                     # Automated test suites
│   ├── test_gateway.py
│   ├── test_cvrde_subsystems.py
│   ├── test_c_engine.py
│   ├── test_pipelines.py
│   ├── test_live_ingest.py                    # Unit & integration tests for R1
│   ├── test_plausibility.py                   # Unit & adversarial tests for R2
│   ├── test_dtc_engine.py                     # Unit & compliance tests for R3
│   ├── test_pipeline.py                       # PHM block-chain wiring
│   ├── test_units.py                          # Unit-contract round trip + envelope
│   ├── test_js_python_parity.py               # docs/lstm.js vs ml/lstm.py
│   └── test_c_python_parity.py                # c_engine vs ml/lstm.py + fail-safe
└── results/
    ├── dtc_flash_log.jsonl                    # Flash ring buffer maintenance log
    └── subsystems_benchmark.json              # Real-corpus benchmark results
```
