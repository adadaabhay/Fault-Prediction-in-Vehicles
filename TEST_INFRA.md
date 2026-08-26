# E2E Test Infra: Defense-Grade PHM/CBM+ Ecosystem

## Test Philosophy
- Requirement-driven, opaque-box and gray-box validation for defense-grade reliability.
- Methodology: Category-Partition + Boundary Value Analysis + Pairwise Combinations + Adversarial Stress / Fuzzing + Real-World Tactical Scenarios.

## Feature Inventory & Test Coverage Matrix
| # | Feature | Source | Tier 1 (Unit) | Tier 2 (Boundary/Fuzz) | Tier 3 (Pairwise) | Tier 4 (E2E Scenario) |
|---|---------|--------|:-------------:|:----------------------:|:-----------------:|:---------------------:|
| 1 | UDP 9000 Ingestion | ORIGINAL_REQUEST §R1 | ≥5 | ≥5 | ✓ | ✓ |
| 2 | Serial COM 115200 Ingest | ORIGINAL_REQUEST §R1 | ≥5 | ≥5 | ✓ | ✓ |
| 3 | REST /api/telemetry/push | ORIGINAL_REQUEST §R1 | ≥5 | ≥5 | ✓ | ✓ |
| 4 | TelemetryBroker | ORIGINAL_REQUEST §R1 | ≥5 | ≥5 | ✓ | ✓ |
| 5 | Range & NaN Sanitizer | ORIGINAL_REQUEST §R2 | ≥5 | ≥5 | ✓ | ✓ |
| 6 | Open / Short Circuit Gate | ORIGINAL_REQUEST §R2 | ≥5 | ≥5 | ✓ | ✓ |
| 7 | Slew Rate & EMI Filter | ORIGINAL_REQUEST §R2 | ≥5 | ≥5 | ✓ | ✓ |
| 8 | Stuck-At / Dual Sensor | ORIGINAL_REQUEST §R2 | ≥5 | ≥5 | ✓ | ✓ |
| 9 | SAE J1939 SPN/FMI Mapper | ORIGINAL_REQUEST §R3 | ≥5 | ≥5 | ✓ | ✓ |
| 10 | J1939 DM1 Active Encoder | ORIGINAL_REQUEST §R3 | ≥5 | ≥5 | ✓ | ✓ |
| 11 | J1939 DM2 Historic Log | ORIGINAL_REQUEST §R3 | ≥5 | ≥5 | ✓ | ✓ |
| 12 | Flash Ring Buffer Persistence | ORIGINAL_REQUEST §R3 | ≥5 | ≥5 | ✓ | ✓ |
| 13 | HUD Stream Indicator & Fallback | ORIGINAL_REQUEST §R1, §AC6 | ≥5 | ≥5 | ✓ | ✓ |
| 14 | <20ms E2E Latency Pipeline | ORIGINAL_REQUEST §AC1 | ≥5 | ≥5 | ✓ | ✓ |

## Test Architecture
- Test Runner: Python `unittest` standard test discovery and custom master runner `run_all_tests.py`.
- Test Locations:
  - `tests/test_live_ingest.py`: Tests UDP port 9000, Serial COM streaming, REST push API, TelemetryBroker.
  - `tests/test_plausibility.py`: Tests physical clamping, NaN/inf sanitization, open/short circuit detection (P_oil=-999), slew rate limiting, stuck-at detection, Hampel EMI filtering.
  - `tests/test_dtc_engine.py`: Tests SPN/FMI mapping, DM1 PGN 65226 binary encoding, DM2 PGN 65227 historic transitions, on-disk flash ring buffer persistence & rollover.
  - `tests/test_e2e_integration.py`: End-to-end multi-vehicle scenario tests (Arjun Mk-1A Thar assault, T-90S combat sprint, Zorawar high-altitude) validating full pipeline from UDP push to DM1 generation.
  - `tests/test_gateway.py`, `tests/test_cvrde_subsystems.py`, `tests/test_c_engine.py`, `tests/test_pipelines.py`: Existing regression suites.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Arjun Mk-1A Thar Assault with Sand Filter Clogging & Wire Cut | UDP 9000 + FDIR Open-Circuit + J1939 DM1 (SPN 100 FMI 04) + LSTM Ingestion | High |
| 2 | T-90S High-Speed Maneuver with Engine Overheat & Severe EMI Spikes | REST Push + Slew Rate Limiter + Hampel EMI Rejection + DM1 (SPN 110 FMI 00) | High |
| 3 | Zorawar Sub-Zero Cold-Start with Stuck Hydraulic Sensor | Serial Ingest + Stuck-At Detector + DM1 (SPN 520104 FMI 02) + Flash Ring Buffer | High |
| 4 | Telemetry Connection Loss & Automatic Watchdog Replay Fallback | Broker + Watchdog 2.0s + HUD Simulation Stream Reconnection | Medium |
| 5 | Master Adversarial Stress Blast (1000 Corrupted Frames) | Zero crashes, 100% clamping, all DTCs properly logged, <20ms per frame | High |

## Coverage Thresholds
- Unit & Boundary tests: 100% test coverage across `telemetry_gateway/live_sensor_ingest.py`, `telemetry_gateway/sensor_plausibility.py`, and `telemetry_gateway/dtc_engine.py`.
- Regressions: 0 failed tests across existing codebase.
- Master Test Harness exit code: 0.
