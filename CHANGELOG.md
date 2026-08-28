# Changelog

All notable changes to **phm-vehicle** are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Deployment-grade project layout: `pyproject.toml`, `Makefile`,
  `requirements-dev.txt`, `LICENSE` (Apache-2.0), `.editorconfig`.
- `sim/generators/` package: one generator per subsystem with a
  uniform `GeneratorSpec` interface and a JSON manifest sidecar on
  every CSV.
- `tools/generate_sensor_data.py`: master CLI (`phm-generate`) for
  running one or all generators; emits a top-level `MANIFEST.json`
  aggregating every CSV written.
- `tools/check_artifacts.py`: CI gate (`phm-check-artifacts`) for the
  dashboard's `model.json` / `config.json` / `live_stream.json`
  consistency.
- `README.md` rewritten as a deployment-grade quickstart.
- `DATA.md` enumerating every external dataset, with sources, licenses,
  and SHA-256 fingerprint policy.
- `ARCHITECTURE.md` with subsystem ↔ module map, dual-tier inference
  contract, and deployment topology.

### Changed
- Moved seven legacy one-off fix-up scripts from the host monorepo's
  root to `archive/` so the deployable repo no longer carries
  one-shot dev debris.

### Removed
- `tools_check_artifacts.py` (top-level) — superseded by
  `tools/check_artifacts.py`.

## [0.1.0] — 2026-08-27

### Added (M5 audit remediation)
- Label-leakage guards (`tests/test_leakage.py`) -- the suite that
  proves a channel is a *measurement* and not a restatement of the
  injected fault parameter.
- Ingest access control (API key, per-client token bucket, payload
  cap, WebSocket origin check).
- Unit contract (`telemetry_gateway/units.py`) -- SI ↔ engineering,
  round-trip and envelope tested.
- J1939-71 conformance (EEC1 / ET1 / EFL_P1 encode + decode).
- EGT / thermostat / diesel-λ physics, anchored to documented limits.
- Grouped three-way ML split (train / val / test on different missions).
- Edge-runtime fail-safe: the C LSTM returns a fail-safe code
  (`PHM_OK | PHM_ERR_INPUT | PHM_ERR_FAULT`) instead of a NaN, and
  the JS consumer surfaces it as a non-fatal HUD chip.
- LSTM backward-gradient fix.
- Input schema D=24→26; combo label determinism; display-only
  discipline (every display channel is flagged `health_exclude=True`
  in `ml/parts.py` so a fault-free run cannot decay the health
  index).

### Added (M4 frontend + verification)
- HUD live integration (`docs/index.html`, `docs/live.js`,
  `docs/dashboard.js`, `docs/lstm.js`, `docs/gauges.js`).
- 2 s watchdog fallback in `LiveSocket`.
- DM1 / DM2 chip rendering with lamp-status colouring.
- 35+ test modules under `tests/`: unit, integration, parity, and
  adversarial FDIR + DTC tests.

### Added (M3 DTC engine)
- `telemetry_gateway/dtc_engine.py`: SAE J1939-73 SPN/FMI mapping,
  PGN 65226 DM1 + PGN 65227 DM2 encoders, on-disk flash ring buffer.

### Added (M2 FDIR plausibility)
- `telemetry_gateway/sensor_plausibility.py`: 6-layer filter
  (NaN/inf sanitiser, physical range clamp, open/short detection,
  slew-rate limiter, EMI / outlier rejection, stuck-at / dual-sensor
  consistency).

### Added (M1 ingest)
- `telemetry_gateway/live_sensor_ingest.py`: UDP 9000, Serial COM
  115200, REST `/api/telemetry/push` + `/api/telemetry/status`,
  thread-safe `TelemetryBroker`.

[Unreleased]: https://github.com/adadaabhay/Fault-Prediction-in-Vehicles/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/adadaabhay/Fault-Prediction-in-Vehicles/releases/tag/v0.1.0
