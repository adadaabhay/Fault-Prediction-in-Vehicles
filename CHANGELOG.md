# Changelog

All notable changes to this project are documented here.  The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `SECURITY.md` — supported-version matrix, reporting policy, threat
  model, and an honest claim about the C engine's MISRA position
  (informed, not compliant; see `c_engine/CODING_STANDARD.md`).
- `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1.
- `CONTRIBUTING.md` — dev setup, test conventions, the six PR
  invariants (`one concern per PR`, `tests first`, `no silent skip`,
  `no silent fallback`, `no hardcoded paths outside the repo`,
  `document the public surface`), changelog rules, and the
  `pr-review-toolkit` agents the maintainers use to review their own
  PRs.
- `.github/dependabot.yml` — weekly pip + GitHub Actions update
  cadence, grouped PRs (`runtime` vs `dev`), `dependencies` + `ci`
  labels, and the security-update priority semantics.
- `.pre-commit-config.yaml` — `pre-commit/pre-commit-hooks` (YAML /
  JSON / TOML / large-file / merge-conflict / debug-statements),
  `ruff` + `ruff-format`, `mypy` on the public surface, and a local
  hook that runs `python -m tools.check_artifacts` and
  `python -m tools.check_links` so the no-skip policy is enforced at
  commit time, not just in CI.
- `telemetry_gateway/metrics.py` — hand-rolled Prometheus text
  exposition (no `prometheus_client` dep) with six named metrics:
  `telemetry_push_accepted_total`, `telemetry_push_rejected_total`
  (labelled by reason), `telemetry_websocket_frames_total`,
  `telemetry_pipeline_errors_total` (labelled by stage),
  `telemetry_websocket_clients`, `telemetry_pipeline_ready`.  See
  the module docstring for why we did not pull in a third-party
  library.
- `GET /healthz` — liveness probe.  Returns 200 if the worker is
  alive.  This is what Kubernetes / Docker / load balancers use
  for the LivenessProbe.
- `GET /readyz` — readiness probe.  Returns 200 only if the broker
  and the pipeline are wired.  Returns 503 otherwise.  A process
  that is alive but not ready is taken out of the LB rotation but
  is not restarted.
- `GET /metrics` — Prometheus text exposition.  See
  `telemetry_gateway/metrics.py` for the metric definitions and the
  exposition contract.
- `Dockerfile` — multi-stage build.  Builder stage uses
  `python:3.12-slim` with `build-essential`, `cmake`, and
  `libusb-1.0-0-dev` to compile the C engine and resolve all wheels.
  Final stage uses `gcr.io/distroless/python3-debian12:nonroot`
  (no shell, no package manager, UID 65532) and exposes
  `:8000` (gateway) and `:8080` (dashboard).  `HEALTHCHECK` hits
  `/healthz` every 30 s.  See the header comment for build / run /
  verify commands.
- `tests/test_metrics.py` — 6 cases covering the exposition
  contract, named-helper behaviour, gauge last-wins, and thread
  safety under concurrent `inc_*` bursts.
- `tests/test_server_obs.py` — 3 cases for `/healthz`, `/readyz`,
  and `/metrics`.  These are contract tests: any drift in the
  observability surface is caught at PR time, not at deploy time.

### Changed

- `telemetry_gateway/server.py` instruments the `push` endpoint and
  the WebSocket loop with the new metrics helpers.  Push-accepted
  is bumped after `_require_api_key` so a 401 never reaches the
  counter; push-rejected carries a `reason` label
  (`rate_limit` today; the helper accepts any low-cardinality
  string).  The WebSocket broadcast loop bumps
  `telemetry_websocket_frames_total` on every successful send and
  `telemetry_pipeline_errors_total` on any unhandled exception;
  the bare `except Exception: disconnect()` is now `except Exception
  as exc: inc_pipeline_error("websocket_broadcast"); ...` so a
  pipeline error is visible in `/metrics`, not just in
  `logger.exception`.
