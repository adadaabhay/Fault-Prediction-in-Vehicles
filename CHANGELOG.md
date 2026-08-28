# Changelog

All notable changes to this project are documented here.  The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- `telemetry_gateway/server.py` push endpoint: `inc_push_accepted()`
  is now bumped **after** every guard (auth, rate-limit, payload-cap,
  parse, type-check) passes, just before `broker.push_telemetry()`.
  The two counter series are now mutually exclusive — a 429, 413, 400,
  or 500 path no longer overcounts as both accepted and rejected.
  New `inc_push_rejected(...)` call sites cover the previously
  uncounted reasons: `auth` (in `_require_api_key`), `payload_cap`,
  `parse`, and `ingest`.  The metric's HELP text — which already
  documented these reasons — now matches what the code emits.
- `telemetry_gateway/server.py` `/readyz` listener predicate:
  `(listener is None or listener.is_running)` short-circuited to
  `True` whenever no listener had been started, making the 503 path
  unreachable for the listeners check.  Replaced with the inverted
  `(listener is not None and listener.is_running)` joined by `or`
  so the readiness probe honestly reports "no listener bound yet"
  as not-ready.
- `telemetry_gateway/server.py` WebSocket handler: a bare
  `except Exception:` swallowed every error with the same logging
  level and the same metric label.  The handler now narrows the
  catch to `(ConnectionResetError, RuntimeError)` for client-abort
  cases (logged at INFO) and reserves a separate `except Exception`
  for genuine pipeline bugs (logged at ERROR with traceback).  The
  double-logged exception string (`logger.exception(..., exc)`) is
  removed — `logger.exception` already appends the traceback.
- `telemetry_gateway/metrics.py` Prometheus label values are now
  escaped per the text-exposition spec (backslash, double-quote,
  newline).  A label like `reason='a"b\\c\nd'` no longer produces a
  malformed exposition line.
- `telemetry_gateway/metrics.py` exposes a public `reset()` for
  tests, so fixtures stop reaching into the private `_lock` /
  `_counters` / `_gauges` triple.
- `Dockerfile` pins the actual install plan (`.[bench]`); the
  previous `.[bench,observability]` referenced an extra that does
  not exist in `pyproject.toml` and was silently masked by a
  `|| pip wheel ...` fallback.  The Dockerfile's `HEALTHCHECK` now
  probes `/readyz` instead of `/healthz` — Docker's HEALTHCHECK
  semantic is "container is doing useful work, restart it if not",
  which is closer to a readiness/restart probe; `/healthz` is by
  design a flat 200 on any live worker, so a container with a
  broken broker would have passed `/healthz` forever and Docker
  would never restart it.
- `SECURITY.md` threat model now names the new unauthenticated
  endpoints (`/healthz`, `/readyz`, `/metrics`) as intentionally
  public, and points the reporting contact at GitHub Security
  Advisories as a non-`.local` fallback for the maintainer's
  mDNS-only address.  The "no TODO/FIXME" compliance claim is
  restated as build-gate-enforced rather than asserted.
- `CONTRIBUTING.md` no longer hard-codes the suite count (which
  drifted past 317/30 after the v0.2.0 audit removed 11 tests);
  readers are pointed at `VERIFICATION.md` for the current numbers.

### Tests

- `tests/test_server_obs.py` adds three new cases:
  `test_readyz_returns_503_when_listeners_unstarted` (pins down the
  corrected listener predicate), `test_readyz_503_when_pipeline_uninitialisable`
  (monkeypatches `get_pipeline` to raise and asserts 503),
  `test_readyz_503_when_broker_is_none` (monkeypatches `broker` to
  `None` and asserts 503), and `test_metrics_active_clients_gauge_reflects_manager`
  (opens a real WebSocket, asserts the gauge flips to 1, closes, asserts
  it drops to 0).  The previous `/readyz` test, which ratifies the
  inverted predicate, is replaced.
- `tests/test_metrics.py` `test_text_has_help_and_type_lines` was
  tautological (iterated the same source-of-truth the renderer
  used); rewritten to assert against a standalone
  `EXPECTED_COUNTERS` tuple so a regression that removes a metric
  is caught.  Adds `test_every_documented_rejection_reason_renders`
  (one case per documented `reason=` label), `test_label_values_are_escaped`
  (backslash / quote / newline escaping per the spec), and
  `test_reset_clears_counters_and_gauges` (the public `reset()`).
  The fixture now calls `metrics.reset()` instead of reaching into
  the module's private state.
- `tests/test_server_obs.py` `test_healthz_returns_ok` now asserts
  `body.get("status") == "ok"` instead of full-body equality, so a
  future contributor adding a `version` or `started_at` field
  does not produce a confusing CI failure.

### Known limitations (carried over)

- The gateway's `lifespan` does not call `start_all_listeners()`,
  so the `udp_listener` / `serial_listener` module-level globals
  are `None` for the lifetime of the process.  This is a
  pre-existing condition that the new `/readyz` predicate now
  surfaces honestly (the readiness probe reports not-ready on a
  fresh container until the lifespan calls the listener starter).
  Fixing the lifespan is a v0.2.1 task; tracked separately.

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
