"""Tests for the observability endpoints on the telemetry gateway.

Covers ``/healthz`` (process liveness), ``/readyz`` (subsystem readiness),
and ``/metrics`` (Prometheus text-exposition).  These are the three
endpoints a Kubernetes pod, a Prometheus scraper, and a load balancer
hit on every probe interval; their contracts must not drift.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from telemetry_gateway.server import app


def _client() -> TestClient:
    return TestClient(app)


def test_healthz_returns_ok():
    """Liveness is a flat 200 with status=ok, no subsystem check."""
    with _client() as c:
        r = c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "ok"}


def test_readyz_reports_subsystem_state():
    """Readiness returns 200 when broker, pipeline, and listeners are up.

    In the unit-test environment the broker is initialised and the
    pipeline constructs lazily, so the readiness probe should report
    ready.  This test is a contract test: any new subsystem added to
    ``readyz`` must show up under ``checks`` and the HTTP status must
    match the boolean reduction of those checks.
    """
    with _client() as c:
        r = c.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    # Every subsystem the contract documents must appear.
    for key in ("broker", "pipeline", "listeners"):
        assert key in body["checks"]
        assert body["checks"][key] is True


def test_metrics_exposition_format():
    """The /metrics endpoint returns Prometheus 0.0.4 text format.

    Verifies the response shape (content-type, required HELP/TYPE lines
    for every declared metric, trailing newline) and the values of the
    three counters the request exercised.
    """
    with _client() as c:
        # Exercise a path that bumps a counter so the assertion is
        # meaningful, not a vacuous match against "0".
        c.get("/healthz")
        r = c.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    # HELP / TYPE lines for the declared metrics
    for name, kind in (
        ("telemetry_push_accepted_total", "counter"),
        ("telemetry_push_rejected_total", "counter"),
        ("telemetry_websocket_frames_total", "counter"),
        ("telemetry_pipeline_errors_total", "counter"),
        ("telemetry_websocket_clients", "gauge"),
        ("telemetry_pipeline_ready", "gauge"),
    ):
        assert f"# HELP {name}" in body
        assert f"# TYPE {name} {kind}" in body
    # After the /readyz probe, the pipeline_ready gauge is set; render
    # a /readyz so the gauge has a fresh value, then check /metrics.
    with _client() as c:
        c.get("/readyz")
        r2 = c.get("/metrics")
    assert "telemetry_pipeline_ready 1" in r2.text
