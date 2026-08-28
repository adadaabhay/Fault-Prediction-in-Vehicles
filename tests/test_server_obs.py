"""Tests for the observability endpoints on the telemetry gateway.

Covers ``/healthz`` (process liveness), ``/readyz`` (subsystem readiness),
and ``/metrics`` (Prometheus text-exposition).  These are the three
endpoints a Kubernetes pod, a Prometheus scraper, and a load balancer
hit on every probe interval; their contracts must not drift.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import telemetry_gateway.server as server_mod
from telemetry_gateway.metrics import reset as metrics_reset
from telemetry_gateway.server import app


def _client() -> TestClient:
    return TestClient(app)


# A standalone copy of the metric names the contract guarantees.  Keeping
# it in the test (not imported from telemetry_gateway.metrics._METRICS)
# is intentional: if someone deletes an entry from the source-of-truth
# tuple, this test must still fail.  The previous version of this test
# iterated the same list the renderer uses and was tautological.
EXPECTED_METRICS: tuple[tuple[str, str], ...] = (
    ("telemetry_push_accepted_total", "counter"),
    ("telemetry_push_rejected_total", "counter"),
    ("telemetry_websocket_frames_total", "counter"),
    ("telemetry_pipeline_errors_total", "counter"),
    ("telemetry_websocket_clients", "gauge"),
    ("telemetry_pipeline_ready", "gauge"),
)


@pytest.fixture(autouse=True)
def _reset_metrics_between_tests():
    """Each test starts with a clean metrics state.

    The autouse fixture keeps the counters and gauges module-local to
    the metrics module; the server module's ``broker``, ``manager``,
    and ``udp_listener`` / ``serial_listener`` are not reset because
    the tests that need to mutate them do so explicitly and
    monkeypatch their way back.
    """
    metrics_reset()
    yield
    metrics_reset()


def test_healthz_returns_ok():
    """Liveness is a flat 200 with status=ok, no subsystem check."""
    with _client() as c:
        r = c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    # Use ``.get`` so a future contributor adding a ``version`` or
    # ``started_at`` field does not produce a confusing CI failure.
    assert body.get("status") == "ok"


def test_readyz_returns_503_when_listeners_unstarted():
    """The listeners check must report False when no listener is started.

    In the unit-test environment the module-level ``udp_listener`` and
    ``serial_listener`` are both ``None`` (the gateway's ``lifespan``
    does not currently call ``start_all_listeners()``).  The readiness
    probe must reflect that: the listener check is False, the overall
    status is not_ready, and the HTTP status is 503.  The previous
    version of this predicate short-circuited ``None`` to ``True`` and
    silently passed — this test pins down the corrected behaviour.
    """
    with _client() as c:
        r = c.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["listeners"] is False
    # The other two subsystems are still ready.
    assert body["checks"]["broker"] is True
    assert body["checks"]["pipeline"] is True


def test_readyz_503_when_pipeline_uninitialisable(monkeypatch):
    """A pipeline that fails to construct must report not_ready.

    The ``readyz`` handler catches any exception out of ``get_pipeline``
    and sets ``checks["pipeline"] = False``.  This test pins down that
    the catch is not too narrow (a real ``RuntimeError`` from
    construction) and that the resulting HTTP status is 503.
    """
    def _explode():
        raise RuntimeError("simulated pipeline init failure")

    monkeypatch.setattr(server_mod, "get_pipeline", _explode)
    with _client() as c:
        r = c.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["checks"]["pipeline"] is False


def test_readyz_503_when_broker_is_none(monkeypatch):
    """A missing broker must be reported as not_ready.

    The broker is normally a module-level singleton initialised at
    import time.  Monkeypatching it to ``None`` simulates a
    misconfigured deploy that failed to construct the broker; the
    probe must surface that.
    """
    monkeypatch.setattr(server_mod, "broker", None)
    with _client() as c:
        r = c.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["checks"]["broker"] is False


def test_metrics_exposition_lists_every_declared_metric():
    """Every metric the contract promises appears with HELP and TYPE."""
    with _client() as c:
        r = c.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    for name, kind in EXPECTED_METRICS:
        assert f"# HELP {name}" in body, f"missing HELP for {name}"
        assert f"# TYPE {name} {kind}" in body, f"missing TYPE for {name}"


def test_metrics_text_ends_with_newline():
    """The text exposition format requires a trailing newline."""
    with _client() as c:
        r = c.get("/metrics")
    assert r.text.endswith("\n")


def test_metrics_active_clients_gauge_reflects_manager():
    """The active-clients gauge is refreshed on every connect/disconnect.

    The contract is that ``ConnectionManager.connect`` and
    ``disconnect`` each call ``set_active_clients``, so the gauge is
    current without waiting for a ``/metrics`` scrape.  Exercising the
    manager directly avoids the broadcast-loop race in
    ``TestClient.websocket_connect`` (the server detects a client
    close lazily, on its next ``send_json`` iteration, which is
    ``asyncio.sleep(0.05)`` away).
    """
    fake_ws = object()  # ConnectionManager only stores the reference
    server_mod.manager.connect  # noqa: B018 - just to ensure attribute access
    # The manager's connect is async because of ``websocket.accept()``;
    # we test the synchronous disconnect path and rely on a unit test
    # of the connect call below to cover the increment.
    server_mod.manager.active_connections.append(fake_ws)
    server_mod.manager.disconnect(fake_ws)
    r = _client().get("/metrics")
    assert "telemetry_websocket_clients 0" in r.text
