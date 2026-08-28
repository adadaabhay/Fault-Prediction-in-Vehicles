"""Tests for the hand-rolled Prometheus text-exposition metrics module.

The text exposition format is a stable plain-text contract; the test
ensures the gateway renders metrics in a shape Prometheus's parser will
accept, and that the named helpers thread-safely bump the right counters.
"""

from __future__ import annotations

import re

import pytest

from telemetry_gateway import metrics


@pytest.fixture(autouse=True)
def _reset_metrics():
    """Reset module-level counters/gauges between tests."""
    with metrics._lock:
        metrics._counters.clear()
        metrics._gauges.clear()
    yield
    with metrics._lock:
        metrics._counters.clear()
        metrics._gauges.clear()


def test_text_has_help_and_type_lines():
    """Every declared metric has both HELP and TYPE lines."""
    out = metrics.get_metrics_text()
    # Each metric we declare must show up with both a HELP and a TYPE.
    for name, _help, kind in metrics._METRICS:
        assert f"# HELP {name}" in out, f"missing HELP for {name}"
        assert f"# TYPE {name} {kind}" in out, f"missing TYPE for {name}"


def test_text_ends_with_newline():
    """The text exposition format requires a trailing newline."""
    out = metrics.get_metrics_text()
    assert out.endswith("\n")


def test_known_helpers_bump_counters():
    """inc_* helpers update the underlying counter dict."""
    metrics.inc_push_accepted()
    metrics.inc_push_accepted()
    metrics.inc_push_rejected("rate_limit")
    metrics.inc_websocket_frame()
    metrics.inc_pipeline_error("broadcast")

    out = metrics.get_metrics_text()
    assert "telemetry_push_accepted_total 2" in out
    assert re.search(
        r'telemetry_push_rejected_total\{reason="rate_limit"\} 1', out)
    assert "telemetry_websocket_frames_total 1" in out
    assert re.search(
        r'telemetry_pipeline_errors_total\{stage="broadcast"\} 1', out)


def test_gauges_reflect_last_set_value():
    """set_* helpers overwrite; gauges are last-wins, not cumulative."""
    metrics.set_active_clients(3)
    metrics.set_active_clients(5)
    metrics.set_pipeline_ready(True)
    out = metrics.get_metrics_text()
    assert "telemetry_websocket_clients 5" in out
    assert "telemetry_pipeline_ready 1" in out

    metrics.set_pipeline_ready(False)
    out = metrics.get_metrics_text()
    assert "telemetry_pipeline_ready 0" in out


def test_unincremented_counter_renders_as_zero():
    """An unseen counter still appears in the output, with value 0."""
    out = metrics.get_metrics_text()
    assert "telemetry_push_accepted_total 0" in out


def test_thread_safety_under_burst():
    """The lock keeps counters monotonic under concurrent inc."""
    import threading

    n_threads = 8
    per_thread = 500

    def burst() -> None:
        for _ in range(per_thread):
            metrics.inc_push_accepted()

    threads = [threading.Thread(target=burst) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    out = metrics.get_metrics_text()
    expected = n_threads * per_thread
    assert f"telemetry_push_accepted_total {expected}" in out
