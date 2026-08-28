"""Tests for the hand-rolled Prometheus text-exposition metrics module.

The text exposition format is a stable plain-text contract; the test
ensures the gateway renders metrics in a shape Prometheus's parser will
accept, and that the named helpers thread-safely bump the right counters.
"""

from __future__ import annotations

import re
import threading

import pytest

from telemetry_gateway import metrics


# A standalone copy of the metric names the contract guarantees.  Kept in
# the test (not imported from ``metrics._METRICS``) so a future change
# that drops a metric from the source-of-truth tuple is caught here.
# The previous version of ``test_text_has_help_and_type_lines`` iterated
# the same list the renderer used and was tautological.
EXPECTED_COUNTERS: tuple[str, ...] = (
    "telemetry_push_accepted_total",
    "telemetry_push_rejected_total",
    "telemetry_websocket_frames_total",
    "telemetry_pipeline_errors_total",
)


@pytest.fixture(autouse=True)
def _reset_metrics():
    """Reset module-level counters/gauges between tests via the public API."""
    metrics.reset()
    yield
    metrics.reset()


def test_text_has_help_and_type_lines():
    """Every declared metric has both HELP and TYPE lines.

    The metric names are checked against the standalone ``EXPECTED_COUNTERS``
    tuple, not against ``metrics._METRICS``: if someone removes a metric
    from the renderer source-of-truth, this test must still fail.
    """
    out = metrics.get_metrics_text()
    for name in EXPECTED_COUNTERS:
        assert f"# HELP {name}" in out, f"missing HELP for {name}"
        assert f"# TYPE {name} counter" in out, f"missing TYPE for {name}"
    # The two gauges are not in EXPECTED_COUNTERS; pin them down here.
    for name, kind in (
        ("telemetry_websocket_clients", "gauge"),
        ("telemetry_pipeline_ready", "gauge"),
    ):
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


def test_every_documented_rejection_reason_renders():
    """All four rejection reasons the CHANGELOG advertises must render.

    The CHANGELOG entry for Phase 5 listed ``auth``, ``rate_limit``,
    ``payload_cap``, and ``parse`` as documented rejection reasons.
    Pinning each of them down here means a future caller that uses
    one of these strings and a future test that wants to assert the
    counter is bumped have a working labelled-variant of the metric.
    """
    for reason in ("auth", "rate_limit", "payload_cap", "parse", "ingest"):
        metrics.inc_push_rejected(reason)
    out = metrics.get_metrics_text()
    for reason in ("auth", "rate_limit", "payload_cap", "parse", "ingest"):
        assert (
            f'telemetry_push_rejected_total{{reason="{reason}"}} 1' in out
        ), f"missing rejection series for reason={reason}"


def test_label_values_are_escaped():
    """Label values with backslash, quote, or newline are escaped.

    The Prometheus text-exposition spec requires backslash, double-quote,
    and newline to be escaped in label values.  Without this, a label like
    ``reason='a"b'`` would break the scraper.
    """
    metrics.inc_push_rejected('a"b\\c\nd')
    out = metrics.get_metrics_text()
    # The escape must produce a well-formed line: a literal backslash
    # before the quote and newline, with a literal double-backslash
    # before the lone backslash.
    assert r'telemetry_push_rejected_total{reason="a\"b\\c\nd"} 1' in out


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
    """Every counter is emitted as 0 before any inc_ call.

    The renderer emits ``name 0`` for unseen counters so a fresh
    process is not a black box to a scraper.  Asserting the contract
    on every counter (not just one) catches a regression where one
    counter's zero-render path was deleted.
    """
    out = metrics.get_metrics_text()
    for name in EXPECTED_COUNTERS:
        assert f"{name} 0" in out, f"{name} not rendered as 0"


def test_thread_safety_under_burst():
    """The lock keeps counters monotonic under concurrent inc."""
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


def test_reset_clears_counters_and_gauges():
    """The public reset() empties the modules' state."""
    metrics.inc_push_accepted()
    metrics.set_pipeline_ready(True)
    metrics.reset()
    out = metrics.get_metrics_text()
    assert "telemetry_push_accepted_total 0" in out
    assert "telemetry_pipeline_ready 0" in out
