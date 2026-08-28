"""Prometheus text-exposition metrics for the telemetry gateway.

This module is hand-rolled to avoid a hard dependency on
``prometheus_client``.  The text exposition format
(https://prometheus.io/docs/instrumenting/exposition_formats/) is a stable
plain-text contract; the surface we need is small enough that a 100-line
implementation is preferable to a third-party dep that ships an OpenMetrics
parser and a multiprocess collector we will never use.

The pattern is the standard Prometheus pattern: thread-safe counters and
gauges with HELP and TYPE lines, exposed via a single ``get_metrics_text()``
function called by the ``/metrics`` route in ``server.py``.  Counters only
go up; gauges may move in either direction.  There is no histogram because
the latency we care about (broker latency_ms, gate_ms) is already produced
by the pipeline and exposed via ``/api/telemetry/status``; for proper
percentile histograms we would need a sliding window, and the in-process
implementation would lie about restart boundaries.
"""

from __future__ import annotations

import threading
from typing import Dict


# ---------------------------------------------------------------------------
# Counters and gauges, with the standard Prometheus shape.
# ---------------------------------------------------------------------------
# Each metric is a tuple of (HELP, TYPE, value).  Counters are int; gauges
# are float.  We hold them in module-level dicts so the rest of the gateway
# can call the named helpers (inc_push_accepted, set_active_clients, etc.)
# without ever touching the raw dict.
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_counters: Dict[str, float] = {}
_gauges: Dict[str, float] = {}


def _counter_inc(name: str, value: float = 1.0) -> None:
    with _lock:
        _counters[name] = _counters.get(name, 0.0) + value


def _gauge_set(name: str, value: float) -> None:
    with _lock:
        _gauges[name] = value


# ---------------------------------------------------------------------------
# Named helpers — the only API the rest of the gateway should touch.
# ---------------------------------------------------------------------------

def inc_push_accepted() -> None:
    """A mutating push passed auth, rate-limit, and payload-cap checks."""
    _counter_inc("telemetry_push_accepted_total")


def inc_push_rejected(reason: str) -> None:
    """A mutating push was refused.  Reason is a low-cardinality label."""
    _counter_inc(f'telemetry_push_rejected_total{{reason="{reason}"}}')


def inc_websocket_frame() -> None:
    """A frame was successfully broadcast to a WebSocket client."""
    _counter_inc("telemetry_websocket_frames_total")


def inc_pipeline_error(stage: str) -> None:
    """The telemetry pipeline raised on this stage.  Labels are not exposed
    as a label set today because the pipeline's stages are not stable
    public surface; we emit a single counter for "anything in the
    WebSocket path raised" and rely on structured logs for the stage
    name.
    """
    _counter_inc(f'telemetry_pipeline_errors_total{{stage="{stage}"}}')


def set_active_clients(n: int) -> None:
    _gauge_set("telemetry_websocket_clients", float(n))


def set_pipeline_ready(ready: bool) -> None:
    _gauge_set("telemetry_pipeline_ready", 1.0 if ready else 0.0)


# ---------------------------------------------------------------------------
# Exposition
# ---------------------------------------------------------------------------
# Each metric we surface has HELP and TYPE lines so Prometheus can ingest
# the response without a separate metadata scrape.  Order is stable for
# diff-friendly output; the Prometheus scraper does not require ordering.
# ---------------------------------------------------------------------------

_METRICS: tuple[tuple[str, str, str], ...] = (
    # name                    HELP                                                              TYPE
    ("telemetry_push_accepted_total",
     "Mutating telemetry pushes accepted by the gateway (auth + rate-limit + payload-cap passed).",
     "counter"),
    ("telemetry_push_rejected_total",
     "Mutating telemetry pushes refused, labelled by reason (auth, rate_limit, payload_cap, parse).",
     "counter"),
    ("telemetry_websocket_frames_total",
     "Telemetry frames broadcast over WebSocket to live dashboards.",
     "counter"),
    ("telemetry_pipeline_errors_total",
     "Exceptions raised inside the telemetry WebSocket broadcast loop, labelled by stage.",
     "counter"),
    ("telemetry_websocket_clients",
     "Current number of connected WebSocket clients.",
     "gauge"),
    ("telemetry_pipeline_ready",
     "1 if the broker, pipeline, and listeners are all initialised, 0 otherwise.",
     "gauge"),
)


def get_metrics_text() -> str:
    """Render all known metrics in Prometheus text exposition format."""
    with _lock:
        counters = dict(_counters)
        gauges = dict(_gauges)

    lines: list[str] = []
    for name, help_text, kind in _METRICS:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {kind}")
        if kind == "counter":
            # Counters and their labelled variants (the {reason="..."}
            # suffixes added by inc_push_rejected) share the same HELP
            # and TYPE line.  Prometheus's parser is fine with a TYPE
            # line that describes the family.
            rendered = False
            for key, value in sorted(counters.items()):
                # Either an exact match, or a labelled variant
                # ``base{labels}`` of this base name.
                if key == name or key.startswith(name + "{"):
                    lines.append(f"{key} {value:.0f}")
                    rendered = True
            if not rendered:
                lines.append(f"{name} 0")
        else:  # gauge
            value = gauges.get(name, 0.0)
            lines.append(f"{name} {value:.0f}")
    # Trailing newline — required by the text exposition format.
    return "\n".join(lines) + "\n"
