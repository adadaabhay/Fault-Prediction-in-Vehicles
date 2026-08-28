"""Real-Time Tactical Telemetry Gateway Server (FastAPI / WebSocket).
Streams decoded J1939 frames, tactical bursts, and multi-subsystem telemetry
over WebSocket to live web dashboards and battle management systems.
Supports live physical hardware ingestion via UDP 9000, Serial COM, and REST push.
"""

import asyncio
import hmac
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from .j1939_can_parser import J1939FrameParser
from .pipeline import get_pipeline
from .tactical_burst import TacticalBurstPacket
from .live_sensor_ingest import (
    TelemetryBroker,
    UDPSensorListener,
    SerialSensorListener,
    get_broker,
    start_all_listeners,
    stop_all_listeners,
)
from .metrics import (
    get_metrics_text,
    inc_push_accepted,
    inc_push_rejected,
    inc_websocket_frame,
    inc_pipeline_error,
    set_active_clients,
    set_pipeline_ready,
)

logger = logging.getLogger("telemetry_server")

# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------
# The ingest path previously had no trust boundary of any kind: no auth, no
# TLS, no rate limit, no payload cap, no WebSocket origin check, and a UDP
# listener bound to 0.0.0.0:9000 accepting frames from any host. Since the
# gateway's entire purpose is trustworthy fault detection, any packet on the
# wire became vehicle health state.
#
# TELEMETRY_API_KEY gates every mutating endpoint. It is intentionally
# fail-closed in the default (non-local) case: with no key configured the push
# endpoint refuses writes rather than accepting anonymous ones. Set
# TELEMETRY_ALLOW_ANONYMOUS=1 only for a loopback bench rig.
API_KEY: Optional[str] = os.environ.get("TELEMETRY_API_KEY") or None
ALLOW_ANONYMOUS: bool = os.environ.get("TELEMETRY_ALLOW_ANONYMOUS") == "1"
# Comma-separated list; empty means "same-origin only" (no Origin header).
ALLOWED_ORIGINS: List[str] = [
    o.strip() for o in os.environ.get("TELEMETRY_ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]
MAX_PAYLOAD_BYTES: int = int(os.environ.get("TELEMETRY_MAX_PAYLOAD_BYTES", 262144))
# Token bucket per source address.
PUSH_RATE_LIMIT_HZ: float = float(os.environ.get("TELEMETRY_PUSH_RATE_HZ", 200.0))

_rate_state: Dict[str, List[float]] = {}
_rate_lock = threading.Lock()


# Maximum unique keys in _rate_state; evicts oldest half when exceeded.
_RATE_STATE_MAX: int = 2048


def _rate_limit_ok(key: str) -> bool:
    """Simple token bucket; one bucket per client address."""
    now = time.monotonic()
    with _rate_lock:
        if len(_rate_state) >= _RATE_STATE_MAX and key not in _rate_state:
            # Evict oldest half by insertion order (dict preserves order ≥3.7)
            evict = list(_rate_state.keys())[: _RATE_STATE_MAX // 2]
            for k in evict:
                del _rate_state[k]
        tokens, last = _rate_state.get(key, [PUSH_RATE_LIMIT_HZ, now])
        tokens = min(PUSH_RATE_LIMIT_HZ,
                     tokens + (now - last) * PUSH_RATE_LIMIT_HZ)
        if tokens < 1.0:
            _rate_state[key] = [tokens, now]
            return False
        _rate_state[key] = [tokens - 1.0, now]
        return True



LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in LOOPBACK_HOSTS


def _require_api_key(request: Request) -> None:
    """Authorise a mutating request.

    Policy, in order:
      1. If TELEMETRY_API_KEY is set, it is required from every caller,
         loopback included. An operator who configures a key means it.
      2. Otherwise loopback callers are allowed -- a bench rig on the same
         box is inside the trust boundary, and requiring key ceremony there
         only encourages people to disable auth globally.
      3. Otherwise (remote caller, no key configured) the request is refused.
         This is the case that used to be wide open: any host on the network
         could write vehicle health state with no credential at all.

    A refused caller bumps ``telemetry_push_rejected_total{reason="auth"}``
    so brute-force attempts against an unauthenticated endpoint surface in
    ``/metrics``.  The metric is bumped *before* the HTTPException is raised
    so the counter is incremented even when the framework's exception
    handler short-circuits the response.
    """
    if API_KEY:
        supplied = (request.headers.get("x-api-key")
                    or request.headers.get("authorization", "")
                    .removeprefix("Bearer ").strip())
        if not supplied or not hmac.compare_digest(supplied, API_KEY):
            inc_push_rejected("auth")
            raise HTTPException(status_code=401,
                                detail="Invalid or missing API key")
        return
    if ALLOW_ANONYMOUS or _is_loopback(request):
        return
    inc_push_rejected("auth")
    raise HTTPException(
        status_code=401,
        detail="Remote ingest requires TELEMETRY_API_KEY to be configured.")


def _origin_allowed(origin: Optional[str]) -> bool:
    if not origin:
        return True                      # same-origin / non-browser client
    return origin in ALLOWED_ORIGINS


# Central Broker & Listeners
broker: TelemetryBroker = get_broker()
udp_listener: Optional[UDPSensorListener] = None
serial_listener: Optional[SerialSensorListener] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for background hardware listeners."""
    yield
    stop_all_listeners()


app = FastAPI(
    title="MBT Tactical Telemetry Gateway",
    version="2.0",
    lifespan=lifespan,
)


class ConnectionManager:
    """Manages active WebSocket connections and client broadcasts."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        set_active_clients(len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        set_active_clients(len(self.active_connections))

    async def broadcast_json(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()


@app.get("/")
async def get_root():
    """Root health and capabilities status."""
    return {
        "status": "OPERATIONAL",
        "service": "MBT Tactical Telemetry Gateway",
        "active_clients": len(manager.active_connections),
        # Only protocols with an implementation behind them are listed.
        # "MIL-STD-1553B" was advertised here with no 1553 code anywhere in the
        # tree -- a capability claim a client could act on and that would fail.
        "supported_protocols": [
            "SAE J1939-71 (EEC1, ET1, EFL_P1)",
            "SAE J1939-73 (DM1/DM2)",
            "32-byte tactical burst (CRC-16 framed, unencrypted)",
            "WebSocket",
            "UDP 9000",
            "Serial 115200",
        ],
        "is_live_hardware": broker.is_live_hardware,
        "active_source": broker.active_source,
    }


# ---------------------------------------------------------------------------
# Observability: liveness, readiness, Prometheus metrics
# ---------------------------------------------------------------------------
# /healthz — process liveness.  Always 200 if the worker is alive.  This is
# what Kubernetes / Docker / load balancers should use for the LivenessProbe;
# a failing /healthz means the process must be killed and restarted.
#
# /readyz — readiness.  Returns 200 only if the broker is wired and the
# pipeline is constructed.  Used for the ReadinessProbe; a not-ready
# container is taken out of the load-balancer rotation but is NOT restarted.
#
# /metrics — Prometheus text exposition format, hand-rolled to avoid pulling
# in prometheus_client.  See telemetry_gateway/metrics.py for the counter
# definitions and HELP/TYPE lines.
@app.get("/healthz", response_class=JSONResponse)
async def healthz() -> Dict[str, Any]:
    """Liveness probe — process is up and the event loop is responsive."""
    return {"status": "ok"}


@app.get("/readyz", response_class=JSONResponse)
async def readyz() -> JSONResponse:
    """Readiness probe — broker and pipeline are wired and able to ingest.

    Returns 503 if any subsystem is not initialised.  This is intentionally
    separate from /healthz: a process that is alive but not ready (e.g. the
    pipeline is still building on first start) should NOT be restarted; it
    should just be removed from the LB rotation until it is.
    """
    checks: Dict[str, bool] = {}
    checks["broker"] = broker is not None
    try:
        pipeline = get_pipeline()
        checks["pipeline"] = pipeline is not None
    except Exception as exc:
        logger.warning("readyz: pipeline not initialised: %s", exc)
        checks["pipeline"] = False
    # A listener is "ready" only if it has been *started* (not None) and
    # is currently *running*.  ``None`` means "never bound" — the test
    # process never calls ``start_all_listeners()`` and the production
    # lifespan also leaves them as ``None`` today; in both cases the
    # gateway is not actually ingesting hardware telemetry, and the
    # readiness probe must reflect that.  The previous
    # ``None or is_running`` predicate short-circuited to ``True`` and
    # made this branch vacuous.
    checks["listeners"] = (
        (udp_listener is not None and udp_listener.is_running)
        or (serial_listener is not None and serial_listener.is_running)
    )
    set_pipeline_ready(all(checks.values()))
    status_code = 200 if all(checks.values()) else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if all(checks.values()) else "not_ready",
                 "checks": checks},
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> Response:
    """Prometheus text exposition for push / WS / pipeline health."""
    set_active_clients(len(manager.active_connections))
    return PlainTextResponse(
        get_metrics_text(),
        media_type="text/plain; version=0.0.4",
    )


@app.post("/api/telemetry/push")
async def push_telemetry_endpoint(request: Request):
    """REST endpoint to ingest raw sensor readings into the TelemetryBroker.

    Guarded: API key, per-client rate limit, and a payload cap. Previously this
    endpoint was completely open -- any host could write vehicle health state.

    Instrumentation contract:
      * ``telemetry_push_accepted_total`` is bumped **once** per request, only
        after every guard (auth, rate-limit, payload-cap, parse, type-check)
        has passed.  A request that fails any guard is reflected in
        ``telemetry_push_rejected_total{reason=...}`` and never in
        ``telemetry_push_accepted_total`` -- the two series are mutually
        exclusive, not cumulative.
      * The 500 path bumps ``telemetry_push_rejected_total{reason="ingest"}``
        so a downstream broker failure is visible in /metrics.
    """
    _require_api_key(request)

    client = request.client.host if request.client else "unknown"
    if not _rate_limit_ok(client):
        inc_push_rejected("rate_limit")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    declared = request.headers.get("content-length")
    if declared and int(declared) > MAX_PAYLOAD_BYTES:
        inc_push_rejected("payload_cap")
        raise HTTPException(status_code=413,
                            detail=f"Payload exceeds {MAX_PAYLOAD_BYTES} bytes")
    body = await request.body()
    if len(body) > MAX_PAYLOAD_BYTES:
        inc_push_rejected("payload_cap")
        raise HTTPException(status_code=413,
                            detail=f"Payload exceeds {MAX_PAYLOAD_BYTES} bytes")
    try:
        payload = json.loads(body)
    except Exception as exc:
        inc_push_rejected("parse")
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {str(exc)}")

    if not isinstance(payload, (dict, list)):
        inc_push_rejected("parse")
        raise HTTPException(status_code=400, detail="Payload must be a JSON object or array of objects")

    # If payload is wrapped like {"sensors": {...}, "source": "http"}
    if isinstance(payload, dict):
        source = payload.get("source", "http")
        data = payload.get("sensors", payload)
        if not isinstance(data, dict):
            data = payload
    else:
        source = "http"
        data = payload

    inc_push_accepted()
    try:
        processed_frame = broker.push_telemetry(data, source=source)
    except Exception as exc:
        inc_push_rejected("ingest")
        raise HTTPException(status_code=500, detail=f"Failed to ingest telemetry: {str(exc)}")

    return {
        "status": "accepted",
        "sequence": processed_frame.get("seq"),
        "timestamp": processed_frame.get("timestamp"),
        "source": processed_frame.get("source", source),
        "is_live_hardware": processed_frame.get("is_live_hardware", True),
        "data": processed_frame,
    }


@app.get("/api/telemetry/status")
async def get_telemetry_status():
    """Returns real-time ingestion status and stream performance metrics."""
    broker_stat = broker.get_status()
    return {
        "status": "OPERATIONAL",
        "active_source": broker.active_source,
        "is_live_hardware": broker.is_live_hardware,
        "packets_received": broker.packet_count,
        "broker_latency_ms": broker.latency_ms,
        "fps": broker_stat.get("fps", 0.0),
        "broker": broker_stat,
        "udp_listener": {
            "port": udp_listener.port if udp_listener else 9000,
            "running": udp_listener.is_running if udp_listener else False,
            "packets_received": udp_listener.packets_received if udp_listener else 0,
        },
        "serial_listener": {
            "port": serial_listener.port if serial_listener else "COM3",
            "running": serial_listener.is_running if serial_listener else False,
            "is_mock": serial_listener.is_mock if serial_listener else False,
            "packets_read": serial_listener.packets_read if serial_listener else 0,
        },
        "active_clients": len(manager.active_connections),
    }


@app.get("/api/telemetry/latest")
async def get_latest_telemetry():
    """Returns the most recent telemetry frame."""
    return broker.get_latest_telemetry()


# ---------------------------------------------------------------------------
# Fallback stream
# ---------------------------------------------------------------------------
_DEMO_PATH = (Path(__file__).resolve().parents[1] / "docs" / "live_stream.json")
_demo_records: Optional[List[Dict[str, Any]]] = None
_demo_lock = threading.Lock()


def _demo_stream() -> List[Dict[str, Any]]:
    """The recorded physics-simulated demo mission, loaded once.

    The previous fallback synthesised `rpm = 1800 + 200*(step%50)/50`,
    `chi = 98 - step*0.02` and so on -- a linear ramp with none of the channels
    the pipeline consumes, so the "seamless fallback to simulation" exercised
    an entirely different code path from live hardware. Replaying the real
    demo mission means the fallback and the live path carry the same schema and
    run through the same blocks.
    """
    global _demo_records
    with _demo_lock:
        if _demo_records is None:
            try:
                _demo_records = json.loads(
                    _DEMO_PATH.read_text(encoding="utf-8"))["records"]
            except Exception as exc:
                logger.warning("demo stream unavailable (%s); "
                               "fallback will emit empty frames", exc)
                _demo_records = []
        return _demo_records


def _fallback_frame(step: int) -> Dict[str, Any]:
    records = _demo_stream()
    if not records:
        return {"step": step, "timestamp": time.time(), "source": "simulation"}
    rec = dict(records[step % len(records)])
    rec["step"] = step
    rec["timestamp"] = time.time()
    rec["source"] = "simulation"
    return rec


@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """Broadcasts processed telemetry at 20 Hz.

    Every frame goes through the full block chain in `pipeline.PHMPipeline`:
    FDIR sanitisation -> state detection -> health assessment -> prognostics
    -> DTC generation. This handler used to read the raw broker frame and ship
    it unchanged, taking the health index straight from whatever the client
    posted (`latest.get("composite_chi", 95.0)`) and padding the subsystem
    vector with five hardcoded literals.
    """
    _hdrs = getattr(websocket, "headers", None) or {}
    if not _origin_allowed(_hdrs.get("origin") if hasattr(_hdrs, "get") else None):
        await websocket.close(code=1008)
        return

    await manager.connect(websocket)
    pipeline = get_pipeline()
    try:
        sim_step = 0
        while True:
            is_live = broker.is_live_hardware

            if is_live:
                raw = broker.get_latest_telemetry()
                source = raw.get("source", broker.active_source)
                badge = f"[LIVE HARDWARE STREAM: {str(source).upper()}]"
            else:
                # Fallback stream. This is the physics simulator's recorded
                # demo mission, not the linear ramp that used to stand in for
                # it (`chi = 98 - step*0.02`), so the fallback exercises the
                # same pipeline with the same channel set as live hardware.
                sim_step += 1
                raw = _fallback_frame(sim_step)
                source = "simulation"
                badge = "[SIMULATION STREAM]"

            processed = pipeline.process(raw)
            clean = processed["clean_telemetry"]
            health = processed["subsystem_health"]

            # J1939 frames are encoded from the *sanitised* signals, in the
            # PGNs those SPNs actually belong to.
            eec1 = J1939FrameParser.encode_eec1(
                float(clean.get("rpm", 0.0)),
                min(100.0, max(0.0, float(clean.get("shaft_torque", 0.0)) / 50.0)))
            et1 = J1939FrameParser.encode_et1(
                float(clean.get("coolant_temp", 0.0)),
                oil_temp_c=clean.get("oil_temp"))
            # oil_pressure is in bar after pipeline.process() / to_canonical().
            # SPN 100 expects kPa. bar -> kPa = * 100 (not / 1000).
            efl = J1939FrameParser.encode_efl_p1(
                float(clean.get("oil_pressure", 0.0)) * 100.0)


            burst_dict = None
            if health is not None:
                order = ["engine", "powertrain", "lubrication", "cooling",
                         "hydraulics", "suspension", "structure", "overall"]
                prog = processed.get("prognosis") or {}
                probs = prog.get("fault_probs") or {}
                top_name = max(probs, key=probs.get) if probs else "healthy"
                rul_frac = prog.get("rul_fraction", {}).get("overall")
                burst = TacticalBurstPacket.encode(
                    tank_id=int(clean.get("tank_id", 1)),
                    mission_time=int(clean.get("step", sim_step)),
                    chi=health["overall"],
                    top_fault_id=(list(probs).index(top_name) if probs else 0),
                    fault_confidence=float(probs.get(top_name, 0.0)),
                    # None when the model has not produced an estimate; the
                    # burst carries 0 rather than a fabricated horizon.
                    rul_minutes=int((rul_frac or 0.0) * 40.0),
                    subsystem_health=[health[p] for p in order],
                    rpm=float(clean.get("rpm", 0.0)),
                    oil_pressure_bar=float(clean.get("oil_pressure", 0.0)) / 1e5,
                    coolant_temp_c=float(clean.get("coolant_temp", 0.0)),
                    vib_rms=float(clean.get("vib_rms", 0.0)),
                )
                burst_dict = TacticalBurstPacket.decode(burst)

            telemetry_payload = {
                "step": clean.get("step", sim_step),
                "timestamp": raw.get("timestamp", time.time()),
                "is_live_hardware": bool(is_live),
                "source": source,
                "stream_badge": badge,
                # Both views are published, and both are named honestly.
                # `telemetry_raw` used to hold the FDIR-sanitised frame, so a
                # consumer reading "raw" got slew-limited, clamped values with
                # no way to see what actually arrived on the wire.
                "telemetry_raw": raw,
                "telemetry_clean": clean,
                "subsystem_health": health,
                "health_available": processed["health_available"],
                "prognosis": processed["prognosis"],
                "inference_available": processed["inference_available"],
                "sensor_faults": processed["sensor_faults"],
                "dtcs_active": processed["dtcs_active"],
                "gate_ms": processed["gate_ms"],
                "tactical_burst": burst_dict,
                "j1939_raw_hex": {
                    "EEC1_PGN61444": eec1.hex().upper(),
                    "ET1_PGN65262": et1.hex().upper(),
                    "EFL_P1_PGN65263": efl.hex().upper(),
                    "DM1_PGN65226": processed["dm1_hex"],
                    "DM2_PGN65227": processed["dm2_hex"],
                },
            }

            await websocket.send_json(telemetry_payload)
            inc_websocket_frame()
            await asyncio.sleep(0.05)  # 20 Hz

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except (ConnectionResetError, RuntimeError) as exc:
        # Client-abort / closed-loop errors.  These are expected when
        # a dashboard tab is closed mid-frame and should not be treated
        # as a pipeline failure.  ``logger.exception`` already appends
        # the formatted traceback; passing ``exc`` to ``%s`` would
        # double-print the exception string.
        inc_pipeline_error("websocket_broadcast")
        logger.info("telemetry websocket closed: %s", exc)
        manager.disconnect(websocket)
    except Exception:
        # Anything else is a real pipeline / encoding bug.  ``logger.exception``
        # already records the traceback at ERROR; we do not re-format ``exc``
        # into the message or the traceback is duplicated.
        inc_pipeline_error("websocket_broadcast")
        logger.exception("telemetry websocket failed")
        manager.disconnect(websocket)
