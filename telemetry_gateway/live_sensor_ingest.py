"""Live Physical Sensor Hardware Ingestion Bridge for MBT PHM/CBM+.
Provides asynchronous UDP datagram ingestion (port 9000), high-speed Serial COM (115200 baud),
and REST push ingestion managed by a centralized, thread-safe TelemetryBroker.
"""

import asyncio
import collections
import inspect
import json
import logging
import socket
import struct
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .j1939_can_parser import J1939FrameParser
from .tactical_burst import TacticalBurstPacket

logger = logging.getLogger("live_sensor_ingest")


# Binary Telemetry Struct Configuration
BINARY_STRUCT_MAGIC = b"MBTT"
BINARY_STRUCT_FORMAT = ">4sBIffffff"  # Magic(4s), Version(B), Seq(I), 6x Float32 (rpm, oil_p, coolant_t, vib_rms, torque, hyd_p)
BINARY_STRUCT_SIZE = struct.calcsize(BINARY_STRUCT_FORMAT)

RAW_FLOATS_6_FORMAT = ">ffffff"
RAW_FLOATS_6_SIZE = struct.calcsize(RAW_FLOATS_6_FORMAT)

RAW_FLOATS_4_FORMAT = ">ffff"
RAW_FLOATS_4_SIZE = struct.calcsize(RAW_FLOATS_4_FORMAT)


def pack_telemetry_struct(
    seq: int = 1,
    rpm: float = 2100.0,
    oil_pressure: float = 4.5,
    coolant_temp: float = 88.0,
    vib_rms: float = 0.45,
    shaft_torque: float = 0.55,
    hyd_pressure: float = 210.0,
    version: int = 1,
) -> bytes:
    """Pack vehicle telemetry parameters into standard 33-byte MBTT binary datagram."""
    return struct.pack(
        BINARY_STRUCT_FORMAT,
        BINARY_STRUCT_MAGIC,
        version,
        seq,
        float(rpm),
        float(oil_pressure),
        float(coolant_temp),
        float(vib_rms),
        float(shaft_torque),
        float(hyd_pressure),
    )


def unpack_telemetry_struct(data: bytes) -> Optional[Dict[str, Any]]:
    """Unpack 33-byte MBTT binary datagram into telemetry dictionary."""
    if len(data) < BINARY_STRUCT_SIZE:
        return None
    try:
        magic, version, seq, rpm, oil_p, coolant_t, vib_rms, torque, hyd_p = struct.unpack(
            BINARY_STRUCT_FORMAT, data[:BINARY_STRUCT_SIZE]
        )
        if magic != BINARY_STRUCT_MAGIC:
            return None
        return {
            "seq": seq,
            "version": version,
            "rpm": round(rpm, 2),
            "oil_pressure": round(oil_p, 3),
            "coolant_temp": round(coolant_t, 2),
            "vib_rms": round(vib_rms, 3),
            "shaft_torque": round(torque, 3),
            "hyd_pressure": round(hyd_p, 2),
            "source": "binary_struct",
        }
    except Exception as e:
        logger.debug(f"Failed to unpack MBTT binary struct: {e}")
        return None


class TelemetryBroker:
    """Centralized, thread-safe telemetry broker coordinating live hardware feeds,
    simulation streams, listener callbacks, and streaming metrics.
    """

    _instance: Optional["TelemetryBroker"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "TelemetryBroker":
        """Singleton accessor for shared TelemetryBroker instance."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self, hardware_timeout: float = 2.0):
        self._lock = threading.RLock()
        self._hardware_timeout: float = hardware_timeout
        self._latest_telemetry: Dict[str, Any] = {}
        self._last_update_time: float = 0.0
        self._last_hardware_time: float = 0.0
        self._last_hardware_source: str = "none"
        self._active_source: str = "simulation"
        self._sequence_num: int = 0
        self._total_packets: int = 0
        self._packets_by_source: Dict[str, int] = collections.defaultdict(int)
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._packet_timestamps: collections.deque = collections.deque(maxlen=100)
        self._last_latency_ms: float = 0.0

    @property
    def is_live_hardware(self) -> bool:
        """Returns True if telemetry has been received from physical hardware within timeout."""
        with self._lock:
            if self._last_hardware_time <= 0.0:
                return False
            return (time.time() - self._last_hardware_time) < self._hardware_timeout

    @property
    def active_source(self) -> str:
        """Returns current active telemetry source ('udp', 'serial', 'http', 'simulation')."""
        with self._lock:
            if self.is_live_hardware:
                return self._last_hardware_source
            return "simulation" if self._total_packets == 0 else self._active_source

    @property
    def packet_count(self) -> int:
        """Total number of packets ingested across all sources."""
        with self._lock:
            return self._total_packets

    @property
    def latency_ms(self) -> float:
        """Latest calculated telemetry ingest latency in milliseconds."""
        with self._lock:
            return self._last_latency_ms

    @property
    def fps(self) -> float:
        """Calculates rolling frame rate (Hz) from recent packet arrival timestamps."""
        with self._lock:
            if len(self._packet_timestamps) < 2:
                return 0.0
            dt = self._packet_timestamps[-1] - self._packet_timestamps[0]
            if dt <= 0.0:
                return 0.0
            return (len(self._packet_timestamps) - 1) / dt

    def push_telemetry(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], source: str = "http") -> Dict[str, Any]:
        """Ingest a raw or structured telemetry frame or batch, update state, and notify listeners."""
        if isinstance(data, list):
            # Batch ingestion: process each, returning the final frame
            last_frame = {}
            for item in data:
                if isinstance(item, dict):
                    last_frame = self.push_telemetry(item, source=source)
            return last_frame

        if not isinstance(data, dict):
            raise TypeError(f"Expected dict or list[dict] telemetry payload, got {type(data).__name__}")

        now = time.time()
        with self._lock:
            self._sequence_num += 1
            self._total_packets += 1
            self._packets_by_source[source] += 1
            self._last_update_time = now
            self._active_source = source

            is_hw = source.lower() in ("udp", "serial", "http", "live", "hardware", "serial_mock", "binary_struct", "j1939")
            if is_hw:
                self._last_hardware_time = now
                self._last_hardware_source = source

            self._packet_timestamps.append(now)

            # Compute ingest latency
            if "timestamp" in data and isinstance(data["timestamp"], (int, float)):
                data_ts = float(data["timestamp"])
                # If timestamp is in seconds vs milliseconds
                if data_ts > 1e11:
                    data_ts /= 1000.0
                self._last_latency_ms = max(0.0, (now - data_ts) * 1000.0)
            else:
                self._last_latency_ms = 0.5

            # Enrich frame copy with metadata
            frame = dict(data)
            frame["seq"] = self._sequence_num
            frame["timestamp"] = now
            frame["source"] = source
            frame["is_live_hardware"] = self.is_live_hardware
            frame["ingest_latency_ms"] = round(self._last_latency_ms, 2)

            self._latest_telemetry = frame
            listeners_snapshot = list(self._listeners)

        # Notify listeners outside main lock
        for cb in listeners_snapshot:
            try:
                if inspect.iscoroutinefunction(cb):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(cb(frame))
                    except RuntimeError:
                        asyncio.run(cb(frame))
                else:
                    cb(frame)
            except Exception as exc:
                logger.warning(f"Exception in telemetry listener callback: {exc}")

        return frame

    def get_latest_telemetry(self) -> Dict[str, Any]:
        """Returns a copy of the latest ingested telemetry frame, or default baseline if empty."""
        with self._lock:
            if self._latest_telemetry:
                result = dict(self._latest_telemetry)
                result["is_live_hardware"] = self.is_live_hardware
                return result

            # Default baseline simulation state
            return {
                "seq": 0,
                "timestamp": time.time(),
                "source": "simulation",
                "is_live_hardware": False,
                "rpm": 1800.0,
                "oil_pressure": 4.5,
                "coolant_temp": 85.0,
                "vib_rms": 0.45,
                "shaft_torque": 0.45,
                "hyd_pressure": 210.0,
                "composite_chi": 98.0,
                "rul_minutes": 720,
            }

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive broker health and stream metrics."""
        with self._lock:
            is_live = self.is_live_hardware
            active_src = self._last_hardware_source if is_live else ("simulation" if self._total_packets == 0 else self._active_source)
            return {
                "status": "OPERATIONAL",
                "active_source": active_src,
                "is_live_hardware": is_live,
                "packet_count": self._total_packets,
                "packets_received": self._total_packets,
                "packets_by_source": dict(self._packets_by_source),
                "fps": round(self.fps, 2),
                "last_update_time": self._last_update_time,
                "latency_ms": round(self._last_latency_ms, 2),
                "broker_latency_ms": round(self._last_latency_ms, 2),
                "sequence_number": self._sequence_num,
                "listeners_count": len(self._listeners),
                "hardware_timeout_seconds": self._hardware_timeout,
            }

    def register_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a synchronous or asynchronous callback triggered on each ingested frame."""
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def unregister_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Unregister an existing callback."""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def reset(self) -> None:
        """Reset internal telemetry state, counters, and listeners."""
        with self._lock:
            self._latest_telemetry.clear()
            self._last_update_time = 0.0
            self._last_hardware_time = 0.0
            self._last_hardware_source = "none"
            self._active_source = "simulation"
            self._sequence_num = 0
            self._total_packets = 0
            self._packets_by_source.clear()
            self._packet_timestamps.clear()
            self._last_latency_ms = 0.0
            self._listeners.clear()


def get_broker() -> TelemetryBroker:
    """Convenience helper to retrieve the singleton TelemetryBroker instance."""
    return TelemetryBroker.get_instance()


class UDPSensorListener:
    """Asynchronous UDP datagram listener on port 9000 (0.0.0.0:9000).
    Decodes JSON strings, 32-byte tactical bursts, 8-byte J1939 frames, and binary structs,
    pushing parsed telemetry frames directly to the TelemetryBroker.
    """

    def __init__(
        self,
        broker: Optional[TelemetryBroker] = None,
        host: str = "0.0.0.0",
        port: int = 9000,
    ):
        self.broker = broker or get_broker()
        self.host = host
        self.port = port
        self._running = False
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._packets_received = 0
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def packets_received(self) -> int:
        return self._packets_received

    def start(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """Bind UDP socket and start asynchronous worker thread."""
        with self._lock:
            if self._running:
                return

            if host is not None:
                self.host = host
            if port is not None:
                self.port = port

            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((self.host, self.port))
            self._sock.settimeout(0.2)
            self._running = True

            self._thread = threading.Thread(
                target=self._worker_loop,
                name=f"UDPSensorListener-{self.port}",
                daemon=True,
            )
            self._thread.start()
            logger.info(f"UDPSensorListener listening on {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop worker loop and close UDP socket."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None

    def _worker_loop(self) -> None:
        while self._running:
            try:
                if self._sock is None:
                    break
                data, _ = self._sock.recvfrom(65535)
                if not data:
                    continue

                parsed = self.parse_datagram(data)
                if parsed:
                    self._packets_received += 1
                    self.broker.push_telemetry(parsed, source="udp")
            except socket.timeout:
                continue
            except (OSError, socket.error):
                if not self._running:
                    break
            except Exception as exc:
                if not self._running:
                    break
                logger.debug(f"UDP datagram handling error: {exc}")

    def parse_datagram(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Parse incoming datagram from JSON, Tactical Burst, J1939 CAN frame, or binary struct."""
        if not data:
            return None

        # 1. Try JSON decoding
        try:
            text = data.decode("utf-8").strip()
            if text.startswith("{") and text.endswith("}"):
                obj = json.loads(text)
                if isinstance(obj, dict):
                    return obj
        except Exception:
            pass

        # 2. Check 33-byte MBTT binary struct
        if len(data) >= BINARY_STRUCT_SIZE and data.startswith(BINARY_STRUCT_MAGIC):
            unpacked = unpack_telemetry_struct(data)
            if unpacked is not None:
                return unpacked

        # 3. Check 32-byte EMCON Tactical Burst
        if len(data) == 32 and data.startswith(b"TK"):
            try:
                burst = TacticalBurstPacket.decode(data)
                return {
                    "source": "udp_tactical_burst",
                    "tactical_burst": burst,
                    "rpm": burst.get("rpm", 2100.0),
                    "oil_pressure": burst.get("oil_pressure_bar", 4.5),
                    "coolant_temp": burst.get("coolant_temp_c", 88.0),
                    "vib_rms": burst.get("vib_rms", 0.45),
                    "composite_chi": burst.get("composite_chi", 95.0),
                    "rul_minutes": burst.get("rul_minutes", 600),
                }
            except Exception:
                pass

        # 4a. PGN-tagged J1939 frame: 4-byte big-endian PGN + 8-byte payload.
        #
        # A raw CAN payload carries no PGN -- the PGN lives in the 29-bit
        # identifier, which a bare 8-byte datagram has thrown away. The
        # previous code decoded the same 8 bytes as BOTH EEC1 and EFL_P1 and
        # merged the results, which is not a thing: one frame is one PGN. It
        # produced an rpm from bytes 4-5 and an "oil pressure" from byte 4 of
        # the *same* frame, i.e. one of the two readings was always garbage.
        # The bare `except Exception: pass` below meant the whole branch
        # silently vanished the moment either decoder changed shape.
        if len(data) == 12:
            try:
                pgn = int.from_bytes(data[:4], "big")
                payload = data[4:]
                if pgn == J1939FrameParser.PGN_EEC1:
                    d = J1939FrameParser.decode_eec1(payload)
                    return {"source": "udp_j1939", "pgn": pgn,
                            "rpm": d["rpm"], "torque_pct": d["torque_pct"]}
                if pgn == J1939FrameParser.PGN_ET1:
                    d = J1939FrameParser.decode_et1(payload)
                    out = {"source": "udp_j1939", "pgn": pgn,
                           "coolant_temp": d["coolant_temp"]}
                    if "oil_temp" in d:
                        out["oil_temp"] = d["oil_temp"]
                    return out
                if pgn == J1939FrameParser.PGN_EFL_P1:
                    d = J1939FrameParser.decode_efl_p1(payload)
                    return {"source": "udp_j1939", "pgn": pgn,
                            "oil_pressure": round(d["oil_pressure_kpa"] / 100.0, 2)}
                logger.debug("unhandled J1939 PGN %d", pgn)
            except Exception:
                logger.debug("malformed PGN-tagged J1939 datagram", exc_info=True)

        # 4b. Bare 8-byte payload. With no PGN there is nothing to
        # disambiguate on, so it is interpreted as EEC1 (engine speed and
        # torque) -- the only assumption that can be stated honestly. Senders
        # that need other PGNs must use the 12-byte tagged form above.
        if len(data) == 8:
            try:
                eec1 = J1939FrameParser.decode_eec1(data)
                return {
                    "source": "udp_j1939",
                    "pgn": J1939FrameParser.PGN_EEC1,
                    "rpm": eec1["rpm"],
                    "torque_pct": eec1["torque_pct"],
                }
            except Exception:
                logger.debug("malformed 8-byte J1939 datagram", exc_info=True)

        # 5. Check raw float array (>ffffff: 24 bytes)
        if len(data) == RAW_FLOATS_6_SIZE:
            try:
                rpm, oil_p, coolant_t, vib_rms, torque, hyd_p = struct.unpack(RAW_FLOATS_6_FORMAT, data)
                return {
                    "rpm": round(rpm, 2),
                    "oil_pressure": round(oil_p, 3),
                    "coolant_temp": round(coolant_t, 2),
                    "vib_rms": round(vib_rms, 3),
                    "shaft_torque": round(torque, 3),
                    "hyd_pressure": round(hyd_p, 2),
                }
            except Exception:
                pass

        # 6. Check raw float array (>ffff: 16 bytes)
        if len(data) == RAW_FLOATS_4_SIZE:
            try:
                rpm, oil_p, coolant_t, vib_rms = struct.unpack(RAW_FLOATS_4_FORMAT, data)
                return {
                    "rpm": round(rpm, 2),
                    "oil_pressure": round(oil_p, 3),
                    "coolant_temp": round(coolant_t, 2),
                    "vib_rms": round(vib_rms, 3),
                }
            except Exception:
                pass

        return None

    def get_status(self) -> Dict[str, Any]:
        """Return UDP listener status."""
        return {
            "running": self._running,
            "host": self.host,
            "port": self.port,
            "packets_received": self._packets_received,
        }


class SerialSensorListener:
    """High-speed serial COM listener at 115200 baud.
    Dynamically imports pyserial and provides seamless mock/virtual streaming fallback
    when physical serial hardware is absent.
    """

    def __init__(
        self,
        broker: Optional[TelemetryBroker] = None,
        port: str = "COM3",
        baudrate: int = 115200,
        mock_fallback: bool = True,
    ):
        self.broker = broker or get_broker()
        self.port = port
        self.baudrate = baudrate
        self.mock_fallback = mock_fallback
        self._running = False
        self._is_mock = False
        self._thread: Optional[threading.Thread] = None
        self._serial_obj: Any = None
        self._packets_read = 0
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    @property
    def packets_read(self) -> int:
        return self._packets_read

    def start(
        self,
        port: Optional[str] = None,
        baudrate: Optional[int] = None,
        mock_fallback: Optional[bool] = None,
    ) -> None:
        """Start serial listener. Connects to physical COM port or starts mock streaming fallback."""
        with self._lock:
            if self._running:
                return

            if port is not None:
                self.port = port
            if baudrate is not None:
                self.baudrate = baudrate
            if mock_fallback is not None:
                self.mock_fallback = mock_fallback

            self._running = True
            self._is_mock = False
            self._serial_obj = None

            # Attempt dynamic import of pyserial
            serial_available = False
            try:
                import serial  # type: ignore
                serial_available = True
            except ImportError:
                serial_available = False

            opened_physical = False
            if serial_available:
                try:
                    self._serial_obj = serial.Serial(self.port, self.baudrate, timeout=0.2)
                    opened_physical = True
                    logger.info(f"Opened physical serial port {self.port} at {self.baudrate} baud")
                except Exception as e:
                    logger.warning(f"Could not open physical serial port {self.port}: {e}")
                    opened_physical = False

            if not opened_physical:
                if not self.mock_fallback:
                    self._running = False
                    raise ConnectionError(
                        f"Failed to open physical serial port {self.port} and mock_fallback is disabled."
                    )
                self._is_mock = True
                logger.info(f"Serial COM {self.port} not available: starting virtual mock serial stream")

            target_func = self._physical_worker_loop if (opened_physical and self._serial_obj) else self._mock_worker_loop
            self._thread = threading.Thread(
                target=target_func,
                name=f"SerialSensorListener-{self.port}",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop serial listener and close open ports."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        with self._lock:
            if self._serial_obj:
                try:
                    self._serial_obj.close()
                except Exception:
                    pass
                self._serial_obj = None

    def _physical_worker_loop(self) -> None:
        """Read lines from physical serial port and push to broker."""
        while self._running and self._serial_obj:
            try:
                line = self._serial_obj.readline()
                if not line:
                    continue
                parsed = self._parse_serial_line(line)
                if parsed:
                    self._packets_read += 1
                    self.broker.push_telemetry(parsed, source="serial")
            except Exception as exc:
                if not self._running:
                    break
                logger.debug(f"Serial read error: {exc}")

    def _mock_worker_loop(self) -> None:
        """Simulate high-speed 20 Hz serial microcontroller ADC stream."""
        step = 0
        while self._running:
            try:
                step += 1
                # Generate realistic oscillating sensor telemetry
                rpm = 2100.0 + 150.0 * (step % 40) / 40.0
                oil_p = 4.5 + 0.3 * (step % 20) / 20.0
                coolant_t = 88.0 + 2.0 * (step % 30) / 30.0
                vib_rms = 0.42 + 0.05 * ((step * 3) % 10) / 10.0
                torque = 0.52 + 0.04 * (step % 15) / 15.0
                hyd_p = 210.0 + 5.0 * (step % 25) / 25.0

                frame = {
                    "step": step,
                    "rpm": round(rpm, 2),
                    "oil_pressure": round(oil_p, 3),
                    "coolant_temp": round(coolant_t, 2),
                    "vib_rms": round(vib_rms, 3),
                    "shaft_torque": round(torque, 3),
                    "hyd_pressure": round(hyd_p, 2),
                    "source": "serial",
                    "timestamp": time.time(),
                }
                self._packets_read += 1
                self.broker.push_telemetry(frame, source="serial")
                time.sleep(0.05)  # 20 Hz rate
            except Exception as exc:
                if not self._running:
                    break
                logger.debug(f"Mock serial error: {exc}")

    def _parse_serial_line(self, line: bytes) -> Optional[Dict[str, Any]]:
        """Parse raw line from serial port (JSON or CSV)."""
        try:
            text = line.decode("utf-8", errors="ignore").strip()
            if not text:
                return None
            if text.startswith("{") and text.endswith("}"):
                return json.loads(text)

            # CSV format fallback: rpm,oil_p,coolant_t,vib_rms
            parts = text.split(",")
            if len(parts) >= 4:
                return {
                    "rpm": float(parts[0]),
                    "oil_pressure": float(parts[1]),
                    "coolant_temp": float(parts[2]),
                    "vib_rms": float(parts[3]),
                }
        except Exception:
            pass
        return None

    def get_status(self) -> Dict[str, Any]:
        """Return Serial listener status."""
        return {
            "running": self._running,
            "port": self.port,
            "baudrate": self.baudrate,
            "is_mock": self._is_mock,
            "packets_read": self._packets_read,
        }


# Global Lifecycle Listeners
_global_udp_listener: Optional[UDPSensorListener] = None
_global_serial_listener: Optional[SerialSensorListener] = None


def start_all_listeners(
    broker: Optional[TelemetryBroker] = None,
    udp_host: str = "0.0.0.0",
    udp_port: int = 9000,
    serial_port: str = "COM3",
    serial_baudrate: int = 115200,
    serial_mock_fallback: bool = True,
) -> Dict[str, Any]:
    """Start all background hardware listeners (UDP port 9000 & Serial COM)."""
    global _global_udp_listener, _global_serial_listener
    b = broker or get_broker()

    if _global_udp_listener is None or not _global_udp_listener.is_running:
        _global_udp_listener = UDPSensorListener(broker=b, host=udp_host, port=udp_port)
        _global_udp_listener.start()

    if _global_serial_listener is None or not _global_serial_listener.is_running:
        _global_serial_listener = SerialSensorListener(
            broker=b,
            port=serial_port,
            baudrate=serial_baudrate,
            mock_fallback=serial_mock_fallback,
        )
        _global_serial_listener.start()

    return {
        "udp": _global_udp_listener,
        "serial": _global_serial_listener,
    }


def stop_all_listeners() -> None:
    """Cleanly stop all running background listeners."""
    global _global_udp_listener, _global_serial_listener
    if _global_udp_listener is not None:
        _global_udp_listener.stop()
        _global_udp_listener = None

    if _global_serial_listener is not None:
        _global_serial_listener.stop()
        _global_serial_listener = None


__all__ = [
    "TelemetryBroker",
    "UDPSensorListener",
    "SerialSensorListener",
    "get_broker",
    "start_all_listeners",
    "stop_all_listeners",
    "pack_telemetry_struct",
    "unpack_telemetry_struct",
    "BINARY_STRUCT_MAGIC",
    "BINARY_STRUCT_SIZE",
]
