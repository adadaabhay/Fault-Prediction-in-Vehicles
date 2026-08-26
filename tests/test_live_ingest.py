"""Unit and Integration Test Suite for Milestone M1: Live Physical Sensor Hardware Ingestion Bridge.
Tests UDP datagram ingestion, Serial COM listener with mock fallback, TelemetryBroker concurrency,
REST push endpoints, and real-time WebSocket streaming.
"""

import asyncio
import json
import socket
import struct
import threading
import time
import unittest
from typing import Any, Dict, List, Optional

import requests
import uvicorn
from fastapi import HTTPException
from starlette.websockets import WebSocketDisconnect

from telemetry_gateway.j1939_can_parser import J1939FrameParser
from telemetry_gateway.tactical_burst import TacticalBurstPacket
from telemetry_gateway.live_sensor_ingest import (
    TelemetryBroker,
    UDPSensorListener,
    SerialSensorListener,
    get_broker,
    pack_telemetry_struct,
    unpack_telemetry_struct,
    start_all_listeners,
    stop_all_listeners,
    BINARY_STRUCT_MAGIC,
    BINARY_STRUCT_SIZE,
    RAW_FLOATS_6_FORMAT,
    RAW_FLOATS_4_FORMAT,
)
from telemetry_gateway.server import (
    app,
    broker as server_broker,
    manager,
    get_root,
    push_telemetry_endpoint,
    get_telemetry_status,
    get_latest_telemetry,
    websocket_telemetry_endpoint,
)


class TestTelemetryBroker(unittest.TestCase):
    """Unit tests for TelemetryBroker concurrency, listener callbacks, and status metrics."""

    def setUp(self):
        self.broker = TelemetryBroker(hardware_timeout=0.5)
        self.broker.reset()

    def tearDown(self):
        self.broker.reset()

    def test_push_and_get_latest_telemetry(self):
        """Verify basic pushing and retrieving of telemetry frames."""
        frame_in = {"rpm": 2350.0, "oil_pressure": 4.8, "coolant_temp": 89.0}
        pushed = self.broker.push_telemetry(frame_in, source="http")

        self.assertEqual(pushed["rpm"], 2350.0)
        self.assertEqual(pushed["source"], "http")
        self.assertEqual(pushed["seq"], 1)
        self.assertTrue(pushed["is_live_hardware"])

        latest = self.broker.get_latest_telemetry()
        self.assertEqual(latest["rpm"], 2350.0)
        self.assertEqual(latest["seq"], 1)
        self.assertEqual(self.broker.packet_count, 1)

    def test_batch_push_telemetry(self):
        """Verify batch telemetry list ingestion."""
        batch = [
            {"rpm": 2100.0, "oil_pressure": 4.2},
            {"rpm": 2200.0, "oil_pressure": 4.4},
            {"rpm": 2300.0, "oil_pressure": 4.6},
        ]
        last = self.broker.push_telemetry(batch, source="http")
        self.assertEqual(last["rpm"], 2300.0)
        self.assertEqual(self.broker.packet_count, 3)
        self.assertEqual(self.broker.get_status()["packets_by_source"]["http"], 3)

    def test_hardware_timeout_and_fallback(self):
        """Verify is_live_hardware transitions to False after timeout."""
        self.broker.push_telemetry({"rpm": 2000.0}, source="udp")
        self.assertTrue(self.broker.is_live_hardware)
        self.assertEqual(self.broker.active_source, "udp")

        # Wait for timeout (hardware_timeout = 0.5s)
        time.sleep(0.6)
        self.assertFalse(self.broker.is_live_hardware)
        self.assertEqual(self.broker.active_source, "udp")

    def test_listener_registration_and_callback(self):
        """Verify synchronous listener registration and notifications."""
        received_frames: List[Dict[str, Any]] = []

        def callback(frame: Dict[str, Any]):
            received_frames.append(frame)

        self.broker.register_listener(callback)
        self.broker.push_telemetry({"rpm": 1950.0}, source="serial")
        self.broker.push_telemetry({"rpm": 2050.0}, source="udp")

        self.assertEqual(len(received_frames), 2)
        self.assertEqual(received_frames[0]["rpm"], 1950.0)
        self.assertEqual(received_frames[1]["rpm"], 2050.0)

        # Test unregister
        self.broker.unregister_listener(callback)
        self.broker.push_telemetry({"rpm": 2150.0}, source="http")
        self.assertEqual(len(received_frames), 2)

    def test_faulty_listener_does_not_break_broker(self):
        """Verify exceptions in listeners are handled without failing push_telemetry."""
        def faulty_listener(frame):
            raise ValueError("Intentional error in listener")

        received = []
        def good_listener(frame):
            received.append(frame)

        self.broker.register_listener(faulty_listener)
        self.broker.register_listener(good_listener)

        frame = self.broker.push_telemetry({"rpm": 2200.0}, source="udp")
        self.assertEqual(len(received), 1)
        self.assertEqual(frame["rpm"], 2200.0)

    def test_concurrent_multithreaded_pushes(self):
        """Verify thread-safe ingestion under high concurrent loads."""
        num_threads = 10
        pushes_per_thread = 50

        def worker(thread_idx: int):
            for i in range(pushes_per_thread):
                self.broker.push_telemetry(
                    {"thread_idx": thread_idx, "iter": i, "rpm": 2000.0 + i},
                    source="udp",
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(self.broker.packet_count, num_threads * pushes_per_thread)
        self.assertEqual(self.broker.get_status()["sequence_number"], num_threads * pushes_per_thread)

    def test_fps_calculation(self):
        """Verify FPS calculation from arrival timestamps."""
        for _ in range(20):
            self.broker.push_telemetry({"rpm": 2000.0}, source="udp")
            time.sleep(0.01)

        fps = self.broker.fps
        self.assertGreater(fps, 0.0)

    def test_status_reporting(self):
        """Verify status metrics dictionary structure."""
        self.broker.push_telemetry({"rpm": 2100.0, "timestamp": time.time() - 0.005}, source="udp")
        stat = self.broker.get_status()

        self.assertEqual(stat["status"], "OPERATIONAL")
        self.assertEqual(stat["active_source"], "udp")
        self.assertTrue(stat["is_live_hardware"])
        self.assertEqual(stat["packet_count"], 1)
        self.assertIn("fps", stat)
        self.assertIn("broker_latency_ms", stat)

    def test_singleton_get_broker(self):
        """Verify singleton get_broker() returns the same instance."""
        b1 = get_broker()
        b2 = get_broker()
        self.assertIs(b1, b2)

    def test_invalid_payload_type_raises(self):
        """Verify non-dict/list payload raises TypeError."""
        with self.assertRaises(TypeError):
            self.broker.push_telemetry("invalid_string_payload")  # type: ignore


class TestUDPSensorListener(unittest.TestCase):
    """Unit and integration tests for UDP Datagram Listener on port 9000."""

    @classmethod
    def setUpClass(cls):
        cls.test_port = 9055
        cls.broker = TelemetryBroker(hardware_timeout=2.0)
        cls.listener = UDPSensorListener(broker=cls.broker, host="127.0.0.1", port=cls.test_port)
        cls.listener.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.listener.stop()

    def setUp(self):
        self.broker.reset()
        self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def tearDown(self):
        self.client_sock.close()

    def test_udp_json_datagram_reception(self):
        """Verify UDP listener receives, parses, and pushes JSON datagrams to broker."""
        payload = {
            "rpm": 2420.0,
            "oil_pressure": 4.75,
            "coolant_temp": 91.5,
            "vib_rms": 0.52,
        }
        data = json.dumps(payload).encode("utf-8")
        self.client_sock.sendto(data, ("127.0.0.1", self.test_port))

        time.sleep(0.1)  # Allow packet arrival
        latest = self.broker.get_latest_telemetry()
        self.assertEqual(latest.get("rpm"), 2420.0)
        self.assertEqual(latest.get("oil_pressure"), 4.75)
        self.assertEqual(latest.get("source"), "udp")
        self.assertTrue(self.broker.is_live_hardware)

    def test_udp_binary_struct_reception(self):
        """Verify UDP listener decodes 33-byte MBTT binary structs."""
        binary_data = pack_telemetry_struct(
            seq=101,
            rpm=2500.0,
            oil_pressure=4.9,
            coolant_temp=93.0,
            vib_rms=0.58,
            shaft_torque=0.62,
            hyd_pressure=215.0,
        )
        self.client_sock.sendto(binary_data, ("127.0.0.1", self.test_port))

        time.sleep(0.1)
        latest = self.broker.get_latest_telemetry()
        self.assertEqual(latest.get("rpm"), 2500.0)
        self.assertEqual(latest.get("oil_pressure"), 4.9)
        self.assertEqual(latest.get("hyd_pressure"), 215.0)
        self.assertEqual(latest.get("source"), "udp")

    def test_udp_tactical_burst_reception(self):
        """Verify UDP listener decodes 32-byte EMCON Tactical Burst packets."""
        burst_bytes = TacticalBurstPacket.encode(
            tank_id=2,
            mission_time=450,
            chi=93.0,
            top_fault_id=0,
            fault_confidence=0.91,
            rul_minutes=480,
            subsystem_health=[93] * 8,
            rpm=2250.0,
            oil_pressure_bar=4.65,
            coolant_temp_c=87.0,
            vib_rms=0.48,
        )
        self.client_sock.sendto(burst_bytes, ("127.0.0.1", self.test_port))

        time.sleep(0.1)
        latest = self.broker.get_latest_telemetry()
        self.assertAlmostEqual(latest.get("rpm"), 2250.0, delta=20.0)
        self.assertAlmostEqual(latest.get("oil_pressure"), 4.65, places=2)
        self.assertEqual(latest.get("composite_chi"), 93.0)


    def test_udp_j1939_can_frame_reception(self):
        """Verify UDP listener decodes 8-byte SAE J1939 CAN datagrams."""
        can_bytes = J1939FrameParser.encode_eec1(rpm=2180.0, torque_pct=80.0)
        self.client_sock.sendto(can_bytes, ("127.0.0.1", self.test_port))

        time.sleep(0.1)
        latest = self.broker.get_latest_telemetry()
        self.assertAlmostEqual(latest.get("rpm", 0.0), 2180.0, delta=1.0)

    def test_udp_raw_6_floats_reception(self):
        """Verify UDP listener decodes 24-byte raw float array."""
        data = struct.pack(RAW_FLOATS_6_FORMAT, 2320.0, 4.6, 88.0, 0.44, 0.55, 210.0)
        self.client_sock.sendto(data, ("127.0.0.1", self.test_port))

        time.sleep(0.1)
        latest = self.broker.get_latest_telemetry()
        self.assertAlmostEqual(latest.get("rpm", 0.0), 2320.0, places=1)
        self.assertAlmostEqual(latest.get("oil_pressure", 0.0), 4.6, places=2)

    def test_udp_raw_4_floats_reception(self):
        """Verify UDP listener decodes 16-byte raw float array."""
        data = struct.pack(RAW_FLOATS_4_FORMAT, 2400.0, 4.8, 90.0, 0.50)
        self.client_sock.sendto(data, ("127.0.0.1", self.test_port))

        time.sleep(0.1)
        latest = self.broker.get_latest_telemetry()
        self.assertAlmostEqual(latest.get("rpm", 0.0), 2400.0, places=1)

    def test_udp_corrupted_datagram_ignored_safely(self):
        """Verify corrupted/random binary data is discarded without crashing."""
        corrupted = b"\xDE\xAD\xBE\xEF\x00\x11\x22\x33\x44\x55\x66\x77"
        self.client_sock.sendto(corrupted, ("127.0.0.1", self.test_port))
        time.sleep(0.05)
        # Broker remains unaffected and listener is still running
        self.assertTrue(self.listener.is_running)

    def test_udp_listener_status_and_lifecycle(self):
        """Verify get_status and start/stop lifecycle."""
        stat = self.listener.get_status()
        self.assertTrue(stat["running"])
        self.assertEqual(stat["port"], self.test_port)
        self.assertIn("packets_received", stat)


class TestSerialSensorListener(unittest.TestCase):
    """Unit tests for Serial COM listener and mock streaming fallback."""

    def setUp(self):
        self.broker = TelemetryBroker(hardware_timeout=2.0)
        self.broker.reset()

    def test_serial_mock_fallback_mode(self):
        """Verify Serial listener starts mock streaming when COM port is unavailable."""
        listener = SerialSensorListener(
            broker=self.broker,
            port="COM99",
            baudrate=115200,
            mock_fallback=True,
        )
        listener.start()

        self.assertTrue(listener.is_running)
        self.assertTrue(listener.is_mock)

        # Allow mock loop to generate a few frames
        time.sleep(0.15)
        listener.stop()

        self.assertFalse(listener.is_running)
        self.assertGreater(self.broker.packet_count, 0)
        latest = self.broker.get_latest_telemetry()
        self.assertIn("rpm", latest)
        self.assertEqual(latest.get("source"), "serial")

    def test_serial_disabled_mock_raises_error_when_unavailable(self):
        """Verify error is raised if mock_fallback is False and physical port is absent."""
        listener = SerialSensorListener(
            broker=self.broker,
            port="COM999",
            baudrate=115200,
            mock_fallback=False,
        )
        with self.assertRaises(Exception):
            listener.start()
        self.assertFalse(listener.is_running)

    def test_parse_serial_line_formats(self):
        """Verify line parser decodes JSON and CSV formatted lines."""
        listener = SerialSensorListener(broker=self.broker)

        # Test JSON line
        json_line = b'{"rpm": 2120.0, "oil_pressure": 4.3, "coolant_temp": 87.0}\n'
        parsed_json = listener._parse_serial_line(json_line)
        self.assertIsNotNone(parsed_json)
        self.assertEqual(parsed_json["rpm"], 2120.0)

        # Test CSV line
        csv_line = b"2200.0,4.5,88.5,0.46\n"
        parsed_csv = listener._parse_serial_line(csv_line)
        self.assertIsNotNone(parsed_csv)
        self.assertEqual(parsed_csv["rpm"], 2200.0)
        self.assertEqual(parsed_csv["oil_pressure"], 4.5)

    def test_serial_lifecycle_and_status(self):
        """Verify status reporting for serial listener."""
        listener = SerialSensorListener(broker=self.broker, port="COM12", baudrate=115200)
        stat = listener.get_status()
        self.assertFalse(stat["running"])
        self.assertEqual(stat["port"], "COM12")
        self.assertEqual(stat["baudrate"], 115200)


class TestServerLiveEndpoints(unittest.TestCase):
    """Integration tests for FastAPI REST push endpoints via live HTTP test server."""

    @classmethod
    def setUpClass(cls):
        cls.port = 8993
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.config = uvicorn.Config(app, host="127.0.0.1", port=cls.port, log_level="error")
        cls.server = uvicorn.Server(cls.config)
        cls.server_thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.server_thread.join(timeout=1.0)

    def setUp(self):
        server_broker.reset()

    def tearDown(self):
        server_broker.reset()

    def test_live_get_root(self):
        """Verify GET / returns OPERATIONAL status and supported protocols."""
        resp = requests.get(f"{self.base_url}/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "OPERATIONAL")
        self.assertIn("UDP 9000", data["supported_protocols"])
        self.assertIn("Serial 115200", data["supported_protocols"])

    def test_live_post_telemetry_push_single(self):
        """Verify POST /api/telemetry/push ingests single sensor record."""
        payload = {
            "rpm": 2320.0,
            "oil_pressure": 4.65,
            "coolant_temp": 90.0,
            "vib_rms": 0.50,
            "source": "http",
        }
        resp = requests.post(f"{self.base_url}/api/telemetry/push", json=payload)
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertEqual(res["status"], "accepted")
        self.assertEqual(res["data"]["rpm"], 2320.0)
        self.assertEqual(res["data"]["source"], "http")
        self.assertTrue(res["is_live_hardware"])

    def test_live_post_telemetry_push_wrapped(self):
        """Verify POST /api/telemetry/push ingests wrapped sensors payload."""
        payload = {
            "source": "http_adc",
            "sensors": {
                "rpm": 2480.0,
                "oil_pressure": 4.85,
                "coolant_temp": 92.5,
            }
        }
        resp = requests.post(f"{self.base_url}/api/telemetry/push", json=payload)
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertEqual(res["data"]["rpm"], 2480.0)
        self.assertEqual(res["data"]["source"], "http_adc")

    def test_live_post_telemetry_push_batch(self):
        """Verify POST /api/telemetry/push ingests batch list payload."""
        batch = [
            {"rpm": 2100.0, "oil_pressure": 4.2},
            {"rpm": 2200.0, "oil_pressure": 4.4},
        ]
        resp = requests.post(f"{self.base_url}/api/telemetry/push", json=batch)
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertEqual(res["data"]["rpm"], 2200.0)

    def test_live_post_telemetry_invalid_json(self):
        """Verify POST /api/telemetry/push returns 400 for invalid JSON payload."""
        resp = requests.post(
            f"{self.base_url}/api/telemetry/push",
            data="Not valid json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_live_get_telemetry_status(self):
        """Verify GET /api/telemetry/status returns complete status metrics."""
        requests.post(f"{self.base_url}/api/telemetry/push", json={"rpm": 2150.0})

        resp = requests.get(f"{self.base_url}/api/telemetry/status")
        self.assertEqual(resp.status_code, 200)
        stat = resp.json()
        self.assertEqual(stat["status"], "OPERATIONAL")
        self.assertTrue(stat["is_live_hardware"])
        self.assertEqual(stat["packets_received"], 1)
        self.assertIn("udp_listener", stat)
        self.assertIn("serial_listener", stat)
        self.assertIn("broker_latency_ms", stat)

    def test_live_get_telemetry_latest(self):
        """Verify GET /api/telemetry/latest returns the most recent frame."""
        requests.post(f"{self.base_url}/api/telemetry/push", json={"rpm": 2600.0, "oil_pressure": 5.2})
        resp = requests.get(f"{self.base_url}/api/telemetry/latest")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["rpm"], 2600.0)


class MockWebSocket:
    """Mock WebSocket for deterministic asynchronous testing of websocket_telemetry_endpoint."""

    def __init__(self, max_messages: int = 2):
        self.accepted = False
        self.sent_messages: List[Dict[str, Any]] = []
        self.max_messages = max_messages

    async def accept(self):
        self.accepted = True

    async def send_json(self, data: Dict[str, Any]):
        self.sent_messages.append(data)
        if len(self.sent_messages) >= self.max_messages:
            raise WebSocketDisconnect()


class TestServerWebSocketStreaming(unittest.TestCase):
    """Direct asynchronous unit tests for WebSocket streaming handler."""

    def setUp(self):
        server_broker.reset()

    def tearDown(self):
        server_broker.reset()

    def test_websocket_simulation_stream(self):
        """Verify WebSocket endpoint produces simulated stream when no hardware telemetry exists."""
        ws = MockWebSocket(max_messages=1)
        asyncio.run(websocket_telemetry_endpoint(ws))  # type: ignore

        self.assertTrue(ws.accepted)
        self.assertEqual(len(ws.sent_messages), 1)

        msg = ws.sent_messages[0]
        self.assertIn("step", msg)
        self.assertIn("timestamp", msg)
        self.assertFalse(msg["is_live_hardware"])
        self.assertEqual(msg["source"], "simulation")
        self.assertEqual(msg["stream_badge"], "[SIMULATION STREAM]")
        self.assertIn("tactical_burst", msg)
        self.assertIn("j1939_raw_hex", msg)
        self.assertIn("EEC1_PGN61444", msg["j1939_raw_hex"])
        self.assertIn("EFL_P1_PGN65263", msg["j1939_raw_hex"])

    def test_websocket_live_hardware_stream(self):
        """Verify WebSocket endpoint streams live hardware frame when hardware telemetry is active."""
        server_broker.push_telemetry(
            {"rpm": 2750.0, "oil_pressure": 5.2, "coolant_temp": 94.0, "vib_rms": 0.62},
            source="udp",
        )

        ws = MockWebSocket(max_messages=1)
        asyncio.run(websocket_telemetry_endpoint(ws))  # type: ignore

        self.assertTrue(ws.accepted)
        self.assertEqual(len(ws.sent_messages), 1)

        msg = ws.sent_messages[0]
        self.assertTrue(msg["is_live_hardware"])
        self.assertEqual(msg["source"], "udp")
        self.assertIn("LIVE HARDWARE STREAM", msg["stream_badge"])
        self.assertEqual(msg["telemetry_raw"]["rpm"], 2750.0)
        self.assertIn("tactical_burst", msg)
        self.assertIn("j1939_raw_hex", msg)


class TestBinaryPackingStructs(unittest.TestCase):
    """Unit tests for standalone binary struct pack/unpack utilities."""

    def test_pack_and_unpack_telemetry_struct(self):
        """Verify packing and unpacking MBTT 33-byte datagrams."""
        data = pack_telemetry_struct(
            seq=42,
            rpm=2345.6,
            oil_pressure=4.78,
            coolant_temp=89.2,
            vib_rms=0.456,
            shaft_torque=0.567,
            hyd_pressure=212.5,
            version=1,
        )
        self.assertEqual(len(data), BINARY_STRUCT_SIZE)
        self.assertTrue(data.startswith(BINARY_STRUCT_MAGIC))

        unpacked = unpack_telemetry_struct(data)
        self.assertIsNotNone(unpacked)
        self.assertEqual(unpacked["seq"], 42)
        self.assertAlmostEqual(unpacked["rpm"], 2345.6, delta=0.01)
        self.assertAlmostEqual(unpacked["oil_pressure"], 4.78, delta=0.01)
        self.assertAlmostEqual(unpacked["coolant_temp"], 89.2, delta=0.01)
        self.assertAlmostEqual(unpacked["vib_rms"], 0.456, delta=0.01)
        self.assertAlmostEqual(unpacked["shaft_torque"], 0.567, delta=0.01)
        self.assertAlmostEqual(unpacked["hyd_pressure"], 212.5, delta=0.01)

    def test_unpack_truncated_or_invalid_binary_returns_none(self):
        """Verify invalid or truncated binary data returns None."""
        self.assertIsNone(unpack_telemetry_struct(b"MBTT"))
        self.assertIsNone(unpack_telemetry_struct(b"XXXX" + b"\x00" * 29))


class TestGlobalLifecycleManagement(unittest.TestCase):
    """Unit tests for start_all_listeners and stop_all_listeners."""

    def test_start_and_stop_all_listeners(self):
        """Verify starting and stopping all background listeners cleanly."""
        listeners = start_all_listeners(
            udp_host="127.0.0.1",
            udp_port=9066,
            serial_port="COM88",
            serial_mock_fallback=True,
        )
        self.assertIn("udp", listeners)
        self.assertIn("serial", listeners)
        self.assertTrue(listeners["udp"].is_running)
        self.assertTrue(listeners["serial"].is_running)

        stop_all_listeners()
        self.assertFalse(listeners["udp"].is_running)
        self.assertFalse(listeners["serial"].is_running)


if __name__ == "__main__":
    unittest.main()
