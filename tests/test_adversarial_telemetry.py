"""Adversarial Stress Test Suite for Milestone M1: Telemetry Gateway & Live Sensor Ingestion.
Tests UDP listener resilience under high-throughput packet blasts, malformed/corrupted datagram fuzzing,
concurrent multi-threaded UDP + REST ingestion, non-numeric payloads, WebSocket stream resilience, and server recovery.
"""

import asyncio
import json
import math
import socket
import struct
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import requests
import uvicorn
from starlette.websockets import WebSocketDisconnect

from telemetry_gateway.j1939_can_parser import J1939FrameParser
from telemetry_gateway.tactical_burst import TacticalBurstPacket
from telemetry_gateway.live_sensor_ingest import (
    TelemetryBroker,
    UDPSensorListener,
    pack_telemetry_struct,
    unpack_telemetry_struct,
    BINARY_STRUCT_MAGIC,
    BINARY_STRUCT_SIZE,
    RAW_FLOATS_6_FORMAT,
    RAW_FLOATS_4_FORMAT,
)
from telemetry_gateway.server import (
    app,
    broker as server_broker,
    websocket_telemetry_endpoint,
)


class MockWebSocket:
    """Mock WebSocket for deterministic testing of WebSocket endpoints under adversarial conditions."""

    def __init__(self, max_messages: int = 5):
        self.accepted = False
        self.sent_messages: List[Dict[str, Any]] = []
        self.max_messages = max_messages

    async def accept(self):
        self.accepted = True

    async def send_json(self, data: Dict[str, Any]):
        self.sent_messages.append(data)
        if len(self.sent_messages) >= self.max_messages:
            raise WebSocketDisconnect()


class TestAdversarialUDPThroughput(unittest.TestCase):
    """Stress tests high-throughput UDP packet floods against the UDP listener."""

    @classmethod
    def setUpClass(cls):
        cls.test_port = 9188
        cls.broker = TelemetryBroker(hardware_timeout=2.0)
        cls.listener = UDPSensorListener(broker=cls.broker, host="127.0.0.1", port=cls.test_port)
        cls.listener.start()
        time.sleep(0.15)

    @classmethod
    def tearDownClass(cls):
        cls.listener.stop()

    def setUp(self):
        self.broker.reset()

    def test_udp_high_throughput_blast_500_packets(self):
        """Blast 500 UDP packets in a tight loop across mixed valid formats."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        target = ("127.0.0.1", self.test_port)

        total_sent = 500
        start_time = time.time()

        for i in range(total_sent):
            fmt_type = i % 4
            if fmt_type == 0:
                # JSON datagram
                pkt = json.dumps({"seq": i, "rpm": 2000.0 + i, "oil_pressure": 4.5, "source": "udp"}).encode("utf-8")
            elif fmt_type == 1:
                # 33-byte MBTT binary struct
                pkt = pack_telemetry_struct(seq=i, rpm=2100.0 + i, oil_pressure=4.8, hyd_pressure=210.0)
            elif fmt_type == 2:
                # 32-byte Tactical Burst
                pkt = TacticalBurstPacket.encode(
                    tank_id=1,
                    mission_time=i,
                    chi=95.0,
                    top_fault_id=0,
                    fault_confidence=0.9,
                    rul_minutes=600,
                    subsystem_health=[95] * 8,
                    rpm=2200.0 + (i % 50),
                    oil_pressure_bar=4.6,
                    coolant_temp_c=88.0,
                    vib_rms=0.45,
                )
            else:
                # 24-byte Raw Floats
                pkt = struct.pack(RAW_FLOATS_6_FORMAT, 2300.0 + i, 4.7, 90.0, 0.48, 0.52, 215.0)

            sock.sendto(pkt, target)

        blast_duration = time.time() - start_time
        sock.close()

        # Wait for listener thread to drain OS socket buffer
        time.sleep(0.4)

        # Verify listener is still running and healthy
        self.assertTrue(self.listener.is_running, "UDP listener crashed during 500-packet blast")
        self.assertGreater(self.listener.packets_received, 0, "No packets received by UDP listener")
        self.assertGreater(self.broker.packet_count, 0, "No packets ingested into broker")
        self.assertTrue(self.broker.is_live_hardware, "Broker should report live hardware after blast")
        self.assertGreaterEqual(self.broker.fps, 0.0, "Broker FPS calculation should be non-negative")
        self.assertFalse(math.isnan(self.broker.fps), "Broker FPS must not be NaN")
        self.assertFalse(math.isinf(self.broker.fps), "Broker FPS must not be Inf")

        # Verify status endpoint integrity
        status = self.broker.get_status()
        self.assertEqual(status["status"], "OPERATIONAL")
        self.assertEqual(status["active_source"], "udp")

    def test_udp_massive_burst_2000_packets(self):
        """Blast 2000 UDP packets continuously at maximum wire rate."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        target = ("127.0.0.1", self.test_port)

        total_sent = 2000
        for i in range(total_sent):
            pkt = pack_telemetry_struct(seq=i, rpm=2000.0 + (i % 500), oil_pressure=4.5)
            sock.sendto(pkt, target)
        sock.close()

        time.sleep(0.5)

        self.assertTrue(self.listener.is_running, "UDP listener died under 2000-packet burst")
        self.assertGreater(self.broker.packet_count, 1000, "Major packet drop in local 2000-packet burst")
        self.assertEqual(self.broker.get_status()["status"], "OPERATIONAL")

    def test_udp_parallel_multiclient_blast_1000_packets(self):
        """Blast 1000 UDP packets concurrently from 10 parallel client threads."""
        threads_count = 10
        packets_per_thread = 100
        target = ("127.0.0.1", self.test_port)

        def worker(thread_id: int):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            for i in range(packets_per_thread):
                seq_num = thread_id * 1000 + i
                payload = json.dumps({
                    "thread": thread_id,
                    "seq": seq_num,
                    "rpm": 2000.0 + (thread_id * 10),
                    "oil_pressure": 4.5,
                }).encode("utf-8")
                sock.sendto(payload, target)
            sock.close()

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(threads_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        time.sleep(0.4)

        self.assertTrue(self.listener.is_running, "UDP listener crashed during multi-client blast")
        self.assertGreaterEqual(self.broker.packet_count, 800, "High packet loss under local UDP blast")
        self.assertEqual(self.broker.get_status()["status"], "OPERATIONAL")


class TestAdversarialMalformedPackets(unittest.TestCase):
    """Fuzzes and stress-tests the UDP listener and parser with corrupt, malformed, and extreme datagrams."""

    @classmethod
    def setUpClass(cls):
        cls.test_port = 9189
        cls.broker = TelemetryBroker(hardware_timeout=2.0)
        cls.listener = UDPSensorListener(broker=cls.broker, host="127.0.0.1", port=cls.test_port)
        cls.listener.start()
        time.sleep(0.15)

    @classmethod
    def tearDownClass(cls):
        cls.listener.stop()

    def setUp(self):
        self.broker.reset()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target = ("127.0.0.1", self.test_port)

    def tearDown(self):
        self.sock.close()

    def test_fuzz_malformed_and_corrupt_datagrams(self):
        """Fuzz listener with a comprehensive corpus of corrupted, truncated, and malicious datagrams."""
        malformed_corpus = [
            # 1. Truncated JSON
            b'{"rpm": 2100.0, "oil_pressure": 4.5,',
            b'{"rpm": ',
            b'{"rpm": 2100.0',
            b'{',
            b'{}',  # Valid empty json
            b'{invalid_json: 123}',

            # 2. Corrupted Binary Structs
            b"MBTT",  # 4 bytes only
            b"MBTT\x01\x00\x00\x00\x01",  # 9 bytes truncated
            b"MBTT" + b"\x00" * 29,  # 33 bytes with null floats
            b"MBTT" + b"\xff" * 29,  # 33 bytes with all 0xFF bytes
            b"XXXX" + b"\x00" * 29,  # 33 bytes with wrong magic
            b"MBTT\x01\x00\x00\x00\x01" + struct.pack(">ffffff", float("nan"), 4.5, 88.0, 0.45, 0.55, 210.0),  # NaN float
            b"MBTT\x01\x00\x00\x00\x01" + struct.pack(">ffffff", float("inf"), 4.5, 88.0, 0.45, 0.55, 210.0),  # Inf float

            # 3. Corrupted Tactical Burst (32 bytes starting with b"TK")
            b"TK" + b"\x00" * 30,  # Invalid CRC / zeros
            b"TK" + b"\xff" * 30,  # Corrupted tactical bytes
            b"TK" + b"CORRUPTED_PAYLOAD_DATA_BYTES!",  # 32 bytes garbage

            # 4. Corrupted J1939 (8 bytes)
            b"\x00" * 8,
            b"\xff" * 8,
            b"\xde\xad\xbe\xef\xca\xfe\xba\xbe",

            # 5. Raw Floats corruptions
            b"\x00" * 24,  # 24 bytes of zeros
            struct.pack(RAW_FLOATS_6_FORMAT, float("nan"), float("inf"), -1e30, 1e30, 0.0, 0.0),
            b"\x00" * 16,  # 16 bytes of zeros

            # 6. Extreme sizes
            b"A" * 1,  # 1 byte
            b"B" * 2,  # 2 bytes
            b"C" * 15,  # 15 bytes
            b"D" * 17,  # 17 bytes
            b"E" * 23,  # 23 bytes
            b"F" * 25,  # 25 bytes
            b"G" * 31,  # 31 bytes
            b"H" * 34,  # 34 bytes
            b"Z" * 4096,  # 4KB payload
            b"\x00" * 8192,  # 8KB null payload

            # 7. Non-ASCII binary garbage
            b"\x80\x81\x82\x83\x90\x91\xfe\xff\x00\x01\x02\x03",
        ]

        for idx, malformed_pkt in enumerate(malformed_corpus):
            self.sock.sendto(malformed_pkt, self.target)

        # Allow processing
        time.sleep(0.3)

        # Assert listener survived without dying
        self.assertTrue(self.listener.is_running, "UDP listener died processing malformed datagram corpus")

        # Send a known valid packet to prove immediate full recovery
        recovery_payload = {"rpm": 2550.0, "oil_pressure": 4.95, "status": "recovered"}
        self.sock.sendto(json.dumps(recovery_payload).encode("utf-8"), self.target)

        time.sleep(0.3)

        # Verify broker ingested valid packet successfully
        latest = self.broker.get_latest_telemetry()
        self.assertEqual(latest.get("rpm"), 2550.0, "Listener failed to process valid packet after fuzzing")
        self.assertTrue(self.broker.is_live_hardware, "Broker failed to establish live hardware state after recovery")

    def test_udp_non_numeric_and_unexpected_json_fields(self):
        """Send JSON packets with string, null, nested lists, and booleans in place of floats."""
        weird_payloads = [
            {"rpm": "HIGH_SPEED", "oil_pressure": "CRITICAL"},
            {"rpm": None, "oil_pressure": None, "coolant_temp": [1, 2, 3]},
            {"rpm": True, "oil_pressure": False, "details": {"sub": "data"}},
            {"rpm": float("inf"), "oil_pressure": -999999.0},
        ]

        for payload in weird_payloads:
            try:
                data = json.dumps(payload).encode("utf-8")
                self.sock.sendto(data, self.target)
            except Exception:
                pass

        time.sleep(0.2)
        self.assertTrue(self.listener.is_running, "Listener died on non-numeric JSON payload")

        # Confirm broker is healthy
        status = self.broker.get_status()
        self.assertEqual(status["status"], "OPERATIONAL")


class TestAdversarialConcurrentUDPAndREST(unittest.TestCase):
    """Stress tests concurrent simultaneous UDP traffic and REST push traffic into the shared broker."""

    @classmethod
    def setUpClass(cls):
        cls.http_port = 8995
        cls.udp_port = 9190
        cls.base_url = f"http://127.0.0.1:{cls.http_port}"

        # Start HTTP server
        cls.config = uvicorn.Config(app, host="127.0.0.1", port=cls.http_port, log_level="error")
        cls.server = uvicorn.Server(cls.config)
        cls.server_thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.server_thread.start()

        # Start UDP listener bound to the server's shared broker
        cls.listener = UDPSensorListener(broker=server_broker, host="127.0.0.1", port=cls.udp_port)
        cls.listener.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.listener.stop()
        cls.server.should_exit = True
        cls.server_thread.join(timeout=1.0)

    def setUp(self):
        server_broker.reset()

    def test_concurrent_udp_and_rest_push_blizzard(self):
        """Send 250 UDP packets and 250 HTTP POST /api/telemetry/push concurrently from 10 threads."""
        udp_workers = 5
        http_workers = 5
        packets_per_udp_worker = 50
        posts_per_http_worker = 50

        errors: List[str] = []

        def udp_task(worker_id: int):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                for i in range(packets_per_udp_worker):
                    data = pack_telemetry_struct(
                        seq=worker_id * 100 + i,
                        rpm=2200.0 + worker_id,
                        oil_pressure=4.6,
                        hyd_pressure=210.0,
                    )
                    sock.sendto(data, ("127.0.0.1", self.udp_port))
                    time.sleep(0.001)
                sock.close()
            except Exception as e:
                errors.append(f"UDP worker {worker_id} error: {e}")

        def http_task(worker_id: int):
            try:
                for i in range(posts_per_http_worker):
                    payload = {
                        "source": f"http_worker_{worker_id}",
                        "rpm": 2400.0 + worker_id,
                        "oil_pressure": 4.8,
                        "coolant_temp": 89.0,
                    }
                    resp = requests.post(
                        f"{self.base_url}/api/telemetry/push",
                        json=payload,
                        timeout=3.0,
                    )
                    if resp.status_code != 200:
                        errors.append(f"HTTP worker {worker_id} status {resp.status_code}: {resp.text}")
            except Exception as e:
                errors.append(f"HTTP worker {worker_id} exception: {e}")

        threads: List[threading.Thread] = []
        for u in range(udp_workers):
            threads.append(threading.Thread(target=udp_task, args=(u,)))
        for h in range(http_workers):
            threads.append(threading.Thread(target=http_task, args=(h,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Wait for UDP loop to finish processing
        time.sleep(0.4)

        self.assertEqual(len(errors), 0, f"Errors occurred during concurrent stress: {errors}")
        self.assertTrue(self.listener.is_running, "UDP listener stopped during concurrent stress")

        # Total expected ingest: up to 250 (UDP) + 250 (HTTP) = 500 packets
        status_resp = requests.get(f"{self.base_url}/api/telemetry/status")
        self.assertEqual(status_resp.status_code, 200)
        status_data = status_resp.json()

        self.assertEqual(status_data["status"], "OPERATIONAL")
        self.assertTrue(status_data["is_live_hardware"])
        self.assertGreaterEqual(status_data["packets_received"], 300, "Too few packets ingested during concurrency test")

        latest_resp = requests.get(f"{self.base_url}/api/telemetry/latest")
        self.assertEqual(latest_resp.status_code, 200)
        latest_data = latest_resp.json()
        self.assertIn("rpm", latest_data)
        self.assertIn("seq", latest_data)


class TestAdversarialRESTPayloads(unittest.TestCase):
    """Adversarial tests against the FastAPI REST push endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.http_port = 8996
        cls.base_url = f"http://127.0.0.1:{cls.http_port}"
        cls.config = uvicorn.Config(app, host="127.0.0.1", port=cls.http_port, log_level="error")
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

    def test_rest_push_invalid_json_types(self):
        """Pushing non-dict non-list JSON payloads should return HTTP 400."""
        # 1. Plain string
        resp = requests.post(
            f"{self.base_url}/api/telemetry/push",
            data='"just a string"',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(resp.status_code, 400)

        # 2. Integer
        resp = requests.post(
            f"{self.base_url}/api/telemetry/push",
            data="12345",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(resp.status_code, 400)

        # 3. Boolean
        resp = requests.post(
            f"{self.base_url}/api/telemetry/push",
            data="true",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_rest_push_large_batch_payload(self):
        """Pushing a 500-item telemetry batch array in a single POST."""
        batch = [
            {"seq": i, "rpm": 2000.0 + (i % 100), "oil_pressure": 4.5, "coolant_temp": 88.0}
            for i in range(500)
        ]
        resp = requests.post(f"{self.base_url}/api/telemetry/push", json=batch)
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertEqual(res["status"], "accepted")
        self.assertEqual(server_broker.packet_count, 500)


class TestAdversarialBrokerEdgeCases(unittest.TestCase):
    """Stress tests internal TelemetryBroker edge conditions, listener failures, and boundary values."""

    def setUp(self):
        self.broker = TelemetryBroker(hardware_timeout=1.0)

    def test_failing_async_and_sync_listeners(self):
        """Verify broker pushes succeed even if multiple sync and async listeners fail catastrophically."""
        sync_called = []
        async_called = []

        def broken_sync_listener(frame):
            raise RuntimeError("Fatal sync listener crash")

        async def broken_async_listener(frame):
            raise RuntimeError("Fatal async listener crash")

        def good_sync_listener(frame):
            sync_called.append(frame)

        self.broker.register_listener(broken_sync_listener)
        self.broker.register_listener(broken_async_listener)
        self.broker.register_listener(good_sync_listener)

        # Push frame
        frame = self.broker.push_telemetry({"rpm": 2100.0}, source="http")
        self.assertEqual(frame["rpm"], 2100.0)
        self.assertEqual(len(sync_called), 1)
        self.assertEqual(self.broker.packet_count, 1)

    def test_broker_latency_calculation_with_extreme_timestamps(self):
        """Test broker latency calculations under future, ancient, millisecond, and second timestamps."""
        # 1. Ancient timestamp (seconds)
        self.broker.push_telemetry({"rpm": 2000.0, "timestamp": 1000.0})
        self.assertGreaterEqual(self.broker.latency_ms, 0.0)

        # 2. Future timestamp (clock skew)
        self.broker.push_telemetry({"rpm": 2000.0, "timestamp": time.time() + 100.0})
        # max(0.0, ...) ensures latency is never negative
        self.assertEqual(self.broker.latency_ms, 0.0)

        # 3. Epoch milliseconds timestamp
        ms_ts = (time.time() - 0.05) * 1000.0
        self.broker.push_telemetry({"rpm": 2000.0, "timestamp": ms_ts})
        self.assertGreaterEqual(self.broker.latency_ms, 0.0)

    def test_broker_reentrant_and_rapid_reset_during_pushes(self):
        """Test thread safety when reset() is called concurrently with push_telemetry()."""
        stop_flag = False

        def pusher():
            while not stop_flag:
                try:
                    self.broker.push_telemetry({"rpm": 2200.0}, source="udp")
                except Exception:
                    pass

        threads = [threading.Thread(target=pusher) for _ in range(4)]
        for t in threads:
            t.start()

        for _ in range(20):
            self.broker.reset()
            time.sleep(0.01)

        stop_flag = True
        for t in threads:
            t.join()

        # Broker must remain in a valid consistent state
        status = self.broker.get_status()
        self.assertEqual(status["status"], "OPERATIONAL")

    def test_broker_data_integrity_and_monotonicity(self):
        """Verify strict monotonicity and sequence integrity across 1000 concurrent multithreaded pushes."""
        num_threads = 10
        pushes_per_thread = 100
        received_sequences = []
        seq_lock = threading.Lock()

        def listener_cb(frame):
            with seq_lock:
                received_sequences.append(frame["seq"])

        self.broker.register_listener(listener_cb)

        def worker(tid: int):
            for i in range(pushes_per_thread):
                self.broker.push_telemetry({"tid": tid, "i": i, "rpm": 2000.0 + i}, source="test")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = num_threads * pushes_per_thread
        self.assertEqual(self.broker.packet_count, total)
        self.assertEqual(len(received_sequences), total)
        # All sequence numbers from 1 to total must be uniquely present
        self.assertEqual(set(received_sequences), set(range(1, total + 1)))

    def test_websocket_streaming_under_adversarial_telemetry(self):
        """Verify WebSocket stream handler handles live frames and does not crash when unexpected keys arrive."""
        server_broker.reset()
        server_broker.push_telemetry(
            {"rpm": 2400.0, "oil_pressure": 4.5, "custom_junk": "extra_data", "source": "udp"},
            source="udp",
        )

        ws = MockWebSocket(max_messages=2)
        asyncio.run(websocket_telemetry_endpoint(ws))  # type: ignore

        self.assertTrue(ws.accepted)
        self.assertEqual(len(ws.sent_messages), 2)
        msg = ws.sent_messages[0]
        self.assertTrue(msg["is_live_hardware"])
        self.assertIn("tactical_burst", msg)
        self.assertIn("j1939_raw_hex", msg)


if __name__ == "__main__":
    unittest.main()
