"""Empirical Challenger Test Suite for Milestone M1 (challenger_m1_2).
Adversarial and stress verification covering:
1. Dynamic pyserial loading & mock generator fallback behaviour.
2. Broker listener registration, unregistration, exception isolation.
3. Delivery latency verification for sub-20ms requirement.
"""

import asyncio
import collections
import gc
import json
import logging
import math
import socket
import struct
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List, Optional

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
    ConnectionManager,
    websocket_telemetry_endpoint,
)


class TestDynamicPyserialAndMockFallback(unittest.TestCase):
    """Adversarial and boundary tests for dynamic pyserial loading & mock generator fallback."""

    def setUp(self):
        self.broker = TelemetryBroker(hardware_timeout=1.0)
        self.broker.reset()

    def tearDown(self):
        self.broker.reset()

    def test_dynamic_pyserial_absent_starts_mock_when_fallback_enabled(self):
        """Verify SerialSensorListener starts mock stream when pyserial is not installed and mock_fallback=True."""
        # Ensure 'serial' is treated as not installed
        with patch.dict(sys.modules, {"serial": None}):
            listener = SerialSensorListener(
                broker=self.broker,
                port="COM_NON_EXISTENT",
                baudrate=115200,
                mock_fallback=True,
            )
            listener.start()

            self.assertTrue(listener.is_running)
            self.assertTrue(listener.is_mock)

            # Wait for mock frames
            time.sleep(0.18)
            listener.stop()

            self.assertFalse(listener.is_running)
            self.assertGreater(self.broker.packet_count, 1)
            latest = self.broker.get_latest_telemetry()
            self.assertEqual(latest.get("source"), "serial")
            self.assertIn("rpm", latest)
            self.assertIn("oil_pressure", latest)
            self.assertIn("coolant_temp", latest)
            self.assertIn("vib_rms", latest)
            self.assertIn("shaft_torque", latest)
            self.assertIn("hyd_pressure", latest)

    def test_dynamic_pyserial_absent_raises_connection_error_when_fallback_disabled(self):
        """Verify SerialSensorListener raises ConnectionError if pyserial missing and mock_fallback=False."""
        with patch.dict(sys.modules, {"serial": None}):
            listener = SerialSensorListener(
                broker=self.broker,
                port="COM_NON_EXISTENT",
                baudrate=115200,
                mock_fallback=False,
            )
            with self.assertRaises(ConnectionError) as ctx:
                listener.start()
            self.assertIn("mock_fallback is disabled", str(ctx.exception))
            self.assertFalse(listener.is_running)
            self.assertFalse(listener.is_mock)

    def test_dynamic_pyserial_present_but_port_fails_falls_back_to_mock(self):
        """Verify that if pyserial is present but COM port throws, it falls back to mock if enabled."""
        mock_serial_mod = MagicMock()
        mock_serial_mod.Serial.side_effect = Exception("Access denied / COM port not found")

        with patch.dict(sys.modules, {"serial": mock_serial_mod}):
            listener = SerialSensorListener(
                broker=self.broker,
                port="COM1",
                baudrate=115200,
                mock_fallback=True,
            )
            listener.start()

            self.assertTrue(listener.is_running)
            self.assertTrue(listener.is_mock)

            time.sleep(0.12)
            listener.stop()
            self.assertFalse(listener.is_running)
            self.assertGreater(self.broker.packet_count, 0)

    def test_dynamic_pyserial_present_and_port_opens_uses_physical_loop(self):
        """Verify that when physical COM port is opened successfully, mock is False and reads lines."""
        mock_serial_obj = MagicMock()
        mock_serial_obj.readline.side_effect = [
            b'{"rpm": 2222.0, "oil_pressure": 4.55, "coolant_temp": 89.0, "vib_rms": 0.44}\n',
            b"",
            b"",
        ]
        mock_serial_mod = MagicMock()
        mock_serial_mod.Serial.return_value = mock_serial_obj

        with patch.dict(sys.modules, {"serial": mock_serial_mod}):
            listener = SerialSensorListener(
                broker=self.broker,
                port="COM4",
                baudrate=115200,
                mock_fallback=True,
            )
            listener.start()

            self.assertTrue(listener.is_running)
            self.assertFalse(listener.is_mock)

            time.sleep(0.1)
            listener.stop()
            self.assertFalse(listener.is_running)
            mock_serial_obj.close.assert_called_once()
            self.assertGreaterEqual(listener.packets_read, 1)

    def test_mock_generator_cadence_and_telemetry_ranges(self):
        """Verify mock generator delivers frames at ~20 Hz with valid physical sensor bounds."""
        listener = SerialSensorListener(broker=self.broker, mock_fallback=True)
        listener.start()

        start_time = time.time()
        time.sleep(0.55)  # Should generate approx 10-11 frames (20 Hz -> 50ms interval)
        listener.stop()
        duration = time.time() - start_time

        count = listener.packets_read
        self.assertGreaterEqual(count, 8, f"Expected at least 8 packets in {duration:.2f}s, got {count}")
        self.assertLessEqual(count, 15, f"Expected at most 15 packets in {duration:.2f}s, got {count}")

        # Check telemetry value validity
        latest = self.broker.get_latest_telemetry()
        self.assertGreaterEqual(latest["rpm"], 2100.0)
        self.assertLessEqual(latest["rpm"], 2300.0)
        self.assertGreaterEqual(latest["oil_pressure"], 4.4)
        self.assertLessEqual(latest["oil_pressure"], 5.0)
        self.assertGreaterEqual(latest["coolant_temp"], 87.0)
        self.assertLessEqual(latest["coolant_temp"], 91.0)
        self.assertGreaterEqual(latest["vib_rms"], 0.40)
        self.assertLessEqual(latest["vib_rms"], 0.50)
        self.assertGreaterEqual(latest["shaft_torque"], 0.50)
        self.assertLessEqual(latest["shaft_torque"], 0.60)
        self.assertGreaterEqual(latest["hyd_pressure"], 209.0)
        self.assertLessEqual(latest["hyd_pressure"], 220.0)

    def test_mock_generator_rapid_lifecycle_stress(self):
        """Stress-test starting and stopping listener 25 times rapidly without leaks or deadlocks."""
        listener = SerialSensorListener(broker=self.broker, mock_fallback=True)
        for _ in range(25):
            listener.start()
            self.assertTrue(listener.is_running)
            time.sleep(0.01)
            listener.stop()
            self.assertFalse(listener.is_running)

    def test_serial_line_parser_adversarial_inputs(self):
        """Stress-test serial line parser with malformed, dirty, and boundary inputs."""
        listener = SerialSensorListener(broker=self.broker)

        # Empty line
        self.assertIsNone(listener._parse_serial_line(b""))
        self.assertIsNone(listener._parse_serial_line(b"   \n"))

        # Broken JSON
        self.assertIsNone(listener._parse_serial_line(b'{"rpm": 2100, incomplete'))
        self.assertIsNone(listener._parse_serial_line(b'{invalid json: 123}'))

        # Corrupted CSV
        self.assertIsNone(listener._parse_serial_line(b"not,a,float,number\n"))
        self.assertIsNone(listener._parse_serial_line(b"2100.0,4.5\n"))  # < 4 parts

        # Valid CSV
        csv_parsed = listener._parse_serial_line(b"2350.5,4.72,89.1,0.48\n")
        self.assertIsNotNone(csv_parsed)
        self.assertEqual(csv_parsed["rpm"], 2350.5)
        self.assertEqual(csv_parsed["oil_pressure"], 4.72)
        self.assertEqual(csv_parsed["coolant_temp"], 89.1)
        self.assertEqual(csv_parsed["vib_rms"], 0.48)

        # Valid JSON
        json_parsed = listener._parse_serial_line(b'{"rpm": 2400.0, "oil_pressure": 4.9}\n')
        self.assertIsNotNone(json_parsed)
        self.assertEqual(json_parsed["rpm"], 2400.0)

        # Non-ASCII / binary garbage
        self.assertIsNone(listener._parse_serial_line(b"\xFF\xFE\x00\x01\x02\x03\x04"))


class TestBrokerListenerSubscriptionsAndIsolation(unittest.TestCase):
    """Adversarial and concurrency tests for TelemetryBroker listener management and exception isolation."""

    def setUp(self):
        self.broker = TelemetryBroker(hardware_timeout=1.0)
        self.broker.reset()

    def tearDown(self):
        self.broker.reset()

    def test_listener_registration_dispatch_and_unregistration(self):
        """Verify listeners receive all dispatched frames in sequence, and unregistering halts delivery."""
        calls_1: List[Dict[str, Any]] = []
        calls_2: List[Dict[str, Any]] = []

        def cb1(f): calls_1.append(f)
        def cb2(f): calls_2.append(f)

        self.broker.register_listener(cb1)
        self.broker.register_listener(cb2)
        self.broker.push_telemetry({"rpm": 2100.0}, source="udp")

        self.assertEqual(len(calls_1), 1)
        self.assertEqual(len(calls_2), 1)
        self.assertEqual(calls_1[0]["seq"], 1)
        self.assertEqual(calls_2[0]["seq"], 1)

        # Unregister cb1
        self.broker.unregister_listener(cb1)
        self.broker.push_telemetry({"rpm": 2200.0}, source="udp")

        self.assertEqual(len(calls_1), 1)  # cb1 not called again
        self.assertEqual(len(calls_2), 2)  # cb2 called again
        self.assertEqual(calls_2[1]["seq"], 2)

    def test_unregister_unregistered_callback_is_safe(self):
        """Verify unregistering a callback not in listeners is a safe no-op (no ValueError)."""
        def dummy_cb(f): pass
        # Should not raise exception
        self.broker.unregister_listener(dummy_cb)
        self.assertEqual(len(self.broker._listeners), 0)

    def test_duplicate_registration_is_idempotent(self):
        """Verify registering the exact same callback multiple times does not result in duplicate calls."""
        calls: List[Dict[str, Any]] = []
        def cb(f): calls.append(f)

        self.broker.register_listener(cb)
        self.broker.register_listener(cb)  # Duplicate register
        self.broker.register_listener(cb)  # Triplicate register

        self.assertEqual(len(self.broker._listeners), 1)
        self.broker.push_telemetry({"rpm": 2000.0}, source="udp")
        self.assertEqual(len(calls), 1)

    def test_exception_isolation_across_multiple_failing_listeners(self):
        """Stress-test exception isolation: multiple failing listeners do not break broker or healthy listeners."""
        healthy_calls: List[int] = []

        def faulty_1(f): raise ValueError("Value error from faulty_1")
        def faulty_2(f): raise ZeroDivisionError("Zero division from faulty_2")
        def faulty_3(f): raise RuntimeError("Runtime error from faulty_3")
        def faulty_4(f): raise KeyError("Missing key error from faulty_4")
        def faulty_5(f): raise Exception("Generic exception from faulty_5")
        def healthy_1(f): healthy_calls.append(f["seq"])
        def healthy_2(f): healthy_calls.append(f["seq"] * 100)

        # Interleave faulty and healthy listeners
        self.broker.register_listener(faulty_1)
        self.broker.register_listener(healthy_1)
        self.broker.register_listener(faulty_2)
        self.broker.register_listener(faulty_3)
        self.broker.register_listener(healthy_2)
        self.broker.register_listener(faulty_4)
        self.broker.register_listener(faulty_5)

        for i in range(1, 11):
            frame = self.broker.push_telemetry({"rpm": 2000.0 + i}, source="serial")
            self.assertEqual(frame["seq"], i)

        # Both healthy listeners must have received all 10 frames
        self.assertEqual(len(healthy_calls), 20)
        self.assertEqual(self.broker.packet_count, 10)

    def test_async_coroutine_listener_execution(self):
        """Verify coroutine functions registered as listeners execute cleanly."""
        received: List[Dict[str, Any]] = []

        async def async_cb(f):
            received.append(f)

        self.broker.register_listener(async_cb)
        self.broker.push_telemetry({"rpm": 2500.0}, source="http")

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["rpm"], 2500.0)

    def test_async_coroutine_failing_listener_isolated(self):
        """Verify throwing coroutine function does not break broker."""
        async def faulty_async(f):
            raise RuntimeError("Async explosion")

        received = []
        def healthy_sync(f):
            received.append(f)

        self.broker.register_listener(faulty_async)
        self.broker.register_listener(healthy_sync)

        frame = self.broker.push_telemetry({"rpm": 2600.0}, source="http")
        self.assertEqual(len(received), 1)
        self.assertEqual(frame["rpm"], 2600.0)

    def test_concurrent_listener_mutation_and_push_stress(self):
        """Stress test: 15 threads pushing telemetry while 5 threads dynamically register/unregister listeners."""
        num_push_threads = 15
        pushes_per_thread = 100
        mutation_duration = 0.5
        stop_mutations = threading.Event()
        errors: List[Exception] = []

        def push_worker(tid: int):
            try:
                for i in range(pushes_per_thread):
                    self.broker.push_telemetry({"tid": tid, "i": i, "rpm": 2000.0 + i}, source="udp")
            except Exception as e:
                errors.append(e)

        def listener_mutator(mid: int):
            def temp_listener(f):
                pass
            try:
                while not stop_mutations.is_set():
                    self.broker.register_listener(temp_listener)
                    time.sleep(0.001)
                    self.broker.unregister_listener(temp_listener)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        push_threads = [threading.Thread(target=push_worker, args=(t,)) for t in range(num_push_threads)]
        mut_threads = [threading.Thread(target=listener_mutator, args=(m,)) for m in range(5)]

        for m in mut_threads: m.start()
        for p in push_threads: p.start()
        for p in push_threads: p.join()

        stop_mutations.set()
        for m in mut_threads: m.join()

        self.assertEqual(len(errors), 0, f"Encountered concurrency errors: {errors}")
        self.assertEqual(self.broker.packet_count, num_push_threads * pushes_per_thread)


class TestDeliveryLatencySub20msBenchmark(unittest.TestCase):
    """Empirical latency and throughput benchmarks verifying sub-20ms delivery requirement."""

    def setUp(self):
        self.broker = TelemetryBroker(hardware_timeout=1.0)
        self.broker.reset()

    def tearDown(self):
        self.broker.reset()

    def test_broker_ingest_and_dispatch_latency_p99(self):
        """Benchmark 5,000 pushes measuring wall-clock latency per push_telemetry call."""
        latencies_ms: List[float] = []

        def dummy_listener(f):
            _ = f["rpm"] * 1.5

        self.broker.register_listener(dummy_listener)

        sample_frame = {
            "rpm": 2150.0,
            "oil_pressure": 4.55,
            "coolant_temp": 88.5,
            "vib_rms": 0.45,
            "shaft_torque": 0.55,
            "hyd_pressure": 210.0,
        }

        # Warm up
        for _ in range(100):
            self.broker.push_telemetry(sample_frame, source="udp")

        # Benchmark 5,000 pushes
        num_samples = 5000
        for _ in range(num_samples):
            t0 = time.perf_counter()
            self.broker.push_telemetry(sample_frame, source="udp")
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        latencies_ms.sort()
        p50 = latencies_ms[int(num_samples * 0.50)]
        p90 = latencies_ms[int(num_samples * 0.90)]
        p99 = latencies_ms[int(num_samples * 0.99)]
        max_lat = latencies_ms[-1]
        avg_lat = sum(latencies_ms) / num_samples

        print(f"\n[Broker Ingest Latency Bench] Samples={num_samples} | "
              f"Avg={avg_lat:.4f}ms | P50={p50:.4f}ms | P90={p90:.4f}ms | P99={p99:.4f}ms | Max={max_lat:.4f}ms")

        # Ingest and dispatch must be < 1.0 ms (far below 20ms requirement)
        self.assertLess(p99, 5.0, f"Broker P99 latency {p99:.2f}ms exceeds 5.0ms threshold")
        self.assertLess(avg_lat, 1.0, f"Broker avg latency {avg_lat:.2f}ms exceeds 1.0ms threshold")

    def test_timestamp_delta_latency_computation(self):
        """Verify broker accurately computes latency_ms from frame timestamps."""
        # 1. Seconds timestamp with 12ms simulated transit delay
        now = time.time()
        frame_sec = {"rpm": 2200.0, "timestamp": now - 0.012}
        pushed = self.broker.push_telemetry(frame_sec, source="udp")
        self.assertAlmostEqual(pushed["ingest_latency_ms"], 12.0, delta=3.0)
        self.assertAlmostEqual(self.broker.latency_ms, 12.0, delta=3.0)

        # 2. Milliseconds timestamp with 8ms simulated transit delay
        now_ms = time.time() * 1000.0
        frame_ms = {"rpm": 2250.0, "timestamp": now_ms - 8.0}
        pushed2 = self.broker.push_telemetry(frame_ms, source="udp")
        self.assertAlmostEqual(pushed2["ingest_latency_ms"], 8.0, delta=3.0)

    def test_websocket_frame_encoding_and_serialization_latency(self):
        """Benchmark full WebSocket payload transformation and serialization under live mode."""
        # Setup live broker state
        self.broker.push_telemetry(
            {
                "rpm": 2250.0,
                "oil_pressure": 4.65,
                "coolant_temp": 89.0,
                "vib_rms": 0.48,
                "shaft_torque": 0.55,
                "hyd_pressure": 210.0,
                "composite_chi": 94.0,
                "rul_minutes": 550,
            },
            source="udp",
        )

        encode_latencies_ms: List[float] = []
        num_cycles = 1000

        for seq in range(num_cycles):
            t0 = time.perf_counter()

            # Execute the exact transformation done inside websocket_telemetry_endpoint
            latest = self.broker.get_latest_telemetry()
            rpm = float(latest.get("rpm", 2100.0))
            oil_p = float(latest.get("oil_pressure", 4.5))
            coolant_t = float(latest.get("coolant_temp", 88.0))
            vib_rms = float(latest.get("vib_rms", 0.45))
            chi = float(latest.get("composite_chi", 95.0))
            torque = float(latest.get("shaft_torque", 0.55))

            eec1_bytes = J1939FrameParser.encode_eec1(rpm, min(100.0, max(0.0, torque * 100.0)))
            # Coolant temperature moved to PGN 65262 (ET1); EFL_P1 carries
            # SPN 100 oil pressure only. See tests/test_gateway.py.
            efl_bytes = J1939FrameParser.encode_efl_p1(oil_p * 100.0)
            et1_bytes = J1939FrameParser.encode_et1(coolant_t)
            assert len(et1_bytes) == 8

            tactical_bytes = TacticalBurstPacket.encode(
                tank_id=1,
                mission_time=int(seq),
                chi=chi,
                top_fault_id=0,
                fault_confidence=0.95,
                rul_minutes=int(chi * 12),
                subsystem_health=[chi, chi, chi, 98, 99, 95, 96, 99],
                rpm=rpm,
                oil_pressure_bar=oil_p,
                coolant_temp_c=coolant_t,
                vib_rms=vib_rms,
            )
            burst_dict = TacticalBurstPacket.decode(tactical_bytes)

            payload = {
                "step": seq,
                "timestamp": latest.get("timestamp", time.time()),
                "is_live_hardware": True,
                "source": "udp",
                "stream_badge": "[LIVE HARDWARE STREAM: UDP]",
                "telemetry_raw": latest,
                "tactical_burst": burst_dict,
                "j1939_raw_hex": {
                    "EEC1_PGN61444": eec1_bytes.hex().upper(),
                    "EFL_P1_PGN65263": efl_bytes.hex().upper(),
                },
            }
            # Serialize to JSON string as websocket would
            json_str = json.dumps(payload)
            t1 = time.perf_counter()

            encode_latencies_ms.append((t1 - t0) * 1000.0)

        encode_latencies_ms.sort()
        p50 = encode_latencies_ms[int(num_cycles * 0.50)]
        p90 = encode_latencies_ms[int(num_cycles * 0.90)]
        p99 = encode_latencies_ms[int(num_cycles * 0.99)]
        avg = sum(encode_latencies_ms) / num_cycles
        max_val = encode_latencies_ms[-1]

        print(f"\n[WebSocket Frame Encoding Latency Bench] Cycles={num_cycles} | "
              f"Avg={avg:.4f}ms | P50={p50:.4f}ms | P90={p90:.4f}ms | P99={p99:.4f}ms | Max={max_val:.4f}ms")

        # Must be well below 20 ms (typically < 0.5 ms)
        self.assertLess(p99, 5.0, f"Encoding P99 latency {p99:.2f}ms exceeds 5.0ms")
        self.assertLess(avg, 1.0, f"Encoding avg latency {avg:.2f}ms exceeds 1.0ms")

    def test_high_frequency_burst_ingest_throughput(self):
        """Verify broker can sustain high burst ingestion (1,000 packets) without packet drops or lag."""
        num_burst = 1000
        start = time.perf_counter()
        for i in range(num_burst):
            self.broker.push_telemetry({"burst_idx": i, "rpm": 2000.0 + (i % 100)}, source="udp")
        elapsed = time.perf_counter() - start

        throughput_fps = num_burst / elapsed
        print(f"\n[Burst Throughput Bench] Ingested {num_burst} pkts in {elapsed:.4f}s -> {throughput_fps:.1f} pkts/sec")

        self.assertEqual(self.broker.packet_count, num_burst)
        self.assertGreater(throughput_fps, 5000.0, "Throughput must exceed 5,000 pkts/sec")


if __name__ == "__main__":
    unittest.main()
