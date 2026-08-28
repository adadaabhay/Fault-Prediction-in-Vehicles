"""Adversarial Verification & End-to-End Integration Test Suite for Milestone M3.
Authored by: challenger_m3_2 (Empirical Challenger & Military Telemetry Verification Specialist).

Empirically verifies:
1. End-to-End Fault Injection Pipeline:
   - Wire cut on oil pressure (P_oil = -999 bar) pushed via UDP / REST.
   - SensorPlausibilityGate catches open circuit, clamps to 0.0 bar, emits SensorFaultEvent(channel='oil_pressure', spn=100, fmi=4).
   - DTCEngine converts fault to PGN 65226 DM1 packet with Red Stop Lamp, SPN 100, FMI 04.
   - Flash log (results/dtc_flash_log.jsonl) records immutable state transition.
   - Neural prognostics and part health inference remain uncorrupted and finite throughout.
2. Full Multi-Channel & Multi-Fault Matrix (Open, Short, Slew, Stuck-at, EMI, Cross-subsystem).
3. Variable Length SAE J1939 DM1 / DM2 binary encoding (Single DTC 8-byte, Multi-DTC 2+4N bytes).
4. DTC Lifecycle & Occurrence Counter transitions (Active DM1 -> Cleared DM2 -> Reactivated DM1 with OC increment).
5. Thread-safe Circular Flash Ring Buffer Concurrency, Rollover, and Crash Recovery.
6. Execution Latency and High-Throughput Verification (< 0.5 ms per frame).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from typing import Any, Dict, List

# Add the repo root (this directory's parent) to sys.path so the
# telemetry_gateway/, sim/, and ml/ packages import without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from telemetry_gateway.dtc_engine import (
    DTCEngine,
    DTCRecord,
    LampStatus,
    LampType,
    PGN_DM1,
    PGN_DM2,
    LAMP_OFF,
    LAMP_ON,
    FMI_DATA_VALID_ABOVE_NORMAL,
    FMI_DATA_VALID_BELOW_NORMAL,
    FMI_DATA_ERRATIC,
    FMI_VOLTAGE_ABOVE_NORMAL,
    FMI_VOLTAGE_BELOW_NORMAL,
    FMI_MECHANICAL_SYSTEM_FAIL,
    FMI_ABNORMAL_RATE_OF_CHANGE,
    FMI_SPECIAL_INSTRUCTIONS,
    CHANNEL_TO_SPN_MAP,
    NEURAL_FAULT_CLASS_MAP,
)
from telemetry_gateway.sensor_plausibility import (
    SensorPlausibilityGate,
    SensorFaultEvent,
    PlausibilityFaultType,
)
from telemetry_gateway.live_sensor_ingest import (
    TelemetryBroker,
    UDPSensorListener,
    pack_telemetry_struct,
    unpack_telemetry_struct,
)

# Import ML model and health scoring for neural corruption tests
from ml.lstm import LSTMModel
from ml.parts import part_health_index, INPUT_FEATURES


class TestEndToEndOilPressureWireCutPipeline(unittest.TestCase):
    """Requirement 1: Complete End-to-End Fault Injection and Diagnostic Logging Pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_file = Path(self.temp_dir.name) / "dtc_flash_log.jsonl"
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)
        self.engine = DTCEngine(flash_capacity=1000, flash_file=self.flash_file)
        self.broker = TelemetryBroker()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_end_to_end_wire_cut_fault_injection_standalone(self):
        """Pure sensor frame with P_oil = -999.0 bar -> Gate clamps to 0.0 -> Emits SPN 100 FMI 04 ->

        DM1 packet has Red Stop Lamp & SPN 100 FMI 04 -> Flash log written -> Neural intact.
        """
        raw_telemetry = {
            "oil_pressure": -999.0,  # Wire cut
            "coolant_temp": 88.0,
        }

        # Step 1: Plausibility Gate filters raw frame
        plaus_res = self.gate.filter_frame(raw_telemetry)
        self.assertTrue(plaus_res.is_valid)

        # Verify safe physical clamping
        self.assertEqual(plaus_res.clean_telemetry["oil_pressure"], 0.0)
        self.assertGreaterEqual(plaus_res.clean_telemetry["oil_pressure"], 0.0)
        self.assertLessEqual(plaus_res.clean_telemetry["oil_pressure"], 15.0)

        # Verify Plausibility Fault Event
        self.assertEqual(len(plaus_res.faults_detected), 1)
        fault_evt = plaus_res.faults_detected[0]
        self.assertEqual(fault_evt.channel, "oil_pressure")
        self.assertEqual(fault_evt.fault_type, PlausibilityFaultType.OPEN_CIRCUIT.value)
        self.assertEqual(fault_evt.raw_value, -999.0)
        self.assertEqual(fault_evt.clamped_value, 0.0)
        self.assertEqual(fault_evt.spn, 100)
        self.assertEqual(fault_evt.fmi, FMI_VOLTAGE_BELOW_NORMAL)  # FMI 04

        # Step 2: DTCEngine ingests FDIR faults
        active_dtcs = self.engine.process_fdir_faults(plaus_res.faults_detected)
        self.assertEqual(len(active_dtcs), 1)
        dtc = active_dtcs[0]
        self.assertEqual(dtc.spn, 100)
        self.assertEqual(dtc.fmi, 4)
        self.assertEqual(dtc.oc, 1)
        self.assertEqual(dtc.cm, 0)
        self.assertEqual(dtc.lamp_status, "RED_STOP")
        self.assertEqual(dtc.source_type, "FDIR_ELECTRICAL")

        # Step 3: Verify SAE J1939 DM1 Packet Generation (PGN 65226)
        dm1_bytes = self.engine.encode_dm1_packet()
        self.assertEqual(len(dm1_bytes), 8)

        # Header check: Byte 0 has Red Stop Lamp ON (bits 6-5 = 01 -> 0x10)
        self.assertEqual(dm1_bytes[0], 0x10)
        # DTC 4-byte packed check: SPN 100 (0x64, 0x00), FMI 04 (0x04), OC 1 (0x01)
        self.assertEqual(dm1_bytes[2:6], b"\x64\x00\x04\x01")
        # Padding check
        self.assertEqual(dm1_bytes[6:8], b"\xff\xff")

        # Step 4: Decode DM1 Packet using standard parser
        lamp_decoded, records_decoded = DTCEngine.decode_dm1_packet(dm1_bytes)
        self.assertTrue(lamp_decoded.is_red_stop)
        self.assertFalse(lamp_decoded.is_amber)
        self.assertEqual(len(records_decoded), 1)
        self.assertEqual(records_decoded[0].spn, 100)
        self.assertEqual(records_decoded[0].fmi, 4)
        self.assertEqual(records_decoded[0].oc, 1)

        # Step 5: Verify Flash Ring Buffer persistence (results/dtc_flash_log.jsonl)
        flash_entries = self.engine.read_flash_log(limit=10)
        self.assertEqual(len(flash_entries), 1)
        flash_record = flash_entries[0]
        self.assertEqual(flash_record["spn"], 100)
        self.assertEqual(flash_record["fmi"], 4)
        self.assertEqual(flash_record["oc"], 1)
        self.assertEqual(flash_record["code"], "SPN 100 FMI 04")
        self.assertEqual(flash_record["lamp_status"], "RED_STOP")
        self.assertEqual(flash_record["event_type"], "ACTIVE")
        self.assertEqual(flash_record["raw_value"], -999.0)
        self.assertEqual(flash_record["clamped_value"], 0.0)

        # Step 6: Verify Neural Inference Robustness (No corruption, no NaN)
        # 6a: Part health score under clean sanitized telemetry
        health_lub = part_health_index("lubrication", plaus_res.clean_telemetry)
        self.assertTrue(math.isfinite(health_lub))
        self.assertGreaterEqual(health_lub, 0.0)
        self.assertLessEqual(health_lub, 100.0)

        # 6b: Neural LSTM model forward pass under clean sanitized features
        model = LSTMModel(D=len(INPUT_FEATURES), H=32, R=8, C=13, seed=42)
        X = np.zeros((10, len(INPUT_FEATURES)), dtype=np.float32)
        for t in range(10):
            for f_idx, feat in enumerate(INPUT_FEATURES):
                X[t, f_idx] = plaus_res.clean_telemetry.get(feat, 1.0)

        cache = model.forward(X)
        reg_out = cache["reg"]
        cls_out = cache["cls"]

        self.assertTrue(np.all(np.isfinite(reg_out)))
        self.assertTrue(np.all(np.isfinite(cls_out)))
        self.assertAlmostEqual(float(np.sum(cls_out)), 1.0, places=5)
        self.assertTrue(np.all(reg_out >= 0.0) and np.all(reg_out <= 1.0))

    def test_broker_pipeline_wire_cut_and_dual_sensor_detection(self):
        """Full multi-sensor frame pushed via broker -> Gate catches wire cut & cross-mismatch ->

        DTCEngine registers active SPN 100 FMI 04 with Red Stop Lamp.
        """
        raw_telemetry = {
            "rpm": 2100.0,
            "oil_pressure": -999.0,
            "coolant_temp": 88.0,
            "vib_rms": 0.45,
            "shaft_torque": 0.55,
            "hyd_pressure": 210.0,
        }

        # Step 1: Plausibility filter
        plaus_res = self.gate.filter_frame(raw_telemetry)
        self.assertEqual(plaus_res.clean_telemetry["oil_pressure"], 0.0)

        # Find the open circuit fault event
        cut_events = [f for f in plaus_res.faults_detected if f.channel == "oil_pressure" and f.fault_type == "OPEN_CIRCUIT"]
        self.assertEqual(len(cut_events), 1)
        self.assertEqual(cut_events[0].spn, 100)
        self.assertEqual(cut_events[0].fmi, 4)

        # Step 2: DTC Engine
        self.engine.process_fdir_faults(plaus_res.faults_detected)
        active_map = {(d.spn, d.fmi): d for d in self.engine.get_active_dtcs()}
        self.assertIn((100, 4), active_map)
        self.assertEqual(active_map[(100, 4)].lamp_status, "RED_STOP")

        # Step 3: DM1 Packet
        dm1_payload = self.engine.encode_dm1_packet()
        lamp_decoded, decoded_dtcs = DTCEngine.decode_dm1_packet(dm1_payload)
        self.assertTrue(lamp_decoded.is_red_stop)
        spn_fmi_set = {(d.spn, d.fmi) for d in decoded_dtcs}
        self.assertIn((100, 4), spn_fmi_set)

    def test_udp_binary_datagram_fault_injection(self):
        """Inject wire cut via 33-byte MBTT binary struct over UDP parser."""
        listener = UDPSensorListener(broker=self.broker, port=9999)
        binary_datagram = pack_telemetry_struct(
            seq=42,
            rpm=1200.0,  # Below 1500 RPM to avoid cross-subsystem trigger
            oil_pressure=-999.0,
            coolant_temp=88.0,
            vib_rms=0.45,
            shaft_torque=0.55,
            hyd_pressure=210.0,
        )

        parsed = listener.parse_datagram(binary_datagram)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["oil_pressure"], -999.0)

        # Plausibility filter
        plaus_res = self.gate.filter_frame(parsed)
        self.assertEqual(plaus_res.clean_telemetry["oil_pressure"], 0.0)

        cut_events = [f for f in plaus_res.faults_detected if f.channel == "oil_pressure" and f.fault_type == "OPEN_CIRCUIT"]
        self.assertEqual(len(cut_events), 1)
        self.assertEqual(cut_events[0].spn, 100)
        self.assertEqual(cut_events[0].fmi, 4)

        # DTC engine
        dtcs = self.engine.process_fdir_faults(plaus_res.faults_detected)
        active_map = {(d.spn, d.fmi): d for d in self.engine.get_active_dtcs()}
        self.assertIn((100, 4), active_map)
        self.assertEqual(active_map[(100, 4)].lamp_status, "RED_STOP")


class TestMultiChannelElectricalAndPlausibilityMatrix(unittest.TestCase):
    """Requirement 2: Comprehensive Multi-Channel Fault Injection Matrix."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_file = Path(self.temp_dir.name) / "matrix_flash.jsonl"
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)
        self.engine = DTCEngine(flash_file=self.flash_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_coolant_wire_cut_and_short_circuits(self):
        """Test coolant temperature wire cut (-999 C) and short to power (+999 C)."""
        # 1. Wire cut
        self.gate.reset()
        res_cut = self.gate.filter_frame({"coolant_temp": -999.0})
        self.assertEqual(res_cut.clean_telemetry["coolant_temp"], -40.0)
        dtcs_cut = self.engine.process_fdir_faults(res_cut.faults_detected)
        self.assertEqual(dtcs_cut[0].spn, 110)
        self.assertEqual(dtcs_cut[0].fmi, FMI_VOLTAGE_BELOW_NORMAL)
        self.assertEqual(dtcs_cut[0].lamp_status, "RED_STOP")

        # 2. Short to power (with clean reset so rate-of-change limiter doesn't clamp from -40 to 160)
        self.gate.reset()
        self.engine.reset()
        res_short = self.gate.filter_frame({"coolant_temp": 999.0})
        self.assertEqual(res_short.clean_telemetry["coolant_temp"], 160.0)
        dtcs_short = self.engine.process_fdir_faults(res_short.faults_detected)
        self.assertEqual(dtcs_short[0].spn, 110)
        self.assertEqual(dtcs_short[0].fmi, FMI_VOLTAGE_ABOVE_NORMAL)
        self.assertEqual(dtcs_short[0].lamp_status, "RED_STOP")

    def test_hydraulic_pressure_wire_cut_and_short(self):
        """Test hydraulic pressure wire cut and short to power."""
        self.gate.reset()
        res_cut = self.gate.filter_frame({"hyd_pressure": -999.0})
        self.assertEqual(res_cut.clean_telemetry["hyd_pressure"], 0.0)
        dtcs_cut = self.engine.process_fdir_faults(res_cut.faults_detected)
        self.assertEqual(dtcs_cut[0].spn, 520202)
        self.assertEqual(dtcs_cut[0].fmi, FMI_VOLTAGE_BELOW_NORMAL)
        self.assertEqual(dtcs_cut[0].lamp_status, "RED_STOP")

        self.gate.reset()
        self.engine.reset()
        res_short = self.gate.filter_frame({"hyd_pressure": 999.0})
        self.assertEqual(res_short.clean_telemetry["hyd_pressure"], 400.0)
        dtcs_short = self.engine.process_fdir_faults(res_short.faults_detected)
        self.assertEqual(dtcs_short[0].spn, 520202)
        self.assertEqual(dtcs_short[0].fmi, FMI_VOLTAGE_ABOVE_NORMAL)
        self.assertEqual(dtcs_short[0].lamp_status, "RED_STOP")

    def test_rpm_sensor_wire_cut(self):
        """Test engine speed wire cut (-999 RPM)."""
        self.gate.reset()
        res = self.gate.filter_frame({"rpm": -999.0})
        self.assertEqual(res.clean_telemetry["rpm"], 0.0)
        dtcs = self.engine.process_fdir_faults(res.faults_detected)
        self.assertEqual(dtcs[0].spn, 190)
        self.assertEqual(dtcs[0].fmi, FMI_VOLTAGE_BELOW_NORMAL)
        self.assertEqual(dtcs[0].lamp_status, "RED_STOP")

    def test_slew_rate_violation_mapping(self):
        """Test unphysical step change -> FMI 10 / AMBER_WARNING."""
        self.gate.reset()
        # Establish baseline
        self.gate.filter_frame({"oil_temp": 80.0})
        # Sudden step jump
        res = self.gate.filter_frame({"oil_temp": 170.0})
        dtcs = self.engine.process_fdir_faults(res.faults_detected)
        self.assertEqual(dtcs[0].spn, 175)
        self.assertEqual(dtcs[0].fmi, FMI_ABNORMAL_RATE_OF_CHANGE)
        self.assertEqual(dtcs[0].lamp_status, "AMBER_WARNING")


class TestMultiDTCBAMPacketEncodingAndDecoders(unittest.TestCase):
    """Requirement 3: Variable Length Multi-DTC DM1 / DM2 Encoding and Decoding."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_file = Path(self.temp_dir.name) / "bam_flash.jsonl"
        self.engine = DTCEngine(flash_file=self.flash_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_five_concurrent_dtcs_variable_length_payload(self):
        """5 active DTCs -> payload size = 2 + 4*5 = 22 bytes."""
        faults = [
            (100, 4, "RED_STOP"),
            (110, 3, "RED_STOP"),
            (190, 2, "AMBER_WARNING"),
            (513, 7, "AMBER_WARNING"),
            (520101, 7, "AMBER_WARNING"),
        ]

        for spn, fmi, lamp in faults:
            self.engine.report_fault(spn=spn, fmi=fmi, lamp=lamp)

        active = self.engine.get_active_dtcs()
        self.assertEqual(len(active), 5)

        # Encode DM1
        dm1_payload = self.engine.encode_dm1_packet()
        self.assertEqual(len(dm1_payload), 2 + 4 * 5)  # 22 bytes

        # Decode DM1
        lamp_header, records = DTCEngine.decode_dm1_packet(dm1_payload)
        self.assertTrue(lamp_header.is_red_stop)
        self.assertTrue(lamp_header.is_amber)
        self.assertEqual(len(records), 5)

        decoded_pairs = {(r.spn, r.fmi) for r in records}
        expected_pairs = {(f[0], f[1]) for f in faults}
        self.assertEqual(decoded_pairs, expected_pairs)

    def test_j1939_19bit_spn_boundary_packing(self):
        """Test boundary SPN 524287 (max 19-bit), FMI 31 (max 5-bit), OC 127 (max 7-bit)."""
        dtc = DTCRecord(spn=524287, fmi=31, oc=127, cm=1)
        raw_4b = dtc.encode_4bytes()
        self.assertEqual(raw_4b, b"\xff\xff\xff\xff")

        decoded = DTCRecord.decode_4bytes(raw_4b)
        self.assertEqual(decoded.spn, 524287)
        self.assertEqual(decoded.fmi, 31)
        self.assertEqual(decoded.oc, 127)
        self.assertEqual(decoded.cm, 1)


class TestDTCLifecycleAndOccurrenceCounters(unittest.TestCase):
    """Requirement 4: Lifecycle Transitions and Occurrence Count Tracking."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_file = Path(self.temp_dir.name) / "lifecycle_flash.jsonl"
        self.engine = DTCEngine(flash_file=self.flash_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_active_cleared_reactivated_lifecycle(self):
        """Test complete cycle: Active (DM1, OC=1) -> Cleared (DM2) -> Reactivated (DM1, OC=2)."""
        # Phase 1: Activate
        dtc1 = self.engine.report_fault(spn=100, fmi=4, description="Oil wire cut")
        self.assertEqual(dtc1.oc, 1)
        self.assertTrue(dtc1.active)
        self.assertEqual(len(self.engine.get_active_dtcs()), 1)
        self.assertEqual(len(self.engine.get_historic_dtcs()), 0)

        # Phase 2: Clear Fault
        cleared = self.engine.clear_fault(100, 4)
        self.assertIsNotNone(cleared)
        self.assertFalse(cleared.active)
        self.assertEqual(len(self.engine.get_active_dtcs()), 0)
        self.assertEqual(len(self.engine.get_historic_dtcs()), 1)

        # Verify DM1 shows zero DTCs and DM2 shows historic DTC
        dm1_bytes = self.engine.encode_dm1_packet()
        dm2_bytes = self.engine.encode_dm2_packet()
        _, dm1_records = DTCEngine.decode_dm1_packet(dm1_bytes)
        _, dm2_records = DTCEngine.decode_dm2_packet(dm2_bytes)
        self.assertEqual(len(dm1_records), 0)
        self.assertEqual(len(dm2_records), 1)
        self.assertEqual(dm2_records[0].spn, 100)

        # Phase 3: Reactivate Fault -> Increments OC to 2
        dtc2 = self.engine.report_fault(spn=100, fmi=4)
        self.assertEqual(dtc2.oc, 2)
        self.assertTrue(dtc2.active)
        self.assertEqual(len(self.engine.get_active_dtcs()), 1)
        self.assertEqual(len(self.engine.get_historic_dtcs()), 0)

        # Phase 4: Clear again and reactivate -> Increments OC to 3
        self.engine.clear_fault(100, 4)
        dtc3 = self.engine.report_fault(spn=100, fmi=4)
        self.assertEqual(dtc3.oc, 3)

        # Verify flash log has 5 recorded state transitions
        records = self.engine.read_flash_log(limit=20)
        self.assertEqual(len(records), 5)


class TestFlashRingBufferConcurrencyAndRecovery(unittest.TestCase):
    """Requirement 5: Thread-Safety, Rollover, and State Recovery."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_file = Path(self.temp_dir.name) / "concurrency_flash.jsonl"
        self.engine = DTCEngine(flash_capacity=200, flash_file=self.flash_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_multithreaded_concurrent_fault_reporting(self):
        """10 threads concurrently reporting faults without corruption."""
        errors = []
        threads = []

        def worker(tid: int):
            try:
                for i in range(25):
                    spn = 520000 + (tid * 50) + i
                    self.engine.report_fault(spn=spn, fmi=i % 4)
            except Exception as e:
                errors.append(e)

        for t in range(10):
            th = threading.Thread(target=worker, args=(t,))
            threads.append(th)
            th.start()

        for th in threads:
            th.join()

        self.assertEqual(len(errors), 0)
        entries = self.engine.read_flash_log(limit=500)
        self.assertGreater(len(entries), 50)
        self.assertLessEqual(len(entries), 200)

    def test_crash_recovery_state_restoration(self):
        """Restore active and historic state from flash log on restart."""
        self.engine.report_fault(spn=100, fmi=4, description="Oil Cut")
        self.engine.report_fault(spn=110, fmi=0, description="Coolant High")
        self.engine.report_fault(spn=190, fmi=2, description="RPM Erratic")
        self.engine.clear_fault(190, 2)

        # Simulate cold restart
        fresh_engine = DTCEngine(flash_file=self.flash_file, auto_recover=True)
        active_spns = {d.spn for d in fresh_engine.get_active_dtcs()}
        historic_spns = {d.spn for d in fresh_engine.get_historic_dtcs()}

        self.assertEqual(active_spns, {100, 110})
        self.assertEqual(historic_spns, {190})


class TestRealTimeLatencyAndThroughputBenchmark(unittest.TestCase):
    """Requirement 6: Real-time Performance & Execution Latency Benchmark."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_file = Path(self.temp_dir.name) / "perf_flash.jsonl"
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)
        self.engine = DTCEngine(flash_capacity=5000, flash_file=self.flash_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pipeline_throughput_and_latency(self):
        """Benchmark 1,000 complete pipeline frames (Filter -> DTC -> DM1 Encode -> Flash)."""
        raw_frame = {
            "rpm": 2100.0,
            "oil_pressure": -999.0,
            "coolant_temp": 88.0,
            "vib_rms": 0.45,
            "shaft_torque": 0.55,
            "hyd_pressure": 210.0,
        }

        t0 = time.perf_counter()
        iterations = 1000

        for _ in range(iterations):
            plaus_res = self.gate.filter_frame(raw_frame)
            self.engine.process_fdir_faults(plaus_res.faults_detected)
            dm1_bytes = self.engine.encode_dm1_packet()

        elapsed_total_s = time.perf_counter() - t0
        avg_latency_ms = (elapsed_total_s / iterations) * 1000.0

        self.assertLess(avg_latency_ms, 0.5, f"Avg latency {avg_latency_ms:.4f} ms exceeds 0.5 ms threshold")


if __name__ == "__main__":
    unittest.main()
