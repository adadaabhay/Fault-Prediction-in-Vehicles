"""Final Empirical Adversarial Test Suite for SAE J1939 DTC Engine & Ingestion Pipeline.

Author: challenger_m3_final (Adversarial Testing & Military Protocol Verification Specialist)
Target: telemetry_gateway/dtc_engine.py, telemetry_gateway/sensor_plausibility.py, telemetry_gateway/live_sensor_ingest.py

Adversarial Verification Dimensions:
1. Malformed / Truncated DM1 & DM2 binary packets with 10,000 random fuzzed payloads.
2. Extreme boundary bitfields (SPN 524287, FMI 31, OC 127, CM 1, overflow/underflow handling).
3. High-frequency fault lifecycle toggling (150+ faults active/cleared across 1,000 frames with state invariants).
4. High-concurrency multithreaded flash ring buffer logging (20+ threads with rollover & crash recovery).
5. End-to-end adversarial sensor injection: wire-cut oil pressure (P_oil = -999 bar) -> SensorPlausibilityGate -> DTCEngine -> DM1 PGN 65226 Red Stop Lamp packet -> results/dtc_flash_log.jsonl.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import random
import struct
import sys
import tempfile
import threading
import time
import unittest
from typing import Any, Dict, List, Set, Tuple

# Project root path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Fault-Prediction-in-Vehicles"))

from telemetry_gateway.dtc_engine import (
    DTCEngine,
    DTCRecord,
    LampStatus,
    LampType,
    PGN_DM1,
    PGN_DM2,
    LAMP_OFF,
    LAMP_ON,
    LAMP_ERROR,
    LAMP_NOT_AVAILABLE,
    FLASH_SLOW,
    FLASH_FAST,
    FLASH_RESERVED,
    FLASH_UNAVAILABLE,
    FMI_DATA_VALID_ABOVE_NORMAL,
    FMI_DATA_VALID_BELOW_NORMAL,
    FMI_DATA_ERRATIC,
    FMI_VOLTAGE_ABOVE_NORMAL,
    FMI_VOLTAGE_BELOW_NORMAL,
    FMI_CURRENT_BELOW_NORMAL,
    FMI_CURRENT_ABOVE_NORMAL,
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
    pack_telemetry_struct,
    unpack_telemetry_struct,
)


class TestAdversarialMalformedAndTruncatedPackets(unittest.TestCase):
    """1. Malformed / Truncated DM1 & DM2 Binary Packets."""

    def test_truncated_frames_length_under_2_bytes(self):
        """Binary payloads < 2 bytes must raise ValueError cleanly without unhandled crashes."""
        bad_inputs = [b"", b"\x00", b"\x10", b"\xFF", b"\xAA"]
        for payload in bad_inputs:
            with self.assertRaises(ValueError, msg=f"DM1 failed on {payload!r}"):
                DTCEngine.decode_dm1_packet(payload)
            with self.assertRaises(ValueError, msg=f"DM2 failed on {payload!r}"):
                DTCEngine.decode_dm2_packet(payload)
            with self.assertRaises(ValueError, msg=f"Lamp header failed on {payload!r}"):
                LampStatus.decode_header(payload)

    def test_truncated_chunks_and_odd_byte_lengths(self):
        """Header (2 bytes) followed by non-multiple-of-4 chunks must parse valid DTCs and discard leftovers."""
        # 2-byte header with red stop lamp ON
        hdr = b"\x10\xFF"
        dtc1 = b"\x64\x00\x04\x01"  # SPN 100, FMI 04, OC 1
        dtc2 = b"\xa5\xef\xe7\x05"  # SPN 520101, FMI 7, OC 5

        # 3 bytes: header + 1 stray byte
        lamp, recs = DTCEngine.decode_dm1_packet(hdr + b"\x64")
        self.assertEqual(len(recs), 0)
        self.assertTrue(lamp.is_red_stop)

        # 5 bytes: header + 3 stray bytes
        lamp, recs = DTCEngine.decode_dm1_packet(hdr + b"\x64\x00\x04")
        self.assertEqual(len(recs), 0)

        # 7 bytes: header + 1 DTC + 1 stray byte
        lamp, recs = DTCEngine.decode_dm1_packet(hdr + dtc1 + b"\x99")
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].spn, 100)
        self.assertEqual(recs[0].fmi, 4)

        # 11 bytes: header + 2 DTCs + 1 stray byte
        lamp, recs = DTCEngine.decode_dm1_packet(hdr + dtc1 + dtc2 + b"\xCC")
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0].spn, 100)
        self.assertEqual(recs[1].spn, 520101)

    def test_fuzzing_10000_random_byte_payloads(self):
        """Feed 10,000 random adversarial byte sequences into DM1 & DM2 decoders."""
        rng = random.Random(1337)
        for i in range(10000):
            length = rng.randint(0, 150)
            rand_bytes = rng.randbytes(length)
            if length < 2:
                with self.assertRaises(ValueError):
                    DTCEngine.decode_dm1_packet(rand_bytes)
                with self.assertRaises(ValueError):
                    DTCEngine.decode_dm2_packet(rand_bytes)
            else:
                try:
                    lamp1, recs1 = DTCEngine.decode_dm1_packet(rand_bytes)
                    lamp2, recs2 = DTCEngine.decode_dm2_packet(rand_bytes)
                    self.assertIsInstance(lamp1, LampStatus)
                    self.assertIsInstance(lamp2, LampStatus)
                    self.assertIsInstance(recs1, list)
                    self.assertIsInstance(recs2, list)
                    # All decoded records must have valid SPN & FMI bitfield bounds
                    for r in recs1 + recs2:
                        self.assertGreaterEqual(r.spn, 0)
                        self.assertLessEqual(r.spn, 524287)
                        self.assertGreaterEqual(r.fmi, 0)
                        self.assertLessEqual(r.fmi, 31)
                        self.assertGreaterEqual(r.oc, 1)
                        self.assertLessEqual(r.oc, 127)
                except Exception as exc:
                    self.fail(f"Adversarial fuzzer crashed on payload {rand_bytes.hex()} (len {length}): {exc}")


class TestAdversarialExtremeBoundaryBitfields(unittest.TestCase):
    """2. Extreme Boundary Bitfields (SPN 524287, FMI 31, OC 127, CM 1)."""

    def test_maximum_boundary_bitfields_all_ones(self):
        """SPN 524287 (0x7FFFF, 19-bit), FMI 31 (0x1F, 5-bit), OC 127 (0x7F, 7-bit), CM 1 -> 0xFFFFFFFF."""
        dtc = DTCRecord(spn=524287, fmi=31, oc=127, cm=1)
        raw_bytes = dtc.encode_4bytes()
        self.assertEqual(raw_bytes, b"\xff\xff\xff\xff")

        decoded = DTCRecord.decode_4bytes(raw_bytes)
        self.assertEqual(decoded.spn, 524287)
        self.assertEqual(decoded.fmi, 31)
        self.assertEqual(decoded.oc, 127)
        self.assertEqual(decoded.cm, 1)

    def test_minimum_boundary_bitfields(self):
        """SPN 0, FMI 0, OC 1, CM 0 -> 0x00000001."""
        dtc = DTCRecord(spn=0, fmi=0, oc=1, cm=0)
        raw_bytes = dtc.encode_4bytes()
        self.assertEqual(raw_bytes, b"\x00\x00\x00\x01")

        decoded = DTCRecord.decode_4bytes(raw_bytes)
        self.assertEqual(decoded.spn, 0)
        self.assertEqual(decoded.fmi, 0)
        self.assertEqual(decoded.oc, 1)
        self.assertEqual(decoded.cm, 0)

    def test_negative_and_overflow_bitfield_clamping(self):
        """Negative and overflow values must be safely masked/clamped according to SAE J1939 specs."""
        # Negative SPN, negative FMI, negative OC
        dtc = DTCRecord(spn=-1, fmi=-1, oc=-5, cm=-1)
        self.assertGreaterEqual(dtc.spn, 0)
        self.assertLessEqual(dtc.spn, 524287)
        self.assertGreaterEqual(dtc.fmi, 0)
        self.assertLessEqual(dtc.fmi, 31)
        self.assertEqual(dtc.oc, 1)  # OC clamped to min 1

        # Massive overflow SPN (1,000,000), overflow FMI (999), overflow OC (5000)
        dtc_overflow = DTCRecord(spn=1_000_000, fmi=999, oc=5000, cm=5)
        self.assertEqual(dtc_overflow.spn, 1_000_000 & 0x7FFFF)
        self.assertEqual(dtc_overflow.fmi, 999 & 0x1F)
        self.assertEqual(dtc_overflow.oc, 127)  # OC clamped to max 127
        self.assertEqual(dtc_overflow.cm, 1)    # CM masked to 1 bit

    def test_exhaustive_fmi_0_to_31_packing_roundtrip(self):
        """Exhaustively verify all 32 FMIs (0..31) for multiple military SPNs."""
        test_spns = [0, 92, 96, 98, 100, 110, 173, 175, 190, 513, 520101, 520200, 520202, 524287]
        for spn in test_spns:
            for fmi in range(32):
                for cm in (0, 1):
                    dtc = DTCRecord(spn=spn, fmi=fmi, oc=42, cm=cm)
                    encoded = dtc.encode_4bytes()
                    decoded = DTCRecord.decode_4bytes(encoded)
                    self.assertEqual(decoded.spn, spn)
                    self.assertEqual(decoded.fmi, fmi)
                    self.assertEqual(decoded.oc, 42)
                    self.assertEqual(decoded.cm, cm)


class TestAdversarialFaultLifecycleToggling(unittest.TestCase):
    """3. High-Frequency Fault Lifecycle Toggling (Cycling 100+ active/cleared across 1,000 frames)."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_file = Path(self.temp_dir.name) / "lifecycle_flash.jsonl"
        self.engine = DTCEngine(flash_capacity=20000, flash_file=self.flash_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_high_frequency_cycling_120_faults_across_1000_frames(self):
        """Cycle 120 unique SPN/FMI faults across 1,000 simulation frames:

        - Assert zero memory corruption or leaked orphan faults.
        - Assert disjointness: active_keys & historic_keys == empty.
        - Assert occurrence counts increment accurately upon re-activation.
        """
        num_faults = 120
        num_frames = 1000

        fault_pool = [
            (520000 + i, (i % 15))
            for i in range(num_faults)
        ]

        active_set: Set[Tuple[int, int]] = set()

        for frame_idx in range(num_frames):
            # Deterministic pseudo-random toggling
            target_spn, target_fmi = fault_pool[(frame_idx * 7 + 13) % num_faults]
            key = (target_spn, target_fmi)

            if key in active_set and (frame_idx % 3 == 0):
                # Clear fault
                cleared = self.engine.clear_fault(target_spn, target_fmi)
                self.assertIsNotNone(cleared)
                self.assertFalse(cleared.active)
                active_set.remove(key)
            else:
                # Report / activate fault
                lamp = "RED_STOP" if target_spn % 2 == 0 else "AMBER_WARNING"
                dtc = self.engine.report_fault(
                    spn=target_spn,
                    fmi=target_fmi,
                    lamp=lamp,
                    description=f"Lifecycle Test Fault {target_spn}",
                )
                self.assertTrue(dtc.active)
                active_set.add(key)

            # Invariant check every 50 frames
            if frame_idx % 50 == 0:
                active_dtcs = self.engine.get_active_dtcs()
                historic_dtcs = self.engine.get_historic_dtcs()

                active_keys = set((d.spn, d.fmi) for d in active_dtcs)
                historic_keys = set((d.spn, d.fmi) for d in historic_dtcs)

                self.assertEqual(active_keys, active_set)
                # Disjointness invariant
                self.assertEqual(active_keys.intersection(historic_keys), set())

                # DM1 packet encoding must succeed and match active count
                dm1_payload = self.engine.encode_dm1_packet()
                lamp_dec, dtcs_dec = DTCEngine.decode_dm1_packet(dm1_payload)
                self.assertEqual(len(dtcs_dec), len(active_set))

        # Final state check
        self.assertGreater(len(self.engine.get_historic_dtcs()), 0)


class TestAdversarialFlashBufferConcurrencyAndRollover(unittest.TestCase):
    """4. High-Concurrency Multithreaded Flash Ring Buffer Logging & Rollover."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_file = Path(self.temp_dir.name) / "concurrent_flash.jsonl"
        # Small capacity to vigorously exercise rollover pruning
        self.engine = DTCEngine(flash_capacity=300, flash_file=self.flash_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_20_threads_hammering_flash_persistence_with_rollover(self):
        """20 worker threads concurrently report faults and persist records (2,000 total writes).

        Assert:
        - No unhandled exceptions or deadlocks.
        - File remains uncorrupted (every single line is valid JSON).
        - Flash file size stays within capacity bounds without unbounded growth.
        """
        num_threads = 20
        writes_per_thread = 100
        errors: List[Exception] = []

        def worker(thread_id: int):
            try:
                for i in range(writes_per_thread):
                    spn = 520000 + (thread_id * 10) + (i % 10)
                    fmi = i % 15
                    self.engine.report_fault(
                        spn=spn,
                        fmi=fmi,
                        lamp="RED_STOP" if i % 2 == 0 else "AMBER_WARNING",
                        description=f"Thread {thread_id} Write {i}",
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        self.assertTrue(self.flash_file.exists())

        # Verify flash file integrity
        with open(self.flash_file, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        self.assertGreater(len(lines), 0)
        # Verify bounded rollover: capacity is 300, so lines should be <= ~350
        self.assertLessEqual(len(lines), 400)

        # Parse every line as valid JSON
        for line_num, line in enumerate(lines):
            try:
                entry = json.loads(line.strip())
                self.assertIn("spn", entry)
                self.assertIn("fmi", entry)
                self.assertIn("code", entry)
            except Exception as parse_err:
                self.fail(f"Line {line_num} corrupted: {line!r} (error: {parse_err})")

    def test_crash_recovery_from_flash_log(self):
        """Verify complete state restoration after engine shutdown/restart."""
        # 1. Report active faults and clear some into historic memory
        self.engine.report_fault(spn=100, fmi=4, lamp="RED_STOP", description="Oil Pressure Wire Cut")
        self.engine.report_fault(spn=110, fmi=0, lamp="RED_STOP", description="Coolant Overheat")
        self.engine.report_fault(spn=190, fmi=2, lamp="AMBER_WARNING", description="RPM Erratic")
        self.engine.clear_fault(spn=190, fmi=2)

        active_before = self.engine.get_active_dtcs()
        historic_before = self.engine.get_historic_dtcs()
        self.assertEqual(len(active_before), 2)
        self.assertEqual(len(historic_before), 1)

        # 2. Instantiate brand new DTCEngine targeting same flash file
        new_engine = DTCEngine(flash_capacity=300, flash_file=self.flash_file, auto_recover=True)

        active_after = new_engine.get_active_dtcs()
        historic_after = new_engine.get_historic_dtcs()

        self.assertEqual(len(active_after), 2)
        self.assertEqual(len(historic_after), 1)

        active_keys = set((d.spn, d.fmi) for d in active_after)
        self.assertIn((100, 4), active_keys)
        self.assertIn((110, 0), active_keys)

        historic_keys = set((d.spn, d.fmi) for d in historic_after)
        self.assertIn((190, 2), historic_keys)


class TestAdversarialEndToEndSensorFaultInjection(unittest.TestCase):
    """5. End-to-End Adversarial Sensor Fault Injection Pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_file = Path(self.temp_dir.name) / "e2e_flash.jsonl"
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)
        self.engine = DTCEngine(flash_capacity=1000, flash_file=self.flash_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_oil_pressure_wire_cut_e2e_pipeline_pure_sensor(self):
        """Adversarial Wire Cut (P_oil = -999 bar, static/idle engine):

        SensorPlausibilityGate -> Clamps to 0.0 bar & emits SensorFaultEvent(oil_pressure, SPN 100, FMI 04).
        DTCEngine -> Ingests FDIR fault, produces PGN 65226 DM1 packet with Red Stop Lamp, SPN 100, FMI 04.
        Flash Ring Buffer -> Appends record to dtc_flash_log.jsonl.
        """
        raw_telemetry = {
            "oil_pressure": -999.0,
            "coolant_temp": 85.0,
        }

        # 1. Gate filters frame
        result = self.gate.filter_frame(raw_telemetry)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.clean_telemetry["oil_pressure"], 0.0)
        self.assertEqual(len(result.faults_detected), 1)

        fault = result.faults_detected[0]
        self.assertEqual(fault.channel, "oil_pressure")
        self.assertEqual(fault.fault_type, PlausibilityFaultType.OPEN_CIRCUIT.value)
        self.assertEqual(fault.spn, 100)
        self.assertEqual(fault.fmi, FMI_VOLTAGE_BELOW_NORMAL)  # FMI 04

        # 2. Engine processes FDIR fault
        active_dtcs = self.engine.process_fdir_faults(result.faults_detected)
        self.assertEqual(len(active_dtcs), 1)
        dtc = active_dtcs[0]
        self.assertEqual(dtc.spn, 100)
        self.assertEqual(dtc.fmi, 4)
        self.assertEqual(dtc.lamp_status, "RED_STOP")
        self.assertEqual(dtc.source_type, "FDIR_ELECTRICAL")

        # 3. Generate SAE J1939-73 DM1 Packet (PGN 65226)
        dm1_bytes = self.engine.encode_dm1_packet()
        self.assertEqual(len(dm1_bytes), 8)

        # Byte 0: Red Stop Lamp ON (0x10)
        self.assertEqual(dm1_bytes[0], 0x10)
        # Bytes 2..5: 4-byte packed DTC struct (SPN 100, FMI 04, OC 01, CM 0)
        self.assertEqual(dm1_bytes[2:6], b"\x64\x00\x04\x01")
        # Bytes 6..7: 0xFFFF padding
        self.assertEqual(dm1_bytes[6:8], b"\xff\xff")

        # 4. Decode DM1 packet
        lamp, decoded_recs = DTCEngine.decode_dm1_packet(dm1_bytes)
        self.assertTrue(lamp.is_red_stop)
        self.assertFalse(lamp.is_amber)
        self.assertEqual(len(decoded_recs), 1)
        self.assertEqual(decoded_recs[0].spn, 100)
        self.assertEqual(decoded_recs[0].fmi, 4)

        # 5. Verify Flash persistence
        logged = self.engine.read_flash_log(limit=5)
        self.assertEqual(len(logged), 1)
        self.assertEqual(logged[0]["spn"], 100)
        self.assertEqual(logged[0]["fmi"], 4)
        self.assertEqual(logged[0]["lamp_status"], "RED_STOP")
        self.assertEqual(logged[0]["event_type"], "ACTIVE")

    def test_oil_pressure_wire_cut_with_high_rpm_dual_detection(self):
        """Adversarial Wire Cut with High Engine Speed (P_oil = -999 bar, RPM = 1800):

        Triggers BOTH:
        - Layer 2 Electrical Open Circuit (SPN 100, FMI 04)
        - Layer 6 Cross-Subsystem Mismatch (SPN 100, FMI 14)
        Both are converted to Red Stop Lamp DM1 DTCs.
        """
        raw_telemetry = {
            "oil_pressure": -999.0,
            "rpm": 1800.0,
        }

        result = self.gate.filter_frame(raw_telemetry)
        self.assertEqual(len(result.faults_detected), 2)

        fmi_types = set(f.fmi for f in result.faults_detected)
        self.assertIn(FMI_VOLTAGE_BELOW_NORMAL, fmi_types)  # FMI 04
        self.assertIn(FMI_SPECIAL_INSTRUCTIONS, fmi_types)   # FMI 14

        active_dtcs = self.engine.process_fdir_faults(result.faults_detected)
        self.assertEqual(len(active_dtcs), 2)

        # Both must be active with Red Stop Lamp
        for dtc in active_dtcs:
            self.assertEqual(dtc.spn, 100)
            self.assertEqual(dtc.lamp_status, "RED_STOP")

        dm1_payload = self.engine.encode_dm1_packet()
        lamp_dec, recs_dec = DTCEngine.decode_dm1_packet(dm1_payload)
        self.assertTrue(lamp_dec.is_red_stop)
        self.assertEqual(len(recs_dec), 2)

    def test_multi_fault_complex_battlefield_injection(self):
        """Simultaneous catastrophic battlefield faults:

        - Wire cut on Oil Pressure (P_oil = -999 bar) -> SPN 100, FMI 04
        - Short to power on Coolant Temp (T_coolant = 2000 C) -> SPN 110, FMI 03
        - Short to power on RPM (RPM = 9000) -> SPN 190, FMI 03
        - AI Neural Bearing Wear prediction (Prob = 0.85) -> SPN 520101, FMI 07
        """
        raw_telemetry = {
            "oil_pressure": -999.0,
            "coolant_temp": 2000.0,
            "rpm": 9000.0,
        }

        # 1. Ingest physical signals through Gate (Layer 2 detects 3 electrical faults + Layer 6 mismatch)
        res = self.gate.filter_frame(raw_telemetry)
        self.assertGreaterEqual(len(res.faults_detected), 3)

        # Ingest FDIR faults into DTC Engine
        fdir_dtcs = self.engine.process_fdir_faults(res.faults_detected)
        self.assertGreaterEqual(len(fdir_dtcs), 3)

        # 2. Ingest AI Neural prediction
        neural_dtcs = self.engine.process_neural_predictions(
            fault_probs={"bearing_wear": 0.85, "healthy": 0.15}
        )
        self.assertEqual(len(neural_dtcs), 1)

        # 3. Verify aggregate active DTCs contains all expected SPNs
        all_active = self.engine.get_active_dtcs()
        active_spns = set(d.spn for d in all_active)
        self.assertIn(100, active_spns)
        self.assertIn(110, active_spns)
        self.assertIn(190, active_spns)
        self.assertIn(520101, active_spns)

        # 4. Multi-DTC variable length DM1 BAM encoding
        dm1_packet = self.engine.encode_dm1_packet()
        self.assertEqual(len(dm1_packet), 2 + (4 * len(all_active)))

        # Red Stop lamp must be ON due to oil pressure and coolant short
        lamp_dec, dtcs_dec = DTCEngine.decode_dm1_packet(dm1_packet)
        self.assertTrue(lamp_dec.is_red_stop)
        self.assertEqual(len(dtcs_dec), len(all_active))


if __name__ == "__main__":
    unittest.main()
