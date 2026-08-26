"""Adversarial Testing & Protocol Fuzzing Suite for SAE J1939-73 DTC Engine.

Author: challenger_m3_1 (Adversarial Testing & Protocol Fuzzing Specialist)
Target: telemetry_gateway/dtc_engine.py

Stress Test Dimensions:
1. Malformed & Truncated DM1/DM2 binary frames, fuzzing with random byte streams.
2. Maximum boundary bitfields (SPN 524287, FMI 31, OC 127, CM 1) and negative/overflow clamping.
3. Rapid occurrence toggling (cycling 100+ faults active/cleared across 1,000 frames) and state consistency.
4. High-concurrency multithreaded flash persistence (16+ threads hammering persist_to_flash with rollover).
"""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import random
import struct
import tempfile
import threading
import time
import unittest
from typing import Any, Dict, List, Set, Tuple

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
    FMI_MECHANICAL_SYSTEM_FAIL,
    FMI_ABNORMAL_RATE_OF_CHANGE,
    FMI_SPECIAL_INSTRUCTIONS,
    CHANNEL_TO_SPN_MAP,
    NEURAL_FAULT_CLASS_MAP,
)


class TestAdversarialMalformedFrames(unittest.TestCase):
    """1. Adversarial Malformed & Truncated Frame Fuzzing."""

    def test_truncated_frames_length_under_2_bytes(self):
        """Frames under 2 bytes must raise ValueError cleanly without unhandled crash."""
        truncated_cases = [b"", b"\x00", b"\xFF", b"\x55"]
        for case in truncated_cases:
            with self.assertRaises(ValueError, msg=f"Failed to raise ValueError on {case!r}"):
                DTCEngine.decode_dm1_packet(case)

            with self.assertRaises(ValueError, msg=f"Failed to raise ValueError on {case!r}"):
                DTCEngine.decode_dm2_packet(case)

            with self.assertRaises(ValueError, msg=f"Failed to raise ValueError on {case!r}"):
                LampStatus.decode_header(case)

    def test_truncated_dtc_chunks_in_payload(self):
        """Header present (2 bytes) but followed by partial DTC bytes (1, 2, or 3 trailing bytes)."""
        header = b"\x00\xFF"  # valid lamp header
        partial_payloads = [
            header + b"\x64",              # 3 bytes (1 partial DTC byte)
            header + b"\x64\x00",          # 4 bytes (2 partial DTC bytes)
            header + b"\x64\x00\x04",      # 5 bytes (3 partial DTC bytes)
            header + b"\x64\x00\x04\x01\x10",  # 7 bytes (1 valid DTC + 1 partial byte)
            header + b"\x64\x00\x04\x01\x10\x20",  # 8 bytes (1 valid DTC + 2 partial bytes)
            header + b"\x64\x00\x04\x01\x10\x20\x30",  # 9 bytes (1 valid DTC + 3 partial bytes)
        ]

        for payload in partial_payloads:
            lamp_status, records = DTCEngine.decode_dm1_packet(payload)
            self.assertIsInstance(lamp_status, LampStatus)
            self.assertIsInstance(records, list)
            # Ensure partial trailing bytes do not crash parser and are safely discarded
            if len(payload) < 6:
                self.assertEqual(len(records), 0)
            elif 6 <= len(payload) < 10:
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0].spn, 100)
                self.assertEqual(records[0].fmi, 4)

    def test_all_zeros_and_all_ff_padding(self):
        """Empty frames filled with 0x00 and 0xFF padding must return 0 DTCs."""
        zero_payload = b"\x00\xFF" + b"\x00\x00\x00\x00\xFF\xFF"
        lamp, records = DTCEngine.decode_dm1_packet(zero_payload)
        self.assertEqual(len(records), 0)

        ff_payload = b"\xFF\xFF" + b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF"
        lamp, records = DTCEngine.decode_dm1_packet(ff_payload)
        self.assertEqual(len(records), 0)

    def test_lamp_status_header_exhaustive_2byte_fuzz(self):
        """Exhaustively decode all 65,536 possible 2-byte header combinations.

        Assert zero crashes, valid 2-bit field extractions, and valid re-encoding.
        """
        for b1 in range(256):
            for b2 in range(256):
                header = bytes([b1, b2])
                status = LampStatus.decode_header(header)
                self.assertEqual(status.mil, (b1 >> 6) & 0x03)
                self.assertEqual(status.red_stop, (b1 >> 4) & 0x03)
                self.assertEqual(status.amber_warning, (b1 >> 2) & 0x03)
                self.assertEqual(status.protect, b1 & 0x03)
                self.assertEqual(status.mil_flash, (b2 >> 6) & 0x03)
                self.assertEqual(status.red_stop_flash, (b2 >> 4) & 0x03)
                self.assertEqual(status.amber_warning_flash, (b2 >> 2) & 0x03)
                self.assertEqual(status.protect_flash, b2 & 0x03)

    def test_random_binary_fuzzing_dm1_dm2(self):
        """Feed 5,000 random adversarial byte sequences into decode_dm1_packet and decode_dm2_packet.

        Must never raise uncaught exceptions (only ValueError when len < 2).
        """
        rng = random.Random(42)
        for _ in range(5000):
            length = rng.randint(0, 128)
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
                except Exception as exc:
                    self.fail(f"Random fuzz payload {rand_bytes.hex()} crashed decoder: {exc}")


class TestAdversarialBoundaryBitfields(unittest.TestCase):
    """2. Maximum Boundary Bitfields & Negative / Overflow Input Hardening."""

    def test_maximum_boundary_bitfields(self):
        """SPN 524287 (19-bit max = 0x7FFFF), FMI 31 (5-bit max = 0x1F), OC 127 (7-bit max = 0x7F), CM 1."""
        dtc = DTCRecord(spn=524287, fmi=31, oc=127, cm=1)
        packed = dtc.encode_4bytes()
        self.assertEqual(len(packed), 4)
        self.assertEqual(packed, b"\xff\xff\xff\xff")

        decoded = DTCRecord.decode_4bytes(packed)
        self.assertEqual(decoded.spn, 524287)
        self.assertEqual(decoded.fmi, 31)
        self.assertEqual(decoded.oc, 127)
        self.assertEqual(decoded.cm, 1)

    def test_minimum_boundary_bitfields(self):
        """SPN 0, FMI 0, OC 1 (clamped min), CM 0."""
        dtc = DTCRecord(spn=0, fmi=0, oc=0, cm=0)
        self.assertEqual(dtc.oc, 1)  # OC 0 clamped to 1 in J1939
        packed = dtc.encode_4bytes()
        self.assertEqual(packed, b"\x00\x00\x00\x01")

        decoded = DTCRecord.decode_4bytes(packed)
        self.assertEqual(decoded.spn, 0)
        self.assertEqual(decoded.fmi, 0)
        self.assertEqual(decoded.oc, 1)
        self.assertEqual(decoded.cm, 0)

    def test_negative_and_overflow_bitfields_clamping(self):
        """Test defensive behavior when negative or overflowing integer values are injected."""
        test_cases = [
            # (input_spn, input_fmi, input_oc, input_cm, expected_spn, expected_fmi, expected_oc, expected_cm)
            (-1, -1, -5, -1, 0x7FFFF, 0x1F, 1, 1),
            (1000000, 255, 9999, 5, 1000000 & 0x7FFFF, 255 & 0x1F, 127, 5 & 0x01),
            (0xFFFFFFFF, 0xFF, 0, 0x10, 0x7FFFF, 0x1F, 1, 0),
            (524288, 32, 128, 2, 0, 0, 127, 0),
        ]

        for in_spn, in_fmi, in_oc, in_cm, exp_spn, exp_fmi, exp_oc, exp_cm in test_cases:
            dtc = DTCRecord(spn=in_spn, fmi=in_fmi, oc=in_oc, cm=in_cm)
            self.assertEqual(dtc.spn, exp_spn)
            self.assertEqual(dtc.fmi, exp_fmi)
            self.assertEqual(dtc.oc, exp_oc)
            self.assertEqual(dtc.cm, exp_cm)

            packed = dtc.encode_4bytes()
            decoded = DTCRecord.decode_4bytes(packed)
            self.assertEqual(decoded.spn, exp_spn)
            self.assertEqual(decoded.fmi, exp_fmi)
            self.assertEqual(decoded.oc, exp_oc)
            self.assertEqual(decoded.cm, exp_cm)

    def test_randomized_bitfield_roundtrip_property_test(self):
        """Property-based verification of 2,000 random SPN/FMI/OC/CM tuples round-tripping."""
        rng = random.Random(1337)
        for _ in range(2000):
            spn = rng.randint(0, 0x7FFFF)
            fmi = rng.randint(0, 0x1F)
            oc = rng.randint(1, 127)
            cm = rng.randint(0, 1)

            dtc = DTCRecord(spn=spn, fmi=fmi, oc=oc, cm=cm)
            packed = dtc.encode_4bytes()
            decoded = DTCRecord.decode_4bytes(packed)

            self.assertEqual(decoded.spn, spn)
            self.assertEqual(decoded.fmi, fmi)
            self.assertEqual(decoded.oc, oc)
            self.assertEqual(decoded.cm, cm)


class TestAdversarialOccurrenceToggling(unittest.TestCase):
    """3. Rapid Occurrence Toggling & State Machine Stress Testing."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_file = Path(self.temp_dir.name) / "test_stress_flash.jsonl"
        self.engine = DTCEngine(
            flash_capacity=5000,
            flash_file=self.flash_file,
            vehicle_id="CVRDE_TEST_HARNESS",
            auto_recover=False,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_rapid_toggling_100_faults_across_1000_frames(self):
        """Cycle 100+ distinct faults active and cleared across 1,000 simulated frames."""
        rng = random.Random(999)
        # Create 120 unique (spn, fmi) fault identities
        fault_pool: List[Tuple[int, int]] = []
        for i in range(120):
            spn = 520000 + i
            fmi = i % 15
            fault_pool.append((spn, fmi))

        active_set: Set[Tuple[int, int]] = set()

        for frame_idx in range(1000):
            # Select random subset to toggle in this frame
            to_activate = rng.sample(fault_pool, k=2)
            to_clear = rng.sample(fault_pool, k=2)

            # Ingest activations
            for spn, fmi in to_activate:
                self.engine.report_fault(
                    spn=spn,
                    fmi=fmi,
                    lamp="RED_STOP" if fmi == 0 else "AMBER_WARNING",
                    description=f"Stress test fault SPN {spn} FMI {fmi}",
                    source_type="MANUAL",
                )
                active_set.add((spn, fmi))

            # Ingest clearings
            for spn, fmi in to_clear:
                self.engine.clear_fault(spn, fmi)
                active_set.discard((spn, fmi))

            # Periodic invariant assertions
            if frame_idx % 100 == 0:
                current_active = self.engine.get_active_dtcs()
                current_historic = self.engine.get_historic_dtcs()

                active_keys = set((d.spn, d.fmi) for d in current_active)
                historic_keys = set((d.spn, d.fmi) for d in current_historic)

                # 1. No overlap between active and historic
                overlap = active_keys.intersection(historic_keys)
                self.assertEqual(len(overlap), 0, f"Overlap between active and historic DTCs: {overlap}")

                # 2. Active keys must match our ground-truth tracking
                self.assertEqual(active_keys, active_set)

                # 3. All active DTCs have active=True, historic have active=False
                self.assertTrue(all(d.active for d in current_active))
                self.assertTrue(all(not d.active for d in current_historic))

        # Final verification that 100+ unique faults were engaged across 1000 frames
        all_tracked = set((d.spn, d.fmi) for d in self.engine.get_active_dtcs()) | set((d.spn, d.fmi) for d in self.engine.get_historic_dtcs())
        self.assertGreaterEqual(len(all_tracked), 100)

    def test_multi_dtc_large_bam_payload_encoding_and_decoding(self):
        """Encode and decode DM1 with 120 simultaneously active faults (multi-packet BAM format)."""
        for i in range(120):
            self.engine.report_fault(
                spn=520100 + i,
                fmi=(i % 14) + 1,
                lamp="AMBER_WARNING",
                description=f"Multi fault {i}",
            )

        active = self.engine.get_active_dtcs()
        self.assertEqual(len(active), 120)

        # Encode DM1
        dm1_payload = self.engine.encode_dm1_packet()
        expected_len = 2 + (120 * 4)  # 2-byte header + 480 bytes = 482 bytes
        self.assertEqual(len(dm1_payload), expected_len)

        # Decode DM1
        lamp, decoded_recs = DTCEngine.decode_dm1_packet(dm1_payload)
        self.assertTrue(lamp.has_active_lamp)
        self.assertEqual(len(decoded_recs), 120)

        # Verify exact preservation of all 120 SPN/FMIs
        original_tuples = [(d.spn, d.fmi) for d in active]
        decoded_tuples = [(d.spn, d.fmi) for d in decoded_recs]
        self.assertEqual(original_tuples, decoded_tuples)

    def test_occurrence_count_increment_and_saturation_at_127(self):
        """Verify occurrence count increments on reactivation and clamps strictly at 127."""
        spn = 520200
        fmi = 4

        dtc = self.engine.report_fault(spn=spn, fmi=fmi)
        self.assertEqual(dtc.oc, 1)

        # Simulate 150 cycles of clear and re-activate
        for cycle in range(2, 150):
            self.engine.clear_fault(spn, fmi)
            dtc = self.engine.report_fault(spn=spn, fmi=fmi)
            expected_oc = min(cycle, 127)
            self.assertEqual(dtc.oc, expected_oc, f"Cycle {cycle} OC mismatch: got {dtc.oc}, expected {expected_oc}")

        self.assertEqual(dtc.oc, 127)


class TestAdversarialFlashPersistenceConcurrency(unittest.TestCase):
    """4. High-Concurrency Multithreaded Flash Persistence Stress Test."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_file = Path(self.temp_dir.name) / "concurrent_flash_log.jsonl"
        # Set capacity 200 to test circular rollover under high thread contention
        self.engine = DTCEngine(
            flash_capacity=200,
            flash_file=self.flash_file,
            vehicle_id="CVRDE_CONCURRENCY_TEST",
            auto_recover=False,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_multithreaded_high_concurrency_persistence_with_rollover(self):
        """Hammer persist_to_flash from 16 concurrent threads while reading and recovering."""
        num_threads = 16
        writes_per_thread = 50  # Total 800 concurrent write operations
        errors: List[Exception] = []

        def worker_write(thread_id: int):
            try:
                for i in range(writes_per_thread):
                    spn = 520000 + (thread_id * 50) + i
                    fmi = (thread_id + i) % 15
                    record = DTCRecord(
                        spn=spn,
                        fmi=fmi,
                        oc=(i % 120) + 1,
                        lamp_status="RED_STOP" if i % 2 == 0 else "AMBER_WARNING",
                        description=f"Thread {thread_id} Iteration {i}",
                        channel="susp_load_kN",
                        subsystem="suspension",
                    )
                    event_type = "ACTIVE" if i % 3 != 0 else "CLEARED"
                    self.engine.persist_to_flash(record, event_type=event_type)
            except Exception as exc:
                errors.append(exc)

        def worker_read():
            try:
                for _ in range(30):
                    logs = self.engine.read_flash_log(limit=50)
                    self.assertIsInstance(logs, list)
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        threads: List[threading.Thread] = []
        for t_id in range(num_threads):
            t = threading.Thread(target=worker_write, args=(t_id,))
            threads.append(t)

        # Add 4 concurrent reader threads
        for _ in range(4):
            t = threading.Thread(target=worker_read)
            threads.append(t)

        # Launch all threads concurrently
        for t in threads:
            t.start()

        # Await completion
        for t in threads:
            t.join()

        # 1. Assert zero thread exceptions
        self.assertEqual(len(errors), 0, f"Concurrency worker threw exceptions: {errors}")

        # 2. Assert flash log exists and adheres to ring buffer capacity limit
        self.assertTrue(self.flash_file.exists())
        with open(self.flash_file, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        self.assertGreater(len(lines), 0)
        # Ring buffer capacity is 200; lines after rollover should be bounded
        self.assertLessEqual(len(lines), 200 + 50)

        # 3. Assert every single line is valid uncorrupted JSON
        for idx, line in enumerate(lines):
            try:
                data = json.loads(line)
                self.assertIn("spn", data)
                self.assertIn("fmi", data)
                self.assertIn("event_type", data)
                self.assertIn("hex_4bytes", data)
            except Exception as exc:
                self.fail(f"Line {idx} in flash log was corrupted JSON: {line!r} ({exc})")

        # 4. Assert recover_from_flash executes cleanly
        recover_engine = DTCEngine(
            flash_capacity=200,
            flash_file=self.flash_file,
            vehicle_id="CVRDE_CONCURRENCY_TEST",
            auto_recover=True,
        )
        active_recovered = recover_engine.get_active_dtcs()
        historic_recovered = recover_engine.get_historic_dtcs()
        self.assertIsInstance(active_recovered, list)
        self.assertIsInstance(historic_recovered, list)
        self.assertGreater(len(active_recovered) + len(historic_recovered), 0)


if __name__ == "__main__":
    unittest.main()
