"""Comprehensive Unit & SAE J1939-73 Protocol Compliance Test Suite for DTC Engine.

Tests:
1. SAE J1939 4-byte packed DTC struct binary bitfield packing and unpacking.
2. 2-byte Lamp Status header encoding, decoding, and lamp state aggregation.
3. DM1 (PGN 65226) active and DM2 (PGN 65227) historic payload binary encoders/decoders.
4. Hook 1: FDIR sensor plausibility fault translation (wire cuts, shorts, slew, stuck-at, EMI, dual mismatch).
5. Hook 2: Neural prognostics (13-class softmax, 8-subsystem RUL regression, CVRDE degradation).
6. Lifecycle management (active -> historic -> reactivate, occurrence count increments).
7. Thread-safe on-disk circular flash ring buffer persistence, rollover, and recovery.
8. End-to-end integration: SensorPlausibilityGate -> DTCEngine pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
from typing import Any, Dict, List

from telemetry_gateway.dtc_engine import (
    DTCEngine,
    DTCRecord,
    DTC,
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


class TestDTCRecordBinaryBitfields(unittest.TestCase):
    """1. Test SAE J1939 4-Byte Packed DTC Struct Binary Bitfield Encoding & Decoding."""

    def test_standard_oil_pressure_wire_cut_4byte_packing(self):
        """SPN 100, FMI 04, CM 0, OC 1 -> exact byte matching."""
        dtc = DTCRecord(spn=100, fmi=4, oc=1, cm=0)
        packed = dtc.encode_4bytes()

        self.assertEqual(len(packed), 4)
        # Byte 1: 100 & 0xFF = 0x64
        # Byte 2: (100 >> 8) & 0xFF = 0x00
        # Byte 3: ((100 >> 16) & 0x07) << 5 | (4 & 0x1F) = 0x00 | 0x04 = 0x04
        # Byte 4: (0 << 7) | (1 & 0x7F) = 0x01
        self.assertEqual(packed, b"\x64\x00\x04\x01")

        # Decode back
        decoded = DTCRecord.decode_4bytes(packed)
        self.assertEqual(decoded.spn, 100)
        self.assertEqual(decoded.fmi, 4)
        self.assertEqual(decoded.cm, 0)
        self.assertEqual(decoded.oc, 1)

    def test_high_19bit_military_spn_packing(self):
        """SPN 520101 (Bearing Vibration = 0x07EFA5), FMI 7, CM 0, OC 5."""
        dtc = DTCRecord(spn=520101, fmi=7, oc=5, cm=0)
        packed = dtc.encode_4bytes()

        self.assertEqual(len(packed), 4)
        # 520101 = 0x07EFA5
        # Byte 1: 0xA5
        # Byte 2: 0xEF
        # Byte 3: (0x07 << 5) | 7 = 0xE0 | 7 = 0xE7
        # Byte 4: 0x05
        self.assertEqual(packed, b"\xa5\xef\xe7\x05")

        decoded = DTCRecord.decode_4bytes(packed)
        self.assertEqual(decoded.spn, 520101)
        self.assertEqual(decoded.fmi, 7)
        self.assertEqual(decoded.oc, 5)
        self.assertEqual(decoded.cm, 0)

    def test_maximum_boundary_spn_fmi_oc_values(self):
        """Test max 19-bit SPN (524287 = 0x7FFFF), max 5-bit FMI (31), max 7-bit OC (127), CM=1."""
        dtc = DTCRecord(spn=524287, fmi=31, oc=127, cm=1)
        packed = dtc.encode_4bytes()

        self.assertEqual(packed, b"\xff\xff\xff\xff")

        decoded = DTCRecord.decode_4bytes(packed)
        self.assertEqual(decoded.spn, 524287)
        self.assertEqual(decoded.fmi, 31)
        self.assertEqual(decoded.oc, 127)
        self.assertEqual(decoded.cm, 1)

    def test_oc_clamping_behavior(self):
        """OC should be clamped to range [1, 127]."""
        dtc_low = DTCRecord(spn=100, fmi=1, oc=0)
        self.assertEqual(dtc_low.oc, 1)

        dtc_high = DTCRecord(spn=100, fmi=1, oc=500)
        self.assertEqual(dtc_high.oc, 127)

    def test_dtc_dict_serialization_and_equality(self):
        """Test dictionary round-trip and equality/hash mechanics."""
        dtc1 = DTCRecord(spn=110, fmi=0, oc=2, description="Coolant Overheat", lamp_status="RED_STOP")
        dtc_dict = dtc1.to_dict()

        self.assertEqual(dtc_dict["spn"], 110)
        self.assertEqual(dtc_dict["fmi"], 0)
        self.assertEqual(dtc_dict["code"], "SPN 110 FMI 00")
        self.assertEqual(dtc_dict["lamp_status"], "RED_STOP")
        self.assertIn("hex_4bytes", dtc_dict)

        dtc2 = DTCRecord.from_dict(dtc_dict)
        self.assertEqual(dtc1, dtc2)
        self.assertEqual(hash(dtc1), hash(dtc2))


class TestLampStatusAndHeader(unittest.TestCase):
    """2. Test SAE J1939 2-Byte Lamp Status Header Encoding and Decoding."""

    def test_all_lamps_off_header(self):
        """All lamps off -> 0x00 byte 1, default 0xFF byte 2."""
        lamp = LampStatus(mil=LAMP_OFF, red_stop=LAMP_OFF, amber_warning=LAMP_OFF, protect=LAMP_OFF)
        header = lamp.encode_header()

        self.assertEqual(len(header), 2)
        self.assertEqual(header[0], 0x00)
        self.assertFalse(lamp.has_active_lamp)

        decoded = LampStatus.decode_header(header)
        self.assertEqual(decoded.mil, LAMP_OFF)
        self.assertEqual(decoded.red_stop, LAMP_OFF)
        self.assertEqual(decoded.amber_warning, LAMP_OFF)
        self.assertEqual(decoded.protect, LAMP_OFF)

    def test_red_stop_lamp_active_header(self):
        """Red Stop Lamp ON (01b at bits 6-5) -> 0b00010000 = 0x10."""
        lamp = LampStatus(red_stop=LAMP_ON)
        header = lamp.encode_header()

        self.assertEqual(header[0], 0x10)
        self.assertTrue(lamp.has_active_lamp)
        self.assertTrue(lamp.is_red_stop)
        self.assertFalse(lamp.is_amber)

        decoded = LampStatus.decode_header(header)
        self.assertEqual(decoded.red_stop, LAMP_ON)
        self.assertEqual(decoded.amber_warning, LAMP_OFF)

    def test_amber_warning_lamp_active_header(self):
        """Amber Warning Lamp ON (01b at bits 4-3) -> 0b00000100 = 0x04."""
        lamp = LampStatus(amber_warning=LAMP_ON)
        header = lamp.encode_header()

        self.assertEqual(header[0], 0x04)
        self.assertTrue(lamp.is_amber)

        decoded = LampStatus.decode_header(header)
        self.assertEqual(decoded.amber_warning, LAMP_ON)

    def test_all_lamps_active_combined(self):
        """All 4 lamps ON -> 0b01010101 = 0x55."""
        lamp = LampStatus(mil=LAMP_ON, red_stop=LAMP_ON, amber_warning=LAMP_ON, protect=LAMP_ON)
        header = lamp.encode_header()

        self.assertEqual(header[0], 0x55)
        decoded = LampStatus.decode_header(header)
        self.assertEqual(decoded.mil, LAMP_ON)
        self.assertEqual(decoded.red_stop, LAMP_ON)
        self.assertEqual(decoded.amber_warning, LAMP_ON)
        self.assertEqual(decoded.protect, LAMP_ON)


class TestDM1AndDM2BinaryPacketPayloads(unittest.TestCase):
    """3. Test DM1 (PGN 65226) & DM2 (PGN 65227) Binary Payload Encoders/Decoders."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_path = Path(self.temp_dir.name) / "test_flash.jsonl"
        self.engine = DTCEngine(flash_file=self.flash_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_zero_active_dtcs_dm1_payload(self):
        """Zero active DTCs -> 8-byte payload: 2-byte header + 0x00000000 + 0xFFFF."""
        payload = self.engine.encode_dm1_packet()

        self.assertEqual(len(payload), 8)
        self.assertEqual(payload[0], 0x00) # Lamps off
        self.assertEqual(payload[2:6], b"\x00\x00\x00\x00")
        self.assertEqual(payload[6:8], b"\xff\xff")

        lamp, records = DTCEngine.decode_dm1_packet(payload)
        self.assertFalse(lamp.has_active_lamp)
        self.assertEqual(len(records), 0)

    def test_single_active_dtc_dm1_payload(self):
        """Single active DTC -> 8-byte payload: 2-byte header + 4-byte DTC + 0xFFFF."""
        self.engine.report_fault(spn=100, fmi=4, lamp="RED_STOP", description="Oil Pressure Wire Cut")
        payload = self.engine.encode_dm1_packet()

        self.assertEqual(len(payload), 8)
        self.assertEqual(payload[0], 0x10) # Red Stop ON
        self.assertEqual(payload[2:6], b"\x64\x00\x04\x01") # SPN 100 FMI 04 OC 1
        self.assertEqual(payload[6:8], b"\xff\xff")

        lamp, records = DTCEngine.decode_dm1_packet(payload)
        self.assertTrue(lamp.is_red_stop)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].spn, 100)
        self.assertEqual(records[0].fmi, 4)
        self.assertEqual(records[0].oc, 1)

    def test_multiple_active_dtcs_variable_length_payload(self):
        """Multiple (N=3) active DTCs -> 2 + 4*3 = 14 bytes."""
        self.engine.report_fault(spn=100, fmi=4, lamp="RED_STOP")
        self.engine.report_fault(spn=110, fmi=0, lamp="RED_STOP")
        self.engine.report_fault(spn=520101, fmi=7, lamp="AMBER_WARNING")

        payload = self.engine.encode_dm1_packet()
        self.assertEqual(len(payload), 2 + 4 * 3) # 14 bytes

        lamp, records = DTCEngine.decode_dm1_packet(payload)
        self.assertTrue(lamp.is_red_stop)
        self.assertTrue(lamp.is_amber)
        self.assertEqual(len(records), 3)
        spns = {r.spn for r in records}
        self.assertEqual(spns, {100, 110, 520101})

    def test_dm2_historic_payload_encoding_and_decoding(self):
        """Active fault cleared -> moved to DM2 historic payload."""
        self.engine.report_fault(spn=100, fmi=4)
        self.engine.clear_fault(100, 4)

        dm1_payload = self.engine.encode_dm1_packet()
        dm2_payload = self.engine.encode_dm2_packet()

        # DM1 should be empty
        _, dm1_records = DTCEngine.decode_dm1_packet(dm1_payload)
        self.assertEqual(len(dm1_records), 0)

        # DM2 should contain the cleared fault
        _, dm2_records = DTCEngine.decode_dm2_packet(dm2_payload)
        self.assertEqual(len(dm2_records), 1)
        self.assertEqual(dm2_records[0].spn, 100)
        self.assertEqual(dm2_records[0].fmi, 4)
        self.assertFalse(dm2_records[0].active)


class TestHook1SensorPlausibilityIngestion(unittest.TestCase):
    """4. Test Hook 1: Sensor Plausibility FDIR Fault Ingestion."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_path = Path(self.temp_dir.name) / "test_flash.jsonl"
        self.engine = DTCEngine(flash_file=self.flash_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_wire_cut_open_circuit_conversion(self):
        """Open circuit on P_oil (-999 bar) translates to SPN 100 FMI 04 with RED_STOP lamp."""
        fault = SensorFaultEvent(
            channel="oil_pressure",
            fault_type=PlausibilityFaultType.OPEN_CIRCUIT.value,
            raw_value=-999.0,
            clamped_value=0.0,
            spn=100,
            fmi=FMI_VOLTAGE_BELOW_NORMAL,
            message="Oil pressure transducer open circuit wire cut",
        )

        dtcs = self.engine.process_fdir_faults([fault])
        self.assertEqual(len(dtcs), 1)
        self.assertEqual(dtcs[0].spn, 100)
        self.assertEqual(dtcs[0].fmi, FMI_VOLTAGE_BELOW_NORMAL)
        self.assertEqual(dtcs[0].lamp_status, "RED_STOP")
        self.assertEqual(dtcs[0].source_type, "FDIR_ELECTRICAL")

    def test_short_circuit_to_power_conversion(self):
        """Short to power on coolant temp (+999 C) translates to SPN 110 FMI 03."""
        fault = SensorFaultEvent(
            channel="coolant_temp",
            fault_type=PlausibilityFaultType.SHORT_CIRCUIT.value,
            raw_value=999.0,
            clamped_value=160.0,
            spn=110,
            fmi=FMI_VOLTAGE_ABOVE_NORMAL,
            message="Coolant RTD harness short to +28V rail",
        )

        dtcs = self.engine.process_fdir_faults([fault])
        self.assertEqual(len(dtcs), 1)
        self.assertEqual(dtcs[0].spn, 110)
        self.assertEqual(dtcs[0].fmi, FMI_VOLTAGE_ABOVE_NORMAL)
        self.assertEqual(dtcs[0].lamp_status, "RED_STOP")

    def test_slew_rate_exceeded_conversion(self):
        """Unphysical step change translates to SPN / FMI 10 with AMBER_WARNING lamp."""
        fault = SensorFaultEvent(
            channel="oil_temp",
            fault_type=PlausibilityFaultType.RATE_OF_CHANGE_EXCEEDED.value,
            raw_value=150.0,
            clamped_value=86.0,
            spn=175,
            fmi=FMI_ABNORMAL_RATE_OF_CHANGE,
            message="Oil temperature slew rate limit breached",
        )

        dtcs = self.engine.process_fdir_faults([fault])
        self.assertEqual(len(dtcs), 1)
        self.assertEqual(dtcs[0].spn, 175)
        self.assertEqual(dtcs[0].fmi, FMI_ABNORMAL_RATE_OF_CHANGE)
        self.assertEqual(dtcs[0].lamp_status, "AMBER_WARNING")

    def test_stuck_at_and_emi_conversion(self):
        """Stuck-at and EMI outlier faults translate to FMI 02."""
        f_stuck = SensorFaultEvent(
            channel="rpm",
            fault_type=PlausibilityFaultType.STUCK_AT.value,
            raw_value=1800.0,
            clamped_value=1800.0,
            spn=190,
            fmi=FMI_DATA_ERRATIC,
        )
        f_emi = SensorFaultEvent(
            channel="hyd_pressure",
            fault_type=PlausibilityFaultType.OUTLIER_EMI.value,
            raw_value=390.0,
            clamped_value=210.0,
            spn=520202,
            fmi=FMI_DATA_ERRATIC,
        )

        dtcs = self.engine.process_fdir_faults([f_stuck, f_emi])
        self.assertEqual(len(dtcs), 2)
        self.assertEqual(dtcs[0].fmi, FMI_DATA_ERRATIC)
        self.assertEqual(dtcs[1].fmi, FMI_DATA_ERRATIC)

    def test_dual_sensor_cross_subsystem_mismatch_conversion(self):
        """Cross-subsystem mismatch (e.g. RPM 2500 with Oil Pressure 0) translates to FMI 14."""
        fault = SensorFaultEvent(
            channel="oil_pressure",
            fault_type=PlausibilityFaultType.DUAL_SENSOR_MISMATCH.value,
            raw_value=0.0,
            clamped_value=0.0,
            spn=100,
            fmi=FMI_SPECIAL_INSTRUCTIONS,
            message="Engine running at 2500 RPM with zero oil pressure",
        )

        dtcs = self.engine.process_fdir_faults([fault])
        self.assertEqual(len(dtcs), 1)
        self.assertEqual(dtcs[0].spn, 100)
        self.assertEqual(dtcs[0].fmi, FMI_SPECIAL_INSTRUCTIONS)
        self.assertEqual(dtcs[0].lamp_status, "RED_STOP")


class TestHook2NeuralPrognosticsIngestion(unittest.TestCase):
    """5. Test Hook 2: AI / Neural Prognostics & Subsystem Degradation Ingestion."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_path = Path(self.temp_dir.name) / "test_flash.jsonl"
        self.engine = DTCEngine(flash_file=self.flash_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_all_13_neural_fault_classes_mapping(self):
        """Verify each of the 13 neural fault classes maps to exact SPN and FMI."""
        for fault_name, expected in NEURAL_FAULT_CLASS_MAP.items():
            self.engine.reset()
            probs = {fault_name: 0.95}
            dtcs = self.engine.process_neural_predictions(fault_probs=probs)

            self.assertEqual(len(dtcs), 1, f"Failed for fault {fault_name}")
            self.assertEqual(dtcs[0].spn, expected["spn"], f"SPN mismatch for {fault_name}")
            self.assertEqual(dtcs[0].fmi, expected["fmi"], f"FMI mismatch for {fault_name}")
            self.assertEqual(dtcs[0].lamp_status, expected["lamp"], f"Lamp mismatch for {fault_name}")
            self.assertEqual(dtcs[0].source_type, "NEURAL_PROGNOSTIC")

    def test_fault_probability_threshold_filtering(self):
        """Probabilities below threshold (e.g. 0.25 < 0.40) must NOT trigger DTCs."""
        probs = {"bearing_wear": 0.25, "structural_crack": 0.15, "healthy": 0.60}
        dtcs = self.engine.process_neural_predictions(fault_probs=probs, fault_prob_threshold=0.40)
        self.assertEqual(len(dtcs), 0)

    def test_subsystem_critical_rul_degradation_trigger(self):
        """Critical RUL breach (< 20%) generates corresponding subsystem DTC."""
        subsystem_health = {
            "engine": 0.85,
            "cooling": 0.10,      # Critical failure (10% RUL)
            "hydraulics": 0.12,   # Critical failure (12% RUL)
            "powertrain": 0.90,
        }

        dtcs = self.engine.process_neural_predictions(
            subsystem_health=subsystem_health,
            critical_rul_threshold=0.20,
        )

        spns = {d.spn for d in dtcs}
        self.assertIn(110, spns)     # Cooling SPN 110
        self.assertIn(520202, spns)  # Hydraulics SPN 520202
        self.assertEqual(len(dtcs), 2)

    def test_cvrde_hardware_degradation_triggers(self):
        """Test CVRDE specific states: HSU pressure drop, gun recoil surge, NBC loss."""
        cvrde_states = {
            "cvrde_hsu_1_pressure_bar": 110.0,    # Normal is 220 bar, < 140 bar triggers fault
            "cvrde_gcs_recoil_force_kn": 520.0,   # Max rating is 450-480 kN, 520 kN is overload
            "cvrde_nbc_overpressure_pa": 120.0,   # Normal is 500 Pa, < 200 Pa is breach
        }

        dtcs = self.engine.process_neural_predictions(cvrde_states=cvrde_states)
        spns = {d.spn for d in dtcs}

        self.assertIn(520200, spns) # HSU Nitrogen Seal
        self.assertIn(520201, spns) # Gun Recoil Buffer
        self.assertIn(520300, spns) # NBC Overpressure


class TestLifecycleAndOccurrenceCount(unittest.TestCase):
    """6. Test Lifecycle Management, Occurrence Count Increments, and Multi-Fault States."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_path = Path(self.temp_dir.name) / "test_flash.jsonl"
        self.engine = DTCEngine(flash_file=self.flash_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_occurrence_count_increments_on_repeated_reentry(self):
        """Fault is activated (OC=1), cleared, reactivated (OC=2), cleared, reactivated (OC=3)."""
        # First trigger
        dtc1 = self.engine.report_fault(spn=100, fmi=4)
        self.assertEqual(dtc1.oc, 1)
        self.assertEqual(len(self.engine.get_active_dtcs()), 1)

        # Clear fault -> becomes historic
        self.engine.clear_fault(100, 4)
        self.assertEqual(len(self.engine.get_active_dtcs()), 0)
        self.assertEqual(len(self.engine.get_historic_dtcs()), 1)

        # Reactivate fault
        dtc2 = self.engine.report_fault(spn=100, fmi=4)
        self.assertEqual(dtc2.oc, 2)
        self.assertEqual(len(self.engine.get_active_dtcs()), 1)
        self.assertEqual(len(self.engine.get_historic_dtcs()), 0)

        # Clear again & reactivate
        self.engine.clear_fault(100, 4)
        dtc3 = self.engine.report_fault(spn=100, fmi=4)
        self.assertEqual(dtc3.oc, 3)

    def test_clear_all_active_and_historic(self):
        """Test bulk clear_active_dtcs and clear_historic_dtcs."""
        self.engine.report_fault(spn=100, fmi=4)
        self.engine.report_fault(spn=110, fmi=0)
        self.engine.report_fault(spn=190, fmi=2)

        self.assertEqual(len(self.engine.get_active_dtcs()), 3)
        self.assertEqual(len(self.engine.get_historic_dtcs()), 0)

        cleared = self.engine.clear_active_dtcs()
        self.assertEqual(len(cleared), 3)
        self.assertEqual(len(self.engine.get_active_dtcs()), 0)
        self.assertEqual(len(self.engine.get_historic_dtcs()), 3)

        purged = self.engine.clear_historic_dtcs()
        self.assertEqual(len(purged), 3)
        self.assertEqual(len(self.engine.get_historic_dtcs()), 0)


class TestFlashRingBufferPersistence(unittest.TestCase):
    """7. Test On-Disk Circular Flash Ring Buffer Persistence, Rollover, and Recovery."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_path = Path(self.temp_dir.name) / "test_flash.jsonl"
        self.engine = DTCEngine(flash_capacity=150, flash_file=self.flash_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_append_and_read_flash_log(self):
        """Verify state transitions write valid JSONL and read_flash_log returns entries."""
        self.engine.report_fault(spn=100, fmi=4, lamp="RED_STOP", description="Oil cut")
        self.engine.report_fault(spn=110, fmi=0, lamp="RED_STOP", description="Coolant high")
        self.engine.clear_fault(100, 4)

        records = self.engine.read_flash_log(limit=10)
        self.assertEqual(len(records), 3)

        # Most recent first
        self.assertEqual(records[0]["event_type"], "CLEARED")
        self.assertEqual(records[0]["spn"], 100)
        self.assertEqual(records[1]["event_type"], "ACTIVE")
        self.assertEqual(records[1]["spn"], 110)
        self.assertEqual(records[2]["event_type"], "ACTIVE")
        self.assertEqual(records[2]["spn"], 100)

    def test_bounded_circular_rollover(self):
        """When log exceeds flash_capacity (150), rollover prunes oldest records."""
        for i in range(200):
            dtc = DTCRecord(spn=100 + (i % 20), fmi=i % 15, oc=1)
            self.engine.persist_to_flash(dtc, event_type="ACTIVE")

        # Read total lines in flash file
        with open(self.flash_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        self.assertLessEqual(len(lines), 150)
        self.assertGreater(len(lines), 40)

    def test_crash_recovery_from_flash_log(self):
        """Instantiate new DTCEngine pointing to existing log file and verify state recovery."""
        self.engine.report_fault(spn=100, fmi=4, description="Oil Wire Cut")
        self.engine.report_fault(spn=110, fmi=0, description="Coolant Overheat")
        self.engine.report_fault(spn=190, fmi=2, description="RPM Intermittent")
        self.engine.clear_fault(190, 2) # Move RPM to historic

        # Create brand new engine instance pointing to the same flash file
        recovered_engine = DTCEngine(flash_file=self.flash_path, auto_recover=True)

        active = recovered_engine.get_active_dtcs()
        historic = recovered_engine.get_historic_dtcs()

        active_spns = {d.spn for d in active}
        historic_spns = {d.spn for d in historic}

        self.assertEqual(active_spns, {100, 110})
        self.assertEqual(historic_spns, {190})

    def test_thread_safe_concurrent_logging(self):
        """Concurrent threads logging DTCs simultaneously without data corruption."""
        threads = []
        errors = []

        def worker(tid: int):
            try:
                for j in range(30):
                    spn = 520000 + tid * 100 + j
                    self.engine.report_fault(spn=spn, fmi=j % 5)
            except Exception as e:
                errors.append(e)

        for t in range(5):
            th = threading.Thread(target=worker, args=(t,))
            threads.append(th)
            th.start()

        for th in threads:
            th.join()

        self.assertEqual(len(errors), 0)
        log_entries = self.engine.read_flash_log(limit=500)
        self.assertEqual(len(log_entries), 150) # 5 * 30 = 150 entries


class TestEndToEndPlausibilityToDTCPipeline(unittest.TestCase):
    """8. Test End-to-End Integration: SensorPlausibilityGate -> DTCEngine Pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_path = Path(self.temp_dir.name) / "test_flash.jsonl"
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)
        self.engine = DTCEngine(flash_file=self.flash_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_adversarial_wire_cut_produces_dm1_binary_packet(self):
        """Wire cut P_oil = -999.0 bar -> Plausibility Gate flags OPEN_CIRCUIT ->

        DTCEngine creates SPN 100 FMI 04 -> DM1 packet encodes Red Stop Lamp + SPN 100 FMI 04.
        """
        raw_telemetry = {
            "oil_pressure": -999.0, # Adversarial wire cut
            "coolant_temp": 88.0,
        }

        # Step 1: Filter frame through FDIR gate
        plaus_res = self.gate.filter_frame(raw_telemetry)
        self.assertTrue(plaus_res.is_valid)
        self.assertEqual(plaus_res.clean_telemetry["oil_pressure"], 0.0) # Clamped safely
        self.assertEqual(len(plaus_res.faults_detected), 1)
        self.assertEqual(plaus_res.faults_detected[0].fault_type, PlausibilityFaultType.OPEN_CIRCUIT.value)

        # Step 2: Feed faults into DTC Engine
        dtcs = self.engine.process_fdir_faults(plaus_res.faults_detected)
        self.assertEqual(len(dtcs), 1)
        self.assertEqual(dtcs[0].spn, 100)
        self.assertEqual(dtcs[0].fmi, FMI_VOLTAGE_BELOW_NORMAL)
        self.assertEqual(dtcs[0].lamp_status, "RED_STOP")

        # Step 3: Encode DM1 Packet
        dm1_payload = self.engine.encode_dm1_packet()
        self.assertEqual(len(dm1_payload), 8)

        # Step 4: Decode DM1 Packet and verify integrity
        lamp, decoded_dtcs = DTCEngine.decode_dm1_packet(dm1_payload)
        self.assertTrue(lamp.is_red_stop)
        self.assertEqual(len(decoded_dtcs), 1)
        self.assertEqual(decoded_dtcs[0].spn, 100)
        self.assertEqual(decoded_dtcs[0].fmi, 4)
        self.assertEqual(decoded_dtcs[0].oc, 1)

        # Step 5: Verify flash log record
        flash_records = self.engine.read_flash_log(limit=10)
        self.assertEqual(len(flash_records), 1)
        self.assertEqual(flash_records[0]["spn"], 100)
        self.assertEqual(flash_records[0]["fmi"], 4)
        self.assertEqual(flash_records[0]["event_type"], "ACTIVE")

    def test_combined_fdir_and_neural_faults_in_dm1_packet(self):
        """Simultaneous electrical fault (coolant RTD short) + AI fault (bearing wear)."""
        # 1. Electrical fault from plausibility
        raw = {"coolant_temp": 999.0, "oil_pressure": 5.0}
        plaus_res = self.gate.filter_frame(raw)
        self.engine.process_fdir_faults(plaus_res.faults_detected)

        # 2. AI fault prediction from neural model
        self.engine.process_neural_predictions(fault_probs={"bearing_wear": 0.88})

        # Verify active DTCs
        active = self.engine.get_active_dtcs()
        self.assertEqual(len(active), 2)
        active_spns = {d.spn for d in active}
        self.assertEqual(active_spns, {110, 520101})

        # Encode DM1
        dm1_bytes = self.engine.encode_dm1_packet()
        self.assertEqual(len(dm1_bytes), 2 + 4 * 2) # 10 bytes

        lamp, decoded = DTCEngine.decode_dm1_packet(dm1_bytes)
        self.assertTrue(lamp.is_red_stop)
        self.assertTrue(lamp.is_amber)
        self.assertEqual(len(decoded), 2)


class TestAdversarialCorruptedPayloadAndEdgeCases(unittest.TestCase):
    """9. Test Adversarial Corrupted Payloads, Malformed Data, and Performance Benchmarks."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.flash_path = Path(self.temp_dir.name) / "edge_flash.jsonl"
        self.engine = DTCEngine(flash_file=self.flash_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_truncated_dm1_packet_raises_error(self):
        """Packet < 2 bytes must raise ValueError on decode."""
        with self.assertRaises(ValueError):
            DTCEngine.decode_dm1_packet(b"")
        with self.assertRaises(ValueError):
            DTCEngine.decode_dm1_packet(b"\x00")

    def test_corrupt_json_lines_in_flash_log_skipped_safely(self):
        """Corrupted/garbage lines in flash file must be skipped without crashing recovery."""
        with open(self.flash_path, "w", encoding="utf-8") as fh:
            fh.write("NOT A JSON LINE\n")
            fh.write("{corrupt: json,\n")
            fh.write(json.dumps({"spn": 100, "fmi": 4, "event_type": "ACTIVE", "oc": 1}) + "\n")
            fh.write("\x00\xFF\xFE\n")

        recovered = DTCEngine(flash_file=self.flash_path, auto_recover=True)
        active = recovered.get_active_dtcs()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].spn, 100)
        self.assertEqual(active[0].fmi, 4)

    def test_13_class_float_list_softmax_ingestion(self):
        """Test neural prediction ingestion when fault_probs is passed as a 13-element float list/array."""
        # Index 3 is cooling_failure (SPN 110, FMI 00)
        softmax_vector = [0.0] * 13
        softmax_vector[3] = 0.92 # cooling_failure prob = 0.92

        dtcs = self.engine.process_neural_predictions(fault_probs=softmax_vector)
        self.assertEqual(len(dtcs), 1)
        self.assertEqual(dtcs[0].spn, 110)
        self.assertEqual(dtcs[0].fmi, 0)
        self.assertEqual(dtcs[0].lamp_status, "RED_STOP")

    def test_high_throughput_dtc_processing_benchmark(self):
        """Benchmark 2,000 DTC processing and encoding cycles, verifying < 0.2 ms per frame."""
        t0 = time.perf_counter()
        fault_event = SensorFaultEvent(
            channel="oil_pressure",
            fault_type=PlausibilityFaultType.OPEN_CIRCUIT.value,
            raw_value=-999.0,
            clamped_value=0.0,
            spn=100,
            fmi=4,
        )

        for i in range(2000):
            self.engine.process_fdir_faults([fault_event])
            payload = self.engine.encode_dm1_packet()

        elapsed_s = time.perf_counter() - t0
        avg_ms = (elapsed_s / 2000.0) * 1000.0
        self.assertLess(avg_ms, 0.5, f"Average DTC processing latency too high: {avg_ms:.4f} ms")


if __name__ == "__main__":
    unittest.main()

