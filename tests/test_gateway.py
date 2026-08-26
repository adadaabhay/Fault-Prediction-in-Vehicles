"""Unit test suite for the Tactical Telemetry Gateway & Protocols."""

import unittest
import os
import json
from pathlib import Path

from telemetry_gateway.j1939_can_parser import J1939FrameParser
from telemetry_gateway.tactical_burst import TacticalBurstPacket


class TestTelemetryGateway(unittest.TestCase):
    def test_j1939_eec1_encoding_decoding(self):
        """Verify J1939 PGN 61444 RPM and Torque encoding."""
        rpm_in = 2150.0
        tq_in = 75.0
        
        frame = J1939FrameParser.encode_eec1(rpm=rpm_in, torque_pct=tq_in)
        self.assertEqual(len(frame), 8, "J1939 CAN frame must be exactly 8 bytes")
        
        decoded = J1939FrameParser.decode_eec1(frame)
        self.assertAlmostEqual(decoded["rpm"], rpm_in, delta=0.125)
        self.assertEqual(decoded["torque_pct"], tq_in)

    def test_j1939_efl_p1_encoding_decoding(self):
        """PGN 65263 (EFL_P1) carries SPN 100 oil pressure -- and no coolant
        temperature.

        This test previously asserted that coolant temperature round-tripped
        through EFL_P1 byte 1. Byte 1 of PGN 65263 is SPN 94, Fuel Delivery
        Pressure; SPN 110 lives in PGN 65262 (ET1). The test was pinning the
        bug rather than the standard, so a conforming J1939 stack would have
        decoded the tank's coolant temperature as a fuel pressure in kPa.
        """
        oil_p_in = 480.0        # kPa
        fuel_p_in = 320.0       # kPa, SPN 94

        frame = J1939FrameParser.encode_efl_p1(
            oil_pressure_kpa=oil_p_in, fuel_delivery_pressure_kpa=fuel_p_in)
        self.assertEqual(len(frame), 8)

        decoded = J1939FrameParser.decode_efl_p1(frame)
        self.assertAlmostEqual(decoded["oil_pressure_kpa"], oil_p_in, delta=4.0)
        self.assertAlmostEqual(decoded["fuel_delivery_pressure_kpa"],
                               fuel_p_in, delta=4.0)
        self.assertNotIn("coolant_temp", decoded,
                         "EFL_P1 must not carry SPN 110")

    def test_j1939_et1_carries_coolant_temperature(self):
        """SPN 110 round-trips through PGN 65262 (ET1), where it belongs."""
        frame = J1939FrameParser.encode_et1(coolant_temp_c=92.0,
                                            oil_temp_c=105.0)
        self.assertEqual(len(frame), 8)
        decoded = J1939FrameParser.decode_et1(frame)
        self.assertAlmostEqual(decoded["coolant_temp"], 92.0, delta=1.0)
        self.assertAlmostEqual(decoded["oil_temp"], 105.0, delta=0.05)

    def test_j1939_et1_offset_and_resolution_match_the_standard(self):
        """SPN 110: 1 degC/bit, -40 offset. Boundary values."""
        for t in (-40.0, 0.0, 92.0, 210.0):
            d = J1939FrameParser.decode_et1(
                J1939FrameParser.encode_et1(coolant_temp_c=t))
            self.assertAlmostEqual(d["coolant_temp"], t, delta=1.0, msg=str(t))

    def test_tactical_burst_32_bytes_and_crc(self):
        """Verify 32-byte EMCON packet encoding and CRC-16 verification."""
        packet = TacticalBurstPacket.encode(
            tank_id=3,
            mission_time=1450,
            chi=94.5,
            top_fault_id=2,
            fault_confidence=0.88,
            rul_minutes=540,
            subsystem_health=[95, 96, 92, 98, 99, 91, 94, 97],
            rpm=2400.0,
            oil_pressure_bar=4.6,
            coolant_temp_c=88.5,
            vib_rms=0.62
        )
        self.assertEqual(len(packet), 32, "Tactical radio burst must be exactly 32 bytes")
        
        decoded = TacticalBurstPacket.decode(packet)
        self.assertEqual(decoded["header"], "TK")
        self.assertEqual(decoded["tank_id"], 3)
        self.assertEqual(decoded["composite_chi"], 94.0)
        self.assertEqual(decoded["rul_minutes"], 540)
        self.assertTrue(decoded["crc_valid"])

    def test_tactical_burst_corrupted_crc_rejected(self):
        """Verify corrupted packet fails CRC-16 check."""
        packet = bytearray(TacticalBurstPacket.encode(
            tank_id=1, mission_time=100, chi=90.0, top_fault_id=0, fault_confidence=0.9,
            rul_minutes=300, subsystem_health=[90]*8, rpm=1800.0, oil_pressure_bar=4.0,
            coolant_temp_c=85.0, vib_rms=0.5
        ))
        
        # Corrupt 1 byte in payload
        packet[5] ^= 0xFF
        with self.assertRaises(ValueError):
            TacticalBurstPacket.decode(bytes(packet))

    def test_multi_streams_json_exists_and_valid(self):
        """Verify live_multi_streams.json has all 4 valid streams."""
        path = "docs/live_multi_streams.json"
        if not os.path.exists(path):
            self.skipTest("live_multi_streams.json not generated; run export_multi_streams.py first")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("streams", data)
        self.assertIn("sim_mbt", data["streams"])
        self.assertIn("cvrde_arjun", data["streams"])
        self.assertIn("real_deutz", data["streams"])
        self.assertIn("real_zema", data["streams"])
        self.assertIn("real_metropt", data["streams"])


if __name__ == "__main__":
    unittest.main()
