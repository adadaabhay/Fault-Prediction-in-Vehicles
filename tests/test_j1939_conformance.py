"""Bit-layout conformance for SAE J1939-73 diagnostic messages.

A DTC packs into 4 bytes: SPN is 19 bits (low 16 in bytes 0-1, high 3 in the
top bits of byte 2), FMI is 5 bits (low bits of byte 2), OC is 7 bits of
byte 3, and CM is the top bit of byte 3.

Occurrence count is deliberately floored at 1: a DTC that has occurred zero
times is not a DTC, and J1939 has no encoding for one.
"""

import unittest

from telemetry_gateway.dtc_engine import DTCEngine, DTCRecord


class TestDTCBitPacking(unittest.TestCase):
    def test_four_byte_encoding_round_trips(self):
        for spn, fmi, oc in ((100, 1, 1), (110, 3, 5), (524287, 31, 127), (190, 0, 2)):
            record = DTCRecord(spn=spn, fmi=fmi, oc=oc, cm=0,
                               description="t", lamp_status="AMBER_WARNING")
            raw = record.encode_4bytes()
            self.assertEqual(len(raw), 4)
            back = DTCRecord.decode_4bytes(raw)
            self.assertEqual(back.spn, spn, f"SPN {spn}")
            self.assertEqual(back.fmi, fmi, f"FMI {fmi}")
            self.assertEqual(back.oc, oc, f"OC {oc}")

    def test_spn_high_bits_land_in_byte_two(self):
        """SPN 524287 is 19 set bits; the top 3 must occupy byte 2 bits 5-7."""
        record = DTCRecord(spn=524287, fmi=0, oc=1, cm=0,
                           description="t", lamp_status="MIL")
        raw = record.encode_4bytes()
        self.assertEqual(raw[0], 0xFF)
        self.assertEqual(raw[1], 0xFF)
        self.assertEqual((raw[2] >> 5) & 0x07, 0x07)

    def test_fmi_occupies_low_five_bits_of_byte_two(self):
        record = DTCRecord(spn=0, fmi=31, oc=1, cm=0,
                           description="t", lamp_status="MIL")
        self.assertEqual(record.encode_4bytes()[2] & 0x1F, 31)

    def test_occurrence_count_is_seven_bits(self):
        record = DTCRecord(spn=100, fmi=1, oc=127, cm=0,
                           description="t", lamp_status="MIL")
        self.assertEqual(record.encode_4bytes()[3] & 0x7F, 127)

    def test_conversion_method_is_the_top_bit_of_byte_three(self):
        record = DTCRecord(spn=100, fmi=1, oc=1, cm=1,
                           description="t", lamp_status="MIL")
        self.assertEqual((record.encode_4bytes()[3] >> 7) & 0x01, 1)

    def test_occurrence_count_saturates_rather_than_wrapping(self):
        """An OC beyond 7 bits must clamp, not alias to a small number."""
        record = DTCRecord(spn=100, fmi=1, oc=500, cm=0,
                           description="t", lamp_status="MIL")
        self.assertEqual(record.oc, 127)


class TestDM1DM2Packets(unittest.TestCase):
    def setUp(self):
        self.engine = DTCEngine()
        self.engine.reset()

    def test_dm1_carries_lamp_header_plus_every_active_dtc(self):
        for spn, fmi in ((100, 1), (110, 3), (190, 2)):
            self.engine.report_fault(spn=spn, fmi=fmi,
                                     description=f"spn{spn}",
                                     lamp="AMBER_WARNING")
        packet = self.engine.encode_dm1_packet()
        self.assertGreaterEqual(len(packet), 2 + 4 * 3,
                                "DM1 must pack all active DTCs, not just the first")
        _lamp, dtcs = DTCEngine.decode_dm1_packet(packet)
        self.assertEqual({d.spn for d in dtcs}, {100, 110, 190})

    def test_empty_dm1_is_the_no_faults_pattern(self):
        packet = self.engine.encode_dm1_packet()
        self.assertGreaterEqual(len(packet), 8)
        _lamp, dtcs = DTCEngine.decode_dm1_packet(packet)
        self.assertEqual(dtcs, [])

    def test_lamp_header_round_trips_through_dm1(self):
        self.engine.report_fault(spn=100, fmi=1, description="oil",
                                 lamp="RED_STOP")
        lamp, _ = DTCEngine.decode_dm1_packet(self.engine.encode_dm1_packet())
        self.assertTrue(lamp.is_red_stop)

    def test_cleared_fault_moves_from_dm1_to_dm2(self):
        self.engine.report_fault(spn=100, fmi=1, description="oil",
                                 lamp="RED_STOP")
        self.engine.clear_fault(100, 1)
        self.assertEqual(self.engine.get_active_dtcs(), [])
        self.assertIn(100, [d.spn for d in self.engine.get_historic_dtcs()])
        _lamp, dtcs = DTCEngine.decode_dm2_packet(self.engine.encode_dm2_packet())
        self.assertIn(100, [d.spn for d in dtcs])

    def test_red_stop_dominates_amber_in_the_lamp_header(self):
        self.engine.report_fault(spn=100, fmi=1, description="a",
                                 lamp="AMBER_WARNING")
        self.engine.report_fault(spn=110, fmi=3, description="b",
                                 lamp="RED_STOP")
        self.assertTrue(self.engine.get_lamp_status().is_red_stop)

    def test_rapid_repeats_do_not_inflate_occurrence_count(self):
        """One continuously present fault is one occurrence, not one per sample.

        Sampled at 20 Hz this previously reached OC 61 in three seconds and
        saturated the 7-bit field in under seven."""
        t0 = 1_000_000.0
        self.engine._activate_or_increment_fault(spn=100, fmi=1, timestamp=t0)
        for i in range(1, 80):
            self.engine._activate_or_increment_fault(
                spn=100, fmi=1, timestamp=t0 + i * 0.05)
        active = self.engine.get_active_dtcs()
        self.assertEqual(len(active), 1, "same SPN/FMI must not duplicate")
        self.assertEqual(active[0].oc, 1,
                         f"continuous fault inflated OC to {active[0].oc}")

    def test_a_genuine_re_occurrence_increments(self):
        """A fault that goes quiet and returns is a second occurrence."""
        t0 = 2_000_000.0
        self.engine._activate_or_increment_fault(spn=100, fmi=1, timestamp=t0)
        self.engine._activate_or_increment_fault(spn=100, fmi=1, timestamp=t0 + 30.0)
        self.assertEqual(self.engine.get_active_dtcs()[0].oc, 2)

    def test_occurrence_count_cannot_exceed_the_seven_bit_field(self):
        t0 = 3_000_000.0
        for i in range(200):
            self.engine._activate_or_increment_fault(
                spn=100, fmi=1, timestamp=t0 + i * 30.0)
        self.assertLessEqual(self.engine.get_active_dtcs()[0].oc, 127)

    def test_dm1_survives_a_full_encode_decode_with_many_dtcs(self):
        """Multi-frame territory: more DTCs than fit a single 8-byte CAN frame."""
        expected = set()
        for i in range(6):
            spn = 100 + i
            self.engine.report_fault(spn=spn, fmi=i % 32,
                                     description=f"f{i}", lamp="AMBER_WARNING")
            expected.add(spn)
        _lamp, dtcs = DTCEngine.decode_dm1_packet(self.engine.encode_dm1_packet())
        self.assertEqual({d.spn for d in dtcs}, expected)


if __name__ == "__main__":
    unittest.main()
