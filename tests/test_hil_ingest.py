"""Replay a recorded vehicle CAN trace through the real ingest path:
UDP socket -> broker -> plausibility gate -> DTC engine.

The gateway had only ever been exercised with synthetic frames. Replaying real
data is what caught the stuck-at detector's 69% false-positive rate, so the
deployed path needs a test that uses a real recording and a real socket.
"""

import json
import socket
import time
import unittest

from telemetry_gateway.dtc_engine import DTCEngine
from telemetry_gateway.live_sensor_ingest import UDPSensorListener, get_broker
from telemetry_gateway.sensor_plausibility import SensorPlausibilityGate

try:
    import h5py  # noqa: F401
    from pipelines.can_aegis import find_trips, load_can_trip, to_telemetry_frames
    HAVE = bool(find_trips())
except Exception:  # pragma: no cover - environment dependent
    HAVE = False

PORT = 9107  # non-default, so a running gateway does not interfere
SAMPLE_HZ = 2.0


@unittest.skipUnless(HAVE, "AEGIS CAN trace not available")
class TestHardwareInTheLoopIngest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = to_telemetry_frames(
            load_can_trip(resample_hz=SAMPLE_HZ), limit=200)

    def setUp(self):
        self.broker = get_broker()
        self.broker.reset()
        self.listener = UDPSensorListener(port=PORT)
        self.listener.start(host="127.0.0.1", port=PORT)
        time.sleep(0.2)

    def tearDown(self):
        self.listener.stop()
        self.broker.reset()

    def _send(self, frames, settle=0.6):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for frame in frames:
                sock.sendto(json.dumps(frame).encode("utf-8"), ("127.0.0.1", PORT))
                time.sleep(0.002)
        finally:
            sock.close()
        time.sleep(settle)

    def test_real_frames_reach_the_broker_over_udp(self):
        self._send(self.frames[:50])
        self.assertGreater(self.broker.packet_count, 0,
                           "no datagrams reached the broker")
        latest = self.broker.get_latest_telemetry()
        self.assertIn("rpm", latest)
        self.assertGreater(latest["rpm"], 100.0,
                           "rpm looks like a placeholder, not a measurement")

    def test_broker_reports_live_hardware_after_real_ingest(self):
        self._send(self.frames[:30])
        self.assertTrue(self.broker.is_live_hardware)
        self.assertIn(self.broker.active_source, ("udp", "serial", "http"))

    def test_healthy_trace_produces_no_dtcs(self):
        gate = SensorPlausibilityGate(sample_rate_hz=SAMPLE_HZ)
        engine = DTCEngine()
        engine.reset()
        for frame in self.frames:
            result = gate.filter_frame(frame)
            engine.process_fdir_faults(result.faults_detected)
        self.assertEqual(engine.get_active_dtcs(), [],
                         "a healthy recorded trip must not raise DTCs")

    def test_injected_open_circuit_raises_a_dtc_with_correct_spn(self):
        gate = SensorPlausibilityGate(sample_rate_hz=SAMPLE_HZ)
        engine = DTCEngine()
        engine.reset()
        for frame in self.frames[:20]:
            gate.filter_frame(frame)
        corrupt = dict(self.frames[20])
        corrupt["oil_pressure"] = -999.0
        records = engine.process_fdir_faults(
            gate.filter_frame(corrupt).faults_detected)
        self.assertTrue(records, "open circuit on real data raised no DTC")
        self.assertIn(100, [r.spn for r in records])  # SPN 100 = engine oil pressure
        self.assertTrue(engine.get_lamp_status().has_active_lamp)

    def test_end_to_end_produces_an_encodable_dm1_frame(self):
        gate = SensorPlausibilityGate(sample_rate_hz=SAMPLE_HZ)
        engine = DTCEngine()
        engine.reset()
        for frame in self.frames[:20]:
            gate.filter_frame(frame)
        corrupt = dict(self.frames[20])
        corrupt["coolant_temp"] = 9999.0
        engine.process_fdir_faults(gate.filter_frame(corrupt).faults_detected)

        packet = engine.encode_dm1_packet()
        self.assertGreaterEqual(len(packet), 8)
        _lamp, dtcs = DTCEngine.decode_dm1_packet(packet)
        self.assertTrue(dtcs, "DM1 carried no DTC after a real fault injection")


if __name__ == "__main__":
    unittest.main()
