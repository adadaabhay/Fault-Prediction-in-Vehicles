"""FDIR plausibility-gate behaviour on real recorded vehicle data.

The gate was only ever exercised with synthetic frames.  Replaying a real CAN
trace surfaced a 69% false-positive rate: the stuck-at detector demanded bit
equality over a fixed frame count, which a thermostat-regulated coolant
temperature and an idle accelerator pedal violate by design.
"""

import collections
import unittest

from telemetry_gateway.sensor_plausibility import SensorPlausibilityGate

try:
    import h5py  # noqa: F401
    from pipelines.can_aegis import find_trips, load_can_trip, to_telemetry_frames
    HAVE_AEGIS = bool(find_trips())
except Exception:  # pragma: no cover - environment dependent
    HAVE_AEGIS = False

SAMPLE_HZ = 2.0


@unittest.skipUnless(HAVE_AEGIS, "AEGIS CAN trace / h5py not available")
class TestGateOnHealthyRealTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        trip = load_can_trip(resample_hz=SAMPLE_HZ)
        cls.frames = to_telemetry_frames(trip, limit=1400)

    def test_healthy_trip_raises_no_faults(self):
        gate = SensorPlausibilityGate(sample_rate_hz=SAMPLE_HZ)
        counts = collections.Counter()
        for frame in self.frames:
            for ev in gate.filter_frame(frame).faults_detected:
                counts[(ev.channel, ev.fault_type)] += 1
        self.assertEqual(sum(counts.values()), 0,
                         f"false positives on healthy real data: {dict(counts)}")

    def test_regulated_channels_may_hold_a_constant_value(self):
        """Coolant temperature legitimately sits at one quantised value for
        minutes on a warmed-up, thermostat-controlled engine."""
        gate = SensorPlausibilityGate(sample_rate_hz=SAMPLE_HZ)
        for frame in self.frames[:400]:
            held = dict(frame)
            held["coolant_temp"] = 93.0
            for ev in gate.filter_frame(held).faults_detected:
                self.assertNotEqual(ev.channel, "coolant_temp")

    def _first_stuck(self, channel, value, limit=1400):
        gate = SensorPlausibilityGate(sample_rate_hz=SAMPLE_HZ)
        for i, frame in enumerate(self.frames[:limit]):
            frozen = dict(frame)
            frozen[channel] = value
            for ev in gate.filter_frame(frozen).faults_detected:
                if ev.fault_type == "STUCK_AT" and ev.channel == channel:
                    return i / SAMPLE_HZ
        return None

    def test_genuinely_frozen_sensor_is_still_caught(self):
        for channel, value, max_latency_s in (("coolant_temp", 93.0, 260.0),
                                              ("rpm", 1500.0, 30.0),
                                              ("shaft_torque", 50.0, 30.0)):
            latency = self._first_stuck(channel, value)
            self.assertIsNotNone(latency, f"{channel} flatline never detected")
            self.assertLessEqual(latency, max_latency_s, channel)

    def test_fast_channels_keep_a_tight_flatline_window(self):
        """Relaxing slow channels must not relax the fast ones."""
        self.assertLessEqual(self._first_stuck("rpm", 1500.0), 30.0)

    def test_injected_electrical_faults_are_still_detected(self):
        gate = SensorPlausibilityGate(sample_rate_hz=SAMPLE_HZ)
        for frame in self.frames[:20]:
            gate.filter_frame(frame)
        corrupt = dict(self.frames[20])
        corrupt["oil_pressure"] = -999.0
        corrupt["coolant_temp"] = 9999.0
        kinds = {ev.fault_type for ev in gate.filter_frame(corrupt).faults_detected}
        self.assertIn("OPEN_CIRCUIT", kinds)
        self.assertIn("SHORT_CIRCUIT", kinds)


if __name__ == "__main__":
    unittest.main()
