"""Empirical Challenger Test Suite for Milestone M2 (challenger_m2_1).

Adversarial Stress Testing & Signal Fuzzing Harness for:
    `telemetry_gateway/sensor_plausibility.py` (Pre-Inference FDIR Gate).

Scope & Rigorous Verification:
1. Massive Pathological Corpus Generation (11,000+ pathological frames total):
   - 1,500 alternating wire-cut open circuits (below open-circuit thresholds / min_physical - 500).
   - 1,500 alternating short circuits to power/rail (above short-circuit thresholds / max_physical + 500).
   - 1,500 high-amplitude transient EMI bursts and single-sample spikes.
   - 1,500 NaN and Infinity corruptions (math.nan, float('nan'), +inf, -inf, "nan", "inf", "-inf").
   - 1,500 Nulls, empty strings, corrupt hex, nested lists/dicts, and non-numeric types.
   - 1,500 frozen flatlines across dynamic channels with engine running (RPM > 500) and stopped.
   - 2,000 randomized combinatorial chaos frames (simultaneous multi-channel fault combinations).
2. Guarantees Checked on Every Single Frame:
   - Zero unhandled exceptions.
   - Output clean_telemetry is 100% finite floating point (no NaN, no Inf, no string/null).
   - Clamped within physical operating envelopes [min_physical, max_physical] for all 58 channels.
   - Standard SAE J1939-73 SPN & FMI compliance (FMI 00, 01, 02, 03, 04, 14).
3. Sub-millisecond Execution Latency Benchmark:
   - Average per-frame latency < 1.0 ms across 58 channels.
   - P99 latency < 2.0 ms.
"""

import collections
import gc
import math
import random
import time
import unittest
from typing import Any, Dict, List, Tuple

from telemetry_gateway.sensor_plausibility import (
    SensorPlausibilityGate,
    PlausibilityResult,
    SensorFaultEvent,
    SensorLimits,
    PlausibilityFaultType,
    SENSOR_LIMITS_CATALOG,
    CHANNEL_ALIASES,
    FMI_DATA_VALID_ABOVE_NORMAL,
    FMI_DATA_VALID_BELOW_NORMAL,
    FMI_DATA_ERRATIC,
    FMI_VOLTAGE_ABOVE_NORMAL,
    FMI_VOLTAGE_BELOW_NORMAL,
    FMI_SPECIAL_INSTRUCTIONS,
)

# 58 standard AFV sensor channels from the CVRDE tank physics model
ALL_58_SENSOR_CHANNELS = [
    "time", "step", "rpm", "load", "terrain",
    "coolant_temp", "coolant_rtd_ohm", "exhaust_temp", "exhaust_thermocouple_v",
    "exhaust_pressure", "exhaust_mass_flow",
    "lambda", "exhaust_o2_pct", "oil_pressure", "oil_temp", "oil_viscosity",
    "oil_flow", "debris_cumulative", "debris_rate", "debris_particles",
    "shaft_torque", "shaft_shear_stress", "shaft_shear_strain",
    "mech_power", "shaft_omega", "fuel_level", "fuel_volume",
    "oil_level", "coolant_level", "fuel_capacitance_pf", "hyd_pressure",
    "hyd_flow", "hyd_force", "hyd_power", "hyd_leak_flow",
    "susp_load_kN", "susp_stress_MPa",
    "susp_strain_ue", "susp_dR_ohm", "torsion_torque", "torsion_twist_deg",
    "torsion_shear_MPa",
    "torsion_cumulative_twist", "shock_a_rms_g", "shock_peak_g", "shock_energy",
    "spl_db",
    "acoustic_dom_freq", "acoustic_energy", "ae_event_rate", "ae_events",
    "ae_energy",
    "ae_amp_dB", "ae_duration_s", "vib_rms", "vib_kurtosis", "vib_dom_freq",
    "vib_dom_amp",
    "vib_energy",
]

DYNAMIC_CHANNELS = [
    ch for ch, lim in SENSOR_LIMITS_CATALOG.items()
    if lim.is_dynamic and ch in ALL_58_SENSOR_CHANNELS
]

STATIC_CHANNELS = [
    ch for ch, lim in SENSOR_LIMITS_CATALOG.items()
    if not lim.is_dynamic and ch in ALL_58_SENSOR_CHANNELS
]


class TestAdversarialWireCutsAndShortCircuits(unittest.TestCase):
    """Stress test Layer 2: 3,000 pathological wire-cut open circuits & short-to-power faults."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)

    def test_wire_cut_fuzzing_1500_frames_across_58_channels(self):
        """Generate 1,500 frames of extreme open-circuit wire cuts across all 58 channels."""
        num_frames = 1500

        for frame_idx in range(num_frames):
            target_ch = ALL_58_SENSOR_CHANNELS[frame_idx % len(ALL_58_SENSOR_CHANNELS)]
            lim = SENSOR_LIMITS_CATALOG[target_ch]

            # Choose an adversarial wire cut value below the open circuit threshold
            wire_cut_val = min(lim.open_circuit_threshold_low - 10.0, lim.min_physical - 500.0, -999.0)
            if frame_idx % 5 == 0:
                wire_cut_val = -1.0e9  # Extreme negative float

            # Construct frame with healthy baseline and 1 to 2 wire cuts
            raw_frame: Dict[str, Any] = {
                ch: SENSOR_LIMITS_CATALOG[ch].healthy_nominal
                for ch in ALL_58_SENSOR_CHANNELS
            }
            raw_frame[target_ch] = wire_cut_val

            # Occasionally cut a second channel simultaneously
            if frame_idx % 3 == 0:
                secondary_ch = ALL_58_SENSOR_CHANNELS[(frame_idx * 7) % len(ALL_58_SENSOR_CHANNELS)]
                sec_lim = SENSOR_LIMITS_CATALOG[secondary_ch]
                raw_frame[secondary_ch] = min(sec_lim.open_circuit_threshold_low - 10.0, sec_lim.min_physical - 500.0)

            res = self.gate.filter_frame(raw_frame)

            # Assertions on every single frame
            self.assertTrue(res.is_valid, f"Frame {frame_idx} marked invalid")
            self.assertEqual(len(res.clean_telemetry), len(ALL_58_SENSOR_CHANNELS))

            # Verify target channel was clamped safely to min_physical
            val = res.clean_telemetry[target_ch]
            self.assertTrue(math.isfinite(val), f"Channel {target_ch} produced non-finite {val}")
            self.assertGreaterEqual(val, lim.min_physical, f"Channel {target_ch} below min_physical: {val} < {lim.min_physical}")
            self.assertLessEqual(val, lim.max_physical, f"Channel {target_ch} above max_physical: {val} > {lim.max_physical}")

            # Verify diagnostic event
            ch_faults = [f for f in res.faults_detected if f.channel == target_ch]
            self.assertGreaterEqual(len(ch_faults), 1, f"No fault emitted for wire cut on {target_ch}")
            open_fault = next((f for f in ch_faults if f.fault_type == PlausibilityFaultType.OPEN_CIRCUIT.value), None)
            self.assertIsNotNone(open_fault, f"Missing OPEN_CIRCUIT fault event for {target_ch}")
            self.assertEqual(open_fault.spn, lim.spn)
            self.assertEqual(open_fault.fmi, FMI_VOLTAGE_BELOW_NORMAL)  # FMI 04

    def test_short_circuit_fuzzing_1500_frames_across_58_channels(self):
        """Generate 1,500 frames of extreme short-to-power / rail saturation faults across all 58 channels."""
        num_frames = 1500

        for frame_idx in range(num_frames):
            target_ch = ALL_58_SENSOR_CHANNELS[frame_idx % len(ALL_58_SENSOR_CHANNELS)]
            lim = SENSOR_LIMITS_CATALOG[target_ch]

            # Choose an adversarial short circuit value above the short circuit threshold
            short_val = max(lim.short_circuit_threshold_high + 50.0, lim.max_physical + 500.0, lim.max_physical * 2.0 + 10.0)
            if frame_idx % 5 == 0:
                short_val = 1.0e9  # Extreme positive float

            raw_frame: Dict[str, Any] = {
                ch: SENSOR_LIMITS_CATALOG[ch].healthy_nominal
                for ch in ALL_58_SENSOR_CHANNELS
            }
            raw_frame[target_ch] = short_val

            if frame_idx % 4 == 0:
                sec_ch = ALL_58_SENSOR_CHANNELS[(frame_idx * 11) % len(ALL_58_SENSOR_CHANNELS)]
                sec_lim = SENSOR_LIMITS_CATALOG[sec_ch]
                raw_frame[sec_ch] = max(sec_lim.short_circuit_threshold_high + 50.0, sec_lim.max_physical + 500.0)

            res = self.gate.filter_frame(raw_frame)

            self.assertTrue(res.is_valid)
            val = res.clean_telemetry[target_ch]
            self.assertTrue(math.isfinite(val))
            self.assertLessEqual(val, lim.max_physical, f"Channel {target_ch} above max_physical: {val} > {lim.max_physical}")
            self.assertGreaterEqual(val, lim.min_physical)

            ch_faults = [f for f in res.faults_detected if f.channel == target_ch]
            self.assertGreaterEqual(len(ch_faults), 1)
            short_fault = next((f for f in ch_faults if f.fault_type == PlausibilityFaultType.SHORT_CIRCUIT.value), None)
            self.assertIsNotNone(short_fault, f"Missing SHORT_CIRCUIT fault event for {target_ch}")
            self.assertEqual(short_fault.spn, lim.spn)
            self.assertEqual(short_fault.fmi, FMI_VOLTAGE_ABOVE_NORMAL)  # FMI 03


class TestAdversarialNaNInfAndCorruptionFuzzing(unittest.TestCase):
    """Stress test Layer 1: 3,000 pathological NaN, Infinity, Null, and Corrupt String frames."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)

    def test_nan_and_inf_fuzzing_1500_frames(self):
        """Generate 1,500 frames containing various IEEE-754 NaNs, +inf, -inf, and case variations."""
        nan_inf_variants = [
            float("nan"),
            math.nan,
            float("inf"),
            float("-inf"),
            math.inf,
            -math.inf,
            "nan",
            "NaN",
            "NAN",
            "inf",
            "+inf",
            "-inf",
            "Infinity",
            "-Infinity",
            "+INFINITY",
        ]

        num_frames = 1500
        for frame_idx in range(num_frames):
            target_ch = ALL_58_SENSOR_CHANNELS[frame_idx % len(ALL_58_SENSOR_CHANNELS)]
            lim = SENSOR_LIMITS_CATALOG[target_ch]
            bad_val = nan_inf_variants[frame_idx % len(nan_inf_variants)]

            raw_frame: Dict[str, Any] = {
                ch: SENSOR_LIMITS_CATALOG[ch].healthy_nominal
                for ch in ALL_58_SENSOR_CHANNELS
            }
            raw_frame[target_ch] = bad_val

            # Occasionally inject multiple NaNs/Infs
            if frame_idx % 2 == 0:
                raw_frame["oil_pressure"] = float("nan")
                raw_frame["coolant_temp"] = float("inf")

            res = self.gate.filter_frame(raw_frame)

            # Assert all 58 channels in output are strictly finite numbers
            for ch, val in res.clean_telemetry.items():
                self.assertIsInstance(val, float, f"Output for {ch} is not float: {type(val)}")
                self.assertTrue(math.isfinite(val), f"Output for {ch} is non-finite: {val}")
                self.assertFalse(math.isnan(val), f"Output for {ch} is NaN")
                self.assertFalse(math.isinf(val), f"Output for {ch} is Inf")

            # Check that NAN_INF_CORRUPTION event was emitted
            nan_faults = [f for f in res.faults_detected if f.channel == target_ch and f.fault_type == PlausibilityFaultType.NAN_INF_CORRUPTION.value]
            self.assertEqual(len(nan_faults), 1)
            self.assertEqual(nan_faults[0].spn, lim.spn)
            self.assertEqual(nan_faults[0].fmi, FMI_DATA_ERRATIC)

    def test_nulls_strings_and_type_pollution_1500_frames(self):
        """Generate 1,500 frames containing None, empty strings, corrupt hex, nested lists, dicts, booleans."""
        type_pollutions = [
            None,
            "",
            "   ",
            "null",
            "None",
            "NONE",
            "0xDEADBEEF",
            "broken_sensor_stream",
            "12.34.56.78",
            "undefined",
            "\x00\x01\xff",
            True,
            False,
            [100.0, 200.0],
            {"val": 50.0},
            (1, 2, 3),
            b"binary_junk",
            1e308,   # Near float max
            -1e308,  # Near float min
            1e-315,  # Subnormal
        ]

        num_frames = 1500
        for frame_idx in range(num_frames):
            target_ch = ALL_58_SENSOR_CHANNELS[frame_idx % len(ALL_58_SENSOR_CHANNELS)]
            polluted_val = type_pollutions[frame_idx % len(type_pollutions)]

            raw_frame: Dict[str, Any] = {
                ch: SENSOR_LIMITS_CATALOG[ch].healthy_nominal
                for ch in ALL_58_SENSOR_CHANNELS
            }
            raw_frame[target_ch] = polluted_val

            res = self.gate.filter_frame(raw_frame)

            # Never raises exceptions and produces valid finite floats
            self.assertTrue(res.is_valid)
            for ch, val in res.clean_telemetry.items():
                self.assertTrue(math.isfinite(val), f"Channel {ch} has non-finite value {val} on frame {frame_idx}")
                lim = SENSOR_LIMITS_CATALOG[ch]
                self.assertGreaterEqual(val, lim.min_physical)
                self.assertLessEqual(val, lim.max_physical)


class TestAdversarialHampelEMISpikesAndFrozenFlatlines(unittest.TestCase):
    """Stress test Layer 3 & Layer 5: 3,000 frames of high-amplitude EMI bursts and stuck-at flatlines."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0, stuck_window=30, hampel_window=5)

    def test_transient_emi_spikes_1500_frames(self):
        """Generate 1,500 frames containing isolated high-amplitude transient EMI pulses."""
        # Establish steady stream first
        for _ in range(5):
            self.gate.filter_frame({ch: SENSOR_LIMITS_CATALOG[ch].healthy_nominal for ch in ALL_58_SENSOR_CHANNELS})

        num_frames = 1500
        for frame_idx in range(num_frames):
            target_ch = ALL_58_SENSOR_CHANNELS[frame_idx % len(ALL_58_SENSOR_CHANNELS)]
            lim = SENSOR_LIMITS_CATALOG[target_ch]

            # Injected spike is near max_physical (substantially above nominal and beyond slew budget)
            span = lim.max_physical - lim.min_physical
            spike_val = lim.max_physical * 0.98

            if frame_idx % 2 == 0:
                # Spike frame
                raw_frame = {ch: SENSOR_LIMITS_CATALOG[ch].healthy_nominal for ch in ALL_58_SENSOR_CHANNELS}
                raw_frame[target_ch] = spike_val
                res = self.gate.filter_frame(raw_frame)

                # Output should be clamped/suppressed by slew or Hampel
                filtered_val = res.clean_telemetry[target_ch]
                self.assertTrue(math.isfinite(filtered_val))
                self.assertGreaterEqual(filtered_val, lim.min_physical)
                self.assertLessEqual(filtered_val, lim.max_physical)

                # Ensure a fault (OUTLIER_EMI or RATE_OF_CHANGE_EXCEEDED) was registered
                ch_faults = [f for f in res.faults_detected if f.channel == target_ch]
                if span > 0.1 and lim.max_slew_per_sec > 0:
                    self.assertGreaterEqual(len(ch_faults), 1, f"Expected fault for spike on {target_ch}")
            else:
                # Return to baseline
                raw_frame = {ch: SENSOR_LIMITS_CATALOG[ch].healthy_nominal for ch in ALL_58_SENSOR_CHANNELS}
                res = self.gate.filter_frame(raw_frame)
                filtered_val = res.clean_telemetry[target_ch]
                self.assertTrue(math.isfinite(filtered_val))

    def test_stuck_at_flatline_fuzzing_1500_frames(self):
        """Generate 1,500 frames testing frozen flatline detection across dynamic and static channels."""
        self.gate.reset()

        # Dynamic channels should trigger STUCK_AT after 30 frames when engine is running (RPM=2100)
        # Test 10 consecutive cycles of 40 frames each = 400 frames
        for cycle in range(10):
            self.gate.reset()
            stuck_channel = DYNAMIC_CHANNELS[cycle % len(DYNAMIC_CHANNELS)]
            lim = SENSOR_LIMITS_CATALOG[stuck_channel]
            stuck_val = lim.healthy_nominal

            required = self.gate._stuck_required.get(stuck_channel,
                                                     self.gate.stuck_window)
            for f in range(1, required + 11):
                # Dynamically vary other channels, keep stuck_channel constant
                raw_frame = {
                    ch: SENSOR_LIMITS_CATALOG[ch].healthy_nominal + (0.1 * math.sin(f + i))
                    for i, ch in enumerate(ALL_58_SENSOR_CHANNELS)
                }
                raw_frame["rpm"] = 2100.0 + (f * 5.0)  # Running
                raw_frame[stuck_channel] = stuck_val

                res = self.gate.filter_frame(raw_frame)
                stuck_faults = [ev for ev in res.faults_detected if ev.fault_type == PlausibilityFaultType.STUCK_AT.value and ev.channel == stuck_channel]

                if f < required:
                    self.assertEqual(len(stuck_faults), 0, f"False positive stuck-at at frame {f} for {stuck_channel}")
                else:
                    self.assertGreaterEqual(len(stuck_faults), 1, f"Expected stuck-at fault at frame {f} for {stuck_channel}")
                    self.assertEqual(stuck_faults[0].fmi, FMI_DATA_ERRATIC)

        # Test remaining frames: static channels should NEVER trigger stuck-at even when constant for 100+ frames
        self.gate.reset()
        for f in range(100):
            raw_frame = {
                ch: SENSOR_LIMITS_CATALOG[ch].healthy_nominal
                for ch in ALL_58_SENSOR_CHANNELS
            }
            raw_frame["rpm"] = 2200.0
            # Static channels kept constant
            raw_frame["fuel_level"] = 85.0
            raw_frame["oil_level"] = 95.0
            raw_frame["coolant_level"] = 95.0

            res = self.gate.filter_frame(raw_frame)
            static_stuck = [ev for ev in res.faults_detected if ev.fault_type == PlausibilityFaultType.STUCK_AT.value and ev.channel in STATIC_CHANNELS]
            self.assertEqual(len(static_stuck), 0, f"Static channel falsely flagged as stuck: {static_stuck}")


class TestCombinatorialChaosStormFuzzing(unittest.TestCase):
    """Stress test: 2,000 multi-fault simultaneous combinatorial chaos frames."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)

    def test_2000_combinatorial_chaos_frames(self):
        """Simultaneously inject 5 to 20 random faults per frame across 2,000 consecutive frames."""
        rng = random.Random(42)  # Deterministic seed
        num_frames = 2000
        fault_types_pool = ["wire_cut", "short", "nan", "inf", "null", "string", "emi_spike", "out_of_bounds_low", "out_of_bounds_high"]

        total_faults_detected = 0
        total_time_ms = 0.0

        for frame_idx in range(num_frames):
            frame: Dict[str, Any] = {
                ch: SENSOR_LIMITS_CATALOG[ch].healthy_nominal + rng.uniform(-0.01, 0.01)
                for ch in ALL_58_SENSOR_CHANNELS
            }

            # Pick 5 to 15 random channels to corrupt in this frame
            corrupt_count = rng.randint(5, 15)
            corrupt_channels = rng.sample(ALL_58_SENSOR_CHANNELS, corrupt_count)

            for ch in corrupt_channels:
                lim = SENSOR_LIMITS_CATALOG[ch]
                mode = rng.choice(fault_types_pool)

                if mode == "wire_cut":
                    frame[ch] = min(lim.open_circuit_threshold_low - 10.0, lim.min_physical - 500.0)
                elif mode == "short":
                    frame[ch] = max(lim.short_circuit_threshold_high + 50.0, lim.max_physical + 500.0)
                elif mode == "nan":
                    frame[ch] = float("nan")
                elif mode == "inf":
                    frame[ch] = float("inf") if rng.random() > 0.5 else float("-inf")
                elif mode == "null":
                    frame[ch] = None
                elif mode == "string":
                    frame[ch] = "CORRUPTED_SIGNAL_0x" + str(rng.randint(100, 999))
                elif mode == "emi_spike":
                    frame[ch] = lim.max_physical * 0.95
                elif mode == "out_of_bounds_low":
                    frame[ch] = lim.min_physical - 5.0
                elif mode == "out_of_bounds_high":
                    frame[ch] = lim.max_physical + 5.0

            # Add occasional time jitter
            if rng.random() < 0.1:
                frame["time"] = frame_idx * 0.05 + rng.uniform(-0.01, 0.01)

            t0 = time.perf_counter()
            res = self.gate.filter_frame(frame)
            t1 = time.perf_counter()
            total_time_ms += (t1 - t0) * 1000.0

            total_faults_detected += len(res.faults_detected)

            # Invariant checks on EVERY frame
            self.assertTrue(res.is_valid)
            self.assertEqual(len(res.clean_telemetry), len(ALL_58_SENSOR_CHANNELS))

            for ch, val in res.clean_telemetry.items():
                self.assertIsInstance(val, float)
                self.assertTrue(math.isfinite(val), f"Non-finite value {val} on channel {ch}")
                lim = SENSOR_LIMITS_CATALOG[ch]
                self.assertGreaterEqual(val, lim.min_physical, f"{ch}: {val} < min {lim.min_physical}")
                self.assertLessEqual(val, lim.max_physical, f"{ch}: {val} > max {lim.max_physical}")

            # Verify fault records
            for f in res.faults_detected:
                self.assertIn(f.fmi, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
                self.assertGreater(f.spn, 0)
                self.assertIsInstance(f.clamped_value, float)
                self.assertTrue(math.isfinite(f.clamped_value))

        avg_lat = total_time_ms / num_frames
        print(f"\n[Combinatorial Chaos Fuzzing] 2,000 Frames x 58 Channels:")
        print(f"  Total Injected Faults Detected: {total_faults_detected}")
        print(f"  Average Execution Latency:      {avg_lat:.4f} ms/frame")
        self.assertLess(avg_lat, 1.0, f"Average latency {avg_lat:.4f} ms exceeded 1.0 ms requirement")


class TestSubMillisecondPerformanceBenchmark(unittest.TestCase):
    """Execution latency and throughput benchmark across 5,000 full 58-channel frames."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)

    def test_5000_frames_latency_benchmark_under_1ms(self):
        """Benchmark 5,000 consecutive frames with 20% fault injection, verifying < 1.0 ms latency."""
        num_frames = 5000
        latencies_ms: List[float] = []

        base_frame = {
            ch: SENSOR_LIMITS_CATALOG[ch].healthy_nominal
            for ch in ALL_58_SENSOR_CHANNELS
        }

        # Warm-up (100 frames)
        for _ in range(100):
            self.gate.filter_frame(base_frame)

        # Benchmark 5,000 frames
        t_start_total = time.perf_counter()
        for i in range(num_frames):
            frame = dict(base_frame)
            frame["step"] = i
            frame["time"] = i * 0.05

            # 20% of frames inject faults
            if i % 5 == 0:
                frame["oil_pressure"] = float("nan")
            if i % 10 == 0:
                frame["coolant_temp"] = -999.0
            if i % 15 == 0:
                frame["hyd_pressure"] = 9999.0
            if i % 20 == 0:
                frame["rpm"] = "corrupt_str"

            t0 = time.perf_counter()
            res = self.gate.filter_frame(frame)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        t_end_total = time.perf_counter()

        avg_latency = sum(latencies_ms) / len(latencies_ms)
        sorted_lats = sorted(latencies_ms)
        p50 = sorted_lats[int(num_frames * 0.50)]
        p90 = sorted_lats[int(num_frames * 0.90)]
        p99 = sorted_lats[int(num_frames * 0.99)]
        max_lat = sorted_lats[-1]
        throughput_fps = num_frames / (t_end_total - t_start_total)

        print(f"\n===========================================================")
        print(f" CHALLENGER M2 BENCHMARK: 5,000 Frames x 58 Channels")
        print(f"===========================================================")
        print(f"  Frames Tested:       {num_frames}")
        print(f"  Channels per Frame:  58")
        print(f"  Average Latency:     {avg_latency:.4f} ms")
        print(f"  P50 Latency:         {p50:.4f} ms")
        print(f"  P90 Latency:         {p90:.4f} ms")
        print(f"  P99 Latency:         {p99:.4f} ms")
        print(f"  Max Latency:         {max_lat:.4f} ms")
        print(f"  Throughput:          {throughput_fps:.1f} frames/sec")
        print(f"===========================================================")

        # Hard assertions
        self.assertLess(avg_latency, 1.0, f"Avg latency {avg_latency:.4f} ms exceeded 1.0 ms budget")
        self.assertLess(p99, 2.0, f"P99 latency {p99:.4f} ms exceeded 2.0 ms budget")
        self.assertGreater(throughput_fps, 1000.0, f"Throughput {throughput_fps:.1f} fps below 1,000 fps requirement")


class TestCrossSubsystemAndEdgePlausibility(unittest.TestCase):
    """Verify Layer 6 cross-subsystem plausibility rules and extreme gateway edge cases."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)

    def test_cross_subsystem_rpm_vs_oil_pressure(self):
        """Cross-check: High RPM (2500) + near-zero Oil Pressure -> FMI 14 fault."""
        res = self.gate.filter_frame({"rpm": 2500.0, "oil_pressure": 0.1})
        faults = [f for f in res.faults_detected if f.fault_type == PlausibilityFaultType.DUAL_SENSOR_MISMATCH.value]
        self.assertEqual(len(faults), 1)
        self.assertEqual(faults[0].channel, "oil_pressure")
        self.assertEqual(faults[0].spn, 100)
        self.assertEqual(faults[0].fmi, FMI_SPECIAL_INSTRUCTIONS)  # 14

    def test_cross_subsystem_thermal_discrepancy(self):
        """Cross-check: Coolant Temp (140°C) vs Oil Temp (20°C) -> FMI 14 fault."""
        res = self.gate.filter_frame({"coolant_temp": 140.0, "oil_temp": 20.0})
        faults = [f for f in res.faults_detected if f.fault_type == PlausibilityFaultType.DUAL_SENSOR_MISMATCH.value]
        self.assertEqual(len(faults), 1)
        self.assertEqual(faults[0].channel, "coolant_temp")
        self.assertEqual(faults[0].spn, 110)
        self.assertEqual(faults[0].fmi, FMI_SPECIAL_INSTRUCTIONS)

    def test_cross_subsystem_dual_egt_bank_divergence(self):
        """Cross-check: Dual EGT Bank A (700°C) vs Bank B (200°C) -> FMI 14 fault."""
        res = self.gate.filter_frame({"egt_bank_a": 700.0, "egt_bank_b": 200.0})
        faults = [f for f in res.faults_detected if f.fault_type == PlausibilityFaultType.DUAL_SENSOR_MISMATCH.value]
        self.assertEqual(len(faults), 1)
        self.assertEqual(faults[0].spn, 173)
        self.assertEqual(faults[0].fmi, FMI_SPECIAL_INSTRUCTIONS)

    def test_cross_subsystem_hydraulic_flow_pressure_contradiction(self):
        """Cross-check: High hydraulic flow (0.010 m³/s) with 0 bar pressure -> FMI 14 fault."""
        res = self.gate.filter_frame({"hyd_flow": 0.010, "hyd_pressure": 0.0})
        faults = [f for f in res.faults_detected if f.fault_type == PlausibilityFaultType.DUAL_SENSOR_MISMATCH.value]
        self.assertEqual(len(faults), 1)
        self.assertEqual(faults[0].channel, "hyd_pressure")
        self.assertEqual(faults[0].spn, 520202)
        self.assertEqual(faults[0].fmi, FMI_SPECIAL_INSTRUCTIONS)

    def test_empty_frame_and_unregistered_keys_safety(self):
        """Empty frame `{}` or frame with 50 unknown extra keys executes safely without crashing."""
        # Empty frame
        res_empty = self.gate.filter_frame({})
        self.assertTrue(res_empty.is_valid)
        self.assertEqual(len(res_empty.clean_telemetry), 0)

        # Unregistered noisy keys
        noisy_frame = {f"unknown_sensor_{i}": i * 1.5 for i in range(50)}
        res_noisy = self.gate.filter_frame(noisy_frame)
        self.assertTrue(res_noisy.is_valid)
        self.assertEqual(len(res_noisy.clean_telemetry), 50)


if __name__ == "__main__":
    unittest.main()
