"""Comprehensive Unit & Adversarial Test Suite for Sensor Plausibility Gate (FDIR).

Tests multi-layer signal plausibility, NaN/Inf sanitization, adversarial electrical faults
(wire cut open-circuit, short-to-power), slew-rate limiting, stuck-at / flatline detection,
sliding-window Hampel EMI outlier filtering, cross-subsystem dual-sensor correlations,
and sub-millisecond execution latency benchmarks.
"""

import collections
import math
import time
import unittest
from typing import Any, Dict, List

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

# 58 standard AFV sensor channels from tank physics model
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


class TestNaNInfNullSanitization(unittest.TestCase):
    """Layer 1: Range Clamping & NaN/Inf/Null/Corrupted Data Sanitization."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)

    def test_nan_values_sanitized_to_nominal_or_last_valid(self):
        """Verify float('nan') is caught, sanitized, and flagged as NaN corruption fault."""
        raw = {"rpm": float("nan"), "oil_pressure": 4.5, "coolant_temp": 88.0}
        res = self.gate.filter_frame(raw)

        self.assertTrue(res.is_valid)
        self.assertFalse(math.isnan(res.clean_telemetry["rpm"]))
        self.assertEqual(res.clean_telemetry["rpm"], 1800.0)  # Healthy nominal
        self.assertEqual(res.clean_telemetry["oil_pressure"], 4.5)

        # Verify fault event
        nan_faults = [f for f in res.faults_detected if f.channel == "rpm"]
        self.assertEqual(len(nan_faults), 1)
        self.assertEqual(nan_faults[0].fault_type, PlausibilityFaultType.NAN_INF_CORRUPTION.value)
        self.assertEqual(nan_faults[0].spn, 190)
        self.assertEqual(nan_faults[0].fmi, FMI_DATA_ERRATIC)

    def test_positive_and_negative_inf_sanitization(self):
        """Verify +inf and -inf are replaced by nominal and flagged."""
        raw = {
            "coolant_temp": float("inf"),
            "oil_temp": float("-inf"),
            "shaft_torque": 450.0,
        }
        res = self.gate.filter_frame(raw)

        self.assertTrue(math.isfinite(res.clean_telemetry["coolant_temp"]))
        self.assertTrue(math.isfinite(res.clean_telemetry["oil_temp"]))
        self.assertEqual(res.clean_telemetry["coolant_temp"], 88.0)
        self.assertEqual(res.clean_telemetry["oil_temp"], 85.0)

        fault_channels = {f.channel for f in res.faults_detected}
        self.assertIn("coolant_temp", fault_channels)
        self.assertIn("oil_temp", fault_channels)

    def test_none_and_corrupt_string_sanitization(self):
        """Verify None, empty string, string representations 'NaN', 'null' are sanitized safely."""
        raw = {
            "rpm": None,
            "oil_pressure": "NaN",
            "coolant_temp": "null",
            "hyd_pressure": "",
            "shaft_torque": "corrupted_hex_0xDEAD",
        }
        res = self.gate.filter_frame(raw)

        for k in raw:
            self.assertIn(k, res.clean_telemetry)
            self.assertTrue(math.isfinite(res.clean_telemetry[k]))
            self.assertIsInstance(res.clean_telemetry[k], float)

    def test_nested_or_non_primitive_types_sanitized(self):
        """Verify nested lists, dictionaries, or objects do not crash the filter and are sanitized."""
        raw = {
            "rpm": [2100.0, 2200.0],
            "oil_pressure": {"val": 4.5},
            "coolant_temp": 85.0,
        }
        res = self.gate.filter_frame(raw)
        self.assertTrue(math.isfinite(res.clean_telemetry["rpm"]))
        self.assertTrue(math.isfinite(res.clean_telemetry["oil_pressure"]))
        self.assertEqual(res.clean_telemetry["coolant_temp"], 85.0)


class TestElectricalOpenCircuitDetection(unittest.TestCase):
    """Layer 2: Electrical Open-Circuit & Wire-Cut Fault Detection (J1939 FMI 04)."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)

    def test_adversarial_wire_cut_oil_pressure(self):
        """Adversarial wire cut: P_oil = -999.0 bar -> clamped to min_physical and J1939 FMI 04 emitted."""
        raw = {"oil_pressure": -999.0, "rpm": 1800.0}
        res = self.gate.filter_frame(raw)

        # Clamped to physical lower bound
        self.assertEqual(res.clean_telemetry["oil_pressure"], 0.0)
        self.assertGreaterEqual(res.clean_telemetry["oil_pressure"], 0.0)

        # Verify J1939 SPN 100 / FMI 04 fault event
        faults = [f for f in res.faults_detected if f.channel == "oil_pressure"]
        self.assertTrue(len(faults) >= 1)
        open_circuit_fault = next(f for f in faults if f.fault_type == PlausibilityFaultType.OPEN_CIRCUIT.value)
        self.assertEqual(open_circuit_fault.spn, 100)
        self.assertEqual(open_circuit_fault.fmi, FMI_VOLTAGE_BELOW_NORMAL)  # 4
        self.assertEqual(open_circuit_fault.clamped_value, 0.0)

    def test_adversarial_wire_cut_coolant_temp(self):
        """Adversarial wire cut: coolant_temp = -999.0 °C -> clamped to -40°C and J1939 FMI 04 emitted."""
        raw = {"coolant_temp": -999.0}
        res = self.gate.filter_frame(raw)

        self.assertEqual(res.clean_telemetry["coolant_temp"], -40.0)
        fault = next(f for f in res.faults_detected if f.channel == "coolant_temp")
        self.assertEqual(fault.fault_type, PlausibilityFaultType.OPEN_CIRCUIT.value)
        self.assertEqual(fault.spn, 110)
        self.assertEqual(fault.fmi, FMI_VOLTAGE_BELOW_NORMAL)

    def test_adversarial_wire_cut_rpm(self):
        """Adversarial wire cut on crank speed sensor: rpm = -999.0 -> clamped to 0 RPM."""
        raw = {"rpm": -999.0}
        res = self.gate.filter_frame(raw)

        self.assertEqual(res.clean_telemetry["rpm"], 0.0)
        fault = next(f for f in res.faults_detected if f.channel == "rpm")
        self.assertEqual(fault.fault_type, PlausibilityFaultType.OPEN_CIRCUIT.value)
        self.assertEqual(fault.spn, 190)
        self.assertEqual(fault.fmi, FMI_VOLTAGE_BELOW_NORMAL)


class TestElectricalShortCircuitDetection(unittest.TestCase):
    """Layer 2: Electrical Short-Circuit & Rail Saturation Detection (J1939 FMI 03)."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)

    def test_adversarial_short_to_power_oil_pressure(self):
        """Short circuit: P_oil = +999.0 bar -> clamped to max_physical and J1939 FMI 03 emitted."""
        raw = {"oil_pressure": 999.0}
        res = self.gate.filter_frame(raw)

        # Clamped to physical upper bound
        self.assertEqual(res.clean_telemetry["oil_pressure"], 15.0)

        # Verify J1939 SPN 100 / FMI 03
        fault = next(f for f in res.faults_detected if f.channel == "oil_pressure")
        self.assertEqual(fault.fault_type, PlausibilityFaultType.SHORT_CIRCUIT.value)
        self.assertEqual(fault.spn, 100)
        self.assertEqual(fault.fmi, FMI_VOLTAGE_ABOVE_NORMAL)  # 3

    def test_adversarial_short_to_power_coolant_temp(self):
        """Short circuit: coolant_temp = +999.0 °C -> clamped to 160°C and J1939 FMI 03 emitted."""
        raw = {"coolant_temp": 999.0}
        res = self.gate.filter_frame(raw)

        self.assertEqual(res.clean_telemetry["coolant_temp"], 160.0)
        fault = next(f for f in res.faults_detected if f.channel == "coolant_temp")
        self.assertEqual(fault.fault_type, PlausibilityFaultType.SHORT_CIRCUIT.value)
        self.assertEqual(fault.spn, 110)
        self.assertEqual(fault.fmi, FMI_VOLTAGE_ABOVE_NORMAL)

    def test_adversarial_short_to_power_hyd_pressure(self):
        """Short circuit on hydraulic sensor: hyd_pressure = 9999.0 bar -> clamped to max physical."""
        raw = {"hyd_pressure": 9999.0}
        res = self.gate.filter_frame(raw)

        self.assertEqual(res.clean_telemetry["hyd_pressure"], 400.0)
        fault = next(f for f in res.faults_detected if f.channel == "hyd_pressure")
        self.assertEqual(fault.fault_type, PlausibilityFaultType.SHORT_CIRCUIT.value)
        self.assertEqual(fault.spn, 520202)
        self.assertEqual(fault.fmi, FMI_VOLTAGE_ABOVE_NORMAL)


class TestSlewRateLimiter(unittest.TestCase):
    """Layer 3: Slew-Rate / Rate-of-Change Limiting & Step-Spike Clamping (J1939 FMI 02)."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)  # dt = 0.05s

    def test_step_change_exceeding_max_slew_clamped_safely(self):
        """Verify unphysical jump in coolant temperature (e.g. +50°C in 50ms) is slew-rate clamped."""
        # Initial steady state frame
        self.gate.filter_frame({"coolant_temp": 80.0})

        # Sudden jump to 130.0 °C (max slew is 15°C/s -> max step is 15 * 0.05 = 0.75°C)
        res = self.gate.filter_frame({"coolant_temp": 130.0})

        self.assertAlmostEqual(res.clean_telemetry["coolant_temp"], 80.0 + 0.75, places=2)
        roc_faults = [f for f in res.faults_detected if f.fault_type == PlausibilityFaultType.RATE_OF_CHANGE_EXCEEDED.value]
        self.assertEqual(len(roc_faults), 1)
        self.assertEqual(roc_faults[0].channel, "coolant_temp")
        self.assertEqual(roc_faults[0].fmi, FMI_DATA_ERRATIC)

    def test_smooth_ramp_within_slew_limit_not_flagged(self):
        """Verify normal gradual temperature rise is not flagged as rate fault."""
        self.gate.filter_frame({"coolant_temp": 80.0})
        res = self.gate.filter_frame({"coolant_temp": 80.3})  # 0.3°C step < 0.75°C max

        self.assertEqual(res.clean_telemetry["coolant_temp"], 80.3)
        roc_faults = [f for f in res.faults_detected if f.fault_type == PlausibilityFaultType.RATE_OF_CHANGE_EXCEEDED.value]
        self.assertEqual(len(roc_faults), 0)

    def test_square_wave_slew_tracking(self):
        """Verify slew limiter gradually tracks a large square step over multiple consecutive frames."""
        self.gate.filter_frame({"coolant_temp": 80.0})
        target = 100.0
        val = 80.0
        for step in range(10):
            res = self.gate.filter_frame({"coolant_temp": target})
            new_val = res.clean_telemetry["coolant_temp"]
            self.assertGreater(new_val, val)
            self.assertLessEqual(new_val, target)
            val = new_val


class TestStuckAtFlatlineDetection(unittest.TestCase):
    """Layer 4: Stuck-At / Frozen Sensor Flatline Detection (J1939 FMI 02)."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0, stuck_window=30)

    def test_stuck_at_sensor_flagged_after_its_channel_tolerance(self):
        """A frozen dynamic sensor triggers STUCK_AT once it exceeds that
        channel's flatline tolerance.

        The tolerance is per channel and expressed in time, not a blanket frame
        count: replaying a real CAN trace showed a thermostat-regulated coolant
        temperature holding one quantised value for minutes, which a universal
        30-frame rule reported as a dead sensor on 69% of healthy frames.
        Shaft torque is genuinely dynamic under load, so it keeps the tight
        default window.
        """
        stuck_val = 88.423
        required = self.gate._stuck_required.get("shaft_torque",
                                                 self.gate.stuck_window)
        detected = False

        for frame_idx in range(1, required + 5):
            res = self.gate.filter_frame({
                "rpm": 2100.0 + (frame_idx * 2.0),  # Vehicle is running dynamically
                "shaft_torque": stuck_val,          # Frozen sensor
            })
            stuck_faults = [f for f in res.faults_detected
                            if f.fault_type == PlausibilityFaultType.STUCK_AT.value
                            and f.channel == "shaft_torque"]
            if frame_idx < required:
                self.assertEqual(len(stuck_faults), 0, f"False positive at frame {frame_idx}")
            else:
                self.assertGreaterEqual(len(stuck_faults), 1, f"Expected stuck-at fault at frame {frame_idx}")
                self.assertEqual(stuck_faults[0].fmi, FMI_DATA_ERRATIC)
                detected = True

        self.assertTrue(detected)

    def test_regulated_channel_may_hold_a_constant_value(self):
        """Coolant temperature must not be flagged merely for being steady."""
        for frame_idx in range(1, 60):
            res = self.gate.filter_frame({
                "rpm": 2100.0 + (frame_idx * 2.0),
                "coolant_temp": 88.423,
            })
            self.assertEqual(
                [f for f in res.faults_detected
                 if f.fault_type == PlausibilityFaultType.STUCK_AT.value
                 and f.channel == "coolant_temp"], [])

    def test_static_channels_or_engine_off_do_not_falsely_trigger_stuck(self):
        """When engine is off (RPM=0) or channel is static (fuel_level), constant value does not false alarm."""
        for _ in range(35):
            res = self.gate.filter_frame({
                "rpm": 0.0,
                "fuel_level": 85.0,
                "oil_level": 95.0,
            })
            stuck_faults = [f for f in res.faults_detected if f.fault_type == PlausibilityFaultType.STUCK_AT.value]
            self.assertEqual(len(stuck_faults), 0)

    def test_normal_dynamic_signal_does_not_trigger_stuck(self):
        """Varying sensor values do not trigger stuck-at filter."""
        for i in range(35):
            res = self.gate.filter_frame({
                "rpm": 2000.0 + (5.0 * (i % 7)),
                "coolant_temp": 85.0 + (0.05 * (i % 5)),
            })
            stuck_faults = [f for f in res.faults_detected if f.fault_type == PlausibilityFaultType.STUCK_AT.value]
            self.assertEqual(len(stuck_faults), 0)


class TestHampelEMIOutlierFilter(unittest.TestCase):
    """Layer 5: Sliding-Window Hampel / Median Outlier & Transient EMI Burst Filter."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0, hampel_window=5, hampel_nsigmas=3.0)

    def test_transient_isolated_emi_spike_suppressed(self):
        """Feed steady stream of 5.0 bar, inject a single-sample EMI spike of 45.0 bar, verify suppression."""
        # Establish baseline history
        for _ in range(5):
            self.gate.filter_frame({"oil_pressure": 5.0})

        # Inject transient spike
        spike_res = self.gate.filter_frame({"oil_pressure": 45.0})

        # Slew rate & Hampel filter should suppress the spike back towards median 5.0
        self.assertLess(spike_res.clean_telemetry["oil_pressure"], 10.0)

        fault_types = {f.fault_type for f in spike_res.faults_detected}
        self.assertTrue(
            PlausibilityFaultType.OUTLIER_EMI.value in fault_types or
            PlausibilityFaultType.RATE_OF_CHANGE_EXCEEDED.value in fault_types
        )

        # Subsequent reading returns to normal
        next_res = self.gate.filter_frame({"oil_pressure": 5.0})
        self.assertAlmostEqual(next_res.clean_telemetry["oil_pressure"], 5.0, places=1)


class TestDualSensorCrossSubsystemPlausibility(unittest.TestCase):
    """Layer 6: Dual-Sensor & Cross-Subsystem Physical Plausibility Correlation Gate."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)

    def test_high_rpm_with_zero_oil_pressure_triggers_mismatch(self):
        """Engine RPM = 2500 with Oil Pressure = 0.0 bar triggers cross-subsystem mismatch."""
        raw = {"rpm": 2500.0, "oil_pressure": 0.0}
        res = self.gate.filter_frame(raw)

        mismatch_faults = [f for f in res.faults_detected if f.fault_type == PlausibilityFaultType.DUAL_SENSOR_MISMATCH.value]
        self.assertGreaterEqual(len(mismatch_faults), 1)
        self.assertEqual(mismatch_faults[0].channel, "oil_pressure")
        self.assertEqual(mismatch_faults[0].spn, 100)
        self.assertEqual(mismatch_faults[0].fmi, FMI_SPECIAL_INSTRUCTIONS)  # 14

    def test_engine_stopped_with_high_oil_pressure_triggers_mismatch(self):
        """Engine RPM = 0 with residual Oil Pressure = 8.0 bar triggers mismatch."""
        raw = {"rpm": 0.0, "oil_pressure": 8.0}
        res = self.gate.filter_frame(raw)

        mismatch_faults = [f for f in res.faults_detected if f.fault_type == PlausibilityFaultType.DUAL_SENSOR_MISMATCH.value]
        self.assertGreaterEqual(len(mismatch_faults), 1)
        self.assertEqual(mismatch_faults[0].channel, "oil_pressure")

    def test_extreme_coolant_oil_temp_divergence_triggers_mismatch(self):
        """Coolant Temp = 125°C but Oil Temp = 10°C (delta = 115°C) triggers thermal mismatch."""
        raw = {"coolant_temp": 125.0, "oil_temp": 10.0}
        res = self.gate.filter_frame(raw)

        mismatch_faults = [f for f in res.faults_detected if f.fault_type == PlausibilityFaultType.DUAL_SENSOR_MISMATCH.value]
        self.assertGreaterEqual(len(mismatch_faults), 1)
        self.assertEqual(mismatch_faults[0].channel, "coolant_temp")
        self.assertEqual(mismatch_faults[0].spn, 110)

    def test_dual_egt_bank_a_vs_bank_b_discrepancy_triggers_mismatch(self):
        """Dual EGT Bank A = 650°C and Bank B = 200°C (delta = 450°C) triggers EGT mismatch."""
        raw = {"egt_bank_a": 650.0, "egt_bank_b": 200.0}
        res = self.gate.filter_frame(raw)

        mismatch_faults = [f for f in res.faults_detected if f.fault_type == PlausibilityFaultType.DUAL_SENSOR_MISMATCH.value]
        self.assertGreaterEqual(len(mismatch_faults), 1)
        self.assertEqual(mismatch_faults[0].spn, 173)

    def test_hydraulic_high_flow_zero_pressure_triggers_mismatch(self):
        """Hydraulic Flow = 0.008 m³/s with Circuit Pressure = 0 bar triggers hydraulic leak/sever mismatch."""
        raw = {"hyd_flow": 0.008, "hyd_pressure": 0.0}
        res = self.gate.filter_frame(raw)

        mismatch_faults = [f for f in res.faults_detected if f.fault_type == PlausibilityFaultType.DUAL_SENSOR_MISMATCH.value]
        self.assertGreaterEqual(len(mismatch_faults), 1)
        self.assertEqual(mismatch_faults[0].channel, "hyd_pressure")
        self.assertEqual(mismatch_faults[0].spn, 520202)


class TestAll58AFVSensorChannelsEnvelopes(unittest.TestCase):
    """Exhaustive coverage of all 58 AFV sensor channels across normal, open, short, and NaN cases."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)

    def test_all_58_channels_have_limits_configured(self):
        """Verify every single AFV channel defined in SENSOR_COLUMNS is registered in the catalog."""
        for ch in ALL_58_SENSOR_CHANNELS:
            lim = self.gate.get_sensor_limits(ch)
            self.assertIsNotNone(lim, f"Missing SensorLimits catalog entry for channel: '{ch}'")
            self.assertGreater(lim.max_physical, lim.min_physical, f"Invalid envelope for {ch}")
            self.assertGreater(lim.spn, 0, f"Invalid SPN for {ch}")

    def test_full_58_channel_nominal_frame_passes_cleanly(self):
        """Construct full 58-channel nominal frame and verify 100% clean output."""
        nominal_frame = {
            ch: self.gate.get_sensor_limits(ch).healthy_nominal
            for ch in ALL_58_SENSOR_CHANNELS
        }
        res = self.gate.filter_frame(nominal_frame)
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.clean_telemetry), len(ALL_58_SENSOR_CHANNELS))
        for ch, val in res.clean_telemetry.items():
            self.assertTrue(math.isfinite(val), f"Non-finite value for channel {ch}")

    def test_full_58_channel_adversarial_fuzzing_corpus(self):
        """Fuzz all 58 channels simultaneously with extreme positive, negative, and NaN inputs."""
        fuzzed_frame = {}
        for idx, ch in enumerate(ALL_58_SENSOR_CHANNELS):
            case = idx % 4
            if case == 0:
                fuzzed_frame[ch] = float("nan")
            elif case == 1:
                fuzzed_frame[ch] = -99999.0  # Extreme wire cut
            elif case == 2:
                fuzzed_frame[ch] = 99999.0   # Extreme short to rail
            else:
                fuzzed_frame[ch] = "invalid_string_fuzz"

        res = self.gate.filter_frame(fuzzed_frame)
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.clean_telemetry), len(ALL_58_SENSOR_CHANNELS))

        for ch, val in res.clean_telemetry.items():
            lim = self.gate.get_sensor_limits(ch)
            self.assertTrue(math.isfinite(val), f"Channel {ch} produced non-finite output: {val}")
            self.assertGreaterEqual(val, lim.min_physical, f"Channel {ch} below min_physical: {val} < {lim.min_physical}")
            self.assertLessEqual(val, lim.max_physical, f"Channel {ch} above max_physical: {val} > {lim.max_physical}")


class TestPerformanceAndExecutionLatencyBenchmark(unittest.TestCase):
    """Performance requirement verification: single frame filtering execution time must be < 1.0 ms."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)

    def test_58_channel_single_frame_filtering_latency_under_1ms(self):
        """Filter 1000 consecutive 58-channel frames and verify average latency is < 1.0 ms."""
        # Create representative test frame
        test_frame = {
            ch: self.gate.get_sensor_limits(ch).healthy_nominal
            for ch in ALL_58_SENSOR_CHANNELS
        }

        # Warm-up
        for _ in range(50):
            self.gate.filter_frame(test_frame)

        # Benchmark 1000 cycles
        num_cycles = 1000
        latencies_ms = []

        t_total_start = time.perf_counter()
        for i in range(num_cycles):
            # Introduce occasional dynamic perturbations
            frame = dict(test_frame)
            if i % 10 == 0:
                frame["oil_pressure"] = float("nan")
            if i % 25 == 0:
                frame["coolant_temp"] = -999.0

            t0 = time.perf_counter()
            res = self.gate.filter_frame(frame)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
        t_total_end = time.perf_counter()

        avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
        sorted_latencies = sorted(latencies_ms)
        p50 = sorted_latencies[int(num_cycles * 0.50)]
        p90 = sorted_latencies[int(num_cycles * 0.90)]
        p99 = sorted_latencies[int(num_cycles * 0.99)]
        max_lat = max(latencies_ms)
        total_fps = num_cycles / (t_total_end - t_total_start)

        print(f"\n[FDIR Plausibility Benchmark] 58 Channels x {num_cycles} Frames:")
        print(f"  Avg Latency: {avg_latency_ms:.4f} ms | P50: {p50:.4f} ms | P90: {p90:.4f} ms | P99: {p99:.4f} ms | Max: {max_lat:.4f} ms")
        print(f"  Throughput:  {total_fps:.1f} frames/sec")

        # Hard assertion: Average latency MUST be well below 1.0 ms requirement (typically < 0.25 ms)
        self.assertLess(avg_latency_ms, 1.0, f"Average execution latency {avg_latency_ms:.4f} ms exceeded 1.0 ms budget!")
        self.assertLess(p99, 2.0, f"P99 latency {p99:.4f} ms exceeded 2.0 ms budget!")


class TestStateResetAndCustomLimits(unittest.TestCase):
    """Test state reset, custom channel registration, and helper methods."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)

    def test_reset_clears_histories_and_state(self):
        """Verify gate.reset() clears all internal rolling buffers and timestamps."""
        self.gate.filter_frame({"rpm": 2000.0, "oil_pressure": 4.5})
        self.assertGreater(len(self.gate.last_valid), 0)
        self.assertGreater(self.gate.frame_count, 0)

        self.gate.reset()
        self.assertEqual(len(self.gate.last_valid), 0)
        self.assertEqual(len(self.gate.stuck_history), 0)
        self.assertEqual(len(self.gate.hampel_history), 0)
        self.assertEqual(self.gate.frame_count, 0)

    def test_custom_limit_registration(self):
        """Verify custom SensorLimits can be registered and applied."""
        custom = SensorLimits(
            channel="custom_laser_gyro_deg_s",
            spn=520901,
            min_physical=-50.0,
            max_physical=50.0,
            healthy_nominal=0.0,
            max_slew_per_sec=20.0,
            open_circuit_threshold_low=-100.0,
            short_circuit_threshold_high=100.0,
            unit="deg/s",
        )
        self.gate.register_custom_limit(custom)
        res = self.gate.filter_frame({"custom_laser_gyro_deg_s": 999.0})
        self.assertEqual(res.clean_telemetry["custom_laser_gyro_deg_s"], 50.0)
        fault = next(f for f in res.faults_detected if f.channel == "custom_laser_gyro_deg_s")
        self.assertEqual(fault.spn, 520901)
        self.assertEqual(fault.fmi, FMI_VOLTAGE_ABOVE_NORMAL)

    def test_validate_packet_tuple_return(self):
        """Verify validate_packet returns (clean_telemetry, faults_detected) tuple."""
        clean, faults = self.gate.validate_packet({"rpm": 2100.0, "oil_pressure": -999.0})
        self.assertEqual(clean["rpm"], 2100.0)
        self.assertEqual(clean["oil_pressure"], 0.0)
        self.assertGreaterEqual(len(faults), 1)


if __name__ == "__main__":
    unittest.main()
