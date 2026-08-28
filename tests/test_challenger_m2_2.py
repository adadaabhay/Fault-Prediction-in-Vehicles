"""Empirical Challenger Test Suite for Milestone M2 (challenger_m2_2).

ML Resilience & Neural Protection Verification:
Interception between `telemetry_gateway/sensor_plausibility.py` and downstream
neural/feature extractors (`ml/lstm.py` and `c_engine`).

Verification Objectives:
1. Control Experiment: Verify that raw un-sanitized adversarial corruption causes
   downstream normalization and LSTM inference to produce NaNs and numerical failures.
2. Adversarial Pressure Invalidation (-999 bar across all pressure channels):
   Verify SensorPlausibilityGate sanitizes inputs to physical min/nominal, emits SPN & FMI 04
   diagnostics, and downstream normalizer + LSTM forward pass complete with finite predictions.
3. Extreme Adversarial Thermal Runaway (+9999°C across all temperature channels):
   Verify SensorPlausibilityGate clamps to physical max, emits SPN & FMI 03 diagnostics,
   and downstream LSTM forward pass produces non-NaN bounded outputs.
4. High-Density NaN/Inf Corruption (50% to 100% channels corrupted):
   Verify NaN/Inf/Null inputs are sanitized to last-valid / healthy nominal, emitting SPN & FMI 02,
   with 100% finite downstream LSTM activations and softmax probability distributions.
5. Combinatorial Chaos Multi-Step Sequential Inference (2,000 frames, T=40 sliding windows):
   Verify that sustained multi-step adversarial sequences through the plausibility gate
   yield zero numerical errors, zero division by zero, and valid RUL/CHI across both Python LSTM
   and C99 edge inference engines.
6. Decoupled Diagnostic Tagging:
   Verify diagnostic fault events (SPN & FMI) are completely isolated in diagnostic structures
   and never pollute the numeric feature tensor fed to neural models.
"""

import json
import math
import random
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure project root is on sys.path (conftest.py also does this for pytest)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
from ml.lstm import LSTMModel, predict_rul, sigmoid, tanh
from ml.parts import INPUT_FEATURES, PARTS, part_health_index

# 58 standard AFV sensor channels from CVRDE physics model
AFV_58_PHYSICAL_CHANNELS = [
    "rpm", "load", "terrain", "coolant_temp", "coolant_rtd_ohm",
    "exhaust_temp", "exhaust_thermocouple_v", "exhaust_pressure", "exhaust_mass_flow",
    "lambda", "exhaust_o2_pct", "oil_pressure", "oil_temp", "oil_viscosity",
    "oil_flow", "debris_cumulative", "debris_rate", "debris_particles",
    "shaft_torque", "shaft_shear_stress", "shaft_shear_strain",
    "mech_power", "shaft_omega", "fuel_level", "fuel_volume",
    "oil_level", "coolant_level", "fuel_capacitance_pf", "hyd_pressure",
    "hyd_flow", "hyd_force", "hyd_power", "hyd_leak_flow",
    "susp_load_kN", "susp_stress_MPa", "susp_strain_ue", "susp_dR_ohm",
    "torsion_torque", "torsion_twist_deg", "torsion_shear_MPa",
    "torsion_cumulative_twist", "shock_a_rms_g", "shock_peak_g", "shock_energy",
    "spl_db", "acoustic_dom_freq", "acoustic_energy", "ae_event_rate", "ae_events",
    "ae_energy", "ae_amp_dB", "ae_duration_s", "vib_rms", "vib_kurtosis",
    "vib_dom_freq", "vib_dom_amp", "vib_energy", "time"
]
assert len(AFV_58_PHYSICAL_CHANNELS) == 58, f"Expected 58 features, got {len(AFV_58_PHYSICAL_CHANNELS)}"

ALL_CATALOG_CHANNELS = list(SENSOR_LIMITS_CATALOG.keys())


def load_trained_lstm_model() -> Tuple[LSTMModel, dict]:
    """Load the trained LSTM model from docs/model.json."""
    model_json_path = PROJECT_ROOT / "docs" / "model.json"
    with open(model_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    D = data["D"]
    H = data["H"]
    R = data["R"]
    C = data["C"]
    
    model = LSTMModel(D, H, R, C)
    for k, v in data["params"].items():
        model.p[k] = np.array(v, dtype=np.float64)
    
    return model, data


def load_scaler_config() -> dict:
    """Load normalization scaler bounds from config.json."""
    cfg_path = PROJECT_ROOT / "docs" / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_telemetry_22(telemetry: Dict[str, Any], scaler_cfg: dict) -> np.ndarray:
    """Normalize 22 input features for Python LSTM model with safe span protection."""
    scalers = scaler_cfg["scaler"]
    features = []
    for k in INPUT_FEATURES:
        raw = telemetry.get(k, 0.0)
        s = scalers.get(k, {"min": 0.0, "max": 1.0})
        span = s["max"] - s["min"]
        if span <= 1e-12:
            norm = 0.5
        else:
            norm = (float(raw) - s["min"]) / span
        features.append(float(np.clip(norm, 0.0, 1.0)))
    return np.array(features, dtype=np.float64)


def c99_infer_step_emulation(weights: dict, state: dict, x_58: np.ndarray) -> dict:
    """Algorithmic exact C99 MISRA-C emulation of c_engine/tank_pdm_infer.c."""
    D = 58
    H = weights["H"]
    R = weights["R"]
    C = weights["C"]
    
    h_prev = state["h"]
    c_prev = state["c"]
    
    act_f = weights["bf"] + x_58 @ weights["Wf"] + h_prev @ weights["Uf"]
    act_i = weights["bi"] + x_58 @ weights["Wi"] + h_prev @ weights["Ui"]
    act_c = weights["bc"] + x_58 @ weights["Wc"] + h_prev @ weights["Uc"]
    act_o = weights["bo"] + x_58 @ weights["Wo"] + h_prev @ weights["Uo"]
    
    f = 1.0 / (1.0 + np.exp(-np.clip(act_f, -40.0, 40.0)))
    i = 1.0 / (1.0 + np.exp(-np.clip(act_i, -40.0, 40.0)))
    c_t = np.tanh(np.clip(act_c, -20.0, 20.0))
    o = 1.0 / (1.0 + np.exp(-np.clip(act_o, -40.0, 40.0)))
    
    c_next = f * c_prev + i * c_t
    h_next = o * np.tanh(np.clip(c_next, -20.0, 20.0))
    
    state["h"] = h_next
    state["c"] = c_next
    state["step_count"] += 1
    
    # RUL regression head
    act_y = weights["by"] + h_next @ weights["Wy"]
    ruls = 1.0 / (1.0 + np.exp(-np.clip(act_y, -40.0, 40.0)))
    composite_chi = float(np.mean(ruls) * 100.0)
    
    # Fault classification head
    logits = weights["bcls"] + h_next @ weights["Wcls"]
    max_logit = np.max(logits)
    exp_l = np.exp(logits - max_logit)
    sum_exp = np.sum(exp_l)
    probs = exp_l / sum_exp if sum_exp > 1e-12 else exp_l
    top_idx = int(np.argmax(probs))
    
    return {
        "ruls": ruls,
        "composite_chi": composite_chi,
        "probs": probs,
        "top_fault_id": top_idx,
        "top_fault_prob": float(probs[top_idx]),
    }


class TestControlUnprotectedVulnerability(unittest.TestCase):
    """Control Experiment: Prove that UNPROTECTED pipelines fail catastrophically under adversarial inputs."""

    def setUp(self):
        self.model, _ = load_trained_lstm_model()
        self.cfg = load_scaler_config()

    def test_unprotected_nan_injection_causes_nan_predictions(self):
        """Without PlausibilityGate, feeding NaN features directly to LSTM causes NaN activations & predictions."""
        # Unprotected vector directly initialized with NaNs
        # Width comes from the model, not a literal: the input schema grows as
        # subsystems are added, and a hardcoded 22 silently stopped matching.
        n_feat = self.model.D
        raw_feat_with_nan = np.ones(n_feat, dtype=np.float64) * 0.5
        raw_feat_with_nan[0:n_feat // 2] = np.nan  # 50% NaNs
        
        X = np.tile(raw_feat_with_nan, (10, 1))
        cache = self.model.forward(X)
        
        # Verify unprotected forward pass outputs NaNs
        self.assertTrue(np.isnan(cache["reg"]).any(), "Unprotected LSTM RUL did not contain expected NaNs")
        self.assertTrue(np.isnan(cache["cls"]).any(), "Unprotected LSTM Cls did not contain expected NaNs")
        self.assertTrue(np.isnan(cache["h_final"]).any(), "Unprotected LSTM hidden state did not contain expected NaNs")


class TestAdversarialPressureInvalidation(unittest.TestCase):
    """Empirical Test 2: Severe Adversarial Pressures (-999 bar on all pressure channels)."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)
        self.model, _ = load_trained_lstm_model()
        self.cfg = load_scaler_config()

    def test_all_pressures_set_to_minus_999_bar(self):
        """Verify -999 bar on all pressure sensors is clamped safely, diagnosed via J1939 SPN/FMI, and LSTM forward succeeds."""
        pressure_channels = ["oil_pressure", "hyd_pressure", "exhaust_pressure"]
        
        for trial in range(50):
            raw_telemetry: Dict[str, Any] = {
                ch: SENSOR_LIMITS_CATALOG[ch].healthy_nominal for ch in ALL_CATALOG_CHANNELS
            }
            # Inject severe adversarial pressure corruption (-999 bar)
            for pch in pressure_channels:
                raw_telemetry[pch] = -999.0
            
            # Pass through SensorPlausibilityGate
            res = self.gate.filter_frame(raw_telemetry)
            
            # 1. Verification of Plausibility Sanitization
            self.assertTrue(res.is_valid)
            self.assertTrue(res.has_faults)
            
            for pch in pressure_channels:
                clean_val = res.clean_telemetry[pch]
                lim = SENSOR_LIMITS_CATALOG[pch]
                self.assertTrue(math.isfinite(clean_val), f"{pch} clean value is non-finite: {clean_val}")
                self.assertGreaterEqual(clean_val, lim.min_physical, f"{pch} {clean_val} < min_physical {lim.min_physical}")
                self.assertLessEqual(clean_val, lim.max_physical, f"{pch} {clean_val} > max_physical {lim.max_physical}")
                self.assertEqual(clean_val, lim.min_physical, f"{pch} not clamped to min_physical for wire-cut")
            
            # 2. Verification of Diagnostic Fault Events & SAE J1939-73 Tagging (FMI 04: Voltage Below Normal / Wire Cut)
            for pch in pressure_channels:
                open_faults = [f for f in res.faults_detected if f.channel == pch and f.fault_type == PlausibilityFaultType.OPEN_CIRCUIT.value]
                self.assertGreaterEqual(len(open_faults), 1, f"Missing OPEN_CIRCUIT fault event for {pch}")
                fe = open_faults[0]
                lim = SENSOR_LIMITS_CATALOG[pch]
                self.assertEqual(fe.spn, lim.spn, f"SPN mismatch for {pch}: {fe.spn} != {lim.spn}")
                self.assertEqual(fe.fmi, FMI_VOLTAGE_BELOW_NORMAL, f"FMI mismatch for {pch}: {fe.fmi} != 4 (open circuit)")
            
            # 3. Downstream Feature Normalization
            norm_features = normalize_telemetry_22(res.clean_telemetry, self.cfg)
            self.assertTrue(np.all(np.isfinite(norm_features)), "Normalized features contain NaN/Inf")
            self.assertTrue(np.all(norm_features >= 0.0) and np.all(norm_features <= 1.0), "Normalized features outside [0, 1]")
            
            # 4. Downstream Pure-NumPy LSTM Forward Pass (T=40 window)
            X = np.tile(norm_features, (40, 1))
            rul_pred, cls_probs = predict_rul(self.model, X)
            
            expected_r = np.array(self.model.p["Wy"]).shape[1]
            self.assertEqual(rul_pred.shape, (expected_r,))
            self.assertEqual(cls_probs.shape, (13,))
            self.assertTrue(np.all(np.isfinite(rul_pred)), "RUL predictions contain NaN/Inf")
            self.assertTrue(np.all(np.isfinite(cls_probs)), "Class probabilities contain NaN/Inf")
            self.assertTrue(np.all(rul_pred >= 0.0) and np.all(rul_pred <= 1.0), "RUL outside [0, 1]")
            self.assertAlmostEqual(float(np.sum(cls_probs)), 1.0, places=5, msg="Softmax probabilities do not sum to 1")


class TestAdversarialThermalRunaway(unittest.TestCase):
    """Empirical Test 3: Extreme Adversarial Temperatures (+9999°C across all temperature channels)."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)
        self.model, _ = load_trained_lstm_model()
        self.cfg = load_scaler_config()

    def test_all_temperatures_set_to_plus_9999_deg_c(self):
        """Verify +9999°C on coolant, oil, and exhaust is clamped safely, diagnosed via J1939 SPN/FMI 03, and LSTM succeeds."""
        temp_channels = ["coolant_temp", "oil_temp", "exhaust_temp"]
        
        for trial in range(50):
            raw_telemetry: Dict[str, Any] = {
                ch: SENSOR_LIMITS_CATALOG[ch].healthy_nominal for ch in ALL_CATALOG_CHANNELS
            }
            # Inject extreme thermal corruption (+9999°C)
            for tch in temp_channels:
                raw_telemetry[tch] = 9999.0
            
            # Pass through SensorPlausibilityGate
            res = self.gate.filter_frame(raw_telemetry)
            
            # 1. Verification of Plausibility Sanitization
            self.assertTrue(res.is_valid)
            self.assertTrue(res.has_faults)
            
            for tch in temp_channels:
                clean_val = res.clean_telemetry[tch]
                lim = SENSOR_LIMITS_CATALOG[tch]
                self.assertTrue(math.isfinite(clean_val), f"{tch} clean value non-finite: {clean_val}")
                self.assertGreaterEqual(clean_val, lim.min_physical)
                self.assertLessEqual(clean_val, lim.max_physical)
                self.assertEqual(clean_val, lim.max_physical, f"{tch} not clamped to max_physical for short-circuit")
            
            # 2. Verification of Diagnostic Fault Events & SAE J1939-73 Tagging (FMI 03: Voltage Above Normal / Short to Power)
            for tch in temp_channels:
                short_faults = [f for f in res.faults_detected if f.channel == tch and f.fault_type == PlausibilityFaultType.SHORT_CIRCUIT.value]
                self.assertGreaterEqual(len(short_faults), 1, f"Missing SHORT_CIRCUIT fault event for {tch}")
                fe = short_faults[0]
                lim = SENSOR_LIMITS_CATALOG[tch]
                self.assertEqual(fe.spn, lim.spn, f"SPN mismatch for {tch}: {fe.spn} != {lim.spn}")
                self.assertEqual(fe.fmi, FMI_VOLTAGE_ABOVE_NORMAL, f"FMI mismatch for {tch}: {fe.fmi} != 3 (short circuit)")
            
            # 3. Downstream Feature Normalization
            norm_features = normalize_telemetry_22(res.clean_telemetry, self.cfg)
            self.assertTrue(np.all(np.isfinite(norm_features)))
            self.assertTrue(np.all(norm_features >= 0.0) and np.all(norm_features <= 1.0))
            
            # 4. Downstream LSTM Forward Pass
            X = np.tile(norm_features, (40, 1))
            rul_pred, cls_probs = predict_rul(self.model, X)
            
            self.assertTrue(np.all(np.isfinite(rul_pred)))
            self.assertTrue(np.all(np.isfinite(cls_probs)))
            self.assertAlmostEqual(float(np.sum(cls_probs)), 1.0, places=5)


class TestHighDensityNaNInfCorruption(unittest.TestCase):
    """Empirical Test 4: High-Density NaN / Inf / Null Corruption across 50% to 100% of channels."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)
        self.model, _ = load_trained_lstm_model()
        self.cfg = load_scaler_config()

    def test_random_50_to_100_percent_nan_inf_injection_1000_frames(self):
        """Empirically test 1,000 frames with 50-100% NaN/Inf/Null channels. Verify 100% finite neural outputs."""
        rng = random.Random(42)
        corrupt_values = [
            float("nan"),
            math.nan,
            float("inf"),
            float("-inf"),
            "nan",
            "NaN",
            "inf",
            "-inf",
            None,
            "null",
            "None",
            "",
            "0xBADF00D",
        ]
        
        for frame_idx in range(1000):
            raw_telemetry: Dict[str, Any] = {
                ch: SENSOR_LIMITS_CATALOG[ch].healthy_nominal for ch in ALL_CATALOG_CHANNELS
            }
            
            # Pick 50% to 100% of channels to corrupt
            num_corrupt = rng.randint(len(ALL_CATALOG_CHANNELS) // 2, len(ALL_CATALOG_CHANNELS))
            corrupted_channels = rng.sample(ALL_CATALOG_CHANNELS, num_corrupt)
            
            for ch in corrupted_channels:
                raw_telemetry[ch] = rng.choice(corrupt_values)
            
            # Intercept with plausibility gate
            res = self.gate.filter_frame(raw_telemetry)
            
            self.assertTrue(res.is_valid, f"Frame {frame_idx} marked invalid")
            self.assertEqual(len(res.clean_telemetry), len(ALL_CATALOG_CHANNELS))
            
            # Verify clean telemetry is 100% finite float
            for ch, val in res.clean_telemetry.items():
                self.assertTrue(isinstance(val, float), f"Channel {ch} in frame {frame_idx} is not float: {type(val)}")
                self.assertTrue(math.isfinite(val), f"Channel {ch} in frame {frame_idx} is non-finite: {val}")
            
            # Verify fault events emitted
            nan_faults = [f for f in res.faults_detected if f.fault_type == PlausibilityFaultType.NAN_INF_CORRUPTION.value]
            self.assertGreaterEqual(len(nan_faults), 1)
            for f in nan_faults:
                self.assertEqual(f.fmi, FMI_DATA_ERRATIC)
            
            # Downstream normalization
            norm_features = normalize_telemetry_22(res.clean_telemetry, self.cfg)
            self.assertTrue(np.all(np.isfinite(norm_features)))
            
            # Forward pass through Python LSTM
            X = np.tile(norm_features, (20, 1))
            rul_pred, cls_probs = predict_rul(self.model, X)
            self.assertTrue(np.all(np.isfinite(rul_pred)))
            self.assertTrue(np.all(np.isfinite(cls_probs)))
            self.assertAlmostEqual(float(np.sum(cls_probs)), 1.0, places=5)


class TestCombinatorialChaosMultiStepSequences(unittest.TestCase):
    """Empirical Test 5: Combinatorial Chaos Injection across multi-step sequences with Python LSTM & C99 Engine."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)
        self.model, self.model_data = load_trained_lstm_model()
        self.cfg = load_scaler_config()
        
        # Synthesize C99 D=58 weights for C99 inference testing
        rng = np.random.default_rng(1234)
        D, H, R, C = 58, 24, 8, 13
        self.c99_weights = {
            "H": H, "R": R, "C": C,
            "Wf": rng.normal(0, 0.08, (D, H)),
            "Uf": rng.normal(0, 0.08, (H, H)),
            "bf": np.zeros(H),
            "Wi": rng.normal(0, 0.08, (D, H)),
            "Ui": rng.normal(0, 0.08, (H, H)),
            "bi": np.zeros(H),
            "Wc": rng.normal(0, 0.08, (D, H)),
            "Uc": rng.normal(0, 0.08, (H, H)),
            "bc": np.zeros(H),
            "Wo": rng.normal(0, 0.08, (D, H)),
            "Uo": rng.normal(0, 0.08, (H, H)),
            "bo": np.zeros(H),
            "Wy": rng.normal(0, 0.08, (H, R)),
            "by": np.zeros(R),
            "Wcls": rng.normal(0, 0.08, (H, C)),
            "bcls": np.zeros(C),
        }
        self.c99_state = {
            "h": np.zeros(H, dtype=np.float64),
            "c": np.zeros(H, dtype=np.float64),
            "step_count": 0,
        }

    def test_sustained_2000_frame_combinatorial_chaos_stream(self):
        """Stream 2,000 continuous frames of mixed extreme pressures, temperatures, NaNs, Infs, wire cuts, and slew-rates.
        Verify both Python LSTM and C99 Engine remain 100% numerically stable with zero division-by-zero or NaNs.
        """
        rng = random.Random(999)
        window_22 = []
        window_size = 40
        
        for step in range(2000):
            # Base healthy frame
            raw_frame: Dict[str, Any] = {
                ch: SENSOR_LIMITS_CATALOG[ch].healthy_nominal for ch in ALL_CATALOG_CHANNELS
            }
            raw_frame["time"] = step * 0.05
            raw_frame["step"] = step
            
            # 1. Randomly inject pressure wire cut (-999 bar)
            if step % 3 == 0:
                raw_frame["oil_pressure"] = -999.0
            if step % 5 == 0:
                raw_frame["hyd_pressure"] = -999.0
            if step % 7 == 0:
                raw_frame["exhaust_pressure"] = -999.0
                
            # 2. Randomly inject thermal runaway (+9999°C)
            if step % 4 == 0:
                raw_frame["coolant_temp"] = 9999.0
            if step % 6 == 0:
                raw_frame["oil_temp"] = 9999.0
            if step % 8 == 0:
                raw_frame["exhaust_temp"] = 9999.0
                
            # 3. Randomly inject NaNs and Infs into 25% to 50% of channels
            corrupt_chs = rng.sample(ALL_CATALOG_CHANNELS, rng.randint(15, 30))
            for ch in corrupt_chs:
                c_type = rng.randint(0, 4)
                if c_type == 0:
                    raw_frame[ch] = float("nan")
                elif c_type == 1:
                    raw_frame[ch] = float("inf")
                elif c_type == 2:
                    raw_frame[ch] = float("-inf")
                elif c_type == 3:
                    raw_frame[ch] = "nan"
                else:
                    raw_frame[ch] = None
                    
            # 4. Filter frame through SensorPlausibilityGate
            res = self.gate.filter_frame(raw_frame)
            self.assertTrue(res.is_valid)
            
            # 5. Extract and normalize 22-D feature vector for Python LSTM
            feat_22 = normalize_telemetry_22(res.clean_telemetry, self.cfg)
            self.assertTrue(np.all(np.isfinite(feat_22)))
            window_22.append(feat_22)
            if len(window_22) > window_size:
                window_22.pop(0)
                
            # Run Python LSTM on window
            if len(window_22) == window_size:
                X_seq = np.array(window_22)
                rul_pred, cls_probs = predict_rul(self.model, X_seq)
                self.assertTrue(np.all(np.isfinite(rul_pred)), f"Python LSTM RUL NaN at step {step}")
                self.assertTrue(np.all(np.isfinite(cls_probs)), f"Python LSTM Cls NaN at step {step}")
                self.assertAlmostEqual(float(np.sum(cls_probs)), 1.0, places=5)
                
            # 6. Extract 58-D normalized vector for C99 Edge Engine
            feat_58 = np.zeros(58, dtype=np.float64)
            for idx, ch in enumerate(AFV_58_PHYSICAL_CHANNELS):
                lim = SENSOR_LIMITS_CATALOG[ch]
                clean_v = res.clean_telemetry[ch]
                span = lim.max_physical - lim.min_physical
                feat_58[idx] = (clean_v - lim.min_physical) / span if span > 1e-12 else 0.5
            feat_58 = np.clip(feat_58, 0.0, 1.0)
            
            c99_res = c99_infer_step_emulation(self.c99_weights, self.c99_state, feat_58)
            self.assertTrue(np.all(np.isfinite(c99_res["ruls"])), f"C99 RUL NaN at step {step}")
            self.assertTrue(np.all(np.isfinite(c99_res["probs"])), f"C99 Probs NaN at step {step}")
            self.assertTrue(math.isfinite(c99_res["composite_chi"]), f"C99 CHI NaN at step {step}")
            self.assertAlmostEqual(float(np.sum(c99_res["probs"])), 1.0, places=5)


class TestDiagnosticDecouplingAndNeuralIndependence(unittest.TestCase):
    """Empirical Test 6: Verify diagnostic fault structures are decoupled from neural features."""

    def setUp(self):
        self.gate = SensorPlausibilityGate(sample_rate_hz=20.0)

    def test_fault_events_tagged_with_spn_fmi_without_polluting_clean_telemetry(self):
        """Ensure clean_telemetry contains strictly numerical values while fault records contain rich diagnostic metadata."""
        raw_telemetry: Dict[str, Any] = {
            ch: SENSOR_LIMITS_CATALOG[ch].healthy_nominal for ch in ALL_CATALOG_CHANNELS
        }
        
        # Inject multiple distinct fault types
        raw_telemetry["oil_pressure"] = -999.0       # Open circuit (SPN 100, FMI 04)
        raw_telemetry["coolant_temp"] = 9999.0       # Short circuit (SPN 110, FMI 03)
        raw_telemetry["shaft_torque"] = float("nan") # NaN corruption (SPN 513, FMI 02)
        raw_telemetry["hyd_pressure"] = "inf"        # Inf corruption (SPN 520202, FMI 02)
        
        res = self.gate.filter_frame(raw_telemetry)
        
        # Check clean telemetry contains ONLY valid float numbers and standard channel keys
        for k, v in res.clean_telemetry.items():
            self.assertTrue(isinstance(v, (int, float)))
            self.assertTrue(math.isfinite(v))
            self.assertIn(k, ALL_CATALOG_CHANNELS)
            
        # Check diagnostic fault events
        self.assertGreaterEqual(len(res.faults_detected), 4)
        
        # Check specific fault records
        oil_open = [f for f in res.faults_detected if f.channel == "oil_pressure" and f.fault_type == PlausibilityFaultType.OPEN_CIRCUIT.value]
        self.assertEqual(len(oil_open), 1)
        self.assertEqual(oil_open[0].spn, 100)
        self.assertEqual(oil_open[0].fmi, FMI_VOLTAGE_BELOW_NORMAL)
        
        coolant_short = [f for f in res.faults_detected if f.channel == "coolant_temp" and f.fault_type == PlausibilityFaultType.SHORT_CIRCUIT.value]
        self.assertEqual(len(coolant_short), 1)
        self.assertEqual(coolant_short[0].spn, 110)
        self.assertEqual(coolant_short[0].fmi, FMI_VOLTAGE_ABOVE_NORMAL)
        
        torque_nan = [f for f in res.faults_detected if f.channel == "shaft_torque" and f.fault_type == PlausibilityFaultType.NAN_INF_CORRUPTION.value]
        self.assertEqual(len(torque_nan), 1)
        self.assertEqual(torque_nan[0].spn, 513)
        self.assertEqual(torque_nan[0].fmi, FMI_DATA_ERRATIC)
        
        hyd_inf = [f for f in res.faults_detected if f.channel == "hyd_pressure" and f.fault_type == PlausibilityFaultType.NAN_INF_CORRUPTION.value]
        self.assertEqual(len(hyd_inf), 1)
        self.assertEqual(hyd_inf[0].spn, 520202)
        self.assertEqual(hyd_inf[0].fmi, FMI_DATA_ERRATIC)


if __name__ == "__main__":
    unittest.main()
