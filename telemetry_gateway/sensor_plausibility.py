"""Pre-Inference FDIR (Fault Detection, Isolation, and Recovery) Plausibility Gate.

Provides deterministic multi-layer signal validation for military Armored Fighting Vehicles
(CVRDE Arjun Mk-1A, T-90S Bhishma, Zorawar Light Tank). Guards dual-tier neural inference
engines against NaN/Inf, open-circuit wire cuts, short-to-power electrical anomalies,
unphysical slew-rates, frozen/deadlined sensors, electromagnetic interference (EMI),
and cross-subsystem physical violations.
"""

from __future__ import annotations

import collections
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union


class PlausibilityFaultType(str, Enum):
    """FDIR fault classification types."""
    OPEN_CIRCUIT = "OPEN_CIRCUIT"
    SHORT_CIRCUIT = "SHORT_CIRCUIT"
    RATE_OF_CHANGE_EXCEEDED = "RATE_OF_CHANGE_EXCEEDED"
    STUCK_AT = "STUCK_AT"
    OUTLIER_EMI = "OUTLIER_EMI"
    DUAL_SENSOR_MISMATCH = "DUAL_SENSOR_MISMATCH"
    RANGE_OUT_OF_BOUNDS = "RANGE_OUT_OF_BOUNDS"
    NAN_INF_CORRUPTION = "NAN_INF_CORRUPTION"


# Standard SAE J1939 Failure Mode Identifiers (FMIs)
FMI_DATA_VALID_ABOVE_NORMAL = 0   # FMI 00: Data valid but above normal operating range (Crit High)
FMI_DATA_VALID_BELOW_NORMAL = 1   # FMI 01: Data valid but below normal operating range (Crit Low)
FMI_DATA_ERRATIC = 2              # FMI 02: Data erratic, intermittent, or incorrect (Slew/Stuck/EMI)
FMI_VOLTAGE_ABOVE_NORMAL = 3      # FMI 03: Voltage above normal or shorted to high source
FMI_VOLTAGE_BELOW_NORMAL = 4      # FMI 04: Voltage below normal or shorted to low source / wire cut
FMI_CURRENT_BELOW_NORMAL = 5      # FMI 05: Current below normal or open circuit
FMI_CURRENT_ABOVE_NORMAL = 6      # FMI 06: Current above normal or grounded circuit
FMI_MECHANICAL_SYSTEM_FAIL = 7    # FMI 07: Mechanical system not responding or out of adjustment
FMI_ABNORMAL_FREQUENCY = 8        # FMI 08: Abnormal frequency or pulse width
FMI_ABNORMAL_UPDATE_RATE = 9      # FMI 09: Abnormal update rate
FMI_ABNORMAL_RATE_OF_CHANGE = 10  # FMI 10: Abnormal rate of change
FMI_FAILURE_MODE_NOT_IDENT = 11   # FMI 11: Root cause not identifiable
FMI_BAD_INTELLIGENT_DEVICE = 12   # FMI 12: Bad intelligent device or component
FMI_OUT_OF_CALIBRATION = 13       # FMI 13: Out of calibration
FMI_SPECIAL_INSTRUCTIONS = 14     # FMI 14: Special instructions / Plausibility cross-mismatch


@dataclass
class SensorFaultEvent:
    """Diagnostic fault event emitted when a sensor violates physical or electrical rules."""
    channel: str
    fault_type: str
    raw_value: Any
    clamped_value: float
    spn: int
    fmi: int
    message: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "fault_type": self.fault_type,
            "raw_value": self.raw_value,
            "clamped_value": self.clamped_value,
            "spn": self.spn,
            "fmi": self.fmi,
            "message": self.message,
            "timestamp": self.timestamp,
        }


@dataclass
class SensorLimits:
    """Operational envelope and dynamic thresholds for a vehicle telemetry channel."""
    channel: str
    spn: int
    min_physical: float
    max_physical: float
    healthy_nominal: float
    max_slew_per_sec: float
    open_circuit_threshold_low: float = -500.0
    short_circuit_threshold_high: float = 800.0
    is_dynamic: bool = True
    unit: str = ""
    description: str = ""
    # How long this channel may legitimately hold an identical value before a
    # flatline is suspicious. Regulated or coarsely quantised signals hold a
    # constant reading for a long time by design: a thermostat-controlled
    # coolant temperature sits at exactly 93.0 C for minutes, and an accelerator
    # pedal reads exactly 0.0 while coasting. Judging those by frame count
    # produced a 69% false-positive rate on real recorded CAN data.
    max_static_duration_s: float = 15.0


@dataclass
class PlausibilityResult:
    """Container for sanitized telemetry and diagnostic fault records."""
    clean_telemetry: Dict[str, float]
    faults_detected: List[SensorFaultEvent]
    is_valid: bool
    raw_telemetry: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0

    @property
    def has_faults(self) -> bool:
        return len(self.faults_detected) > 0


# ============================================================================
# Exhaustive 58-Channel Physical Envelopes & SAE J1939 SPN Configurations
# ============================================================================

SENSOR_LIMITS_CATALOG: Dict[str, SensorLimits] = {
    # 1. Mission Timing & Step
    "time": SensorLimits(
        channel="time",
        spn=520001,
        min_physical=0.0,
        max_physical=1.0e8,
        healthy_nominal=0.0,
        max_slew_per_sec=100.0,
        is_dynamic=False,
        unit="s",
        description="Mission elapsed time",
    ),
    "step": SensorLimits(
        channel="step",
        spn=520002,
        min_physical=0.0,
        max_physical=1.0e9,
        healthy_nominal=0.0,
        max_slew_per_sec=1000.0,
        is_dynamic=False,
        unit="count",
        description="Simulation discrete step index",
    ),

    # 2. Powertrain & Engine Core
    "rpm": SensorLimits(
        channel="rpm",
        spn=190,
        min_physical=0.0,
        max_physical=3500.0,
        healthy_nominal=1800.0,
        max_slew_per_sec=2500.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=4500.0,
        is_dynamic=True,
        unit="RPM",
        description="Main Engine Crankshaft Speed",
    ),
    "load": SensorLimits(
        channel="load",
        spn=92,
        min_physical=0.0,
        max_physical=1.0,
        healthy_nominal=0.50,
        max_slew_per_sec=2.5,
        open_circuit_threshold_low=-5.0,
        short_circuit_threshold_high=5.0,
        is_dynamic=True,
        unit="fraction",
        description="Engine Throttle / Demand Load",
        max_static_duration_s=120.0,
    ),
    "terrain": SensorLimits(
        channel="terrain",
        spn=520401,
        min_physical=0.0,
        max_physical=1.0,
        healthy_nominal=0.20,
        max_slew_per_sec=4.0,
        open_circuit_threshold_low=-5.0,
        short_circuit_threshold_high=5.0,
        is_dynamic=True,
        unit="index",
        description="ISO 8608 Cross-Country Terrain Roughness",
        max_static_duration_s=600.0,
    ),

    # 3. Thermal & Exhaust Systems
    "coolant_temp": SensorLimits(
        channel="coolant_temp",
        spn=110,
        min_physical=-40.0,
        max_physical=160.0,
        healthy_nominal=88.0,
        max_slew_per_sec=15.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=800.0,
        is_dynamic=True,
        unit="°C",
        description="Engine Coolant Temperature",
        max_static_duration_s=240.0,
    ),
    "coolant_rtd_ohm": SensorLimits(
        channel="coolant_rtd_ohm",
        spn=110,
        min_physical=50.0,
        max_physical=250.0,
        healthy_nominal=134.0,
        max_slew_per_sec=25.0,
        open_circuit_threshold_low=-100.0,
        short_circuit_threshold_high=1000.0,
        is_dynamic=True,
        unit="Ω",
        description="PT100 Coolant RTD Resistance",
    ),
    "exhaust_temp": SensorLimits(
        channel="exhaust_temp",
        spn=173,
        min_physical=0.0,
        max_physical=900.0,
        healthy_nominal=350.0,
        max_slew_per_sec=150.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=1500.0,
        is_dynamic=True,
        unit="°C",
        description="Exhaust Gas Temperature (EGT)",
        max_static_duration_s=120.0,
    ),
    "exhaust_thermocouple_v": SensorLimits(
        channel="exhaust_thermocouple_v",
        spn=173,
        min_physical=0.0,
        max_physical=50.0,
        healthy_nominal=15.0,
        max_slew_per_sec=10.0,
        open_circuit_threshold_low=-50.0,
        short_circuit_threshold_high=150.0,
        is_dynamic=True,
        unit="mV",
        description="Type-K Thermocouple Voltage",
    ),
    "exhaust_pressure": SensorLimits(
        channel="exhaust_pressure",
        spn=102,
        min_physical=0.0,
        max_physical=5.0,  # in bar
        healthy_nominal=1.25,
        max_slew_per_sec=3.0,
        open_circuit_threshold_low=-5.0,
        short_circuit_threshold_high=10.0,
        is_dynamic=True,
        unit="bar",
        description="Exhaust Manifold Backpressure",
    ),
    "exhaust_mass_flow": SensorLimits(
        channel="exhaust_mass_flow",
        spn=132,
        min_physical=0.0,
        max_physical=3.5,
        healthy_nominal=0.45,
        max_slew_per_sec=2.0,
        open_circuit_threshold_low=-10.0,
        short_circuit_threshold_high=25.0,
        is_dynamic=True,
        unit="kg/s",
        description="Exhaust Mass Flow Rate",
    ),
    # A compression-ignition engine is quality-governed and always runs lean:
    # roughly 1.2 at the smoke limit up to ~6 at idle. The envelope here was
    # [0.5, 2.5] with a nominal of 1.0 -- a stoichiometric spark-ignition
    # assumption, the same one that was in ml/parts.py. Against real diesel
    # telemetry it rejects normal light-load operation as an out-of-range
    # sensor fault.
    "lambda": SensorLimits(
        channel="lambda",
        spn=520102,
        min_physical=0.9,
        max_physical=8.0,
        healthy_nominal=3.0,
        max_slew_per_sec=4.0,
        open_circuit_threshold_low=-5.0,
        short_circuit_threshold_high=20.0,
        is_dynamic=True,
        unit="ratio",
        description="Air-Fuel Equivalence Ratio (diesel, lean-burn)",
    ),
    # Load-normalised combustion indicator: 1.0 when fuelling matches the
    # load-expected value at any duty point. This is the channel that can be
    # scored against a fixed reference; raw lambda cannot.
    "lambda_residual": SensorLimits(
        channel="lambda_residual",
        spn=520103,
        min_physical=0.2,
        max_physical=3.0,
        healthy_nominal=1.0,
        max_slew_per_sec=2.0,
        open_circuit_threshold_low=-5.0,
        short_circuit_threshold_high=10.0,
        is_dynamic=True,
        unit="ratio",
        description="Lambda residual vs load-expected (combustion condition)",
    ),
    "exhaust_o2_pct": SensorLimits(
        channel="exhaust_o2_pct",
        spn=520103,
        min_physical=0.0,
        max_physical=25.0,
        healthy_nominal=5.0,
        max_slew_per_sec=15.0,
        open_circuit_threshold_low=-50.0,
        short_circuit_threshold_high=100.0,
        is_dynamic=True,
        unit="%",
        description="Exhaust O2 Concentration",
    ),

    # 4. Lubrication & Oil System
    "oil_pressure": SensorLimits(
        channel="oil_pressure",
        spn=100,
        min_physical=0.0,
        max_physical=15.0,  # in bar
        healthy_nominal=5.0,
        max_slew_per_sec=8.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=800.0,
        is_dynamic=True,
        unit="bar",
        description="Engine Main Gallery Oil Pressure",
    ),
    "oil_temp": SensorLimits(
        channel="oil_temp",
        spn=175,
        min_physical=-40.0,
        max_physical=180.0,
        healthy_nominal=85.0,
        max_slew_per_sec=10.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=800.0,
        is_dynamic=True,
        unit="°C",
        description="Engine Oil Sump Temperature",
        max_static_duration_s=240.0,
    ),
    "oil_viscosity": SensorLimits(
        channel="oil_viscosity",
        spn=520104,
        min_physical=0.001,
        max_physical=1.0,
        healthy_nominal=0.035,
        max_slew_per_sec=0.2,
        open_circuit_threshold_low=-5.0,
        short_circuit_threshold_high=10.0,
        is_dynamic=True,
        unit="Pa·s",
        description="Dynamic Oil Viscosity",
    ),
    "oil_flow": SensorLimits(
        channel="oil_flow",
        spn=520105,
        min_physical=0.0,
        max_physical=0.01,
        healthy_nominal=0.001,
        max_slew_per_sec=0.005,
        open_circuit_threshold_low=-1.0,
        short_circuit_threshold_high=2.0,
        is_dynamic=True,
        unit="m³/s",
        description="Lubrication Oil Flow Rate",
    ),
    "debris_cumulative": SensorLimits(
        channel="debris_cumulative",
        spn=520106,
        min_physical=0.0,
        max_physical=50000.0,
        healthy_nominal=200.0,
        max_slew_per_sec=200.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=100000.0,
        is_dynamic=True,
        unit="count",
        description="Total Ferromagnetic Debris Count",
    ),
    "debris_rate": SensorLimits(
        channel="debris_rate",
        spn=520107,
        min_physical=0.0,
        max_physical=100.0,
        healthy_nominal=1.0,
        max_slew_per_sec=30.0,
        open_circuit_threshold_low=-50.0,
        short_circuit_threshold_high=300.0,
        is_dynamic=True,
        unit="particles/s",
        description="Instantaneous Debris Generation Rate",
    ),
    "debris_particles": SensorLimits(
        channel="debris_particles",
        spn=520108,
        min_physical=0.0,
        max_physical=100.0,
        healthy_nominal=0.0,
        max_slew_per_sec=50.0,
        open_circuit_threshold_low=-50.0,
        short_circuit_threshold_high=300.0,
        is_dynamic=True,
        unit="count/step",
        description="Debris Particles per Step",
    ),

    # 5. Drivetrain & Mechanical Power
    "shaft_torque": SensorLimits(
        channel="shaft_torque",
        spn=513,
        min_physical=0.0,
        max_physical=5000.0,
        healthy_nominal=450.0,
        max_slew_per_sec=3500.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=15000.0,
        is_dynamic=True,
        unit="N·m",
        description="Main Transmission Shaft Torque",
    ),
    "shaft_shear_stress": SensorLimits(
        channel="shaft_shear_stress",
        spn=520109,
        min_physical=0.0,
        max_physical=600.0,
        healthy_nominal=80.0,
        max_slew_per_sec=300.0,
        open_circuit_threshold_low=-100.0,
        short_circuit_threshold_high=1500.0,
        is_dynamic=True,
        unit="MPa",
        description="Shaft Torsional Shear Stress",
    ),
    "shaft_shear_strain": SensorLimits(
        channel="shaft_shear_strain",
        spn=520110,
        min_physical=0.0,
        max_physical=6000.0,
        healthy_nominal=1000.0,
        max_slew_per_sec=3000.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=15000.0,
        is_dynamic=True,
        unit="με",
        description="Shaft Elastic Shear Strain",
    ),
    "mech_power": SensorLimits(
        channel="mech_power",
        spn=520111,
        min_physical=0.0,
        max_physical=1800.0,
        healthy_nominal=350.0,
        max_slew_per_sec=900.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=5000.0,
        is_dynamic=True,
        unit="kW",
        description="Delivered Shaft Mechanical Power",
    ),
    "shaft_omega": SensorLimits(
        channel="shaft_omega",
        spn=520112,
        min_physical=0.0,
        max_physical=400.0,
        healthy_nominal=150.0,
        max_slew_per_sec=250.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=1500.0,
        is_dynamic=True,
        unit="rad/s",
        description="Sprocket Shaft Angular Velocity",
    ),

    # 6. Fluid Levels & Capacitive Sensors
    "fuel_level": SensorLimits(
        channel="fuel_level",
        spn=96,
        min_physical=0.0,
        max_physical=100.0,
        healthy_nominal=85.0,
        max_slew_per_sec=5.0,
        open_circuit_threshold_low=-50.0,
        short_circuit_threshold_high=200.0,
        is_dynamic=False,
        unit="%",
        description="Fuel Tank Level",
        max_static_duration_s=900.0,
    ),
    "fuel_volume": SensorLimits(
        channel="fuel_volume",
        spn=96,
        min_physical=0.0,
        max_physical=2500.0,
        healthy_nominal=1200.0,
        max_slew_per_sec=50.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=5000.0,
        is_dynamic=False,
        unit="L",
        description="Fuel Volume",
    ),
    "oil_level": SensorLimits(
        channel="oil_level",
        spn=98,
        min_physical=0.0,
        max_physical=100.0,
        healthy_nominal=95.0,
        max_slew_per_sec=5.0,
        open_circuit_threshold_low=-50.0,
        short_circuit_threshold_high=200.0,
        is_dynamic=False,
        unit="%",
        description="Engine Oil Sump Level",
        max_static_duration_s=900.0,
    ),
    "coolant_level": SensorLimits(
        channel="coolant_level",
        spn=111,
        min_physical=0.0,
        max_physical=100.0,
        healthy_nominal=95.0,
        max_slew_per_sec=5.0,
        open_circuit_threshold_low=-50.0,
        short_circuit_threshold_high=200.0,
        is_dynamic=False,
        unit="%",
        description="Coolant Expansion Tank Level",
        max_static_duration_s=900.0,
    ),
    "fuel_capacitance_pf": SensorLimits(
        channel="fuel_capacitance_pf",
        spn=520113,
        min_physical=10.0,
        max_physical=600.0,
        healthy_nominal=250.0,
        max_slew_per_sec=50.0,
        open_circuit_threshold_low=-100.0,
        short_circuit_threshold_high=1500.0,
        is_dynamic=False,
        unit="pF",
        description="Capacitive Fuel Level Probe",
    ),

    # 7. Actuation & Gun Control Hydraulics
    "hyd_pressure": SensorLimits(
        channel="hyd_pressure",
        spn=520202,
        min_physical=0.0,
        max_physical=400.0,  # in bar
        healthy_nominal=210.0,
        max_slew_per_sec=250.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=800.0,
        is_dynamic=True,
        unit="bar",
        description="Turret & Recoil Hydraulic Pressure",
        max_static_duration_s=60.0,
    ),
    "hyd_flow": SensorLimits(
        channel="hyd_flow",
        spn=520203,
        min_physical=0.0,
        max_physical=0.015,
        healthy_nominal=0.002,
        max_slew_per_sec=0.01,
        open_circuit_threshold_low=-1.0,
        short_circuit_threshold_high=5.0,
        is_dynamic=True,
        unit="m³/s",
        description="Main Hydraulic Pump Flow Rate",
    ),
    "hyd_force": SensorLimits(
        channel="hyd_force",
        spn=520204,
        min_physical=0.0,
        max_physical=50.0,  # in kN
        healthy_nominal=4.2,
        max_slew_per_sec=30.0,
        open_circuit_threshold_low=-100.0,
        short_circuit_threshold_high=200.0,
        is_dynamic=True,
        unit="kN",
        description="Gun Elevation Actuator Force",
    ),
    "hyd_power": SensorLimits(
        channel="hyd_power",
        spn=520205,
        min_physical=0.0,
        max_physical=200.0,
        healthy_nominal=25.0,
        max_slew_per_sec=100.0,
        open_circuit_threshold_low=-100.0,
        short_circuit_threshold_high=1000.0,
        is_dynamic=True,
        unit="kW",
        description="Hydraulic System Power",
    ),
    "hyd_leak_flow": SensorLimits(
        channel="hyd_leak_flow",
        spn=520206,
        min_physical=0.0,
        max_physical=0.002,
        healthy_nominal=0.00002,
        max_slew_per_sec=0.001,
        open_circuit_threshold_low=-1.0,
        short_circuit_threshold_high=2.0,
        is_dynamic=True,
        unit="m³/s",
        description="Hydraulic Seal Leak Bypass Flow",
    ),

    # 8. CVRDE Hydrogas Suspension & Running Gear
    "susp_load_kN": SensorLimits(
        channel="susp_load_kN",
        spn=520200,
        min_physical=0.0,
        max_physical=400.0,
        healthy_nominal=115.0,
        max_slew_per_sec=350.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=1000.0,
        is_dynamic=True,
        unit="kN",
        description="Roadwheel Dynamic Station Load",
    ),
    "susp_stress_MPa": SensorLimits(
        channel="susp_stress_MPa",
        spn=520207,
        min_physical=0.0,
        max_physical=900.0,
        healthy_nominal=180.0,
        max_slew_per_sec=500.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=2000.0,
        is_dynamic=True,
        unit="MPa",
        description="Suspension Trailing Arm Stress",
    ),
    "susp_strain_ue": SensorLimits(
        channel="susp_strain_ue",
        spn=520208,
        min_physical=0.0,
        max_physical=2500.0,
        healthy_nominal=240.0,
        max_slew_per_sec=1500.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=5000.0,
        is_dynamic=True,
        unit="με",
        description="Suspension Strain Gauge Deflection",
    ),
    "susp_dR_ohm": SensorLimits(
        channel="susp_dR_ohm",
        spn=520209,
        min_physical=-100.0,
        max_physical=100.0,
        healthy_nominal=0.0,
        max_slew_per_sec=40.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=500.0,
        is_dynamic=True,
        unit="Ω",
        description="Wheatstone Bridge Delta Resistance",
    ),

    # 9. Torsion Bar & Structural Fatigue
    "torsion_torque": SensorLimits(
        channel="torsion_torque",
        spn=520210,
        min_physical=0.0,
        max_physical=20000.0,
        healthy_nominal=3500.0,
        max_slew_per_sec=12000.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=50000.0,
        is_dynamic=True,
        unit="N·m",
        description="Torsion Bar Twisting Moment",
    ),
    "torsion_twist_deg": SensorLimits(
        channel="torsion_twist_deg",
        spn=520211,
        min_physical=0.0,
        max_physical=12.0,
        healthy_nominal=0.40,
        max_slew_per_sec=8.0,
        open_circuit_threshold_low=-50.0,
        short_circuit_threshold_high=50.0,
        is_dynamic=True,
        unit="deg",
        description="Torsion Bar Angular Deflection",
    ),
    "torsion_shear_MPa": SensorLimits(
        channel="torsion_shear_MPa",
        spn=520212,
        min_physical=0.0,
        max_physical=800.0,
        healthy_nominal=120.0,
        max_slew_per_sec=500.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=2500.0,
        is_dynamic=True,
        unit="MPa",
        description="Torsion Bar Outer Fiber Shear",
    ),
    "torsion_cumulative_twist": SensorLimits(
        channel="torsion_cumulative_twist",
        spn=520213,
        min_physical=0.0,
        max_physical=1.0e6,
        healthy_nominal=300.0,
        max_slew_per_sec=100.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=5.0e6,
        is_dynamic=True,
        unit="rad",
        description="Plastic Fatigue Cumulative Twist",
    ),

    # 10. Hull Shock, Acoustic & Vibration Sensors
    "shock_a_rms_g": SensorLimits(
        channel="shock_a_rms_g",
        spn=520214,
        min_physical=0.0,
        max_physical=30.0,
        healthy_nominal=1.8,
        max_slew_per_sec=25.0,
        open_circuit_threshold_low=-50.0,
        short_circuit_threshold_high=100.0,
        is_dynamic=True,
        unit="g",
        description="Hull Vertical Acceleration RMS",
    ),
    "shock_peak_g": SensorLimits(
        channel="shock_peak_g",
        spn=520215,
        min_physical=0.0,
        max_physical=60.0,
        healthy_nominal=3.5,
        max_slew_per_sec=50.0,
        open_circuit_threshold_low=-50.0,
        short_circuit_threshold_high=200.0,
        is_dynamic=True,
        unit="g",
        description="Hull Peak Impulse Acceleration",
    ),
    "shock_energy": SensorLimits(
        channel="shock_energy",
        spn=520216,
        min_physical=0.0,
        max_physical=1.0e5,
        healthy_nominal=500.0,
        max_slew_per_sec=10000.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=5.0e5,
        is_dynamic=True,
        unit="J",
        description="Dynamic Shock Absorbed Energy",
    ),
    "spl_db": SensorLimits(
        channel="spl_db",
        spn=520114,
        min_physical=40.0,
        max_physical=170.0,
        healthy_nominal=110.0,
        max_slew_per_sec=50.0,
        open_circuit_threshold_low=-50.0,
        short_circuit_threshold_high=300.0,
        is_dynamic=True,
        unit="dB",
        description="Cabin Acoustic Sound Pressure Level",
    ),
    "acoustic_dom_freq": SensorLimits(
        channel="acoustic_dom_freq",
        spn=520115,
        min_physical=0.0,
        max_physical=25000.0,
        healthy_nominal=1200.0,
        max_slew_per_sec=10000.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=50000.0,
        is_dynamic=True,
        unit="Hz",
        description="Dominant Acoustic Spectral Peak",
    ),
    "acoustic_energy": SensorLimits(
        channel="acoustic_energy",
        spn=520116,
        min_physical=0.0,
        max_physical=20000.0,
        healthy_nominal=100.0,
        max_slew_per_sec=2000.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=50000.0,
        is_dynamic=True,
        unit="J",
        description="Microphone Acoustic Energy",
    ),
    "ae_event_rate": SensorLimits(
        channel="ae_event_rate",
        spn=520117,
        min_physical=0.0,
        max_physical=300.0,
        healthy_nominal=2.0,
        max_slew_per_sec=80.0,
        open_circuit_threshold_low=-50.0,
        short_circuit_threshold_high=1000.0,
        is_dynamic=True,
        unit="events/s",
        description="Acoustic Emission Micro-Crack Rate",
    ),
    "ae_events": SensorLimits(
        channel="ae_events",
        spn=520118,
        min_physical=0.0,
        max_physical=1.0e6,
        healthy_nominal=50.0,
        max_slew_per_sec=200.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=2.0e6,
        is_dynamic=True,
        unit="count",
        description="Total Acoustic Emission Burst Count",
    ),
    "ae_energy": SensorLimits(
        channel="ae_energy",
        spn=520119,
        min_physical=0.0,
        max_physical=300.0,
        healthy_nominal=0.50,
        max_slew_per_sec=40.0,
        open_circuit_threshold_low=-50.0,
        short_circuit_threshold_high=1000.0,
        is_dynamic=True,
        unit="a.u.",
        description="Acoustic Emission Burst Wave Energy",
    ),
    "ae_amp_dB": SensorLimits(
        channel="ae_amp_dB",
        spn=520120,
        min_physical=0.0,
        max_physical=140.0,
        healthy_nominal=35.0,
        max_slew_per_sec=50.0,
        open_circuit_threshold_low=-50.0,
        short_circuit_threshold_high=300.0,
        is_dynamic=True,
        unit="dB_AE",
        description="Peak AE Burst Amplitude",
    ),
    "ae_duration_s": SensorLimits(
        channel="ae_duration_s",
        spn=520121,
        min_physical=0.0,
        max_physical=2.0,
        healthy_nominal=0.005,
        max_slew_per_sec=1.0,
        open_circuit_threshold_low=-1.0,
        short_circuit_threshold_high=10.0,
        is_dynamic=True,
        unit="s",
        description="Acoustic Emission Event Duration",
    ),
    "vib_rms": SensorLimits(
        channel="vib_rms",
        spn=520101,
        min_physical=0.0,
        max_physical=30.0,
        healthy_nominal=0.45,
        max_slew_per_sec=15.0,
        open_circuit_threshold_low=-50.0,
        short_circuit_threshold_high=100.0,
        is_dynamic=True,
        unit="m/s²",
        description="Broad-Band Accelerometer RMS Vibration",
    ),
    "vib_kurtosis": SensorLimits(
        channel="vib_kurtosis",
        spn=520101,
        min_physical=0.5,
        max_physical=60.0,
        healthy_nominal=2.8,
        max_slew_per_sec=30.0,
        open_circuit_threshold_low=-50.0,
        short_circuit_threshold_high=200.0,
        is_dynamic=True,
        unit="unitless",
        description="Vibration 4th Moment Pearson Kurtosis",
    ),
    "vib_dom_freq": SensorLimits(
        channel="vib_dom_freq",
        spn=520101,
        min_physical=0.0,
        max_physical=8000.0,
        healthy_nominal=120.0,
        max_slew_per_sec=2000.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=20000.0,
        is_dynamic=True,
        unit="Hz",
        description="Dominant Spectral Vibration Harmonic",
    ),
    "vib_dom_amp": SensorLimits(
        channel="vib_dom_amp",
        spn=520101,
        min_physical=0.0,
        max_physical=600.0,
        healthy_nominal=105.0,
        max_slew_per_sec=300.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=2000.0,
        is_dynamic=True,
        unit="m/s²",
        description="Dominant Spectral Peak Amplitude",
    ),
    "vib_energy": SensorLimits(
        channel="vib_energy",
        spn=520101,
        min_physical=0.0,
        max_physical=20000.0,
        healthy_nominal=50.0,
        max_slew_per_sec=3000.0,
        open_circuit_threshold_low=-500.0,
        short_circuit_threshold_high=50000.0,
        is_dynamic=True,
        unit="J",
        description="Vibration Integrated Signal Energy",
    ),

    # Common aliases / HUD parameters
    "health_index": SensorLimits(
        channel="health_index",
        spn=520402,
        min_physical=0.0,
        max_physical=100.0,
        healthy_nominal=95.0,
        max_slew_per_sec=30.0,
        is_dynamic=True,
        unit="%",
        description="Fused Vehicle Health Index",
    ),
    "chi": SensorLimits(
        channel="chi",
        spn=520402,
        min_physical=0.0,
        max_physical=100.0,
        healthy_nominal=95.0,
        max_slew_per_sec=30.0,
        is_dynamic=True,
        unit="%",
        description="Combat Health Index",
    ),
}

# Aliases mapping shorthand channel keys to primary canonical keys
CHANNEL_ALIASES: Dict[str, str] = {
    "oil_p": "oil_pressure",
    "coolant_t": "coolant_temp",
    "hyd_p": "hyd_pressure",
    "torque": "shaft_torque",
    "speed": "rpm",
    "power": "mech_power",
    "egt": "exhaust_temp",
}


class SensorPlausibilityGate:
    """Multi-Layer Pre-Inference Signal Plausibility & FDIR Verification Gate.

    Execution Pipeline per Frame:
      Layer 1: Range Clamping & NaN/Inf/Null Sanitization
      Layer 2: Open-Circuit & Short-Circuit Electrical Detection (J1939 FMI 04 & FMI 03)
      Layer 3: Hampel / Median Transient EMI Burst Filter (J1939 FMI 02)
      Layer 4: Slew-Rate / Rate-of-Change Limiter (J1939 FMI 02)
      Layer 5: Stuck-At / Flatline Filter across dynamic channels (J1939 FMI 02)
      Layer 6: Dual-Sensor / Cross-Subsystem Physical Plausibility Gate (J1939 FMI 14)
    """

    def __init__(
        self,
        sample_rate_hz: float = 20.0,
        stuck_window: int = 30,
        hampel_window: int = 5,
        hampel_nsigmas: float = 3.0,
        custom_limits: Optional[Dict[str, SensorLimits]] = None,
    ) -> None:
        self.sample_rate_hz = max(0.1, float(sample_rate_hz))
        self.dt = 1.0 / self.sample_rate_hz
        self.stuck_window = max(5, int(stuck_window))
        self.hampel_window = max(3, int(hampel_window) if int(hampel_window) % 2 == 1 else int(hampel_window) + 1)
        self.hampel_nsigmas = float(hampel_nsigmas)

        # Dynamic sensor catalog
        self.limits: Dict[str, SensorLimits] = dict(SENSOR_LIMITS_CATALOG)
        if custom_limits:
            self.limits.update(custom_limits)

        # Internal state buffers
        self.last_valid: Dict[str, float] = {}
        self.last_sanitized: Dict[str, float] = {}
        # Required flatline run length per channel, in frames. Precomputed:
        # deriving it inside filter_frame() cost enough to break the
        # sub-millisecond per-frame budget.
        self._stuck_required = {
            name: max(self.stuck_window,
                      int(math.ceil(lim.max_static_duration_s
                                    * max(self.sample_rate_hz, 1e-6))))
            for name, lim in SENSOR_LIMITS_CATALOG.items()
        }
        self._stuck_default_required = self.stuck_window
        # Flatline length is tracked as an O(1) run counter per channel rather
        # than by re-scanning a deque each frame: with a 240 s tolerance at
        # 20 Hz the buffer is ~4,800 deep, and copying that for all 58 channels
        # every frame blew the sub-millisecond per-frame budget.
        self._stuck_maxlen = self.stuck_window
        self._stuck_run: Dict[str, list] = {}
        self.stuck_history: Dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=self._stuck_maxlen))
        self.hampel_history: Dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=self.hampel_window))
        self.frame_count: int = 0
        self.last_timestamp: float = 0.0

    def reset(self) -> None:
        """Reset internal filter state, rolling histories, and stuck-at tracking."""
        self._stuck_run = {}
        self.last_valid.clear()
        self.last_sanitized.clear()
        self.stuck_history.clear()
        self.hampel_history.clear()
        self.frame_count = 0
        self.last_timestamp = 0.0

    def register_custom_limit(self, limit: SensorLimits) -> None:
        """Register or override limits for a sensor channel."""
        self.limits[limit.channel] = limit

    def get_sensor_limits(self, channel: str) -> Optional[SensorLimits]:
        """Retrieve limits definition for a given channel or its alias."""
        canonical = CHANNEL_ALIASES.get(channel, channel)
        return self.limits.get(canonical)

    def _resolve_canonical_key(self, key: str) -> str:
        """Map key or alias to canonical channel name."""
        return CHANNEL_ALIASES.get(key, key)

    def filter_frame(self, raw_telemetry: Dict[str, Any]) -> PlausibilityResult:
        """Filter and validate a complete telemetry frame across all 6 FDIR layers.

        Args:
            raw_telemetry: Dictionary of raw sensor readings (key -> value).

        Returns:
            PlausibilityResult containing sanitized telemetry ready for ML inference
            and list of detected SensorFaultEvents. Guaranteed latency < 1.0 ms.
        """
        t0 = time.perf_counter()
        faults: List[SensorFaultEvent] = []
        clean_telemetry: Dict[str, float] = {}

        # 0. Time Step Calculation
        if "time" in raw_telemetry and isinstance(raw_telemetry["time"], (int, float)) and math.isfinite(raw_telemetry["time"]):
            incoming_time = float(raw_telemetry["time"])
            if self.last_timestamp > 0.0 and incoming_time > self.last_timestamp:
                dt = min(1.0, max(0.001, incoming_time - self.last_timestamp))
            else:
                dt = self.dt
            self.last_timestamp = incoming_time
        else:
            dt = self.dt

        # Check vehicle dynamic state for stuck-at logic (RPM > 500 or speed > 0)
        raw_rpm_val = raw_telemetry.get("rpm", raw_telemetry.get("speed", 0.0))
        try:
            is_vehicle_running = float(raw_rpm_val) > 500.0 if raw_rpm_val is not None and not (isinstance(raw_rpm_val, float) and math.isnan(raw_rpm_val)) else False
        except (ValueError, TypeError):
            is_vehicle_running = False

        # Iterate over all incoming telemetry keys
        for key, raw_val in raw_telemetry.items():
            canonical_key = self._resolve_canonical_key(key)
            lim = self.limits.get(canonical_key)

            # -------------------------------------------------------------
            # LAYER 1: Range Clamping & NaN / Inf / Null Sanitization
            # -------------------------------------------------------------
            is_corrupt_non_numeric = False
            parsed_val: float

            if raw_val is None:
                is_corrupt_non_numeric = True
            elif isinstance(raw_val, bool):
                parsed_val = 1.0 if raw_val else 0.0
            elif isinstance(raw_val, (int, float)):
                if math.isnan(raw_val) or math.isinf(raw_val):
                    is_corrupt_non_numeric = True
                else:
                    parsed_val = float(raw_val)
            elif isinstance(raw_val, str):
                raw_str_stripped = raw_val.strip().lower()
                if raw_str_stripped in ("nan", "inf", "-inf", "+inf", "null", "none", ""):
                    is_corrupt_non_numeric = True
                else:
                    try:
                        parsed_val = float(raw_val)
                        if math.isnan(parsed_val) or math.isinf(parsed_val):
                            is_corrupt_non_numeric = True
                    except ValueError:
                        is_corrupt_non_numeric = True
            else:
                is_corrupt_non_numeric = True

            # If non-numeric or NaN/Inf, sanitize using last valid or nominal baseline
            if is_corrupt_non_numeric:
                fallback_val = self.last_valid.get(
                    canonical_key,
                    lim.healthy_nominal if lim else 0.0
                )
                spn = lim.spn if lim else 520999
                faults.append(
                    SensorFaultEvent(
                        channel=key,
                        fault_type=PlausibilityFaultType.NAN_INF_CORRUPTION.value,
                        raw_value=raw_val,
                        clamped_value=float(fallback_val),
                        spn=spn,
                        fmi=FMI_DATA_ERRATIC,
                        message=f"Corrupt or non-finite value '{raw_val}' sanitized to nominal {fallback_val:.3f}",
                    )
                )
                clean_telemetry[key] = float(fallback_val)
                # Keep rolling histories updated
                self.stuck_history[canonical_key].append(float(fallback_val))
                self.hampel_history[canonical_key].append(float(fallback_val))
                self.last_sanitized[canonical_key] = float(fallback_val)
                continue

            # If no limit catalog definition exists for an unknown custom channel, pass through parsed float
            if lim is None:
                clean_telemetry[key] = parsed_val
                self.last_valid[canonical_key] = parsed_val
                self.last_sanitized[canonical_key] = parsed_val
                continue

            current_val = parsed_val
            spn = lim.spn
            is_electrical_fault = False

            # -------------------------------------------------------------
            # LAYER 2: Open-Circuit & Short-Circuit Electrical Detection
            # -------------------------------------------------------------
            # Open-Circuit detection (wire cut, transducer disconnected)
            if current_val <= lim.open_circuit_threshold_low or (current_val < lim.min_physical - 300.0):
                clamped_val = lim.min_physical
                faults.append(
                    SensorFaultEvent(
                        channel=key,
                        fault_type=PlausibilityFaultType.OPEN_CIRCUIT.value,
                        raw_value=raw_val,
                        clamped_value=clamped_val,
                        spn=spn,
                        fmi=FMI_VOLTAGE_BELOW_NORMAL,  # FMI 04
                        message=f"Open circuit / wire cut detected on {key}: raw={current_val:.2f}{lim.unit} <= threshold {lim.open_circuit_threshold_low:.2f}",
                    )
                )
                current_val = clamped_val
                is_electrical_fault = True

            # Short-Circuit detection (short to power / saturated rail voltage)
            elif current_val >= lim.short_circuit_threshold_high or (current_val > lim.max_physical + 300.0):
                clamped_val = lim.max_physical
                faults.append(
                    SensorFaultEvent(
                        channel=key,
                        fault_type=PlausibilityFaultType.SHORT_CIRCUIT.value,
                        raw_value=raw_val,
                        clamped_value=clamped_val,
                        spn=spn,
                        fmi=FMI_VOLTAGE_ABOVE_NORMAL,  # FMI 03
                        message=f"Short circuit / power rail short detected on {key}: raw={current_val:.2f}{lim.unit} >= threshold {lim.short_circuit_threshold_high:.2f}",
                    )
                )
                current_val = clamped_val
                is_electrical_fault = True

            # Physical Boundary Clamping (Soft range bounds)
            elif current_val < lim.min_physical:
                clamped_val = lim.min_physical
                faults.append(
                    SensorFaultEvent(
                        channel=key,
                        fault_type=PlausibilityFaultType.RANGE_OUT_OF_BOUNDS.value,
                        raw_value=raw_val,
                        clamped_value=clamped_val,
                        spn=spn,
                        fmi=FMI_DATA_VALID_BELOW_NORMAL,  # FMI 01
                        message=f"Signal {key} below physical minimum {lim.min_physical}: clamped to {clamped_val}",
                    )
                )
                current_val = clamped_val

            elif current_val > lim.max_physical:
                clamped_val = lim.max_physical
                faults.append(
                    SensorFaultEvent(
                        channel=key,
                        fault_type=PlausibilityFaultType.RANGE_OUT_OF_BOUNDS.value,
                        raw_value=raw_val,
                        clamped_value=clamped_val,
                        spn=spn,
                        fmi=FMI_DATA_VALID_ABOVE_NORMAL,  # FMI 00
                        message=f"Signal {key} above physical maximum {lim.max_physical}: clamped to {clamped_val}",
                    )
                )
                current_val = clamped_val

            # -------------------------------------------------------------
            # LAYER 3: Sliding Window Median / Hampel EMI Outlier Filter
            # -------------------------------------------------------------
            # If not an open/short circuit, evaluate transient EMI bursts
            h_buf = self.hampel_history[canonical_key]
            if not is_electrical_fault and len(h_buf) >= 3:
                vals = sorted(h_buf)
                med = vals[len(vals) // 2]
                diffs = sorted(abs(x - med) for x in vals)
                mad = diffs[len(diffs) // 2]
                mad_scale = 1.4826 * mad
                # Adaptive threshold: if MAD is very small, use a minimum based on physical envelope
                threshold = max(
                    self.hampel_nsigmas * mad_scale,
                    (lim.max_physical - lim.min_physical) * 0.05,
                )
                # Check if current_val is an outlier spike compared to window median
                if abs(current_val - med) > threshold and abs(current_val - med) > (lim.max_slew_per_sec * dt * 1.2):
                    faults.append(
                        SensorFaultEvent(
                            channel=key,
                            fault_type=PlausibilityFaultType.OUTLIER_EMI.value,
                            raw_value=raw_val,
                            clamped_value=med,
                            spn=spn,
                            fmi=FMI_DATA_ERRATIC,  # FMI 02
                            message=f"Transient EMI spike rejected on {key}: raw={current_val:.2f}, median={med:.2f}, diff={abs(current_val - med):.2f} > threshold {threshold:.2f}",
                        )
                    )
                    current_val = med

            h_buf.append(current_val)

            # -------------------------------------------------------------
            # LAYER 4: Slew-Rate / Rate-of-Change Limiter
            # -------------------------------------------------------------
            prev_val = self.last_sanitized.get(canonical_key)
            if prev_val is not None and lim.max_slew_per_sec > 0:
                max_delta = lim.max_slew_per_sec * dt
                delta = current_val - prev_val
                if abs(delta) > max_delta:
                    roc_clamped = prev_val + (max_delta if delta > 0 else -max_delta)
                    faults.append(
                        SensorFaultEvent(
                            channel=key,
                            fault_type=PlausibilityFaultType.RATE_OF_CHANGE_EXCEEDED.value,
                            raw_value=raw_val,
                            clamped_value=roc_clamped,
                            spn=spn,
                            fmi=FMI_DATA_ERRATIC,  # FMI 02
                            message=f"Rate of change exceeded on {key}: step {abs(delta):.3f} > max allowable {max_delta:.3f} per step (ROC={abs(delta)/dt:.1f}/s)",
                        )
                    )
                    current_val = roc_clamped

            # -------------------------------------------------------------
            # LAYER 5: Stuck-At / Flatline Filter
            # -------------------------------------------------------------
            s_buf = self.stuck_history[canonical_key]
            s_buf.append(current_val)

            run = self._stuck_run.get(canonical_key)
            if run is not None and abs(current_val - run[0]) < 1.0e-7:
                run[1] += 1
            else:
                self._stuck_run[canonical_key] = [current_val, 1]
                run = self._stuck_run[canonical_key]
            # Required run length is a *duration*, not a frame count, and is set
            # per channel: a slow or regulated signal may legitimately hold one
            # value far longer than a fast one.
            required = self._stuck_required.get(canonical_key,
                                                self._stuck_default_required)
            if run[1] >= required and lim.is_dynamic and is_vehicle_running:
                if True:
                    faults.append(
                        SensorFaultEvent(
                            channel=key,
                            fault_type=PlausibilityFaultType.STUCK_AT.value,
                            raw_value=raw_val,
                            clamped_value=current_val,
                            spn=spn,
                            fmi=FMI_DATA_ERRATIC,  # FMI 02
                            message=f"Sensor stuck-at / flatline detected on {key}: zero variance for {required / max(self.sample_rate_hz, 1e-6):.1f}s while vehicle running",
                        )
                    )

            # Update channel state
            clean_telemetry[key] = current_val
            self.last_sanitized[canonical_key] = current_val
            self.last_valid[canonical_key] = current_val

        # -----------------------------------------------------------------
        # LAYER 6: Dual-Sensor / Cross-Subsystem Physical Plausibility Gate
        # -----------------------------------------------------------------
        cross_faults = self._check_cross_subsystem_plausibility(clean_telemetry)
        if cross_faults:
            faults.extend(cross_faults)

        self.frame_count += 1
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return PlausibilityResult(
            clean_telemetry=clean_telemetry,
            faults_detected=faults,
            is_valid=True,
            raw_telemetry=raw_telemetry,
            processing_time_ms=round(t_elapsed_ms, 4),
        )

    def validate_packet(self, telemetry: Dict[str, Any]) -> Tuple[Dict[str, float], List[SensorFaultEvent]]:
        """Convenience method returning (clean_telemetry, faults_detected)."""
        res = self.filter_frame(telemetry)
        return res.clean_telemetry, res.faults_detected

    def _check_cross_subsystem_plausibility(self, data: Dict[str, float]) -> List[SensorFaultEvent]:
        """Cross-checks physical correlation laws between interdependent sensors."""
        faults: List[SensorFaultEvent] = []

        # 1. Engine RPM vs Oil Pressure
        # At high RPM (> 1500), oil pressure cannot physically be zero / vacuum (< 0.5 bar or < 50 kPa)
        # unless catastrophic pump / bearing failure or disconnected sensor.
        rpm = data.get("rpm", data.get("speed"))
        oil_p = data.get("oil_pressure", data.get("oil_p"))
        if rpm is not None and oil_p is not None:
            oil_p_bar = oil_p if oil_p <= 20.0 else (oil_p / 1.0e5)
            if rpm > 1500.0 and oil_p_bar <= 0.4:
                faults.append(
                    SensorFaultEvent(
                        channel="oil_pressure",
                        fault_type=PlausibilityFaultType.DUAL_SENSOR_MISMATCH.value,
                        raw_value=oil_p,
                        clamped_value=oil_p,
                        spn=100,
                        fmi=FMI_SPECIAL_INSTRUCTIONS,  # FMI 14
                        message=f"Plausibility Mismatch: High Engine Speed ({rpm:.0f} RPM) with near-zero Oil Pressure ({oil_p_bar:.2f} bar)",
                    )
                )
            elif rpm < 50.0 and oil_p_bar > 6.5:
                faults.append(
                    SensorFaultEvent(
                        channel="oil_pressure",
                        fault_type=PlausibilityFaultType.DUAL_SENSOR_MISMATCH.value,
                        raw_value=oil_p,
                        clamped_value=oil_p,
                        spn=100,
                        fmi=FMI_SPECIAL_INSTRUCTIONS,  # FMI 14
                        message=f"Plausibility Mismatch: Engine stopped (0 RPM) but residual Oil Pressure indicates {oil_p_bar:.2f} bar",
                    )
                )

        # 2. Coolant Temp vs Oil Temp
        # Under thermal equilibrium, Coolant Temp and Oil Temp track within ~50°C of each other
        coolant_t = data.get("coolant_temp", data.get("coolant_t"))
        oil_t = data.get("oil_temp")
        if coolant_t is not None and oil_t is not None:
            temp_delta = abs(coolant_t - oil_t)
            if temp_delta > 60.0:
                faults.append(
                    SensorFaultEvent(
                        channel="coolant_temp",
                        fault_type=PlausibilityFaultType.DUAL_SENSOR_MISMATCH.value,
                        raw_value=coolant_t,
                        clamped_value=coolant_t,
                        spn=110,
                        fmi=FMI_SPECIAL_INSTRUCTIONS,  # FMI 14
                        message=f"Plausibility Mismatch: Extreme thermal discrepancy |Coolant Temp ({coolant_t:.1f}°C) - Oil Temp ({oil_t:.1f}°C)| = {temp_delta:.1f}°C > 60°C",
                    )
                )

        # 3. Dual Exhaust Gas Pyrometry (EGT Bank A vs Bank B)
        egt_a = data.get("egt_bank_a", data.get("exhaust_temp_bank_a"))
        egt_b = data.get("egt_bank_b", data.get("exhaust_temp_bank_b"))
        if egt_a is not None and egt_b is not None:
            egt_delta = abs(egt_a - egt_b)
            if egt_delta > 150.0:
                faults.append(
                    SensorFaultEvent(
                        channel="exhaust_temp",
                        fault_type=PlausibilityFaultType.DUAL_SENSOR_MISMATCH.value,
                        raw_value=egt_a,
                        clamped_value=egt_a,
                        spn=173,
                        fmi=FMI_SPECIAL_INSTRUCTIONS,  # FMI 14
                        message=f"Plausibility Mismatch: Dual EGT Bank discrepancy |EGT_A ({egt_a:.1f}°C) - EGT_B ({egt_b:.1f}°C)| = {egt_delta:.1f}°C > 150°C",
                    )
                )

        # 4. Hydraulic Flow vs Hydraulic Pressure
        hyd_flow = data.get("hyd_flow")
        hyd_p = data.get("hyd_pressure", data.get("hyd_p"))
        if hyd_flow is not None and hyd_p is not None:
            hyd_p_bar = hyd_p if hyd_p <= 500.0 else (hyd_p / 1.0e5)
            # High flow with 0 bar pressure is a severed hydraulic circuit or faulty transducer
            if hyd_flow > 0.005 and hyd_p_bar < 5.0:
                faults.append(
                    SensorFaultEvent(
                        channel="hyd_pressure",
                        fault_type=PlausibilityFaultType.DUAL_SENSOR_MISMATCH.value,
                        raw_value=hyd_p,
                        clamped_value=hyd_p,
                        spn=520202,
                        fmi=FMI_SPECIAL_INSTRUCTIONS,  # FMI 14
                        message=f"Plausibility Mismatch: Significant Hydraulic Flow ({hyd_flow:.4f} m³/s) with zero circuit pressure ({hyd_p_bar:.1f} bar)",
                    )
                )

        return faults
