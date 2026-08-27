"""SAE J1939-73 Diagnostic Trouble Code (DTC) Engine & Military Flash Ring Buffer.

Implements standard SAE J1939-73 DM1 (PGN 65226 / 0xFECA) Active Malfunction and
DM2 (PGN 65227 / 0xFECB) Historic Malfunction diagnostic protocol encoding, decoding,
FDIR plausibility fault conversion, neural prognostics fault mapping, and thread-safe
on-disk circular flash ring buffer persistence for military armored fighting vehicles
(CVRDE Arjun Mk-1A, T-90S Bhishma, Zorawar Light Tank).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import math
import os
from pathlib import Path
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger("dtc_engine")


# ============================================================================
# SAE J1939-73 Protocol Constants
# ============================================================================

PGN_DM1 = 65226   # 0xFECA: Active Diagnostic Trouble Codes
PGN_DM2 = 65227   # 0xFECB: Previously Active Diagnostic Trouble Codes
PGN_DM3 = 65228   # 0xFECC: Diagnostic Data Clear/Reset for Previously Active DTCs
PGN_DM11 = 65235  # 0xFED3: Diagnostic Data Clear/Reset for Active DTCs

# Lamp Status 2-bit bitfield values (SAE J1939-73 Table 1)
LAMP_OFF = 0b00            # 0: Lamp Off
LAMP_ON = 0b01             # 1: Lamp On
LAMP_ERROR = 0b10          # 2: Error
LAMP_NOT_AVAILABLE = 0b11  # 3: Not Available / Unaffected

# Flash State 2-bit bitfield values (SAE J1939-73 Table 2)
FLASH_SLOW = 0b00          # 0: Slow Flash (1 Hz)
FLASH_FAST = 0b01          # 1: Fast Flash (2 Hz or continuous)
FLASH_RESERVED = 0b10      # 2: Reserved
FLASH_UNAVAILABLE = 0b11   # 3: Unavailable / Do Not Flash (default)


class LampType(str, Enum):
    """J1939 Warning and Indicator Lamp classifications."""
    MIL = "MIL"                     # Malfunction Indicator Lamp (Emissions / Core Powertrain)
    RED_STOP = "RED_STOP"           # Red Stop Lamp (Critical Safety / Severe Mechanical Shutdown)
    AMBER_WARNING = "AMBER_WARNING" # Amber Warning Lamp (Degraded System / Check Required)
    PROTECT = "PROTECT"             # Protect Lamp (Thermal / Boundary Protection Action)
    NONE = "NONE"                   # No lamp illuminated


# Standard SAE J1939 Failure Mode Identifiers (FMIs)
FMI_DATA_VALID_ABOVE_NORMAL = 0   # FMI 00: Data valid but above normal operational range (Crit High)
FMI_DATA_VALID_BELOW_NORMAL = 1   # FMI 01: Data valid but below normal operational range (Crit Low)
FMI_DATA_ERRATIC = 2              # FMI 02: Data erratic, intermittent, or incorrect (Slew/Stuck/EMI)
FMI_VOLTAGE_ABOVE_NORMAL = 3      # FMI 03: Voltage above normal or shorted to high source
FMI_VOLTAGE_BELOW_NORMAL = 4      # FMI 04: Voltage below normal or shorted to low source / wire cut
FMI_CURRENT_BELOW_NORMAL = 5      # FMI 05: Current below normal or open circuit
FMI_CURRENT_ABOVE_NORMAL = 6      # FMI 06: Current above normal or grounded circuit
FMI_MECHANICAL_SYSTEM_FAIL = 7    # FMI 07: Mechanical system not responding or degraded
FMI_ABNORMAL_FREQUENCY = 8        # FMI 08: Abnormal frequency, pulse width, or period
FMI_ABNORMAL_UPDATE_RATE = 9      # FMI 09: Abnormal update rate
FMI_ABNORMAL_RATE_OF_CHANGE = 10  # FMI 10: Abnormal rate of change
FMI_ROOT_CAUSE_NOT_IDENT = 11     # FMI 11: Root cause not identifiable
FMI_BAD_INTELLIGENT_DEVICE = 12   # FMI 12: Bad intelligent device or component
FMI_OUT_OF_CALIBRATION = 13       # FMI 13: Out of calibration
FMI_SPECIAL_INSTRUCTIONS = 14     # FMI 14: Special instructions / Plausibility cross-mismatch

FMI_DESCRIPTIONS: Dict[int, str] = {
    0: "Data Valid Above Normal Range (Most Severe)",
    1: "Data Valid Below Normal Range (Most Severe)",
    2: "Data Erratic, Intermittent, or Incorrect",
    3: "Voltage Above Normal / Shorted to High Source",
    4: "Voltage Below Normal / Shorted to Low Source (Wire Cut)",
    5: "Current Below Normal / Open Circuit",
    6: "Current Above Normal / Grounded Circuit",
    7: "Mechanical System Not Responding / Mechanical Degradation",
    8: "Abnormal Frequency, Pulse Width, or Period",
    9: "Abnormal Update Rate",
    10: "Abnormal Rate of Change",
    11: "Root Cause Not Identifiable",
    12: "Bad Intelligent Device or Component",
    13: "Out of Calibration",
    14: "Special Instructions / Cross-Subsystem Mismatch",
}


# Comprehensive SAE J1939 Suspect Parameter Numbers (SPNs) for Military AFVs
SPN_DESCRIPTIONS: Dict[int, str] = {
    92: "Engine Percent Load At Current Speed",
    96: "Fuel Level",
    98: "Engine Oil Level",
    100: "Engine Oil Pressure",
    102: "Engine Intake Manifold Pressure / Turbo Boost",
    105: "Engine Intake Manifold Temperature",
    110: "Engine Coolant Temperature",
    111: "Coolant Level",
    132: "Engine Air Mass Flow Rate",
    173: "Engine Exhaust Gas Temperature (EGT)",
    175: "Engine Oil Sump Temperature",
    190: "Engine Crankshaft Speed (RPM)",
    513: "Actual Engine Percent Torque / Transmission Shaft Torque",
    520001: "Mission Elapsed Time",
    520002: "Simulation Step Counter",
    520101: "Drivetrain / Final Drive Bearing Vibration",
    520102: "Hydraulic Actuator Pressure / Fuel Injection Lambda",
    520103: "HSU Suspension Nitrogen Pressure / Exhaust O2 Concentration",
    520104: "Hydraulic Recoil Flow / Oil Viscosity",
    520105: "Lubrication Oil Flow Rate",
    520106: "Ferromagnetic Debris Cumulative Count",
    520107: "Instantaneous Debris Generation Rate",
    520108: "Debris Particles per Step",
    520109: "Shaft Torsional Shear Stress",
    520110: "Shaft Elastic Shear Strain",
    520111: "Delivered Shaft Mechanical Power",
    520112: "Sprocket Shaft Angular Velocity",
    520113: "Capacitive Fuel Level Probe",
    520114: "Cabin Acoustic Sound Pressure Level (SPL)",
    520115: "Dominant Acoustic Spectral Peak",
    520116: "Microphone Acoustic Energy",
    520117: "Acoustic Emission Micro-Crack Rate",
    520118: "Total Acoustic Emission Burst Count",
    520119: "Acoustic Emission Burst Wave Energy",
    520120: "Peak Acoustic Emission Burst Amplitude",
    520121: "Acoustic Emission Event Duration",
    520200: "CVRDE Hydrogas Suspension Station (HSU 1-14)",
    520201: "CVRDE Gun Control System & Recoil Buffer",
    520202: "Turret & Recoil Hydraulic Pressure",
    520203: "Main Hydraulic Pump Flow Rate",
    520204: "Gun Elevation Actuator Force",
    520205: "Hydraulic System Power",
    520206: "Hydraulic Seal Leak Bypass Flow",
    520207: "Suspension Trailing Arm Stress",
    520208: "Suspension Strain Gauge Deflection",
    520209: "Wheatstone Bridge Delta Resistance",
    520210: "Torsion Bar Twisting Moment",
    520211: "Torsion Bar Angular Deflection",
    520212: "Torsion Bar Outer Fiber Shear",
    520213: "Plastic Fatigue Cumulative Twist",
    520214: "Hull Vertical Acceleration RMS",
    520215: "Hull Peak Impulse Acceleration",
    520216: "Dynamic Shock Absorbed Energy",
    520300: "CVRDE NBC Cabin Filtration & Positive Overpressure",
    520401: "ISO 8608 Cross-Country Terrain Roughness",
}

# Channel Name to Standard SPN Mapping
CHANNEL_TO_SPN_MAP: Dict[str, int] = {
    "rpm": 190,
    "load": 92,
    "terrain": 520401,
    "coolant_temp": 110,
    "coolant_rtd_ohm": 110,
    "exhaust_temp": 173,
    "exhaust_thermocouple_v": 173,
    "exhaust_pressure": 102,
    "exhaust_mass_flow": 132,
    "lambda": 520102,
    "exhaust_o2_pct": 520103,
    "oil_pressure": 100,
    "oil_temp": 175,
    "oil_viscosity": 520104,
    "oil_flow": 520105,
    "debris_cumulative": 520106,
    "debris_rate": 520107,
    "debris_particles": 520108,
    "shaft_torque": 513,
    "shaft_shear_stress": 520109,
    "shaft_shear_strain": 520110,
    "mech_power": 520111,
    "shaft_omega": 520112,
    "fuel_level": 96,
    "fuel_volume": 96,
    "oil_level": 98,
    "coolant_level": 111,
    "fuel_capacitance_pf": 520113,
    "hyd_pressure": 520202,
    "hyd_flow": 520203,
    "hyd_force": 520204,
    "hyd_power": 520205,
    "hyd_leak_flow": 520206,
    "susp_load_kN": 520200,
    "susp_stress_MPa": 520207,
    "susp_strain_ue": 520208,
    "susp_dR_ohm": 520209,
    "torsion_torque": 520210,
    "torsion_twist_deg": 520211,
    "torsion_shear_MPa": 520212,
    "torsion_cumulative_twist": 520213,
    "shock_a_rms_g": 520214,
    "shock_peak_g": 520215,
    "shock_energy": 520216,
    "spl_db": 520114,
    "acoustic_dom_freq": 520115,
    "acoustic_energy": 520116,
    "ae_event_rate": 520117,
    "ae_events": 520118,
    "ae_energy": 520119,
    "ae_amp_dB": 520120,
    "ae_duration_s": 520121,
    "vib_rms": 520101,
    "vib_kurtosis": 520101,
    "vib_dom_freq": 520101,
    "vib_dom_amp": 520101,
    "vib_energy": 520101,
}

# 13 AI / Neural Fault Class Name to SPN & FMI Mapping
NEURAL_FAULT_CLASS_MAP: Dict[str, Dict[str, Any]] = {
    "bearing_wear": {
        "spn": 520101,
        "fmi": FMI_MECHANICAL_SYSTEM_FAIL,
        "lamp": "AMBER_WARNING",
        "description": "Drive Bearing Mechanical Wear / Vibration Degradation",
        "subsystem": "powertrain",
    },
    "bearing_clearance_wear": {
        "spn": 520101,
        "fmi": FMI_MECHANICAL_SYSTEM_FAIL,
        "lamp": "AMBER_WARNING",
        "description": "Bearing Journal Clearance Wear / Debris Emission",
        "subsystem": "powertrain",
    },
    "gear_wear": {
        "spn": 513,
        "fmi": FMI_MECHANICAL_SYSTEM_FAIL,
        "lamp": "AMBER_WARNING",
        "description": "Transmission Gear Tooth Mesh Degradation / Torque Ripple",
        "subsystem": "powertrain",
    },
    "cooling_failure": {
        "spn": 110,
        "fmi": FMI_DATA_VALID_ABOVE_NORMAL,
        "lamp": "RED_STOP",
        "description": "Engine Cooling System Severe Thermal Overheat",
        "subsystem": "cooling",
    },
    "oil_pump_degradation": {
        "spn": 100,
        "fmi": FMI_DATA_VALID_BELOW_NORMAL,
        "lamp": "RED_STOP",
        "description": "Lubrication Oil Pump Mechanical Loss / Gallery Starvation",
        "subsystem": "lubrication",
    },
    "seal_leakage": {
        "spn": 520206,
        "fmi": FMI_MECHANICAL_SYSTEM_FAIL,
        "lamp": "AMBER_WARNING",
        "description": "Hydraulic / Oil Dynamic Seal Bypass Leakage",
        "subsystem": "hydraulics",
    },
    "fuel_injector_fault": {
        "spn": 520102,
        "fmi": FMI_SPECIAL_INSTRUCTIONS,
        "lamp": "MIL",
        "description": "Fuel Injector Drift / Cylinder Mixture Imbalance",
        "subsystem": "engine",
    },
    "exhaust_restriction": {
        "spn": 102,
        "fmi": FMI_DATA_VALID_ABOVE_NORMAL,
        "lamp": "AMBER_WARNING",
        "description": "Exhaust Manifold Restriction / Backpressure Surge",
        "subsystem": "engine",
    },
    "torsion_fatigue": {
        "spn": 520210,
        "fmi": FMI_MECHANICAL_SYSTEM_FAIL,
        "lamp": "AMBER_WARNING",
        "description": "Suspension Torsion Bar Structural Fatigue / Modulus Loss",
        "subsystem": "structure",
    },
    "hydraulic_valve_fault": {
        "spn": 520202,
        "fmi": FMI_MECHANICAL_SYSTEM_FAIL,
        "lamp": "AMBER_WARNING",
        "description": "Turret Electro-Hydraulic Elevation Servo Valve Sticking",
        "subsystem": "hydraulics",
    },
    "structural_crack": {
        "spn": 520117,
        "fmi": FMI_MECHANICAL_SYSTEM_FAIL,
        "lamp": "RED_STOP",
        "description": "Hull Structural Micro-Crack Growth / Acoustic Emission Surge",
        "subsystem": "structure",
    },
    "drivetrain_efficiency_loss": {
        "spn": 520111,
        "fmi": FMI_MECHANICAL_SYSTEM_FAIL,
        "lamp": "AMBER_WARNING",
        "description": "Final Drive Kinematic Power Transfer Efficiency Loss",
        "subsystem": "powertrain",
    },
}

# 8 Monitored Subsystems Default SPNs for Critical Low-RUL Breaches
SUBSYSTEM_RUL_SPN_MAP: Dict[str, Dict[str, Any]] = {
    "engine": {"spn": 190, "fmi": FMI_MECHANICAL_SYSTEM_FAIL, "lamp": "RED_STOP", "desc": "Engine Core Subsystem End-of-Life"},
    "powertrain": {"spn": 520101, "fmi": FMI_MECHANICAL_SYSTEM_FAIL, "lamp": "RED_STOP", "desc": "Powertrain / Transmission End-of-Life"},
    "lubrication": {"spn": 100, "fmi": FMI_DATA_VALID_BELOW_NORMAL, "lamp": "RED_STOP", "desc": "Lubrication Oil System Depletion"},
    "cooling": {"spn": 110, "fmi": FMI_DATA_VALID_ABOVE_NORMAL, "lamp": "RED_STOP", "desc": "Cooling System Thermal Depletion"},
    "hydraulics": {"spn": 520202, "fmi": FMI_DATA_VALID_BELOW_NORMAL, "lamp": "RED_STOP", "desc": "Gun / Turret Hydraulic System Failure"},
    "suspension": {"spn": 520200, "fmi": FMI_MECHANICAL_SYSTEM_FAIL, "lamp": "RED_STOP", "desc": "Hydrogas Suspension Station Failure"},
    "structure": {"spn": 520210, "fmi": FMI_MECHANICAL_SYSTEM_FAIL, "lamp": "RED_STOP", "desc": "Hull / Torsion Structural Limit Reached"},
    "overall": {"spn": 520002, "fmi": FMI_MECHANICAL_SYSTEM_FAIL, "lamp": "RED_STOP", "desc": "Overall Vehicle Readiness Deadline Breached"},
}


# ============================================================================
# LampStatus Data Structure & 2-Byte Binary Header
# ============================================================================

@dataclass
class LampStatus:
    """SAE J1939 DM1/DM2 2-byte Lamp Status Header representation.

    Byte 1: Lamp State (2 bits per lamp)
      - Bits 8-7: Malfunction Indicator Lamp (MIL)
      - Bits 6-5: Red Stop Lamp
      - Bits 4-3: Amber Warning Lamp
      - Bits 2-1: Protect Lamp
    Byte 2: Lamp Flash State (2 bits per lamp)
      - Bits 8-7: MIL Flash State
      - Bits 6-5: Red Stop Lamp Flash State
      - Bits 4-3: Amber Warning Lamp Flash State
      - Bits 2-1: Protect Lamp Flash State
    """
    mil: int = LAMP_OFF
    red_stop: int = LAMP_OFF
    amber_warning: int = LAMP_OFF
    protect: int = LAMP_OFF
    mil_flash: int = FLASH_UNAVAILABLE
    red_stop_flash: int = FLASH_UNAVAILABLE
    amber_warning_flash: int = FLASH_UNAVAILABLE
    protect_flash: int = FLASH_UNAVAILABLE

    def encode_header(self) -> bytes:
        """Encodes lamp status into 2-byte SAE J1939 header."""
        b1 = (
            ((self.mil & 0x03) << 6)
            | ((self.red_stop & 0x03) << 4)
            | ((self.amber_warning & 0x03) << 2)
            | (self.protect & 0x03)
        )
        b2 = (
            ((self.mil_flash & 0x03) << 6)
            | ((self.red_stop_flash & 0x03) << 4)
            | ((self.amber_warning_flash & 0x03) << 2)
            | (self.protect_flash & 0x03)
        )
        return struct.pack("<BB", b1, b2)

    @classmethod
    def decode_header(cls, header: bytes) -> LampStatus:
        """Decodes 2-byte header into LampStatus instance."""
        if len(header) < 2:
            raise ValueError(f"Lamp status header must be at least 2 bytes (got {len(header)})")
        b1, b2 = struct.unpack("<BB", header[:2])
        return cls(
            mil=(b1 >> 6) & 0x03,
            red_stop=(b1 >> 4) & 0x03,
            amber_warning=(b1 >> 2) & 0x03,
            protect=b1 & 0x03,
            mil_flash=(b2 >> 6) & 0x03,
            red_stop_flash=(b2 >> 4) & 0x03,
            amber_warning_flash=(b2 >> 2) & 0x03,
            protect_flash=b2 & 0x03,
        )

    @property
    def has_active_lamp(self) -> bool:
        """Returns True if any warning/stop/protect/MIL lamp is active."""
        return (
            self.mil == LAMP_ON
            or self.red_stop == LAMP_ON
            or self.amber_warning == LAMP_ON
            or self.protect == LAMP_ON
        )

    @property
    def is_red_stop(self) -> bool:
        return self.red_stop == LAMP_ON

    @property
    def is_amber(self) -> bool:
        return self.amber_warning == LAMP_ON

    @property
    def is_mil(self) -> bool:
        return self.mil == LAMP_ON

    @property
    def is_protect(self) -> bool:
        return self.protect == LAMP_ON

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mil": self.mil,
            "red_stop": self.red_stop,
            "amber_warning": self.amber_warning,
            "protect": self.protect,
            "mil_flash": self.mil_flash,
            "red_stop_flash": self.red_stop_flash,
            "amber_warning_flash": self.amber_warning_flash,
            "protect_flash": self.protect_flash,
            "has_active_lamp": self.has_active_lamp,
        }


# ============================================================================
# DTCRecord (4-Byte SAE J1939 Packed Diagnostic Trouble Code)
# ============================================================================

@dataclass
class DTCRecord:
    """SAE J1939 Diagnostic Trouble Code (DTC) Record.

    Standard 4-Byte Binary Representation (SAE J1939-73 §5.7.1):
      Byte 1: SPN bits 7-0 (LSB)
      Byte 2: SPN bits 15-8
      Byte 3: [SPN bits 18-16 (upper 3 bits shifted left by 5)] | [FMI bits 4-0 (lower 5 bits)]
      Byte 4: [CM bit (bit 7, 0 standard)] | [Occurrence Count bits 6-0 (0-127)]
    """
    spn: int                              # 19-bit Suspect Parameter Number (0 .. 524287)
    fmi: int                              # 5-bit Failure Mode Identifier (0 .. 31)
    oc: int = 1                           # 7-bit Occurrence Count (1 .. 127)
    cm: int = 0                           # 1-bit SPN Conversion Method (0 standard)
    description: str = ""                 # Human-readable fault description
    lamp_status: str = "AMBER_WARNING"    # "MIL", "RED_STOP", "AMBER_WARNING", "PROTECT", "NONE"
    timestamp: str = ""                   # ISO-8601 formatted timestamp string
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    active: bool = True
    # Timestamp of the most recent occurrence-count increment. Debouncing
    # against first_seen only suppressed the first second, after which every
    # sample of a *single* continuous fault incremented -- a 20 Hz feed drove
    # OC to 61 in three seconds and saturated at 127 in under seven. J1939
    # occurrence count means distinct occurrences, not samples.
    last_oc_increment: float = 0.0
    source_type: str = "FDIR_PLAUSIBILITY" # "FDIR_ELECTRICAL", "FDIR_PLAUSIBILITY", "NEURAL_PROGNOSTIC", "MANUAL"
    raw_value: Any = None
    clamped_value: Any = None
    channel: str = ""
    subsystem: str = ""

    def __post_init__(self):
        # Enforce range limits on SPN, FMI, OC, CM
        self.spn = int(self.spn) & 0x7FFFF       # 19-bit mask
        self.fmi = int(self.fmi) & 0x1F          # 5-bit mask
        self.oc = min(max(int(self.oc), 1), 127) # 7-bit [1, 127]
        self.cm = int(self.cm) & 0x01            # 1-bit

        if not self.description:
            spn_desc = SPN_DESCRIPTIONS.get(self.spn, f"SPN {self.spn}")
            fmi_desc = FMI_DESCRIPTIONS.get(self.fmi, f"FMI {self.fmi}")
            self.description = f"{spn_desc} - {fmi_desc}"

        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(self.last_seen))

        if self.last_oc_increment <= 0.0:
            self.last_oc_increment = self.first_seen

    def encode_4bytes(self) -> bytes:
        """Encodes DTC into standard SAE J1939 4-byte packed binary struct."""
        b1 = self.spn & 0xFF
        b2 = (self.spn >> 8) & 0xFF
        b3 = (((self.spn >> 16) & 0x07) << 5) | (self.fmi & 0x1F)
        b4 = ((self.cm & 0x01) << 7) | (self.oc & 0x7F)
        return struct.pack("<BBBB", b1, b2, b3, b4)

    @classmethod
    def decode_4bytes(cls, data: bytes, **kwargs: Any) -> DTCRecord:
        """Decodes 4-byte SAE J1939 packed binary into a DTCRecord."""
        if len(data) < 4:
            raise ValueError(f"DTC binary chunk must be at least 4 bytes (got {len(data)})")
        b1, b2, b3, b4 = struct.unpack("<BBBB", data[:4])
        spn = b1 | (b2 << 8) | (((b3 >> 5) & 0x07) << 16)
        fmi = b3 & 0x1F
        cm = (b4 >> 7) & 0x01
        oc = b4 & 0x7F
        return cls(spn=spn, fmi=fmi, oc=max(oc, 1), cm=cm, **kwargs)

    @property
    def formatted_code(self) -> str:
        """Formatted human code string, e.g. 'SPN 100 FMI 04'."""
        return f"SPN {self.spn} FMI {self.fmi:02d}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spn": self.spn,
            "fmi": self.fmi,
            "oc": self.oc,
            "cm": self.cm,
            "code": self.formatted_code,
            "description": self.description,
            "lamp_status": self.lamp_status,
            "timestamp": self.timestamp,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "active": self.active,
            "source_type": self.source_type,
            "raw_value": self.raw_value,
            "clamped_value": self.clamped_value,
            "channel": self.channel,
            "subsystem": self.subsystem,
            "hex_4bytes": self.encode_4bytes().hex().upper(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DTCRecord:
        return cls(
            spn=d["spn"],
            fmi=d["fmi"],
            oc=d.get("oc", 1),
            cm=d.get("cm", 0),
            description=d.get("description", ""),
            lamp_status=d.get("lamp_status", "AMBER_WARNING"),
            timestamp=d.get("timestamp", ""),
            first_seen=d.get("first_seen", time.time()),
            last_seen=d.get("last_seen", time.time()),
            active=d.get("active", True),
            source_type=d.get("source_type", "FDIR_PLAUSIBILITY"),
            raw_value=d.get("raw_value"),
            clamped_value=d.get("clamped_value"),
            channel=d.get("channel", ""),
            subsystem=d.get("subsystem", ""),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DTCRecord):
            return False
        return self.spn == other.spn and self.fmi == other.fmi

    def __hash__(self) -> int:
        return hash((self.spn, self.fmi))

    def __repr__(self) -> str:
        return (
            f"DTCRecord({self.formatted_code}, oc={self.oc}, lamp={self.lamp_status}, "
            f"active={self.active}, desc='{self.description}')"
        )


# Alias for backward compatibility & spec parity
DTC = DTCRecord


# ============================================================================
# DTCEngine Core SAE J1939-73 Implementation
# ============================================================================

class DTCEngine:
    """SAE J1939-73 Diagnostic Trouble Code (DTC) Engine & Flash Ring Buffer.

    Manages active (DM1) and historic (DM2) DTC states, dual-hook ingestion from FDIR
    sensor plausibility and AI prognostics models, binary packet encoding/decoding,
    and thread-safe persistent flash ring buffer logging.
    """

    def __init__(
        self,
        flash_capacity: int = 10000,
        flash_file: Union[str, Path] = "results/dtc_flash_log.jsonl",
        vehicle_id: str = "CVRDE_ARJUN_MK1A",
        auto_recover: bool = True,
        reoccurrence_window_s: float = 1.0,
    ):
        # A fault must go quiet for at least this long before being counted as
        # a fresh occurrence. Without it a continuously present fault on a
        # 20 Hz feed increments once per sample.
        self.reoccurrence_window_s = float(reoccurrence_window_s)
        self.flash_capacity = max(100, int(flash_capacity))
        self.flash_file = Path(flash_file)
        self.vehicle_id = vehicle_id
        self._lock = threading.RLock()

        self._active_dtcs: Dict[Tuple[int, int], DTCRecord] = {}
        self._historic_dtcs: Dict[Tuple[int, int], DTCRecord] = {}

        # Ensure directory exists
        self.flash_file.parent.mkdir(parents=True, exist_ok=True)

        # O(1) line counter — seeded once at startup, then maintained via increment.
        # Avoids reading the whole file on every persist_to_flash() call.
        self._flash_line_count: int = 0
        if self.flash_file.exists():
            try:
                with open(self.flash_file, "rb") as _fh:
                    self._flash_line_count = _fh.read().count(b"\n")
            except OSError:
                self._flash_line_count = 0

        if auto_recover and self.flash_file.exists():
            self.recover_from_flash()


    # ------------------------------------------------------------------------
    # Ingress Hook 1: Sensor Plausibility FDIR Faults
    # ------------------------------------------------------------------------

    def process_fdir_faults(self, faults: Sequence[Any]) -> List[DTCRecord]:
        """Ingests a sequence of `SensorFaultEvent`s or fault dicts from `sensor_plausibility.py`.

        Converts electrical, physical, and cross-subsystem plausibility breaches
        into standard J1939 SPN/FMI DTCs and activates them in the DM1 registry.
        """
        if not faults:
            return []

        generated: List[DTCRecord] = []
        with self._lock:
            for fault in faults:
                if hasattr(fault, "to_dict"):
                    f_dict = fault.to_dict()
                elif isinstance(fault, dict):
                    f_dict = fault
                else:
                    f_dict = {
                        "channel": getattr(fault, "channel", ""),
                        "fault_type": getattr(fault, "fault_type", "FDIR_FAULT"),
                        "raw_value": getattr(fault, "raw_value", None),
                        "clamped_value": getattr(fault, "clamped_value", None),
                        "spn": getattr(fault, "spn", 0),
                        "fmi": getattr(fault, "fmi", FMI_DATA_ERRATIC),
                        "message": getattr(fault, "message", ""),
                        "timestamp": getattr(fault, "timestamp", time.time()),
                    }

                channel = f_dict.get("channel", "")
                fault_type = str(f_dict.get("fault_type", "")).upper()
                spn = int(f_dict.get("spn", 0))
                fmi = int(f_dict.get("fmi", FMI_DATA_ERRATIC))
                raw_val = f_dict.get("raw_value")
                clamped_val = f_dict.get("clamped_value")
                ts = float(f_dict.get("timestamp", time.time()))

                # Resolve SPN if missing or 0
                if spn <= 0 and channel:
                    spn = CHANNEL_TO_SPN_MAP.get(channel, 520000)

                # Determine Lamp Status based on fault type & channel criticality
                lamp = "AMBER_WARNING"
                if "OPEN_CIRCUIT" in fault_type or "WIRE_CUT" in fault_type:
                    fmi = FMI_VOLTAGE_BELOW_NORMAL
                    lamp = "RED_STOP" if channel in ("oil_pressure", "coolant_temp", "rpm", "hyd_pressure") else "AMBER_WARNING"
                elif "SHORT_CIRCUIT" in fault_type or "SHORT_TO_POWER" in fault_type:
                    fmi = FMI_VOLTAGE_ABOVE_NORMAL
                    lamp = "RED_STOP" if channel in ("oil_pressure", "coolant_temp", "hyd_pressure") else "AMBER_WARNING"
                elif "RATE_OF_CHANGE" in fault_type or "ROC" in fault_type:
                    fmi = FMI_ABNORMAL_RATE_OF_CHANGE
                    lamp = "AMBER_WARNING"
                elif "STUCK_AT" in fault_type or "FLATLINE" in fault_type:
                    fmi = FMI_DATA_ERRATIC
                    lamp = "AMBER_WARNING"
                elif "OUTLIER_EMI" in fault_type or "EMI" in fault_type:
                    fmi = FMI_DATA_ERRATIC
                    lamp = "AMBER_WARNING"
                elif "DUAL_SENSOR_MISMATCH" in fault_type or "MISMATCH" in fault_type:
                    fmi = FMI_SPECIAL_INSTRUCTIONS
                    lamp = "RED_STOP" if "oil_pressure" in channel or "rpm" in channel else "AMBER_WARNING"
                elif "NAN_INF_CORRUPTION" in fault_type:
                    fmi = FMI_DATA_ERRATIC
                    lamp = "AMBER_WARNING"

                dtc = self._activate_or_increment_fault(
                    spn=spn,
                    fmi=fmi,
                    lamp=lamp,
                    source_type="FDIR_ELECTRICAL" if "CIRCUIT" in fault_type else "FDIR_PLAUSIBILITY",
                    channel=channel,
                    raw_value=raw_val,
                    clamped_value=clamped_val,
                    timestamp=ts,
                    description=f_dict.get("message", ""),
                )
                generated.append(dtc)

        return generated

    # ------------------------------------------------------------------------
    # Ingress Hook 2: AI / Neural Prognostics & Subsystem Degradation
    # ------------------------------------------------------------------------

    def process_neural_predictions(
        self,
        subsystem_health: Optional[Dict[str, float]] = None,
        fault_probs: Optional[Union[Dict[str, float], Sequence[float]]] = None,
        cvrde_states: Optional[Dict[str, Any]] = None,
        fault_prob_threshold: float = 0.40,
        critical_rul_threshold: float = 0.20,
    ) -> List[DTCRecord]:
        """Ingests AI / Neural Prognostic outputs:

        - `fault_probs`: Softmax distribution across 13 fault classes (dict or list/array).
        - `subsystem_health`: RUL fraction [0.0, 1.0] or Health Index [0, 100] across 8 subsystems.
        - `cvrde_states`: Optional CVRDE physical simulation states (HSU, GCS recoil, NBC).
        """
        generated: List[DTCRecord] = []
        now = time.time()

        with self._lock:
            # 1. Process 13-Class Fault Softmax Distribution
            if fault_probs is not None:
                if isinstance(fault_probs, dict):
                    prob_dict = fault_probs
                elif isinstance(fault_probs, (list, tuple)) or hasattr(fault_probs, "__iter__"):
                    class_names = [
                        "healthy",
                        "bearing_clearance_wear",
                        "bearing_wear",
                        "cooling_failure",
                        "drivetrain_efficiency_loss",
                        "exhaust_restriction",
                        "fuel_injector_fault",
                        "gear_wear",
                        "hydraulic_valve_fault",
                        "oil_pump_degradation",
                        "seal_leakage",
                        "structural_crack",
                        "torsion_fatigue",
                    ]
                    prob_dict = {
                        class_names[i]: float(p)
                        for i, p in enumerate(fault_probs)
                        if i < len(class_names)
                    }
                else:
                    prob_dict = {}

                for fault_name, prob in prob_dict.items():
                    if fault_name == "healthy":
                        continue
                    if prob >= fault_prob_threshold and fault_name in NEURAL_FAULT_CLASS_MAP:
                        mapping = NEURAL_FAULT_CLASS_MAP[fault_name]
                        desc = f"{mapping['description']} (AI Conf: {prob*100:.1f}%)"
                        dtc = self._activate_or_increment_fault(
                            spn=mapping["spn"],
                            fmi=mapping["fmi"],
                            lamp=mapping["lamp"],
                            source_type="NEURAL_PROGNOSTIC",
                            subsystem=mapping["subsystem"],
                            raw_value=prob,
                            clamped_value=None,
                            timestamp=now,
                            description=desc,
                        )
                        generated.append(dtc)

            # 2. Process Subsystem RUL Health Scores
            if subsystem_health is not None:
                for sub, health in subsystem_health.items():
                    sub_lower = sub.lower()
                    # If health is passed as [0, 100] index, convert to fraction
                    rul_frac = health / 100.0 if health > 1.0 else health
                    if rul_frac <= critical_rul_threshold and sub_lower in SUBSYSTEM_RUL_SPN_MAP:
                        sub_meta = SUBSYSTEM_RUL_SPN_MAP[sub_lower]
                        desc = f"{sub_meta['desc']} (RUL remaining: {rul_frac*100:.1f}%)"
                        dtc = self._activate_or_increment_fault(
                            spn=sub_meta["spn"],
                            fmi=sub_meta["fmi"],
                            lamp=sub_meta["lamp"],
                            source_type="NEURAL_PROGNOSTIC",
                            subsystem=sub_lower,
                            raw_value=rul_frac,
                            clamped_value=None,
                            timestamp=now,
                            description=desc,
                        )
                        generated.append(dtc)

            # 3. Process CVRDE Subsystem State Signatures
            if cvrde_states is not None:
                # CVRDE HSU Nitrogen Seal Leakage (< 140 bar)
                hsu_p = cvrde_states.get("cvrde_hsu_1_pressure_bar") or cvrde_states.get("hsu_pressure_bar")
                if hsu_p is not None and float(hsu_p) < 140.0:
                    dtc = self._activate_or_increment_fault(
                        spn=520200,
                        fmi=FMI_DATA_VALID_BELOW_NORMAL,
                        lamp="RED_STOP",
                        source_type="FDIR_PLAUSIBILITY",
                        subsystem="suspension",
                        channel="susp_load_kN",
                        raw_value=hsu_p,
                        description=f"HSU Station 1 Nitrogen Pre-Charge Pressure Loss ({hsu_p:.1f} bar < 140 bar)",
                    )
                    generated.append(dtc)

                # CVRDE GCS Main Gun Recoil Overload (> 480 kN)
                recoil_force = cvrde_states.get("cvrde_gcs_recoil_force_kn") or cvrde_states.get("recoil_force_kn")
                if recoil_force is not None and float(recoil_force) > 480.0:
                    dtc = self._activate_or_increment_fault(
                        spn=520201,
                        fmi=FMI_DATA_VALID_ABOVE_NORMAL,
                        lamp="AMBER_WARNING",
                        source_type="FDIR_PLAUSIBILITY",
                        subsystem="hydraulics",
                        channel="hyd_force",
                        raw_value=recoil_force,
                        description=f"120mm Recoil Buffer Peak Force Impulse Overload ({recoil_force:.1f} kN > 480 kN)",
                    )
                    generated.append(dtc)

                # CVRDE NBC Positive Cabin Overpressure Loss (< 200 Pa)
                nbc_p = cvrde_states.get("cvrde_nbc_overpressure_pa") or cvrde_states.get("nbc_overpressure_pa")
                if nbc_p is not None and float(nbc_p) < 200.0:
                    dtc = self._activate_or_increment_fault(
                        spn=520300,
                        fmi=FMI_DATA_VALID_BELOW_NORMAL,
                        lamp="RED_STOP",
                        source_type="FDIR_PLAUSIBILITY",
                        subsystem="auxiliary",
                        raw_value=nbc_p,
                        description=f"NBC Cabin Positive Overpressure Barrier Compromised ({nbc_p:.1f} Pa < 200 Pa)",
                    )
                    generated.append(dtc)

        return generated

    # ------------------------------------------------------------------------
    # Fault State Management (Active / Cleared / Increment)
    # ------------------------------------------------------------------------

    def _activate_or_increment_fault(
        self,
        spn: int,
        fmi: int,
        lamp: str = "AMBER_WARNING",
        source_type: str = "FDIR_PLAUSIBILITY",
        channel: str = "",
        subsystem: str = "",
        raw_value: Any = None,
        clamped_value: Any = None,
        timestamp: Optional[float] = None,
        description: str = "",
    ) -> DTCRecord:
        """Internal helper to activate a new DTC or increment occurrence on an existing DTC."""
        now = time.time() if timestamp is None else timestamp
        key = (spn, fmi)

        if key in self._active_dtcs:
            existing = self._active_dtcs[key]
            previous_last_seen = existing.last_seen
            existing.last_seen = now
            existing.raw_value = raw_value
            existing.clamped_value = clamped_value
            if description and not existing.description:
                existing.description = description
            # Increment only when the fault has been quiet for longer than the
            # re-occurrence window, measured from the last increment. Comparing
            # against first_seen made every sample after the first second count.
            gap_since_last_report = now - previous_last_seen
            if (gap_since_last_report > self.reoccurrence_window_s
                    and now - existing.last_oc_increment > self.reoccurrence_window_s):
                existing.oc = min(existing.oc + 1, 127)
                existing.last_oc_increment = now
                self.persist_to_flash(existing, event_type="OCCURRENCE_INCREMENT")
            return existing

        # If fault was previously in historic memory, reactivate it with incremented OC
        if key in self._historic_dtcs:
            historic = self._historic_dtcs.pop(key)
            historic.active = True
            historic.oc = min(historic.oc + 1, 127)
            historic.last_oc_increment = now
            historic.last_seen = now
            historic.lamp_status = lamp
            historic.raw_value = raw_value
            historic.clamped_value = clamped_value
            if description:
                historic.description = description
            self._active_dtcs[key] = historic
            self.persist_to_flash(historic, event_type="ACTIVE")
            return historic

        # Brand new DTC
        dtc = DTCRecord(
            spn=spn,
            fmi=fmi,
            oc=1,
            cm=0,
            description=description,
            lamp_status=lamp,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)),
            first_seen=now,
            last_seen=now,
            active=True,
            source_type=source_type,
            raw_value=raw_value,
            clamped_value=clamped_value,
            channel=channel,
            subsystem=subsystem,
        )
        self._active_dtcs[key] = dtc
        self.persist_to_flash(dtc, event_type="ACTIVE")
        return dtc

    def report_fault(
        self,
        spn: int,
        fmi: int,
        lamp: str = "AMBER_WARNING",
        description: str = "",
        channel: str = "",
        subsystem: str = "",
        raw_value: Any = None,
        clamped_value: Any = None,
        source_type: str = "MANUAL",
    ) -> DTCRecord:
        """Explicitly reports a fault code."""
        with self._lock:
            return self._activate_or_increment_fault(
                spn=spn,
                fmi=fmi,
                lamp=lamp,
                source_type=source_type,
                channel=channel,
                subsystem=subsystem,
                raw_value=raw_value,
                clamped_value=clamped_value,
                description=description,
            )

    def clear_fault(self, spn: int, fmi: int) -> Optional[DTCRecord]:
        """Transitions an active DTC from DM1 to DM2 (historic memory)."""
        with self._lock:
            key = (spn, fmi)
            if key in self._active_dtcs:
                dtc = self._active_dtcs.pop(key)
                dtc.active = False
                dtc.last_seen = time.time()
                self._historic_dtcs[key] = dtc
                self.persist_to_flash(dtc, event_type="CLEARED")
                return dtc
            return None

    def clear_active_dtcs(self) -> List[DTCRecord]:
        """Clears all active DTCs, moving them to historic memory (emulates DM11)."""
        cleared = []
        with self._lock:
            keys = list(self._active_dtcs.keys())
            for k in keys:
                res = self.clear_fault(k[0], k[1])
                if res:
                    cleared.append(res)
        return cleared

    def clear_historic_dtcs(self) -> List[DTCRecord]:
        """Purges all historic DTCs (emulates DM3)."""
        cleared = []
        with self._lock:
            cleared = list(self._historic_dtcs.values())
            self._historic_dtcs.clear()
        return cleared

    def get_active_dtcs(self) -> List[DTCRecord]:
        """Returns a snapshot copy of all currently active DTC records."""
        with self._lock:
            return list(self._active_dtcs.values())

    def get_historic_dtcs(self) -> List[DTCRecord]:
        """Returns a snapshot copy of all previously active (historic) DTC records."""
        with self._lock:
            return list(self._historic_dtcs.values())

    def get_all_dtcs(self) -> Dict[str, List[DTCRecord]]:
        """Returns both active and historic DTC lists."""
        with self._lock:
            return {
                "active": self.get_active_dtcs(),
                "historic": self.get_historic_dtcs(),
            }

    # ------------------------------------------------------------------------
    # SAE J1939 DM1 / DM2 Binary Encoders & Decoders
    # ------------------------------------------------------------------------

    def get_lamp_status(self) -> LampStatus:
        """Computes aggregate LampStatus from currently active DTCs."""
        with self._lock:
            if not self._active_dtcs:
                return LampStatus(
                    mil=LAMP_OFF,
                    red_stop=LAMP_OFF,
                    amber_warning=LAMP_OFF,
                    protect=LAMP_OFF,
                    mil_flash=FLASH_UNAVAILABLE,
                    red_stop_flash=FLASH_UNAVAILABLE,
                    amber_warning_flash=FLASH_UNAVAILABLE,
                    protect_flash=FLASH_UNAVAILABLE,
                )

            mil = LAMP_OFF
            red_stop = LAMP_OFF
            amber = LAMP_OFF
            protect = LAMP_OFF

            for dtc in self._active_dtcs.values():
                lamp = dtc.lamp_status.upper()
                if lamp == "RED_STOP":
                    red_stop = LAMP_ON
                elif lamp == "AMBER_WARNING" or lamp == "AMBER":
                    amber = LAMP_ON
                elif lamp == "MIL":
                    mil = LAMP_ON
                elif lamp == "PROTECT":
                    protect = LAMP_ON

            return LampStatus(
                mil=mil,
                red_stop=red_stop,
                amber_warning=amber,
                protect=protect,
            )

    def encode_dm1_packet(self) -> bytes:
        """Encodes active DTCs into standard SAE J1939-73 DM1 (PGN 65226) binary payload.

        Payload Format:
        - Bytes 1-2: 2-Byte Lamp Status Header.
        - If 0 DTCs: Header (2 bytes) + 4-byte zero DTC (0x00, 0x00, 0x00, 0x00) + 2-byte padding (0xFF, 0xFF) = 8 bytes.
        - If 1 DTC: Header (2 bytes) + 4-byte DTC + 2-byte padding (0xFF, 0xFF) = 8 bytes.
        - If N >= 2 DTCs: Header (2 bytes) + (4 * N) bytes = (2 + 4 * N) bytes.
        """
        with self._lock:
            lamp_header = self.get_lamp_status().encode_header()
            active_list = list(self._active_dtcs.values())

            if len(active_list) == 0:
                # 8-byte payload representing NO ACTIVE DTCs
                return lamp_header + struct.pack("<BBBBBB", 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF)
            elif len(active_list) == 1:
                # Single DTC standard 8-byte CAN frame
                dtc_bytes = active_list[0].encode_4bytes()
                return lamp_header + dtc_bytes + struct.pack("<BB", 0xFF, 0xFF)
            else:
                # Multi-DTC variable length BAM/TP payload
                payload = bytearray(lamp_header)
                for dtc in active_list:
                    payload.extend(dtc.encode_4bytes())
                return bytes(payload)

    # Alias for method name parity
    encode_dm1_payload = encode_dm1_packet

    def encode_dm2_packet(self) -> bytes:
        """Encodes historic DTCs into standard SAE J1939-73 DM2 (PGN 65227) binary payload."""
        with self._lock:
            # DM2 lamps are always OFF unless specific historic indicator requested
            lamp_header = LampStatus(
                mil=LAMP_OFF,
                red_stop=LAMP_OFF,
                amber_warning=LAMP_OFF,
                protect=LAMP_OFF,
            ).encode_header()

            historic_list = list(self._historic_dtcs.values())
            if len(historic_list) == 0:
                return lamp_header + struct.pack("<BBBBBB", 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF)
            elif len(historic_list) == 1:
                dtc_bytes = historic_list[0].encode_4bytes()
                return lamp_header + dtc_bytes + struct.pack("<BB", 0xFF, 0xFF)
            else:
                payload = bytearray(lamp_header)
                for dtc in historic_list:
                    payload.extend(dtc.encode_4bytes())
                return bytes(payload)

    # Alias for method name parity
    encode_dm2_payload = encode_dm2_packet

    @staticmethod
    def decode_dm1_packet(data: bytes) -> Tuple[LampStatus, List[DTCRecord]]:
        """Decodes standard SAE J1939 DM1 binary payload into LampStatus and DTCRecord list."""
        if len(data) < 2:
            raise ValueError(f"DM1 packet must be at least 2 bytes (got {len(data)})")

        lamp_status = LampStatus.decode_header(data[:2])
        records: List[DTCRecord] = []

        # Parse 4-byte DTC blocks starting at offset 2
        offset = 2
        while offset + 4 <= len(data):
            chunk = data[offset : offset + 4]
            # Check if empty zero DTC or all 0xFF padding
            if chunk == b"\x00\x00\x00\x00" or chunk == b"\xFF\xFF\xFF\xFF":
                offset += 4
                continue
            rec = DTCRecord.decode_4bytes(chunk, active=True)
            records.append(rec)
            offset += 4

        return lamp_status, records

    @staticmethod
    def decode_dm2_packet(data: bytes) -> Tuple[LampStatus, List[DTCRecord]]:
        """Decodes standard SAE J1939 DM2 binary payload into LampStatus and DTCRecord list."""
        if len(data) < 2:
            raise ValueError(f"DM2 packet must be at least 2 bytes (got {len(data)})")

        lamp_status = LampStatus.decode_header(data[:2])
        records: List[DTCRecord] = []

        offset = 2
        while offset + 4 <= len(data):
            chunk = data[offset : offset + 4]
            if chunk == b"\x00\x00\x00\x00" or chunk == b"\xFF\xFF\xFF\xFF":
                offset += 4
                continue
            rec = DTCRecord.decode_4bytes(chunk, active=False)
            records.append(rec)
            offset += 4

        return lamp_status, records

    # ------------------------------------------------------------------------
    # Thread-Safe On-Disk Circular Flash Ring Buffer Persistence
    # ------------------------------------------------------------------------

    def persist_to_flash(self, record: DTCRecord, event_type: str = "ACTIVE") -> None:
        """Appends a DTC state transition record to `results/dtc_flash_log.jsonl`.

        Enforces circular rollover with maximum capacity `self.flash_capacity`
        to prevent flash storage exhaustion on embedded military hardware.
        """
        with self._lock:
            entry = {
                "timestamp": record.timestamp or time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "utc_float": record.last_seen,
                "event_type": event_type,
                "vehicle_id": self.vehicle_id,
                "spn": record.spn,
                "fmi": record.fmi,
                "oc": record.oc,
                "cm": record.cm,
                "code": record.formatted_code,
                "lamp_status": record.lamp_status,
                "description": record.description,
                "source_type": record.source_type,
                "raw_value": record.raw_value,
                "clamped_value": record.clamped_value,
                "channel": record.channel,
                "subsystem": record.subsystem,
                "hex_4bytes": record.encode_4bytes().hex().upper(),
            }

            try:
                # Append record line
                with open(self.flash_file, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
                self._flash_line_count += 1

                # Only read the file when the O(1) counter says we're over capacity.
                if self._flash_line_count > self.flash_capacity:
                    self._check_and_rollover_flash()
            except Exception as exc:
                logger.error(f"Failed to persist DTC record to flash: {exc}")

    def _check_and_rollover_flash(self) -> None:
        """Prunes oldest lines if flash log exceeds `self.flash_capacity`."""
        if not self.flash_file.exists():
            return

        try:
            with open(self.flash_file, "r", encoding="utf-8") as fh:
                lines = fh.readlines()

            if len(lines) > self.flash_capacity:
                # Keep latest (capacity - 100) records to minimize frequent rewrites
                prune_to = max(50, self.flash_capacity - 100)
                retained = lines[-prune_to:]
                tmp_file = self.flash_file.with_suffix(".tmp")
                with open(tmp_file, "w", encoding="utf-8") as fh:
                    fh.writelines(retained)
                tmp_file.replace(self.flash_file)
                self._flash_line_count = len(retained)
        except Exception as exc:
            logger.warning(f"Flash ring buffer rollover warning: {exc}")


    def read_flash_log(self, limit: int = 100, reverse: bool = True) -> List[Dict[str, Any]]:
        """Reads recent maintenance records from the flash ring buffer."""
        with self._lock:
            if not self.flash_file.exists():
                return []
            try:
                with open(self.flash_file, "r", encoding="utf-8") as fh:
                    lines = [line.strip() for line in fh if line.strip()]

                records = []
                for line in lines:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue

                if reverse:
                    records.reverse()

                return records[:limit]
            except Exception as exc:
                logger.error(f"Failed reading flash log: {exc}")
                return []

    def clear_flash_log(self) -> None:
        """Clears the on-disk flash ring buffer log."""
        with self._lock:
            try:
                if self.flash_file.exists():
                    self.flash_file.unlink()
            except Exception as exc:
                logger.error(f"Failed to clear flash log: {exc}")

    def recover_from_flash(self, max_age_hours: float = 72.0) -> None:
        """Recovers active and historic DTC state from the on-disk flash log.

        ``max_age_hours`` gates how stale an ACTIVE record can be before it
        is demoted to historic on recovery.  Faults older than this have almost
        certainly been investigated or cleared; re-activating them silently is
        worse than promoting them to historic for audit.
        """
        with self._lock:
            if not self.flash_file.exists():
                return

            stale_cutoff = time.time() - max_age_hours * 3600.0

            try:
                with open(self.flash_file, "r", encoding="utf-8") as fh:
                    lines = [line.strip() for line in fh if line.strip()]

                for line in lines:
                    try:
                        entry = json.loads(line)
                        spn = entry.get("spn")
                        fmi = entry.get("fmi")
                        if spn is None or fmi is None:
                            continue

                        key = (int(spn), int(fmi))
                        event_type = entry.get("event_type", "ACTIVE")
                        dtc = DTCRecord.from_dict(entry)

                        if event_type == "ACTIVE" or event_type == "OCCURRENCE_INCREMENT":
                            # Demote stale ACTIVE records rather than re-activating them.
                            # Missing/null utc_float is treated as now (recent) — not stale.
                            raw_ts = entry.get("utc_float")
                            last_seen = float(raw_ts) if raw_ts is not None else time.time()
                            if last_seen < stale_cutoff:
                                dtc.active = False
                                self._historic_dtcs[key] = dtc
                                continue
                            dtc.active = True
                            self._active_dtcs[key] = dtc
                            if key in self._historic_dtcs:
                                del self._historic_dtcs[key]
                        elif event_type == "CLEARED" or event_type == "MANUAL_CLEAR":
                            dtc.active = False
                            self._historic_dtcs[key] = dtc
                            if key in self._active_dtcs:
                                del self._active_dtcs[key]
                    except Exception:
                        continue
            except Exception as exc:
                logger.warning(f"Error during flash recovery: {exc}")

    def reset(self) -> None:
        """Resets all in-memory DTC state and clears active/historic registries."""
        with self._lock:
            self._active_dtcs.clear()
            self._historic_dtcs.clear()
