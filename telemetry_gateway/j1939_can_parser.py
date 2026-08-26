"""SAE J1939 CAN & Vectronics Frame Parser.
Encodes and decodes standard Parameter Group Numbers (PGNs) and Suspect Parameter Numbers (SPNs)
matching heavy vehicle and military vectronics architectures (STANAG 4754 / J1939-71).
"""

from __future__ import annotations

import struct
from typing import Dict, Any


class J1939FrameParser:
    # Standard PGN Definitions
    PGN_EEC1 = 61444     # 0xF004: Electronic Engine Controller 1 (RPM, Torque)
    PGN_ET1 = 65262      # 0xFEEE: Engine Temperature 1 (SPN 110 coolant temp)
    PGN_EFL_P1 = 65263   # 0xFEEF: Engine Fluid Level/Pressure 1 (SPN 100 oil pressure)
    PGN_TF = 65272       # 0xFEF8: Transmission Fluids (Transmission Oil Temp)
    PGN_TURRET_HYD = 61184 # 0xEF00: Proprietary A (Turret Pressure, Recoil Accumulator)

    @staticmethod
    def encode_eec1(rpm: float, torque_pct: float = 50.0) -> bytes:
        """Encodes PGN 61444 (EEC1) 8-byte CAN payload.
        Layout per SAE J1939-71:
          byte 1  Engine Torque Mode (SPN 899)
          byte 2  Driver's Demand Engine Percent Torque (SPN 512), 1%/bit, -125 offset
          byte 3  Actual Engine Percent Torque (SPN 513),          1%/bit, -125 offset
          bytes 4-5  Engine Speed (SPN 190), 0.125 rpm/bit, LSB first
          byte 6  Source Address of Controlling Device (SPN 1483)
          byte 7  Engine Starter Mode (SPN 1675)
          byte 8  Engine Demand Percent Torque (SPN 2432)

        The previous docstring attributed bytes 2-3 to SPN 898. SPN 898 is
        "Engine Requested Torque / Torque Limit" and does not live in EEC1;
        the byte positions themselves were correct.
        """
        raw_torque = int(min(max(torque_pct + 125, 0), 250))
        raw_rpm = int(min(max(rpm / 0.125, 0), 64255))
        
        # 8 bytes: [Engine Torque mode, Driver Demand Torque, Actual Torque, RPM_L, RPM_H, Source Addr, Starter State, Reserved]
        return struct.pack("<BBBHBBB", 0x00, raw_torque, raw_torque, raw_rpm, 0x00, 0xFF, 0xFF)

    @staticmethod
    def decode_eec1(data: bytes) -> Dict[str, float]:
        """Decodes PGN 61444 (EEC1) 8-byte CAN payload."""
        if len(data) < 8:
            raise ValueError("J1939 EEC1 frame must be at least 8 bytes.")
        _, raw_demand_tq, raw_tq, raw_rpm, _, _, _ = struct.unpack("<BBBHBBB", data[:8])
        return {
            "rpm": raw_rpm * 0.125,
            "torque_pct": raw_tq - 125.0,
        }

    @staticmethod
    def encode_et1(coolant_temp_c: float, fuel_temp_c: float = -40.0,
                   oil_temp_c: float | None = None) -> bytes:
        """Encodes PGN 65262 (ET1, Engine Temperature 1) 8-byte CAN payload.

        Layout per SAE J1939-71:
          byte 1     Engine Coolant Temperature (SPN 110), 1 degC/bit, -40 offset
          byte 2     Engine Fuel Temperature 1  (SPN 174), 1 degC/bit, -40 offset
          bytes 3-4  Engine Oil Temperature 1   (SPN 175), 0.03125 degC/bit, -273 offset
          bytes 5-6  Engine Turbo Oil Temp      (SPN 176)
          byte 7     Engine Intercooler Temp    (SPN 52)
          byte 8     Intercooler Thermostat Opening (SPN 1134)

        SPN 110 belongs HERE, not in EFL_P1. It was previously written into
        byte 1 of PGN 65263, which is SPN 94 (Fuel Delivery Pressure) -- any
        conforming J1939 stack decoded the coolant temperature as a fuel
        pressure in kPa.
        """
        raw_coolant_t = int(min(max(coolant_temp_c + 40.0, 0), 250))
        raw_fuel_t = int(min(max(fuel_temp_c + 40.0, 0), 250))
        if oil_temp_c is None:
            raw_oil_t = 0xFFFF                      # not available
        else:
            raw_oil_t = int(min(max((oil_temp_c + 273.0) / 0.03125, 0), 64255))
        return struct.pack("<BBHHBB", raw_coolant_t, raw_fuel_t, raw_oil_t,
                           0xFFFF, 0xFF, 0xFF)

    @staticmethod
    def decode_et1(data: bytes) -> Dict[str, float]:
        """Decodes PGN 65262 (ET1) 8-byte CAN payload."""
        if len(data) < 8:
            raise ValueError("J1939 ET1 frame must be at least 8 bytes.")
        raw_coolant_t, raw_fuel_t, raw_oil_t, _, _, _ = struct.unpack(
            "<BBHHBB", data[:8])
        out = {
            "coolant_temp": float(raw_coolant_t - 40),
            "fuel_temp": float(raw_fuel_t - 40),
        }
        if raw_oil_t != 0xFFFF:
            out["oil_temp"] = float(raw_oil_t * 0.03125 - 273.0)
        return out

    @staticmethod
    def encode_efl_p1(oil_pressure_kpa: float,
                      fuel_delivery_pressure_kpa: float = 0.0,
                      oil_level_pct: float | None = None) -> bytes:
        """Encodes PGN 65263 (EFL_P1) 8-byte CAN payload.

        Layout per SAE J1939-71:
          byte 1     Fuel Delivery Pressure  (SPN 94),  4 kPa/bit
          byte 2     Extended Crankcase Blow-by Pressure (SPN 22)
          byte 3     Engine Oil Level        (SPN 98),  0.4 %/bit
          byte 4     Engine Oil Pressure     (SPN 100), 4 kPa/bit
          bytes 5-6  Crankcase Pressure      (SPN 101)
          bytes 7-8  Engine Coolant Pressure (SPN 109)

        There is no coolant *temperature* in this PGN -- see `encode_et1`.
        """
        raw_oil_p = int(min(max(oil_pressure_kpa / 4.0, 0), 250))
        raw_fuel_p = int(min(max(fuel_delivery_pressure_kpa / 4.0, 0), 250))
        raw_oil_lvl = (0xFF if oil_level_pct is None
                       else int(min(max(oil_level_pct / 0.4, 0), 250)))
        return struct.pack("<BBBBHH", raw_fuel_p, 0xFF, raw_oil_lvl, raw_oil_p,
                           0xFFFF, 0xFFFF)

    @staticmethod
    def decode_efl_p1(data: bytes) -> Dict[str, float]:
        """Decodes PGN 65263 (EFL_P1) 8-byte CAN payload."""
        if len(data) < 8:
            raise ValueError("J1939 EFL_P1 frame must be at least 8 bytes.")
        raw_fuel_p, _, raw_oil_lvl, raw_oil_p, _, _ = struct.unpack(
            "<BBBBHH", data[:8])
        out = {
            "fuel_delivery_pressure_kpa": float(raw_fuel_p * 4.0),
            "oil_pressure_kpa": float(raw_oil_p * 4.0),
        }
        if raw_oil_lvl != 0xFF:
            out["oil_level_pct"] = float(raw_oil_lvl * 0.4)
        return out
