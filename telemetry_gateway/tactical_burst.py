"""Tactical Radio Low-Bandwidth Burst Protocol (EMCON Mode).

Compresses full vehicle health state into a 32-byte binary burst suitable for
9.6 kbps VHF/SDR tactical data links, framed with a 16-bit CRC.

SECURITY NOTICE -- this transport is *unencrypted and unauthenticated*.
This docstring previously described the burst as "encrypted".  It is not:
the payload is plaintext ``struct.pack`` and CRC-16-CCITT is an
error-detecting code, not a cipher and not a MAC.  A CRC is trivially
recomputed by anyone who alters the payload, so it provides no
confidentiality and no authenticity whatsoever.

Any operational use over a contested link must layer this inside a transport
that provides both (e.g. an approved COMSEC device or an AEAD such as
AES-GCM with a per-vehicle key).  Do not rely on the CRC for anything beyond
detecting accidental bit errors.
"""

import struct
from typing import Dict, Any, List


class TacticalBurstPacket:
    HEADER = b"TK"  # 2 bytes Sync Header (0x54 0x4B)
    PACKET_SIZE = 32

    @staticmethod
    def compute_crc16(data: bytes) -> int:
        """CRC-16-CCITT (Polynomial 0x1021, Init 0xFFFF).

        Integrity only. Not a message authentication code -- an attacker who
        modifies the payload simply recomputes this.
        """
        crc = 0xFFFF
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ 0x1021) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc

    @classmethod
    def encode(cls, tank_id: int, mission_time: int, chi: float,
               top_fault_id: int, fault_confidence: float,
               rul_minutes: int, subsystem_health: List[float],
               rpm: float, oil_pressure_bar: float, coolant_temp_c: float,
               vib_rms: float) -> bytes:
        """Encodes vehicle health into a 32-byte tactical packet."""
        # Ensure 8 subsystem health bytes
        sub_bytes = [int(min(max(h, 0), 100)) for h in subsystem_health[:8]]
        while len(sub_bytes) < 8:
            sub_bytes.append(100)

        # Packed payload (30 bytes before CRC: 2+1+4+1+1+1+2+8+1+1+1+1+4+2 = 30)
        payload = struct.pack(
            ">2s B I B B B H 8B B B B B 4s H",
            cls.HEADER,
            tank_id & 0xFF,
            mission_time & 0xFFFFFFFF,
            int(min(max(chi, 0), 100)),
            top_fault_id & 0xFF,
            int(min(max(fault_confidence * 100, 0), 100)),
            min(max(rul_minutes, 0), 65535),
            *sub_bytes,
            int(min(max(rpm / 20.0, 0), 255)),            # RPM scaled by 20 (0-5100 RPM)
            int(min(max(oil_pressure_bar * 20.0, 0), 255)),# Oil P scaled (0-12.75 bar)
            int(min(max(coolant_temp_c + 40.0, 0), 255)),  # Coolant T (-40 to 215 C)
            int(min(max(vib_rms * 50.0, 0), 255)),         # Vib RMS (0-5.1 g)
            b"\x00\x00\x00\x00",                          # Reserved 4 bytes
            0x0000                                         # Status word 2 bytes
        )

        # Append 16-bit CRC
        crc = cls.compute_crc16(payload)
        return payload + struct.pack(">H", crc)

    @classmethod
    def decode(cls, packet: bytes) -> Dict[str, Any]:
        """Decodes 32-byte tactical packet and validates CRC."""
        if len(packet) != cls.PACKET_SIZE:
            raise ValueError(f"Packet size must be exactly {cls.PACKET_SIZE} bytes (got {len(packet)}).")

        payload = packet[:30]
        expected_crc = struct.unpack(">H", packet[30:32])[0]
        actual_crc = cls.compute_crc16(payload)

        if actual_crc != expected_crc:
            raise ValueError(f"CRC-16 mismatch: expected {hex(expected_crc)}, got {hex(actual_crc)}")

        unpacked = struct.unpack(">2s B I B B B H 8B B B B B 4s H", payload)
        
        return {
            "header": unpacked[0].decode("ascii", errors="ignore"),
            "tank_id": unpacked[1],
            "mission_time_sec": unpacked[2],
            "composite_chi": float(unpacked[3]),
            "top_fault_id": unpacked[4],
            "fault_confidence_pct": float(unpacked[5]),
            "rul_minutes": unpacked[6],
            "subsystem_health": list(unpacked[7:15]),
            "rpm": float(unpacked[15] * 20.0),
            "oil_pressure_bar": float(unpacked[16] / 20.0),
            "coolant_temp_c": float(unpacked[17] - 40.0),
            "vib_rms": float(unpacked[18] / 50.0),
            "crc_valid": True,
        }
