"""Unit contract between the simulator/native telemetry and the gateway.

The problem this solves
-----------------------
There were two unit conventions in the system and no contract between them:

* `tank_sim` is SI-native. `oil_pressure` is Pa (5.3e5), `hyd_pressure` is Pa
  (2.1e7), levels are fractions (0.89).
* `sensor_plausibility.SENSOR_LIMITS_CATALOG` and the J1939 SPN encodings are
  engineering-native. `oil_pressure` is bar with envelope [0, 15],
  `hyd_pressure` is bar with envelope [0, 400], levels are percent [0, 100].

Nothing converted between them, and the two failure modes were different:

1. **Loud** -- the three pressure channels blew straight through the FDIR
   envelope. Every frame raised a spurious SHORT_CIRCUIT and got clamped:
   oil_pressure 527705 Pa -> 15.0 bar, hyd_pressure 2.1e7 Pa -> 400 bar. That
   drove lubrication and hydraulics health to 0.0 on a healthy mission.
2. **Silent, and worse** -- `fuel_level` 0.89 (a fraction) sits *inside* the
   [0, 100] percent envelope, so no fault fired at all. It was simply read as
   0.89 %, i.e. a nearly-empty tank, all the way through health scoring and
   DTC generation. A conversion error that trips a range check is a bug; one
   that doesn't is a wrong answer delivered with confidence.

`server.py` also did `encode_efl_p1(oil_p * 100.0)` assuming bar, so a
simulator frame (Pa) saturated SPN 100 at its 1000 kPa ceiling.

The contract
------------
`to_canonical` converts a native/SI frame into the engineering units the FDIR
catalog, the J1939 encoders and the HUD all declare. `to_native` is its exact
inverse, for the health scorer in `ml/parts.py`, whose thresholds carry their
own `scale` factors and therefore expect the SI form.

`tests/test_units.py` asserts round-trip exactness and, more importantly, that
every converted channel lands inside the catalog envelope it is destined for --
so the next channel added with the wrong unit fails a test instead of quietly
reporting an empty fuel tank.
"""

from __future__ import annotations

from typing import Any, Dict

# channel -> multiplicative factor applied going native (SI) -> canonical.
# Only channels whose two conventions actually differ appear here; anything
# absent is passed through unchanged and is asserted consistent by the test.
NATIVE_TO_CANONICAL: Dict[str, float] = {
    # Pressures: simulator Pa -> catalog/J1939 bar.
    "oil_pressure": 1e-5,
    "hyd_pressure": 1e-5,
    "exhaust_pressure": 1e-5,
    "gcs_hyd_pressure": 1e-5,
    "boost_pressure": 1e-5,
    "rail_pressure": 1e-5,
    # Levels: simulator fraction -> catalog/HUD percent.
    "fuel_level": 100.0,
    "oil_level": 100.0,
    "coolant_level": 100.0,
}

# Units each canonical channel is expressed in, for documentation and for the
# test that cross-checks them against SENSOR_LIMITS_CATALOG[...].unit.
CANONICAL_UNITS: Dict[str, str] = {
    "oil_pressure": "bar",
    "hyd_pressure": "bar",
    "exhaust_pressure": "bar",
    "gcs_hyd_pressure": "bar",
    "boost_pressure": "bar",
    "rail_pressure": "bar",
    "fuel_level": "%",
    "oil_level": "%",
    "coolant_level": "%",
}


def to_canonical(frame: Dict[str, Any]) -> Dict[str, Any]:
    """Native/SI telemetry -> engineering units (FDIR catalog, J1939, HUD)."""
    out = dict(frame)
    for key, factor in NATIVE_TO_CANONICAL.items():
        v = out.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v) * factor
    return out


def to_native(frame: Dict[str, Any]) -> Dict[str, Any]:
    """Exact inverse of :func:`to_canonical`."""
    out = dict(frame)
    for key, factor in NATIVE_TO_CANONICAL.items():
        v = out.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v) / factor
    return out


def looks_canonical(frame: Dict[str, Any]) -> bool:
    """Heuristic: is this frame already in engineering units?

    Used so a client that already speaks bar/percent is not converted twice.
    Pressure is the reliable discriminator -- 5.3e5 is unambiguously Pa and
    5.3 is unambiguously bar, with four orders of magnitude between them.
    """
    p = frame.get("oil_pressure")
    if isinstance(p, (int, float)) and not isinstance(p, bool):
        return abs(float(p)) < 1000.0
    p = frame.get("hyd_pressure")
    if isinstance(p, (int, float)) and not isinstance(p, bool):
        return abs(float(p)) < 10000.0
    return True


__all__ = ["NATIVE_TO_CANONICAL", "CANONICAL_UNITS", "to_canonical",
           "to_native", "looks_canonical"]
