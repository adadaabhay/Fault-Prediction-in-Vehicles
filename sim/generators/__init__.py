"""Per-subsystem simulated sensor data generators.

The package exposes a uniform interface for producing deployment-grade
CSV datasets for every subsystem listed in :data:`ml.parts.PARTS`.
Each module defines a :class:`GeneratorSpec` describing its output
schema (channels, units, fault profiles, sample rate) and a thin
``generate()`` wrapper that defers to :func:`sim.generators._base.run_subsystem`.

Why a per-subsystem package and not one file with a ``switch``?
    * Each subsystem has a distinct channel set, scaling, and fault
      profile -- so the column manifest, units, and threshold blocks
      differ enough that a "one generator" god-object would obscure
      the data contract.
    * Per-subsystem modules are independently importable, so a
      downstream user can ``from sim.generators.engine import SPEC``
      without paying the import cost of every other subsystem's
      physics module.

The shared protocol -- :class:`GeneratorSpec` with a
``generate(out_path, *, steps, seed, faults, dt)`` callable -- is
what the master CLI introspects.  Add a new subsystem by creating a
new module with those two exports and registering it in
:data:`sim.generators.SUBSYSTEMS` below.
"""

from __future__ import annotations

from . import (  # noqa: F401 -- re-exports for `from sim.generators import <name>`
    acoustics,
    engine,
    exhaust,
    fuel_levels,
    gun_control,
    hydraulics,
    nbc,
    powertrain,
    structure,
    suspension,
)
from ._base import GeneratorSpec, run_subsystem, sha256_file, write_manifest

# Authoritative registry.  Order is preserved (Python 3.7+ dict is
# ordered) so the CLI's help text is deterministic.  ``part_key`` is
# the key in :data:`ml.parts.PARTS` whose channels this generator
# covers; the master CLI uses it to validate the dataset against the
# LSTM input schema.
#
# Note on aliases: ``lubrication`` and ``cooling`` are exposed as
# channel views of the engine generator (their sensors are co-located
# on the powerpack).  Each call invokes the simulator independently,
# so two CSV writes follow -- the duplicates are bit-identical given
# the same seed.
SUBSYSTEMS: dict[str, GeneratorSpec] = {
    s.name: s for s in (
        engine.SPEC,
        powertrain.SPEC,
        hydraulics.SPEC,
        suspension.SPEC,
        structure.SPEC,
        gun_control.SPEC,
        nbc.SPEC,
        exhaust.SPEC,
        acoustics.SPEC,
        fuel_levels.SPEC,
    )
}

# Subsystems that share a generator but present a different *channel
# view* of the same simulation run.  These resolve at CLI dispatch
# time -- the user-facing name points at the real generator above.
SUBSYSTEM_ALIASES: dict[str, str] = {
    "lubrication": "engine",
    "cooling": "engine",
}

ALL_SUBSYSTEMS: tuple[str, ...] = tuple(
    list(SUBSYSTEMS) + list(SUBSYSTEM_ALIASES)
)


def _validate_registry() -> None:
    """Startup check: every spec is well-formed.

    Catches the bug class "declared a channel the simulator does not
    produce" at import time instead of leaving silently empty cells in
    the CSV.  A real channel-mismatch check would require a sim run
    (expensive); we settle for the cheap checks: unique names, unique
    channel keys per spec, and unique part_keys (allowing the
    intentional ``gun_control -> hydraulics`` mapping as a documented
    exception).
    """
    seen_names: set[str] = set()
    seen_part_keys: set[str] = set()
    for spec in SUBSYSTEMS.values():
        if spec.name in seen_names:
            raise RuntimeError(
                f"duplicate subsystem name: {spec.name!r}")
        seen_names.add(spec.name)
        # ``gun_control`` deliberately shares a part_key with
        # ``hydraulics`` (the GCS is fed by the 21 MPa circuit).  Allow
        # exactly one extra collision, flagged in the docstring.
        if spec.part_key in seen_part_keys and spec.name != "gun_control":
            raise RuntimeError(
                f"duplicate part_key {spec.part_key!r} for "
                f"subsystem {spec.name!r}")
        seen_part_keys.add(spec.part_key)
        keys = [c.key for c in spec.channels]
        if len(set(keys)) != len(keys):
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            raise RuntimeError(
                f"duplicate channel keys in {spec.name!r}: {dupes}")


_validate_registry()


__all__ = [
    "GeneratorSpec",
    "SUBSYSTEMS",
    "SUBSYSTEM_ALIASES",
    "ALL_SUBSYSTEMS",
    "run_subsystem",
    "sha256_file",
    "write_manifest",
    "engine",
    "powertrain",
    "hydraulics",
    "suspension",
    "structure",
    "gun_control",
    "nbc",
    "exhaust",
    "acoustics",
    "fuel_levels",
]
