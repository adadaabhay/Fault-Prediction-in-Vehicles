"""Fluid-level (fuel / oil / coolant) sensor data generator.

Capacitive level probes measure dielectric-permittivity changes; the
published channels are:

* ``fuel_level`` -- 0-1 fraction in the main tank.
* ``fuel_capacitance_pf`` -- probe capacitance.

All channels in this generator are ``health_exclude=True`` in
``ml.parts.PARTS``: a fault-free mission still drains fuel, so
including them in the health index would decay the score towards
failure on a healthy run.
"""

from __future__ import annotations

from typing import Sequence

from ._base import (ChannelSpec, FaultProfile, GeneratorSpec,
                    run_subsystem)


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("time", "s", healthy=0.0, description="mission elapsed time"),
    ChannelSpec("step", "count", healthy=0.0, description="integer time step"),
    ChannelSpec("fuel_level", "frac", healthy=0.95, warn_lo=0.20, crit_lo=0.10,
                description="fuel level in main tank"),
    ChannelSpec("fuel_capacitance_pf", "pF", healthy=420.0, warn_lo=80.0, crit_lo=40.0,
                description="probe capacitance (display only)"),
)


FAULTS: tuple[FaultProfile, ...] = (
    FaultProfile("seal_leakage",
                 description="fuel_leak, oil_leak, coolant_leak all rise"),
)


def generate(out_path: str, steps: int = 2000, seed: int = 42,
             faults: Sequence[str] | None = None, dt: float = 0.05) -> int:
    """Run the fluid-level simulation."""
    return run_subsystem(SPEC, out_path, steps=steps, seed=seed,
                         faults=tuple(faults or ()), dt=dt)


SPEC = GeneratorSpec(
    name="fuel_levels",
    part_key="overall",  # maps to the dashboard "Overall" card
    label="Fluid Levels (Fuel / Oil / Coolant)",
    description=(
        "Capacitive level probes.  All channels are consumables "
        "(health_exclude=True) -- a healthy mission drains fuel."
    ),
    channels=CHANNELS,
    faults=FAULTS,
    sample_rate_hz=20.0,
    generate=generate,
)
