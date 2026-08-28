"""Hydraulics sensor data generator (turret / stabilizer circuit).

The hydraulic circuit drives the gun-control elevation/traverse
actuators and the recoil buffer.  Two channels reach the LSTM:

* ``hyd_pressure`` -- regulated circuit pressure (21 MPa nominal).
* ``hyd_leak_flow`` -- return-line flow that is not the commanded
  flow; a rising trend means seal degradation.

``hyd_force`` is display-only (readback of the joystick command, not a
measurement) and is flagged ``health_exclude=True`` in the schema but
still written here so the dashboard card has a value.
"""

from __future__ import annotations

from typing import Sequence

from ._base import (ChannelSpec, FaultProfile, GeneratorSpec,
                    run_subsystem)


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("time", "s", healthy=0.0, description="mission elapsed time"),
    ChannelSpec("step", "count", healthy=0.0, description="integer time step"),
    ChannelSpec("hyd_pressure", "bar", healthy=210.0, scale=1e-5,
                warn_lo=140.0, crit_lo=90.0,
                description="regulated circuit pressure (Pa stored, bar published)"),
    ChannelSpec("hyd_flow", "m^3/s", healthy=1.2e-3,
                description="commanded flow through the servo valve"),
    ChannelSpec("hyd_force", "kN", healthy=4.2, scale=1e-3,
                warn_lo=2.9, crit_lo=2.0,
                description="actuator force (display only, not an LSTM input)"),
    ChannelSpec("hyd_leak_flow", "L/s", healthy=0.0087, scale=1000.0,
                warn_hi=0.025, crit_hi=0.045,
                description="seal leak flow (m^3/s stored, L/s published)"),
    ChannelSpec("hyd_power", "W", healthy=250.0, warn_hi=350.0, crit_hi=420.0,
                description="hydraulic power dissipation"),
)


FAULTS: tuple[FaultProfile, ...] = (
    FaultProfile("seal_leakage",
                 description="hyd_seal_leak rises; hyd_pressure drops under load"),
    FaultProfile("hydraulic_valve_fault",
                 description="valve_fault adds hysteresis; hyd_pressure undershoots"),
)


def generate(out_path: str, steps: int = 2000, seed: int = 42,
             faults: Sequence[str] | None = None, dt: float = 0.05) -> int:
    """Run the hydraulics simulation."""
    return run_subsystem(SPEC, out_path, steps=steps, seed=seed,
                         faults=tuple(faults or ()), dt=dt)


SPEC = GeneratorSpec(
    name="hydraulics",
    part_key="hydraulics",
    label="Hydraulics (Turret / Stabilizer Circuit)",
    description="21 MPa regulated circuit; seal-leak and valve-fault discriminator.",
    channels=CHANNELS,
    faults=FAULTS,
    sample_rate_hz=20.0,
    generate=generate,
    notes=(
        "hyd_force is published for display only; the LSTM input schema "
        "flags it ``health_exclude=True`` in ml.parts.PARTS[hydraulics].",
    ),
)
