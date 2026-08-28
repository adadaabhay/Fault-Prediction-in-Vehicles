"""Gun Control System (GCS) -- display-only generator.

The GCS drives the 120 mm main armament's elevation / traverse and the
recoil buffer.  The simulator does not currently model GCS dynamics
(no CVRDE gun-control channel in the raw telemetry), so this generator
publishes only the mission timing columns.  The dashboard renders the
GCS card against the ``hyd_pressure`` channel from the hydraulics
generator, which is the upstream of the 21 MPa circuit that drives the
servo valves.

When a fielded GCS physics model is added to ``sim.tank`` (or
``sim.cvrde.gun_control`` is wired in), add the
``gcs_elevation``/``gcs_los_error``/``recoil_peak_force``/
``recoil_stroke_mm``/``barrel_efc`` channels here and they will flow
through the existing pipeline without further changes.
"""

from __future__ import annotations

from typing import Sequence

from ._base import (ChannelSpec, FaultProfile, GeneratorSpec,
                    run_subsystem)


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("time", "s", healthy=0.0, description="mission elapsed time"),
    ChannelSpec("step", "count", healthy=0.0, description="integer time step"),
)


# Placeholder fault profiles for future GCS physics; the master CLI
# will accept these names without error once the underlying model
# exists, but they are no-ops today (no matching key in the shared
# FaultManager FAULT_MAP).
FAULTS: tuple[FaultProfile, ...] = (
    FaultProfile("hydraulic_valve_fault",
                 description="servo hysteresis; los_error widens on the move"),
    FaultProfile("torsion_fatigue",
                 description="gun mount stiffness loss; los_error bias grows"),
)


def generate(out_path: str, steps: int = 2000, seed: int = 42,
             faults: Sequence[str] | None = None, dt: float = 0.05) -> int:
    """Run the simulator; the GCS card is fed by hydraulics.hyd_pressure."""
    return run_subsystem(SPEC, out_path, steps=steps, seed=seed,
                         faults=tuple(faults or ()), dt=dt)


SPEC = GeneratorSpec(
    name="gun_control",
    part_key="hydraulics",  # the GCS is fed by the hydraulics subsystem
    label="Gun Control System (Elevation / Traverse / Recoil)",
    description=(
        "GCS hydraulics, recoil buffer, and barrel EFC tracking.  The "
        "hydraulic pressure and leak-flow channels live in the "
        "hydraulics generator; the GCS-specific channels (elevation, "
        "LOS error, recoil stroke, barrel EFC) are planned but not yet "
        "wired into sim.tank."
    ),
    channels=CHANNELS,
    faults=FAULTS,
    sample_rate_hz=20.0,
    generate=generate,
    notes=(
        "Mapped to ml.parts.PARTS[hydraulics] in the dashboard (the GCS "
        "shares the 21 MPa circuit); the barrel EFC accumulator will live "
        "in the structure card once modelled.",
    ),
)
