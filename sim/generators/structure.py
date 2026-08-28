"""Structure / Torsion sensor data generator (hull + gun mount).

The structure channel is the discriminant for fatigue damage in the
hull torsion bars and the gun mount.  The four LSTM input channels
are:

* ``torsion_twist_deg`` -- instantaneous bar twist.
* ``torsion_cumulative_twist`` -- lifetime fatigue accumulator (it
  grows monotonically with mission time, so it is flagged
  ``health_exclude=True`` in the schema to keep it out of the health
  index, but the LSTM still sees it as a feature).
* ``ae_event_rate`` -- acoustic emission hit rate, the load-bearing
  discriminant for crack initiation.
* ``ae_energy`` -- burst energy, the load-bearing discriminant for
  crack propagation.
"""

from __future__ import annotations

from typing import Sequence

from ._base import (ChannelSpec, FaultProfile, GeneratorSpec,
                    run_subsystem)


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("time", "s", healthy=0.0, description="mission elapsed time"),
    ChannelSpec("step", "count", healthy=0.0, description="integer time step"),
    ChannelSpec("terrain", "frac", healthy=0.4, warn_hi=0.8, crit_hi=0.95,
                description="normalized terrain roughness"),
    ChannelSpec("torsion_twist_deg", "deg", healthy=1.8, warn_hi=5.0, crit_hi=6.0,
                description="instantaneous torsion bar twist"),
    ChannelSpec("torsion_cumulative_twist", "rad", healthy=300.0,
                warn_hi=1200.0, crit_hi=2000.0,
                description="lifetime fatigue accumulator (LSTM input, health_exclude)"),
    ChannelSpec("torsion_shear_MPa", "MPa", healthy=320.0, warn_hi=480.0,
                crit_hi=560.0,
                description="bar shear stress"),
    ChannelSpec("ae_event_rate", "/s", healthy=2.0, warn_hi=7.0, crit_hi=12.0,
                description="acoustic emission hit rate"),
    ChannelSpec("ae_energy", "-", healthy=0.5, warn_hi=8.0, crit_hi=15.0,
                description="acoustic emission burst energy"),
)


FAULTS: tuple[FaultProfile, ...] = (
    FaultProfile("torsion_fatigue",
                 description="stiffness_mult drops; AE event rate rises"),
    FaultProfile("structural_crack",
                 description="ae_severity + fatigue_factor; AE energy spikes"),
)


def generate(out_path: str, steps: int = 2000, seed: int = 42,
             faults: Sequence[str] | None = None, dt: float = 0.05) -> int:
    """Run the structure / torsion-bar simulation."""
    return run_subsystem(SPEC, out_path, steps=steps, seed=seed,
                         faults=tuple(faults or ()), dt=dt)


SPEC = GeneratorSpec(
    name="structure",
    part_key="structure",
    label="Structure / Torsion / Hull",
    description="Hull torsion bars + gun-mount structure; AE event/energy channels.",
    channels=CHANNELS,
    faults=FAULTS,
    sample_rate_hz=20.0,
    generate=generate,
    notes=(
        "torsion_cumulative_twist is health_exclude=True: it grows "
        "monotonically and would dominate the health index if treated as "
        "a fault (see ml.parts.PARTS[structure]).",
    ),
)
