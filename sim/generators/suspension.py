"""Suspension sensor data generator (CVRDE hydrogas + road wheels).

The CVRDE tank rides on 14 hydrogas units (7 per side) -- a hydro-
pneumatic suspension with nitrogen pre-charge.  The four LSTM input
channels are:

* ``susp_load_kN`` -- road-wheel load.
* ``susp_strain_ue`` -- bridge strain.
* ``susp_compliance`` -- strain / load (derived health feature).
* ``shock_a_rms_g`` -- ride shock RMS (used by the C engine's edge
  runtime).

The CVRDE-specific high-frequency simulation lives in
:mod:`sim.cvrde.hydrogas_suspension`; this generator uses the lower-
rate ``sim.tank`` physics and is the channel-set the rest of the ML
pipeline consumes.
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
    ChannelSpec("susp_load_kN", "kN", healthy=50.0, warn_hi=88.0, crit_hi=100.0,
                description="road-wheel load (LSTM input)"),
    ChannelSpec("susp_stress_MPa", "MPa", healthy=180.0, warn_hi=290.0, crit_hi=330.0,
                description="suspension link stress"),
    ChannelSpec("susp_compliance", "ue/kN", healthy=1.25, warn_hi=1.8, crit_hi=2.3,
                description="strain / load (health-feature ratio, LSTM input)"),
    ChannelSpec("susp_strain_ue", "ue", healthy=62.0, warn_hi=110.0, crit_hi=125.0,
                description="bridge strain (LSTM input)"),
    ChannelSpec("shock_a_rms_g", "g", healthy=1.8, warn_hi=3.2, crit_hi=4.5,
                description="ride shock RMS (LSTM input)"),
    ChannelSpec("shock_peak_g", "g", healthy=4.5, warn_hi=7.0, crit_hi=9.0,
                description="ride shock peak"),
)


FAULTS: tuple[FaultProfile, ...] = (
    FaultProfile("torsion_fatigue",
                 description="stiffness_mult drops; strain and shock rise"),
    FaultProfile("structural_crack",
                 description="ae_severity rises; fatigue_factor multiplies strain"),
)


def generate(out_path: str, steps: int = 2000, seed: int = 42,
             faults: Sequence[str] | None = None, dt: float = 0.05) -> int:
    """Run the suspension simulation."""
    return run_subsystem(SPEC, out_path, steps=steps, seed=seed,
                         faults=tuple(faults or ()), dt=dt)


SPEC = GeneratorSpec(
    name="suspension",
    part_key="suspension",
    label="Suspension (Hydrogas + Road Wheels)",
    description="Hydrogas suspension channels; ride shock + compliance health.",
    channels=CHANNELS,
    faults=FAULTS,
    sample_rate_hz=20.0,
    generate=generate,
)
