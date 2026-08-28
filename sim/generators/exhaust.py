"""Exhaust / Aftertreatment sensor data generator.

The exhaust-aftertreatment channels (mass flow, O2, pyrometer) are
display-only in the LSTM input schema -- the regression head already
sees ``exhaust_temp`` (also published in ``engine.csv``) and
``exhaust_pressure`` (in the engine / cooling view), which is what the
model actually needs.  This generator publishes the remaining
exhaust-specific channels so the dashboard card has values, with no
synthetic degradation profile.
"""

from __future__ import annotations

from typing import Sequence

from ._base import (ChannelSpec, FaultProfile, GeneratorSpec,
                    run_subsystem)


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("time", "s", healthy=0.0, description="mission elapsed time"),
    ChannelSpec("step", "count", healthy=0.0, description="integer time step"),
    ChannelSpec("exhaust_mass_flow", "kg/s", healthy=1.35, warn_lo=0.8, crit_lo=0.5,
                description="exhaust mass flow (hot-film)"),
    ChannelSpec("exhaust_o2_pct", "%", healthy=8.0, warn_lo=2.0, crit_lo=1.0,
                description="exhaust O2 (post-catalyst)"),
    ChannelSpec("exhaust_temp", "C", healthy=560.0, warn_hi=680.0, crit_hi=750.0,
                description="exhaust pyrometer (also published in engine.csv)"),
)


FAULTS: tuple[FaultProfile, ...] = (
    FaultProfile("exhaust_restriction",
                 description="backpressure rises; cooling_eff degrades as a side effect"),
    FaultProfile("fuel_injector_fault",
                 description="air_mult drops, fuel_mult rises; lambda residual shifts"),
)


def generate(out_path: str, steps: int = 2000, seed: int = 42,
             faults: Sequence[str] | None = None, dt: float = 0.05) -> int:
    """Run the exhaust / aftertreatment simulation."""
    return run_subsystem(SPEC, out_path, steps=steps, seed=seed,
                         faults=tuple(faults or ()), dt=dt)


SPEC = GeneratorSpec(
    name="exhaust",
    part_key="exhaust",
    label="Exhaust / Aftertreatment",
    description=(
        "Exhaust mass flow, post-catalyst O2, pyrometer.  All display only; "
        "the LSTM uses exhaust_temp (engine) and exhaust_pressure (cooling)."
    ),
    channels=CHANNELS,
    faults=FAULTS,
    sample_rate_hz=20.0,
    generate=generate,
)
