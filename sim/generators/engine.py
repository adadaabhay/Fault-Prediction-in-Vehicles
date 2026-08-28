"""Engine (and co-located lubrication + cooling) sensor data generator.

The engine powerpack carries the engine, lubrication, and cooling
sensors -- they share the same physical assembly and the same telemetry
record.  This generator writes the engine + lubrication + cooling
channels, so a single run feeds three of the eleven ``ml.parts.PARTS``
views.

Channels written (per ``ml.parts.PARTS``):
    engine:       coolant_temp, oil_temp, exhaust_temp, lambda_residual
    lubrication:  oil_pressure, oil_temp, oil_viscosity,
                  debris_rate, debris_cumulative
    cooling:      coolant_temp, coolant_level, exhaust_pressure

Fault profiles injected on demand:
    cooling_failure, oil_pump_degradation, fuel_injector_fault,
    bearing_wear (raises oil_temp_bias as a side-channel cue).
"""

from __future__ import annotations

from typing import Sequence

from ._base import (ChannelSpec, FaultProfile, GeneratorSpec,
                    run_subsystem)


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("time", "s", healthy=0.0,
                description="mission elapsed time"),
    ChannelSpec("step", "count", healthy=0.0,
                description="integer time step"),
    ChannelSpec("rpm", "rpm", healthy=1500.0, warn_lo=800.0, crit_lo=600.0,
                warn_hi=2400.0, crit_hi=2600.0,
                description="engine crankshaft speed"),
    ChannelSpec("load", "frac", healthy=0.5, warn_hi=0.9, crit_hi=1.0,
                description="normalized load (0-1)"),
    ChannelSpec("coolant_temp", "C", healthy=92.0, warn_hi=105.0, crit_hi=120.0,
                description="engine coolant out temperature"),
    ChannelSpec("coolant_level", "frac", healthy=0.95, warn_lo=0.55, crit_lo=0.35,
                description="coolant expansion tank level (cooling LSTM input)"),
    ChannelSpec("oil_temp", "C", healthy=92.0, warn_hi=115.0, crit_hi=135.0,
                description="sump oil temperature"),
    ChannelSpec("oil_pressure", "bar", healthy=5.0, warn_lo=3.2, crit_lo=2.0,
                scale=1e-5,
                description="main gallery oil pressure (Pa stored, bar published)"),
    ChannelSpec("oil_viscosity", "cSt", healthy=15.0, warn_lo=8.0, crit_lo=5.0,
                description="sump oil kinematic viscosity"),
    ChannelSpec("exhaust_temp", "C", healthy=560.0, warn_hi=680.0, crit_hi=750.0,
                description="exhaust gas temperature (pyrometer)"),
    ChannelSpec("exhaust_pressure", "bar", healthy=1.29, warn_hi=2.2, crit_hi=2.6,
                scale=1e-5,
                description="exhaust back-pressure (post-turbine)"),
    ChannelSpec("debris_rate", "pts/s", healthy=1.0, warn_hi=8.0, crit_hi=15.0,
                description="oil debris particle count rate"),
    ChannelSpec("debris_cumulative", "pts", healthy=500.0, warn_hi=4000.0,
                crit_hi=8000.0,
                description="cumulative debris count since last oil change"),
    ChannelSpec("lambda_residual", "-", healthy=1.0,
                warn_lo=0.82, warn_hi=1.25, crit_lo=0.70, crit_hi=1.45,
                description="load-normalised air-fuel ratio residual"),
)


FAULTS: tuple[FaultProfile, ...] = (
    FaultProfile("cooling_failure",
                 description="loss of radiator effectiveness; coolant_temp climbs"),
    FaultProfile("oil_pump_degradation",
                 description="pump_eff decay; oil_pressure drops, oil_temp rises"),
    FaultProfile("fuel_injector_fault",
                 description="air_mult drops, fuel_mult rises; lambda residual shifts"),
    FaultProfile("bearing_wear",
                 description="vib_severity + debris + oil_temp bias; side-channel cue"),
)


def generate(out_path: str, steps: int = 2000, seed: int = 42,
             faults: Sequence[str] | None = None, dt: float = 0.05) -> int:
    """Run the engine / lubrication / cooling simulation."""
    return run_subsystem(SPEC, out_path, steps=steps, seed=seed,
                         faults=tuple(faults or ()), dt=dt)


SPEC = GeneratorSpec(
    name="engine",
    part_key="engine",
    label="Engine / Lubrication / Cooling",
    description=(
        "Engine powerpack: coolant + oil + exhaust sensors.  One run covers "
        "the engine, lubrication, and cooling subsystem views "
        "(ml.parts.PARTS[engine|lubrication|cooling])."
    ),
    channels=CHANNELS,
    faults=FAULTS,
    sample_rate_hz=20.0,
    generate=generate,
    notes=(
        "lambda_residual is load-normalised; the absolute lambda channel is "
        "in the engine raw stream but excluded here because it is not an "
        "LSTM input (see ml.parts.PARTS[engine]).",
        "oil_pressure and exhaust_pressure are stored in Pa in the raw "
        "telemetry; this generator publishes them in bar to match the "
        "schema's scale=1e-5 contract.",
    ),
)
