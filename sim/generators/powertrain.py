"""Powertrain (driveline, vibration, gear-mesh) sensor data generator.

The powertrain is the mechanical link between the engine output shaft
and the sprocket/track drive.  Three orthogonal discriminators live
here:

* ``shaft_torque`` and ``driveline_efficiency`` -- the energy transfer
  path.  Efficiency loss is one of the declared fault classes.
* ``vib_rms`` / ``vib_kurtosis`` / ``vib_dom_amp`` -- the rolling-
  element and gear-mesh fault surface.  Bearing and gear wear drive
  these.

All channels are LSTM inputs (no ``health_exclude`` flag in
``ml.parts.PARTS[powertrain]``).
"""

from __future__ import annotations

from typing import Sequence

from ._base import (ChannelSpec, FaultProfile, GeneratorSpec,
                    run_subsystem)


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("time", "s", healthy=0.0, description="mission elapsed time"),
    ChannelSpec("step", "count", healthy=0.0, description="integer time step"),
    ChannelSpec("rpm", "rpm", healthy=1500.0, warn_hi=2400.0, crit_hi=2600.0,
                description="driveshaft speed"),
    ChannelSpec("shaft_torque", "kN*m", healthy=1.6, scale=0.001,
                description="shaft torque (N*m stored, kN*m published)"),
    ChannelSpec("driveline_efficiency", "%", healthy=100.0, scale=100.0,
                warn_lo=88.0, crit_lo=80.0,
                description="driveline mechanical efficiency (frac stored, % published)"),
    ChannelSpec("vib_rms", "m/s^2", healthy=0.46, warn_hi=0.75, crit_hi=1.2,
                description="wideband vibration RMS"),
    ChannelSpec("vib_kurtosis", "-", healthy=2.0, warn_hi=4.5, crit_hi=8.0,
                description="vibration kurtosis (impulsiveness indicator)"),
    ChannelSpec("vib_dom_amp", "-", healthy=33.0, warn_hi=90.0, crit_hi=130.0,
                description="dominant FFT bin amplitude"),
    ChannelSpec("sprocket_torque", "N*m", healthy=12000.0, warn_hi=18000.0,
                crit_hi=22000.0,
                description="final-drive sprocket torque"),
    ChannelSpec("shaft_omega", "rad/s", healthy=157.0,
                description="driveshaft angular velocity"),
    ChannelSpec("delivered_power", "kW", healthy=600.0, warn_hi=900.0, crit_hi=1000.0,
                description="power delivered to the sprocket"),
)


FAULTS: tuple[FaultProfile, ...] = (
    FaultProfile("drivetrain_efficiency_loss",
                 description="efficiency drops; vibration rises"),
    FaultProfile("bearing_wear",
                 description="vib_rms + vib_kurtosis + debris_cumulative"),
    FaultProfile("gear_wear",
                 description="vib_rms + sidebands on gear-mesh frequency"),
)


def generate(out_path: str, steps: int = 2000, seed: int = 42,
             faults: Sequence[str] | None = None, dt: float = 0.05) -> int:
    """Run the powertrain simulation."""
    return run_subsystem(SPEC, out_path, steps=steps, seed=seed,
                         faults=tuple(faults or ()), dt=dt)


SPEC = GeneratorSpec(
    name="powertrain",
    part_key="powertrain",
    label="Powertrain / Driveline / Vibration",
    description=(
        "Driveline, shaft torque, gear-mesh and rolling-element vibration "
        "channels -- the SVM/LSTM feature surface for bearing and gear "
        "faults."
    ),
    channels=CHANNELS,
    faults=FAULTS,
    sample_rate_hz=20.0,
    generate=generate,
    notes=(
        "Sample rate is the simulation step (20 Hz by default).  "
        "vibration kurtosis is computed on cfg.window_samples=2048-sample "
        "windows at 4 kHz, then downsampled to the run rate (see "
        "sim.config.TankConfig.sample_rate).",
    ),
)
