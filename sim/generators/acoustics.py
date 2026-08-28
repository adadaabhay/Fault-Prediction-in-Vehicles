"""Acoustics sensor data generator.

Acoustic emissions are a load-bearing discriminant for early-stage
bearing and structural faults.  The high-rate AE burst data is
computed on ``cfg.window_samples=2048``-sample windows at 4 kHz (see
``sim.config.TankConfig.sample_rate``), then downsampled to the
simulation rate for the LSTM.

Two channels reach the LSTM:
    * ``ae_event_rate`` -- the hit rate (Hz), shared with the structure
      view in the dashboard.
    * ``spl_db`` -- sound pressure level, the cabin noise floor.

The wideband vibration channels (rms / kurtosis / dominant amplitude)
are also written here for completeness; the LSTM ingests the
powertrain generator's copy of the same channels (same source, same
sample stream).
"""

from __future__ import annotations

from typing import Sequence

from ._base import (ChannelSpec, FaultProfile, GeneratorSpec,
                    run_subsystem)


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("time", "s", healthy=0.0, description="mission elapsed time"),
    ChannelSpec("step", "count", healthy=0.0, description="integer time step"),
    ChannelSpec("rpm", "rpm", healthy=1500.0,
                description="engine crankshaft speed (drives firing rate)"),
    ChannelSpec("spl_db", "dB", healthy=107.0, warn_hi=122.0, crit_hi=132.0,
                description="cabin sound pressure level"),
    ChannelSpec("acoustic_dom_freq", "Hz", healthy=120.0,
                description="dominant acoustic frequency"),
    ChannelSpec("acoustic_energy", "dB", healthy=80.0, warn_hi=95.0, crit_hi=105.0,
                description="acoustic energy (display only)"),
    ChannelSpec("ae_event_rate", "Hz", healthy=2.0, warn_hi=25.0, crit_hi=40.0,
                description="acoustic emission hit rate"),
    ChannelSpec("vib_rms", "m/s^2", healthy=0.46, warn_hi=0.75, crit_hi=1.2,
                description="wideband vibration RMS (also in powertrain)"),
    ChannelSpec("vib_kurtosis", "-", healthy=2.0, warn_hi=4.5, crit_hi=8.0,
                description="vibration kurtosis (also in powertrain)"),
    ChannelSpec("vib_dom_amp", "-", healthy=33.0, warn_hi=90.0, crit_hi=130.0,
                description="dominant FFT bin amplitude (also in powertrain)"),
)


FAULTS: tuple[FaultProfile, ...] = (
    FaultProfile("bearing_wear",
                 description="ae_event_rate rises; vib kurtosis rises"),
    FaultProfile("structural_crack",
                 description="AE energy spikes; AE event rate climbs"),
    FaultProfile("gear_wear",
                 description="sidebands on gear-mesh frequency in SPL FFT"),
)


def generate(out_path: str, steps: int = 2000, seed: int = 42,
             faults: Sequence[str] | None = None, dt: float = 0.05) -> int:
    """Run the acoustics / AE simulation."""
    return run_subsystem(SPEC, out_path, steps=steps, seed=seed,
                         faults=tuple(faults or ()), dt=dt)


SPEC = GeneratorSpec(
    name="acoustics",
    part_key="acoustics",
    label="Acoustics / SPL / AE",
    description=(
        "Cabin SPL, AE hit rate, wideband vibration.  All four LSTM "
        "inputs (ae_event_rate, spl_db, vib_rms, vib_kurtosis) are "
        "produced here; the powertrain generator publishes the same "
        "vibration channels for a driveline view."
    ),
    channels=CHANNELS,
    faults=FAULTS,
    sample_rate_hz=20.0,
    generate=generate,
)
