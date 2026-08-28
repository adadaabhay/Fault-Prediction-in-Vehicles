"""NBC (Nuclear / Biological / Chemical) Protection + APU generator.

Wired to :class:`sim.cvrde.auxiliary_nbc.CVRDEAuxiliaryNBC` -- a 10 Hz
first-order APU + cabin overpressure physics model that already ships in
the repo.  The generator drives the module for ``steps`` ticks at the
caller's ``dt`` (the cvrde physics is dt-agnostic; its 10 Hz design
intent is just an integrator time-constant choice, not a hard contract),
re-keys the returned ``cvrde_*`` dict to the bare-prefix channel names
the dashboard's NBC card reads, and emits the standard CSV + manifest.

NBC has no synthetic fault model today (a chemical/biological threat
cannot be credibly reproduced).  ``ml.parts.PARTS['nbc']`` flags both
channels ``health_exclude=True`` so a steady-state healthy APU never
moves the LSTM's health index.  The physics module exposes
``filter_dust_load`` and ``cabin_seal_leak`` inputs that a domain
expert can wire into ``sim.faults.FaultManager.FAULT_MAP`` for future
fault injection.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from ._base import (ChannelSpec, FaultProfile, GeneratorSpec, write_csv,
                    write_manifest)


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("time", "s", healthy=0.0, description="mission elapsed time"),
    ChannelSpec("step", "count", healthy=0.0, description="integer time step"),
    ChannelSpec("nbc_overpressure", "Pa", healthy=500.0,
                warn_lo=150.0, crit_lo=50.0,
                description="cabin overpressure vs ambient (HEPA barrier)"),
    ChannelSpec("nbc_filter_dp", "Pa", healthy=120.0,
                warn_hi=1500.0, crit_hi=1800.0,
                description="pressure drop across the HEPA / carbon filter"),
)


# NBC has no synthetic fault model (see module docstring).  Real
# deployment would add ``filter_clog`` and ``cabin_seal_leak`` here
# once ``sim.faults.FaultManager.FAULT_MAP`` learns the keys.
FAULTS: tuple[FaultProfile, ...] = ()


def generate(out_path: str, steps: int = 2000, seed: int = 42,
             faults: Sequence[str] | None = None, dt: float = 0.05) -> int:
    """Drive :class:`CVRDEAuxiliaryNBC` for ``steps`` ticks; write CSV + manifest.

    The cvrde physics module returns 6 channels under the ``cvrde_*``
    prefix; we re-key the two the dashboard reads (``nbc_overpressure``,
    ``nbc_filter_dp``) and drop the rest.  Same seed + same steps + same
    dt -> bit-identical CSV (the cvrde module uses
    ``cfg.noise_seed + 200`` as its RNG seed).
    """
    # Lazy import so this module loads even if sim.cvrde is missing --
    # the import only fails at generate() time, with a clear traceback.
    from sim.cvrde.auxiliary_nbc import CVRDEAuxiliaryNBC
    from sim.cvrde.cvrde_config import CVRDETankConfig

    cfg = CVRDETankConfig()
    cfg.dt = dt
    cfg.noise_seed = seed
    apu = CVRDEAuxiliaryNBC(cfg)

    cols = [c.key for c in CHANNELS]
    rows: list[list[float]] = []
    for step in range(steps):
        t = step * dt
        # Slow 30 s mission-load oscillation: 1.5-4.5 kW.  The APU
        # charges / discharges the bus against this load, so bus_voltage
        # wobbles by a few hundred mV per cycle.
        electrical_load_kw = 3.0 + 1.5 * math.sin(2.0 * math.pi * t / 30.0)
        rec = apu.step(
            electrical_load_kw=electrical_load_kw,
            nbc_blower_on=True,
            filter_dust_load=0.0,
            cabin_seal_leak=0.0,
        )
        rows.append([
            t,
            float(step),
            rec["cvrde_nbc_overpressure_pa"],
            rec["cvrde_nbc_filter_dp_pa"],
        ])

    csv_path = Path(out_path)
    rows_written = write_csv(csv_path, cols, rows)
    write_manifest(
        SPEC, out_path=csv_path, steps=steps, seed=seed,
        faults=tuple(faults or ()), dt=dt, rows_written=rows_written,
    )
    return rows_written


SPEC = GeneratorSpec(
    name="nbc",
    part_key="nbc",
    label="NBC Protection + APU",
    description=(
        "Cabin overpressure, HEPA filter dp, and APU bus voltage.  "
        "Wired to sim.cvrde.auxiliary_nbc.CVRDEAuxiliaryNBC; re-keys "
        "the cvrde_* physics output to the bare-prefix channel names "
        "the dashboard reads."
    ),
    channels=CHANNELS,
    faults=FAULTS,
    sample_rate_hz=20.0,
    generate=generate,
    notes=(
        "Wired to sim.cvrde.auxiliary_nbc.CVRDEAuxiliaryNBC; the "
        "module's 10 Hz integrator time-constant is honoured "
        "implicitly by the per-step call (the model is dt-agnostic).",
        "No synthetic fault model: filter_dust_load and cabin_seal_leak "
        "are hard-zero.  A domain expert would wire them into "
        "sim.faults.FaultManager.FAULT_MAP for future fault injection.",
        "Both published channels are flagged health_exclude=True in "
        "ml.parts.PARTS['nbc'] -- a healthy APU never moves the LSTM "
        "health index.",
    ),
)
