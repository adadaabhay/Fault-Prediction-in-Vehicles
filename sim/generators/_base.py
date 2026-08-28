"""Shared types and helpers for the per-subsystem sensor data generators.

Every subsystem module in :mod:`sim.generators` exposes a
:class:`GeneratorSpec` that declares what it produces (channels, units,
fault profiles, sample rate).  The master CLI introspects those specs to
render help text, to validate user-supplied fault names, and to emit a
JSON manifest sidecar next to each generated CSV so downstream consumers
can verify what they got without reverse-engineering the binary.

There is a single :func:`run_subsystem` implementation behind every
``SPEC.generate``.  The per-subsystem module just supplies the
declarative bits (channels, faults, label); the helper handles the
mission scaling, fault injection, record projection, CSV write, and
manifest write.  This keeps the per-subsystem modules thin and uniform.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


# ---------------------------------------------------------------------------
# Declarative types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelSpec:
    """One column the generator writes.

    ``unit`` is a free-form string (SI symbols allowed).  ``scale`` is
    the linear factor applied to the raw physical value before it is
    written -- this matches the ``scale`` field in
    :data:`ml.parts.PARTS` so a generator output is drop-in compatible
    with the LSTM input schema.  ``healthy`` and ``warn_*`` / ``crit_*``
    are the same anchor values the schema uses, copied here so a
    downstream user can validate a dataset without importing
    ``ml.parts`` (the manifest is self-describing).
    """

    key: str
    unit: str
    healthy: float
    warn_lo: float | None = None
    warn_hi: float | None = None
    crit_lo: float | None = None
    crit_hi: float | None = None
    scale: float = 1.0
    description: str = ""


@dataclass(frozen=True)
class FaultProfile:
    """One fault the generator can inject on demand.

    ``name`` must be one of the keys in
    :class:`sim.faults.FaultManager`'s ``FAULT_MAP`` so the master CLI
    can hand the request through to the shared fault manager.
    ``onset_frac`` is the fraction of the run at which the fault begins
    (deterministic across runs of the same length).  ``ramp_frac`` is the
    duration of the severity ramp.  ``max_severity`` caps the sigmoid
    ramp.  All three fractions are in [0, 1] and the dataclass itself
    enforces this in :meth:`__post_init__`.
    """

    name: str
    onset_frac: float = 0.30
    ramp_frac: float = 0.20
    max_severity: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        for label, value in (("onset_frac", self.onset_frac),
                             ("ramp_frac", self.ramp_frac),
                             ("max_severity", self.max_severity)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"FaultProfile({self.name!r}).{label}={value!r} "
                    f"must be in [0, 1]"
                )


@dataclass(frozen=True)
class GeneratorSpec:
    """Static description of a generator.

    The :attr:`generate` callable is bound to a partial of
    :func:`run_subsystem` at module import time -- every per-subsystem
    module's ``SPEC.generate`` is functionally identical apart from the
    captured spec.  This dataclass describes what the callable produces;
    the master CLI uses it to validate user input, render help text, and
    emit a manifest.
    """

    name: str
    part_key: str
    label: str
    description: str
    channels: tuple[ChannelSpec, ...]
    faults: tuple[FaultProfile, ...]
    sample_rate_hz: float
    generate: Callable[..., int]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def manifest(self, *, out_path: Path, steps: int, seed: int,
                 faults: Sequence[str], dt: float, sha256: str,
                 rows_written: int) -> dict:
        """JSON-serialisable manifest for this generation run."""
        return {
            "schema_version": 1,
            "subsystem": self.name,
            "part_key": self.part_key,
            "label": self.label,
            "description": self.description,
            "sample_rate_hz": self.sample_rate_hz,
            "dt_s": dt,
            "steps": steps,
            "duration_s": round(steps * dt, 6),
            "rows_written": rows_written,
            "seed": seed,
            "faults_injected": list(faults),
            "channels": [asdict(c) for c in self.channels],
            "fault_profiles": [asdict(f) for f in self.faults],
            "notes": list(self.notes),
            "output_csv": out_path.name,
            "output_sha256": sha256,
            "generated_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
        }


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def write_manifest(spec: GeneratorSpec, *, out_path: Path, steps: int,
                   seed: int, faults: Sequence[str], dt: float,
                   rows_written: int) -> Path:
    """Write the manifest sidecar JSON next to ``out_path``."""
    manifest = spec.manifest(
        out_path=out_path, steps=steps, seed=seed, faults=faults, dt=dt,
        sha256=sha256_file(out_path), rows_written=rows_written,
    )
    manifest_path = out_path.with_name(out_path.name + ".manifest.json")
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    return manifest_path


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    """Hex SHA-256 of the file at ``path`` -- 1 MB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (write-to-tmp + rename).

    Avoids the half-written-file race when the master CLI writes a CSV
    and a manifest in parallel, or when a verifier opens the manifest
    while a generator is still flushing.  On Windows, ``os.replace`` is
    atomic at the filesystem level.
    """
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv(out_path: Path, columns: Sequence[str],
              rows: Sequence[Sequence[float]]) -> int:
    """Write a CSV with the given column order.  Returns the row count."""
    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(list(columns))
        w.writerows([_fmt(v) for v in r] for r in rows)
    import os
    os.replace(tmp, out_path)
    return len(rows)


def _fmt(v: float) -> str:
    """Format a CSV cell value.

    * ``None`` -> empty string.
    * ``NaN`` -> empty string (pandas reads blanks; Excel shows the cell
      as empty).
    * ``Inf`` -> ``"inf"`` / ``"-inf"`` (rare, but ``%.6f`` would crash
      on it).
    * Integer-valued floats (e.g. the ``step`` counter) -> ``"1"``, not
      ``"1.000000"``.
    * Everything else -> ``%.6f`` (no scientific notation -- it is what
      makes ``1e-9`` look like an error to a human reviewer).
    """
    if v is None:
        return ""
    f = float(v)
    if math.isnan(f):
        return ""
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    if f.is_integer():
        return str(int(f))
    return f"{f:.6f}"


# ---------------------------------------------------------------------------
# Shared run helper -- the one implementation behind every SPEC.generate
# ---------------------------------------------------------------------------
_NAN = float("nan")  # hoisted to avoid re-allocation per missing key per record


def run_subsystem(spec: GeneratorSpec, out_path: str, *, steps: int,
                  seed: int, dt: float, faults: Sequence[str] = ()) -> int:
    """Run the simulator, project onto ``spec.channels``, write CSV + manifest.

    Every per-subsystem ``generate()`` is a thin wrapper around this
    function.  The simulator is invoked once per call -- the master CLI
    does not memoize runs across subsystems, so the ``lubrication`` and
    ``cooling`` aliases each call this independently and produce CSVs
    derived from independent (but identical, given the same seed) runs.

    Returns the row count actually written (the simulator's record
    count, not the requested step count -- they can differ by one when
    ``steps * dt`` does not divide the mission length evenly).
    """
    import numpy as np
    from sim.config import TankConfig
    from sim.faults import FaultManager
    from sim.tank import MissionStep, TankSimulator, default_mission

    cfg = TankConfig()
    cfg.dt = dt
    cfg.noise_seed = seed

    base = default_mission(cfg)
    factor = (steps * dt) / sum(m.duration_s for m in base)
    mission = [
        MissionStep(m.duration_s * factor, m.rpm, m.load, m.terrain)
        for m in base
    ]

    fm = FaultManager(np.random.default_rng(seed))
    for i, profile in enumerate(spec.faults):
        if profile.name in faults:
            fm.add(
                profile.name,
                start_step=int(steps * (profile.onset_frac + 0.10 * i)),
                ramp_steps=max(int(steps * profile.ramp_frac), 10),
            )

    sim = TankSimulator(cfg, faults=fm, mission=mission, seed=seed)
    records = sim.run()

    cols = [c.key for c in spec.channels]
    # Apply each channel's declared ``scale`` so the CSV writes the
    # *published* unit (per ``ChannelSpec.unit``), not the raw SI value
    # the simulator produced.  ``NaN * scale == NaN`` in IEEE 754, so
    # missing keys still get an empty cell via ``_fmt``.
    scales = [c.scale for c in spec.channels]
    rows = [
        [r.get(c, _NAN) * s for c, s in zip(cols, scales)]
        for r in records
    ]
    csv_path = Path(out_path)
    rows_written = write_csv(csv_path, cols, rows)
    write_manifest(
        spec, out_path=csv_path, steps=steps, seed=seed,
        faults=faults, dt=dt, rows_written=rows_written,
    )
    return rows_written
