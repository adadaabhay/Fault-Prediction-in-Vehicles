"""Master CLI for the per-subsystem sensor data generators.

Usage examples
--------------

    # One CSV per subsystem, 500 steps, fault-free.
    python -m tools.generate_sensor_data --subsystem all --steps 500

    # Just the engine view, with two faults injected.
    python -m tools.generate_sensor_data --subsystem engine --steps 2000 \\
        --fault cooling_failure oil_pump_degradation

    # List every registered subsystem + its channels and fault profiles.
    python -m tools.generate_sensor_data --list

    # Custom output directory.
    python -m tools.generate_sensor_data --subsystem all --out-dir data/sim

Every run writes:

    <out-dir>/<subsystem>.csv               the data
    <out-dir>/<subsystem>.csv.manifest.json the schema/fault/sample-rate manifest
    <out-dir>/MANIFEST.json                 aggregate manifest (when --subsystem all)

The aggregate manifest is the data-quality contract: it lists every CSV
written, its SHA-256, and a back-reference to the per-subsystem spec.
A training run can refuse to start if the aggregate is missing or stale.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from sim.generators import (ALL_SUBSYSTEMS, SUBSYSTEMS, SUBSYSTEM_ALIASES,
                            GeneratorSpec, sha256_file)
from sim.generators._base import atomic_write_text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phm-generate",
        description=(
            "Generate simulated per-subsystem sensor datasets for the "
            "phm-vehicle PHM/CBM+ stack.  Writes a CSV + JSON manifest "
            "sidecar for every selected subsystem."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--subsystem", "-s", action="append", default=None, metavar="NAME",
        help=(
            "Subsystem to generate.  Repeat the flag for multiple, or pass "
            "'all'.  Aliases: 'lubrication' and 'cooling' resolve to the "
            "engine generator.  Available: "
            + ", ".join(ALL_SUBSYSTEMS)
        ),
    )
    p.add_argument(
        "--out-dir", default="data/simulated",
        help="Output directory; created if it does not exist.",
    )
    p.add_argument(
        "--steps", type=int, default=2000,
        help="Number of simulation time steps per CSV.  Total duration is "
             "steps * dt seconds.",
    )
    p.add_argument(
        "--dt", type=float, default=0.05,
        help="Simulation time step in seconds (default 0.05 s = 20 Hz).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.  Same seed + same faults + same steps -> "
             "bit-identical CSV.",
    )
    p.add_argument(
        "--fault", "-f", action="append", default=None, metavar="NAME",
        help=(
            "Fault to inject (repeatable).  Applies to every selected "
            "subsystem; if a subsystem does not declare the fault, it is "
            "skipped with a warning.  If a name is skipped for *every* "
            "selected subsystem, the run aborts (typo guard)."
        ),
    )
    p.add_argument(
        "--list", action="store_true",
        help="Print the registered subsystems and exit.",
    )
    p.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-subsystem progress output.",
    )
    return p


# ---------------------------------------------------------------------------
# Subsystem resolution
# ---------------------------------------------------------------------------
def _resolve_targets(requested: Iterable[str] | None) -> list[str]:
    """Normalise user input -> ordered list of generator names to run.

    Unknown names and aliases that point nowhere both raise.  The order
    is deterministic (declaration order in
    :data:`sim.generators.SUBSYSTEMS`) so the aggregate manifest's CSV
    list is reproducible across runs.
    """
    if not requested or "all" in requested:
        return list(SUBSYSTEMS)

    seen: set[str] = set()
    out: list[str] = []
    for raw in requested:
        if raw in SUBSYSTEMS:
            if raw not in seen:
                out.append(raw)
                seen.add(raw)
        elif raw in SUBSYSTEM_ALIASES:
            target = SUBSYSTEM_ALIASES[raw]
            if target not in seen:
                out.append(target)
                seen.add(target)
        else:
            raise SystemExit(
                f"unknown subsystem {raw!r}.  Available: "
                f"{', '.join(ALL_SUBSYSTEMS)}"
            )
    return out


# ---------------------------------------------------------------------------
# Per-subsystem dispatch
# ---------------------------------------------------------------------------
def _filter_faults(spec: GeneratorSpec, faults: list[str] | None) -> list[str]:
    """Return the subset of ``faults`` the spec knows about.

    Unknown faults for this subsystem are skipped with a stderr warning
    so a single ``--fault`` list can target multiple subsystems without
    the operator having to remember which fault is declared where.
    """
    if not faults:
        return []
    declared = {f.name for f in spec.faults}
    return [f for f in faults if f in declared]


# ---------------------------------------------------------------------------
# Aggregate manifest
# ---------------------------------------------------------------------------
def _write_aggregate_manifest(out_dir: Path, runs: list[dict], *,
                               steps: int, seed: int, dt: float) -> Path:
    """Emit ``<out-dir>/MANIFEST.json`` summarising every CSV in the run.

    The aggregate manifest is the artefact a training run, a release
    pipeline or a data-quality check would consume.  It is the only
    file the operator needs to inspect to know what was generated.
    """
    payload = {
        "schema_version": 1,
        "tool": "phm-generate",
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "out_dir": str(out_dir),
        "steps": steps,
        "seed": seed,
        "dt_s": dt,
        "subsystems": [],
    }
    for r in runs:
        # Each run stores the basename of the CSV/manifest; both live
        # directly in ``out_dir`` so the join is unambiguous regardless
        # of platform.
        csv = out_dir / r["csv"]
        man = out_dir / r["manifest"]
        payload["subsystems"].append({
            **r,
            "csv_sha256": sha256_file(csv) if csv.exists() else None,
            "manifest_sha256": sha256_file(man) if man.exists() else None,
            "csv_exists": csv.exists(),
            "manifest_exists": man.exists(),
        })
    path = out_dir / "MANIFEST.json"
    atomic_write_text(path, json.dumps(payload, indent=2))
    return path


# ---------------------------------------------------------------------------
# List mode
# ---------------------------------------------------------------------------
def _print_registry() -> None:
    print("Registered per-subsystem generators:")
    print()
    for name, spec in SUBSYSTEMS.items():
        print(f"  {name:<14}  {spec.label}")
        print(f"  {'':14}  part_key: {spec.part_key}, "
              f"sample_rate: {spec.sample_rate_hz} Hz, "
              f"channels: {len(spec.channels)}, "
              f"faults: {len(spec.faults)}")
        print(f"  {'':14}  {spec.description}")
        if spec.faults:
            for f in spec.faults:
                print(f"  {'':14}    - {f.name:<28} {f.description}")
        if spec.notes:
            for note in spec.notes:
                print(f"  {'':14}  note: {note}")
        print()
    if SUBSYSTEM_ALIASES:
        print("Aliases (share a generator):")
        for alias, target in SUBSYSTEM_ALIASES.items():
            print(f"  {alias:<14} -> {target}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        _print_registry()
        return 0

    targets = _resolve_targets(args.subsystem)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(
            f"phm-generate: writing {len(targets)} subsystem CSV(s) "
            f"to {out_dir}",
            file=sys.stderr,
        )

    runs: list[dict] = []
    skipped: dict[str, list[str]] = defaultdict(list)  # fault -> [subsystem, ...]
    for name in targets:
        spec = SUBSYSTEMS[name]
        declared = {f.name for f in spec.faults}
        applicable = [f for f in (args.fault or []) if f in declared]
        for f in (args.fault or []):
            if f not in declared:
                skipped[f].append(name)
        t0 = time.perf_counter()
        rows_written = spec.generate(
            str(out_dir / f"{name}.csv"),
            steps=args.steps, seed=args.seed, faults=applicable, dt=args.dt,
        )
        elapsed = time.perf_counter() - t0
        csv_path = out_dir / f"{name}.csv"
        manifest_path = csv_path.with_name(csv_path.name + ".manifest.json")
        if not args.quiet:
            print(
                f"  [{name:<14}] {rows_written:>6} rows, "
                f"{len(applicable)} fault(s), {elapsed:5.2f}s  "
                f"->  {csv_path.name}",
                file=sys.stderr,
            )
        runs.append({
            "subsystem": name,
            "part_key": spec.part_key,
            "label": spec.label,
            "csv": csv_path.name,
            "manifest": manifest_path.name,
            "rows_written": rows_written,
            "faults": applicable,
            "elapsed_s": round(elapsed, 3),
        })

    # Typo guard: a fault name skipped for *every* selected subsystem
    # is a typo, not a per-subsystem-not-declared case.  Abort.
    if args.fault:
        for fault, subs in skipped.items():
            if len(subs) == len(targets):
                raise SystemExit(
                    f"fault {fault!r} is not declared by any selected "
                    f"subsystem.  Check spelling with `phm-generate --list`."
                )

    # Per-subsystem warnings for the legit case (multi-subsystem run,
    # fault declared by some but not all).
    for fault, subs in skipped.items():
        print(
            f"warning: fault {fault!r} skipped for {', '.join(subs)} "
            f"(not declared by those subsystems)",
            file=sys.stderr,
        )

    aggregate = _write_aggregate_manifest(
        out_dir, runs, steps=args.steps, seed=args.seed, dt=args.dt,
    )
    if not args.quiet:
        print(f"phm-generate: aggregate manifest -> {aggregate}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
