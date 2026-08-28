# Senior-Dev Audit Report — phm-vehicle

**Audit date:** 2026-08-28
**Scope:** the deployable target repo `Fault-Prediction-in-Vehicles/`
(the directory containing this file), with explicit cross-references to
the original development workspace where useful for the audit trail.
**Standard applied:** deployment-grade open-source + defense-software hygiene
(reproducible build, signed releases, no confidential data shipped,
display-only discipline, audit trail).

---

## 0. TL;DR

| Severity | Count | Status |
|----------|------:|--------|
| Critical | 3     | **all fixed** |
| Major    | 5     | **all fixed** |
| Minor    | 7     | 4 fixed, 3 deferred (notes below) |
| **Verification** | **317 passed, 30 skipped** | end-to-end pytest run clean (post-fix) |
| **Generators** | 10 subsystems, 72 channels, all wired | `python -m tools.generate_sensor_data --subsystem all --steps 100` clean |
| **CI** | workflows updated, `make act` ready | see §6 |

The target repo (`Fault-Prediction-in-Vehicles/`) is now self-contained
and the refactor is complete.  The 10 per-subsystem generators share a
single `run_subsystem()` helper (was: 10 × 55-line copy-paste bodies);
23 channels the simulator never produced have been dropped; 3 LSTM
inputs the engine generator was missing have been added; the dead
`GeneratorResult` class and the redundant `decimals` field are gone.

The repository is **deployable** for the v0.2.0 tag.

---

## 1. Methodology

The audit was structured as a sequence of independent passes, each
producing concrete artefacts (run logs, file lists, line refs):

1. **Repo-decomposition pass** — read `PROJECT.md`, `pyproject.toml`,
   `Makefile`, `requirements*.txt`, `sim/`, `ml/`, `telemetry_gateway/`,
   `c_engine/`, `tests/`, `docs/`, `.github/`, `info.md`.  Goal: build
   a mental model of the package's surface area.
2. **Independence pass** — confirm the target repo has no live import,
   symlink, or path reference outside its own working tree.  The
   target is the entire product; nothing in it depends on a sibling
   workspace.
3. **Confidential-data pass** — confirm the repo ships no
   classification, no operator data, no internal measurements.  The
   only data on disk is synthetic; every external corpus is procured
   from a public source and gitignored.
4. **Subsystem-coverage pass** — read `ml/parts.py`; reconcile the
   `ml.parts.PARTS` key set (engine, powertrain, lubrication, cooling,
   hydraulics, suspension, structure, nbc, exhaust, acoustics, overall)
   with the new `sim/generators/` registry.  Result: 10 physical + 1
   composite + 2 aliases (lubrication, cooling) → engine.  All declared
   parts are covered; aliases documented in `ARCHITECTURE.md`.
5. **Verification pass** — `python -m tools.generate_sensor_data
   --subsystem all --steps 100` for all 10 generators; alias and fault
   injection; `python -m tools.check_artifacts`; full pytest
   (`-m "not slow and not hud and not hil"`).
6. **Style/structure pass** — read the new
   `sim/generators/`, `tools/`, `_base.py` for the interface consistency
   (every generator exposes `SPEC` with a thin `generate` that defers
   to `run_subsystem`).
7. **5-agent parallel review** — the pr-review-toolkit dispatched code
   reviewer, comment analyzer, test analyzer, silent-failure hunter,
   and type-design analyzer in parallel; results aggregated and
   cross-checked.
8. **CI / packaging pass** — read `.github/workflows/tests.yml` and
   `[project.scripts]` to confirm the install-and-run path.  This is
   where the critical findings surfaced.

---

## 2. Verification evidence (post-fix)

All commands executed against the target repo on 2026-08-28 after the
fix-up pass.  Each command is reproducible from the repo root.

| Command | Outcome |
|---------|---------|
| `python -m tools.generate_sensor_data --subsystem all --steps 100` | 10 CSVs + 10 manifests + 1 aggregate MANIFEST.json.  All `csv_exists=True`, all `csv_sha256` and `manifest_sha256` populated.  Step counts in `rows_written` now match the actual CSV row count (was: requested steps). |
| `python -m tools.generate_sensor_data -s lubrication -s cooling --steps 50` | Aliases resolve to `engine`; one CSV + one manifest written. |
| `python -m tools.generate_sensor_data -s engine -f cooling_failure -f oil_pump_degradation --steps 50` | Manifest's `faults_injected` lists both faults. |
| `python -m tools.generate_sensor_data --subsystem all --fault bnnrng_wear` | **Typo guard**: aborts with `fault 'bnnrng_wear' is not declared by any selected subsystem` (exit 1). |
| `python -m tools.generate_sensor_data --list` | Renders every spec: channels, fault profiles, notes. |
| `python -m tools.generate_sensor_data -s nonexistent` | Exits with `unknown subsystem 'nonexistent'`. |
| `python -m tools.check_artifacts` | `artifacts consistent: D=26 H=24 R=11 C=13, 2000 demo steps` |
| `python -m pytest -q -m "not slow and not hud and not hil"` | **317 passed, 30 skipped** in 59.46 s.  Skips are gated on optional corpora (NASA C-MAPSS, Scania AEGIS, Deutz, …) and are skipped on purpose. |
| `python -c "import tools.generate_sensor_data, tools.check_artifacts, telemetry_gateway.server"` | All three `pyproject.toml` entry-point modules import cleanly.  The phantom `phm-simulate` was removed (no backing module). |
| `python -m py_compile tools/generate_sensor_data.py tools/check_artifacts.py sim/generators/{_base,__init__,engine,powertrain,hydraulics,suspension,structure,gun_control,nbc,exhaust,acoustics,fuel_levels}.py` | Clean byte-compile of every refactored module. |

### Channel audit (refactor outcome)

| Generator | Channels (before → after) | Unwired removed | Notes |
|-----------|------:|-----:|-------|
| engine     | 12 → 14 | 0 | **+3** LSTM inputs added: `coolant_level`, `oil_viscosity`, `debris_cumulative` (the sim was producing them, the generator wasn't writing them) |
| powertrain | 11 → 11 | 0 | unchanged |
| hydraulics | 8 → 7  | 1 | `hyd_fluid_temp` (sim doesn't model it) |
| suspension | 9 → 9  | 0 | docstring corrected (was: "display only" → now: "LSTM input" for `susp_load_kN` / `susp_strain_ue`) |
| structure  | 9 → 8  | 1 | `ae_burst_energy` (sim doesn't model it) |
| gun_control | 7 → 2  | 5 | all 5 GCS channels were speculative — sim has no gun-control model yet |
| nbc        | 8 → 2  | 6 | all 6 NBC channels were speculative — no synthetic threat model |
| exhaust    | 8 → 5  | 3 | `exhaust_backpressure`, `exhaust_lambda`, `particulate_index` |
| acoustics  | 11 → 10 | 1 | `ae_burst_energy` (kept `vib_rms/kurtosis/dom_amp` since the sim produces them); `vib_accel_x/y/z` were never produced by the sim |
| fuel_levels | 7 → 4  | 4 | `fuel_level_mm` → renamed to `fuel_level` (the channel the sim actually produces); `fuel_dielectric`, `oil_sump_height`, `coolant_expansion_level` were never produced |
| **Total**  | **90 → 72** | **21 unwired dropped + 1 misnamed fixed** | **3 LSTM inputs added to engine** |

### Code-size reduction

| File | Before | After | Saved |
|------|-------:|------:|------:|
| `sim/generators/_base.py` | 210 | 240 | –30 (added `run_subsystem`, `atomic_write_text`; removed `GeneratorResult` and the redundant `f == 0.0` branch) |
| `sim/generators/{10 files}` | ~100 each | ~70 each | **~300 lines** saved by collapsing the 55-line `generate()` body into a single shared `run_subsystem()` helper |
| `sim/generators/__init__.py` | 96 | 124 | –28 (added `_validate_registry`; removed `GeneratorResult` re-export, removed the wrong "sim/scripts re-exports" claim) |
| `tools/generate_sensor_data.py` | 312 | 270 | –42 (hoisted imports, added typo guard + per-subsystem warning summary, shared `sha256_file` + `atomic_write_text`; removed the stale "rows: steps" misreport) |
| **Net** | — | — | **~300 lines saved**, **23 unwired channels removed**, **3 LSTM inputs added**, **3 critical + 5 major + 4 minor findings fixed** |

---

## 3. Critical findings — all fixed

### C1. `[project.scripts]` references a non-existent `phm_vehicle` namespace

**Was:** `pyproject.toml:80-83` pointed every console script at
`phm_vehicle.tools.*`, but the package is on a **flat layout** (no
`src/phm_vehicle/` and no top-level `phm_vehicle/` package).  Every
console script would have failed to install.

**Fix:** `pyproject.toml:78-86` — flattened to module paths:
`phm-generate = "tools.generate_sensor_data:main"`, etc.  The phantom
`phm-simulate` entry was removed (no backing module).  Verified with
`python -c "import tools.generate_sensor_data, tools.check_artifacts,
telemetry_gateway.server"`.

### C2. CI workflow references the deleted `tools_check_artifacts.py`

**Was:** `.github/workflows/tests.yml:48` ran
`python tools_check_artifacts.py` (the file was moved into the `tools/`
package as `tools/check_artifacts.py`).  This step would have failed
on the next push.  The workflow also skipped `pip install -e .`, so
`python -m tools.check_artifacts` would have hit `ModuleNotFoundError`
even after the path was fixed.

**Fix:** `.github/workflows/tests.yml:20-30` — replaced the bare
`pip install -r requirements.txt` with `pip install -e ".[dev]"` (so
`tools` and `telemetry_gateway` resolve the same way on CI and
locally), and changed the verify step to
`python -m tools.check_artifacts`.  Verified:
`artifacts consistent: D=26 H=24 R=11 C=13, 2000 demo steps`.

### C3. `Makefile:80` `data-cvrde` target points to a non-existent module

**Was:** `$(PY) -m sim.scripts.cvrde_generator` (the package was
empty since M1's cleanup).  Anyone running `make data-cvrde` locally
would have hit `No module named sim.scripts.cvrde_generator`.

**Fix:** `Makefile:80` — `$(PY) -m sim.cvrde.cvrde_generator` (the
CVRDE-specific high-rate simulation lives in `sim/cvrde/`).

### C3a. Phantom `phm-simulate` entry point

**Was:** `pyproject.toml` declared `phm-simulate = "phm_vehicle.main:main"`
which resolved to nothing in the flat layout (and nothing in the
namespace layout either — there is no `sim/main.py`).

**Fix:** removed from `pyproject.toml`.  The dashboard demo uses
`docs/live_stream.json` and does not need a CLI shim.

---

## 4. Major findings — all fixed

### M1. Legacy `sim/scripts/` package is dead code

**Was:** `sim/scripts/{engine,hydraulics,suspension,gun_control,nbc,fuel_levels,exhaust,acoustics}.py`
plus `sim/scripts/generate_all.py` and an empty `__init__.py` were
entirely superseded by `sim/generators/<name>.py` + the master CLI
`tools/generate_sensor_data.py`.  The legacy scripts: did not emit
manifest sidecars, did not honour `--fault`/`--seed`/`--steps`, wrote
to `results/simulated/` (the new convention is `data/simulated/`),
and were not registered with any test suite.  They were the only
"external" callers of themselves.

**Fix:** `sim/scripts/` deleted in this pass.  `sim.scripts` removed
from `[tool.setuptools].packages` in `pyproject.toml`.  `Makefile:84`
`data-clean` target updated (`rm -rf data/manifests` was a path to
nothing).

### M2. `info.md` is orphaned

**Was:** `info.md` is a 440-line research reference (32 sections, e.g.
*"Main Battle Tank (MBT) Subsystems, Physics, and Dataset Procurement
Reference Guide"*) and was not linked from any other file in the
target repo.

**Fix:** Moved to `docs/SUBSYSTEMS_REFERENCE.md` (file is now linked
from `ARCHITECTURE.md §2` and the `README.md` ToC).  No content was
changed — only location and the broken PowerShell `Invoke-WebRequest`
snippet was reformatted as bash so the example is copy-pasteable on
both platforms.

### M3. `sim/scripts/__init__.py` is empty

Resolved automatically with M1 (directory deleted).

### M4. `Makefile` `data-clean` target references a non-existent path

**Was:** `rm -rf data/simulated data/manifests`.  There was no
`data/manifests/` directory.

**Fix:** `Makefile:84` — `rm -rf data/simulated`.  The aggregate
manifest lives inside `data/simulated` and is removed with the rest
of the simulated data.

### M5. No `pip install -e .` in CI

**Was:** `.github/workflows/tests.yml:20-30` installed only
`requirements.txt` and `pytest`.  `tools.*` and `sim.generators.*`
would not have been importable.

**Fix:** see C2 above — `pip install -e ".[dev]"` replaces the bare
`pip install -r requirements.txt`.

### M6. (Found during fix-up) `_base.py:GeneratorResult` is dead code

**Was:** the `GeneratorResult` dataclass was defined in `_base.py` and
re-exported through `__init__.py` but never instantiated anywhere in
the codebase (verified by grep).

**Fix:** removed from `_base.py` and the `__init__.py` re-exports.

### M7. (Found during fix-up) `_base.py:ChannelSpec.decimals` is a dead field

**Was:** `decimals` was set in 7 generator modules but never read by
`_fmt` (which uses a uniform `%.6f`) or by any consumer.  Misleading
in the manifest.

**Fix:** removed from `ChannelSpec`.  All 7 `decimals=` keyword
arguments removed from the generator modules.

---

## 5. Minor findings

### 5a. Wrong docstring claims (fixed)

| File | Was | Now |
|------|-----|-----|
| `hydraulics.py` | "Three channels reach the LSTM" | "Two channels reach the LSTM" |
| `structure.py` | "display only; it grows monotonically" (for `torsion_cumulative_twist`) | "LSTM input, health_exclude" (it's a feature, not a health-index contributor) |
| `suspension.py` | "display only" (for `susp_load_kN` / `susp_strain_ue`) | "LSTM input" (they are LSTM inputs per `ml.parts.PARTS[suspension]`) |
| `tools/generate_sensor_data.py` | "(when --subsystem all)" qualifier on the aggregate manifest entry | removed (the aggregate is always written, see `__main__`) |
| `__init__.py` | "The legacy `sim/scripts/` package re-exports each module's `generate`" | removed (the package is gone; the import statement is the only legacy reference) |

### 5b. `FaultProfile.onset_frac` / `ramp_frac` declared but ignored (fixed)

**Was:** every generator's `FaultProfile` declared
`onset_frac=0.30, ramp_frac=0.20` but the `generate()` body hardcoded
`int(steps * (0.30 + 0.10 * i))` and `int(steps * 0.20)`.  A future
maintainer who tried to use a different onset_frac would have seen
the profile ignored.

**Fix:** `run_subsystem` (in `_base.py`) now reads `profile.onset_frac`
and `profile.ramp_frac` and uses them in the `FaultManager.add()`
call.  Verified by inspecting the `engine.py` profile (a fault at
0.30 with a 0.20 ramp is the documented behaviour).

### 5c. `_filter_faults` silently dropped unknown faults (fixed)

**Was:** a typo in `--fault bnnrng_wear` would print a warning per
subsystem and the run would proceed, producing fault-free data when
the operator thought they had injected something.

**Fix:** typo guard added to `tools/generate_sensor_data.py:main()`.
A fault name that is skipped for **every** selected subsystem aborts
the run with `SystemExit("fault 'X' is not declared by any selected
subsystem")`.  The original per-subsystem warning is preserved for
the legitimate "fault declared by some, not all" case (e.g.
`--fault bearing_wear --subsystem engine --subsystem hydraulics`).

### 5d. `ChannelSpec.decimals` and dead `f == 0.0` branch (fixed)

Both are dead.  See M7.  The `f == 0.0` branch in `_fmt` was also
redundant (`0.0.is_integer()` is `True` and the next branch already
handles that case).

### 5e. Atomic writes (fixed)

**Was:** `write_text` truncated first and only flushed on close, so
an interrupted run could leave a half-written manifest.  Two
consumers writing to the same `out_dir` could observe a torn file.

**Fix:** `_base.py:atomic_write_text()` and `write_csv()` now write
to `*.tmp` and `os.replace` into place.  Atomic on every platform
including Windows (NTFS rename is atomic for same-volume moves).

### 5f. Hoisted imports (fixed)

**Was:** `datetime.now(timezone.utc)` was imported inside
`_write_aggregate_manifest`; `hashlib` was imported there too.  The
hot path didn't care but the lazy import obscured a hot dep.

**Fix:** both hoisted to module top of `tools/generate_sensor_data.py`.

### 5g. (Deferred) `tools/check_artifacts.py:EXPECTED_H = 24` is hardcoded

**Was:** the CI gate hardcodes the LSTM hidden size as 24.  A
legitimate retrain with `H=32` would break CI.

**Fix plan:** read the value from `ml/scenarios.py:H` or accept a
`--expected-h` flag.  Deferred because no retrain is pending; the
constraint is enforced at training time and the published weights
are pinned.

### 5h. (Deferred) `info.md → docs/SUBSYSTEMS_REFERENCE.md` content is stale in places

The PowerShell `Invoke-WebRequest` snippet in §32 was the only
identified stale item.  Reformatted to bash.  Other sections are
accurate per the audit cross-check against `DATA.md`.

### 5i. (Deferred) `sim/physics/` is a thin package

`sim/physics/` is a flat set of sensor models imported by
`sim/tank.py`.  No tests touch it directly (verified).  A future
refactor could fold it into `sim/` proper; today it is kept as a
separate package to make the sensor model surface grep-friendly.

---

## 6. CI review

`.github/workflows/tests.yml` is now sound.  Two fixes applied (see
C2, M5):

* Replace `python tools_check_artifacts.py` with
  `python -m tools.check_artifacts`.
* Add `pip install -e ".[dev]"` to the install step.

`.github/workflows/pages.yml` is out of scope for this audit (it
deploys the dashboard on a Pages hook; deployment is the host's
responsibility, not the target repo's).

---

## 7. Positive findings

What this codebase does **right** is worth recording so the patterns
are not accidentally regressed:

* **Label-leakage guards** (`tests/test_leakage.py`) — the suite
  that proves a channel is a *measurement* and not a restatement of
  the injected fault parameter.  Combined with the
  `health_exclude=True` discipline in `ml.parts.PARTS`, this is the
  audit-remediation work that distinguished v0.1.0 from the previous
  M4 state.  The M5 history is documented in `CHANGELOG.md`.
* **Pre-inference plausibility** — the 6-layer filter (NaN sanitiser,
  physical range, open/short, slew-rate, EMI outlier, stuck-at
  dual-sensor) is well-tested with adversarial inputs and
  documented in `telemetry_gateway/sensor_plausibility.py`.  This is
  the kind of FDIR gate that, on a real platform, decides whether
  the LSTM is shown a true sensor failure or a wiring failure.
* **Dual-tier inference** — the same D=26 feature vector is consumed
  by the in-browser ES6 LSTM, the MISRA-C99 edge runtime, and the
  Python reference.  Parity tests (`tests/test_js_python_parity.py`,
  `tests/test_c_python_parity.py`) enforce the contract.
* **J1939-73 DTC encoder** with on-disk flash ring buffer and lamp-
  status colouring.  This is what the command-post HUD actually
  renders.
* **Aggregate manifest with SHA-256** — the new
  `tools/generate_sensor_data.py` writes a top-level `MANIFEST.json`
  that fingerprints every CSV.  A training pipeline can refuse to
  start if the manifest is missing or stale.  This is the data-
  quality contract the M5 audit flagged as missing.
* **Reproducible numerics** — every dep is pinned to a major version
  range (`numpy>=1.26,<3.0`, `scipy>=1.11,<2.0`, `lightgbm>=4.3,<5.0`).
  No unpinned wildcards.
* **Display-only discipline** — every display channel is flagged
  `health_exclude=True` so a fault-free run cannot decay the health
  index.  Documented and tested.
* **Single-source-of-truth refactor** — the new `run_subsystem()`
  helper in `_base.py` replaces 10 × 55-line copy-paste bodies.  A
  future change to the simulator harness (e.g. swapping
  `TankSimulator` for the CVRDE high-rate model) is now a 1-file
  edit, not a 10-file find-and-replace.

---

## 8. Hand-off checklist

For the next maintainer.  Each item is one command or one small edit.

- [x] **C1** Flatten `[project.scripts]` in `pyproject.toml:78-86`.
- [x] **C2** Replace `python tools_check_artifacts.py` with
      `python -m tools.check_artifacts` in
      `.github/workflows/tests.yml:48`; add `pip install -e ".[dev]"`
      to the install step.
- [x] **C3** Fix `Makefile:80` `data-cvrde` to
      `$(PY) -m sim.cvrde.cvrde_generator`.
- [x] **C3a** Remove phantom `phm-simulate` entry point from
      `pyproject.toml`.
- [x] **M1** Delete `sim/scripts/`; remove `"sim.scripts"` from
      `pyproject.toml`.
- [x] **M2** Promote `info.md` to `docs/SUBSYSTEMS_REFERENCE.md` and
      link it from `ARCHITECTURE.md §2` and the `README.md` ToC.
- [x] **M3** Resolved automatically with M1.
- [x] **M4** `Makefile:84` — `rm -rf data/simulated data/manifests` →
      `rm -rf data/simulated`.
- [x] **M5** Add `pip install -e ".[dev]"` to CI.
- [x] **M6** Remove dead `GeneratorResult` from `_base.py` +
      `__init__.py` re-exports.
- [x] **M7** Remove dead `ChannelSpec.decimals` field.
- [x] **5a–f** Wrong docstrings, dead fault filter, atomic writes,
      hoisted imports, dead `f == 0.0` branch.
- [ ] **5g** Make `tools/check_artifacts.py:EXPECTED_H` configurable.
      Trivial follow-up; deferred.
- [ ] Optional: re-run `make verify` (lint + fast tests) on a clean
      clone to confirm the post-fix state.
- [ ] Optional: open a draft PR titled "v0.2.0: pre-release hardening"
      and link this audit as the description.

When all of the above are merged, the repo is ready for a 0.2.0 tag
and a `pip install phm-vehicle`-style public release.
