# Fault-Prediction-in-Vehicles

Physics-informed simulator for **predictive maintenance of
military battle tanks**. Every sensor is simulated from the physics-based
equations in
`Physics_Based_Sensor_Equations_Military_Tank_Preventive_Maintenance.docx`
and coupled through an evolving shared vehicle state, so that injected
faults degrade the readings in a physically consistent way.  The
resulting labelled datasets feed AI models for anomaly detection, fault
diagnosis and remaining-useful-life (RUL) prediction.

```
Sensors -> Physics-Based Features -> Vehicle State -> AI -> Anomaly / Fault / RUL
```

## Sensor families and governing physics

| # | Sensor | Physics | Key features | Associated fault |
|---|--------|---------|--------------|------------------|
| 1 | Vibration (accelerometer) | `a = d²x/dt²`, `x = A·sin(2πft+φ)` | RMS, kurtosis, FFT peaks, BPFO/BPFI | Bearing / gear wear |
| 2 | Engine/exhaust temperature | `m·cₚ·dT/dt = Q̇_gen − Q̇_cool − Q̇_exh`, RTD `R = R₀(1+αΔT)` | Temperature trend | Overheating / cooling failure |
| 3 | Oil pressure | Hagen–Poiseuille `ΔP = 8μLQ/(πr⁴)`, `μ(T)=A·e^{E/RT}` | Pressure deviation vs. speed/load/temp | Pump / leakage / blockage |
| 4 | Oil debris | Inductive `L ≈ μN²A/l`, `ṅₚ = dNₚ/dt` | Particle count & rate | Accelerated wear |
| 5 | Torque | `P = τω`, `τ = Tr/J`, `ω = 2πN/60` | Torque / power / shear | Load / efficiency degradation |
| 6 | Exhaust | `PV = nRT`, `ṁ = ρAv`, `λ` | EGT, pressure, λ, O₂ | Combustion abnormality |
| 7 | Fluid levels | `C = εA/d`, `V = πr²h` | Level / volume / capacitance | Leakage / depletion |
| 8 | Hydraulics | Pascal `P = F/A`, `P_hyd = PQ` | Pressure–flow relation | Pump / valve / seal fault |
| 9 | Suspension strain | `σ = Eε`, `ΔR/R = GF·ε` | Strain / load | Fatigue / overload |
| 10 | Torsion bar | `θ = TL/JG`, `T = JGθ/L` | Twist / torque history | Torsion-bar fatigue |
| 11 | Shock | `F = ma`, `a_RMS` | RMS / peak g | Terrain / component loading |
| 12 | Acoustic | `p(t)=P₀+A·sin(2πft)`, `SPL = 20·log₁₀(p/p_ref)` | SPL, FFT signatures | Gear / bearing noise |
| 13 | Acoustic emission | Wave equation, event features | Event rate / energy | Crack / impact / friction |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Generate a labelled dataset with injected faults:

```bash
python main.py --steps 12000 \
    --faults bearing_wear,cooling_failure,oil_pump_degradation \
    --out data/sim_dataset.csv
```

The CSV contains raw sensor readings, physics-derived features
(`vib_rms`, `vib_kurtosis`, `spl_db`, `debris_rate`, ...), a fused
`health_index` (0–100), an extrapolated `rul_steps` estimate, an
`anomaly_score`, and one-hot `fault_*` label columns.

### Available fault profiles

```
bearing_wear gear_wear cooling_failure oil_pump_degradation
bearing_clearance_wear seal_leakage fuel_injector_fault
exhaust_restriction torsion_fatigue hydraulic_valve_fault
structural_crack drivetrain_efficiency_loss
```

### Programmatic use

```python
import numpy as np
from tank_sim.config import TankConfig
from tank_sim.faults import FaultManager
from tank_sim.tank import TankSimulator
from tank_sim.dataset import write_dataset

fm = FaultManager(np.random.default_rng(0))
fm.add("bearing_wear", start_step=3000, ramp_steps=1500)
fm.add("cooling_failure", start_step=6000, ramp_steps=1500)

sim = TankSimulator(TankConfig(), faults=fm, seed=0)
records = sim.run()                 # list of dict records
write_dataset(sim, "data/run.csv")  # CSV with health features + labels
```

`run()` resets the simulator first, so calling it repeatedly (as
`write_dataset` does internally) always reproduces the same mission rather
than continuing from the previous end state.

## Health fusion

Following section 15 of the physics document, the fused health index
combines vibration RMS/kurtosis, oil temperature/pressure, debris rate,
AE activity, structural stress and lambda into a single 0–100 score;
`rul_steps` is obtained by linear extrapolation of the health-index
trajectory to the failure threshold (section 16).

Health is anchored to the documented thresholds: a subsystem whose worst
parameter sits exactly at its critical threshold scores `FAIL_HEALTH`, so the
health curve and the threshold alarms cannot disagree. Deviation is one-sided
(being better than the healthy reference is never penalised), and monotonically
accumulating counters (total debris, cumulative twist) and consumables (fuel
level) are excluded from scoring — including them made health a proxy for
elapsed mission time rather than condition. `overall` is fused from the
subsystem healths rather than scored from its own parameters.

Load-driven channels (shaft torque, road-wheel load, shock RMS) are shown but
excluded from health, since duty is not degradation. The condition indicators
are the load-normalised ones: `susp_compliance` (strain per unit load) and
`driveline_efficiency`.

## Live dashboard (GitHub Pages)

A self-contained dashboard is published at
`https://sheenapravin.github.io/Fault-Prediction-in-Vehicles/`
(source in `docs/`, deployed automatically by
`.github/workflows/pages.yml` on every push).

- **Live feed** — streams physics-simulated telemetry in real time with
  play/pause and 1–10× speed control.
- **Clickable modules** — select any subsystem card (engine, powertrain,
  lubrication, cooling, hydraulics, suspension, structure) to open a
  detail view with live parameter trends, per-module LSTM RUL and its
  detection history.
- **Failure-mode log** — every warning/critical threshold crossing,
  health-failure crossing and AI fault-mode classification is timestamped
  with mission time.
- **Per-module RUL** — a pure-JS LSTM forward pass (`docs/lstm.js`) runs
  the trained weights (`docs/model.json`) in the browser.

Regenerate model + stream after changing the simulator or training:

```bash
python -m ml.train --epochs 28 --stride 8 --demo-steps 3000
```

## Validation

The model is trained on simulated data, so what it scores and on what has to be
stated precisely.

**Split.** Scenarios are grouped into (fault family x duty profile) cells across
five duty profiles (road march, cross country, assault, convoy idle, recovery
tow). A whole cell goes to train, validation or test; no cell straddles a split,
so a held-out scenario has no near-duplicate sibling in training. Every fault
class is present in all three splits -- holding out a class entirely would make
its accuracy vacuously zero rather than informative. The feature scaler is
fitted on the training scenarios only.

**Reported figures** (`python -m ml.train`):

| Split | RUL MAE | Fault-class accuracy |
|---|---|---|
| validation (selected the checkpoint) | 221 steps | 0.850 |
| **test** (scored once, after selection) | **213 steps** | **0.854** |

Only the test row is a generalisation estimate. Validation drove checkpoint
selection, so quoting it would be quoting a training statistic.

Per-class accuracy is not uniform, and the weak cells are real diagnostic gaps
rather than noise: `gear_wear` under a low-load duty cycle scores 0.31-0.58,
because the mesh-sideband signature is not separable when the drivetrain is
barely loaded. `exhaust_restriction` (0.42-0.75) and `cooling_failure`
(0.51-0.74) are similar. Bearing, seal-leak, structural-crack and
torsion-fatigue cells score 0.96-0.99.

**What this does not demonstrate.** Generalisation across duty cycle and fault
onset, within one simulator and one vehicle. It says nothing about transfer to
real hardware. The only sim-to-real evidence in this project is the real-corpus
benchmark in `../benchmark/evaluate_subsystems.py` (ZeMA hydraulic rig,
MetroPT-3 APU, Scania APS, naval gas turbine, SCANIA Component X).

### Label leakage

Nine channels were previously published as noiseless readbacks of the injected
fault severity rather than as measurements -- `driveline_efficiency` was
literally `max(1 - 0.18*s, 0.7)`, so a depth-1 threshold on it scored 0.9988 on
its own fault class. `debris_rate` and `ae_event_rate` also fed the health
index, so the leak reached the RUL targets and the regression was circular.

`tests/test_leakage.py` guards this with a general invariant rather than a
blocklist: hold the fault trajectory fixed, re-roll only the measurement noise,
and every emitted channel must move. A channel identical across two independent
noise realisations is not a measurement -- it is whichever parameter produced
it, published under a sensor's name. That catches the next one before it ships.

## Testing

```bash
python -m unittest discover -s tests -v
```

## License

Educational / research use for the IDP project.