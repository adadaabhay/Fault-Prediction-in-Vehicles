# Defense-Grade PHM/CBM+ Platform for Armored Fighting Vehicles

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![C99 Embedded](https://img.shields.io/badge/C99-MISRA--Informed-green.svg)](c_engine/CODING_STANDARD.md)
[![ISO 13374](https://img.shields.io/badge/Architecture-ISO%2013374%20%2F%20OSA--CBM-orange.svg)](PROJECT.md)
[![SAE J1939](https://img.shields.io/badge/Diagnostics-SAE%20J1939--71%2F73-red.svg)](telemetry_gateway/dtc_engine.py)
[![Verification](https://img.shields.io/badge/Tests-390%20Passed-brightgreen.svg)](tests/)

An end-to-end Condition-Based Maintenance Plus (**CBM+**) and Integrated Vehicle Health Management (**IVHM**) platform engineered for Heavy Armored Fighting Vehicles and Main Battle Tanks (e.g., CVRDE Arjun Mk-1A, T-90S Bhishma, Zorawar Light Tank). 

The platform couples first-principles digital-twin physics with deterministic sensor plausibility filtering (FDIR), zero-allocation embedded C99 neural inference, SAE J1939-73 Diagnostic Trouble Code (DTC) generation, and real-time WebSocket streaming to an interactive tactical HUD.

---

## 1. System Architecture (ISO 13374 / OSA-CBM)

The system implements the full 6-layer ISO 13374 condition-monitoring specification:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 ISO 13374 / OSA-CBM 6-LAYER PIPELINE                             │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
 1. DATA ACQUISITION       UDP 9000 Socket · Serial COM (115200 baud) · REST `/api/telemetry/push`
    (telemetry_gateway/)   Digital Twin Physics Simulator (`tank_sim/`) & CVRDE Combat Missions
                                                 │
                                                 ▼
 2. DATA MANIPULATION      58-Channel FDIR Plausibility Gate (`sensor_plausibility.py`)
    (FDIR Pre-Filter)      • Physical bounds clamping       • Open/short circuit isolation (-999 bar)
                           • Hampel EMI spike filter        • Stuck-at dwell timeout & Slew limiter
                                                 │
                                                 ▼
 3. STATE DETECTION        SAE J1939-73 DTC Engine (`dtc_engine.py`)
    (DTC Engine)           • SPN/FMI translation            • PGN 65226 DM1 Active Malfunctions
                           • PGN 65227 DM2 Historic Log     • Flash circular ring buffer persistence
                                                 │
                                                 ▼
 4. HEALTH ASSESSMENT      Subsystem Health Index (`ml/parts.py` / `pipeline.py`)
    (Health Matrix)        • One-sided physical deviation   • Failure threshold anchor (25 FAIL_HEALTH)
                           • Logarithmic rate normalization • Driveline & compliance load scaling
                                                 │
                                                 ▼
 5. PROGNOSTICS (RUL)      Dual-Tier Neural Inference:
    (Neural Inference)     • Embedded C99 Edge Engine: Zero-allocation, <32 KB SRAM (`c_engine/`)
                           • Python Training & Export: Grouped 3-way split by duty cell (`ml/`)
                           • In-Browser Inference: Pure ES6 Matrix Forward Pass (`docs/lstm.js`)
                                                 │
                                                 ▼
 6. ADVISORY & TACTICAL    Real-Time Command HUD (`docs/index.html`)
    (Command HUD)          • 270° Radial Canvas Gauges      • Subsystem Drill-Down Health Modals
                           • Active DM1 Fault Chips         • 2.0s Watchdog Fallback to Flight Replay
```

---

## 2. Vehicle Subsystem Map & Governing Physics

The platform models 13 coupled mechanical, hydraulic, thermal, and electrical subsystems:

| # | Subsystem | Physical Sensors Simulated | Governing Equations & Physical Laws | Module Path |
|---|---|---|---|---|
| **1** | **Main Powerpack** | Coolant RTD Pt100, Type-K Thermocouple, Hot-film Mass Airflow, Turbo Boost, Dual EGT Pyrometry | • Lumped thermal ODE: $mc_p \frac{dT}{dt} = \dot{Q}_{gen} - \dot{Q}_{cool} - \dot{Q}_{exh}$<br>• Diesel $\lambda$ residual: $1.4\text{--}5.0$<br>• Ideal gas combustion: $PV = nRT$, $\dot{m} = \rho Av$ | [`tank_sim/physics/temperature.py`](tank_sim/physics/temperature.py)<br>[`tank_sim/physics/exhaust.py`](tank_sim/physics/exhaust.py)<br>[`tank_sim/cvrde/powerpack.py`](tank_sim/cvrde/powerpack.py) |
| **2** | **Transmission & Final Drive** | Piezoelectric Accelerometers (RMS, Kurtosis, FFT), Shaft Torque Strain Bridges | • Fundamental shaft frequency: $f_r = N/60$<br>• Gear-mesh sidebands: $f_{GMF} = N_p Z_p/60 \pm f_r$<br>• Bearing defect harmonics: $BPFO / BPFI$<br>• Driveline efficiency: $\eta = \tau_{sprocket}/(\tau_{shaft}\cdot i)$ | [`tank_sim/physics/vibration.py`](tank_sim/physics/vibration.py)<br>[`tank_sim/physics/torque.py`](tank_sim/physics/torque.py) |
| **3** | **Lubrication System** | Main gallery pressure transducer, Sump temperature probe, Inductive wear debris coil | • Hagen-Poiseuille laminar drop: $\Delta P = \frac{8\mu(T)LQ}{\pi r^4}$<br>• Arrhenius viscosity: $\mu(T) = A \exp(\frac{E_a}{RT})$<br>• Metal particle count: $L \sim \frac{\mu N^2 A}{l}$ + Poisson hits | [`tank_sim/physics/oil.py`](tank_sim/physics/oil.py) |
| **4** | **Cooling System** | Radiator RTD probe, Expansion tank capacitive volume | • Thermostat opening hysteresis ($82\text{--}95^\circ\text{C}$)<br>• Convective core dissipation<br>• Dielectric capacitance: $C = \frac{\epsilon A}{d}$ | [`tank_sim/physics/temperature.py`](tank_sim/physics/temperature.py)<br>[`tank_sim/physics/level.py`](tank_sim/physics/level.py) |
| **5** | **Hydraulics & Turret** | 210–300 bar circuit transducers, Servo flowmeters, Actuator LVDTs | • Pascal's law: $P = F/A$, $P_{hyd} = PQ$<br>• Orifice seal leak: $Q_{leak} = C_d A_{leak}\sqrt{\frac{2\Delta P}{\rho}}$<br>• Azimuth PMSM motor current draw | [`tank_sim/physics/hydraulics.py`](tank_sim/physics/hydraulics.py)<br>[`tank_sim/cvrde/gun_control.py`](tank_sim/cvrde/gun_control.py) |
| **6** | **Suspension & Running Gear** | Hydrogas Suspension Units (HSU), Torsion bar strain gauges, Shock accelerometers | • Adiabatic gas spring: $P V^\gamma = \text{const}$<br>• Torsion bar twist: $\theta = \frac{TL}{JG}$, $\tau = \frac{Tr}{J}$<br>• Wheatstone bridge compliance: $\frac{\Delta R}{R} = GF\cdot\epsilon$<br>• ISO 8608 cross-country vertical shock: $a_{RMS}$ | [`tank_sim/physics/suspension.py`](tank_sim/physics/suspension.py)<br>[`tank_sim/cvrde/hydrogas_suspension.py`](tank_sim/cvrde/hydrogas_suspension.py) |
| **7** | **120mm Gun & Hull Structure** | Piezoelectric Acoustic Emission (AE) burst sensors, Recoil buffer stroke LVDT | • Acoustic Emission wave propagation: $\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$<br>• Paris-Erdogan crack fatigue: $\frac{da}{dN} = C(\Delta K)^m$<br>• 120mm Equivalent Full Charge (EFC) shot wear | [`tank_sim/physics/acoustics.py`](tank_sim/physics/acoustics.py)<br>[`tank_sim/cvrde/gun_control.py`](tank_sim/cvrde/gun_control.py) |
| **8** | **Auxiliary Power & NBC** | APU 28V DC bus voltmeter/ammeter, NBC differential pressure gauge | • 8.5 kW APU generator electric load balance<br>• Positive cabin barrier ($+500\text{ Pa}$)<br>• Darcy's law filter dust load: $\Delta P = \frac{1}{2}\rho v^2 \zeta$ | [`tank_sim/cvrde/auxiliary_nbc.py`](tank_sim/cvrde/auxiliary_nbc.py)<br>[`pipelines/apu_metropt.py`](pipelines/apu_metropt.py) |

---

## 3. Directory Layout

```
Fault-Prediction-in-Vehicles/
├── tank_sim/               # Physics Simulator & Digital Twin
│   ├── physics/            # 9 sensor physics modules (temperature, vibration, oil, etc.)
│   ├── cvrde/              # CVRDE Arjun Mk-1A subsystem modules (powerpack, HSU, gun)
│   ├── tank.py             # Central multi-subsystem simulation orchestrator
│   ├── faults.py           # 12 physical fault injection profiles
│   └── dataset.py          # CSV dataset exporter
├── telemetry_gateway/      # Ingestion, FDIR Plausibility Gate & J1939 Engine
│   ├── server.py           # FastAPI WebSocket & REST streaming server
│   ├── sensor_plausibility.py # 58-channel pre-inference FDIR plausibility gate
│   ├── dtc_engine.py       # SAE J1939-73 DM1/DM2 Diagnostic Trouble Code engine
│   ├── live_sensor_ingest.py # UDP 9000 & Serial COM 115200 hardware ingestion
│   └── pipeline.py         # End-to-end ISO 13374 pipeline composition
├── c_engine/               # MISRA-C:2012 Informed C99 Edge Runtime
│   ├── tank_pdm_infer.c    # Zero-allocation inference (<32 KB SRAM, fail-safe NaN trap)
│   ├── tank_pdm_infer.h    # C API interface
│   ├── binding.py          # Python ctypes bindings
│   ├── build.py            # Automatic compiler discovery & shared library builder
│   └── CODING_STANDARD.md  # MISRA compliance and static safety verification
├── ml/                     # Machine Learning & Scenario Generator
│   ├── lstm.py             # Pure NumPy dual-head LSTM (RUL regression + Fault classification)
│   ├── scenarios.py        # Grouped 3-way dataset splitting (fault x duty profile)
│   ├── parts.py            # Subsystem health index formulation & threshold tables
│   └── train.py            # Training pipeline with checkpoint selection
├── pipelines/              # Real-World Empirical Subsystem Pipelines
│   ├── zema_hydraulics.py  # ZeMA Hydraulic Rig (UCI 447)
│   ├── apu_metropt.py      # MetroPT-3 Train APU (UCI 791)
│   ├── scania_aps.py       # Scania APS Heavy Trucks (UCI 421)
│   └── naval_gasturbine.py # Naval Vessel Propulsion Plant (UCI 316)
├── benchmark/              # Multi-Corpus Evaluation Harness
│   └── evaluate_subsystems.py # Cross-corpus benchmark with leakage guards
├── docs/                   # Tactical Command HUD & Web Frontend
│   ├── index.html          # Real-time operational dashboard
│   ├── dashboard.js        # UI gauge updater, DM1 fault chips, modal views
│   ├── live.js             # WebSocket streaming client with 2.0s watchdog
│   ├── lstm.js             # ES6 in-browser neural forward pass
│   └── model.json          # Shipped dual-head neural network weights
├── tests/                  # Verification Harness (390+ Automated Tests)
├── PROJECT.md              # Complete ISO 13374 & SAE J1939 architecture guide
├── info.md                 # Detailed physical equations & military gap roadmap
├── TEST_INFRA.md           # Category-Partition & boundary test matrix
└── requirements.txt        # Pinned production runtime dependencies
```

---

## 4. Quickstart Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/adadaabhay/Fault-Prediction-in-Vehicles.git
cd Fault-Prediction-in-Vehicles

# Create virtual environment and install pinned dependencies
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate Simulated Multi-Sensor Telemetry

Generate 12,000 discrete 10 Hz time-steps of 58-channel vehicle telemetry with injected bearing wear, cooling failure, and oil pump degradation:

```bash
python main.py --steps 12000 \
    --faults bearing_wear,cooling_failure,oil_pump_degradation \
    --out data/sim_dataset.csv
```

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