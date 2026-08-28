# Main Battle Tank (MBT) Subsystems, Physics, and Dataset Procurement Reference Guide

> **Platform Architecture Reference**: Heavy Tracked Armored Combat Vehicle (58–68 Metric Tons)  
> **Comparative Platforms**: M1A2 Abrams (USA), Leopard 2A7 (Germany), Arjun Mk1A (India)  
> **Target Framework**: Condition-Based Maintenance Plus (CBM+) & Prognostic Health Management (PHM)

---

## 1. Operational Environment, Terrain Features & Mission Cycles

Modern Main Battle Tanks operate under extreme thermal, mechanical, and dynamic environmental stressors. Prognostic models must account for real-world excitation profiles rather than assuming static operating points.

```
                                  +-------------------------------------------------------------+
                                  |                 MBT OPERATIONAL STRESS SPECTRUM             |
                                  +-------------------------------------------------------------+
                                                                 │
    ┌─────────────────────────────┼──────────────────────────────┴─────────────────────────────┬─────────────────────────────┐
    ▼                             ▼                                                             ▼                             ▼
[1. Terrain Roughness]       [2. Mission Duty Cycles]                                      [3. Ballistic & Shock]        [4. Environmental Extremes]
• ISO 8608 Class C–G         • Idle / Silent Watch (Low RPM / Battery)                     • 120mm Gun Recoil (350 kN)   • Ambient: -40°C to +55°C
• Severe Cross-Country Rocks • Road Cruise (1200–1600 RPM, 45% Load)                       • ATGM Impact Acceleration    • Desert Silica Dust Ingress
• Mud / Soft Soil Sinking    • Tactical Sprint (2200–2600 RPM, 95% Load)                   • Anti-Tank Mine Belly Blast  • Fording / Water Submergence
• Obstacle / Trench Impacts  • High-G Pivot Turns (Steering Clutch Slip)                   • Roadwheel Obstacle Slap     • Tactical IR Signature Rise
```

### 1.1 Terrain Roughness Classification & Dynamic Shock (ISO 8608)
Combat vehicle suspension and drivetrain fatigue are directly driven by terrain spatial frequency:
$$G_d(n) = G_d(n_0) \cdot \left(\frac{n}{n_0}\right)^{-w}$$
* $n$: Spatial frequency ($\text{cycles/m}$).
* $G_d(n_0)$: Terrain roughness coefficient (ISO 8608 geometric mean values: Class A paved highway $\approx 16 \times 10^{-6}\text{ m}^3$, Class B $\approx 64 \times 10^{-6}\text{ m}^3$, Class C $\approx 256 \times 10^{-6}\text{ m}^3$, Class D $\approx 1024 \times 10^{-6}\text{ m}^3$, Class E severe cross-country $\approx 4096 \times 10^{-6}\text{ m}^3$, Class F extreme rough terrain $\approx 16384 \times 10^{-6}\text{ m}^3$).
* Dynamic vertical shock acceleration severity is quantified via RMS acceleration:
  $$a_{RMS} = \sqrt{\frac{1}{T}\int_0^T a_z^2(t) \, dt}$$

### 1.2 Mission Duty Profiles
* **Silent Watch**: Engine off, APU or 24V battery bank driving active sights, radar, and radios (high electrical discharge rate, low mechanical strain).
* **Road Cruise**: Steady-state power delivery ($1400\text{–}1600\text{ RPM}$, nominal lubrication pressure, laminar thermal flow).
* **Tactical Sprint**: Maximum governor speed ($2400\text{–}2600\text{ RPM}$), turbocharger full boost, high thermal generation rate $\dot{Q}_{gen}$.
* **High-G Pivot Turns**: One track locked or reversed, transmitting extreme instantaneous torque through cross-drive differential and planetary gearsets.
* **Hot Soak**: Engine shutdown immediately following sprint, halting coolant circulation while residual heat dissipates into engine block and turbocharger bearings.

---

## 2. First-Principles Governing Physics Formulations

Every sensor channel in an MBT CBM+ framework is governed by fundamental mechanics, thermodynamics, fluid dynamics, and wave kinematics:

| Physical Domain | Governing Equation / Analytical Formulation | Monitored Subsystem & Fault Mode |
| :--- | :--- | :--- |
| **Mechanics & Vibration** | $a = \frac{d^2x}{dt^2}$, $BPFO = \frac{n}{2}f_s(1 - \frac{d}{D}\cos\alpha)$ | Final Drive & Transmission Bearing Spalls |
| **Structural Torsion** | $\theta = \frac{TL}{JG}$, $\tau_{shear} = \frac{Tr}{J}$, $J = \frac{\pi r^4}{2}$ | Suspension Torsion Bars, Main Drive Shaft |
| **Hooke's Law Strain** | $\sigma = E\epsilon$, $\frac{\Delta R}{R} = GF \cdot \epsilon$ | Hull Chassis Seams, Roadwheel Spindles |
| **Lumped Thermal Balance** | $m c_p \frac{dT}{dt} = \dot{Q}_{gen} - \dot{Q}_{cool} - \dot{Q}_{exh}$ | Engine Block, Turbocharger, Oil Coolers |
| **RTD Temperature Sensor** | $R(T) = R_0 [1 + \alpha (T - T_0)]$ | Coolant & Cylinder Head Temperature |
| **Fluid Dynamics (Laminar)** | $\Delta P = \frac{8\mu(T)LQ}{\pi r^4}$ [Hagen-Poiseuille] | Engine Oil Gallery, Filter Blockage |
| **Arrhenius Viscosity** | $\mu(T) = A \exp(\frac{E_a}{RT})$ | Lubrication Breakdown, Thermal Sludge |
| **Pascal Hydraulic Power** | $P = \frac{F}{A}$, $P_{hyd} = PQ$, $Q_{leak} = C_d A_{leak}\sqrt{\frac{2\Delta P}{\rho}}$ | Turret Traverse, Elevation, Recoil Buffer |
| **Ideal Gas Combustion** | $PV = nRT$, $\dot{m} = \rho Av$, $\lambda = \frac{(A/F)_{actual}}{(A/F)_{stoich}}$ | Exhaust Manifold, Turbo Lag, EGT Bias |
| **Inductive Wear Debris** | $L \approx \frac{\mu_r \mu_0 N^2 A}{l}$, $\dot{N}_p = \frac{dN_p}{dt}$ | Ferromagnetic Metallic Gear Tooth Debris |
| **Capacitive Liquid Level** | $C = \frac{\epsilon_r \epsilon_0 A}{d}$, $V = \pi r^2 h$ | Fuel Cell, Oil Sump, Coolant Reservoir |
| **Acoustic Sound Pressure** | $SPL = 20\log_{10}(\frac{p_{RMS}}{20\ \mu\text{Pa}})$ | Cabin Noise, Bearing/Gear Whine |
| **Acoustic Emission (AE)** | $\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$, $E = \int v^2(t) dt$ | Micro-Crack Propagation, Armor Fracture |
| **Fatigue Crack & Barrel Wear** | $\frac{da}{dN} = C (\Delta K)^m$ [Paris-Erdogan], $EFC = \sum N_i w_i$ | Main Gun Barrel, Planetary Gear Teeth |
| **Electrochemical Battery** | $V_{term} = V_{oc}(SoC) - I \cdot R_{int}(SoH, T)$ | 24V DC Vehicle Battery State of Health |

---

## 3. Subsystem Hierarchy, Dataset Procurement & Confidentiality Matrix

```
Legend:
  • Status:
      - [OPEN PROXY PROCURED]        : Publicly downloadable dataset directly representing subsystem physics.
      - [CONFIDENTIAL / MILITARY GAP]: Classified/restricted military data; addressed via physics simulation or proxy.
      - [HYBRID PROXY / CV]          : Real physical baseline or computer vision ground-truth dataset.
  • Provenance:
      - 🟢 Real In-Service Fleet     : Recorded from vehicles/vessels in active daily operation with natural failures.
      - 🔵 Real Physical Test Rig    : Recorded on physical hardware/motor/bearing/hydraulic testbeds with real sensors.
      - 🟡 Hybrid (Bench + CFD/Sim)  : Real test bench telemetry paired with matched CFD or mission profiles.
      - 🔴 Synthetic Simulation      : First-principles physics simulation, numerical ODEs, or Simulink models.
  • Priority:
      - Priority 1 (Core)            : Fundamental to primary mission availability & daily operations.
      - Priority 2 (Secondary)       : High-fidelity subsystem training & operational regime modulation.
      - Priority 3 (Auxiliary)       : Specialized electrical, benchmark, or computer vision data.
      - Priority 4 (Synthetic/Sim)   : Mathematical digital twin / physics simulation required.
```

---

### System 1: Powerplant System
*Generates primary motive mechanical power and electrical generation.*

#### Subsystems & Components:
* **1.1 Main Engine (Diesel / Gas Turbine)**: 1500 hp MTU MB 873 Ka-501 V12 twin-turbo diesel (Leopard 2), 1400 hp MTU 838 / DATRAN 1500 diesel (Arjun Mk1A), 1500 hp Lycoming AGT1500 gas turbine (M1 Abrams).
* **1.2 Fuel Delivery & Injection**: Armored self-sealing fuel cells, common-rail injection pumps, fuel/water separators.
* **1.3 Auxiliary Power Unit (APU)**: Under-armor APU generator (6–10 kW, 28V DC) powering systems during silent watch.
* **1.4 Cooling & Thermal Dissipation**: Radiator cores, hydraulic-driven suction cooling fans, oil-to-water heat exchangers.
* **1.5 Air Induction & Exhaust**: Cyclone pre-cleaners, HEPA engine air filters, exhaust manifolds, infrared (IR) smoke ejectors.

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **1.1 Engine Core** | `OPEN PROXY PROCURED` | Combustion imbalance, injector fouling, boost loss, EGT deltas.<br>$m c_p \frac{dT}{dt} = \dot{Q}_{gen} - \dot{Q}_{loss}$, $PV=nRT$ | **Deutz TCD 12.0 V6 Diesel Air Path**<br>**Naval Vessel Propulsion Plant**<br>**Diesel Engine Original Dataset**<br>**Diesel Engine Faults Features**<br>**NASA C-MAPSS / N-CMAPSS** | 🟡 Hybrid (Bench+CFD)<br>🟡 Hybrid (Bench+Sim)<br>🟢 Real In-Service<br>🔴 Synthetic Sim<br>🔴 Sim / 🟡 Hybrid | **P1 (Core)**<br>**P1 (Core)**<br>**P1 (Core)**<br>**P2**<br>**P3** | • [Deutz Zenodo #5766940](https://zenodo.org/records/5766940): Transient NRTC non-road cycles.<br>• [Naval Propulsion UCI](https://archive.ics.uci.edu/dataset/316/condition+based+maintenance+of+naval+propulsion+plants): Gas turbine decay for M1 Abrams.<br>• [Marine Diesel Mendeley](https://data.mendeley.com/datasets/p92gj2732w/2): Multi-cylinder EGT deltas.<br>• [Diesel Faults IEEE](https://ieee-dataport.org/documents/diesel-engine-faults-features-dataset-default): 3,500 severity sweeps.<br>• [NASA C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data): Turbofan RUL baseline. |
| **1.2 Fuel System** | `OPEN PROXY PROCURED` | Fuel pump cavitation, filter clogging, fuel level capacitance.<br>$C = \frac{\epsilon A}{d}$, $\Delta P = \frac{8\mu LQ}{\pi r^4}$ | **SCANIA DT-CARGO**<br>**In-house `level.py`** | 🟢 Real In-Service<br>🔴 Synthetic Sim | **P2**<br>**P4 (Sim)** | • [DT-CARGO Zenodo #7599687](https://zenodo.org/records/7599687): Dynamic fuel burn across 1.26M km.<br>• `level.py`: Dielectric capacitance level. |
| **1.3 APU** | `OPEN PROXY PROCURED` | Turbine/generator thermal fatigue, motor current spikes.<br>$P = VI$, $SPL = 20\log_{10}(\frac{p}{p_{ref}})$ | **MetroPT-3 Train APU**<br>**SCANIA Component X** | 🟢 Real In-Service<br>🟢 Real In-Service | **P1 (Core)**<br>**P1 (Core)** | • [MetroPT-3 UCI #791](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset): 11M rows of continuous 1 Hz APU breakdown data.<br>• [SCANIA Comp X](https://researchdata.se/en/catalogue/dataset/2024-34): Real fleet time-to-event failures. |
| **1.4 Cooling System** | `OPEN PROXY PROCURED` | Radiator clogging, pump seal leak, oil viscosity decay.<br>$\mu(T) = A e^{\frac{E_a}{RT}}$, $\Delta P = \frac{8\mu LQ}{\pi r^4}$ | **ZeMA Hydraulic (Cooler)**<br>**In-house `temperature.py`** | 🔵 Real Hardware Rig<br>🔴 Synthetic Sim | **P1 (Core)**<br>**P4 (Sim)** | • [ZeMA UCI #447](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems): Quantitative cooler decay ($100\% \to 3\%$).<br>• `temperature.py`: Dynamic lumped thermal balance. |
| **1.5 Air & Exhaust** | `OPEN PROXY PROCURED` | Pre-cleaner dust loading, exhaust manifold restriction.<br>$\dot{m} = \rho Av$, $\lambda = \frac{(A/F)_{act}}{(A/F)_{stoich}}$ | **Deutz TCD 12.0 V6 Air Path**<br>**APS Failure at Scania Trucks** | 🟡 Hybrid (Bench+CFD)<br>🟢 Real In-Service | **P1 (Core)**<br>**P1 (Core)** | • [Deutz Zenodo #5766940](https://zenodo.org/records/5766940): Exhaust pressure/temp transients.<br>• [Scania APS UCI #421](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks): Air intake and compressor failure logs. |

---

### System 2: Drivetrain System
*Transmits high torque from engine to drive sprockets and enables differential steering.*

#### Subsystems & Components:
* **2.1 Main Cross-Drive Transmission**: Allison X-1100-3B (Abrams), Renk HSWL 354 (Leopard 2), DATRAN (Arjun).
* **2.2 Final Drives**: Heavy planetary gear reduction units inside armored sponsons.
* **2.3 Steering & Service Brakes**: Hydrostatic/hydrokinetic differential steering clutches, multi-disc wet brakes.
* **2.4 Power Take-Off (PTO)**: Auxiliary drives powering high-pressure hydraulic pumps and generators.

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **2.1 Transmission** | `OPEN PROXY PROCURED` | Planetary gear pitting, clutch pack slippage, bearing spalling.<br>$BPFO/BPFI$, $GMF = Z \cdot f_s$, Kurtosis $> 8.0$ | **Univ. of Ottawa Time-Varying Bearings**<br>**Multi-Mode Gearbox Variable**<br>**PHM 2023 Gearbox Challenge**<br>**PHM 2026 Spur Gear Challenge** | 🔵 Real Hardware Rig<br>🔵 Real Hardware Rig<br>🔵 Real Hardware Rig<br>🔵 Real Hardware Rig | **P1 (Core)**<br>**P1 (Core)**<br>**P2**<br>**P2** | • [Ottawa Mendeley](https://data.mendeley.com/datasets/v43hmbwxpm/2): Time-varying speed vibration for non-stationary order tracking.<br>• [Multi-Mode Mendeley](https://data.mendeley.com/datasets/p92gj2732w/2): Multi-speed/load gearbox pitting.<br>• [PHM 2023](https://data.phmsociety.org/phm2023-conference-data-challenge/): 11 severity levels.<br>• [PHM 2026](https://data.phmsociety.org/phm-north-america-2026-conference-data-challenge/): Accelerated run-to-failure. |
| **2.2 Final Drives** | `OPEN PROXY PROCURED` | Casing bearing fatigue, reduction tooth shearing, metallic debris.<br>$\tau_{shear} = \frac{Tr}{J}$, $L \approx \frac{\mu N^2 A}{l}$, $\dot{N}_p = \frac{dN_p}{dt}$ | **NASA IMS Bearing Dataset**<br>**FEMTO-ST PRONOSTIA**<br>**Paderborn Bearing Dataset** | 🔵 Real Hardware Rig<br>🔵 Real Hardware Rig<br>🔵 Real Hardware Rig | **P2**<br>**P2**<br>**P2** | • [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/): Natural run-to-failure bearing spalling.<br>• [FEMTO-ST GitHub](https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset): Variable unit lifetimes.<br>• [Uni Paderborn](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download): 64 kHz vibration. |
| **2.3 Steering & Brakes** | `OPEN PROXY PROCURED` | Steering clutch overheating, hydraulic pressure drop.<br>$P = \frac{F}{A}$, $W_{fric} = \mu F_N v$ | **ZeMA Hydraulic (Valve/Pump)**<br>**Bosch Hydraulic EoL** | 🔵 Real Hardware Rig<br>🔵 Real Hardware Rig | **P1 (Core)**<br>**P3** | • [ZeMA UCI #447](https://archive.ics.uci.edu/dataset/447/): Internal valve leakage & pump pressure loss.<br>• [Bosch GitHub](https://github.com/boschresearch/Hydraulic-EoL-Testing): Pump control cycles. |
| **2.4 Power Take-Off (PTO)**| `OPEN PROXY PROCURED` | Auxiliary shaft shear, bearing misalignment, pump load decay.<br>$P_{mech} = \tau \omega$ | **CWRU Bearing Data Center**<br>**Univ. of Ottawa Motor** | 🔵 Real Hardware Rig<br>🔵 Real Hardware Rig | **P3**<br>**P3** | • [CWRU Bearings](https://engineering.case.edu/bearingdatacenter/download-data-file): Baseline vibration physics.<br>• [Ottawa Motor Mendeley](https://data.mendeley.com/datasets/msxs4vj48g): Drive motor vibration. |

---

### System 3: Running Gear & Track Assembly
*Converts power to traction, supports 58+ tons, and absorbs cross-country shock.*

#### Subsystems & Components:
* **3.1 Suspension Assembly**: Torsion bars with rotary dampers (Leopard 2), hydropneumatic suspension units (Arjun Mk1A).
* **3.2 Road Wheels**: 7 dual road wheels per side with solid-rubber tires.
* **3.3 Idler Wheels & Tensioners**: Front track-tensioning idler wheels with hydraulic recoil cylinders.
* **3.4 Drive Sprockets**: Rear-mounted toothed drive sprockets meshing with track end-connectors.
* **3.5 Return Rollers & Track Assemblies**: Double-pin continuous steel tracks with replaceable rubber pads (Diehl 570F / L&T tracks).

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **3.1 Suspension (HSU/Torsion)** | `HYBRID PROXY / PARTIAL GAP` | Torsion bar fatigue twist, hydropneumatic damper cavitation, nitrogen seal leakage.<br>$\theta = \frac{TL}{JG}$, $a_{RMS} = \sqrt{\frac{1}{T}\int a^2 dt}$ | **TartanDrive 2.0 Off-Road**<br>**In-house `suspension.py`** | 🟢 Real In-Service<br>🔴 Synthetic Sim | **P1 (Core)**<br>**P4 (Sim)** | • [TartanDrive GitHub](https://github.com/castacks/tartan_drive_2.0): 7 hours of real off-road suspension travel & IMU terrain excitation.<br>• `suspension.py`: Torsion stiffness degradation. |
| **3.2 Road Wheels** | `HYBRID PROXY / CV` | Rubber delamination, rim hub bearing fatigue, metal rim spalling.<br>Rolling contact fatigue, $\sigma = E\epsilon$ | **FaultSeg Wheel Defect Dataset**<br>**Paderborn Bearings** | 🟢 Real In-Service (CV)<br>🔵 Real Hardware Rig | **P2**<br>**P2** | • [FaultSeg Zenodo #13162335](https://zenodo.org/records/13162335): 829 annotated images of wheel shelling, spalling, and cracks (Computer Vision). |
| **3.3 Idler Wheels** | `CONFIDENTIAL / GAP` | Track tensioner cylinder pressure leak, idler bearing seizure.<br>Pascal $P = \frac{F}{A}$ | **ZeMA Hydraulic (Accumulator)** | 🔵 Real Hardware Rig | **P2** | • [ZeMA UCI #447](https://archive.ics.uci.edu/dataset/447/): Accumulator pressure drop models track tensioner bleed-down. |
| **3.4 Drive Sprockets** | `CONFIDENTIAL / GAP` | Sprocket tooth flank wear, tooth shearing, pitch mismatch.<br>Tachometer vs. INS slip ratio | **Multi-Mode Gearbox (Tooth)**<br>**In-house `tank.py`** | 🔵 Real Hardware Rig<br>🔴 Synthetic Sim | **P2**<br>**P4 (Sim)** | • [Multi-Mode Mendeley](https://data.mendeley.com/datasets/p92gj2732w/2): Sprocket meshing harmonics.<br>• `tank.py`: Velocity differential slip. |
| **3.5 Tracks (Double-Pin)** | `CONFIDENTIAL / MILITARY GAP` | Track pin bushing galling, track horn wear, track shedding.<br>Tribological dry sliding wear | **FaultSeg (Visual inspection)**<br>**Tracked Heavy Excavator**<br>**In-house digital twin** | 🟢 Real In-Service (CV)<br>🟢 Real In-Service<br>🔴 Synthetic Sim | **P3**<br>**P2**<br>**P4 (Sim)** | • [Tracked Excavator MDPI](https://www.researchgate.net/publication/351717382_Anomaly_Detection_for_Excavators_Based_on_Sensor_Data): Crawler chassis motor load.<br>• `suspension.py`: Synthetic track fatigue. |

---

### System 4: Hull & Armored Structure
*Armored chassis housing crew, powerpack, and subfloor mine protection.*

#### Subsystems & Components:
* **4.1 Hull Base Armor**: Chobham / Kanchan composite armored front glacis and side sponsons.
* **4.2 Hull Structure & Frame**: Welded high-hardness armored steel monocoque hull chassis.
* **4.3 Hatches & Doors**: Driver's sliding/lifting hatch, hull bottom emergency escape hatch.
* **4.4 Subfloor (Mine Protection)**: V-shaped belly plates, anti-mine blast sheets.

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **4.1 Hull Armor** | `CONFIDENTIAL / CLASSIFIED` | Ceramic tile shattering, delamination under repeated non-penetrating blast.<br>Acoustic Emission elastic wave | **In-house `acoustics.py` (AE)** | 🔴 Synthetic Sim | **P4 (Sim)** | • Modeled via Acoustic Emission wave propagation ($\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$). |
| **4.2 Structure & Frame** | `OPEN PROXY PROCURED` | Welded seam stress corrosion cracking, hull twisting under terrain shock.<br>Hooke $\sigma = E\epsilon$, $\frac{\Delta R}{R} = GF \cdot \epsilon$ | **In-house `suspension.py`**<br>**TartanDrive 2.0 (IMU)** | 🔴 Synthetic Sim<br>🟢 Real In-Service | **P2**<br>**P2** | • [TartanDrive GitHub](https://github.com/castacks/tartan_drive_2.0): Tri-axial frame shock load spectra.<br>• `suspension.py`: Strain gauge modeling. |
| **4.3 Hatches & Doors** | `OPEN PROXY PROCURED` | Hatch seal degradation, hinge binding.<br>Seal leak flow $Q = C_d A \sqrt{\frac{2\Delta P}{\rho}}$ | **ZeMA Single Cylinder (Festo)** | 🔵 Real Hardware Rig | **P3** | • [ZeMA Zenodo #5185953](https://zenodo.org/records/5185953): Linear actuator stroke & friction telemetry. |
| **4.4 Subfloor Mine Plates**| `CONFIDENTIAL / CLASSIFIED` | Belly plate deformation, mount shearing.<br>Impulsive shock $F = \int m \cdot a \, dt$ | **PHM 2022 Rock Drill** | 🔵 Real Hardware Rig | **P2** | • [PHM 2022 Rock Drill](https://data.phmsociety.org/2022-phm-conference-data-challenge/): Extreme mechanical shock & impulse telemetry. |

---

### System 5: Turret System
*Rotating armored fighting compartment housing gun, sights, and fire-control drives.*

#### Subsystems & Components:
* **5.1 Turret Armor & Structure**: Modular applique composite armor wedges, spall liners, turret race ring bearing.
* **5.2 Turret Traverse Mechanism**: All-electric PMSM motor drive (or electro-hydraulic motor) driving azimuth bull gear.
* **5.3 Gun Elevation Mechanism**: Electro-hydraulic elevation actuator / ballscrew linear servo driving gun trunnions.
* **5.4 Turret Race Ring & Slip-Ring**: 360° rotary electrical slip-ring, high-load turret ball race bearings.

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **5.2 Traverse Drive** | `OPEN PROXY PROCURED` | Azimuth drive gearhead backlash, PMSM motor winding thermal decay, hydraulic leak.<br>$P = \tau \omega$, $Z = R + j\omega L$ | **Electro-Hydraulic Servo (13 Faults)**<br>**Inverter-Driven PMSM Faults**<br>**ZeMA Hydraulic System (447)** | 🔴 Synthetic Sim<br>🔵 Real Hardware Rig<br>🔵 Real Hardware Rig | **P1 (Core)**<br>**P2**<br>**P1 (Core)** | • [IEEE DataPort Servo](https://ieee-dataport.org/documents/hydraulicsimulationwith13injectedfaults-0): Closed-loop position servo driving an inertia load.<br>• [PMSM Zenodo #13974503](https://zenodo.org/records/13974503): Motor electrical & thermal faults.<br>• [ZeMA UCI #447](https://archive.ics.uci.edu/dataset/447/): Hydraulic drive pressure. |
| **5.3 Elevation Drive** | `OPEN PROXY PROCURED` | Elevation actuator seal blowout, trunnion bearing galling, valve hysteresis.<br>$P_{hyd} = PQ$, $F_{out} = PA$ | **Electro-Hydraulic Servo**<br>**ZeMA Electromech Cylinders** | 🔴 Synthetic Sim<br>🔵 Real Hardware Rig | **P1 (Core)**<br>**P2** | • [IEEE DataPort Servo](https://ieee-dataport.org/documents/hydraulicsimulationwith13injectedfaults-0): Turret lay-on dynamics.<br>• [ZeMA Zenodo #3364431](https://zenodo.org/records/3364431): Cylinder force & velocity decay. |
| **5.4 Turret Race Ring** | `OPEN PROXY PROCURED` | Turret race ring ball bearing spalling, ring gear tooth wear.<br>Sideband energy ratio, $BPFO/BPFI$ | **PHM 2026 Spur Gear Challenge**<br>**Paderborn Bearings** | 🔵 Real Hardware Rig<br>🔵 Real Hardware Rig | **P2**<br>**P2** | • [PHM 2026](https://data.phmsociety.org/phm-north-america-2026-conference-data-challenge/): Spur gear tooth pitting.<br>• [Paderborn](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download): Low-speed high-load bearing fatigue. |

---

### System 6: Main Armament System
*120mm/125mm smoothbore/rifled tank cannon, recoil absorption, and loading systems.*

#### Subsystems & Components:
* **6.1 Main Gun Barrel & Breech**: 120mm M256 (Abrams), 120mm Rh-120 L/55 (Leopard 2), 120mm rifled (Arjun). Automatic wedge breech block.
* **6.2 Gun Recoil & Recuperator**: Hydro-pneumatic recoil buffer cylinder and nitrogen recuperator.
* **6.3 Ammunition Storage & Autoloader**: Blow-out ammo racks or cassette carousel autoloader.
* **6.4 Muzzle Brake & Bore Evacuator**: Glass-reinforced plastic fume extractor chamber on barrel.

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **6.1 Gun Barrel / Breech** | `CONFIDENTIAL / MILITARY GAP` | Bore rifling/land erosion, propellant obturation gas leak, thermal droop bowing.<br>$EFC = \sum N_i w_i$, MVR exit velocity decay | **In-house Digital Twin EFC**<br>**Paris-Erdogan fatigue model** | 🔴 Synthetic Sim<br>🔴 Synthetic Sim | **P4 (Sim)**<br>**P4 (Sim)** | • *Classified Gun Firing Data*: Mitigated by tracking Equivalent Full Charge (EFC) shot wear and thermal gradient droop in code. |
| **6.2 Gun Recoil Mechanism**| `HYBRID PROXY PROCURED` | Recoil buffer seal cavitation, nitrogen charge bleed, excessive recoil stroke length.<br>Pascal $P = \frac{F}{A}$, damping velocity | **PHM 2022 Rock Drill Challenge**<br>**ZeMA Hydraulic (Accumulator)** | 🔵 Real Hardware Rig<br>🔵 Real Hardware Rig | **P1 (Core)**<br>**P1 (Core)** | • [PHM 2022 Rock Drill](https://data.phmsociety.org/2022-phm-conference-data-challenge/): High-frequency cyclic hydraulic shock directly mimics recoil buffer decay.<br>• [ZeMA UCI #447](https://archive.ics.uci.edu/dataset/447/): Accumulator pre-charge leak. |
| **6.3 Autoloader / Loader** | `OPEN PROXY PROCURED` | Carousel indexing misalignment, rammer motor current spikes, mechanical binding.<br>$P = IV$, cycle time latency | **ZeMA Electromechanical Cylinders (3 Cyl)**<br>**ZeMA Single Festo Cylinder** | 🔵 Real Hardware Rig<br>🔵 Real Hardware Rig | **P2**<br>**P2** | • [ZeMA Zenodo #3364431](https://zenodo.org/records/3364431) & [#5185953](https://zenodo.org/records/5185953): Motor current shunt spikes during mechanical binding directly map to autoloader jams. |
| **6.4 Bore Evacuator** | `CONFIDENTIAL / GAP` | Evacuator check-valve fouling, diaphragm failure.<br>$\Delta P$ gas venting differential | **APS Failure at Scania (Valves)** | 🟢 Real In-Service | **P3** | • [Scania APS UCI #421](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks): Pneumatic check-valve sticking. |

---

### System 7: Secondary Armament System
* **7.1 Coaxial MG**: 7.62mm M240 / MG3 / MAG-58.
* **7.2 Commander’s RWS / Heavy MG**: 12.7mm M2HB .50 cal or Remote Weapon Station (CROWS II).
* **7.3 Smoke Grenade Dischargers**: Multi-barrel 76mm/81mm electrical smoke launchers (6–8 per turret flank).

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **7.2 Commander's RWS** | `OPEN PROXY PROCURED` | Remote weapon station slew motor overheating, optical feed lag.<br>$P = I^2 R$, resolver latency | **Inverter-Driven PMSM Faults**<br>**Univ. of Ottawa Motor** | 🔵 Real Hardware Rig<br>🔵 Real Hardware Rig | **P3**<br>**P3** | • [PMSM Zenodo #13974503](https://zenodo.org/records/13974503): Stator winding thermal stress.<br>• [Ottawa Motor Mendeley](https://data.mendeley.com/datasets/msxs4vj48g): Drive motor vibration. |
| **7.3 Smoke Dischargers** | `OPEN PROXY PROCURED` | Firing tube igniter continuity break, loop impedance drift.<br>$V = IR$, loopback continuity | **In-house BIT logic** | 🔴 Synthetic Sim | **P4 (Sim)** | • Software loopback circuit test checking circuit impedance ($R_{nom} \approx 2\ \Omega$). |

---

### System 8: Fire Control & Ballistics System
* **8.1 Digital Ballistic Computer**: Target range, crosswind, lead angle, barrel temperature, and cant.
* **8.2 Two-Axis Gun Stabilization**: Gyro-stabilized sight mirrors, resolvers, and electro-hydraulic servo loops.
* **8.3 Muzzle Reference System (MRS) & Sensors**: Crosswind mast sensors, laser rangefinders.

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **8.1 Ballistic Computer** | `CONFIDENTIAL / GAP` | Execution latency, MIL-STD-1553B bus frame drops, parity errors.<br>Bit Error Rate (BER) | **In-house decision audit log** | 🟢 Real In-Service Pipeline | **P1 (Core)** | • Monitored via execution time latencies and automated JSON-Lines audit verification. |
| **8.2 Gun Stabilization** | `OPEN PROXY PROCURED` | Gyroscopic rate drift, ADRC feedback servo valve latency, axis hunting.<br>Resolver error standard deviation | **Electro-Hydraulic Servo (13 Faults)**<br>**Inverter-Driven PMSM Faults** | 🔴 Synthetic Sim<br>🔵 Real Hardware Rig | **P1 (Core)**<br>**P2** | • [IEEE DataPort Servo](https://ieee-dataport.org/documents/hydraulicsimulationwith13injectedfaults-0): Closed-loop position error & valve response latency.<br>• [PMSM Zenodo](https://zenodo.org/records/13974503): Electrical feedback. |

---

### System 9: Observation & Sighting System
* **9.1 Gunner’s Primary Sight (GPS)**: Dual-axis stabilized day optical channel + 3rd-Gen FLIR with Stirling cryocooler + Eye-Safe LRF.
* **9.2 Commander’s Independent Thermal Viewer (CITV)**: 360° panoramic stabilized day/night sight.
* **9.3 Driver’s Vision Enhancer (DVE)**: Uncooled forward/rearward thermal camera.

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **9.1 Gunner Primary Sight** | `CONFIDENTIAL / MILITARY GAP` | Stirling cryocooler pump decay, thermal NETD drift.<br>Cryocooler thermodynamic cycle | **ZeMA Hydraulic (Cooler)**<br>**In-house `temperature.py`** | 🔵 Real Hardware Rig<br>🔴 Synthetic Sim | **P2**<br>**P4 (Sim)** | • [ZeMA UCI #447](https://archive.ics.uci.edu/dataset/447/): Cryocooler compressor thermodynamic decay analogue. |
| **9.2 Commander Sight (CITV)**| `OPEN PROXY PROCURED` | 360° slip-ring brush contact resistance spike, motor torque loss.<br>$R_{contact}$, $P = \tau \omega$ | **Univ. of Ottawa Motor**<br>**PMSM Inverter Dataset** | 🔵 Real Hardware Rig<br>🔵 Real Hardware Rig | **P3**<br>**P3** | • [Ottawa Motor Mendeley](https://data.mendeley.com/datasets/msxs4vj48g): Continuous motor speed & contact harmonics.<br>• [PMSM Zenodo](https://zenodo.org/records/13974503): Winding resistance. |

---

### System 10: Electronics & Data Infrastructure
* **10.1 Vehicle Digital Bus**: Dual-redundant MIL-STD-1553B databus, CAN bus (SAE J1939), STANAG 4754 Ethernet.
* **10.2 Multi-Function Displays (MFDs)**: Ruggedized touchscreen flat-panel displays.
* **10.3 Battle Management System (BMS)**: Tactical C4ISR mission computer.

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **10.1 Vehicle Data Bus** | `OPEN PROXY PROCURED` | Harness pin corrosion, EMI noise injection, dropped frames, Bit Error Rate (BER).<br>$\text{BER} = \frac{N_{err}}{N_{total}}$ | **US Army CBM Demo Subset (Kuiper)**<br>**Automotive CAN Bus (AEGIS)**<br>**SCANIA Component X**<br>**Kidmose CANid Dataset (KCID)** | 🟢 Real In-Service (Military)<br>🟢 Real In-Service<br>🟢 Real In-Service<br>🟢 Real In-Service | **P1 (Core)**<br>**P1 (Core)**<br>**P1 (Core)**<br>**P2** | • [gen_pm GitHub](https://github.com/patrick-kuiper/gen_pm): Real military vehicle CAN bus sensor streams and timestamped DTCs.<br>• [AEGIS Zenodo #3267184](https://zenodo.org/record/3267184): CAN bus time-series traces.<br>• [SCANIA Comp X](https://researchdata.se/en/catalogue/dataset/2024-34): Fleet telemetry logs.<br>• [KCID arXiv](https://arxiv.org/abs/2108.03875): Raw CAN traffic anomalies. |

---

### System 11: Communications & Navigation System
* **11.1 Tactical Radios**: SINCGARS, AN/VRC-92F SDR datalinks.
* **11.2 Digital Intercom**: Noise-canceling crew headsets.
* **11.3 Navigation System**: GPS/GLONASS with Land Navigation System (FOG INS).

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **11.3 Navigation (GPS/INS)**| `OPEN PROXY PROCURED` | Gyro bias drift, accelerometer scale-factor error, GPS spoofing/loss.<br>Kalman filter innovation variance | **DT-CARGO Operation Dataset**<br>**TartanDrive 2.0 (GPS/IMU)** | 🟢 Real In-Service<br>🟢 Real In-Service | **P2**<br>**P2** | • [DT-CARGO Zenodo #7599687](https://zenodo.org/records/7599687): 10 Hz GPS HDOP and trajectory tracking.<br>• [TartanDrive GitHub](https://github.com/castacks/tartan_drive_2.0): 9-DOF IMU data. |

---

### System 12: Electrical Power Generation & Distribution
* **12.1 24V DC Armored Battery Bank**: 6–8 Hawker Armasafe Plus AGM or Lithium-ion battery packs.
* **12.2 High-Output Alternator / Generator**: Oil-cooled 650A / 28V DC main engine alternator.
* **12.3 Solid-State Power Distribution Unit (PDU)**: Microprocessor-controlled solid-state circuit breakers and bus switches.

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **12.1 24V Battery Bank** | `OPEN PROXY PROCURED` | Capacity loss, internal resistance growth, thermal runaway, Silent Watch droop.<br>$V = V_{oc} - I R_{int}$, $SoH = \frac{C_{act}}{C_{nom}}$ | **NASA Randomized Battery (PCoE)**<br>**CALCE Battery Datasets**<br>**UNIBO Powertools Battery** | 🔵 Real Hardware Rig<br>🔵 Real Hardware Rig<br>🔵 Real Hardware Rig | **P1 (Core)**<br>**P2**<br>**P3** | • [NASA PCoE Battery](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/): Randomized load cycling.<br>• [CALCE Maryland](https://calce.umd.edu/battery-data): Dynamic driving degradation.<br>• [UNIBO Mendeley](https://data.mendeley.com/datasets/n6xg5fzsbv/1): Run-to-failure cycles. |
| **12.2 Alternator / Gen** | `OPEN PROXY PROCURED` | Rectifier diode breakdown, stator phase open-circuit, slip ring wear.<br>3-phase AC voltage ripple | **Univ. of Ottawa Electric Motor**<br>**NEV Fault Diagnosis (Kaggle)** | 🔵 Real Hardware Rig<br>🔵 Real Hardware Rig | **P2**<br>**P3** | • [Ottawa Motor Mendeley](https://data.mendeley.com/datasets/msxs4vj48g): Generator vibration & electrical harmonics.<br>• [NEV Kaggle](https://www.kaggle.com/datasets/ziya07/fault-diagnosis-dataset-for-new-energy-vehicles): Stator winding fault classification. |
| **12.3 Solid-State PDU** | `OPEN PROXY PROCURED` | MOSFET thermal overstress, contactor welding, overcurrent tripping.<br>Joule heating $Q = I^2 R t$ | **NASA PCoE MOSFET / IGBT Aging**<br>**In-house `ml/parts.py`** | 🔵 Real Hardware Rig<br>🟢 Real In-Service | **P2**<br>**P1 (Core)** | • [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/): Accelerated aging for power transistors.<br>• `parts.py`: Threshold tracking. |

---

### System 13: Environmental Control System (HVAC)
* **13.1 Vapor-Compression HVAC Unit**: Under-armor Thermal Management System (TMS) utilizing R134a refrigerant.
* **13.2 Cabin Circulation Blowers**: High-flow brushless blower fans and air duct diverter valves.

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **13.1 HVAC TMS Unit** | `OPEN PROXY PROCURED` | Refrigerant gas leak, compressor valve slap, condenser fan failure.<br>Thermodynamic COP $= \frac{Q_L}{W_{in}}$ | **MetroPT-3 Train APU**<br>**ZeMA Hydraulic (Cooler)** | 🟢 Real In-Service<br>🔵 Real Hardware Rig | **P1 (Core)**<br>**P1 (Core)** | • [MetroPT-3 UCI #791](https://archive.ics.uci.edu/dataset/791/): Industrial compressor pressure, temperature, and electrical load.<br>• [ZeMA UCI #447](https://archive.ics.uci.edu/dataset/447/): Heat exchanger decay. |
| **13.2 Cabin Blowers** | `OPEN PROXY PROCURED` | Blower motor bearing wear, air duct damper sticking.<br>$SPL = 20\log_{10}(\frac{p}{p_{ref}})$, $BPFO$ | **CWRU Fan Bearing Data**<br>**In-house `acoustics.py`** | 🔵 Real Hardware Rig<br>🔴 Synthetic Sim | **P3**<br>**P4 (Sim)** | • [CWRU Bearings](https://engineering.case.edu/bearingdatacenter/download-data-file): Fan bearing vibration signatures. |

---

### System 14: CBRN / NBC Life Support System
* **14.1 Multi-Stage NBC Particulate & Charcoal Filters**: Radial HEPA particulate filter + activated carbon gas adsorption filters.
* **14.2 Overpressure Blower & Blast Valves**: Positive pressure centrifugal blower maintaining $+200\text{ to }+500\text{ Pa}$ cabin overpressure.

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **14.1 NBC Filter Banks** | `HYBRID PROXY PROCURED` | HEPA dust loading, activated charcoal chemical breakthrough, differential pressure spike.<br>$\Delta P = \frac{1}{2} \rho v^2 \zeta$, Darcy's law | **MetroPT-3 Train Air Treatment**<br>**APS Failure at Scania Trucks** | 🟢 Real In-Service<br>🟢 Real In-Service | **P1 (Core)**<br>**P1 (Core)** | • [MetroPT-3 Zenodo #6854240](https://zenodo.org/records/6854240): Air filter differential pressure & desiccant dryer telemetry.<br>• [Scania APS UCI #421](https://archive.ics.uci.edu/dataset/421/): Air filtration failure records. |
| **14.2 Overpressure Blower**| `OPEN PROXY PROCURED` | Cabin positive pressure loss ($< 200\text{ Pa}$), blast overpressure valve sticking.<br>Pressure differential $\Delta P_{cabin}$ | **APS Failure at Scania Trucks**<br>**In-house `exhaust.py`** | 🟢 Real In-Service<br>🔴 Synthetic Sim | **P1 (Core)**<br>**P4 (Sim)** | • [Scania APS UCI #421](https://archive.ics.uci.edu/dataset/421/): Pressure regulation failure.<br>• `exhaust.py`: Manifold pressure physics. |

---

### System 15: Passive, Reactive & Spall Protection
* **15.1 Base Armor**: Chobham / Kanchan composite matrix (silicon carbide / alumina ceramic tiles in steel matrix).
* **15.2 Explosive Reactive Armor (ERA)**: TUSK ERA (Abrams), ERA MK-II (Arjun Mk1A) with explosive sandwich cassettes.
* **15.3 Internal Spall Liners**: Multi-ply woven Aramid / Kevlar fabric bolted inside fighting compartment.

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **15.1 Composite Armor** | `CONFIDENTIAL / CLASSIFIED` | Subsurface ceramic fracture, delamination under repeated non-penetrating blast.<br>Acoustic Emission elastic wave energy | **In-house `acoustics.py` (AE)** | 🔴 Synthetic Sim | **P4 (Sim)** | • [AE Module in `acoustics.py`](./sim/physics/acoustics.py): Micro-crack acoustic emission event rate and energy ($\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$). |
| **15.2 ERA Cassettes** | `CONFIDENTIAL / CLASSIFIED` | Explosive cassette detachment, environmental casing seal breach.<br>Vibration resonance shift | **In-house `vibration.py`** | 🔴 Synthetic Sim | **P4 (Sim)** | • Modeled via structural natural frequency shifts ($\omega_n = \sqrt{\frac{k}{m}}$). |

---

### System 16: Active Protection System (APS)
* **16.1 Active Threat Detection Radars**: 4-panel Active Electronically Scanned Array (AESA) millimeter-wave radar (Trophy / Iron Fist).
* **16.2 Laser Warning Receivers (ALWCS)**: Threat warning sensors detecting laser rangefinders and ATGM beam-riders.
* **16.3 Countermeasure Launchers**: Explosively formed projectile (EFP) interceptor launchers or directional aerosol smoke.

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **16.1 AESA Radar Panels** | `CONFIDENTIAL / CLASSIFIED` | Radar T/R module silicon-germanium power decay, receiver phase noise floor rise.<br>$P_{rx} = \frac{P_{tx} G^2 \lambda^2 \sigma}{(4\pi)^3 R^4}$, BIT RF level | **In-house BIT logic** | 🔴 Synthetic Sim | **P4 (Sim)** | • *Strictly Classified Defense Radar*: Modeled via software Built-In Test (BIT) RF power monitoring ($P_{tx} / P_{ref} < 0.85$). |
| **16.2 Laser Warning ALWCS**| `CONFIDENTIAL / CLASSIFIED` | Photodiode lens dust/mud attenuation, photodiode sensitivity loss.<br>Beer-Lambert law $I = I_0 e^{-\alpha x}$ | **In-house BIT loopback** | 🔴 Synthetic Sim | **P4 (Sim)** | • Software optical self-test LED loopback measuring lens transmissivity. |
| **16.3 Launchers** | `CONFIDENTIAL / CLASSIFIED` | Launcher slewing servo delay, interceptor igniter circuit open-circuit/impedance drift.<br>$Z = R + j\omega L$ | **ZeMA Electromech Actuators**<br>**In-house loopback impedance** | 🔵 Real Hardware Rig<br>🔴 Synthetic Sim | **P2**<br>**P4 (Sim)** | • [ZeMA Zenodo #3364431](https://zenodo.org/records/3364431): Actuator positioning latency.<br>• Igniter loopback continuity testing ($R \approx 1.5 - 2.5\ \Omega$). |

---

### System 17: Fire Detection & Suppression System
* **17.1 Optical Flame Detectors**: Dual-spectrum optical IR/UV sensors in crew and engine compartments.
* **17.2 Rapid Extinguisher Canisters**: Pressurized Halon 1301 / FM-200 gas cylinders with explosive squib valves ($< 200\text{ ms}$ discharge).

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **17.2 Fire Extinguishers** | `OPEN PROXY PROCURED` | Halon/clean agent cylinder pressure leakage, squib discharge circuit continuity.<br>Pascal $P = \frac{F}{A}$ | **MetroPT-3 (Pressure Vessels)**<br>**ZeMA Hydraulic (Accumulator)** | 🟢 Real In-Service<br>🔵 Real Hardware Rig | **P1 (Core)**<br>**P1 (Core)** | • [MetroPT-3 UCI #791](https://archive.ics.uci.edu/dataset/791/): Pressure vessel slow leakage telemetry.<br>• [ZeMA UCI #447](https://archive.ics.uci.edu/dataset/447/): Gas pre-charge monitoring. |

---

### System 18: Crew Systems & Ergonomics
* **18.1 Driver Controls & Pedals**: Multi-axis steering wheel / t-bar levers, throttle/brake pedals with rotary potentiometers.
* **18.2 Impact-Absorbing Seating**: Roof-suspended driver seat with variable-stroke mechanical damper.

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **18.1 Driver Controls** | `OPEN PROXY PROCURED` | Potentiometer wiper wear, deadband enlargement, return spring fatigue.<br>$R_{pot}$, spring $F = -kx$ | **TartanDrive 2.0 (Controls)** | 🟢 Real In-Service | **P2** | • [TartanDrive GitHub](https://github.com/castacks/tartan_drive_2.0): Continuous steering angle, throttle position, and brake commands. |
| **18.2 Seating Damper** | `CONFIDENTIAL / GAP` | Under-seat damper gas loss, mine-blast energy absorber stroke depletion.<br>Stroke LVDT $x(t)$ | **In-house `suspension.py`** | 🔴 Synthetic Sim | **P4 (Sim)** | • Mechanical damper stroke and acceleration attenuation physics. |

---

### System 19: Diagnostics, Prognostics & Health Monitoring (HUMS/PHM)
* **19.1 Probabilistic Sensor Validation (PNN)**: Validates multi-channel sensor correlation to reject sensor drift/wiring detachment.
* **19.2 Hybrid RUL Prognostic Engine**: Fuses physics-of-failure equations with multi-head LSTM and Bayesian particle filtering.
* **19.3 Automated Decision Audit & BMS Bridge**: Generates immutable JSON-Lines audit logs linking onboard RUL to military supply chain depots.

| Subsystem | Status | Monitored Failure Modes & Governing Physics | Applicable Datasets | Provenance | Priority | Source Link & Scope |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **19.1 Sensor Validation** | `OPEN PROXY PROCURED` | Transducer detachment, wiring harness short/open, spurious signal drift.<br>Multi-sensor cross-channel correlation | **All Catalogue Datasets**<br>**In-house `sim/`** | Multi-Source<br>🔴 Synthetic Sim | **P1 (Core)**<br>**P1 (Core)** | • PNN cross-checks vibration, temperature, and oil debris before triggering mechanical failure alerts. |
| **19.2 Hybrid RUL Engine** | `OPEN PROXY PROCURED` | Remaining Useful Life probability distribution collapse, particle filter variance.<br>$\text{RUL} = \int P(\text{fail} \mid x) dt$ | **SCANIA Component X**<br>**NASA N-CMAPSS (DS01–08)**<br>**US Army CBM (Kuiper)** | 🟢 Real In-Service<br>🟡 Hybrid (Flight+Sim)<br>🟢 Real In-Service (Military) | **P1 (Core)**<br>**P2**<br>**P1 (Core)** | • [SCANIA Comp X](https://researchdata.se/en/catalogue/dataset/2024-34): Fleet time-to-event ground truth.<br>• [N-CMAPSS PHM](https://data.phmsociety.org/2021-phm-conference-data-challenge/): Multi-failure-mode RUL.<br>• [gen_pm GitHub](https://github.com/patrick-kuiper/gen_pm): Military CAN prediction. |
| **19.3 Decision Audit Log** | `OPEN PROXY PROCURED` | Maintenance recommendation misclassification, spare part demand error.<br>Machine Learning Forecasting | **In-house `decision_audit_log.jsonl`** | 🟢 Real In-Service Pipeline | **P1 (Core)** | • [Results Audit Log in `results/`](./results/decision_audit_log.jsonl): UTC-timestamped JSON-Lines audit stream linking onboard predictions to depot logistics. |

---

## 4. Master Dataset Procurement Priority Roadmap

> [!NOTE]
> **Subsystem Proxy Caveats & Provenance Disclosures**:
> 1. **ZeMA Hydraulic System (UCI 447)**: Models generic industrial hydraulic testbed degradation (cooler, valve, pump, accumulator). It serves as an empirical physical circuit degradation proxy, not tank-specific physical dimensions.
> 2. **Naval Vessel Propulsion Plant (UCI 316)**: Tagged as 🟡 **Hybrid** (numerical thermodynamic simulator calibrated against naval propulsion machinery), ideal for gas turbine powerpacks (Abrams AGT1500).
> 3. **US Army CBM (`gen_pm` / arXiv 2407.17654)**: Generative modeling code and 1-vehicle demonstration pickle; full 200-vehicle military ground vehicle fleet data requires contact with author (patrick.kendal.kuiper@gmail.com).
> 4. **Zenodo Record #15626055 (Obike & Obot, 2025)**: Contains qualitative JSON fault/symptom/diagnosis text graphs (`automotive_faults_aktc_obike_et_al.json`), not tabular numerical sensor time-series.
> 5. **NASA N-CMAPSS**: Elevated to **Priority 1 (Core)** for Powerpack RUL modeling due to multi-failure-mode real flight profiles.

```
+─────────────────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    MASTER DATASET PROCUREMENT PRIORITY ROADMAP                              |
+─────────────────────────────────────────────────────────────────────────────────────────────────────────────+
|                                                                                                             |
|  [ PRIORITY 1: CORE FLEET INFRASTRUCTURE & BENCHMARKS ] (Immediate Download & Pipeline Ingestion)           |
|  1. ZeMA Hydraulic System Condition Monitoring (UCI 447)     ──► Turret, Elevation & Steering Hydraulics     |
|  2. Univ. of Ottawa Bearings under Time-Varying Speed (Mendeley) ─► Non-Stationary Transmission & Final Drive|
|  3. MetroPT-3 Train Air Production Unit (UCI 791 / Zenodo)   ──► Auxiliaries, APU, Pneumatics & Filters      |
|  4. SCANIA Component X — Fleet Run-to-Failure (ResearchData.se)─► Fleet-Wide Survival & RUL Ground Truth     |
|  5. Deutz TCD 12.0 V6 Diesel Engine Air Path (Zenodo 5766940) ──► Main Engine Core & Turbocharging Dynamics  |
|  6. APS Failure at Scania Trucks (UCI 421)                   ──► Air Filtration, NBC & Pneumatic Valves      |
|  7. NASA N-CMAPSS Flight Degradation (PHM Society Challenge) ──► Multi-Regime Powerpack RUL Benchmark        |
|  8. Naval Vessel Propulsion Plant (UCI 316 / Kaggle)         ──► Gas Turbine Powerpack (Abrams AGT1500)      |
|  9. Automotive CAN Bus Dataset — AEGIS (Zenodo 3267184)     ──► CAN Bus Traffic Integrity & Fault Traces    |
|                                                                                                             |
|  [ PRIORITY 2: HIGH-FIDELITY PHM & REGIME EXCITATION ] (Secondary Ingestion)                               |
|  10. TartanDrive 2.0 Off-Road Driving Dynamics (CMU GitHub)  ──► Suspension Roughness & Dynamic Shock        |
|  11. PHM 2022 Data Challenge — Hydraulic Rock Drill (PHM Soc) ──► Main Cannon Recoil Buffer Hydraulic Shock  |
|  12. ZeMA Electromechanical Cylinders — 3 Actuators (Zenodo) ──► Ammunition Autoloader Mechanical Rammers    |
|  13. Multi-Mode Gearbox Variable Conditions (Mendeley/IEEE)  ──► Cross-Drive Transmission Planetary Sets     |
|  14. PHM 2023 / PHM 2026 Gearbox Data Challenges (PHM Soc)   ──► Gear Tooth Pitting & Run-to-Failure Fatigue |
|  15. Paderborn University Bearing Dataset (KAt DataCenter)   ──► Real Accelerated Bearing Fatigue & MCSA     |
|  16. NASA Randomized Battery Usage Dataset (NASA PCoE)       ──► Silent Watch 24V DC Battery Degradation     |
|  17. Electro-Hydraulic Position Servo — 13 Faults (IEEE DP)  ──► Closed-Loop Turret Stabilization Control    |
|  18. FaultSeg Train Wheel Defect Segmentation (Zenodo)       ──► Roadwheel & Sprocket Computer Vision        |
|  19. Tracked Heavy Excavator Telemetry (MDPI / ResearchGate) ──► Tracked Chassis Dynamics & Heavy Hydraulics |
|  20. US Army Condition Based Maintenance (Kuiper gen_pm)     ──► Military Vehicle Generative Fault Modeling   |
|                                                                                                             |
|  [ PRIORITY 3: SPECIALIZED BENCHMARKS & ELECTRICAL ] (Auxiliary Evaluation)                                  |
|  21. NASA C-MAPSS Classic Dataset (NASA Open Data)           ──► Baseline Single-Mode RUL Regressors         |
|  22. Inverter-Driven PMSM Systems Fault Dataset (Zenodo)     ──► All-Electric Turret Traverse Motors         |
|  23. CALCE & UNIBO Battery Datasets (Maryland / Mendeley)    ──► Lithium-Ion Electrochemical Impedance       |
|  24. University of Ottawa Electric Motor Dataset (Mendeley)  ──► Auxiliary Motor Harmonics & Vibration       |
|  25. DT-CARGO Heavy Vehicle Operation (Zenodo 7599687)       ──► 10 Hz GPS, Load & Trajectory Excitation     |
|  26. Diesel Engine Faults Features (IEEE DataPort)           ──► Graded Engine Compression & Injection Drop  |
|                                                                                                             |
|  [ PRIORITY 4: MATHEMATICAL DIGITAL TWIN SYNTHESIS ] (Classified Defense Gaps)                               |
|  27. First-Principles Multi-Physics Core (`sim/`)       ──► 13 Coupled Physics Equations (58t MBT)      |
|  28. Equivalent Full Charge (EFC) + Paris-Erdogan Model      ──► Main Gun Barrel Thermo-Chemical Erosion     |
|  29. Built-In Test (BIT) Software Simulation                 ──► AESA Active Radar T/R Decay & Optronics     |
|                                                                                                             |
+─────────────────────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 5. Automated Data Staging Workflow (bash)

Run the following command in bash from the project root to create the structured proxy directories and stage the core verified datasets:

```bash
# Execute from the project root (any POSIX shell: bash, zsh, WSL).
set -euo pipefail

mkdir -p datasets/procured/zema_hydraulic
mkdir -p datasets/procured/metropt3_apu
mkdir -p datasets/procured/deutz_engine
mkdir -p datasets/procured/scania_aps

# 1. ZeMA Hydraulic Condition Monitoring (UCI 447)
echo "Downloading ZeMA Hydraulic Dataset (UCI 447)..."
curl -L --fail -o datasets/procured/zema_hydraulic/zema.zip \
    "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
unzip -o datasets/procured/zema_hydraulic/zema.zip -d datasets/procured/zema_hydraulic/

# 2. MetroPT-3 Train Air Production Unit (UCI 791)
echo "Downloading MetroPT-3 APU Dataset (UCI 791)..."
curl -L --fail -o datasets/procured/metropt3_apu/metropt3.zip \
    "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
unzip -o datasets/procured/metropt3_apu/metropt3.zip -d datasets/procured/metropt3_apu/

# 3. Deutz TCD 12.0 V6 Diesel Engine Air Path (Zenodo 5766940)
echo "Downloading Deutz TCD 12.0 V6 Air Path Dataset (Zenodo 5766940)..."
curl -L --fail -o datasets/procured/deutz_engine/tb_nrtc.csv \
    "https://zenodo.org/records/5766940/files/tb_nrtc.csv"
curl -L --fail -o datasets/procured/deutz_engine/gt_nrtc.csv \
    "https://zenodo.org/records/5766940/files/gt_nrtc.csv"

# 4. APS Failure at Scania Trucks (UCI 421)
echo "Downloading APS Failure at Scania Trucks (UCI 421)..."
curl -L --fail -o datasets/procured/scania_aps/scania_aps.zip \
    "https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip"
unzip -o datasets/procured/scania_aps/scania_aps.zip -d datasets/procured/scania_aps/
```
