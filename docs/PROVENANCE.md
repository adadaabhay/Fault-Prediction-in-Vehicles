# Data Provenance and Subsystem Coverage

Every corpus in this project is a **declared surrogate**. Real fleet telemetry
for the platforms of interest (Arjun Mk-1A, T-90S, Zorawar) is classified, so
the approach follows CVRDE proposal §9.4: bootstrap on public datasets whose
physics transfers, and swap the loaders for real DAQ/CAN feeds when field data
becomes available.

The rule applied throughout: **a supervised target must come from an annotation
channel independent of the features it is predicted from.** A corpus that
cannot meet that rule is not scored — it is listed under "Excluded" with the
reason, and retained only for what it can honestly supply.

---

## Scored corpora

| Corpus | Source | Label provenance | Subsystem proxied |
|---|---|---|---|
| ZeMA Hydraulic Rig | UCI 447 | Rig `profile.txt` condition annotations | Turret traverse / stabiliser hydraulics |
| MetroPT-3 APU | UCI 791 | Operator air-leak failure reports (4 episodes) | Auxiliary power unit / pneumatics |
| Scania APS | UCI 421 | Workshop pass/fail component records | Heavy fleet air-pressure system |
| Naval GT Propulsion | UCI 316 | Experimental decay coefficients (design variable) | Gas-turbine powerplant (AGT1500-class) |
| SCANIA Component X | researchdata.se 2024-34 | Workshop repair records (time-to-event) | Fleet component RUL |

### Sampling notes

- **ZeMA** cycles are ordered by experimental block. A head slice (`iloc[:200]`)
  lands inside a single condition — the first 200 cycles contain only
  `valve_condition == 100` and only `cooler_condition == 3`, i.e. no fault
  variation at all. All sampling goes through `stratified_cycle_sample`.
- **MetroPT-3** is a 10-second-interval time series. Folds are split by failure
  episode (`GroupKFold`), never shuffled: adjacent samples are seconds apart and
  a random split puts the same episode in train and test.
- **Scania APS** median imputation is a whole-frame statistic and therefore
  leaks slightly across folds. The benchmark passes `impute=False` when a
  NaN-aware learner is available.
- **SCANIA Component X** is the project's only non-simulated RUL ground truth.
  Two properties dominate and are handled explicitly:
  - **90.4% of vehicles are right-censored** (21,278 of 23,550 never had the
    repair). For those, `length_of_study_time_step` is a lower bound on
    component life, not a life. Regressing on it directly teaches the model
    that healthy trucks fail the moment observation stopped. The benchmark
    scores the uncensored subset only, and `censoring_summary()` carries the
    corpus rate through so a stratified sample can never be mistaken for it.
  - Readouts are a **time series per vehicle** (~48 rows each, 1.12 M rows
    total), so folds are grouped by `vehicle_id`. A random row split would put
    the same truck in train and test.
  - `time_step` is excluded from the feature set because the target is defined
    as `study_end - time_step`; leaving the clock in lets the model
    reconstruct the label from it.

---

## Excluded from scoring, with reasons

| Corpus | Reason | Retained for |
|---|---|---|
| Deutz TCD 12.0 V6 (Zenodo 5766940) | Ships **no fault labels**. The engine is healthy throughout. A threshold indicator over `lambda` and `engine_power_kw` scores ROC-AUC 1.0 over **5 positives** purely by re-deriving its own rule. | Transient NRTC duty cycle; 42-channel CFD-vs-bench model residual baseline |
| AEGIS instrumented CAN trace | All **16 diagnostic lamp/warning channels read zero** for the whole trip — verified, not assumed (`can_aegis.lamp_activity`). A healthy trip has no fault to predict. | Real CAN framing and signal values for the J1939 gateway; road duty cycle |
| `engine_fault_db` | No pipeline written; provenance not yet established. | Nothing — currently unused |

## Measured results

Scored with the rules above (grouped CV where units are correlated, label
provenance asserted, positive counts reported):

| Subsystem | Metric | Note |
|---|---|---|
| Hydraulics (cooler / valve / pump / accumulator) | ROC-AUC 0.999x | Rig-annotated, well-separated by design |
| APU leak **detection** | ROC-AUC 0.9932 ± 0.0040, F1 0.684 | Leave-one-episode-out |
| APU leak **24 h prediction** | ROC-AUC 0.489 ± 0.134 | **Chance.** No evidence of lead-time predictability from four episodes |
| Heavy fleet APS | ROC-AUC 0.988, F1 0.686 | Real imbalanced workshop labels |
| Gas-turbine decay (compressor / turbine) | R² 0.998 / 0.988 | Continuous experimental design variable |
| **Fleet component RUL** | **R² 0.450 ± 0.024, MAE 48.4** | Real time-to-event, vehicle-grouped, uncensored subset |

The last row is the most representative of a real PdM problem: a moderate score
on genuine workshop data is worth more than a near-perfect one on a target
derived from its own features.

---

## Subsystems with no public sensor data

Against the 20-system MBT taxonomy, the following have **no releasable sensor
corpus**, confirmed by search rather than assumed:

- **Tracked running gear** — road wheels, idlers, sprockets, track pin/bushing
  wear. No open dataset exists; rail axle-box literature independently reports
  the same absence. `FaultSeg` (Zenodo 13162335) offers a computer-vision route
  via rail-wheel shelling, which is the same rolling-contact-fatigue physics.
- **Fire control and sighting** — ballistic computer, stabilised sights, gyro
  drift. No raw corpus. A published dual-mass MEMS gyroscope fault taxonomy
  (7 classes: normal, bias, blocking, drift, multiplicative, cyclic, internal)
  is implementable in simulation without data.
- **Active protection systems** — radar T/R decay, interceptor loopback.
- **CBRN filtration and fire suppression** — filter loading, agent discharge.

**CBRN filtration is the one addressable gap.** The LBNL Fault Detection and
Diagnostics dataset (`data.openei.org/submissions/5763`) includes graded
filter-restriction faults at 10 / 20 / 50 % increased airflow resistance —
the same differential-pressure-versus-dust-loading physics as an NBC filter
bank. It is free, and with Component X now procured it is the single
remaining recommended acquisition.

Realistic honest coverage: **7–8 of 20 systems evidenced by real data.** Stating
the remaining twelve as findings is deliberate. A submission claiming twenty
green subsystems where fifteen are constants is weaker, not stronger, than one
that names its gaps.

---

## Simulation-derived streams

`sim_mbt` and `cvrde_arjun` are physics simulations, labelled `kind: simulation`
in the stream metadata. Their health curves are ground truth **by construction**
— which is legitimate for a simulator, provided it is declared. The CVRDE
mission scores health from the simulated telemetry against physical limits
rather than reading back the injected severity, so the label is not a
restatement of the fault parameter.

## Dashboard channel provenance

Each `real_*` stream declares three lists in its metadata:

- `channels_measured` — model-schema channels backed by the source recording
- `channels_measured_extra` — real source channels outside the model schema
- `channels_synthetic` — schema placeholders, **not measurements**

The model input vector is fixed-width, so absent channels need a value. They are
enumerated so the dashboard renders them as `n/a` rather than presenting a
scaler midpoint as if it were a reading. An engine dyno genuinely has no
suspension strain channel; the fix is to say so, not to invent one.
