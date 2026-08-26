"""CVRDE Two-Axis Gun Control System (GCS) & 120mm Recoil Simulator.
Simulates:
- 210 bar electro-hydraulic elevation servo cylinder
- 120mm main gun firing shock (450 kN peak recoil impulse across 35 ms)
- Line-of-sight (LOS) stabilization gyro drift (microradians).
"""

from __future__ import annotations
import numpy as np
try:
    from .cvrde_config import CVRDETankConfig
except ImportError:
    from cvrde_config import CVRDETankConfig


class CVRDEGunControlSystem:
    def __init__(self, cfg: CVRDETankConfig | None = None, rng: np.random.Generator | None = None):
        self.cfg = cfg or CVRDETankConfig()
        self.rng = rng or np.random.default_rng(self.cfg.noise_seed + 100)
        
        # State
        self.elevation_deg = 3.0
        self.azimuth_deg = 0.0
        self.elevation_hyd_pressure_bar = 205.0
        self.recoil_buffer_pressure_bar = 85.0
        self.recoil_cycles_count = 0
        self.servo_valve_hysteresis = 0.0
        # Cumulative barrel wear in Equivalent Full Charges.
        self.barrel_efc = 0.0
        self.recoil_stroke_mm = self.cfg.recoil_stroke_nominal_mm

    # Propellant charge weighting per natured round, relative to a full charge.
    EFC_PER_ROUND = {"APFSDS": 1.0, "HEAT": 0.55, "HE": 0.40, "PRACTICE": 0.30}

    def step(self, elev_cmd_deg: float, azim_rate_dps: float,
             hull_pitch_deg: float, hull_roll_deg: float,
             trigger_fire: bool = False, servo_leak: float = 0.0,
             round_type: str = "APFSDS",
             recuperator_gas_loss: float = 0.0) -> dict[str, float]:
        """Simulates one 10 Hz step of the Gun Control System."""
        cfg = self.cfg
        
        # Elevation cylinder servo dynamics
        self.elevation_deg += (elev_cmd_deg - self.elevation_deg) * 0.35
        self.elevation_deg = float(np.clip(self.elevation_deg, cfg.gcs_min_depression_deg, cfg.gcs_max_elevation_deg))
        
        # Hydraulic system pressure (drops during rapid elevation movement)
        flow_demand = abs(elev_cmd_deg - self.elevation_deg) * 2.5
        self.elevation_hyd_pressure_bar = cfg.gcs_system_pressure_bar - flow_demand - (servo_leak * 40.0)
        self.elevation_hyd_pressure_bar += float(self.rng.normal(0.0, 1.2))
        
        # 120mm Main Gun Recoil Dynamics
        recoil_force_kn = 0.0
        recoil_shock_g = 0.0
        if trigger_fire:
            self.recoil_cycles_count += 1
            # Barrel wear accrues by charge energy, not by shot count.
            self.barrel_efc += self.EFC_PER_ROUND.get(round_type.upper(), 1.0)
            recoil_force_kn = cfg.recoil_peak_force_kn + float(self.rng.normal(0.0, 15.0))
            recoil_shock_g = 18.5 + float(self.rng.normal(0.0, 1.2))
            self.recoil_buffer_pressure_bar = 165.0 # Recoil accumulator spike
            # Recoil travel lengthens as the recuperator loses gas charge --
            # the documented physical indicator of buffer/seal degradation.
            self.recoil_stroke_mm = (cfg.recoil_stroke_nominal_mm
                                     * (1.0 + 0.16 * float(np.clip(recuperator_gas_loss, 0.0, 1.0)))
                                     + float(self.rng.normal(0.0, 1.5)))
        else:
            self.recoil_buffer_pressure_bar += (85.0 - self.recoil_buffer_pressure_bar) * 0.2
            
        # Line-of-sight stabilization gyro error (microradians):
        # Baseline = 0.2 mrad = 200 urad, increases with hull motion and servo wear
        hull_disturbance = np.sqrt(hull_pitch_deg**2 + hull_roll_deg**2)
        los_error_urad = 180.0 + (hull_disturbance * 120.0) + (servo_leak * 350.0)
        if trigger_fire:
            los_error_urad += 800.0 # Momentary disruption during firing shock
        los_error_urad += float(self.rng.normal(0.0, 15.0))
        
        # Azimuth electric drive motor current (A)
        azim_current_a = 12.0 + abs(azim_rate_dps) * 8.5 + float(self.rng.normal(0.0, 0.8))

        return {
            "cvrde_gcs_elevation_deg": float(self.elevation_deg),
            "cvrde_gcs_hyd_pressure_bar": float(max(self.elevation_hyd_pressure_bar, 50.0)),
            "cvrde_gcs_recoil_buffer_bar": float(self.recoil_buffer_pressure_bar),
            "cvrde_gcs_recoil_force_kn": float(recoil_force_kn),
            "cvrde_gcs_recoil_shock_g": float(recoil_shock_g),
            "cvrde_gcs_los_error_urad": float(max(los_error_urad, 50.0)),
            "cvrde_gcs_azim_current_a": float(azim_current_a),
            "cvrde_gcs_rounds_fired": float(self.recoil_cycles_count),
            "cvrde_gcs_barrel_efc": float(self.barrel_efc),
            "cvrde_gcs_barrel_life_pct": float(
                max(0.0, 100.0 * (1.0 - self.barrel_efc / max(cfg.barrel_life_efc, 1e-9)))),
            "cvrde_gcs_recoil_stroke_mm": float(self.recoil_stroke_mm),
        }
