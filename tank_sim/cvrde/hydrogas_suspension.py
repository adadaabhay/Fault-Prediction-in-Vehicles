"""CVRDE Hydrogas Suspension Unit (HSU) Physical Simulator.
Simulates CVRDE's patented Hydrogas suspension stations (14 units on Arjun MBT Mk-1A):
- Non-linear adiabatic nitrogen gas spring (P * V^gamma = const)
- Multi-stage velocity-dependent hydraulic orifice damping
- LVDT suspension travel sensor and nitrogen seal leakage detection.
"""

from __future__ import annotations
import numpy as np
try:
    from .cvrde_config import CVRDETankConfig
except ImportError:
    from cvrde_config import CVRDETankConfig


class CVRDEHydrogasUnit:
    def __init__(self, station_id: int = 1, cfg: CVRDETankConfig | None = None,
                 rng: np.random.Generator | None = None):
        self.station_id = station_id
        self.cfg = cfg or CVRDETankConfig()
        self.rng = rng or np.random.default_rng(self.cfg.noise_seed + station_id)
        
        # State variables
        self.piston_area = np.pi * (self.cfg.hsu_cylinder_bore_m / 2.0) ** 2
        self.v0_gas = self.cfg.hsu_nitrogen_volume_m3
        self.p0_gas_pa = self.cfg.hsu_n2_precharge_bar * 1e5
        # Hydraulic fluid acts on the rod annulus, not the full bore, so the
        # gas spring supports a load of the right order. Using the full bore
        # area gave 249 kN per station against a 48 kN static share -- the
        # suspension sat ~5 g out of equilibrium at rest.
        self.rod_area = np.pi * (self.cfg.hsu_rod_diameter_m / 2.0) ** 2
        self.damper_oil_temp_c = 45.0
        self.gas_seal_integrity = 1.0 # 1.0 = sealed, < 0.7 = nitrogen leak
        self.prev_stroke_m = 0.0
        self.static_share_n = (self.cfg.combat_mass_kg * 9.81
                               / max(self.cfg.roadwheels_per_side * 2, 1))

    def step(self, road_elevation_m: float, tank_velocity_mps: float,
             seal_leak_rate: float = 0.0, damper_valve_stick: float = 0.0) -> dict[str, float]:
        """Simulates one 10 Hz step of an HSU station."""
        cfg = self.cfg
        
        # Seal degradation drops pre-charge pressure over time
        self.gas_seal_integrity = max(0.2, self.gas_seal_integrity - seal_leak_rate)
        effective_p0 = self.p0_gas_pa * self.gas_seal_integrity
        
        # Wheel stroke (m) from terrain profile and vehicle roll/pitch
        stroke_m = float(np.clip(road_elevation_m, -0.14, 0.14))
        # Stroke velocity is d(stroke)/dt. The previous expression multiplied
        # displacement by vehicle speed, giving units of m^2/s -- not a velocity,
        # which is why the damping term needed a fudge factor to look sane.
        stroke_velocity = (stroke_m - self.prev_stroke_m) / max(cfg.dt, 1e-6)
        self.prev_stroke_m = stroke_m
        
        # 1. Adiabatic Gas Spring Law: P(x) = P0 * (V0 / (V0 - A*x))^gamma
        displaced_vol = self.piston_area * stroke_m
        v_current = max(self.v0_gas - displaced_vol, self.v0_gas * 0.2)
        p_gas_pa = effective_p0 * ((self.v0_gas / v_current) ** cfg.gas_adiabatic_gamma)
        p_gas_bar = p_gas_pa / 1e5
        
        # 2. Hydraulic Damping Force: F = 0.5 * rho * Cd * A * v^2 * sgn(v)
        cd_eff = 0.65 * (1.0 + damper_valve_stick * 1.5)
        orifice_area = 2.0e-4
        f_damp_n = (0.5 * 870.0 * cd_eff * orifice_area
                    * (stroke_velocity ** 2) * np.sign(stroke_velocity))
        
        # Total wheel support force
        # The fluid acts on the rod annulus, not the full bore: using the bore
        # gave 249 kN per station against a 48 kN static share, leaving the
        # suspension ~5 g out of equilibrium at rest.
        f_total_kn = (p_gas_pa * self.rod_area + f_damp_n) / 1000.0
        
        # Damper oil friction heating
        work_done = abs(f_damp_n * stroke_velocity)
        self.damper_oil_temp_c += (work_done * 0.0002 - (self.damper_oil_temp_c - 40.0) * 0.05) * cfg.dt
        
        # Station accelerometer (g): net of the static share the wheel already
        # carries, so a stationary vehicle reads ~0 g rather than its own weight.
        net_n = f_total_kn * 1000.0 - self.static_share_n
        station_accel_g = net_n / max(self.static_share_n, 1e-6)
        station_accel_g += float(self.rng.normal(0.0, 0.05))

        return {
            f"cvrde_hsu_{self.station_id}_pressure_bar": float(p_gas_bar),
            f"cvrde_hsu_{self.station_id}_stroke_mm": float(stroke_m * 1000.0),
            f"cvrde_hsu_{self.station_id}_force_kn": float(f_total_kn),
            f"cvrde_hsu_{self.station_id}_temp_c": float(self.damper_oil_temp_c),
            f"cvrde_hsu_{self.station_id}_accel_g": float(station_accel_g),
            f"cvrde_hsu_{self.station_id}_seal_health_pct": float(self.gas_seal_integrity * 100.0),
        }
