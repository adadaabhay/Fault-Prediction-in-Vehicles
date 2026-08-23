"""CVRDE 1400 hp Multi-Fuel Powerpack Simulator.
Models high-ambient (+50 C Thar Desert) thermal de-rating, dual exhaust gas pyrometry (EGT_A, EGT_B),
turbocharger boost dynamics, common-rail injection pressure, and oil gallery circulation.
"""

from __future__ import annotations
import numpy as np
try:
    from .cvrde_config import CVRDETankConfig
except ImportError:
    from cvrde_config import CVRDETankConfig


class CVRDEPowerpack:
    def __init__(self, cfg: CVRDETankConfig | None = None, rng: np.random.Generator | None = None):
        self.cfg = cfg or CVRDETankConfig()
        self.rng = rng or np.random.default_rng(self.cfg.noise_seed)
        
        # State variables
        self.coolant_temp_c = 85.0
        self.oil_temp_c = 90.0
        self.intercooler_out_temp_c = 45.0
        self.air_filter_clog_pct = 0.0 # Sand ingestion in desert
        self.cooling_fan_efficiency = 1.0  # settable; degrades radiator capacity

    def step(self, rpm: float, load: float, ambient_temp_c: float = 45.0,
             injector_wear: float = 0.0, turbo_decay: float = 0.0,
             sand_clog_rate: float = 0.0, oil_pump_wear: float = 0.0,
             fan_efficiency: float | None = None) -> dict[str, float]:
        """Simulates one 10 Hz step of the 1400 hp powerpack."""
        cfg = self.cfg
        if fan_efficiency is not None:
            self.cooling_fan_efficiency = float(np.clip(fan_efficiency, 0.05, 1.0))
        
        # Progressive sand filter clogging
        self.air_filter_clog_pct = min(100.0, self.air_filter_clog_pct + sand_clog_rate)
        filter_flow_loss = self.air_filter_clog_pct * 0.003
        
        # High-ambient engine thermal de-rating: power drops ~0.5% per deg C above 35 C
        ambient_derate = max(0.0, (ambient_temp_c - 35.0) * 0.005)
        effective_power_fraction = max(0.2, (1.0 - ambient_derate - filter_flow_loss) * (1.0 - turbo_decay * 0.3))
        
        # Turbocharger boost pressure: p_boost = 1.0 + (boost_max - 1.0) * (rpm/max_rpm)^1.5 * load * eff
        rpm_ratio = min(max(rpm / cfg.engine_max_rpm, 0.0), 1.2)
        p_boost_bar = 1.0 + (cfg.boost_pressure_max_bar - 1.0) * (rpm_ratio ** 1.5) * load * effective_power_fraction
        p_boost_bar += float(self.rng.normal(0.0, 0.04))
        
        # Dual Exhaust Gas Temperature (EGT) Pyrometers (Bank A and Bank B)
        base_egt = 250.0 + load * 420.0 + (rpm_ratio * 100.0) + (ambient_temp_c * 0.8)
        egt_a = base_egt + (injector_wear * 85.0) + float(self.rng.normal(0.0, 4.0))
        egt_b = base_egt + (injector_wear * 15.0) + float(self.rng.normal(0.0, 4.0)) # Uneven injector drift
        
        # Intercooler temperature
        self.intercooler_out_temp_c += 0.02 * (ambient_temp_c + (p_boost_bar - 1.0) * 22.0 - self.intercooler_out_temp_c)
        
        # Engine Coolant and Oil Thermal Dynamics
        heat_gen = load * (rpm_ratio ** 1.2) * 180.0
        heat_diss_coolant = (self.coolant_temp_c - ambient_temp_c) * 1.6 * self.cooling_fan_efficiency
        heat_diss_oil = (self.oil_temp_c - self.coolant_temp_c) * 0.8
        
        self.coolant_temp_c += (heat_gen - heat_diss_coolant) * (cfg.dt / 15.0)
        self.oil_temp_c += (heat_gen * 0.6 - heat_diss_oil) * (cfg.dt / 25.0)
        
        # Oil gallery pressure (drops with higher temperature and pump wear)
        # Oil pressure falls with thinning oil and with pump wear. It was
        # previously reduced by turbo_decay, which is an unrelated subsystem.
        visc_factor = max(0.5, 1.0 - (self.oil_temp_c - 90.0) * 0.008)
        oil_p_bar = (cfg.nominal_oil_pressure_bar * visc_factor
                     * (1.0 - float(np.clip(oil_pump_wear, 0.0, 1.0)) * 0.55))
        oil_p_bar += float(self.rng.normal(0.0, 0.05))
        
        # Common Rail Fuel Injection Pressure (1800 bar rail)
        rail_pressure_bar = 600.0 + (load * 1150.0) + float(self.rng.normal(0.0, 10.0))

        return {
            "cvrde_engine_rpm": float(rpm),
            "cvrde_engine_load": float(load),
            "cvrde_boost_pressure_bar": float(max(p_boost_bar, 1.0)),
            "cvrde_egt_bank_a_c": float(egt_a),
            "cvrde_egt_bank_b_c": float(egt_b),
            "cvrde_coolant_temp_c": float(self.coolant_temp_c),
            "cvrde_oil_temp_c": float(self.oil_temp_c),
            "cvrde_oil_pressure_bar": float(max(oil_p_bar, 0.5)),
            "cvrde_rail_pressure_bar": float(rail_pressure_bar),
            "cvrde_intercooler_out_temp_c": float(self.intercooler_out_temp_c),
            "cvrde_air_filter_clog_pct": float(self.air_filter_clog_pct),
        }
