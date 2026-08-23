"""CVRDE Auxiliary Power Unit (APU) & NBC Overpressure Simulator.
Simulates:
- 8.5 kW silent watch auxiliary diesel generator and 28V DC bus battery charging
- Nuclear-Biological-Chemical (NBC) positive cabin overpressure (500 Pa) filtration.
"""

from __future__ import annotations
import numpy as np
try:
    from .cvrde_config import CVRDETankConfig
except ImportError:
    from cvrde_config import CVRDETankConfig


class CVRDEAuxiliaryNBC:
    def __init__(self, cfg: CVRDETankConfig | None = None, rng: np.random.Generator | None = None):
        self.cfg = cfg or CVRDETankConfig()
        self.rng = rng or np.random.default_rng(self.cfg.noise_seed + 200)
        
        # State
        self.apu_running = True
        self.battery_soc_pct = 92.0
        self.bus_voltage_v = 28.2
        self.cabin_overpressure_pa = 505.0
        self.nbc_filter_dp_pa = 120.0

    def step(self, electrical_load_kw: float = 3.5, nbc_blower_on: bool = True,
             filter_dust_load: float = 0.0, cabin_seal_leak: float = 0.0,
             apu_running: bool | None = None) -> dict[str, float]:
        """Simulates one 10 Hz step of APU and NBC systems."""
        cfg = self.cfg
        # `apu_running` was initialised True and never written, so the
        # silent-watch / battery-discharge branch below was unreachable.
        if apu_running is not None:
            self.apu_running = bool(apu_running)
        
        # APU Generator and 28V DC Bus
        if self.apu_running:
            gen_power_kw = cfg.apu_rated_power_kw
            net_power_kw = gen_power_kw - electrical_load_kw
            self.battery_soc_pct = min(100.0, self.battery_soc_pct + net_power_kw * 0.001)
            self.bus_voltage_v = 28.0 + (self.battery_soc_pct - 50.0) * 0.02
            apu_rpm = 3000.0 + float(self.rng.normal(0.0, 15.0))
            apu_oil_p_bar = 3.8 + float(self.rng.normal(0.0, 0.05))
        else:
            self.battery_soc_pct = max(10.0, self.battery_soc_pct - electrical_load_kw * 0.005)
            self.bus_voltage_v = 24.0 + (self.battery_soc_pct / 100.0) * 3.5
            apu_rpm = 0.0
            apu_oil_p_bar = 0.0
            
        # NBC Positive Cabin Overpressure (500 Pa barrier prevents ingress of toxic agents)
        if nbc_blower_on:
            self.nbc_filter_dp_pa += filter_dust_load * 0.05
            blower_head_pa = 650.0 - self.nbc_filter_dp_pa
            target_pressure = max(blower_head_pa, 0.0) * (1.0 - cabin_seal_leak * 0.6)
            # Clean-filter target is the specified cabin overpressure.
            target_pressure = min(target_pressure, cfg.nbc_cabin_overpressure_pa * 1.3)
            self.cabin_overpressure_pa += (target_pressure - self.cabin_overpressure_pa) * 0.1
            self.cabin_overpressure_pa += float(self.rng.normal(0.0, 3.0))
        else:
            self.cabin_overpressure_pa += (0.0 - self.cabin_overpressure_pa) * 0.05

        return {
            "cvrde_apu_rpm": float(apu_rpm),
            "cvrde_apu_oil_pressure_bar": float(apu_oil_p_bar),
            "cvrde_bus_voltage_v": float(self.bus_voltage_v),
            "cvrde_battery_soc_pct": float(self.battery_soc_pct),
            "cvrde_nbc_overpressure_pa": float(max(self.cabin_overpressure_pa, 0.0)),
            "cvrde_nbc_filter_dp_pa": float(self.nbc_filter_dp_pa),
        }
