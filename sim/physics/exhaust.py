"""Exhaust-gas sensors for combustion health.

Physics
-------
Ideal gas:

    P V = n R T  ->  P = rho R_s T

Gas mass flow:

    m_dot = rho A v

Hydrocarbon combustion:

    C_x H_y + (x + y/4) O2 -> x CO2 + (y/2) H2O

Air-fuel equivalence ratio:

    lambda = (A/F)_actual / (A/F)_stoich

Exhaust temperature, pressure, oxygen concentration and flow jointly
reveal combustion abnormalities, incomplete combustion, air-fuel
imbalance and exhaust restrictions.
"""

from __future__ import annotations

import numpy as np

from ..config import TankConfig
from .temperature import ThermalSystem



class ExhaustSensor:
    """Simulates EGT, exhaust pressure, mass flow and lambda."""

    def __init__(self, cfg: TankConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)

    def read(self, thermal: "ThermalSystem", rpm: float, load: float,
             fuel_mult: float = 1.0, air_mult: float = 1.0,
             restriction: float = 0.0) -> dict[str, float]:
        cfg = self.cfg
        t_k = thermal.T_exhaust + 273.15
        rho = cfg.exhaust_pressure_base / (cfg.exhaust_Rs * t_k)
        v = 8.0 * load * (0.4 + 0.6 * rpm / cfg.max_speed_rpm) * air_mult
        # Hot-film / pitot mass-flow measurement.  Without noise this was
        # rho*A*v with v linear in `air_mult`, i.e. the injected
        # fuel_injector_fault parameter recoverable in closed form.
        m_dot = rho * cfg.exhaust_area * v
        m_dot = max(m_dot * (1.0 + self.rng.normal(0.0, cfg.mass_flow_noise)), 0.0)
        # Back-pressure rises with restriction and with the square of flow.
        # Previously this was a function of `restriction` alone with no noise,
        # leaving the channel at exactly one value across a healthy run.
        flow_term = 1.0 + 0.25 * (v / 8.0) ** 2
        p_restricted = cfg.exhaust_pressure_base * (1.0 + 0.4 * restriction) * flow_term
        p_restricted += self.rng.normal(0.0, 0.004 * cfg.exhaust_pressure_base)

        # Diesels are quality-governed: they run unthrottled with a roughly
        # fixed air charge and vary fuelling, so lambda is high at light load
        # and falls toward the smoke limit at full load. Typical range is
        # ~6 at idle down to ~1.2 at rated power.
        #
        # This was `afr_actual / stoich_afr` with both multipliers at 1.0,
        # giving lambda ~= 1.0 across the whole mission -- a stoichiometric
        # spark-ignition assumption applied to a compression-ignition engine.
        # A diesel never runs at lambda 1.0 in normal operation, so both the
        # channel and the lambda-based fault logic were modelling the wrong
        # combustion system.
        lam_nominal = cfg.lambda_idle - (cfg.lambda_idle - cfg.lambda_rated) * min(
            max(load, 0.0), 1.0)
        lam = lam_nominal * air_mult / max(fuel_mult, 1e-6)
        lam += self.rng.normal(0.0, cfg.lambda_noise * lam_nominal)
        o2_pct = 20.9 * (lam - 1.0) / lam if lam > 1.0 else 0.0
        # Raw lambda on a diesel swings 1.4-5.0 with duty alone, so it cannot
        # be scored against a fixed reference -- doing so penalises idling
        # exactly as hard as over-fuelling. The condition indicator is the
        # *residual* against the load-expected value: 1.0 when combustion is
        # nominal at any load, falling under over-fuelling and rising under a
        # air-side restriction. This is the same load-normalisation rule the
        # project already applies to susp_compliance and driveline_efficiency.
        lam_residual = lam / max(lam_nominal, 1e-6)
        return {
            "exhaust_pressure": float(p_restricted),
            "exhaust_mass_flow": float(m_dot),
            "lambda": float(lam),
            "lambda_residual": float(lam_residual),
            "exhaust_o2_pct": float(o2_pct),
        }