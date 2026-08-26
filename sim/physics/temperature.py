"""Engine, transmission and exhaust temperature sensors.

Physics
-------
Thermocouple (Seebeck effect):

    V = integral[Tc..Th] S(T) dT  ~  S (Th - Tc)

RTD resistance:

    R(T) = R0 [1 + alpha (T - T0)]

Simplified engine thermal balance:

    m c_p dT/dt = Q_gen - Q_cool - Q_exhaust

Thermal deviations indicate cooling-system degradation, excessive
engine load, lubrication problems, combustion abnormalities or
transmission friction.
"""

from __future__ import annotations

import numpy as np

from ..config import TankConfig


class ThermalSystem:
    """Time-integrates the engine thermal balance.

    The heat-generation term scales with throttle/load; cooling removes
    heat at a rate proportional to the temperature difference to the
    ambient (its efficiency degrades under a cooling fault).  A fraction
    of the generated heat is rejected through the exhaust.
    """

    def __init__(self, cfg: TankConfig, ambient: float = 25.0):
        self.cfg = cfg
        self.ambient = ambient
        # Initial warm-up state. This used to be `45.0 + 0.1 * cfg.noise_seed`,
        # which made a *physical initial condition* a function of the RNG seed:
        # changing the seed changed the starting engine temperature by up to
        # several kelvin, so runs were not comparable across seeds.
        self.T_engine = 45.0
        self.T_oil = self.T_engine - 8.0
        # Cold-start EGT, before combustion loading.
        self.T_exhaust = 180.0

    def step(self, load: float, rpm: float,
             cooling_eff: float = 1.0, load_multiplier: float = 1.0,
             lambda_actual: float = 1.8) -> None:
        cfg = self.cfg
        q_gen = cfg.max_fuel_energy_rate * load * load_multiplier
        # Radiator + fan + oil cooler effectiveness rises with duty cycle,
        # otherwise the linear balance never reaches an equilibrium.
        k_eff = cfg.coolant_power * (1.0 + 3.0 * load)
        # Thermostat. A real cooling circuit is regulated, not proportional:
        # below the setpoint the radiator loop is bypassed so the engine warms
        # up quickly, and above it the valve opens progressively. Without this
        # the balance settled wherever load happened to put it -- 43-78 C on a
        # healthy mission, against a real MBT operating band of 85-105 C and
        # against parts.py's own `healthy: 90` reference. The warn/crit
        # thresholds (105/120 C) were therefore unreachable except under an
        # injected fault, which is what made the cooling alarm chain
        # self-referential.
        opening = min(max((self.T_engine - cfg.thermostat_open_c)
                          / max(cfg.thermostat_range_c, 1e-6), 0.0), 1.0)
        k_eff *= cfg.thermostat_bypass_leak + (1.0 - cfg.thermostat_bypass_leak) * opening
        q_cool = k_eff * max(self.T_engine - self.ambient, 0.0) * cooling_eff
        q_exhaust = cfg.exhaust_heat_fraction * q_gen
        dT = (q_gen - q_cool - q_exhaust) / (cfg.engine_mass_thermal * cfg.c_p_coolant)
        self.T_engine += dT * cfg.dt
        self.T_oil += (self.T_engine - self.T_oil) * cfg.dt * 0.8

        # --- Exhaust gas temperature ------------------------------------
        # EGT is set by combustion, not by the coolant loop. This was
        # `T_engine*0.85 + 90`, which tied it to coolant temperature and
        # produced 121-158 C on a healthy mission -- roughly a quarter of the
        # 400-700 C a loaded diesel actually runs, and four times lower than
        # the CVRDE model in this same repository, which uses a 750 C
        # pyrometer limit. The `parts.py` thresholds (healthy 142, warn 240,
        # crit 300) had then been fitted to the simulator's wrong numbers
        # rather than to engine reality, so the whole EGT alarm chain was
        # self-referential.
        #
        # Energy balance on the exhaust stream:
        #     T_egt = T_intake + Q_exh / (m_dot_exh * c_p_gas)
        # Excess air (high lambda) dilutes the same heat into more mass and
        # cools the stream, which is why a diesel at light load runs cool and
        # a fuelling fault that drops lambda drives EGT up.
        # A diesel is quality-governed: air flow per revolution is roughly
        # fixed, so exhaust mass flow tracks engine speed. Lambda is the
        # *result* of how much fuel goes into that air, which is why a
        # fuelling fault (falling lambda) raises EGT at constant airflow.
        m_dot = max(cfg.exhaust_mdot_ref * (0.12 + 0.88 * rpm / cfg.max_speed_rpm),
                    1e-3)
        # Excess air beyond the reference dilutes the same heat into more mass.
        m_dot *= max(lambda_actual, 0.3) / cfg.lambda_reference
        t_target = (self.ambient
                    + q_exhaust / (m_dot * cfg.c_p_exhaust_gas))
        # Gas leaving the head cannot be meaningfully colder than the metal it
        # just passed through, so the port temperature floors the estimate.
        # Without it a long idle segment drove EGT toward ambient.
        t_target = max(t_target, self.T_engine + cfg.exhaust_port_soak_c)
        t_target = min(t_target, cfg.max_egt_c)
        # First-order thermal lag of the manifold and turbine housing.
        tau = max(cfg.exhaust_tau_s, cfg.dt)
        self.T_exhaust += (t_target - self.T_exhaust) * (cfg.dt / tau)


class EngineTemperatureSensor:
    """RTD reading of engine coolant temperature."""

    def __init__(self, cfg: TankConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)
        self.alpha = 0.00385  # 1/K for platinum RTD
        self.R0 = 100.0
        self.T0 = 0.0

    def resistance(self, temperature_c: float) -> float:
        return self.R0 * (1.0 + self.alpha * (temperature_c - self.T0))

    def read(self, thermal: ThermalSystem) -> dict[str, float]:
        t = thermal.T_engine + self.rng.normal(0.0, 0.4)
        return {"coolant_temp": t, "coolant_rtd_ohm": self.resistance(t)}


class ExhaustThermocoupleSensor:
    """Exhaust-gas temperature (thermocouple) reading."""

    def __init__(self, cfg: TankConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)
        self.seebeck = 41.0e-6  # V/K (type-K approximation)
        self.T_cold = 25.0

    def read(self, thermal: ThermalSystem) -> dict[str, float]:
        t = thermal.T_exhaust + self.rng.normal(0.0, 1.2)
        emf = self.seebeck * (t - self.T_cold)
        return {"exhaust_temp": t, "exhaust_thermocouple_v": emf}


# Backwards compatibility alias
CoolantSensor = ExhaustThermocoupleSensor