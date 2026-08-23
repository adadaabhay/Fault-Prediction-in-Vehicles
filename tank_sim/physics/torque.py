"""Torque sensors for engine and drivetrain mechanical load.

Physics
-------
Torque from tangential force and shaft radius:

    tau = r F

Mechanical power:

    P_mech = tau * omega,    omega = 2 pi N / 60

Circular-shaft shear stress and strain:

    tau_shear = T r / J
    gamma      = T r / (J G)

A reduction gearbox multiplies torque and divides speed, so sprocket torque
is engine torque * gear_ratio * efficiency.  Drivetrain efficiency degradation
therefore *reduces* the torque delivered to the sprocket for the same engine
power -- it does not raise it.
"""

from __future__ import annotations

import numpy as np

from ..config import TankConfig


class TorqueSensor:
    """Simulates a strain-based shaft torque sensor on the drivetrain."""

    def __init__(self, cfg: TankConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)

    def omega(self, rpm: float) -> float:
        return 2.0 * np.pi * rpm / 60.0

    def read(self, rpm: float, engine_power_w: float,
             efficiency: float = 1.0) -> dict[str, float]:
        cfg = self.cfg
        omega = self.omega(rpm)
        # Engine output-shaft torque, where the sensor is mounted.
        tau_shaft = engine_power_w / max(omega, 1e-3) if omega > 0 else 0.0
        tau_shaft += self.rng.normal(0.0, cfg.torque_noise)

        # Torque actually delivered at the sprocket after the reduction and
        # the (degrading) drivetrain efficiency.  The final-drive sensor is a
        # distinct transducer from the input-shaft one, so it carries its own
        # noise -- without it the efficiency ratio inverts exactly.
        tau_sprocket = tau_shaft * cfg.gear_ratio * max(efficiency, 0.05)
        tau_sprocket += self.rng.normal(0.0, cfg.torque_noise * cfg.gear_ratio)
        delivered_power = tau_sprocket * (omega / cfg.gear_ratio)

        r = cfg.shaft_radius
        shear = tau_shaft * r / cfg.shaft_J
        gamma = tau_shaft * r / (cfg.shaft_J * cfg.shaft_shear_modulus)
        # Efficiency is *inferred* from the two independently-measured torques,
        # never read back from the injected parameter.  The previous version
        # published `max(efficiency, 0.05)` verbatim: a depth-1 threshold on it
        # scored 99.88% on its own fault class because it *was* the label.
        eff_measured = tau_sprocket / max(tau_shaft * cfg.gear_ratio, 1e-6)
        return {
            "shaft_torque": float(tau_shaft),
            "sprocket_torque": float(tau_sprocket),
            "driveline_efficiency": float(eff_measured),
            "shaft_shear_stress": float(shear),
            "shaft_shear_strain": float(gamma),
            # Reported from the *measured* shaft torque, as a torque-telemetry
            # channel is.  Echoing the commanded engine_power_w made this a
            # noise-free exogenous readback sitting in a sensor column.
            "mech_power": float(tau_shaft * omega),
            "delivered_power": float(delivered_power),
            "shaft_omega": float(omega),
        }