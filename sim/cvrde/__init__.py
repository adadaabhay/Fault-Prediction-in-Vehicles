"""CVRDE (Combat Vehicles Research and Development Establishment) Military Tank Subsystem Simulation Package.
Modeled on Arjun MBT Mk-1A, Zorawar Light Tank, and T-90S Bhishma architectures.
"""

from .cvrde_config import CVRDETankConfig
from .powerpack import CVRDEPowerpack
from .hydrogas_suspension import CVRDEHydrogasUnit
from .gun_control import CVRDEGunControlSystem
from .auxiliary_nbc import CVRDEAuxiliaryNBC
from .cvrde_generator import CVRDEMissionGenerator

__all__ = [
    "CVRDETankConfig",
    "CVRDEPowerpack",
    "CVRDEHydrogasUnit",
    "CVRDEGunControlSystem",
    "CVRDEAuxiliaryNBC",
    "CVRDEMissionGenerator",
]
