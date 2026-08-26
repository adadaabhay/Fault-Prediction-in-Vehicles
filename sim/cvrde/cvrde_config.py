"""CVRDE Tank Configuration & Parametric Baseline.
Specifies engineering dimensions, capacities, and ratings for CVRDE armored platforms:
- Arjun MBT Mk-1A (68.5 tonne Main Battle Tank)
- Zorawar (25 tonne High-Altitude Light Tank)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CVRDETankConfig:
    # Vehicle Class & Mass
    vehicle_name: str = "Arjun MBT Mk-1A"
    combat_mass_kg: float = 68500.0        # 68.5 tonnes
    track_width_m: float = 0.65            # Diehl 840 double-pin track
    roadwheels_per_side: int = 7           # 14 HSU stations total
    
    # 1400 hp Multi-Fuel Powerpack
    engine_rated_power_kw: float = 1030.0  # 1400 hp @ 2400 RPM
    engine_max_rpm: float = 2600.0
    engine_idle_rpm: float = 800.0
    displacement_litres: float = 39.2      # V10 multi-fuel diesel
    boost_pressure_max_bar: float = 3.2    # Dual turbochargers
    nominal_oil_pressure_bar: float = 5.2  # Main oil gallery
    max_coolant_temp_c: float = 115.0      # High-ambient cooling threshold
    max_egt_c: float = 750.0               # Exhaust pyrometer limit
    desert_ambient_ref_c: float = 50.0     # Thar desert rating (+50 C)
    cold_ambient_ref_c: float = -30.0      # Ladakh sub-zero rating (-30 C)

    # CVRDE Hydrogas Suspension Unit (HSU)
    hsu_n2_precharge_bar: float = 220.0    # Nitrogen pre-charge pressure
    hsu_max_operating_pressure_bar: float = 350.0
    hsu_cylinder_bore_m: float = 0.12      # 120 mm hydraulic cylinder
    hsu_rod_diameter_m: float = 0.06       # 60 mm rod
    hsu_max_stroke_m: float = 0.30         # 300 mm total wheel travel
    hsu_nitrogen_volume_m3: float = 0.0035 # 3.5 litres initial gas volume
    gas_adiabatic_gamma: float = 1.4       # Diatomic nitrogen gamma

    # Gun Control System (GCS) & 120mm Armament
    gcs_system_pressure_bar: float = 210.0 # 21 MPa hydraulic supply
    gcs_elevation_cylinder_area_m2: float = 0.015
    gcs_max_elevation_deg: float = 20.0
    gcs_min_depression_deg: float = -9.0
    gcs_los_error_target_mrad: float = 0.2 # 0.2 mrad line-of-sight stabilization
    recoil_peak_force_kn: float = 450.0    # 120mm rifled gun recoil peak
    recoil_duration_s: float = 0.035       # 35 ms hydro-pneumatic buffer time

    # Barrel life is tracked in Equivalent Full Charges (EFC): each round is
    # weighted by its propellant charge energy, so a barrel's remaining life
    # depends on what has been fired through it, not on a raw shot count.
    barrel_life_efc: float = 1500.0        # 120 mm rifled tube condemnation limit
    recoil_stroke_nominal_mm: float = 310.0
    recoil_stroke_limit_mm: float = 370.0  # buffer bottoming risk beyond this

    # APU & NBC System
    apu_rated_power_kw: float = 8.5        # 8.5 kW auxiliary diesel
    apu_bus_voltage_v: float = 28.0        # 28V DC military standard
    nbc_cabin_overpressure_pa: float = 500.0 # 500 Pa positive pressure barrier
    nbc_blower_flow_m3_min: float = 4.5    # 4.5 m^3/min filtered air

    # Sampling & Simulation
    dt: float = 0.1                        # 10 Hz sampling rate
    noise_seed: int = 42
