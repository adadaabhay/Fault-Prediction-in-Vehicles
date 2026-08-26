"""Default parameters of the simulated battle-tank model.

All values are representative for a heavy tracked military vehicle
(e.g. a main battle tank) and are used by the physics modules to turn
raw physical quantities into realistic sensor readings.
"""

from dataclasses import dataclass, field


@dataclass
class TankConfig:
    # --- Universal constants ----------------------------------------------------
    R_UNIVERSAL: float = 8.314          # J/(mol K)
    P_REF_AIR: float = 20e-6            # Pa, acoustic reference pressure
    EPSILON_0: float = 8.854e-12        # F/m, vacuum permittivity

    # --- Engine -----------------------------------------------------------------
    engine_displacement: float = 28.0      # L (diesel V12)
    cylinders: int = 12                    # for firing-frequency acoustics
    max_speed_rpm: float = 2600.0          # RPM
    idle_speed_rpm: float = 800.0          # RPM
    engine_mass_thermal: float = 120.0     # kg (effective thermal mass)
    c_p_coolant: float = 4180.0            # J/(kg K) water-glycol coolant
    max_fuel_energy_rate: float = 2.5e6    # J/s full-throttle heat release
    brake_thermal_efficiency: float = 0.42 # shaft power / fuel heat release
    coolant_power: float = 6000.0          # J/s per K of coolant delta-T (radiator + fan + oil cooler)
    # --- Exhaust thermodynamics -------------------------------------------
    exhaust_heat_fraction: float = 0.32    # of fuel heat release leaving via exhaust
    c_p_exhaust_gas: float = 1150.0        # J/(kg K) combustion products
    exhaust_mdot_ref: float = 1.35         # kg/s reference exhaust mass flow
    lambda_reference: float = 1.8          # typical diesel part-load excess air
    exhaust_tau_s: float = 4.0             # s manifold/turbine thermal lag
    max_egt_c: float = 760.0               # pyrometer limit (cf. CVRDE 750 C)
    exhaust_port_soak_c: float = 25.0      # K, exhaust port above head metal temp
    # --- Cooling circuit regulation ---------------------------------------
    thermostat_open_c: float = 88.0        # C, valve cracks open
    thermostat_range_c: float = 10.0       # K, fully open at open_c + range
    thermostat_bypass_leak: float = 0.06   # residual rejection when closed

    # --- Vibration --------------------------------------------------------------
    n_bearings: int = 12                   # rolling elements per bearing
    bearing_pitch_d: float = 0.130         # m
    ball_d: float = 0.024                  # m
    contact_angle: float = 15.0            # deg
    drive_pinion_teeth: int = 18           # Z_p gear-mesh
    shaft_amp_base: float = 0.6            # m/s^2 base sinusoidal amplitude
    vibration_noise: float = 0.05          # m/s^2 sensor noise sigma

    # --- Lubrication oil --------------------------------------------------------
    oil_density: float = 860.0             # kg/m^3
    oil_A: float = 1.6e-6                  # Pa*s Arrhenius pre-exponential
    oil_E: float = 26000.0                 # J/mol activation energy
    oil_ref_temp: float = 60.0             # deg C reference for nominal mu
    filter_r: float = 0.004                # m filter passage radius
    filter_L: float = 0.06                 # m filter passage length
    main_gallery_r: float = 0.008          # m main oil gallery radius
    main_gallery_L: float = 0.5            # m main oil gallery length
    pump_discharge_pressure: float = 5.5e5 # Pa nominal pump head
    oil_pressure_noise: float = 0.02e5     # Pa
    # Flowmeter noise, ~2% of the nominal 0.6-1.8e-3 m^3/s gallery flow.  Every
    # transducer needs a noise term: a channel with none is not a measurement,
    # it is a readback of whatever parameter produced it, and a fault-injection
    # parameter published that way is a label leak.  See tests/test_leakage.py.
    oil_flow_noise: float = 2.0e-5         # m^3/s

    # --- Oil debris -------------------------------------------------------------
    debris_gain: float = 1.0               # particles per second baseline
    debris_noise: float = 0.2

    # --- Torque / drivetrain ----------------------------------------------------
    gear_ratio: float = 12.0               # overall drivetrain ratio
    drive_r: float = 0.4                   # m sprocket / track drive radius
    shaft_radius: float = 0.06             # m
    # Solid circular shaft: J = pi r^4 / 2 for shaft_radius = 0.06 m.
    shaft_J: float = 2.036e-5              # m^4 polar moment of inertia
    shaft_shear_modulus: float = 79e9      # Pa steel G
    torque_efficiency: float = 0.92        # nominal drivetrain efficiency
    torque_noise: float = 15.0             # N*m

    # --- Exhaust ----------------------------------------------------------------
    exhaust_area: float = 0.03             # m^2
    exhaust_Rs: float = 287.0              # J/(kg K) for air-combustion mix
    stoich_afr: float = 14.5               # diesel stoichiometric A/F ratio
    lambda_idle: float = 5.5               # excess-air ratio at no load
    lambda_rated: float = 1.25             # excess-air ratio at rated power (smoke limit)
    lambda_noise: float = 0.01
    mass_flow_noise: float = 0.015         # relative, hot-film flow noise
    exhaust_pressure_base: float = 1.25e5  # Pa

    # --- Fuel / fluid levels ----------------------------------------------------
    fuel_tank_r: float = 0.30              # m
    fuel_tank_h: float = 0.90              # m
    fuel_permittivity: float = 2.1         # relative dielectric of diesel
    oil_sump_h: float = 0.35               # m oil level equivalent height
    coolant_h: float = 0.40                # m coolant expansion tank height
    fuel_burn_rate: float = 1.6e-4         # m^3/s full-load fuel burn
    level_noise: float = 0.004             # fraction, capacitive probe noise

    # --- Hydraulics -------------------------------------------------------------
    hyd_pump_pressure: float = 2.1e7       # Pa (turret/stabilizer circuit)
    hyd_valve_area: float = 2.0e-4         # m^2
    # Command flow is hyd_valve_area * cmd ~ 1e-4 m^3/s; sensor noise must be a
    # small fraction of that, not 25-50x it (which forced ~49% of samples to
    # clip at exactly zero and made seal-leak detection impossible).
    hyd_flow_noise: float = 4.0e-6         # m^3/s
    hyd_leak_area: float = 6.0e-6          # m^2 seal-leak equivalent area
    hyd_leak_noise: float = 8.0e-7         # m^3/s return-line flowmeter noise

    # --- Suspension / structure -------------------------------------------------
    suspension_E: float = 200e9            # Pa
    torsion_L: float = 1.8                 # m torsion bar length
    torsion_r: float = 0.03                # m torsion bar radius
    torsion_G: float = 79e9                # Pa
    strain_gauge_GF: float = 2.1           # gauge factor
    strain_gauge_R: float = 350.0          # Ohm nominal resistance
    suspension_area: float = 0.004         # m^2 effective loaded section
    vehicle_mass: float = 58000.0          # kg
    roadwheel_stations: int = 14           # 7 per side
    suspension_noise: float = 0.5          # kN load-cell noise
    strain_noise_ue: float = 1.5           # microstrain, bridge excitation noise

    # --- Acoustic ---------------------------------------------------------------
    acoustic_base_spl: float = 105.0       # dB baseline SPL
    acoustic_noise: float = 0.4            # dB
    ae_event_rate_base: float = 2.0        # events/s baseline
    ae_noise: float = 0.3                  # dB, AE amplitude readout noise
    ae_energy_noise: float = 0.25          # lognormal sigma on burst energy

    # --- Control / sampling -----------------------------------------------------
    dt: float = 0.05                       # s simulation time step
    # Gear-mesh frequency reaches drive_pinion_teeth * max_speed_rpm / 60
    # = 18 * 2600 / 60 = 780 Hz, so the burst rate must clear 1560 Hz to avoid
    # folding the mesh tone (and its sidebands) back onto the shaft orders.
    sample_rate: float = 4000.0            # Hz for high-frequency bursts
    window_samples: int = 2048             # samples per feature window
    noise_seed: int = 42