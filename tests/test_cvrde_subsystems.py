"""Unit tests for CVRDE Military Tank Subsystems & Simulation Suite."""

import unittest
import sys
from pathlib import Path

# Add project roots
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Fault-Prediction-in-Vehicles"))

from sim.cvrde.cvrde_config import CVRDETankConfig
from sim.cvrde.powerpack import CVRDEPowerpack
from sim.cvrde.hydrogas_suspension import CVRDEHydrogasUnit
from sim.cvrde.gun_control import CVRDEGunControlSystem
from sim.cvrde.auxiliary_nbc import CVRDEAuxiliaryNBC
from sim.cvrde.cvrde_generator import CVRDEMissionGenerator


class TestCVRDESubsystems(unittest.TestCase):
    def setUp(self):
        self.cfg = CVRDETankConfig()

    def test_powerpack_desert_derating_and_egt(self):
        """Verify powerpack derating under +50 C ambient and dual EGT pyrometry."""
        pp = CVRDEPowerpack(self.cfg)
        rec_nominal = pp.step(rpm=2200.0, load=0.8, ambient_temp_c=25.0)
        
        pp_desert = CVRDEPowerpack(self.cfg)
        rec_desert = pp_desert.step(rpm=2200.0, load=0.8, ambient_temp_c=50.0, sand_clog_rate=0.05)
        
        self.assertGreater(rec_desert["cvrde_coolant_temp_c"], rec_nominal["cvrde_coolant_temp_c"])
        self.assertGreater(rec_desert["cvrde_egt_bank_a_c"], 400.0)
        self.assertGreater(rec_nominal["cvrde_boost_pressure_bar"], 1.5)

    def test_hydrogas_adiabatic_spring_and_leakage(self):
        """Verify CVRDE Hydrogas Unit adiabatic gas spring P*V^gamma behavior."""
        hsu = CVRDEHydrogasUnit(station_id=1, cfg=self.cfg)
        
        # Nominal neutral stroke (x = 0)
        r0 = hsu.step(road_elevation_m=0.0, tank_velocity_mps=10.0)
        p0 = r0["cvrde_hsu_1_pressure_bar"]
        
        # Compression bump (x = +0.10 m)
        r_bump = hsu.step(road_elevation_m=0.10, tank_velocity_mps=10.0)
        p_bump = r_bump["cvrde_hsu_1_pressure_bar"]
        
        # Rebound droop (x = -0.10 m)
        r_rebound = hsu.step(road_elevation_m=-0.10, tank_velocity_mps=10.0)
        p_rebound = r_rebound["cvrde_hsu_1_pressure_bar"]
        
        self.assertGreater(p_bump, p0, "Adiabatic gas pressure must increase on compression")
        self.assertLess(p_rebound, p0, "Adiabatic gas pressure must decrease on rebound")

    def test_gun_control_recoil_impulse(self):
        """Verify 120mm main gun recoil shock force and buffer pressure."""
        gcs = CVRDEGunControlSystem(self.cfg)
        
        # Steady tracking
        r_steady = gcs.step(elev_cmd_deg=5.0, azim_rate_dps=1.0, hull_pitch_deg=0.0, hull_roll_deg=0.0)
        self.assertEqual(r_steady["cvrde_gcs_recoil_force_kn"], 0.0)
        self.assertEqual(r_steady["cvrde_gcs_rounds_fired"], 0)
        
        # Fire main gun
        r_fire = gcs.step(elev_cmd_deg=5.0, azim_rate_dps=1.0, hull_pitch_deg=0.0, hull_roll_deg=0.0, trigger_fire=True)
        self.assertGreater(r_fire["cvrde_gcs_recoil_force_kn"], 400.0)
        self.assertGreater(r_fire["cvrde_gcs_recoil_shock_g"], 15.0)
        self.assertEqual(r_fire["cvrde_gcs_rounds_fired"], 1)

    def test_auxiliary_nbc_positive_overpressure(self):
        """Verify APU 28V DC bus charging and 500 Pa NBC cabin protection."""
        apu = CVRDEAuxiliaryNBC(self.cfg)
        rec = apu.step(electrical_load_kw=3.0, nbc_blower_on=True)
        
        self.assertAlmostEqual(rec["cvrde_bus_voltage_v"], 28.8, delta=2.0)
        self.assertGreater(rec["cvrde_nbc_overpressure_pa"], 400.0)
        self.assertGreater(rec["cvrde_apu_oil_pressure_bar"], 2.5)

    def test_cvrde_mission_generator_output(self):
        """Verify multi-subsystem Thar desert assault mission generation."""
        gen = CVRDEMissionGenerator(self.cfg)
        mission = gen.generate_thar_desert_mission(duration_s=10.0)
        
        self.assertIn("records", mission)
        self.assertIn("health", mission)
        self.assertEqual(len(mission["records"]), 100) # 10s @ 10 Hz = 100 steps
        self.assertIn("hyd_pressure", mission["records"][0])
        self.assertIn("hsu_l_stroke_mm", mission["records"][0])
        self.assertIn("gcs_hyd_pressure", mission["records"][0])
        self.assertIn("nbc_overpressure_pa", mission["records"][0])


if __name__ == "__main__":
    unittest.main()
