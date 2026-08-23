"""CVRDE subsystem integrity tests.

Guards the defects that shipped in the CVRDE package: health arrays that
restated the injected fault severity, a suspension sitting several g out of
equilibrium, and unreachable code paths.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tank_sim.cvrde import (CVRDEAuxiliaryNBC, CVRDEHydrogasUnit,
                            CVRDEMissionGenerator, CVRDEPowerpack,
                            CVRDETankConfig)


class TestHydrogasEquilibrium(unittest.TestCase):
    def setUp(self):
        self.cfg = CVRDETankConfig()
        self.hsu = CVRDEHydrogasUnit(1, self.cfg)
        self.static_kn = (self.cfg.combat_mass_kg * 9.81
                          / (self.cfg.roadwheels_per_side * 2) / 1000.0)

    def test_static_force_is_the_right_order_as_wheel_share(self):
        """Using the full bore area gave 249 kN against a 48 kN static share."""
        r = self.hsu.step(road_elevation_m=0.0, tank_velocity_mps=0.0)
        force = r["cvrde_hsu_1_force_kn"]
        self.assertGreater(force, self.static_kn * 0.7)
        self.assertLess(force, self.static_kn * 2.0)

    def test_stationary_vehicle_reads_near_zero_g(self):
        r = self.hsu.step(road_elevation_m=0.0, tank_velocity_mps=0.0)
        self.assertLess(abs(r["cvrde_hsu_1_accel_g"]), 1.0)

    def test_stroke_velocity_is_a_rate_not_a_product(self):
        """A constant stroke implies zero stroke velocity, so damping force
        must vanish.  The old expression multiplied displacement by vehicle
        speed (units m^2/s), so a stationary wheel under a moving vehicle
        produced damping force out of nothing."""
        hsu = CVRDEHydrogasUnit(2, self.cfg)
        hsu.step(road_elevation_m=0.05, tank_velocity_mps=10.0)
        a = hsu.step(road_elevation_m=0.05, tank_velocity_mps=10.0)
        b = hsu.step(road_elevation_m=0.05, tank_velocity_mps=0.0)
        self.assertAlmostEqual(a["cvrde_hsu_2_force_kn"],
                               b["cvrde_hsu_2_force_kn"], places=6)

    def test_seal_leak_lowers_gas_pressure(self):
        hsu = CVRDEHydrogasUnit(3, self.cfg)
        base = hsu.step(0.0, 5.0)["cvrde_hsu_3_pressure_bar"]
        for _ in range(200):
            hsu.step(0.0, 5.0, seal_leak_rate=0.002)
        leaked = hsu.step(0.0, 5.0, seal_leak_rate=0.002)["cvrde_hsu_3_pressure_bar"]
        self.assertLess(leaked, base)


class TestPowerpackCoupling(unittest.TestCase):
    def test_oil_pressure_responds_to_pump_wear_not_turbo_decay(self):
        """Turbocharger decay and oil-pump wear are unrelated subsystems; the
        original model reduced oil pressure from turbo_decay."""
        cfg = CVRDETankConfig()
        a = CVRDEPowerpack(cfg).step(rpm=1800, load=0.6, turbo_decay=0.8)
        b = CVRDEPowerpack(cfg).step(rpm=1800, load=0.6, oil_pump_wear=0.8)
        clean = CVRDEPowerpack(cfg).step(rpm=1800, load=0.6)
        self.assertAlmostEqual(a["cvrde_oil_pressure_bar"],
                               clean["cvrde_oil_pressure_bar"], delta=0.4)
        self.assertLess(b["cvrde_oil_pressure_bar"],
                        clean["cvrde_oil_pressure_bar"] - 1.0)

    def test_injector_wear_opens_the_egt_bank_delta(self):
        """Differential EGT between banks is the documented discriminator for
        cylinder-specific injector faults."""
        cfg = CVRDETankConfig()
        clean = CVRDEPowerpack(cfg).step(rpm=1800, load=0.7)
        worn = CVRDEPowerpack(cfg).step(rpm=1800, load=0.7, injector_wear=1.5)
        d_clean = abs(clean["cvrde_egt_bank_a_c"] - clean["cvrde_egt_bank_b_c"])
        d_worn = abs(worn["cvrde_egt_bank_a_c"] - worn["cvrde_egt_bank_b_c"])
        self.assertGreater(d_worn, d_clean + 40.0)

    def test_fan_efficiency_is_reachable(self):
        """cooling_fan_efficiency was initialised and never written."""
        cfg = CVRDETankConfig()
        p = CVRDEPowerpack(cfg)
        p.step(rpm=2000, load=0.9, fan_efficiency=0.3)
        self.assertAlmostEqual(p.cooling_fan_efficiency, 0.3, places=6)


class TestAuxiliaryApuBranch(unittest.TestCase):
    def test_silent_watch_branch_is_reachable(self):
        """apu_running was always True, so battery discharge was dead code."""
        aux = CVRDEAuxiliaryNBC()
        aux.step(electrical_load_kw=5.0, apu_running=True)
        charged = aux.battery_soc_pct
        for _ in range(50):
            aux.step(electrical_load_kw=5.0, apu_running=False)
        self.assertLess(aux.battery_soc_pct, charged)
        self.assertEqual(aux.step(apu_running=False)["cvrde_apu_rpm"], 0.0)


class TestMissionHealthIsNotCircular(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mission = CVRDEMissionGenerator().generate_thar_desert_mission(60.0)
        cls.health = cls.mission["health"]

    # Structure tracks lifetime barrel wear (EFC), so it legitimately starts
    # below full health on a tube with service history; overall fuses it.
    LIFETIME_PARTS = {"structure", "overall"}

    def test_per_mission_subsystems_start_healthy(self):
        for part, series in self.health.items():
            if part in self.LIFETIME_PARTS:
                continue
            self.assertGreaterEqual(series[0], 90.0, part)

    def test_lifetime_parts_reflect_declared_service_history(self):
        """A fresh tube must start at full structural health; a worn one must not."""
        fresh = CVRDEMissionGenerator().generate_thar_desert_mission(
            60.0, initial_barrel_efc=0.0)
        self.assertGreaterEqual(fresh["health"]["structure"][0], 95.0)
        self.assertLess(self.health["structure"][0], 95.0)
        self.assertEqual(self.mission["meta"]["initial_barrel_efc"], 1150.0)

    def test_barrel_wear_accrues_by_charge_energy(self):
        from tank_sim.cvrde import CVRDEGunControlSystem
        efc = {}
        for rt in ("APFSDS", "HEAT", "PRACTICE"):
            g = CVRDEGunControlSystem()
            g.step(4.5, 2.0, 0.0, 0.0, trigger_fire=True, round_type=rt)
            efc[rt] = g.barrel_efc
        self.assertGreater(efc["APFSDS"], efc["HEAT"])
        self.assertGreater(efc["HEAT"], efc["PRACTICE"])

    def test_recoil_stroke_grows_with_recuperator_gas_loss(self):
        from tank_sim.cvrde import CVRDEGunControlSystem
        tight = CVRDEGunControlSystem()
        tight.step(4.5, 2.0, 0.0, 0.0, trigger_fire=True, recuperator_gas_loss=0.0)
        loose = CVRDEGunControlSystem()
        loose.step(4.5, 2.0, 0.0, 0.0, trigger_fire=True, recuperator_gas_loss=1.0)
        self.assertGreater(loose.recoil_stroke_mm, tight.recoil_stroke_mm + 20.0)

    def test_no_health_series_is_constant(self):
        """Every subsystem card must carry information."""
        flat = [p for p, v in self.health.items() if len(set(v)) == 1]
        self.assertEqual(flat, [], f"constant health series: {flat}")

    def test_no_series_exceeds_full_health(self):
        for part, series in self.health.items():
            self.assertLessEqual(max(series), 100.0, part)

    def test_subsystems_are_not_collinear(self):
        """powertrain was engine*0.9+9.5 and hydraulics was a copy of
        suspension, so three of the eight cards carried no new information."""
        eng = np.array(self.health["engine"])
        ptrn = np.array(self.health["powertrain"])
        self.assertFalse(np.allclose(ptrn, eng * 0.9 + 9.5))
        self.assertNotEqual(self.health["hydraulics"], self.health["suspension"])

    def test_faults_degrade_the_affected_subsystems(self):
        for part in ("engine", "powertrain", "cooling", "suspension"):
            self.assertLess(min(self.health[part]), 70.0, part)

    def test_short_mission_does_not_divide_by_zero(self):
        """Fault onsets were absolute step indices against (n_steps - 400)."""
        short = CVRDEMissionGenerator().generate_thar_desert_mission(20.0)
        self.assertGreater(len(short["records"]), 0)
        self.assertEqual(len(short["health"]["overall"]), len(short["records"]))


if __name__ == "__main__":
    unittest.main()
