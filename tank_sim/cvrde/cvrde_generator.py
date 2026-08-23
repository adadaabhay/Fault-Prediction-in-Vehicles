import os
import sys
import json
from pathlib import Path
import numpy as np

# Add tank_sim to sys.path for direct execution
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(current_dir))

try:
    from .cvrde_config import CVRDETankConfig
    from .powerpack import CVRDEPowerpack
    from .hydrogas_suspension import CVRDEHydrogasUnit
    from .gun_control import CVRDEGunControlSystem
    from .auxiliary_nbc import CVRDEAuxiliaryNBC
except ImportError:
    from cvrde_config import CVRDETankConfig
    from powerpack import CVRDEPowerpack
    from hydrogas_suspension import CVRDEHydrogasUnit
    from gun_control import CVRDEGunControlSystem
    from auxiliary_nbc import CVRDEAuxiliaryNBC


def _score(value: float, warn: float, crit: float) -> float:
    """0-100 health for a rising-is-bad channel, anchored to its limits.

    ``warn`` scores 100 or better; ``crit`` scores 25 (the project-wide failure
    threshold) so the health curve and the threshold alarms agree by
    construction.  Pass negated values for falling-is-bad channels.
    """
    span = max(crit - warn, 1e-9)
    dev = max((value - warn) / span, 0.0)
    return float(np.clip(100.0 - 75.0 * dev, 0.0, 100.0))


class CVRDEMissionGenerator:
    def __init__(self, cfg: CVRDETankConfig | None = None):
        self.cfg = cfg or CVRDETankConfig()

    def generate_thar_desert_mission(self, duration_s: float = 60.0,
                                     initial_barrel_efc: float = 1150.0) -> dict:
        """Generates a 10 Hz Thar Desert (+50 C) combat assault scenario.

        ``initial_barrel_efc`` is the tube's accumulated service history in
        Equivalent Full Charges. A fielded gun arrives with wear already on it,
        and barrel life is a lifetime quantity, not a per-mission one -- three
        rounds inside a 60 second sortie is not degradation. The default places
        the tube near the end of its serviceable life so the structural channel
        reflects a real maintenance decision.
        """
        cfg = self.cfg
        n_steps = int(duration_s / cfg.dt)
        
        powerpack = CVRDEPowerpack(cfg)
        hsu_left = CVRDEHydrogasUnit(station_id=1, cfg=cfg)
        hsu_right = CVRDEHydrogasUnit(station_id=2, cfg=cfg)
        gcs = CVRDEGunControlSystem(cfg)
        gcs.barrel_efc = float(initial_barrel_efc)
        apu = CVRDEAuxiliaryNBC(cfg)
        
        records = []
        health = {
            "overall": [],
            "engine": [],
            "powertrain": [],
            "lubrication": [],
            "cooling": [],
            "hydraulics": [],
            "suspension": [],
            "structure": [],
        }
        
        for i in range(n_steps):
            t = i * cfg.dt
            
            # Combat driving cycle (Thar Desert dunes)
            speed_kmh = 25.0 + 20.0 * np.sin(2 * np.pi * 0.02 * t)
            rpm = 1200.0 + (speed_kmh / 50.0) * 1100.0
            load = 0.4 + 0.45 * (speed_kmh / 50.0)
            ambient_t = 48.0 + 4.0 * (i / n_steps) # Heat buildup to 52 C
            
            # Injected anomalies after step 400 (Sand filter clogging & HSU Station 1 nitrogen seal degradation)
            # Fault onsets are expressed as fractions of the run so a shorter
            # mission does not divide by zero (or index past the end).
            onset_sand, onset_turbo, onset_hsu = 0.63, 0.67, 0.70
            i_frac = i / max(n_steps - 1, 1)
            sand_rate = 0.08 if i_frac > onset_sand else 0.01
            turbo_decay = (0.6 * (i_frac - onset_turbo) / max(1.0 - onset_turbo, 1e-6)
                           if i_frac > onset_turbo else 0.0)
            hsu_leak = 0.001 if i_frac > onset_hsu else 0.0
            # Bank-A injector fouling: drives the EGT bank delta, which is the
            # documented discriminator for cylinder-specific combustion faults.
            onset_inj = 0.55
            injector_wear = (1.6 * (i_frac - onset_inj) / max(1.0 - onset_inj, 1e-6)
                             if i_frac > onset_inj else 0.0)
            
            # Firing main gun at t = 20s, t = 35s, t = 48s
            trigger_fire = i in {int(n_steps * f) for f in (0.33, 0.58, 0.80)}
            
            # 1. Step Subsystems
            p_rec = powerpack.step(rpm=rpm, load=load, ambient_temp_c=ambient_t,
                                  sand_clog_rate=sand_rate, turbo_decay=turbo_decay,
                                  injector_wear=injector_wear,
                                  oil_pump_wear=min(turbo_decay * 0.5, 0.6),
                                  fan_efficiency=max(0.35, 1.0 - turbo_decay * 0.4))
            
            # Terrain wave for hydrogas units
            terrain_z = 0.08 * np.sin(2 * np.pi * 0.5 * t) + (0.12 if i % 60 < 5 else 0.0)
            hsu_l_rec = hsu_left.step(road_elevation_m=terrain_z, tank_velocity_mps=speed_kmh/3.6,
                                      seal_leak_rate=hsu_leak)
            hsu_r_rec = hsu_right.step(road_elevation_m=-terrain_z*0.8, tank_velocity_mps=speed_kmh/3.6)
            
            # Recuperator gas loss grows with firing exposure, lengthening the
            # recoil stroke -- the structural wear channel for the gun mount.
            gas_loss = min(gcs.barrel_efc / 900.0, 1.0)
            gcs_rec = gcs.step(elev_cmd_deg=4.5, azim_rate_dps=2.0, hull_pitch_deg=terrain_z*50.0,
                               hull_roll_deg=0.5, trigger_fire=trigger_fire,
                               recuperator_gas_loss=gas_loss)
            
            apu_rec = apu.step(electrical_load_kw=4.2, nbc_blower_on=True, filter_dust_load=sand_rate)
            
            # Combine into standardized telemetry dictionary
            combined_rec = {
                # Engine & Powertrain
                "rpm": float(rpm),
                "load": float(load),
                "oil_pressure": float(p_rec["cvrde_oil_pressure_bar"] * 1e5),
                "oil_temp": float(p_rec["cvrde_oil_temp_c"]),
                "coolant_temp": float(p_rec["cvrde_coolant_temp_c"]),
                "coolant_level": 0.85,
                "boost_pressure": float(p_rec["cvrde_boost_pressure_bar"] * 1e5),
                "intercooler_out_temp": float(p_rec["cvrde_intercooler_out_temp_c"]),
                "egt_bank_a": float(p_rec["cvrde_egt_bank_a_c"]),
                "egt_bank_b": float(p_rec["cvrde_egt_bank_b_c"]),
                "exhaust_temp": float(p_rec["cvrde_egt_bank_a_c"]),
                "rail_pressure": float(p_rec["cvrde_rail_pressure_bar"] * 1e5),
                
                # CVRDE Hydrogas Suspension Units (HSU)
                "hyd_pressure": float(hsu_l_rec["cvrde_hsu_1_pressure_bar"] * 1e5),
                "hsu_l_stroke_mm": float(hsu_l_rec["cvrde_hsu_1_stroke_mm"]),
                "hsu_r_stroke_mm": float(hsu_r_rec["cvrde_hsu_2_stroke_mm"]),
                "hsu_seal_health": float(hsu_l_rec["cvrde_hsu_1_seal_health_pct"]),
                
                # Gun Control & Recoil
                "gcs_hyd_pressure": float(gcs_rec["cvrde_gcs_hyd_pressure_bar"] * 1e5),
                "recoil_force_kn": float(gcs_rec["cvrde_gcs_recoil_force_kn"]),
                "los_error_urad": float(gcs_rec["cvrde_gcs_los_error_urad"]),
                "barrel_efc": float(gcs_rec["cvrde_gcs_barrel_efc"]),
                "barrel_life_pct": float(gcs_rec["cvrde_gcs_barrel_life_pct"]),
                "recoil_stroke_mm": float(gcs_rec["cvrde_gcs_recoil_stroke_mm"]),
                
                # APU & NBC System
                "apu_rpm": float(apu_rec["cvrde_apu_rpm"]),
                "bus_voltage_v": float(apu_rec["cvrde_bus_voltage_v"]),
                "nbc_overpressure_pa": float(apu_rec["cvrde_nbc_overpressure_pa"]),
                
                # Dynamic Vibration & Shock
                "vib_rms": float(0.40 + (0.8 if trigger_fire else (0.1 if i > 400 else 0.0))),
                "terrain": 0.45,
            }
            records.append(combined_rec)
            
            # Subsystem Health indices (0..100)
            # Scored from the telemetry against physical limits, not read back
            # from the injected severity. Deriving health from the fault
            # parameters made the label a restatement of the ground truth.
            # Loaded diesel EGT normally sits 600-700 C; only the approach to the
            # pyrometer limit is a fault.
            eng_h = _score(p_rec["cvrde_egt_bank_a_c"], cfg.max_egt_c * 0.87, cfg.max_egt_c)
            cool_h = _score(p_rec["cvrde_coolant_temp_c"], 95.0, cfg.max_coolant_temp_c)
            lube_h = min(_score(p_rec["cvrde_oil_temp_c"], 100.0, 135.0),
                         _score(-p_rec["cvrde_oil_pressure_bar"], -4.2, -2.5))
            bank_delta = abs(p_rec["cvrde_egt_bank_a_c"] - p_rec["cvrde_egt_bank_b_c"])
            ptrn_h = min(_score(bank_delta, 25.0, 70.0),
                         _score(p_rec["cvrde_air_filter_clog_pct"], 35.0, 85.0))
            hsu_h = _score(-hsu_l_rec["cvrde_hsu_1_pressure_bar"],
                           -cfg.hsu_n2_precharge_bar * 0.92,
                           -cfg.hsu_n2_precharge_bar * 0.60)
            # LOS error rises with hull motion on the move; the fault case is
            # sustained error well beyond normal cross-country disturbance.
            gcs_h = _score(gcs_rec["cvrde_gcs_los_error_urad"], 950.0, 1800.0)
            # Structural condition of the gun mount: cumulative barrel wear in
            # EFC plus recoil-stroke growth from recuperator gas loss. Peak
            # recoil force alone was constant, because nominal firing is not
            # degradation -- it is the accumulated exposure that is.
            struct_h = min(
                _score(gcs_rec["cvrde_gcs_barrel_efc"],
                       cfg.barrel_life_efc * 0.70, cfg.barrel_life_efc),
                _score(gcs_rec["cvrde_gcs_recoil_stroke_mm"],
                       cfg.recoil_stroke_nominal_mm * 1.03, cfg.recoil_stroke_limit_mm))
            subs = [eng_h, ptrn_h, lube_h, cool_h, gcs_h, hsu_h, struct_h]
            overall_h = 0.6 * min(subs) + 0.4 * (sum(subs) / len(subs))
            
            health["engine"].append(round(eng_h, 1))
            health["powertrain"].append(round(ptrn_h, 1))
            health["lubrication"].append(round(lube_h, 1))
            health["cooling"].append(round(cool_h, 1))
            health["hydraulics"].append(round(gcs_h, 1))
            health["suspension"].append(round(hsu_h, 1))
            health["structure"].append(round(struct_h, 1))
            health["overall"].append(round(overall_h, 1))
            
        return {
            "meta": {
                "name": "CVRDE Arjun Mk-1A Thar Desert Assault (50C)",
                "description": ("High-thermal desert offensive with sand filter choking, "
                                "injector bank-A fouling, HSU nitrogen seal drift and 120mm "
                                "main gun recoil cycles. Barrel enters the sortie with "
                                f"{initial_barrel_efc:.0f} EFC of accumulated service life."),
                "initial_barrel_efc": float(initial_barrel_efc),
                "faults": ["sand_filter_flow_loss", "turbo_boost_decay", "injector_fouling_bank_a",
                       "oil_pump_wear", "hsu_n2_seal_drift", "120mm_firing_recoil_shocks"],
            },
            "records": records,
            "health": health
        }


def update_live_multi_streams_with_cvrde():
    """Appends CVRDE Arjun Mk-1A stream to live_multi_streams.json."""
    # Resolve relative to this file, not the caller's working directory.
    path = str(Path(__file__).resolve().parents[2] / "docs" / "live_multi_streams.json")
    gen = CVRDEMissionGenerator()
    cvrde_mission = gen.generate_thar_desert_mission(duration_s=60.0)
    
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"metadata": {"streams_available": []}, "streams": {}}
        
    # Add metadata item if not present
    avail = data["metadata"]["streams_available"]
    if not any(s["id"] == "cvrde_arjun" for s in avail):
        avail.append({
            "id": "cvrde_arjun",
            "label": "CVRDE Arjun Mk-1A (Desert 50C)",
            "origin": "CVRDE-parameterised physics simulation",
            "kind": "simulation",
        })
        
    data["streams"]["cvrde_arjun"] = cvrde_mission
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Successfully integrated CVRDE Arjun Mk-1A stream into {path} ({os.path.getsize(path)/(1024*1024):.2f} MB)")


if __name__ == "__main__":
    update_live_multi_streams_with_cvrde()
