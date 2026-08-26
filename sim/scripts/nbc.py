"""Nbc subsystem simulated sensor data generator."""

import csv
from pathlib import Path
from sim.config import TankConfig
from sim.tank import TankSimulator

def generate(out_path: str, steps: int = 2000, fault_rate: float = 0.05):
    print(f"Generating Nbc subsystem data -> {out_path}")
    cfg = TankConfig()
    sim = TankSimulator(cfg=cfg, seed=42)
    records = sim.run()
    
    # Filter columns relevant to this subsystem
    keep_cols = ['time', 'step', 'nbc_overpressure', 'nbc_fan_rpm', 'nbc_filter_dp', 'nbc_valve_state']
    
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keep_cols)
        for r in records:
            writer.writerow([r.get(c, 0.0) for c in keep_cols])
    print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/simulated/nbc.csv")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    generate(args.out, args.steps)
