"""Master script to generate all subsystem simulated datasets."""

from sim.scripts import engine, hydraulics, suspension, gun_control, nbc, fuel_levels, exhaust, acoustics

def main():
    steps = 2000
    print(f"Generating all subsystem datasets ({steps} steps)...")
    engine.generate("results/simulated/engine.csv", steps=steps)
    hydraulics.generate("results/simulated/hydraulics.csv", steps=steps)
    suspension.generate("results/simulated/suspension.csv", steps=steps)
    gun_control.generate("results/simulated/gun_control.csv", steps=steps)
    nbc.generate("results/simulated/nbc.csv", steps=steps)
    fuel_levels.generate("results/simulated/fuel_levels.csv", steps=steps)
    exhaust.generate("results/simulated/exhaust.csv", steps=steps)
    acoustics.generate("results/simulated/acoustics.csv", steps=steps)
    print("All datasets generated successfully in results/simulated/")

if __name__ == "__main__":
    main()
