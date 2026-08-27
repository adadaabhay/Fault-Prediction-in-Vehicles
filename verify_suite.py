"""Final verification suite — structure, imports, bug fixes, generators, config."""
import json
import os
import importlib
import tempfile
import csv
from pathlib import Path

root = Path(".")
PASS = "PASS"
FAIL = "FAIL"
results = []

def chk(label, ok, detail=""):
    status = PASS if ok else FAIL
    results.append((status, label, detail))
    print(f"  {status}  {label}" + (f" | {detail}" if detail else ""))

# ── 1. Repo structure ────────────────────────────────────────────────────────
print("\n── 1. Repo Structure ──")
chk("sim/ exists",                (root / "sim").is_dir())
chk("sim/scripts/ exists",        (root / "sim" / "scripts").is_dir())
chk("sim/cvrde/ exists",          (root / "sim" / "cvrde").is_dir())
chk("sim/physics/ exists",        (root / "sim" / "physics").is_dir())
chk("tank_sim/ is gone",          not (root / "tank_sim").exists())
chk("docs/config.json exists",    (root / "docs" / "config.json").exists())
chk("docs/model.json exists",     (root / "docs" / "model.json").exists())
chk("docs/index.html exists",     (root / "docs" / "index.html").exists())
chk("ml/constants.py exists",     (root / "ml" / "constants.py").exists())
chk("requirements.txt exists",    (root / "requirements.txt").exists())

scripts = ["engine","hydraulics","suspension","gun_control","nbc","fuel_levels","exhaust","acoustics","generate_all"]
for s in scripts:
    chk(f"sim/scripts/{s}.py", (root / "sim" / "scripts" / f"{s}.py").exists())

# ── 2. Python imports ────────────────────────────────────────────────────────
print("\n── 2. Python Imports ──")
try:
    from sim.tank import TankSimulator
    from sim.config import TankConfig
    from sim.cvrde.cvrde_generator import CVRDEMissionGenerator
    from ml.parts import PARTS, PART_ORDER
    from ml.constants import DEFAULT_W_CLS, DEFAULT_W_REG
    chk("Core sim + ml imports", True)
except Exception as e:
    chk("Core sim + ml imports", False, str(e))

# ── 3. config.json ───────────────────────────────────────────────────────────
print("\n── 3. config.json ──")
cfg = json.load(open("docs/config.json"))
expected = ["engine","powertrain","lubrication","cooling","hydraulics","suspension","structure","nbc","exhaust","acoustics","overall"]
chk("part_order == 11 parts", cfg["part_order"] == expected, str(cfg["part_order"]))
chk("nbc in parts",      "nbc" in cfg["parts"])
chk("exhaust in parts",  "exhaust" in cfg["parts"])
chk("acoustics in parts","acoustics" in cfg["parts"])

# ── 4. live_multi_streams.json health completeness ──────────────────────────
print("\n── 4. live_multi_streams.json ──")
streams = json.load(open("docs/live_multi_streams.json"))
for sid, stream in streams["streams"].items():
    n = len(stream["records"])
    missing = [p for p in cfg["part_order"] if p not in stream["health"]]
    wrong_len = [p for p in cfg["part_order"] if p in stream["health"] and len(stream["health"][p]) != n]
    chk(f"stream {sid} health keys",    not missing,   f"missing={missing}" if missing else "")
    chk(f"stream {sid} health lengths", not wrong_len, f"wrong_len={wrong_len}" if wrong_len else "")

# ── 5. requirements pinned ───────────────────────────────────────────────────
print("\n── 5. Requirements ──")
reqs = open("requirements.txt").read()
unpinned = [l.strip() for l in reqs.splitlines() if l.strip() and "==" not in l and not l.startswith("#")]
chk("All deps pinned with ==", not unpinned, str(unpinned) if unpinned else "")
chk("pyserial==3.5 present",   "pyserial==3.5" in reqs)
chk("websockets==14.1 present","websockets==14.1" in reqs)

# ── 6. Bug fix spot-checks ──────────────────────────────────────────────────
print("\n── 6. Bug Fix Spot-Checks ──")
server = open("telemetry_gateway/server.py", encoding="utf-8").read()
chk("server.py EFL_P1 * 100.0 (bar->kPa)",   "* 100.0" in server)
chk("server.py _RATE_STATE_MAX eviction",     "_RATE_STATE_MAX" in server)

dtc = open("telemetry_gateway/dtc_engine.py", encoding="utf-8").read()
chk("dtc_engine.py _flash_line_count O(1)",   "_flash_line_count" in dtc)
chk("dtc_engine.py max_age_hours staleness",  "max_age_hours" in dtc)

ingest_lines = open("telemetry_gateway/live_sensor_ingest.py", encoding="utf-8").readlines()
hw_whitelist_lines = [l for l in ingest_lines if "is_hw" in l and "source.lower()" in l]
http_in_whitelist = any('"http"' in l for l in hw_whitelist_lines)
chk("live_sensor_ingest.py http removed from HW whitelist", not http_in_whitelist)

c_src = open("c_engine/tank_pdm_infer.c", encoding="utf-8").read()
chk("c_engine fast_softmax len<=0 guard", "len <= 0" in c_src)
chk("c_engine shadow var ci (not c)",     "int ci " in c_src or "ci = 0" in c_src)

ml_const = open("ml/constants.py", encoding="utf-8").read()
chk("ml/constants.py DEFAULT_W_CLS", "DEFAULT_W_CLS" in ml_const)
chk("ml/constants.py DEFAULT_W_REG", "DEFAULT_W_REG" in ml_const)

# ── 7. All 8 generators E2E ──────────────────────────────────────────────────
print("\n── 7. Generator E2E ──")
GEN_COLS = {
    "engine":       9,
    "hydraulics":   7,
    "suspension":   6,
    "gun_control":  6,
    "nbc":          6,
    "fuel_levels":  6,
    "exhaust":      6,
    "acoustics":    8,
}
for name, expected_cols in GEN_COLS.items():
    try:
        mod = importlib.import_module(f"sim.scripts.{name}")
        tmp = tempfile.mktemp(suffix=".csv")
        mod.generate(tmp)
        with open(tmp) as f:
            rows = list(csv.reader(f))
        os.unlink(tmp)
        cols_ok = len(rows[0]) == expected_cols
        rows_ok = len(rows) > 1000
        chk(f"sim/scripts/{name}.py", cols_ok and rows_ok,
            f"{len(rows[0])} cols, {len(rows)-1} rows")
    except Exception as e:
        chk(f"sim/scripts/{name}.py", False, str(e))

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n── Summary ──")
failed = [r for r in results if r[0] == FAIL]
print(f"Total: {len(results)} checks | PASS: {len(results)-len(failed)} | FAIL: {len(failed)}")
if failed:
    print("\nFailed checks:")
    for s, l, d in failed:
        print(f"  FAIL  {l}  {d}")
    exit(1)
else:
    print("All checks passed.")
