"""Run the E1 main experiment (5 seeds) and the E1b missing-data sensitivity grid in parallel."""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "results", "logs")
os.makedirs(LOG, exist_ok=True)
jobs = []
for s in range(1, 6):
    jobs.append(["--seed", str(s), "--n", "2000", "--samples", "1000", "--mode", "full"])
for mech in ("MCAR", "MNAR"):
    for m in ("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"):
        for s in (1, 2):
            jobs.append(["--seed", str(s), "--n", "2000", "--missing", m, "--mech", mech, "--samples", "500",
                         "--methods", "M2,M2c,M3,M3-noML", "--mode", "summary",
                         "--out", os.path.join(HERE, "..", "results", "e1b")])


def run(args):
    name = "_".join(a.replace("-", "").replace(".", "") for a in args[:8] if not a.startswith("--"))
    env = dict(os.environ, OMP_NUM_THREADS="3")
    with open(os.path.join(LOG, f"e1_{name}.log"), "w") as f:
        return subprocess.call([sys.executable, os.path.join(HERE, "run_e1.py"), *args], stdout=f, stderr=subprocess.STDOUT,
                               cwd=HERE, env=env)


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    with ThreadPoolExecutor(workers) as ex:
        codes = list(ex.map(run, jobs))
    print("exit codes:", codes)
