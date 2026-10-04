"""E7: scalability and system performance of the single-node prototype.

For increasing numbers of batch passports: scenario generation, analytics
execution (M2, M3) with stage breakdown, compliance evaluation, provenance
graph construction and loading, event-store ingestion throughput; API latency
and throughput at the largest provenance-enabled size.

Wall-clock times on the test laptop varied by a factor of several between
identical runs (probably CPU power management and core scheduling), so every size is
measured in several rounds, with the order of sizes shuffled per round; the
analysis reports the median and range. Each measurement is appended to
``scalability_runs.jsonl`` as soon as it is complete.
Run alone (no concurrent workloads) for representative timings.

Usage: python run_e7.py [sizes] [--rounds R] [--api]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import RESULTS, save_json  # noqa: E402
from tsa.synth import generate, ScenarioConfig  # noqa: E402
from tsa.engine import Engine, METHODS, observed_view  # noqa: E402
from tsa.compliance import ComplianceEngine  # noqa: E402
from tsa.factors import build_factor_db  # noqa: E402
from tsa import provenance, interop, store  # noqa: E402

OUT = os.path.join(RESULTS, "e7")
os.makedirs(OUT, exist_ok=True)
RUNS = os.path.join(OUT, "scalability_runs.jsonl")
PROV_MAX = 20000
API_SIZE = 20000


def _rm(db):
    for ext in ("", "-wal", "-shm"):
        try:
            os.remove(db + ext)
        except OSError:
            pass


def measure(n: int, api: bool) -> dict:
    row = dict(n_batches=n, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    t = time.perf_counter(); sc = generate(ScenarioConfig(n_batches=n, seed=41)); row["generate_s"] = time.perf_counter() - t
    v = observed_view(sc)
    row.update(events=len(v.events), observations=len(v.observations), claims=len(v.claims))
    for k in ("M2", "M3"):
        c = time.process_time()
        r = Engine(v, METHODS[k], n_samples=200, seed=41).run()
        row[f"{k}_cpu_s"] = time.process_time() - c
        row[f"{k}_total_s"] = r.timings["total"]
        for part in ("screen", "gapfill", "contributions", "propagate", "other_indicators"):
            row[f"{k}_{part}_s"] = r.timings.get(part, 0.0)
    t = time.perf_counter(); ComplianceEngine(v).run(); row["compliance_s"] = time.perf_counter() - t
    db = os.path.join(tempfile.gettempdir(), f"tsa_store_{n}.sqlite")   # local disk
    _rm(db)
    S = store.DppStore(db)
    row["event_ingest_s"] = S.append_events(v.events)
    row["event_ingest_per_s"] = len(v.events) / row["event_ingest_s"]
    row["obs_ingest_s"] = S.load_observations(v.observations)
    row["obs_ingest_per_s"] = len(v.observations) / row["obs_ingest_s"]
    if n <= PROV_MAX:
        t = time.perf_counter(); N, E = provenance.build_graph(r, v, build_factor_db()); row["prov_build_s"] = time.perf_counter() - t
        row["prov_nodes"], row["prov_edges"] = len(N), len(E)
        row["prov_load_s"] = S.attach_provenance(N, E)
        ps = provenance.ProvenanceStore(db)
        bs = np.random.default_rng(0).choice(v.dpp.batch_id.to_numpy(), 100, replace=False)
        lat = []
        for b in bs:
            t = time.perf_counter(); ps.explain(f"ind:{b}:gwp_per_unit"); lat.append(time.perf_counter() - t)
        row["explain_p50_ms"], row["explain_p95_ms"] = float(np.percentile(lat, 50) * 1000), float(np.percentile(lat, 95) * 1000)
        t = time.perf_counter(); ps.impact("fac:grid:CN"); row["impact_gridCN_ms"] = (time.perf_counter() - t) * 1000
        ps.con.close()
    if api and n == API_SIZE:
        docs = {b: interop.passport_jsonld(v, r, b, "2026-09-25T00:00:00Z") for b in v.dpp.batch_id}
        S.load_passports(v, docs)
        S.load_indicators(r, "2026-09-25T00:00:00Z")
        srv = store.serve(db)
        port = srv.server_address[1]
        b = list(np.random.default_rng(1).choice(v.dpp.batch_id.to_numpy(), 2000, replace=False))
        res = []
        for name, paths in (("GET passport", [f"/dpp/{x}" for x in b]),
                            ("GET events", [f"/dpp/{x}/events" for x in b]),
                            ("GET indicators", [f"/dpp/{x}/indicators" for x in b]),
                            ("GET provenance (explain)", [f"/indicator/ind:{x}:gwp_per_unit/explain" for x in b]),
                            ("GET search", [f"/search?category={c}&max_gwp={g}" for c in ("tshirt", "jeans", "dress")
                                            for g in (2, 4, 6, 8)])):
            for c in (1, 8, 32):
                res.append(dict(endpoint=name, **store.benchmark(port, paths, concurrency=c, n_requests=1600)))
        srv.shutdown()
        pd.DataFrame(res).to_csv(os.path.join(OUT, "api_latency.csv"), index=False)
        print(pd.DataFrame(res).round(2).to_string(), flush=True)
    _rm(db)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="?", default="1000,2000,5000,10000,20000,50000")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--api", action="store_true", help="also run the API benchmark (first round only)")
    ap.add_argument("--api-only", action="store_true",
                    help=f"run only the API benchmark at {API_SIZE:,} passports (timings are not recorded)")
    a = ap.parse_args()
    if a.api_only:
        measure(API_SIZE, True)
        raise SystemExit(0)
    sizes = [int(x) for x in a.sizes.split(",")]
    for rnd in range(a.rounds):
        for n in np.random.default_rng(100 + rnd).permutation(sizes):
            row = dict(round=rnd, **measure(int(n), a.api and rnd == 0))
            line = json.dumps(row, default=float)
            print(line, flush=True)
            with open(RUNS, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    import sklearn, scipy
    save_json(dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                   sklearn=sklearn.__version__, scipy=scipy.__version__, platform=platform.platform(),
                   processor=platform.processor(), omp_threads=os.environ.get("OMP_NUM_THREADS"),
                   rounds=a.rounds, sizes=sizes), os.path.join(OUT, "environment.json"))
