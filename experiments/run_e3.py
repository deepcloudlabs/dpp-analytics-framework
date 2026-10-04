"""E3: analytical provenance - graph size, lineage coverage, query latency,
competency questions (CQ) across passport representations, and factor-revision
impact analysis with targeted recomputation.

Representations compared:
  A  static passport document (attributes, claims, indicator value + unit)
  B  A + lifecycle event log + observations with metadata (event-based traceability)
  C  B + analytical records + analytical provenance graph (proposed)
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import RESULTS, save_json  # noqa: E402
from tsa.synth import generate, ScenarioConfig  # noqa: E402
from tsa.engine import Engine, METHODS, observed_view  # noqa: E402
from tsa.factors import build_factor_db  # noqa: E402
from tsa import provenance, interop  # noqa: E402
from tsa.store import DppStore  # noqa: E402

OUT = os.path.join(RESULTS, "e3")
os.makedirs(OUT, exist_ok=True)
REP_TABLES = {"A": {"passport", "indicator_value"},
              "B": {"passport", "indicator_value", "event", "observation", "certificate"},
              "C": {"passport", "indicator_value", "event", "observation", "certificate", "indicator_record", "nodes", "edges"}}


class NotAvailable(Exception):
    pass


def need(rep, *tables):
    if not set(tables) <= REP_TABLES[rep]:
        raise NotAvailable(tables)


def make_cqs(con, view):
    certs = view.certificates.set_index("cert_id")
    claims = view.claims

    def cq1(rep, b):   # facilities/countries involved
        if rep == "A":
            return ("partial", con.execute("SELECT manufacturer_facility FROM passport WHERE batch_id=?", (b,)).fetchall())
        need(rep, "event")
        return ("yes", con.execute("SELECT DISTINCT facility_id FROM event WHERE batch_id=?", (b,)).fetchall())

    def cq2(rep, b):   # indicator value and unit
        need(rep, "indicator_value")
        return ("yes", con.execute("SELECT value, unit FROM indicator WHERE node_id=?", (f"ind:{b}:gwp_per_unit",)).fetchall())

    def cq3(rep, b):   # interval and DQ
        need(rep, "indicator_record")
        return ("yes", con.execute("SELECT lo, hi, dq, primary_share FROM indicator WHERE node_id=?",
                                   (f"ind:{b}:gwp_per_unit",)).fetchall())

    def lineage(b):
        q = """WITH RECURSIVE lin(node) AS (SELECT ? UNION SELECT e.dst FROM lin CROSS JOIN edges e ON e.src = lin.node)
               SELECT n.node_id, n.kind, n.attrs FROM lin CROSS JOIN nodes n ON n.node_id = lin.node"""
        return con.execute(q, (f"ind:{b}:gwp_per_unit",)).fetchall()

    def cq4(rep, b):   # contributing observations with source/timestamp and shares
        need(rep, "nodes", "edges")
        rows = [r for r in lineage(b) if r[1] == "observation"]
        return ("yes", rows)

    def cq5(rep, b):   # share by data tier
        need(rep, "nodes", "edges")
        q = """SELECT json_extract(n.attrs,'$.tier'), SUM(json_extract(n.attrs,'$.value'))
               FROM (SELECT e2.dst AS agg FROM edges e2 WHERE e2.src = ?) a
               CROSS JOIN edges e ON e.src = a.agg CROSS JOIN nodes n ON n.node_id = e.dst
               WHERE n.kind='contribution' GROUP BY 1"""
        return ("yes", con.execute(q, (f"ind:{b}:gwp_per_unit",)).fetchall())

    def cq6(rep, b):   # rules, factor datasets, model versions
        need(rep, "nodes", "edges")
        q = """WITH RECURSIVE lin(node) AS (SELECT ? UNION SELECT e.dst FROM lin CROSS JOIN edges e ON e.src = lin.node)
               SELECT DISTINCT e.rule FROM lin CROSS JOIN edges e ON e.src = lin.node WHERE e.rule <> ''"""
        rules = con.execute(q, (f"ind:{b}:gwp_per_unit",)).fetchall()
        facs = [r for r in lineage(b) if r[1] in ("factor", "model")]
        return ("yes", (rules, facs))

    def cq7(rep, f):   # passports affected by a factor revision
        need(rep, "nodes", "edges")
        q = """WITH RECURSIVE imp(node) AS (SELECT ? UNION SELECT e.src FROM imp CROSS JOIN edges e ON e.dst = imp.node)
               SELECT node FROM imp WHERE substr(node, 1, 4) = 'ind:'"""
        return ("yes", con.execute(q, (f,)).fetchall())

    def cq8(rep, b):   # certificate validity at event time
        need(rep, "event", "certificate")
        out = []
        for evs in claims[claims.batch_id == b].evidence:
            for cid in json.loads(evs):
                if cid.startswith("CERT"):
                    c = certs.loc[cid]
                    t = con.execute("SELECT MIN(time) FROM event WHERE batch_id=? AND facility_id=?",
                                    (b, c.holder_facility)).fetchone()[0]
                    out.append((cid, t is not None and str(c.valid_from) <= t <= str(c.valid_to)))
        return ("yes", out)

    def cq9(rep, b):   # missing upstream events
        need(rep, "event")
        types = {r[0] for r in con.execute("SELECT DISTINCT event_type FROM event WHERE batch_id=?", (b,))}
        return ("yes", {"raw_material_sourcing", "spinning"} - types)

    def cq10(rep, b):  # rejected or imputed observations and model
        need(rep, "nodes", "edges")
        # prefix range on the indexed src column ('-' < '.' in ASCII)
        q = """SELECT e.src, e.dst, e.rel FROM edges e WHERE e.src >= ? AND e.src < ?
               AND e.rel IN ('rejected','wasGeneratedBy')"""
        return ("yes", con.execute(q, (f"act:{b}-", f"act:{b}.")).fetchall())

    return [("CQ1", "Facilities and countries involved in producing a batch", cq1),
            ("CQ2", "Value and unit of an indicator", cq2),
            ("CQ3", "Uncertainty interval and data-quality score of an indicator", cq3),
            ("CQ4", "Observations (source system, time) that contributed to an indicator", cq4),
            ("CQ5", "Share of an indicator based on measured, estimated, imputed or default data", cq5),
            ("CQ6", "Calculation rules, factor datasets and model versions used", cq6),
            ("CQ7", "Passports affected by the revision of an external factor", cq7),
            ("CQ8", "Validity of claim evidence at the time of the related event", cq8),
            ("CQ9", "Missing upstream lifecycle events (traceability gaps)", cq9),
            ("CQ10", "Observations rejected by screening or replaced by imputation, and the model used", cq10)]


if __name__ == "__main__":
    t0 = time.perf_counter()
    sc = generate(ScenarioConfig(n_batches=2000, seed=21))
    v = observed_view(sc)
    r = Engine(v, METHODS["M3"], n_samples=500, seed=21).run()
    F = build_factor_db()
    t = time.perf_counter(); N, E = provenance.build_graph(r, v, F); t_build = time.perf_counter() - t
    # database on the local disk (the results folder may be on a network/virtual file system)
    import tempfile
    db = os.path.join(tempfile.gettempdir(), "tsa_e3.sqlite")
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(db + ext):
            os.remove(db + ext)
    S = DppStore(db)
    docs = {b: interop.passport_jsonld(v, r, b, "2026-09-25T00:00:00Z") for b in v.dpp.batch_id}
    S.load_passports(v, docs); S.append_events(v.events); S.load_observations(v.observations)
    S.load_indicators(r, "2026-09-25T00:00:00Z")
    t_load = S.attach_provenance(N, E)
    stats = dict(n_batches=2000, nodes=len(N), edges=len(E), nodes_per_batch=len(N) / 2000, edges_per_batch=len(E) / 2000,
                 build_s=t_build, load_s=t_load, node_kinds=N.kind.value_counts().to_dict(),
                 edge_rel=E.rel.value_counts().to_dict())
    # --- lineage coverage: contribution value reaching typed leaves ----------------------
    act_edges = E[E.src.str.startswith("act:")]
    leaf_ok = set(act_edges.src[act_edges.dst.str.startswith(("obs:", "model:", "fac:", "spec:"))])
    con_edges = E[E.src.str.startswith("con:") & E.dst.str.startswith("act:")]
    con_to_act = dict(zip(con_edges.src, con_edges.dst))
    cn = N[N.kind == "contribution"].copy()
    cn["value"] = cn["attrs"].map(lambda a: json.loads(a)["value"])
    cn["tier"] = cn["attrs"].map(lambda a: json.loads(a)["tier"])
    cn["ok"] = cn.node_id.map(lambda c: con_to_act.get(c) in leaf_ok)
    stats["lineage_coverage_value_weighted"] = float((cn.value.abs() * cn.ok).sum() / cn.value.abs().sum())
    stats["lineage_coverage_contributions"] = float(cn.ok.mean())
    stats["value_share_by_tier"] = (cn.groupby("tier").value.apply(lambda s: s.abs().sum()) / cn.value.abs().sum()).to_dict()
    # --- query latency --------------------------------------------------------------------
    ps = provenance.ProvenanceStore(db)
    rng = np.random.default_rng(0)
    bs = rng.choice(v.dpp.batch_id.to_numpy(), 200, replace=False)
    lat = {"explain": [], "rules_used": [], "impact_factor": []}
    for b in bs:
        t = time.perf_counter(); ps.explain(f"ind:{b}:gwp_per_unit"); lat["explain"].append(time.perf_counter() - t)
        t = time.perf_counter(); ps.rules_used(f"ind:{b}:gwp_per_unit"); lat["rules_used"].append(time.perf_counter() - t)
    facs = sorted(N.node_id[N.kind == "factor"])
    for f in facs:
        t = time.perf_counter(); ps.impact(f); lat["impact_factor"].append(time.perf_counter() - t)
    stats["latency_ms"] = {k: dict(p50=float(np.percentile(np.array(v_) * 1000, 50)),
                                   p95=float(np.percentile(np.array(v_) * 1000, 95)), n=len(v_)) for k, v_ in lat.items()}
    # --- competency questions ---------------------------------------------------------------
    con = sqlite3.connect(db)
    cqs = make_cqs(con, v)
    cq_rows = []
    for cid, desc, fn in cqs:
        row = dict(cq=cid, question=desc)
        for rep in ("A", "B", "C"):
            status, times = "no", []
            for b in bs[:50]:
                arg = "fac:grid:CN" if cid == "CQ7" else b
                try:
                    t = time.perf_counter(); st, ans = fn(rep, arg); times.append(time.perf_counter() - t)
                    status = st
                except NotAvailable:
                    status = "no"
                    break
            row[rep] = status
            if rep == "C":
                row["C_latency_ms_p50"] = float(np.median(times) * 1000) if times else None
        cq_rows.append(row)
    pd.DataFrame(cq_rows).to_csv(os.path.join(OUT, "competency_questions.csv"), index=False)
    # --- factor revision impact analysis ---------------------------------------------------------
    C = r.contrib
    base = C[C.indicator == "gwp"].groupby("batch_id").point.sum()
    units = v.batch_master.set_index("batch_id").units
    imp_rows = []
    for fid in [f for f in facs if f.startswith(("fac:grid:", "fac:fibre_gwp:", "fac:transport:"))]:
        t = time.perf_counter()
        affected = ps.impact(fid)
        t_q = time.perf_counter() - t
        bset = {x.split(":")[1] for x in affected.node_id}
        # targeted recomputation (+10 % revision of the factor) vs full recomputation
        t = time.perf_counter()
        sub = C[(C.indicator == "gwp") & C.batch_id.isin(bset)]
        upd = sub.point * np.where(sub.factor_id == fid[4:], 1.10, 1.0)
        new_t = upd.groupby(sub.batch_id).sum()
        t_targ = time.perf_counter() - t
        t = time.perf_counter()
        allc = C[C.indicator == "gwp"]
        new_f = (allc.point * np.where(allc.factor_id == fid[4:], 1.10, 1.0)).groupby(allc.batch_id).sum()
        t_full = time.perf_counter() - t
        changed = set(new_f.index[(new_f - base).abs() > 1e-12])
        imp_rows.append(dict(factor=fid, affected_passports=len(bset), affected_share=len(bset) / len(base),
                             impact_query_ms=t_q * 1000, targeted_recompute_ms=t_targ * 1000, full_recompute_ms=t_full * 1000,
                             exact_match=bool(np.allclose(new_t.reindex(sorted(bset)), new_f.reindex(sorted(bset)))),
                             missed=len(changed - bset), superfluous=len(bset - changed)))
    pd.DataFrame(imp_rows).to_csv(os.path.join(OUT, "factor_revision_impact.csv"), index=False)
    # example PROV-O JSON-LD lineage for one indicator
    ex = ps.explain(f"ind:{bs[0]}:gwp_per_unit")
    prov_doc = provenance.to_prov_jsonld(ex, E[E.src.isin(ex.node_id)])
    save_json(prov_doc, os.path.join(OUT, "example_prov_lineage.jsonld"))
    stats["example_batch"] = str(bs[0])
    stats["runtime_s"] = time.perf_counter() - t0
    save_json(stats, os.path.join(OUT, "provenance_stats.json"))
    print(json.dumps({k: v_ for k, v_ in stats.items() if k not in ("node_kinds", "edge_rel")}, indent=1, default=float))
    print(pd.DataFrame(cq_rows)[["cq", "A", "B", "C", "C_latency_ms_p50"]])
