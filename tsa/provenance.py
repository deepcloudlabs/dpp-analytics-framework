"""Analytical provenance graph.

Every indicator value is linked, through derived variables and calculation
rules, to the observations, external factors, models, evidence and data
providers it depends on. The graph follows W3C PROV-O semantics:

  indicator --wasDerivedFrom--> stage aggregate --wasDerivedFrom--> contribution
  contribution --wasDerivedFrom--> activity value --wasDerivedFrom--> observation
  contribution --used--> factor
  activity value --wasGeneratedBy--> model (imputation) | used --> default factor
  activity value --wasInvalidatedBy(rejected)--> observation (screened out)
  observation --wasAttributedTo--> agent (facility)
  observation --wasGeneratedBy--> lifecycle event

Edges carry the identifier and version of the calculation rule ("hadPlan").
The graph is stored in SQLite (nodes, edges) and queried with recursive CTEs.
"""
from __future__ import annotations

import json
import sqlite3
import time

import numpy as np
import pandas as pd

ENGINE_VERSION = "tsa-0.1.0"
RULES = {
    "R-AGG-01": "sum of contributions and normalisation per functional unit",
    "R-SEL-01": "evidence-tier selection (measured > calculated > estimated; verification rank)",
    "R-CAL-01": "bias correction of lower-tier data against independent audit measurements",
    "R-SCR-01": "plausibility screening (physical ranges and model-based robust residuals)",
    "R-IMP-01": "ML imputation of activity intensity x lot mass",
    "R-DEF-01": "secondary default intensity x lot mass",
    "R-ELE-01": "electricity x grid factor x (1 - evidenced renewable share)",
    "R-HEA-01": "fuel energy x fuel emission factor",
    "R-FIB-01": "fibre mass x fibre cradle-to-gate factor",
    "R-TRN-01": "transport work x mode emission factor",
    "R-TRN-02": "default transport model (regional distance table)",
    "R-WAT-01": "water withdrawal x consumption ratio",
    "R-AWA-01": "water consumption x AWARE characterisation factor",
    "R-ENE-01": "final energy (electricity + fuel/3.6)",
    "R-MB-01": "evidence-based mass balance of recycled content",
    "R-TRF-01": "transformation: lot masses derived from product specification and default yields",
}
ACT_RULE = {("gwp", "electricity"): "R-ELE-01", ("gwp", "heat"): "R-HEA-01", ("gwp", "fibre_gwp"): "R-FIB-01",
            ("gwp", "transport"): "R-TRN-01", ("water", "water_consumption"): "R-WAT-01",
            ("water", "fibre_water"): "R-FIB-01", ("aware", "water_consumption"): "R-AWA-01",
            ("aware", "fibre_water"): "R-AWA-01", ("energy", "electricity"): "R-ENE-01", ("energy", "heat"): "R-ENE-01"}
IND_NAME = {"gwp": "gwp_per_unit", "energy": "energy_per_unit", "water": "water_per_unit", "aware": "aware_per_unit"}


def build_graph(result, view, factors: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct node and edge tables from an engine result."""
    C = result.contrib.reset_index(drop=True)
    C = C[C.indicator.isin(IND_NAME)].copy()
    C["cid"] = "con:" + C.index.astype(str)
    dec = result.decisions
    O = view.observations.set_index("obs_id")
    nodes, edges = [], []

    ind_vals = result.indicators.set_index("batch_id")
    # --- indicators and stage aggregates --------------------------------------------
    agg = C.groupby(["batch_id", "indicator", "stage"]).point.sum().reset_index()
    agg["aid"] = "agg:" + agg.batch_id + ":" + agg.indicator + ":" + agg.stage
    agg["iid"] = "ind:" + agg.batch_id + ":" + agg.indicator.map(IND_NAME)
    ii = agg.drop_duplicates("iid")
    for b, ind, iid in zip(ii.batch_id, ii.indicator, ii.iid):
        name = IND_NAME[ind]
        r = ind_vals.loc[b]
        attrs = dict(value=float(r[name]), lo=float(r.get(f"{name}_lo", np.nan)), hi=float(r.get(f"{name}_hi", np.nan)),
                     method=result.method.name, engine=ENGINE_VERSION,
                     dq=float(r["dq_score_gwp"]) if ind == "gwp" else None)
        nodes.append((iid, "indicator", name, json.dumps(attrs)))
    nodes += [(a, "aggregate", f"{i}:{s}", json.dumps(dict(value=float(v)))) for a, i, s, v in
              zip(agg.aid, agg.indicator, agg.stage, agg.point)]
    edges += [(i, a, "wasDerivedFrom", "R-AGG-01") for i, a in zip(agg.iid, agg.aid)]
    # --- contributions ----------------------------------------------------------------
    C["aid"] = "agg:" + C.batch_id + ":" + C.indicator + ":" + C.stage
    nodes += [(c, "contribution", f"{i}:{a}", json.dumps(dict(value=float(p), tier=t))) for c, i, a, p, t in
              zip(C.cid, C.indicator, C.activity, C.point, C.tier)]
    edges += [(a, c, "wasDerivedFrom", "R-AGG-01") for a, c in zip(C.aid, C.cid)]
    # activity node id per contribution
    fibre = C.activity.str.startswith("fibre")
    act_q = np.where(fibre, "fibre_mass", np.where(C.activity == "water_consumption", "water", C.activity))
    lot = np.where(fibre | (C.activity == "transport"), C.obs_id.astype(str), "")
    # stage activities: lot id from decisions
    dkey = dec.set_index(["batch_id", "stage", "quantity"]).lot_id
    stage_lot = [dkey.get((b, s, q), "") if not l else l for b, s, q, l in
                 zip(C.batch_id, C.stage, np.where(C.activity == "water_consumption", "water", C.activity), lot)]
    tran = (C.activity == "transport").to_numpy()
    C["act_id"] = np.where(tran, "act:" + C.obs_id.astype(str) + ":tkm", "act:" + pd.Series(stage_lot, index=C.index) + ":" + act_q)
    rule = [ACT_RULE.get((i, a if a != "fibre_gwp" or i == "gwp" else "fibre_water"), "R-AGG-01")
            for i, a in zip(C.indicator, C.activity)]
    rule = np.where((C.activity == "transport") & (C.tier == "default"), "R-TRN-02", rule)
    edges += [(c, a, "wasDerivedFrom", r) for c, a, r in zip(C.cid, C.act_id, rule)]
    fz = C[C.factor_id != ""]
    edges += [(c, "fac:" + f, "used", r) for c, f, r in zip(fz.cid, fz.factor_id, np.array(rule)[C.factor_id.to_numpy() != ""])]
    for f in fz.factor_id.unique():
        F = factors[f]
        nodes.append(("fac:" + f, "factor", f, json.dumps(dict(value=F.value, unit=F.unit, dataset=F.dataset,
                                                                version=F.version))))
    # --- activity values ----------------------------------------------------------------
    acts = C.drop_duplicates("act_id")
    nodes += [(a, "activity_value", a[4:], json.dumps(dict(tier=t))) for a, t in zip(acts.act_id, acts.tier)]
    # stage activity evidence
    d = dec.copy()
    d["act_id"] = "act:" + d.lot_id + ":" + d.quantity
    sel = d[d.tier.isin(["measured", "calculated", "estimated"])]
    r_sel = np.where(sel.tier.isin(["calculated", "estimated"]) & result.method.calibration, "R-CAL-01", "R-SEL-01")
    edges += [(a, "obs:" + o, "wasDerivedFrom", r) for a, o, r in zip(sel.act_id, sel.obs_id, r_sel)]
    massobs = view.observations[view.observations.quantity == "mass_in"].set_index("lot_id").obs_id
    gap = d[d.tier.isin(["imputed", "default"])]
    for a, l, t, q, st in zip(gap.act_id, gap.lot_id, gap.tier, gap.quantity, gap.stage):
        mo = massobs.get(l)
        if isinstance(mo, str):
            edges.append((a, "obs:" + mo, "wasDerivedFrom", "R-IMP-01" if t == "imputed" else "R-DEF-01"))
        if t == "imputed":
            edges.append((a, f"model:{st}:{q}", "wasGeneratedBy", "R-IMP-01"))
        else:
            edges.append((a, f"fac:default:{st}:{q}", "used", "R-DEF-01"))
    for st, q in gap[gap.tier == "imputed"][["stage", "quantity"]].drop_duplicates().itertuples(index=False):
        imp = result.imputers.get((st, q))
        nodes.append((f"model:{st}:{q}", "model", f"intensity-imputer:{st}:{q}",
                      json.dumps(dict(algorithm="HistGradientBoosting+facility-effect+conformal", version="1.0",
                                      n_train=int(len(imp.resid_)) if imp is not None else None))))
    for st, q in gap[gap.tier == "default"][["stage", "quantity"]].drop_duplicates().itertuples(index=False):
        nodes.append((f"fac:default:{st}:{q}", "factor", f"default:{st}:{q}", json.dumps(dict(dataset="generic-process-data"))))
    # rejected observations
    rej = view.observations[view.observations.obs_id.isin(getattr(result, "rejected", set()))]
    edges += [("act:" + l + ":" + q, "obs:" + o, "rejected", "R-SCR-01") for l, q, o in zip(rej.lot_id, rej.quantity, rej.obs_id)]
    # fibre and transport activity evidence
    fmass = view.observations[(view.observations.quantity == "mass_out") & (view.observations.stage == "fibre")].set_index("lot_id").obs_id
    prim = view.observations[view.observations.quantity.isin(["fibre_gwp_intensity", "fibre_water_intensity"])]
    fl = acts[acts.activity.str.startswith("fibre")]
    for a, l in zip(fl.act_id, fl.obs_id):
        mo = fmass.get(l)
        if isinstance(mo, str):
            edges.append((a, "obs:" + mo, "wasDerivedFrom", "R-FIB-01"))
    edges += [("act:" + l + ":fibre_mass", "obs:" + o, "wasDerivedFrom", "R-FIB-01") for l, o in zip(prim.lot_id, prim.obs_id)]
    tr = acts[acts.activity == "transport"]
    edges += [(a, "obs:" + o, "wasDerivedFrom", "R-SEL-01") for a, o, t in zip(tr.act_id, tr.obs_id, tr.tier) if t == "measured"]
    # --- activity values derived from the product specification (no direct observation) -------
    has_out = {s for s, _, _, _ in edges if s.startswith("act:")}
    spec_batches = set()
    for a, b in zip(acts.act_id, acts.batch_id):
        if a in has_out:
            continue
        spec_batches.add(b)
        edges.append((a, f"spec:{b}", "wasDerivedFrom", "R-TRF-01"))
        if a.endswith(":tkm"):
            edges.append((a, "fac:default:transport-distances", "used", "R-TRN-02"))
    if spec_batches:
        nodes += [(f"spec:{b}", "specification", f"product specification {b}",
                   json.dumps(dict(source="PLM", content="units, unit mass, composition, cut waste"))) for b in spec_batches]
        nodes.append(("fac:default:transport-distances", "factor", "default:transport-distances",
                      json.dumps(dict(dataset="regional distance table", version="assumption-1"))))
    # --- observations, agents, events ------------------------------------------------------
    E = pd.DataFrame(edges, columns=["src", "dst", "rel", "rule"]).drop_duplicates()
    obs_ids = E.dst[E.dst.str.startswith("obs:")].str[4:].unique()
    Ob = O.loc[O.index.intersection(obs_ids)]
    nodes += [("obs:" + i, "observation", q, json.dumps(dict(value=None if pd.isna(v) else float(v), unit=u, data_type=dt,
                                                              source=s, timestamp=str(ts), verification=vf)))
              for i, q, v, u, dt, s, ts, vf in zip(Ob.index, Ob.quantity, Ob.value, Ob.unit, Ob.data_type,
                                                   Ob.source_system, Ob.timestamp, Ob.verification)]
    extra = [("obs:" + i, "agent:" + f, "wasAttributedTo", "") for i, f in zip(Ob.index, Ob.facility_id)]
    ev = view.events
    out_lot = {json.loads(o)[0]: e for o, e in zip(ev.outputs, ev.event_id) if o not in ("[]", "")}
    extra += [("obs:" + i, "event:" + out_lot[l], "wasGeneratedBy", "") for i, l in zip(Ob.index, Ob.lot_id) if l in out_lot]
    agents = Ob.facility_id.unique()
    nodes += [("agent:" + a, "agent", a, "{}") for a in agents]
    evs = {out_lot[l] for l in Ob.lot_id if l in out_lot}
    evt = ev.set_index("event_id")
    nodes += [("event:" + e, "event", evt.at[e, "event_type"], json.dumps(dict(time=str(evt.at[e, "time"]),
                                                                            facility=evt.at[e, "facility_id"]))) for e in evs]
    E = pd.concat([E, pd.DataFrame(extra, columns=E.columns)], ignore_index=True)
    N = pd.DataFrame(nodes, columns=["node_id", "kind", "label", "attrs"]).drop_duplicates("node_id")
    return N, E


class ProvenanceStore:
    def __init__(self, path: str = ":memory:"):
        self.con = sqlite3.connect(path)
        self.con.executescript("""
            PRAGMA journal_mode=WAL; PRAGMA synchronous=OFF;
            CREATE TABLE IF NOT EXISTS nodes(node_id TEXT PRIMARY KEY, kind TEXT, label TEXT, attrs TEXT);
            CREATE TABLE IF NOT EXISTS edges(src TEXT, dst TEXT, rel TEXT, rule TEXT);
            CREATE TABLE IF NOT EXISTS rules(rule_id TEXT PRIMARY KEY, description TEXT);
        """)

    def load(self, N: pd.DataFrame, E: pd.DataFrame):
        t = time.perf_counter()
        self.con.executemany("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?)", N.itertuples(index=False, name=None))
        self.con.executemany("INSERT INTO edges VALUES (?,?,?,?)", E.itertuples(index=False, name=None))
        self.con.executemany("INSERT OR REPLACE INTO rules VALUES (?,?)", RULES.items())
        self.con.executescript("CREATE INDEX IF NOT EXISTS e_src ON edges(src); CREATE INDEX IF NOT EXISTS e_dst ON edges(dst);"
                               "CREATE INDEX IF NOT EXISTS n_kind ON nodes(kind);")
        self.con.commit()
        return time.perf_counter() - t

    # backward lineage (explain)
    def explain(self, node_id: str) -> pd.DataFrame:
        # CROSS JOIN fixes the join order (lineage first, then primary-key lookups in nodes);
        # otherwise the planner may scan all nodes to satisfy the grouping.
        q = """WITH RECURSIVE lin(node, depth) AS (
                 SELECT ?, 0
                 UNION SELECT e.dst, lin.depth + 1 FROM lin CROSS JOIN edges e ON e.src = lin.node),
               agg AS (SELECT node, MIN(depth) AS depth FROM lin GROUP BY node)
               SELECT n.node_id, n.kind, n.label, n.attrs, agg.depth
               FROM agg CROSS JOIN nodes n ON n.node_id = agg.node"""
        return pd.read_sql_query(q, self.con, params=(node_id,))

    # forward impact (which indicators depend on a node)
    def impact(self, node_id: str) -> pd.DataFrame:
        q = """WITH RECURSIVE imp(node) AS (
                 SELECT ?
                 UNION SELECT e.src FROM imp CROSS JOIN edges e ON e.dst = imp.node)
               SELECT n.node_id, n.kind FROM imp CROSS JOIN nodes n ON n.node_id = imp.node WHERE n.kind = 'indicator'"""
        return pd.read_sql_query(q, self.con, params=(node_id,))

    def rules_used(self, node_id: str) -> list:
        q = """WITH RECURSIVE lin(node) AS (SELECT ? UNION SELECT e.dst FROM lin CROSS JOIN edges e ON e.src = lin.node)
               SELECT DISTINCT e.rule FROM lin CROSS JOIN edges e ON e.src = lin.node WHERE e.rule <> ''"""
        return [r[0] for r in self.con.execute(q, (node_id,))]

    def size(self):
        return (self.con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
                self.con.execute("SELECT COUNT(*) FROM edges").fetchone()[0])


def to_prov_jsonld(sub: pd.DataFrame, edges: pd.DataFrame) -> dict:
    """Serialise a lineage sub-graph as PROV-O JSON-LD (qualified derivations carry the rule)."""
    kind_type = {"indicator": "prov:Entity", "aggregate": "prov:Entity", "contribution": "prov:Entity",
                 "activity_value": "prov:Entity", "observation": "prov:Entity", "factor": "prov:Entity",
                 "model": "prov:SoftwareAgent", "agent": "prov:Agent", "event": "prov:Activity"}
    ids = set(sub.node_id)
    graph = []
    for r in sub.itertuples():
        node = {"@id": f"urn:tsa:{r.node_id}", "@type": [kind_type.get(r.kind, "prov:Entity"), f"tsa:{r.kind}"],
                "rdfs:label": r.label}
        node.update({f"tsa:{k}": v for k, v in json.loads(r.attrs).items() if v is not None})
        out = edges[(edges.src == r.node_id) & edges.dst.isin(ids)]
        for e in out.itertuples():
            tgt = {"@id": f"urn:tsa:{e.dst}"}
            if e.rel == "wasDerivedFrom":
                node.setdefault("prov:qualifiedDerivation", []).append(
                    {"@type": "prov:Derivation", "prov:entity": tgt,
                     "prov:hadPlan": {"@id": f"urn:tsa:rule:{e.rule}"}} if e.rule else {"prov:entity": tgt})
            else:
                node.setdefault(f"prov:{e.rel}" if e.rel != "rejected" else "tsa:rejected", []).append(tgt)
        graph.append(node)
    return {"@context": {"prov": "http://www.w3.org/ns/prov#", "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                         "tsa": "https://example.org/tsa#"}, "@graph": graph}
