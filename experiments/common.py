"""Shared helpers for the experiments (ground-truth labels and metrics)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tsa import params as P  # noqa: E402
from tsa.compliance import GENERIC, COMP_TOL, CLAIM_TOL  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)
EPCIS_SCHEMA = os.environ.get("EPCIS_SCHEMA", os.path.join(ROOT, "external", "epcis-json-schema.json"))

EVAL_INDICATORS = ["gwp_per_unit", "energy_per_unit", "water_per_unit", "aware_per_unit"]


def batch_flags(sc) -> pd.DataFrame:
    """Diagnostic flags derived from ground truth (never used by the engine)."""
    B = sc.batches.set_index("batch_id")
    fac = sc.facilities.set_index("facility_id")
    out = pd.DataFrame(index=B.index)
    out["fraud"] = sc.fibre_lots.groupby("batch_id").fraud.any()
    unev = fac.renew_share.gt(0) & ~fac.renew_evidenced
    out["unevidenced_re"] = B[["fac_spinning", "fac_fabric", "fac_wet", "fac_garment"]].apply(
        lambda r: bool(unev.loc[list(r)].any()), axis=1)
    lab = sc.obs_labels.merge(sc.observations[["obs_id", "batch_id"]], on="obs_id")
    out["anomaly"] = lab.groupby("batch_id").anomaly.any()
    out["fuel_unreported"] = ~fac.loc[B.fac_wet, "fuel_reported"].to_numpy()
    out["category"] = B.category
    out["brand"] = B.brand
    out["variant_id"] = B.variant_id
    out["country_garment"] = fac.loc[B.fac_garment, "country"].to_numpy()
    return out.fillna(False)


def accuracy_metrics(est: pd.DataFrame, truth: pd.DataFrame, col: str) -> dict:
    e = est.set_index("batch_id")
    t = truth.set_index("batch_id")[col].reindex(e.index)
    x = e[col]
    rel = x / t - 1
    out = dict(MAPE=float(rel.abs().mean()), MdAPE=float(rel.abs().median()), bias=float(rel.mean()),
               p90APE=float(rel.abs().quantile(0.9)), spearman=float(spearmanr(x, t).statistic))
    if f"{col}_lo" in e:
        lo, hi, med = e[f"{col}_lo"], e[f"{col}_hi"], e[f"{col}_med"]
        out["coverage95"] = float(((t >= lo) & (t <= hi)).mean())
        out["rel_width95"] = float(((hi - lo) / med).median())
        # interval score (Gneiting & Raftery 2007), relative to truth, alpha = 0.05
        a = 0.05
        isc = (hi - lo) + 2 / a * (lo - t).clip(lower=0) + 2 / a * (t - hi).clip(lower=0)
        out["rel_interval_score"] = float((isc / t).median())
    return out


def compliance_truth(sc) -> pd.DataFrame:
    """Batch-level ground-truth violation labels per rule."""
    B = sc.batches.set_index("batch_id")
    idx = B.index
    F = sc.faults
    lab = pd.DataFrame(False, index=idx, columns=[f"R{i:02d}" for i in range(1, 11)])
    lab["R01"] = idx.isin(F[F.fault_type == "missing_mandatory"].batch_id)
    # R02: declared composition missing or inconsistent with true fibre masses
    fl = sc.fibre_lots.assign(g=sc.fibre_lots.true_material.map(GENERIC))
    true = fl.groupby(["batch_id", "g"]).mass_kg.sum().unstack(fill_value=0.0)
    true = true.div(true.sum(axis=1), axis=0)
    comp = sc.components.assign(g=sc.components.material.map(GENERIC)).groupby(["batch_id", "g"]).share.sum().unstack(fill_value=0.0)
    cols = sorted(set(true.columns) | set(comp.columns))
    comp = comp.reindex(index=idx, columns=cols)
    true = true.reindex(index=idx, columns=cols, fill_value=0.0)
    missing = comp.isna().all(axis=1)
    comp = comp.fillna(0.0)
    lab["R02"] = missing | ((comp - true).abs().max(axis=1) > COMP_TOL) | ((comp.sum(axis=1) - 1).abs() > 0.005)
    # R03
    rec_true = sc.truth.set_index("batch_id").recycled_content_true
    cl = sc.claims
    c3 = cl[cl.claim_type == "recycled_content"]
    lab["R03"] = idx.isin(c3.batch_id[c3.value.to_numpy() > rec_true.reindex(c3.batch_id).to_numpy() + CLAIM_TOL])
    # R04 and R05 need true event times
    certs = sc.certificates
    by = {k: g for k, g in certs.groupby(["holder_facility", "cert_type"])}

    def valid(f, kind, t):
        g = by.get((f, kind))
        return g is not None and bool(((g.valid_from <= t) & (g.valid_to >= t)).any())
    st = sc.stages.set_index(["batch_id", "facility_id"]).time
    fl_t = sc.fibre_lots.set_index(["batch_id", "facility_id"]).time
    org_share = B.organic_share
    r4 = []
    for b in cl[cl.claim_type == "organic"].batch_id:
        bad = org_share[b] < 0.95
        r = B.loc[b]
        for f in [r.fac_spinning, r.fac_fabric, r.fac_wet, r.fac_garment]:
            bad |= not valid(f, "organic", st[(b, f)] if not isinstance(st[(b, f)], pd.Series) else st[(b, f)].iat[0])
        for f, m, t in sc.fibre_lots[sc.fibre_lots.batch_id == b][["facility_id", "true_material", "time"]].itertuples(index=False):
            if m == "cotton_org":
                bad |= not valid(f, "organic", t)
        if bad:
            r4.append(b)
    lab["R04"] = idx.isin(r4)
    cert_ix = certs.set_index("cert_id")
    r5 = []
    for b, evs in zip(cl.batch_id, cl.evidence):
        for cid in json.loads(evs):
            if not cid.startswith("CERT"):
                continue
            c = cert_ix.loc[cid]
            key = (b, c.holder_facility)
            t = st[key] if key in st.index else (fl_t[key] if key in fl_t.index else None)
            if isinstance(t, pd.Series):
                t = t.iat[0]
            if t is not None and not (c.valid_from <= t <= c.valid_to):
                r5.append(b)
                break
    lab["R05"] = idx.isin(r5)
    lab["R06"] = idx.isin(cl.batch_id[cl.evidence.map(lambda e: len(json.loads(e)) == 0)])
    lab["R07"] = idx.isin(F[F.fault_type == "traceability_gap"].batch_id)
    LT = sc.lab_truth
    exc = set(LT.batch_id[LT.true_conc > LT["limit"]])
    lab["R08"] = idx.isin(list(exc))
    c9 = cl[cl.claim_type == "restricted_substances_free"]
    lab["R09"] = idx.isin([b for b in c9.batch_id if b in exc])
    ol = sc.obs_labels.merge(sc.observations[["obs_id", "batch_id"]], on="obs_id")
    lab["R10"] = idx.isin(ol.batch_id[ol.anomaly])
    return lab


def prf(pred: np.ndarray, true: np.ndarray) -> dict:
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum()); fn = int((~pred & true).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if (p == p and r == r and p + r) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f, prevalence=int(true.sum()))


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
