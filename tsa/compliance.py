"""Compliance and claim-verification rules engine.

Rules are declarative records (identifier, version, description, basis,
severity) bound to vectorised check functions over OBSERVED passport data.
Each check returns one row per flagged batch with outcome ``violation``
(evidence contradicts the requirement) or ``unverifiable`` (the evidence
needed to decide is absent). The rule set is ILLUSTRATIVE: no textile
delegated act under Regulation (EU) 2024/1781 had been adopted at the time
of writing, so the mandatory-attribute set and thresholds are assumptions,
except where a specific legal threshold is cited in the manuscript.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import time

import numpy as np
import pandas as pd

from . import params as P

GENERIC = {"cotton_conv": "cotton", "cotton_org": "cotton", "cotton_rec": "cotton", "pes_virgin": "polyester",
           "pes_rec": "polyester", "pa_virgin": "polyamide", "elastane": "elastane", "viscose": "viscose",
           "lyocell": "lyocell", "wool": "wool"}
MANDATORY = ["gtin", "operator_id", "manufacturer_facility", "country_of_manufacture", "declared_composition",
             "care_information"]
COMP_TOL = 0.03       # tolerance on fibre composition (fraction)
CLAIM_TOL = 0.01      # tolerance on recycled-content claims (fraction)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    version: str
    description: str
    basis: str
    severity: str


RULES = [
    Rule("R01", "1.0", "Mandatory passport attributes are present", "illustrative (ESPR Annex III type data)", "error"),
    Rule("R02", "1.0", "Declared fibre composition consistent with mass balance of sourced fibre lots",
         "illustrative (textile labelling tolerance concept)", "error"),
    Rule("R03", "1.0", "Recycled-content claim does not exceed evidenced recycled content", "claim substantiation", "error"),
    Rule("R04", "1.0", "Organic claim supported by valid certificates at every chain stage", "claim substantiation", "error"),
    Rule("R05", "1.0", "Evidence certificates valid at the time of the related lifecycle event", "evidence validity", "error"),
    Rule("R06", "1.0", "Every claim references at least one evidence item", "claim substantiation", "warning"),
    Rule("R07", "1.0", "Transformation chain is linked back to fibre-sourcing events", "traceability", "error"),
    Rule("R08", "1.0", "Tested restricted substances below limit values", "illustrative limits", "error"),
    Rule("R09", "1.0", "'Restricted-substance free' claim substantiated by test reports", "claim substantiation", "error"),
    Rule("R10", "1.0", "Reported activity data within physical plausibility ranges", "data quality", "warning"),
]


class ComplianceEngine:
    def __init__(self, view, canon_obs: pd.DataFrame | None = None):
        self.v = view
        from .engine import canonicalise
        self.O = canon_obs if canon_obs is not None else canonicalise(view.observations)
        self._prep()

    def _prep(self):
        v = self.v
        ev = v.events
        self.ev = ev
        # event time per (batch, facility, event_type)
        self.t_src = {json.loads(o)[0]: t for o, t in
                      zip(ev[ev.event_type == "raw_material_sourcing"].outputs, ev[ev.event_type == "raw_material_sourcing"].time)}
        stage_ev = ev[ev.event_type.isin(["spinning", "weaving", "knitting", "dyeing_finishing", "cutting_assembly"])]
        self.t_stage = {(b, f): t for b, f, t in zip(stage_ev.batch_id, stage_ev.facility_id, stage_ev.time)}
        self.stage_ev = stage_ev
        c = v.certificates
        self.cert = c.set_index("cert_id")
        self.cert_by = {k: g for k, g in c.groupby(["holder_facility", "cert_type"])}
        fl = v.fibre_lot_master
        m = self.O[(self.O.quantity == "mass_out") & (self.O.stage == "fibre")].set_index("lot_id").value_c
        self.fl = fl.assign(mass=m.reindex(fl.lot_id).to_numpy(), t=fl.lot_id.map(self.t_src))

    def _valid(self, fac, kind, t) -> bool | None:
        if t is None or pd.isna(t):
            return None
        g = self.cert_by.get((fac, kind))
        if g is None:
            return False
        return bool(((g.valid_from <= t) & (g.valid_to >= t)).any())

    # ------------------------------------------------------------------
    def run(self) -> tuple[pd.DataFrame, dict]:
        out, timing = [], {}
        for rule in RULES:
            t0 = time.perf_counter()
            res = getattr(self, f"_{rule.rule_id.lower()}")()
            timing[rule.rule_id] = time.perf_counter() - t0
            if len(res):
                res = res.assign(rule_id=rule.rule_id, rule_version=rule.version)
                out.append(res)
        R = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
            columns=["batch_id", "outcome", "detail", "rule_id", "rule_version"])
        return R, timing

    def _r01(self):
        d = self.v.dpp
        miss = d[MANDATORY].isna()
        flag = miss.any(axis=1)
        det = miss[flag].apply(lambda r: ",".join(r.index[r]), axis=1)
        return pd.DataFrame(dict(batch_id=d.batch_id[flag].values, outcome="violation", detail=det.values))

    def _r02(self):
        comp = self.v.components.assign(g=lambda x: x.material.map(GENERIC))
        decl = comp.groupby(["batch_id", "g"]).share.sum().unstack(fill_value=0.0)
        fl = self.fl.assign(g=self.fl.material.map(GENERIC))
        complete = fl.groupby("batch_id").mass.apply(lambda s: s.notna().all())
        mb = fl.groupby(["batch_id", "g"]).mass.sum().unstack(fill_value=0.0)
        mb = mb.div(mb.sum(axis=1), axis=0)
        cols = sorted(set(decl.columns) | set(mb.columns))
        decl, mb = decl.reindex(columns=cols, fill_value=0.0), mb.reindex(index=decl.index, columns=cols, fill_value=0.0)
        sum_bad = (decl.sum(axis=1) - 1).abs() > 0.005
        dev = (decl - mb).abs().max(axis=1) > COMP_TOL
        comp_ok = complete.reindex(decl.index).fillna(False).astype(bool)
        viol = (sum_bad | (dev & comp_ok))
        unver = ~comp_ok & ~sum_bad
        rows = [pd.DataFrame(dict(batch_id=decl.index[viol], outcome="violation", detail="composition")),
                pd.DataFrame(dict(batch_id=decl.index[unver], outcome="unverifiable", detail="fibre lot masses missing"))]
        nodecl = sorted(set(self.v.dpp.batch_id) - set(decl.index))
        rows.append(pd.DataFrame(dict(batch_id=nodecl, outcome="unverifiable", detail="no declared composition")))
        return pd.concat(rows, ignore_index=True)

    def _evidenced_recycled(self):
        fl = self.fl
        rec = fl.material.map(lambda m: P.MATERIALS[m]["recycled"]).to_numpy()
        ok = np.array([self._valid(f, "recycled", t) if r else False for f, t, r in zip(fl.facility_id, fl.t, rec)],
                      dtype=object)
        unknown = np.array([o is None for o in ok])
        evid = np.array([bool(o) for o in ok])
        g = fl.assign(ev_mass=fl.mass * evid, unk=unknown & rec).groupby("batch_id")
        share = g.ev_mass.sum() / g.mass.sum(min_count=1)
        incomplete = g.mass.apply(lambda s: s.isna().any()) | g.unk.any()
        return share, incomplete

    def _r03(self):
        cl = self.v.claims[self.v.claims.claim_type == "recycled_content"]
        share, incomplete = self._evidenced_recycled()
        s = share.reindex(cl.batch_id).to_numpy()
        inc = incomplete.reindex(cl.batch_id).fillna(True).to_numpy().astype(bool)
        viol = (cl.value.to_numpy() > s + CLAIM_TOL) & ~inc
        unver = inc & (cl.value.to_numpy() > 0)
        return pd.concat([pd.DataFrame(dict(batch_id=cl.batch_id[viol].values, outcome="violation", detail="overclaim")),
                          pd.DataFrame(dict(batch_id=cl.batch_id[unver & ~viol].values, outcome="unverifiable",
                                            detail="incomplete evidence"))], ignore_index=True)

    def _chain(self, b):
        bm = self.bm_ix.loc[b]
        return [bm.fac_spinning, bm.fac_fabric, bm.fac_wet, bm.fac_garment]

    def _r04(self):
        self.bm_ix = self.v.batch_master.set_index("batch_id")
        cl = self.v.claims[self.v.claims.claim_type == "organic"]
        comp = self.v.components
        org = comp[comp.material == "cotton_org"].groupby("batch_id").share.sum()
        fl_by = self.fl.groupby("batch_id")
        rows = []
        for b in cl.batch_id:
            if org.get(b, 0.0) < 0.95:
                rows.append((b, "violation", "declared organic share < 95%"))
                continue
            status = []
            for _, r in fl_by.get_group(b).iterrows():
                if r.material == "cotton_org":
                    status.append(self._valid(r.facility_id, "organic", r.t))
            for f in self._chain(b):
                status.append(self._valid(f, "organic", self.t_stage.get((b, f))))
            if any(s is False for s in status):
                rows.append((b, "violation", "chain certificate missing/invalid"))
            elif any(s is None for s in status):
                rows.append((b, "unverifiable", "event missing"))
        return pd.DataFrame(rows, columns=["batch_id", "outcome", "detail"])

    def _r05(self):
        cl = self.v.claims
        fl_t = {(b, f): t for b, f, t in zip(self.fl.batch_id, self.fl.facility_id, self.fl.t)}
        rows = []
        for b, evs in zip(cl.batch_id, cl.evidence):
            for cid in json.loads(evs):
                if not cid.startswith("CERT"):
                    continue
                c = self.cert.loc[cid]
                t = self.t_stage.get((b, c.holder_facility), fl_t.get((b, c.holder_facility)))
                if t is None or pd.isna(t):
                    rows.append((b, "unverifiable", cid))
                elif not (c.valid_from <= t <= c.valid_to):
                    rows.append((b, "violation", cid))
        R = pd.DataFrame(rows, columns=["batch_id", "outcome", "detail"])
        if len(R):   # one row per batch, violation dominates
            R = R.sort_values("outcome", ascending=False).drop_duplicates("batch_id")
        return R

    def _r06(self):
        cl = self.v.claims
        empty = cl.evidence.map(lambda e: len(json.loads(e)) == 0)
        return pd.DataFrame(dict(batch_id=cl.batch_id[empty].values, outcome="violation",
                                 detail=cl.claim_type[empty].values)).drop_duplicates("batch_id")

    def _r07(self):
        ev = self.ev
        have_src = set(self.t_src)
        fl_ok = self.fl.assign(ok=self.fl.lot_id.isin(have_src)).groupby("batch_id").ok.all()
        spin = set(ev[ev.event_type == "spinning"].batch_id)
        bad = [b for b in self.v.dpp.batch_id if not fl_ok.get(b, False) or b not in spin]
        return pd.DataFrame(dict(batch_id=bad, outcome="violation", detail="upstream event missing"))

    def _r08(self):
        lab = self.v.lab_tests
        exc = lab[lab.conc_mg_kg > lab["limit"]]
        return pd.DataFrame(dict(batch_id=exc.batch_id.unique(), outcome="violation", detail="exceedance"))

    def _r09(self):
        cl = self.v.claims[self.v.claims.claim_type == "restricted_substances_free"]
        lab = self.v.lab_tests
        exc = set(lab[lab.conc_mg_kg > lab["limit"]].batch_id)
        tested = set(lab.batch_id)
        viol = cl.batch_id.isin(exc)
        unver = ~cl.batch_id.isin(tested)
        return pd.concat([pd.DataFrame(dict(batch_id=cl.batch_id[viol].values, outcome="violation", detail="exceedance")),
                          pd.DataFrame(dict(batch_id=cl.batch_id[unver].values, outcome="unverifiable",
                                            detail="no test report"))], ignore_index=True)

    def _r10(self):
        from .ml import plausibility_violation
        O = self.O
        a = O[O.quantity.isin(["electricity", "heat", "water"]) & O.value_c.notna()]
        m = O[(O.quantity == "mass_in")].set_index("lot_id").value_c
        mass = m.reindex(a.lot_id).to_numpy()
        ok = ~np.isnan(mass)
        a = a[ok]
        inten = a.value_c.to_numpy() / mass[ok]
        flag = plausibility_violation(a.stage.to_numpy(), a.quantity.to_numpy(), inten)
        return pd.DataFrame(dict(batch_id=a.batch_id[flag].unique(), outcome="violation", detail="implausible activity"))
