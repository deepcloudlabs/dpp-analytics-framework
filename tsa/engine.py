"""Sustainability analytics engine.

Transforms OBSERVED passport data into batch-level sustainability indicators
with uncertainty, data-quality scores and a contribution table that feeds the
analytical provenance graph.

Pipeline (proposed method, M3):
  1. unit normalisation to canonical units
  2. evidence-tier selection per (lot, quantity)
  3. plausibility screening (physical range rules + model-based residuals)
  4. calibration of lower-tier data (bias and dispersion) against
     independent verification (audit) measurements
  5. gap filling: ML imputation (M3) or secondary default data (baselines)
  6. contribution table: activity x factor rows with tier, verification,
     factor specificity and uncertainty
  7. Monte Carlo propagation with shared (correlated) factor draws
  8. indicator aggregation, normalisation and data-quality scoring

Baselines: M0 (naive: available data, gaps ignored), M1 (secondary data
only), M2 (primary data + default gap filling + pedigree uncertainty).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import params as P
from .model import to_canonical, UnitError, QUANTITIES
from .factors import build_factor_db
from .dq import pedigree_sigma, row_dq
from .ml import lot_features, IntensityImputer, screening_scores, plausibility_violation

ACT = [("spinning", "electricity"), ("fabric", "electricity"), ("wet", "electricity"), ("garment", "electricity"),
       ("wet", "heat"), ("wet", "water")]
TIER_RANK = {"measured": 0, "calculated": 1, "estimated": 2}
VER_RANK = {"third_party": 0, "second_party": 1, "self_declared": 2, "none": 3}
DEFAULT_YIELD = dict(wet_loss=0.04, fabric_waste=0.035, spin_staple=0.115, spin_filament=0.02)
E_F_ASSUMED = 0.80   # efficiency of the recycling process producing recycled feedstock (MCI), assumption


@dataclass
class Method:
    name: str
    use_primary: bool = True
    missing: str = "default"          # zero | default | ml
    screening: str = "none"           # none | rules | rules+model
    calibration: bool = False         # bias/dispersion calibration against audit measurements
    uncertainty: str = "pedigree"     # none | pedigree | empirical
    pedigree: str = "ecoinvent"       # ecoinvent | ciroth (pedigree factor scheme)
    shared_factors: bool = True       # correlated factor draws across rows/batches
    screen_threshold: float = 4.0


METHODS = {
    "M0": Method("M0 naive aggregation", missing="zero", uncertainty="none"),
    "M1": Method("M1 secondary data only", use_primary=False, missing="default", uncertainty="pedigree"),
    "M2": Method("M2 primary+default, pedigree (ecoinvent)", missing="default", screening="rules"),
    "M2c": Method("M2 primary+default, pedigree (Ciroth)", missing="default", screening="rules", pedigree="ciroth"),
    "M3": Method("M3 proposed", missing="ml", screening="rules+model", calibration=True, uncertainty="empirical"),
    "M3-noML": Method("M3 without ML imputation", missing="default", screening="rules+model", calibration=True,
                      uncertainty="empirical"),
    "M3-noscreen": Method("M3 without model screening", missing="ml", screening="rules", calibration=True,
                          uncertainty="empirical"),
    "M3-nocal": Method("M3 without empirical calibration", missing="ml", screening="rules+model",
                       calibration=False, uncertainty="pedigree", pedigree="ciroth"),
}


@dataclass
class ObservedView:
    """The subset of a scenario visible to the analytics layer (no ground truth)."""
    dpp: pd.DataFrame
    components: pd.DataFrame
    events: pd.DataFrame
    observations: pd.DataFrame
    certificates: pd.DataFrame
    claims: pd.DataFrame
    lab_tests: pd.DataFrame
    facility_master: pd.DataFrame
    batch_master: pd.DataFrame
    fibre_lot_master: pd.DataFrame


def observed_view(sc) -> ObservedView:
    obs = sc.observations.drop(columns=["true_value"])
    lab = sc.lab_tests.drop(columns=["true_conc"])
    flm = sc.fibre_lots[["lot_id", "batch_id", "material", "facility_id", "country"]].copy()   # labelled material
    return ObservedView(sc.dpp, sc.components, sc.events, obs, sc.certificates, sc.claims, lab,
                        sc.facility_master, sc.batch_master, flm)


@dataclass
class Result:
    method: Method
    indicators: pd.DataFrame
    contrib: pd.DataFrame
    decisions: pd.DataFrame
    calibration: pd.DataFrame
    samples: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    imputers: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
def canonicalise(O: pd.DataFrame) -> pd.DataFrame:
    O = O.copy()
    O["value_c"] = np.nan
    O["unit_ok"] = True
    for (q, u), idx in O.groupby(["quantity", "unit"]).groups.items():
        if q in QUANTITIES:
            try:
                f = to_canonical(1.0, u, q)
            except UnitError:
                O.loc[idx, "unit_ok"] = False
                continue
        elif q == "fibre_water_intensity":
            f = 1e-3 if u == "L/kg" else 1.0
        else:
            f = 1.0
        O.loc[idx, "value_c"] = O.loc[idx, "value"].to_numpy(float) * f
    return O


class Engine:
    def __init__(self, view: ObservedView, method: Method, n_samples: int = 1000, seed: int = 0,
                 keep_samples: bool = False, chunk: int = 1000):
        self.v, self.m, self.S, self.seed = view, method, n_samples, seed
        self.keep_samples, self.chunk = keep_samples, chunk
        self.F = build_factor_db()
        self.t = {}

    # ------------------------------------------------------------------
    def run(self) -> Result:
        t0 = time.perf_counter()
        v, m = self.v, self.m
        if m.uncertainty == "empirical":
            self.calibrate_fibre_factors()
        O = canonicalise(v.observations)
        self.t["normalise"] = time.perf_counter() - t0
        bm = v.batch_master.set_index("batch_id")
        fm = v.facility_master.set_index("facility_id")

        # ---------------- stage lots and masses ------------------------------
        t = time.perf_counter()
        L = pd.concat([pd.DataFrame(dict(batch_id=bm.index, stage=st, facility_id=bm[f"fac_{st}"].values,
                                         lot_id=bm.index + "-" + st[:3].upper())) for st in ("spinning", "fabric", "wet", "garment")],
                      ignore_index=True)
        L["country"] = fm.loc[L.facility_id, "country"].values
        mass = O[O.quantity.isin(["mass_in", "mass_out"]) & O.value_c.notna()].pivot_table(
            index="lot_id", columns="quantity", values="value_c", aggfunc="first")
        L = L.join(mass, on="lot_id")
        derived = self._derived_masses(bm)
        for col in ("mass_in", "mass_out"):
            key = L.lot_id + "|" + col
            dm = key.map(derived)
            L[f"{col}_derived"] = L[col].isna()
            L[col] = L[col].fillna(dm)
        self.t["masses"] = time.perf_counter() - t

        # ---------------- evidence-tier selection ------------------------------
        t = time.perf_counter()
        A = O[O.quantity.isin(["electricity", "heat", "water", "wastewater", "chemicals", "waste", "waste_recovered"])].copy()
        A["tier_rank"] = A.data_type.map(TIER_RANK).fillna(9)
        A["ver_rank"] = A.verification.map(VER_RANK).fillna(9)
        A["is_audit"] = A.source_system == "audit"
        avail = A[A.value_c.notna() & A.unit_ok].sort_values(["lot_id", "quantity", "tier_rank", "ver_rank", "is_audit"],
                                                            ascending=[True, True, True, True, False])
        sel = avail.drop_duplicates(["lot_id", "quantity"], keep="first").set_index(["lot_id", "quantity"])
        self.t["select"] = time.perf_counter() - t

        # ---------------- screening ----------------------------------------------
        t = time.perf_counter()
        rejected = set()
        scores = pd.Series(dtype=float)
        if m.screening != "none":
            rejected, scores = self._screen(avail, L, bm, fm, m.screen_threshold, m.screening == "rules+model")
            # re-select among non-rejected values
            avail2 = avail[~avail.obs_id.isin(rejected)]
            sel = avail2.drop_duplicates(["lot_id", "quantity"], keep="first").set_index(["lot_id", "quantity"])
        self.t["screen"] = time.perf_counter() - t

        # ---------------- calibration against audits --------------------------------
        t = time.perf_counter()
        calib = self._calibrate(avail[~avail.obs_id.isin(rejected)]) if m.calibration else pd.DataFrame()
        self.t["calibrate"] = time.perf_counter() - t

        # ---------------- activity values per stage lot --------------------------------
        t = time.perf_counter()
        imputers = {}
        rows = []
        for st, q in ACT:
            Ls = L[L.stage == st].copy()
            key = list(zip(Ls.lot_id, [q] * len(Ls)))
            s = sel.reindex(key)
            Ls["obs_id"] = s.obs_id.values
            Ls["value"] = s.value_c.values
            Ls["tier"] = np.where(s.value_c.notna().values, s.data_type.values, "missing")
            Ls["verification"] = np.where(s.value_c.notna().values, s.verification.values, "none")
            Ls["sigma"] = 0.0
            Ls["resid_group"] = ""
            if not m.use_primary:
                Ls["value"], Ls["tier"], Ls["obs_id"] = np.nan, "missing", None
            # tier-specific uncertainty / calibration
            for tier in ("measured", "calculated", "estimated"):
                mk = Ls.tier.values == tier
                if not mk.any():
                    continue
                if m.uncertainty == "pedigree":
                    Ls.loc[mk, "sigma"] = pedigree_sigma(tier, scheme=m.pedigree)
                elif m.uncertainty == "empirical":
                    if tier == "measured":
                        Ls.loc[mk, "sigma"] = 0.05
                    else:
                        b, sd = self._calib_lookup(calib, q, tier)
                        Ls.loc[mk, "value"] = Ls.loc[mk, "value"] / np.exp(b)
                        Ls.loc[mk, "sigma"] = sd
            # gap filling
            gap = Ls.tier.values == "missing"
            if gap.any():
                if m.missing == "zero":
                    Ls.loc[gap, "value"] = 0.0
                    Ls.loc[gap, "tier"] = "zero"
                elif m.missing == "default":
                    dval = self._default_intensity(st, q) * Ls.loc[gap, "mass_in"].to_numpy()
                    Ls.loc[gap, "value"] = dval
                    Ls.loc[gap, "tier"] = "default"
                    Ls.loc[gap, "verification"] = "platform"
                    if m.uncertainty == "pedigree":
                        Ls.loc[gap, "sigma"] = pedigree_sigma("default", scheme=m.pedigree)
                    elif m.uncertainty == "empirical":
                        Ls.loc[gap, "sigma"] = self._default_rms(sel, L, st, q)
                elif m.missing == "ml":
                    imp, mu, seen = self._impute(sel, L, bm, fm, st, q, Ls[gap])
                    imputers[(st, q)] = imp
                    Ls.loc[gap, "value"] = np.exp(mu) * Ls.loc[gap, "mass_in"].to_numpy()
                    Ls.loc[gap, "tier"] = "imputed"
                    Ls.loc[gap, "verification"] = "platform"
                    if m.uncertainty == "pedigree":
                        Ls.loc[gap, "sigma"] = pedigree_sigma("imputed", scheme=m.pedigree)
                    else:
                        Ls.loc[gap, "resid_group"] = np.where(seen, f"{st}:{q}:seen", f"{st}:{q}:unseen")
            Ls["quantity"] = q
            rows.append(Ls)
        LA = pd.concat(rows, ignore_index=True)
        self.t["gapfill"] = time.perf_counter() - t

        # ---------------- contribution table ----------------------------------------------
        t = time.perf_counter()
        C = self._contributions(LA, O, sel, L, bm, fm, calib)
        self.t["contributions"] = time.perf_counter() - t

        # ---------------- propagation -----------------------------------------------------
        t = time.perf_counter()
        ind, samples = self._propagate(C, bm, imputers)
        self.t["propagate"] = time.perf_counter() - t
        t = time.perf_counter()
        ind = ind.join(self._other_indicators(O, sel, L, bm, fm, calib))
        self.t["other_indicators"] = time.perf_counter() - t
        self.t["total"] = time.perf_counter() - t0
        dec = LA[["batch_id", "stage", "quantity", "lot_id", "facility_id", "obs_id", "tier", "verification", "value",
                  "sigma", "resid_group"]].copy()
        dec["screen_rejected"] = dec.lot_id.map(lambda x: False)
        if len(rejected):
            rj = avail[avail.obs_id.isin(rejected)]
            rej_lots = set(zip(rj.lot_id, rj.quantity))
            dec["screen_rejected"] = [(l, q) in rej_lots for l, q in zip(dec.lot_id, dec.quantity)]
        res = Result(m, ind.reset_index(), C, dec, calib, samples if self.keep_samples else {}, dict(self.t), imputers)
        res.rejected = rejected
        res.scores = scores
        return res

    # ------------------------------------------------------------------
    def _derived_masses(self, bm) -> dict:
        """Transformation rules: derive stage masses from product mass and default yields."""
        out = {}
        for b, r in bm.iterrows():
            Mg = r.units * r.unit_mass
            Mf = Mg / (1 - r.cut_waste)
            Mgf = Mf / (1 - DEFAULT_YIELD["wet_loss"])
            My = Mgf / (1 - DEFAULT_YIELD["fabric_waste"])
            ws = DEFAULT_YIELD["spin_staple"] if r.category != "jacket" else DEFAULT_YIELD["spin_filament"]
            Mfib = My / (1 - ws)
            for st, mi, mo in (("SPI", Mfib, My), ("FAB", My, Mgf), ("WET", Mgf, Mf), ("GAR", Mf, Mg)):
                out[f"{b}-{st}|mass_in"] = mi
                out[f"{b}-{st}|mass_out"] = mo
        return out

    def _default_intensity(self, st, q):
        D = self.F
        if q == "electricity":
            k = {"spinning": "spin_kwh_per_kg", "wet": "wet_elec_kwh_per_kg", "garment": None, "fabric": None}[st]
            if st == "fabric":
                return D["default:weave_kwh_per_kg"].value * 0.6 + D["default:knit_kwh_per_kg"].value * 0.4
            if st == "garment":
                return D["default:garment_kwh_per_kg"].value
            return D[f"default:{k}"].value
        if q == "heat":
            return D["default:wet_heat_mj_per_kg"].value
        if q == "water":
            return D["default:wet_water_l_per_kg"].value
        raise KeyError((st, q))

    def _intensity_table(self, sel, L, st, q, measured_only=True):
        s = sel.reset_index()
        s = s[(s.quantity == q) & (s.stage == st)]
        if measured_only:
            s = s[s.data_type == "measured"]
        Ls = L[L.stage == st].set_index("lot_id")
        s = s[s.lot_id.isin(Ls.index)]
        mass = Ls.loc[s.lot_id, "mass_in"].to_numpy()
        return s.assign(mass_in=mass, intensity=s.value_c.to_numpy() / mass)

    def _default_rms(self, sel, L, st, q):
        s = self._intensity_table(sel, L, st, q)
        if len(s) < 10:
            return pedigree_sigma("default", scheme="ciroth")
        r = np.log(s.intensity.to_numpy() / self._default_intensity(st, q))
        return float(np.sqrt(np.mean(r ** 2)))

    def _impute(self, sel, L, bm, fm, st, q, gaprows):
        s = self._intensity_table(sel, L, st, q)
        X = lot_features(s.assign(stage=st), bm.reset_index(), fm.reset_index())
        y = np.log(s.intensity.to_numpy())
        imp = IntensityImputer(seed=self.seed).fit(X, y, s.facility_id.to_numpy())
        Xg = lot_features(gaprows.assign(stage=st), bm.reset_index(), fm.reset_index())
        mu, seen = imp.predict(Xg, gaprows.facility_id.to_numpy())
        return imp, mu, seen

    def _screen(self, avail, L, bm, fm, thr, use_model=True):
        rejected, allscores = set(), []
        Lm = L.set_index("lot_id")
        for st, q in ACT:
            s = avail[(avail.stage == st) & (avail.quantity == q) & ~avail.is_audit]
            s = s[s.lot_id.isin(Lm.index)]
            if len(s) == 0:
                continue
            mass = Lm.loc[s.lot_id, "mass_in"].to_numpy()
            inten = s.value_c.to_numpy() / mass
            rule = plausibility_violation(np.full(len(s), st), np.full(len(s), q), inten)
            if use_model and len(s) >= 30:
                X = lot_features(s.assign(mass_in=mass), bm.reset_index(), fm.reset_index())
                sc = screening_scores(X, np.log(np.maximum(inten, 1e-9)), s.facility_id.to_numpy(), seed=self.seed)
            else:
                sc = np.zeros(len(s))
            flag = rule | (sc > thr)
            rejected |= set(s.obs_id[flag])
            allscores.append(pd.Series(np.where(rule, np.inf, sc), index=s.obs_id.values))
        return rejected, (pd.concat(allscores) if allscores else pd.Series(dtype=float))

    def _calibrate(self, avail):
        au = avail[avail.is_audit].set_index(["lot_id", "quantity"]).value_c
        na = avail[~avail.is_audit & avail.quantity.isin(["electricity", "heat", "water"])]
        na = na[na.set_index(["lot_id", "quantity"]).index.isin(au.index)]
        r = np.log(na.value_c.to_numpy() / au.reindex(list(zip(na.lot_id, na.quantity))).to_numpy())
        na = na.assign(r=r)
        out = []
        for (q, tier), g in na.groupby(["quantity", "data_type"]):
            med = float(np.median(g.r))
            mad = float(np.median(np.abs(g.r - med)) * 1.4826)
            out.append(dict(quantity=q, tier=tier, n=len(g), bias=med, sigma=mad))
        for tier, g in na.groupby("data_type"):
            med = float(np.median(g.r))
            out.append(dict(quantity="*", tier=tier, n=len(g), bias=med, sigma=float(np.median(np.abs(g.r - med)) * 1.4826)))
        return pd.DataFrame(out)

    @staticmethod
    def _calib_lookup(calib, q, tier):
        if calib is None or len(calib) == 0:
            return 0.0, pedigree_sigma(tier, scheme="ciroth")
        r = calib[(calib.quantity == q) & (calib.tier == tier)]
        if len(r) and r.n.iat[0] >= 20:
            return r.bias.iat[0], max(r.sigma.iat[0], 0.02)
        r = calib[(calib.quantity == "*") & (calib.tier == tier)]
        if len(r) and r.n.iat[0] >= 10:
            return r.bias.iat[0], max(r.sigma.iat[0], 0.02)
        return 0.0, pedigree_sigma(tier, scheme="ciroth")

    # ------------------------------------------------------------------
    def _contributions(self, LA, O, sel, L, bm, fm, calib) -> pd.DataFrame:
        m, F = self.m, self.F
        parts = []
        # --- electricity (GWP + energy) ------------------------------------------------
        el = LA[LA.quantity == "electricity"]
        re = fm.loc[el.facility_id, "renew_share_declared"].fillna(0).to_numpy()
        re = np.where(fm.loc[el.facility_id, "renew_evidence"].to_numpy().astype(bool) & m.use_primary, re, 0.0)
        base = dict(batch_id=el.batch_id.values, stage=el.stage.values, activity="electricity", obs_id=el.obs_id.values,
                    act=el.value.values, act_sigma=el.sigma.values, resid_group=el.resid_group.values,
                    tier=el.tier.values, verification=el.verification.values)
        parts.append(pd.DataFrame(dict(base, indicator="gwp", factor_id="grid:" + el.country.values,
                                       mult=1 - re, specificity="country")))
        parts.append(pd.DataFrame(dict(base, indicator="energy", factor_id="", mult=1.0, specificity="none")))
        parts.append(pd.DataFrame(dict(base, indicator="renewable_kwh", factor_id="", mult=re, specificity="none")))
        parts.append(pd.DataFrame(dict(base, indicator="renewable_kwh_location", factor_id="",
                                       mult=el.country.map(lambda c: P.COUNTRIES[c]["renew"]).values, specificity="none")))
        # --- process heat --------------------------------------------------------------------
        he = LA[LA.quantity == "heat"]
        fuel = fm.loc[he.facility_id, "fuel"].to_numpy()
        use_fuel = pd.notna(fuel) & m.use_primary
        fid = np.where(use_fuel, "fuel:" + np.where(pd.notna(fuel), fuel, "gas").astype(str), "fuelmix:" + he.country.values)
        base = dict(batch_id=he.batch_id.values, stage=he.stage.values, activity="heat", obs_id=he.obs_id.values,
                    act=he.value.values, act_sigma=he.sigma.values, resid_group=he.resid_group.values,
                    tier=he.tier.values, verification=he.verification.values)
        parts.append(pd.DataFrame(dict(base, indicator="gwp", factor_id=fid, mult=1.0,
                                       specificity=np.where(use_fuel, "specific", "country"))))
        parts.append(pd.DataFrame(dict(base, indicator="energy", factor_id="", mult=1 / 3.6, specificity="none")))
        bio = np.where(use_fuel & (fuel == "biomass"), 1 / 3.6, 0.0)
        parts.append(pd.DataFrame(dict(base, indicator="renewable_energy_heat", factor_id="", mult=bio, specificity="none")))
        # --- wet-processing water consumption (withdrawal x consumption ratio) ---------------------
        wa = LA[LA.quantity == "water"]
        ww = sel.reindex(list(zip(wa.lot_id, ["wastewater"] * len(wa))))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = 1 - ww.value_c.to_numpy() / wa.value.to_numpy()
        ok = np.isfinite(ratio) & (ratio > 0) & (ratio < 0.6) & m.use_primary
        if m.uncertainty == "empirical" and m.use_primary:
            obs_ratio = ratio[ok & (ww.data_type.to_numpy() == "measured") & (wa.tier.to_numpy() == "measured")]
            dflt = float(np.median(obs_ratio)) if len(obs_ratio) > 10 else F["default:consumption_ratio"].value
            dsig = float(np.std(np.log(obs_ratio))) if len(obs_ratio) > 10 else F["default:consumption_ratio"].sigma
        else:
            dflt, dsig = F["default:consumption_ratio"].value, F["default:consumption_ratio"].sigma
        cratio = np.where(ok, ratio, dflt)
        csig = np.where(ok, 0.25, dsig)   # ratio of two reported quantities is itself uncertain
        base = dict(batch_id=wa.batch_id.values, stage=wa.stage.values, activity="water_consumption",
                    obs_id=wa.obs_id.values, act=wa.value.values * cratio,
                    act_sigma=np.sqrt(wa.sigma.values ** 2 + csig ** 2), resid_group=wa.resid_group.values,
                    tier=wa.tier.values, verification=wa.verification.values)
        parts.append(pd.DataFrame(dict(base, indicator="water", factor_id="", mult=1.0, specificity="none")))
        parts.append(pd.DataFrame(dict(base, indicator="aware", factor_id="",
                                       mult=wa.country.map(lambda c: P.COUNTRIES[c]["aware_na"]).values, specificity="country")))
        parts.append(pd.DataFrame(dict(base, indicator="water_withdrawal", factor_id="", mult=1 / np.maximum(cratio, 1e-9),
                                       specificity="none")))
        # --- fibre production ---------------------------------------------------------------------
        FLm = self.v.fibre_lot_master
        fmass = O[(O.quantity == "mass_out") & (O.stage == "fibre")].set_index("lot_id").value_c
        comp = self.v.batch_master.set_index("batch_id")
        Lsp = L[L.stage == "spinning"].set_index("batch_id").mass_in
        share = FLm.groupby("batch_id").lot_id.transform("count")
        # declared composition share is not always available: use lot master order and equal split as last resort
        mass_fl = fmass.reindex(FLm.lot_id).to_numpy()
        missing_mass = np.isnan(mass_fl)
        if missing_mass.any():
            decl = self._declared_shares(FLm)
            mass_fl = np.where(missing_mass, Lsp.reindex(FLm.batch_id).to_numpy() * decl, mass_fl)
        prim_g = O[O.quantity == "fibre_gwp_intensity"].set_index("lot_id").value_c
        prim_w = O[O.quantity == "fibre_water_intensity"].set_index("lot_id").value_c
        pg = prim_g.reindex(FLm.lot_id).to_numpy() if m.use_primary else np.full(len(FLm), np.nan)
        pw = prim_w.reindex(FLm.lot_id).to_numpy() if m.use_primary else np.full(len(FLm), np.nan)
        fac_ver = fm.loc[FLm.facility_id, "verification"].to_numpy()
        for ind, prim, pref in (("gwp", pg, "fibre_gwp"), ("water", pw, "fibre_water")):
            has = ~np.isnan(prim)
            fid = np.where(has, "", pref + ":" + FLm.material.values)
            multv = np.where(has, prim, 1.0)
            base = dict(batch_id=FLm.batch_id.values, stage="fibre", activity=f"fibre_{ind}", obs_id=FLm.lot_id.values,
                        act=mass_fl * multv, act_sigma=np.where(has, 0.05, 0.0), resid_group="",
                        tier=np.where(has, "measured", "default"), verification=np.where(has, fac_ver, "platform"))
            parts.append(pd.DataFrame(dict(base, indicator=ind, factor_id=fid, mult=1.0,
                                           specificity=np.where(has, "specific", "generic"))))
            if ind == "water":
                parts.append(pd.DataFrame(dict(base, indicator="aware", factor_id=fid,
                                               mult=FLm.country.map(lambda c: P.COUNTRIES[c]["aware_un"]).values,
                                               specificity=np.where(has, "specific", "generic"))))
        # --- transport ----------------------------------------------------------------------------
        T = O[O.quantity == "transport_work"]
        known = T.value_c.notna().to_numpy() & m.use_primary
        dist = self._default_transport(T, L, FLm, fmass, fm)
        tm = np.where(known, T.tkm_main.to_numpy(), dist["tkm_main"])
        tr = np.where(known, T.tkm_road.to_numpy(), dist["tkm_road"])
        mode = np.where(known, T["mode"].to_numpy(), dist["mode"])
        mode = np.where(pd.isna(mode), "sea", mode).astype(str)
        tier = np.where(known, "measured", "default")
        tsig = np.where(known, 0.10, pedigree_sigma("default", "transport", m.pedigree) if m.uncertainty != "empirical" else 0.35)
        ver = np.where(known, "second_party", "platform")
        main_fid = np.where(mode == "road", "transport:road", "transport:" + mode)
        base = dict(batch_id=T.batch_id.values, stage="transport", activity="transport", obs_id=T.obs_id.values,
                    act_sigma=tsig, resid_group="", tier=tier, verification=ver)
        parts.append(pd.DataFrame(dict(base, act=tm, indicator="gwp", factor_id=main_fid, mult=1.0, specificity="generic")))
        parts.append(pd.DataFrame(dict(base, act=tr, indicator="gwp", factor_id="transport:road", mult=1.0,
                                       specificity="generic")))
        C = pd.concat(parts, ignore_index=True)
        C["act"] = C.act.astype(float).fillna(0.0)
        C["f_val"] = [F[f].value if f else 1.0 for f in C.factor_id]
        if self.m.uncertainty == "none":
            C["f_sigma"] = 0.0
            C["act_sigma"] = 0.0
        else:
            C["f_sigma"] = [self._factor_sigma(f) for f in C.factor_id]
        C["point"] = C.act * C.f_val * C["mult"]
        C["dq"] = row_dq(C.tier.to_numpy(), C.verification.to_numpy(), C.specificity.to_numpy())
        return C

    def _factor_sigma(self, fid):
        if not fid:
            return 0.0
        f = self.F[fid]
        if self.m.uncertainty == "pedigree" and f.specificity == "generic":
            return pedigree_sigma("generic_factor", "emission_factor", self.m.pedigree)
        if self.m.uncertainty == "empirical" and fid.startswith(("fibre_gwp", "fibre_water")):
            return self._fibre_sigma.get(fid, f.sigma) if hasattr(self, "_fibre_sigma") else f.sigma
        return f.sigma

    def calibrate_fibre_factors(self):
        """Empirical dispersion of supplier-specific fibre data around generic factors."""
        O = canonicalise(self.v.observations)
        FLm = self.v.fibre_lot_master.set_index("lot_id")
        self._fibre_sigma = {}
        for q, pref in (("fibre_gwp_intensity", "fibre_gwp"), ("fibre_water_intensity", "fibre_water")):
            s = O[O.quantity == q]
            mat = FLm.loc[s.lot_id, "material"].to_numpy()
            for mt in np.unique(mat):
                v = s.value_c.to_numpy()[mat == mt]
                if len(v) >= 10:
                    r = np.log(v / self.F[f"{pref}:{mt}"].value)
                    self._fibre_sigma[f"{pref}:{mt}"] = float(np.sqrt(np.mean(r ** 2)))

    def _declared_shares(self, FLm):
        comp = self.v.components
        d = {(b, m_): s for b, m_, s in zip(comp.batch_id, comp.material, comp.share)}
        n = FLm.groupby("batch_id").lot_id.transform("count").to_numpy()
        return np.array([d.get((b, m_), 1.0 / k) for b, m_, k in zip(FLm.batch_id, FLm.material, n)])

    def _default_transport(self, T, L, FLm, fmass, fm):
        """Default transport model: regional distance table and default modes."""
        ev = self.v.events[self.v.events.event_type == "shipment"]
        to = {json.loads(i)[0]: json.loads(o)[0] for i, o in zip(ev.inputs, ev.outputs)}
        reg = fm["country"].map(lambda c: P.COUNTRIES[c]["region"]).to_dict()
        cty = fm["country"].to_dict()
        role = fm["role"].to_dict()
        reg["RETAIL"], cty["RETAIL"], role["RETAIL"] = "eu", "DE", "retail"
        Lm = L.set_index("lot_id")
        fl_by = FLm.set_index(["batch_id", "facility_id"]).lot_id.to_dict()
        tkm_main, tkm_road, mode = [], [], []
        for leg, b, frm in zip(T.lot_id, T.batch_id, T.facility_id):
            dst = to.get(leg)
            r = role.get(frm)
            if r == "fibre":
                mass = fmass.get(fl_by.get((b, frm)), np.nan)
                if np.isnan(mass):
                    mass = Lm.at[f"{b}-SPI", "mass_in"] * 0.5
            elif r in ("spinning", "fabric", "wet"):
                mass = Lm.at[f"{b}-{r[:3].upper()}", "mass_out"]
            else:
                mass = Lm.at[f"{b}-GAR", "mass_out"]
            if dst is None:
                tkm_main.append(0.0), tkm_road.append(mass / 1000 * 500), mode.append("road")
                continue
            if r == "dc":
                tkm_main.append(0.0), tkm_road.append(mass / 1000 * 600), mode.append("road")
                continue
            if cty[frm] == cty[dst]:
                tkm_main.append(0.0), tkm_road.append(mass / 1000 * 275), mode.append("road")
                continue
            key = (reg[frm], reg[dst]) if (reg[frm], reg[dst]) in P.REGION_DIST else (reg[dst], reg[frm])
            s_km, r_km = P.REGION_DIST[key]
            tkm_main.append(mass / 1000 * s_km), tkm_road.append(mass / 1000 * r_km)
            mode.append("sea" if s_km > 0 else "road")
        return dict(tkm_main=np.array(tkm_main), tkm_road=np.array(tkm_road), mode=np.array(mode))

    # ------------------------------------------------------------------
    def _propagate(self, C, bm, imputers):
        batches = bm.index.to_numpy()
        bidx = pd.Series(np.arange(len(batches)), index=batches)
        units = bm.units.to_numpy().astype(float)
        mass = units * bm.unit_mass.to_numpy()
        inds = ["gwp", "energy", "water", "aware", "water_withdrawal", "renewable_kwh", "renewable_energy_heat",
                "renewable_kwh_location"]
        point = {}
        for ind in inds:
            c = C[C.indicator == ind]
            point[ind] = np.bincount(bidx.loc[c.batch_id].to_numpy(), weights=c.point.to_numpy(), minlength=len(batches))
        out = pd.DataFrame(index=pd.Index(batches, name="batch_id"))
        out["gwp_total"] = point["gwp"]
        out["gwp_per_unit"] = point["gwp"] / units
        out["gwp_per_kg"] = point["gwp"] / mass
        out["energy_per_unit"] = point["energy"] / units
        el = C[(C.indicator == "energy") & (C.activity == "electricity")]
        elec = np.bincount(bidx.loc[el.batch_id].to_numpy(), weights=el.point.to_numpy(), minlength=len(batches))
        out["renewable_elec_share"] = point["renewable_kwh"] / np.maximum(elec, 1e-9)
        out["renewable_elec_share_location"] = point["renewable_kwh_location"] / np.maximum(elec, 1e-9)
        out["renewable_energy_share"] = (point["renewable_kwh"] + point["renewable_energy_heat"]) / np.maximum(point["energy"], 1e-9)
        out["water_per_unit"] = point["water"] / units
        out["aware_per_unit"] = point["aware"] / units
        out["water_withdrawal_per_unit"] = point["water_withdrawal"] / units
        # stage breakdown of GWP (explanation)
        g = C[C.indicator == "gwp"]
        for st in ("fibre", "spinning", "fabric", "wet", "garment", "transport"):
            gs = g[g.stage == st]
            out[f"gwp_{st}"] = np.bincount(bidx.loc[gs.batch_id].to_numpy(), weights=gs.point.to_numpy(),
                                           minlength=len(batches)) / units
        # data-quality score and primary-data share (GWP)
        w = np.abs(g.point.to_numpy())
        bi = bidx.loc[g.batch_id].to_numpy()
        out["dq_score_gwp"] = np.bincount(bi, weights=w * g.dq.to_numpy(), minlength=len(batches)) / \
            np.maximum(np.bincount(bi, weights=w, minlength=len(batches)), 1e-12)
        prim = np.isin(g.tier.to_numpy(), ["measured", "calculated"])
        out["primary_share_gwp"] = np.bincount(bi, weights=w * prim, minlength=len(batches)) / \
            np.maximum(np.bincount(bi, weights=w, minlength=len(batches)), 1e-12)
        for tier in ("measured", "calculated", "estimated", "imputed", "default", "zero"):
            mk = g.tier.to_numpy() == tier
            out[f"gwp_share_{tier}"] = np.bincount(bi, weights=w * mk, minlength=len(batches)) / \
                np.maximum(np.bincount(bi, weights=w, minlength=len(batches)), 1e-12)
        samples = {}
        if self.m.uncertainty == "none" or self.S == 0:
            return out, samples
        # ---------- Monte Carlo ----------------------------------------------------------
        rng = np.random.default_rng(self.seed + 7)
        resid = {}
        for (st, q), imp in imputers.items():
            resid[f"{st}:{q}:seen"] = imp.resid_
            resid[f"{st}:{q}:unseen"] = imp.resid_unseen_
        fids = sorted(set(C.factor_id) - {""})
        fpos = {f: i for i, f in enumerate(fids)}
        S = self.S
        Zf = rng.normal(size=(len(fids), S)).astype(np.float32)
        for ind, cols in (("gwp", ("gwp_total", "gwp_per_unit")), ("energy", ("energy_per_unit",)),
                          ("water", ("water_per_unit",)), ("aware", ("aware_per_unit",))):
            c = C[C.indicator == ind].reset_index(drop=True)
            n = len(c)
            bi = bidx.loc[c.batch_id].to_numpy()
            tot = np.zeros((len(batches), S), dtype=np.float64)
            for s0 in range(0, n, 20000):
                cc = c.iloc[s0:s0 + 20000]
                k = len(cc)
                la = rng.normal(size=(k, S)).astype(np.float32) * cc.act_sigma.to_numpy(np.float32)[:, None]
                rg = cc.resid_group.to_numpy()
                for grp in np.unique(rg):
                    if grp:
                        mk = rg == grp
                        pool = resid[grp]
                        la[mk] = pool[rng.integers(0, len(pool), size=(mk.sum(), S))]
                fsig = cc.f_sigma.to_numpy(np.float32)[:, None]
                if self.m.shared_factors:
                    pos = np.array([fpos.get(f, -1) for f in cc.factor_id])
                    zf = np.where(pos[:, None] >= 0, Zf[np.maximum(pos, 0)], 0.0)
                else:
                    zf = rng.normal(size=(k, S)).astype(np.float32)
                val = (cc.point.to_numpy()[:, None] * np.exp(la + fsig * zf)).astype(np.float64)
                M = sp.csr_matrix((np.ones(k), (bi[s0:s0 + k], np.arange(k))), shape=(len(batches), k))
                tot += M @ val
            if self.keep_samples:
                samples[ind] = tot
            norm = units if cols[-1].endswith("unit") else np.ones_like(units)
            for col in cols:
                d = units if col.endswith("per_unit") else 1.0
                lo, med, hi = np.percentile(tot / (d if np.ndim(d) == 0 else d[:, None]), [2.5, 50, 97.5], axis=1)
                out[f"{col}_lo"], out[f"{col}_med"], out[f"{col}_hi"] = lo, med, hi
        return out, samples

    # ------------------------------------------------------------------
    def _other_indicators(self, O, sel, L, bm, fm, calib):
        """Material, chemical, waste and circularity indicators (deterministic,
        evidence-based) with explicit non-computability."""
        v = self.v
        out = pd.DataFrame(index=bm.index)
        FLm = v.fibre_lot_master
        fmass = O[(O.quantity == "mass_out") & (O.stage == "fibre")].set_index("lot_id").value_c
        mass = fmass.reindex(FLm.lot_id).to_numpy()
        FLm = FLm.assign(mass=mass)
        # time of sourcing event (evidence of when the lot was produced)
        ev = v.events[v.events.event_type == "raw_material_sourcing"]
        t_src = {json.loads(o)[0]: t for o, t in zip(ev.outputs, ev.time)}
        FLm["t"] = FLm.lot_id.map(t_src)
        certs = v.certificates
        valid = {}
        for kind in ("recycled", "organic"):
            cc = certs[certs.cert_type == kind]
            by = cc.groupby("holder_facility")
            ok = []
            for f, t in zip(FLm.facility_id, FLm.t):
                if pd.isna(t) or f not in by.groups:
                    ok.append(False)
                    continue
                g = by.get_group(f)
                ok.append(bool(((g.valid_from <= t) & (g.valid_to >= t)).any()))
            valid[kind] = np.array(ok)
        rec_label = FLm.material.map(lambda m_: P.MATERIALS[m_]["recycled"]).to_numpy()
        org_label = FLm.material.map(lambda m_: P.MATERIALS[m_]["organic"]).to_numpy()
        FLm["rec_evid"] = rec_label & valid["recycled"]
        FLm["rec_decl"] = rec_label
        FLm["cert_mat"] = (rec_label & valid["recycled"]) | (org_label & valid["organic"])
        g = FLm.groupby("batch_id")
        tot = g.mass.sum(min_count=1)
        complete = g.mass.apply(lambda s: s.notna().all())
        out["recycled_content"] = (FLm.mass * FLm.rec_evid).groupby(FLm.batch_id).sum() / tot
        out["recycled_content_declared"] = (FLm.mass * FLm.rec_decl).groupby(FLm.batch_id).sum() / tot
        out["virgin_share"] = 1 - out.recycled_content
        out["certified_material_ratio"] = (FLm.mass * FLm.cert_mat).groupby(FLm.batch_id).sum() / tot
        out.loc[~complete.reindex(out.index).fillna(False).astype(bool), ["recycled_content", "virgin_share",
                                                                      "certified_material_ratio"]] = np.nan
        # mono-material share and recycling disruptors (design-based proxies)
        comp = v.components
        fam = comp.material.map(lambda m_: P.MATERIALS[m_]["family"])
        fs = comp.assign(fam=fam).groupby(["batch_id", "fam"]).share.sum()
        out["mono_material_share"] = fs.groupby(level=0).max()
        out["elastane_present"] = comp[comp.material == "elastane"].groupby("batch_id").share.sum().reindex(out.index).fillna(0) > 0
        out["fluorinated_finish"] = bm.finish == "wr_fluorinated"
        # chemicals, waste
        garment_mass = bm.units * bm.unit_mass
        s = sel.reset_index()
        chem = s[s.quantity == "chemicals"].set_index("batch_id").value_c
        out["chem_intensity"] = chem.reindex(out.index) / garment_mass
        w = s[s.quantity == "waste"]
        wr = s[s.quantity == "waste_recovered"]
        nwaste = w.groupby("batch_id").value_c.count()
        out["waste_ratio"] = (w.groupby("batch_id").value_c.sum() / garment_mass).where(nwaste.reindex(out.index) == 3)
        out["waste_recovery_rate"] = (wr.groupby("batch_id").value_c.sum() / w.groupby("batch_id").value_c.sum()).where(
            (nwaste.reindex(out.index) == 3) & (wr.groupby("batch_id").value_c.count().reindex(out.index) == 3))
        # restricted substances (only where tested)
        lab = v.lab_tests
        exc = (lab.conc_mg_kg > lab["limit"]).groupby(lab.batch_id).sum()
        out["rs_tested"] = out.index.isin(lab.batch_id.unique())
        out["rs_exceedances"] = exc.reindex(out.index)
        out["svhc_above_threshold"] = ((lab.conc_mg_kg > P.SVHC_THRESHOLD_MG_KG).groupby(lab.batch_id).any()
                                       .reindex(out.index))
        # MCI (design level) with category-level end-of-life data
        eol = O[O.stage == "eol"].pivot_table(index="batch_id", columns="quantity", values="value_c", aggfunc="first")
        eol = eol.join(bm[["category", "units", "unit_mass"]])
        eol["sold_kg"] = eol.units * eol.unit_mass
        cat = eol.groupby("category")[["reuse_kg", "recycling_in_kg", "recycled_out_kg", "sold_kg"]].sum()
        C_U = (cat.reuse_kg / cat.sold_kg).to_dict()
        C_R = (cat.recycling_in_kg / cat.sold_kg).to_dict()
        E_C = (cat.recycled_out_kg / cat.recycling_in_kg).to_dict()
        # EMF (2019): whole-product reuse is not component reuse (C_U = 0); it extends the lifetime and enters
        # through the utility X = L/L_av, approximated as 1 + reused share (second life of equal length; assumption)
        out["mci"] = [mci(fr if pd.notna(fr) else 0.0, 0.0, C_R.get(c, 0.0), 0.0, E_C.get(c, 0.0), E_F_ASSUMED,
                          X=1.0 + C_U.get(c, 0.0))
                      for fr, c in zip(out.recycled_content, bm.category)]
        out["material_recovery_rate_cat"] = [E_C.get(c, np.nan) * C_R.get(c, np.nan) /
                                             max(C_R.get(c, np.nan) + C_U.get(c, np.nan), 1e-9) for c in bm.category]
        return out


def mci(F_R, F_U, C_R, C_U, E_C, E_F, C_C=0.0, C_E=0.0, X=1.0, M=1.0):
    """Material Circularity Indicator (product level, per unit mass M).

    V = M(1-F_R-F_U); W0 = M(1-C_R-C_U-C_C-C_E); W_C = M(1-E_C)C_R;
    W_F = M(1-E_F)F_R/E_F; W = W0 + (W_F+W_C)/2;
    LFI = (V+W) / (2M + (W_F-W_C)/2); F(X) = 0.9/X; MCI = max(0, 1 - LFI*F(X))."""
    V = M * (1 - F_R - F_U)
    W0 = M * (1 - C_R - C_U - C_C - C_E)
    WC = M * (1 - E_C) * C_R
    WF = M * (1 - E_F) * F_R / E_F
    W = W0 + (WF + WC) / 2
    LFI = (V + W) / (2 * M + (WF - WC) / 2)
    return max(0.0, 1 - LFI * 0.9 / X)   # MCI_P = max(0, 1 - LFI * F(X)), F(X) = 0.9 / X
