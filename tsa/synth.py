"""Synthetic textile lifecycle scenario generator.

Generates (i) a GROUND-TRUTH representation of a textile value chain and (ii)
an OBSERVED Digital Product Passport (DPP) layer derived from it by an explicit
data-degradation model (metering level, supplier estimation, missingness,
anomalies, fraud and compliance faults). Ground truth is never visible to the
analytics engine; it is used only for evaluation.

Chain modelled per garment production batch (the DPP unit of analysis):

    fibre lots -> spinning -> fabric formation (weaving|knitting) ->
    wet processing (pretreatment, dyeing, finishing) -> cut-make-trim ->
    logistics -> distribution/retail -> end-of-life (collection, sorting,
    reuse, recycling, residual treatment)

All values are synthetic. See ``params.py`` for parameter provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd

from . import params as P
from .model import gtin14, gln13, digital_link, record_hash

STAGES = ("spinning", "fabric", "wet", "garment")
YEAR_START = np.datetime64("2025-01-01")
ACTIVITY_Q = ("electricity", "heat", "water")          # quantities subject to anomaly injection
UNITS_ALT = {"electricity": ["kWh", "kWh", "kWh", "MWh", "Wh"], "heat": ["MJ", "MJ", "GJ", "kWh"],
             "water": ["m3", "m3", "L"], "wastewater": ["m3", "L"], "chemicals": ["kg", "kg", "t"],
             "waste": ["kg", "t"], "waste_recovered": ["kg", "t"]}
# multiply a canonical value by this to express it in the reported unit
TO_UNIT = {"kWh": 1.0, "MWh": 1e-3, "Wh": 1e3, "MJ": 1.0, "GJ": 1e-3, "m3": 1.0, "L": 1e3, "kg": 1.0, "t": 1e-3}


@dataclass
class ScenarioConfig:
    n_batches: int = 2000
    seed: int = 42
    n_brands: int = 4
    n_facilities: dict = field(default_factory=lambda: {"spinning": 40, "fabric": 50, "wet": 50, "garment": 60})
    fibre_suppliers_per_material: int = 4
    missing_rate: float = 0.25            # additional probability of non-disclosure of an activity datum
    missing_mechanism: str = "MAR"        # MCAR | MAR | MNAR
    anomaly_rate: float = 0.03            # share of reported activity values corrupted
    fraud_rate: float = 0.05              # share of 'recycled' fibre lots that are in truth virgin
    fault_rate: float = 0.04              # base probability for each injected DPP fault type
    primary_fibre_data_prob: float = 0.30
    lab_test_prob: float = 0.35
    eol_prob: float = 0.30                # share of batches with an end-of-life cohort record
    audit_prob: float = 0.08              # share of lots with an independent verification measurement


@dataclass
class Scenario:
    config: ScenarioConfig
    # --- ground truth -----------------------------------------------------
    facilities: pd.DataFrame
    batches: pd.DataFrame
    fibre_lots: pd.DataFrame
    stages: pd.DataFrame
    transport: pd.DataFrame
    lab_truth: pd.DataFrame
    eol: pd.DataFrame
    truth: pd.DataFrame               # true batch-level indicators
    # --- observed DPP layer ----------------------------------------------
    dpp: pd.DataFrame                 # static passport attributes (declared)
    components: pd.DataFrame          # declared fibre composition rows
    events: pd.DataFrame
    observations: pd.DataFrame
    certificates: pd.DataFrame
    claims: pd.DataFrame
    lab_tests: pd.DataFrame
    facility_master: pd.DataFrame     # observed facility attributes
    batch_master: pd.DataFrame        # observed product/process specification (PLM)
    # --- labels -------------------------------------------------------------
    obs_labels: pd.DataFrame          # anomaly labels per observation
    faults: pd.DataFrame              # injected compliance faults (batch level)


# ---------------------------------------------------------------------------
def _choice(rng, weights: dict, n: int):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], float)
    return np.array(keys)[rng.choice(len(keys), size=n, p=p / p.sum())]


def _lognorm(rng, sd, n):
    return np.exp(rng.normal(0.0, sd, n))


# ---------------------------------------------------------------------------
# facilities and certificates
# ---------------------------------------------------------------------------
def _make_facilities(cfg: ScenarioConfig, rng) -> pd.DataFrame:
    rows = []
    fid = 0
    for mat, origins in P.FIBRE_ORIGINS.items():
        for _ in range(cfg.fibre_suppliers_per_material):
            fid += 1
            rows.append(dict(facility_id=f"FAC{fid:04d}", role="fibre", material=mat,
                             country=_choice(rng, origins, 1)[0], route=""))
    for role, n in cfg.n_facilities.items():
        countries = _choice(rng, P.ROLE_COUNTRIES[role], n)
        for i in range(n):
            fid += 1
            route = ("knit" if i % 2 == 0 else "woven") if role == "fabric" else ""
            rows.append(dict(facility_id=f"FAC{fid:04d}", role=role, material="", country=countries[i], route=route))
    for role, c in (("collection", "DE"), ("sorting", "DE"), ("recycling", "PT"), ("dc", "DE")):
        fid += 1
        rows.append(dict(facility_id=f"FAC{fid:04d}", role=role, material="", country=c, route=""))
    f = pd.DataFrame(rows)
    n = len(f)
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    f["eff"] = np.exp(0.20 * z1)                                  # energy efficiency multiplier (>1 = worse)
    f["water_eff"] = np.exp(0.20 * (0.5 * z1 + np.sqrt(0.75) * z2))
    has_re = rng.random(n) < 0.30
    f["renew_share"] = np.where(has_re, rng.uniform(0.1, 0.8, n), 0.0)
    f["renew_evidenced"] = has_re & (rng.random(n) < P.REPORTING["renewable_evidence_prob"])
    f["metering"] = _choice(rng, P.REPORTING["metering_levels"], n)
    f["verification"] = _choice(rng, P.REPORTING["verification_levels"], n)
    f["fuel"] = [_choice(rng, P.COUNTRIES[c]["fuel"], 1)[0] for c in f.country]
    f["fuel_reported"] = rng.random(n) < 0.7
    f["boiler_eff"] = rng.uniform(*P.PROCESS["boiler_eff"], n)
    f["overhead"] = rng.uniform(*P.PROCESS["overhead"], n)
    f["waste_recovery"] = rng.uniform(0.1, 0.95, n)
    f["estimates_when_unmetered"] = rng.random(n) < 0.6
    f["org_id"] = [f"ORG{i // 2 + 1:03d}" for i in range(n)]   # two facilities per organisation
    f["gln"] = [gln13(4000000 + i // 2, i + 1) for i in range(n)]
    f["region"] = [P.COUNTRIES[c]["region"] for c in f.country]
    f["cert_organic"] = rng.random(n) < 0.45
    f["cert_recycled"] = rng.random(n) < 0.45
    f["cert_chemical"] = rng.random(n) < 0.30
    f["cert_ems"] = rng.random(n) < 0.35
    f.loc[f.material == "cotton_org", "cert_organic"] = True
    f.loc[f.material.isin(["cotton_rec", "pes_rec"]), "cert_recycled"] = True
    return f


def _make_certificates(fac: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    cid = 0
    for kind in ("organic", "recycled", "chemical", "ems", "renewable_electricity"):
        col = "renew_evidenced" if kind == "renewable_electricity" else f"cert_{kind}"
        for r in fac[fac[col]].itertuples():
            cid += 1
            start = YEAR_START - np.timedelta64(int(rng.integers(30, 900)), "D")
            if rng.random() < 0.12:        # ~12 % lapse during the observation year (not renewed)
                end = YEAR_START + np.timedelta64(int(rng.integers(30, 330)), "D")
            else:
                end = YEAR_START + np.timedelta64(int(rng.integers(400, 1100)), "D")
            scope = r.material if r.role == "fibre" else r.role
            rows.append(dict(cert_id=f"CERT{cid:05d}", cert_type=kind, holder_facility=r.facility_id,
                             scope=scope, valid_from=start, valid_to=end, issuer=f"CB{int(rng.integers(1, 9)):02d}"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# main generator
# ---------------------------------------------------------------------------
def generate(cfg: ScenarioConfig | None = None) -> Scenario:
    cfg = cfg or ScenarioConfig()
    rng = np.random.default_rng(cfg.seed)
    fac = _make_facilities(cfg, rng)
    certs = _make_certificates(fac, rng)
    fac_ix = fac.set_index("facility_id")

    # ---------------- variants and batches ---------------------------------
    variants = []
    vid = 0
    for cat, spec in P.CATEGORIES.items():
        for ci, comp in enumerate(spec["comps"]):
            for depth in ("light", "medium", "dark"):
                for b in range(cfg.n_brands):
                    vid += 1
                    variants.append(dict(variant_id=f"V{vid:04d}", category=cat, comp_idx=ci,
                                         composition=json.dumps(comp, sort_keys=True), depth=depth,
                                         brand=f"BR{b + 1}", gtin=gtin14(5000000 + b, vid),
                                         finish=spec["finishes"][(ci + b) % len(spec["finishes"])]))
    variants = pd.DataFrame(variants)
    nB = cfg.n_batches
    B = variants.iloc[rng.integers(0, len(variants), nB)].reset_index(drop=True)
    B.insert(0, "batch_id", [f"B{i + 1:06d}" for i in range(nB)])
    B["lot"] = [f"L{x}" for x in rng.integers(10 ** 7, 10 ** 8, nB)]
    B["units"] = np.round(np.exp(rng.uniform(np.log(300), np.log(5000), nB))).astype(int)
    cat = B.category.values
    spec = [P.CATEGORIES[c] for c in cat]
    B["route"] = [s["route"] for s in spec]
    B["unit_mass"] = [rng.uniform(*s["mass"]) for s in spec]
    B["gsm"] = [rng.uniform(*s["gsm"]) for s in spec]
    B["ne"] = [rng.uniform(*s["ne"]) if s["ne"][1] > 0 else 0.0 for s in spec]
    B["cut_waste"] = [rng.uniform(*s["cut"]) for s in spec]
    comps = [json.loads(c) for c in B.composition]
    B["organic_share"] = [sum(v for k, v in c.items() if P.MATERIALS[k]["organic"]) for c in comps]
    B["recycled_share_declared"] = [sum(v for k, v in c.items() if P.MATERIALS[k]["recycled"]) for c in comps]
    staple_set = {m for m in P.MATERIALS if P.MATERIALS[m]["staple"]}
    B["staple"] = [max(c, key=c.get) in staple_set for c in comps]
    fam_share = [{} for _ in comps]
    for d, c in zip(fam_share, comps):
        for k, v in c.items():
            if k != "elastane":
                fam = P.MATERIALS[k]["family"]
                d[fam] = d.get(fam, 0) + v
    dye_map = {"cellulosic": "reactive", "polyester": "disperse", "polyamide": "acid", "protein": "acid"}
    B["dye_classes"] = ["|".join(dict.fromkeys(("vat_indigo" if (ct == "jeans" and fm == "cellulosic") else dye_map[fm])
                                               for fm in sorted(d, key=lambda k: -d[k]))) for ct, d in zip(cat, fam_share)]
    B["garment_mass"] = B.units * B.unit_mass
    B["start"] = YEAR_START + rng.integers(0, 300, nB).astype("timedelta64[D]")

    # ---------------- facility selection (certified chains preferred) ------
    wants_org = B.organic_share.values >= 0.95
    wants_rec = B.recycled_share_declared.values > 0
    pools = {}
    for role in STAGES:
        for route in (("knit", "woven") if role == "fabric" else ("",)):
            base = fac[(fac.role == role) & (fac.route == route)]
            pools[(role, route, "any")] = base.facility_id.values
            pools[(role, route, "org")] = base[base.cert_organic].facility_id.values
            pools[(role, route, "rec")] = base[base.cert_recycled].facility_id.values
    pref = np.where(wants_org & (rng.random(nB) < 0.9), "org", np.where(wants_rec & (rng.random(nB) < 0.9), "rec", "any"))
    for role in STAGES:
        out = np.empty(nB, dtype=object)
        for i in range(nB):
            route = B.route.iat[i] if role == "fabric" else ""
            pool = pools[(role, route, pref[i])]
            if len(pool) == 0:
                pool = pools[(role, route, "any")]
            out[i] = pool[rng.integers(0, len(pool))]
        B[f"fac_{role}"] = out

    # ---------------- mass balance (true) ---------------------------------
    M_g = B.garment_mass.values
    M_f = M_g / (1 - B.cut_waste.values)                       # dyed fabric into cutting
    w_wet = rng.uniform(*P.PROCESS["wet_loss"], nB)
    M_gf = M_f / (1 - w_wet)                                   # greige fabric into wet processing
    w_fab = rng.uniform(*P.PROCESS["fabric_waste"], nB)
    M_y = M_gf / (1 - w_fab)                                   # yarn into fabric formation
    w_spin = np.where(B.staple, rng.uniform(*P.PROCESS["spin_waste_staple"], nB),
                      rng.uniform(*P.PROCESS["spin_waste_filament"], nB))
    M_fib = M_y / (1 - w_spin)                                 # fibre into spinning

    t_fibre = B.start.values
    t_spin = t_fibre + rng.integers(5, 20, nB).astype("timedelta64[D]")
    t_fab = t_spin + rng.integers(5, 15, nB).astype("timedelta64[D]")
    t_wet = t_fab + rng.integers(3, 10, nB).astype("timedelta64[D]")
    t_gar = t_wet + rng.integers(5, 20, nB).astype("timedelta64[D]")
    B["t_garment"] = t_gar

    # ---------------- fibre lots ------------------------------------------
    rows = [(B.batch_id.iat[i], k, mat, share, M_fib[i] * share, t_fibre[i])
            for i in range(nB) for k, (mat, share) in enumerate(sorted(comps[i].items()))]
    FL = pd.DataFrame(rows, columns=["batch_id", "k", "material", "share", "mass_kg", "time"])
    FL["lot_id"] = FL.batch_id + "-F" + FL.k.astype(str)
    sup_pool = {m: fac[(fac.role == "fibre") & (fac.material == m)].facility_id.values for m in P.MATERIALS}
    FL["facility_id"] = [sup_pool[m][rng.integers(0, len(sup_pool[m]))] for m in FL.material]
    FL["country"] = fac_ix.loc[FL.facility_id, "country"].values
    nF = len(FL)
    is_rec_label = FL.material.map(lambda m: P.MATERIALS[m]["recycled"]).values
    fraud = is_rec_label & (rng.random(nF) < cfg.fraud_rate)
    virgin_of = {"cotton_rec": "cotton_conv", "pes_rec": "pes_virgin"}
    FL["true_material"] = np.where(fraud, FL.material.map(lambda m: virgin_of.get(m, m)), FL.material)
    FL["fraud"] = fraud
    sup_factor = pd.Series(np.exp(rng.normal(0, 0.10, len(fac))), index=fac.facility_id)
    sd = FL.true_material.map(lambda m: P.MATERIALS[m]["sd"]).values
    sf = sup_factor.loc[FL.facility_id].values
    FL["gwp_int_true"] = FL.true_material.map(lambda m: P.MATERIALS[m]["gwp"]).values * sf * np.exp(rng.normal(0, sd))
    base_w = np.array([P.FIBRE_WATER_COUNTRY.get((m, c), P.MATERIALS[m]["water"]) for m, c in zip(FL.true_material, FL.country)])
    FL["water_int_true"] = base_w * sf * np.exp(rng.normal(0, sd))
    FL["gwp_true"] = FL.gwp_int_true * FL.mass_kg
    FL["water_true_m3"] = FL.water_int_true * FL.mass_kg / 1000.0
    FL["primary_data"] = rng.random(nF) < cfg.primary_fibre_data_prob
    FL["aware"] = FL.country.map(lambda c: P.COUNTRIES[c]["aware_un"]).values

    # ---------------- process stages (true) -------------------------------
    def fcol(col, attr):
        return fac_ix.loc[B[col], attr].values
    spin_int = np.where(B.staple, P.PROCESS["spin_staple_a"] + P.PROCESS["spin_staple_b"] * B["ne"].values,
                        P.PROCESS["spin_filament"])
    e_spin = M_y * spin_int * fcol("fac_spinning", "eff") * _lognorm(rng, 0.08, nB) * (1 + fcol("fac_spinning", "overhead"))
    ref = np.where(B.route == "woven", P.PROCESS["weave_ref"], P.PROCESS["knit_ref"])
    e_fab = M_gf * ref * np.sqrt(200.0 / B.gsm.values) * fcol("fac_fabric", "eff") * _lognorm(rng, 0.10, nB) * \
        (1 + fcol("fac_fabric", "overhead"))
    water_L = np.zeros(nB)
    heat_useful = np.zeros(nB)
    aux = np.zeros(nB)
    dfac = B.depth.map(P.DEPTH_FACTOR).values
    for i in range(nB):
        for j, dc in enumerate(B.dye_classes.iat[i].split("|")):
            d = P.DYE_CLASSES[dc]
            w = 1.0 if j == 0 else 0.6
            wl = M_gf[i] * rng.uniform(*d["lr"]) * rng.uniform(*d["baths"]) * dfac[i] * w
            water_L[i] += wl
            heat_useful[i] += wl * 4.186e-3 * rng.uniform(*d["dT"])
            aux[i] += M_gf[i] * rng.uniform(*d["aux"]) * w
    weff = fcol("fac_wet", "water_eff")
    water_L *= weff * _lognorm(rng, 0.10, nB)
    heat_useful = heat_useful * weff + M_f * rng.uniform(*P.PROCESS["dryer_moisture"], nB) * \
        P.PROCESS["latent_heat"] / rng.uniform(*P.PROCESS["dryer_eff"], nB)
    fuel_wet = heat_useful / fcol("fac_wet", "boiler_eff") * fcol("fac_wet", "eff") * (1 + fcol("fac_wet", "overhead"))
    e_wet = M_gf * rng.uniform(*P.PROCESS["wet_elec"], nB) * fcol("fac_wet", "eff") * _lognorm(rng, 0.10, nB) * \
        (1 + fcol("fac_wet", "overhead"))
    owf = np.array([rng.uniform(*P.DEPTH_OWF[d]) for d in B.depth])
    fin = np.array([rng.uniform(*P.FINISHES[f]["chem"]) for f in B.finish])
    chem = M_gf * (owf + fin) + aux
    wastewater_m3 = water_L / 1000 * rng.uniform(0.85, 0.95, nB)
    cmt = np.array([P.PROCESS["cmt_kwh"][c] for c in cat])
    e_gar = B.units.values * cmt * fcol("fac_garment", "eff") * _lognorm(rng, 0.10, nB) * (1 + fcol("fac_garment", "overhead"))
    z = np.zeros(nB)
    S = pd.concat([pd.DataFrame(dict(batch_id=B.batch_id, stage=st, facility_id=B[f"fac_{st}"].values, time=t,
                                     mass_in=mi, mass_out=mo, electricity=el, heat=he, water=wa, wastewater=ww,
                                     chemicals=ch))
                   for st, t, mi, mo, el, he, wa, ww, ch in (
                       ("spinning", t_spin, M_fib, M_y, e_spin, z, z, z, z),
                       ("fabric", t_fab, M_y, M_gf, e_fab, z, z, z, z),
                       ("wet", t_wet, M_gf, M_f, e_wet, fuel_wet, water_L / 1000, wastewater_m3, chem),
                       ("garment", t_gar, M_f, M_g, e_gar, z, z, z, z))], ignore_index=True)
    S["lot_id"] = S.batch_id + "-" + S.stage.str[:3].str.upper()
    S["country"] = fac_ix.loc[S.facility_id, "country"].values
    S["waste"] = np.where(S.stage == "wet", 0.0, S.mass_in - S.mass_out)   # wet losses leave with effluent
    S["waste_recovered"] = S.waste * fac_ix.loc[S.facility_id, "waste_recovery"].values
    S["month"] = S.time.values.astype("datetime64[M]")
    grid = S.country.map(lambda c: P.COUNTRIES[c]["grid_ef"]).values
    S["renew_share"] = fac_ix.loc[S.facility_id, "renew_share"].values
    S["ef_elec_true"] = grid * (1 - S.renew_share)       # market-based; contractual renewables = 0
    S["fuel"] = fac_ix.loc[S.facility_id, "fuel"].values
    S["ef_heat_true"] = S.fuel.map(P.FUEL_EF).values
    S["gwp_true"] = S.electricity * S.ef_elec_true + S.heat * S.ef_heat_true
    S["water_consumption"] = S.water - S.wastewater
    S["aware"] = S.country.map(lambda c: P.COUNTRIES[c]["aware_na"]).values
    S["grid_renew"] = S.country.map(lambda c: P.COUNTRIES[c]["renew"]).values

    # ---------------- transport legs (true) -------------------------------
    TL = _transport(rng, B, FL, M_y, M_gf, M_f, M_g, fac, fac_ix)

    # ---------------- restricted substances (true) ------------------------
    lab_rows = []
    for i in range(nB):
        fin_ = B.finish.iat[i]
        for sub, sp in P.SUBSTANCES.items():
            if sub == "pfas":
                if fin_ == "wr_fluorinated":
                    conc = float(np.exp(rng.normal(np.log(200), 0.5)))
                elif fin_ == "wr_fluorine_free" and rng.random() < 0.05:
                    conc = float(np.exp(rng.normal(np.log(10), 1.0)))
                else:
                    conc = 0.0
            else:
                conc = float(np.exp(rng.normal(np.log(sp["limit"]), 0.8))) if rng.random() < sp["p_base"] else 0.0
            lab_rows.append((B.batch_id.iat[i], sub, conc, sp["limit"]))
    LT = pd.DataFrame(lab_rows, columns=["batch_id", "substance", "true_conc", "limit"])

    # ---------------- end-of-life cohort (true) ---------------------------
    has_eol = rng.random(nB) < cfg.eol_prob
    E = B.loc[has_eol, ["batch_id", "units", "unit_mass"]].copy()
    nE = len(E)
    E["collected_kg"] = E.units * E.unit_mass * rng.uniform(*P.EOL["collection_rate"], nE)
    E["reuse_kg"] = E.collected_kg * rng.uniform(*P.EOL["reuse_share"], nE)
    E["recycling_in_kg"] = E.collected_kg * rng.uniform(*P.EOL["recycling_share"], nE)
    E["residual_kg"] = E.collected_kg - E.reuse_kg - E.recycling_in_kg
    E["recycled_out_kg"] = E.recycling_in_kg * rng.uniform(*P.EOL["recycling_yield"], nE)
    E["time"] = B.loc[has_eol, "t_garment"].values + np.timedelta64(365, "D")

    truth = _true_indicators(B, FL, S, TL)

    # ======================================================================
    # OBSERVED DPP LAYER
    # ======================================================================
    obs, obs_lab = _observe(cfg, rng, FL, S, TL, E, fac_ix)
    events = _events(B, FL, S, TL, E, fac)
    lab_obs = _lab_observed(cfg, rng, LT, B)
    dpp, comp_rows, claims, faults, events = _passport_and_claims(cfg, rng, B, FL, fac_ix, certs, lab_obs, events)
    fm = fac[["facility_id", "gln", "org_id", "role", "material", "country", "route"]].copy()
    fm["fuel"] = np.where(fac.fuel_reported, fac.fuel, None)
    fm["renew_share_declared"] = np.where(fac.renew_evidenced, fac.renew_share, 0.0)
    fm["renew_evidence"] = fac.renew_evidenced
    fm["verification"] = fac.verification
    fm["metering"] = fac.metering
    bm = B[["batch_id", "variant_id", "category", "brand", "route", "gsm", "ne", "depth", "finish", "dye_classes",
            "units", "unit_mass", "cut_waste", "fac_spinning", "fac_fabric", "fac_wet", "fac_garment"]].copy()
    return Scenario(cfg, fac, B, FL, S, TL, LT, E, truth, dpp, comp_rows, events, obs,
                    certs, claims, lab_obs, fm, bm, obs_lab, faults)


# ---------------------------------------------------------------------------
def _transport(rng, B, FL, M_y, M_gf, M_f, M_g, fac, fac_ix):
    dc = fac[fac.role == "dc"].facility_id.iat[0]
    bt = B.set_index("batch_id")
    legs = [pd.DataFrame(dict(batch_id=FL.batch_id, leg="fibre->spinning", from_fac=FL.facility_id,
                              to_fac=bt.loc[FL.batch_id, "fac_spinning"].values, mass_kg=FL.mass_kg,
                              time=FL.time.values))]
    for a, b_, m, nm in (("fac_spinning", "fac_fabric", M_y, "spinning->fabric"),
                         ("fac_fabric", "fac_wet", M_gf, "fabric->wet"),
                         ("fac_wet", "fac_garment", M_f, "wet->garment")):
        legs.append(pd.DataFrame(dict(batch_id=B.batch_id, leg=nm, from_fac=B[a].values, to_fac=B[b_].values,
                                      mass_kg=m, time=B.start.values)))
    legs.append(pd.DataFrame(dict(batch_id=B.batch_id, leg="garment->dc", from_fac=B.fac_garment.values, to_fac=dc,
                                  mass_kg=M_g, time=B.t_garment.values)))
    legs.append(pd.DataFrame(dict(batch_id=B.batch_id, leg="dc->retail", from_fac=dc, to_fac="RETAIL",
                                  mass_kg=M_g, time=B.t_garment.values + np.timedelta64(40, "D"))))
    TL = pd.concat(legs, ignore_index=True)
    reg = fac_ix["region"].to_dict()
    cty = fac_ix["country"].to_dict()
    reg["RETAIL"], cty["RETAIL"] = "eu", "DE"
    n = len(TL)
    km_main, km_road, mode = np.zeros(n), np.zeros(n), np.empty(n, dtype=object)
    u1, u2, u3 = rng.uniform(0.8, 1.2, n), rng.uniform(0.7, 1.3, n), rng.random(n)
    loc = rng.uniform(50, 500, n)
    for j, (a, b_, leg) in enumerate(zip(TL.from_fac.values, TL.to_fac.values, TL.leg.values)):
        if leg == "dc->retail":
            km_road[j], mode[j] = 200 + 800 * u3[j], "road"
            continue
        if cty[a] == cty[b_]:
            km_road[j], mode[j] = loc[j], "road"
            continue
        key = (reg[a], reg[b_]) if (reg[a], reg[b_]) in P.REGION_DIST else (reg[b_], reg[a])
        s_km, r_km = P.REGION_DIST[key]
        s_km, r_km = s_km * u1[j], r_km * u2[j]
        if s_km > 0 and leg == "garment->dc" and u3[j] < P.AIR_SHARE:
            km_main[j], km_road[j], mode[j] = 0.5 * s_km, r_km, "air"
        elif s_km > 0:
            km_main[j], km_road[j], mode[j] = s_km, r_km, "sea"
        else:
            km_road[j], mode[j] = r_km, "road"
    TL["km_main"], TL["km_road"], TL["mode"] = km_main, km_road, mode
    TL["tkm_main"] = TL.mass_kg / 1000 * TL.km_main
    TL["tkm_road"] = TL.mass_kg / 1000 * TL.km_road
    ef_main = TL["mode"].map(lambda m: P.TRANSPORT_EF[m]).values
    TL["gwp_true"] = TL.tkm_main * np.where(TL["mode"] == "road", 0.0, ef_main) + TL.tkm_road * P.TRANSPORT_EF["road"]
    TL["leg_id"] = TL.batch_id + "-T" + TL.groupby("batch_id").cumcount().astype(str)
    return TL


# ---------------------------------------------------------------------------
def _true_indicators(B, FL, S, TL) -> pd.DataFrame:
    FL = FL.assign(aw=FL.water_true_m3 * FL.aware,
                   rec_mass=FL.mass_kg * FL.true_material.map(lambda m: P.MATERIALS[m]["recycled"]).astype(float))
    g_fibre = FL.groupby("batch_id").agg(gwp_fibre=("gwp_true", "sum"), water_fibre=("water_true_m3", "sum"),
                                         aware_fibre=("aw", "sum"), fibre_mass=("mass_kg", "sum"),
                                         rec_mass=("rec_mass", "sum"))
    S = S.assign(aw=S.water_consumption * S.aware, ren_kwh=S.electricity * S.renew_share,
                 loc_kwh=S.electricity * S.grid_renew,
                 bio_kwh=np.where(S.fuel == "biomass", S.heat / 3.6, 0.0))
    g_st = S.groupby("batch_id").agg(gwp_mfg=("gwp_true", "sum"), elec=("electricity", "sum"), heat=("heat", "sum"),
                                     water_withdrawal=("water", "sum"), water_cons_mfg=("water_consumption", "sum"),
                                     aware_mfg=("aw", "sum"), chem=("chemicals", "sum"), waste=("waste", "sum"),
                                     waste_rec=("waste_recovered", "sum"), ren_kwh=("ren_kwh", "sum"),
                                     bio_kwh=("bio_kwh", "sum"), loc_kwh=("loc_kwh", "sum"))
    g_tr = TL.groupby("batch_id").agg(gwp_transport=("gwp_true", "sum"))
    T = B[["batch_id", "units", "garment_mass"]].set_index("batch_id").join([g_fibre, g_st, g_tr])
    T["gwp_total"] = T.gwp_fibre + T.gwp_mfg + T.gwp_transport
    T["gwp_per_unit"] = T.gwp_total / T.units
    T["gwp_per_kg"] = T.gwp_total / T.garment_mass
    T["energy_kwh"] = T.elec + T.heat / 3.6
    T["energy_per_unit"] = T.energy_kwh / T.units
    T["renewable_elec_share"] = T.ren_kwh / T.elec
    T["renewable_elec_share_location"] = T.loc_kwh / T.elec
    T["renewable_energy_share"] = (T.ren_kwh + T.bio_kwh) / T.energy_kwh
    T["water_cons_total"] = T.water_fibre + T.water_cons_mfg
    T["water_per_unit"] = T.water_cons_total / T.units
    T["aware_per_unit"] = (T.aware_fibre + T.aware_mfg) / T.units
    T["chem_intensity"] = T.chem / T.garment_mass
    T["waste_ratio"] = T.waste / T.garment_mass
    T["waste_recovery_rate"] = T.waste_rec / T.waste
    T["recycled_content_true"] = T.rec_mass / T.fibre_mass
    return T.reset_index()


# ---------------------------------------------------------------------------
def _observe(cfg, rng, FL, S, TL, E, fac_ix):
    """Derive observed activity data from ground truth (data-degradation model).

    Values are first generated in canonical units, anomalies are injected in
    canonical space, and values are finally expressed in heterogeneous
    reporting units (to exercise unit normalisation)."""
    R = P.REPORTING
    S = S.copy()
    S["metering"] = fac_ix.loc[S.facility_id, "metering"].values
    S["verification"] = fac_ix.loc[S.facility_id, "verification"].values
    S["estimates"] = fac_ix.loc[S.facility_id, "estimates_when_unmetered"].values
    S["region"] = fac_ix.loc[S.facility_id, "region"].values
    grp = S.groupby(["facility_id", "month"])
    for q in ("electricity", "heat", "water"):
        tot = grp[q].transform("sum").values * np.exp(rng.normal(0, 0.02, len(S)))   # metered facility-month total
        S[f"{q}_alloc"] = tot * S.mass_in.values / grp.mass_in.transform("sum").values  # mass-based allocation
    recs = []
    spec = {"electricity": (S.index, "MES"), "heat": (S.index[S.stage == "wet"], "MES"),
            "water": (S.index[S.stage == "wet"], "IoT"), "wastewater": (S.index[S.stage == "wet"], "IoT"),
            "chemicals": (S.index[S.stage == "wet"], "ERP"), "waste": (S.index[S.stage != "wet"], "ERP"),
            "waste_recovered": (S.index[S.stage != "wet"], "ERP")}
    for q, (ix, src) in spec.items():
        sub = S.loc[ix]
        n = len(sub)
        true = sub[q].values
        met = sub.metering.values
        if q in ("electricity", "heat", "water"):
            dtype = np.where(met == "lot", "measured", np.where(met == "facility", "calculated", "estimated"))
            alloc = sub[f"{q}_alloc"].values
        elif q == "wastewater":
            dtype = np.where((met == "lot") & (rng.random(n) < 0.6), "measured", "estimated")
            alloc = true
        else:   # ERP-based: purchases / waste manifests allocated to lots
            dtype = np.where(rng.random(n) < 0.7, "calculated", "estimated")
            alloc = true * np.exp(rng.normal(0, 0.15, n))
        est = true * np.exp(rng.normal(-R["estimate_mean_bias"], R["estimate_bias_sd"], n))
        val = np.where(dtype == "measured", true * np.exp(rng.normal(0, R["measured_noise"], n)),
                       np.where(dtype == "calculated", alloc, est))
        no_est = (met == "none") & ~sub.estimates.values & (dtype == "estimated") & (q in ACTIVITY_Q)
        if cfg.missing_mechanism == "MCAR":
            p_miss = np.full(n, cfg.missing_rate)
        elif cfg.missing_mechanism == "MNAR":          # poorer performers disclose less
            inten = np.log(np.maximum(true, 1e-12) / sub.mass_in.values)
            zz = np.zeros(n)
            for st in np.unique(sub.stage.values):
                m = sub.stage.values == st
                zz[m] = (inten[m] - inten[m].mean()) / (inten[m].std() + 1e-9)
            p_miss = np.clip(cfg.missing_rate * 2 / (1 + np.exp(-1.5 * zz)), 0, 0.95)
        else:                                           # MAR: depends on an observed covariate (region)
            p_miss = np.clip(cfg.missing_rate * np.where(sub.region.values == "asia", 1.3, 0.6), 0, 0.95)
        miss = no_est | (rng.random(n) < p_miss)
        dtype = np.where(miss, "missing", dtype)
        ver = np.where(np.isin(dtype, ["measured", "calculated"]), sub.verification.values, "self_declared")
        ver = np.where(dtype == "missing", "none", ver)
        recs.append(pd.DataFrame(dict(batch_id=sub.batch_id.values, lot_id=sub.lot_id.values, stage=sub.stage.values,
                                      facility_id=sub.facility_id.values, quantity=q,
                                      value_c=np.where(miss, np.nan, val), data_type=dtype, verification=ver,
                                      source_system=src, timestamp=sub.time.values, true_value=true)))
        if q in ACTIVITY_Q:
            # independent third-party verification sample (audit): measured value for ~8 % of lots
            a = rng.random(n) < cfg.audit_prob
            recs.append(pd.DataFrame(dict(batch_id=sub.batch_id.values[a], lot_id=sub.lot_id.values[a],
                                          stage=sub.stage.values[a], facility_id=sub.facility_id.values[a], quantity=q,
                                          value_c=true[a] * np.exp(rng.normal(0, R["measured_noise"], a.sum())),
                                          data_type="measured", verification="third_party", source_system="audit",
                                          timestamp=sub.time.values[a], true_value=true[a])))
    # fibre-lot supplier-specific data (cradle-to-gate intensity), when provided
    fl = FL[FL.primary_data]
    for q, col in (("fibre_gwp_intensity", "gwp_int_true"), ("fibre_water_intensity", "water_int_true")):
        v = fl[col].values * np.exp(rng.normal(0, 0.05, len(fl)))
        recs.append(pd.DataFrame(dict(batch_id=fl.batch_id.values, lot_id=fl.lot_id.values, stage="fibre",
                                      facility_id=fl.facility_id.values, quantity=q, value_c=v, data_type="measured",
                                      verification=fac_ix.loc[fl.facility_id, "verification"].values,
                                      source_system="supplier_portal", timestamp=fl.time.values, true_value=fl[col].values)))
    # masses (ERP/MES)
    for q in ("mass_in", "mass_out"):
        n = len(S)
        ok = rng.random(n) < 0.97
        recs.append(pd.DataFrame(dict(batch_id=S.batch_id.values, lot_id=S.lot_id.values, stage=S.stage.values,
                                      facility_id=S.facility_id.values, quantity=q,
                                      value_c=np.where(ok, S[q].values * np.exp(rng.normal(0, 0.01, n)), np.nan),
                                      data_type=np.where(ok, "measured", "missing"), verification=S.verification.values,
                                      source_system="ERP", timestamp=S.time.values, true_value=S[q].values)))
    ok = rng.random(len(FL)) < 0.97
    recs.append(pd.DataFrame(dict(batch_id=FL.batch_id.values, lot_id=FL.lot_id.values, stage="fibre",
                                  facility_id=FL.facility_id.values, quantity="mass_out",
                                  value_c=np.where(ok, FL.mass_kg.values, np.nan),
                                  data_type=np.where(ok, "measured", "missing"), verification="self_declared",
                                  source_system="supplier_portal", timestamp=FL.time.values, true_value=FL.mass_kg.values)))
    # transport (logistics system): mode and t*km per leg
    n = len(TL)
    known = rng.random(n) < 0.85
    recs.append(pd.DataFrame(dict(batch_id=TL.batch_id.values, lot_id=TL.leg_id.values, stage="transport",
                                  facility_id=TL.from_fac.values, quantity="transport_work",
                                  value_c=np.where(known, (TL.tkm_main + TL.tkm_road).values, np.nan),
                                  data_type=np.where(known, "measured", "missing"), verification="second_party",
                                  source_system="logistics", timestamp=TL.time.values,
                                  true_value=(TL.tkm_main + TL.tkm_road).values,
                                  mode=np.where(known, TL["mode"].values, None),
                                  tkm_main=np.where(known, TL.tkm_main.values, np.nan),
                                  tkm_road=np.where(known, TL.tkm_road.values, np.nan))))
    # end-of-life (recycling system)
    for q in ("collected_kg", "reuse_kg", "recycling_in_kg", "residual_kg", "recycled_out_kg"):
        recs.append(pd.DataFrame(dict(batch_id=E.batch_id.values, lot_id=E.batch_id.values + "-EOL", stage="eol",
                                      facility_id="EOL", quantity=q, value_c=E[q].values, data_type="measured",
                                      verification="second_party", source_system="recycling_system",
                                      timestamp=E.time.values, true_value=E[q].values)))
    O = pd.concat(recs, ignore_index=True)
    O.insert(0, "obs_id", [f"O{i + 1:08d}" for i in range(len(O))])

    # --- anomaly injection (canonical space) --------------------------------------
    lab = pd.DataFrame(dict(obs_id=O.obs_id.values, anomaly=False, anomaly_type=""))
    act = np.where(O.quantity.isin(ACTIVITY_Q).values & O.value_c.notna().values &
                   (O.source_system.values != "audit"))[0]
    pick = act[rng.random(len(act)) < cfg.anomaly_rate]
    types = rng.choice(["unit_error", "under_report", "spike", "stale_copy"], size=len(pick), p=[0.25, 0.35, 0.2, 0.2])
    vals = O.value_c.values.copy()
    keyfq = O.facility_id.values + "|" + O.quantity.values
    for j, t in zip(pick, types):
        v = vals[j]
        if t == "unit_error":
            nv = v * 1000.0                     # value in Wh / L entered under kWh / m3
        elif t == "under_report":
            nv = v * rng.uniform(0.2, 0.6)
        elif t == "spike":
            nv = v * rng.uniform(2.5, 6.0)
        else:                                    # stale copy of another lot of the same facility
            cand = np.where((keyfq == keyfq[j]) & ~np.isnan(vals))[0]
            cand = cand[cand != j]
            nv = vals[cand[rng.integers(0, len(cand))]] if len(cand) else v * 0.4
            if not len(cand):
                t = "under_report"
        if abs(np.log(max(nv, 1e-12) / max(v, 1e-12))) < np.log(1.25):
            continue                              # not a material deviation: leave unlabelled and unchanged
        vals[j] = nv
        lab.iat[j, 1] = True
        lab.iat[j, 2] = t
    O["value_c"] = vals
    # --- heterogeneous reporting units ------------------------------------------------
    unit = np.array(["kg"] * len(O), dtype=object)
    unit[O.quantity.values == "transport_work"] = "tkm"
    unit[O.quantity.values == "fibre_gwp_intensity"] = "kgCO2e/kg"
    unit[O.quantity.values == "fibre_water_intensity"] = "L/kg"
    factor = np.ones(len(O))
    for q, alts in UNITS_ALT.items():
        m = np.where(O.quantity.values == q)[0]
        u = np.array(alts, dtype=object)[rng.integers(0, len(alts), len(m))]
        unit[m] = u
        f = np.array([TO_UNIT[x] for x in u])
        if q == "heat":
            f = np.where(u == "kWh", 1 / 3.6, f)
        factor[m] = f
    O["unit"] = unit
    O["value"] = O.value_c.values * factor
    O = O.drop(columns=["value_c"])
    return O, lab


# ---------------------------------------------------------------------------
def _events(B, FL, S, TL, E, fac):
    """Lifecycle event log: actor + location + time + inputs + outputs (+ links to
    observations via lot identifiers)."""
    route = B.set_index("batch_id").route
    parts = [pd.DataFrame(dict(batch_id=FL.batch_id, event_type="raw_material_sourcing", facility_id=FL.facility_id,
                               time=FL.time.values, inputs="[]", outputs=FL.lot_id.map(lambda x: json.dumps([x]))))]
    fib = FL.groupby("batch_id").lot_id.apply(list)
    prev = {"spinning": None, "fabric": "SPI", "wet": "FAB", "garment": "WET"}
    for st in STAGES:
        s = S[S.stage == st]
        if st == "spinning":
            ins = [json.dumps(fib[b]) for b in s.batch_id]
            et = "spinning"
        else:
            ins = [json.dumps([f"{b}-{prev[st]}"]) for b in s.batch_id]
            et = None
        types = (np.where(route.loc[s.batch_id].values == "knit", "knitting", "weaving") if st == "fabric"
                 else np.full(len(s), {"spinning": "spinning", "wet": "dyeing_finishing", "garment": "cutting_assembly"}[st]))
        parts.append(pd.DataFrame(dict(batch_id=s.batch_id.values, event_type=types, facility_id=s.facility_id.values,
                                       time=s.time.values, inputs=ins, outputs=[json.dumps([x]) for x in s.lot_id])))
    parts.append(pd.DataFrame(dict(batch_id=TL.batch_id, event_type="shipment", facility_id=TL.from_fac,
                                   time=TL.time.values, inputs=TL.leg_id.map(lambda x: json.dumps([x])),
                                   outputs=TL.to_fac.map(lambda x: json.dumps([x])))))
    eol_fac = dict(zip(fac.role, fac.facility_id))
    for et, role in (("collection", "collection"), ("sorting", "sorting"), ("recycling", "recycling")):
        parts.append(pd.DataFrame(dict(batch_id=E.batch_id, event_type=et, facility_id=eol_fac[role],
                                       time=E.time.values, inputs="[]", outputs="[]")))
    ev = pd.concat(parts, ignore_index=True)
    ev.insert(0, "event_id", [f"E{i + 1:08d}" for i in range(len(ev))])
    ev["record_hash"] = [record_hash(a, b, c, d) for a, b, c, d in
                         zip(ev.event_id.values, ev.event_type.values, ev.facility_id.values, ev.time.values)]
    return ev


def _lab_observed(cfg, rng, LT, B):
    tested = set(B.batch_id[rng.random(len(B)) < cfg.lab_test_prob])
    L = LT[LT.batch_id.isin(tested)].copy()
    L["conc_mg_kg"] = L.true_conc * np.exp(rng.normal(0, 0.10, len(L)))
    L["report_id"] = ["LAB" + record_hash(b, s)[:10] for b, s in zip(L.batch_id, L.substance)]
    return L


# ---------------------------------------------------------------------------
def _passport_and_claims(cfg, rng, B, FL, fac_ix, certs, lab_obs, events):
    """Declared passport attributes, claims with evidence, and injected faults."""
    nB = len(B)
    fr = cfg.fault_rate
    faults = []
    dpp = pd.DataFrame(dict(batch_id=B.batch_id, variant_id=B.variant_id, brand=B.brand, gtin=B.gtin, lot=B.lot,
                            units=B.units, unit_mass_kg=B.unit_mass, category=B.category))
    dpp["dpp_uri"] = [digital_link(g, l) for g, l in zip(B.gtin, B.lot)]
    dpp["operator_id"] = [f"urn:example:operator:{b}" for b in B.brand]
    dpp["manufacturer_facility"] = fac_ix.loc[B.fac_garment, "gln"].values
    dpp["country_of_manufacture"] = fac_ix.loc[B.fac_garment, "country"].values
    dpp["care_information"] = "present"
    dpp["declared_composition"] = B.composition.values
    for col in ("gtin", "operator_id", "manufacturer_facility", "country_of_manufacture", "declared_composition",
                "care_information"):
        drop = rng.random(nB) < fr / 2
        dpp[col] = dpp[col].astype(object)
        dpp.loc[drop, col] = None
        faults += [(b, "missing_mandatory", col) for b in B.batch_id[drop]]
    comp_fault = rng.random(nB) < fr
    newcomp = []
    for i in range(nB):
        dc = dpp.declared_composition.iat[i]
        if comp_fault[i] and isinstance(dc, str):
            c = json.loads(dc)
            if len(c) > 1:                           # omit the minor fibre and shift the major share
                minor = min(c, key=c.get)
                c.pop(minor)
                major = max(c, key=c.get)
                c[major] = round(c[major] - 0.08, 4)
                others = [k for k in c if k != major]
                if others:
                    c[others[0]] = round(c[others[0]] + 0.08, 4)
                else:
                    c[major] = round(c[major] + 0.08, 4)
                    c["pes_virgin" if major != "pes_virgin" else "cotton_conv"] = 0.08
                    c[major] = round(1 - 0.08, 4)
            else:
                k0 = next(iter(c))
                c[k0] = 0.9
                c["pes_virgin" if k0 != "pes_virgin" else "cotton_conv"] = 0.1
            faults.append((B.batch_id.iat[i], "composition_mismatch", json.dumps(c)))
            newcomp.append(json.dumps(c, sort_keys=True))
        else:
            newcomp.append(dc)
    dpp["declared_composition"] = newcomp
    comp_rows = pd.DataFrame([(b, m, s) for b, c in zip(dpp.batch_id, dpp.declared_composition) if isinstance(c, str)
                              for m, s in json.loads(c).items()], columns=["batch_id", "material", "share"])

    # -- claims with evidence references ----------------------------------------------
    cert_by_holder = certs.groupby(["holder_facility", "cert_type"]).cert_id.apply(list).to_dict()
    fib_sup = FL.groupby("batch_id").facility_id.apply(list).to_dict()
    lab_by_batch = lab_obs.groupby("batch_id").report_id.apply(list).to_dict()

    def evidence_for(r, kind):
        ids = []
        for f in fib_sup[r.batch_id] + [r.fac_spinning, r.fac_fabric, r.fac_wet, r.fac_garment]:
            ids += cert_by_holder.get((f, kind), [])
        return ids

    claims = []
    over = rng.random(nB) < fr
    org_false = rng.random(nB) < fr / 2
    hz = rng.random(nB) < 0.5
    for i, r in enumerate(B.itertuples()):
        if r.recycled_share_declared > 0 or over[i]:
            val = r.recycled_share_declared + (rng.uniform(0.10, 0.30) if over[i] else 0.0)
            if over[i]:
                faults.append((r.batch_id, "recycled_overclaim", f"{val:.3f}"))
            claims.append((r.batch_id, "recycled_content", round(val, 4), evidence_for(r, "recycled")))
        if r.organic_share >= 0.95 or org_false[i]:
            if org_false[i] and r.organic_share < 0.95:
                faults.append((r.batch_id, "organic_false_claim", ""))
            claims.append((r.batch_id, "organic", 1.0, evidence_for(r, "organic")))
        if hz[i]:
            claims.append((r.batch_id, "restricted_substances_free", 1.0, lab_by_batch.get(r.batch_id, [])))
    C = pd.DataFrame(claims, columns=["batch_id", "claim_type", "value", "evidence"])
    drop_ev = rng.random(len(C)) < fr
    for j in np.where(drop_ev)[0]:
        if C.evidence.iat[j]:
            faults.append((C.batch_id.iat[j], "missing_evidence", C.claim_type.iat[j]))
            C.at[C.index[j], "evidence"] = []
    C["evidence"] = C.evidence.map(json.dumps)
    C.insert(0, "claim_id", [f"CL{i + 1:07d}" for i in range(len(C))])

    # -- traceability gaps: remove one upstream event per affected batch -----------------------
    gap = B.batch_id[rng.random(nB) < fr].tolist()
    up = events[events.batch_id.isin(gap) & events.event_type.isin(["raw_material_sourcing", "spinning"])]
    drop_ids = up.groupby("batch_id").event_id.apply(lambda s: s.iloc[rng.integers(0, len(s))]).tolist()
    events = events[~events.event_id.isin(drop_ids)].reset_index(drop=True)
    faults += [(b, "traceability_gap", "upstream event missing") for b in gap]
    F = pd.DataFrame(faults, columns=["batch_id", "fault_type", "detail"])
    return dpp, comp_rows, C, F, events
