"""E8: multi-granularity aggregation rules and event granularity; E9: fragility of
composite sustainability scores.

E8 uses the ground truth of the E1 seed-1 scenario (regenerated deterministically)
to quantify errors caused by methodologically invalid aggregation:
  (a) intensity indicators aggregated as mean of ratios instead of ratio of sums;
  (b) renewable-electricity share of supplier organisations (mean of facility
      shares vs energy-weighted);
  (c) scarcity-weighted water computed with a global average instead of
      location-specific characterisation factors;
  (d) uncertainty of aggregated footprints with shared (correlated) versus
      independent factor draws;
  (e) loss of batch-level information when a model-level (static) value is used.
E9 ranks product variants by composite scores under alternative normalisation,
weighting and aggregation choices and measures rank instability.
"""
from __future__ import annotations

import itertools
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from common import RESULTS, save_json  # noqa: E402
from tsa import params as P  # noqa: E402
from tsa.synth import generate, ScenarioConfig  # noqa: E402
from tsa.engine import Engine, METHODS, observed_view  # noqa: E402

OUT = os.path.join(RESULTS, "e8_e9")
os.makedirs(OUT, exist_ok=True)
AWARE_GLO = dict(nonagri=17.9, unspecified=39.5)   # AWARE 2.0 GLO annual (F_parameter_sources, section 5)


def e8(sc, v):
    res = {}
    T = sc.truth.set_index("batch_id")
    B = sc.batches.set_index("batch_id")
    # (a) intensity aggregation per brand x category
    g = T.join(B[["brand", "category"]]).groupby(["brand", "category"])
    rs = g.gwp_total.sum() / g.garment_mass.sum()
    mr = g.gwp_per_kg.mean()
    err = (mr / rs - 1)
    res["a_intensity_mean_of_ratios"] = dict(groups=len(err), median_abs_err=float(err.abs().median()),
                                             max_abs_err=float(err.abs().max()), mean_signed=float(err.mean()))
    # (b) renewable share per supplier organisation
    S = sc.stages.merge(sc.facilities[["facility_id", "org_id"]], on="facility_id")
    S = S.assign(ren=S.electricity * S.renew_share)
    go = S.groupby("org_id")
    rs_o = go.ren.sum() / go.electricity.sum()
    fac_share = S.groupby(["org_id", "facility_id"]).renew_share.first().groupby("org_id").mean()
    d = (fac_share - rs_o)
    res["b_org_renewable_share"] = dict(orgs=int(len(d)), median_abs_pp=float(d.abs().median() * 100),
                                        p90_abs_pp=float(d.abs().quantile(0.9) * 100), max_abs_pp=float(d.abs().max() * 100),
                                        orgs_diff_gt_5pp=float((d.abs() > 0.05).mean()))
    # (c) AWARE with location-specific vs global CFs
    FL = sc.fibre_lots
    wf = FL.groupby("batch_id").water_true_m3.sum()
    wm = S.groupby("batch_id").water_consumption.sum()
    avg = (wf * AWARE_GLO["unspecified"] + wm.reindex(wf.index).fillna(0) * AWARE_GLO["nonagri"]) / T.units.reindex(wf.index)
    e = avg / T.aware_per_unit.reindex(wf.index) - 1
    res["c_aware_global_cf"] = dict(median_abs_err=float(e.abs().median()), p90_abs_err=float(e.abs().quantile(0.9)),
                                    share_gt_50pct=float((e.abs() > 0.5).mean()), mean_signed=float(e.mean()))
    # (e) batch vs model-level (variant mean) values
    ev = {}
    for col in ("gwp_per_unit", "energy_per_unit", "water_per_unit"):
        vm = T[col].groupby(B.variant_id).transform("mean")
        dev = (vm / T[col] - 1).abs()
        ev[col] = dict(median=float(dev.median()), p90=float(dev.quantile(0.9)), share_gt_20pct=float((dev > 0.2).mean()))
    res["e_model_level_vs_batch"] = ev
    return res


def e8_uncertainty(v, truth, B):
    out = {}
    T = truth.set_index("batch_id")
    grp = (B.brand + "|" + pd.Series(B.t_garment.values.astype("datetime64[M]").astype(str), index=B.index)).rename("grp")
    for shared in (True, False):
        m = METHODS["M3"]
        from dataclasses import replace
        eng = Engine(v, replace(m, shared_factors=shared, name=m.name + (" shared" if shared else " independent")),
                     n_samples=1000, seed=5, keep_samples=True)
        r = eng.run()
        samp = r.samples["gwp"]                        # batches x samples (kg CO2e per batch)
        idx = r.indicators.batch_id.to_numpy()
        g = grp.reindex(idx).to_numpy()
        rows = []
        for key in np.unique(g):
            mk = g == key
            tot = samp[mk].sum(axis=0)
            lo, med, hi = np.percentile(tot, [2.5, 50, 97.5])
            t = T.loc[idx[mk], "gwp_total"].sum()
            rows.append(dict(group=key, lo=lo, med=med, hi=hi, truth=t, relw=(hi - lo) / med, covered=lo <= t <= hi))
        df = pd.DataFrame(rows)
        # batch-level for reference
        bl = r.indicators.set_index("batch_id")
        bcov = ((T.gwp_per_unit >= bl.gwp_per_unit_lo) & (T.gwp_per_unit <= bl.gwp_per_unit_hi)).mean()
        out["shared" if shared else "independent"] = dict(groups=len(df), coverage=float(df.covered.mean()),
                                                          median_rel_width=float(df.relw.median()),
                                                          batch_coverage=float(bcov))
        df.to_csv(os.path.join(OUT, f"aggregate_uncertainty_{'shared' if shared else 'independent'}.csv"), index=False)
    return out


def e9(ind: pd.DataFrame, B: pd.DataFrame, n_w=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = ind.set_index("batch_id").join(B[["variant_id", "units", "unit_mass"]])
    X["mass"] = X.units * X.unit_mass
    X["gwp_kg"] = X.gwp_per_unit * X.units / X.mass
    X["aware_kg"] = X.aware_per_unit * X.units / X.mass
    X["energy_kg"] = X.energy_per_unit * X.units / X.mass
    X["virgin"] = 1 - X.recycled_content.fillna(0)
    cols = ["gwp_kg", "aware_kg", "energy_kg", "chem_intensity", "virgin", "waste_ratio"]
    V = X.groupby("variant_id").apply(lambda g: pd.Series({c: np.average(g[c].fillna(g[c].mean()), weights=g.mass)
                                                           for c in cols})).dropna()
    norms = {
        "min-max": lambda D: (D - D.min()) / (D.max() - D.min()),
        "z-score": lambda D: (D - D.mean()) / D.std(),
        "rank": lambda D: D.rank(pct=True),
    }
    W = rng.dirichlet(np.ones(len(cols)), n_w)
    eq = np.full(len(cols), 1 / len(cols))
    rows, rank_ranges = [], {}
    for (nn, nf), agg in itertools.product(norms.items(), ("arithmetic", "geometric")):
        Z = nf(V[cols])
        if agg == "geometric":
            Zp = Z - Z.min() + 0.01 if nn == "z-score" else Z + 0.01
            score = lambda w: np.exp(np.log(Zp.to_numpy()) @ w)
        else:
            score = lambda w: Z.to_numpy() @ w
        base = pd.Series(score(eq)).rank().to_numpy()
        ranks = np.array([pd.Series(score(w)).rank().to_numpy() for w in W])
        taus = np.array([kendalltau(base, r_).statistic for r_ in ranks])
        n = len(V)
        order = np.argsort(base)
        top10 = set(order[:10])
        overlap = np.array([len(top10 & set(np.argsort(r_)[:10])) / 10 for r_ in ranks])
        # pairwise reversal frequency
        rev = 0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                pairs += 1
                s = np.sign(base[i] - base[j])
                f = np.mean(np.sign(ranks[:, i] - ranks[:, j]) != s)
                rev += f >= 0.10
        rr = np.percentile(ranks, 95, axis=0) - np.percentile(ranks, 5, axis=0)
        rows.append(dict(normalisation=nn, aggregation=agg, variants=n, kendall_tau_median=float(np.median(taus)),
                         kendall_tau_p05=float(np.percentile(taus, 5)), top10_overlap_median=float(np.median(overlap)),
                         pairs_reversed_ge10pct=float(rev / pairs), rank_range_90_median=float(np.median(rr)),
                         rank_range_90_max=float(rr.max())))
        rank_ranges[f"{nn}|{agg}"] = rr.tolist()
    # cross-method: equal weights, different normalisation/aggregation
    base_rank = {}
    for (nn, nf), agg in itertools.product(norms.items(), ("arithmetic", "geometric")):
        Z = nf(V[cols])
        Zp = (Z - Z.min() + 0.01) if nn == "z-score" else Z + 0.01
        s = np.exp(np.log(Zp.to_numpy()) @ eq) if agg == "geometric" else Z.to_numpy() @ eq
        base_rank[f"{nn}|{agg}"] = pd.Series(s).rank().to_numpy()
    keys = list(base_rank)
    cross = [kendalltau(base_rank[a], base_rank[b]).statistic for a, b in itertools.combinations(keys, 2)]
    return pd.DataFrame(rows), dict(cross_method_tau_min=float(min(cross)), cross_method_tau_median=float(np.median(cross))), V


if __name__ == "__main__":
    t0 = time.perf_counter()
    sc = generate(ScenarioConfig(n_batches=2000, seed=1))
    v = observed_view(sc)
    res = dict(e8=e8(sc, v))
    res["e8"]["d_uncertainty_aggregation"] = e8_uncertainty(v, sc.truth, sc.batches.set_index("batch_id"))
    ind = pd.read_csv(os.path.join(RESULTS, "e1", "MAR_m0.25_a0.03_s1_M3.csv"))
    df, cross, V = e9(ind, sc.batches.set_index("batch_id"))
    df.to_csv(os.path.join(OUT, "composite_fragility.csv"), index=False)
    V.to_csv(os.path.join(OUT, "variant_profiles.csv"))
    res["e9"] = dict(cross, table=df.to_dict(orient="records"))
    res["runtime_s"] = time.perf_counter() - t0
    save_json(res, os.path.join(OUT, "e8_e9.json"))
    print(json.dumps(res, indent=1, default=float))
