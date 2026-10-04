"""Tables, figures and macros for E3, E5, E7, E8 and E9 (called from analyze.py)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INK, INK2 = "#0b0b0b", "#52514e"


def _pct(x):
    return 100 * x


def e3(mac, RES, TAB):
    p = os.path.join(RES, "e3", "provenance_stats.json")
    if not os.path.exists(p):
        return
    s = json.load(open(p))
    mac("ProvNodes", s["nodes"], "{:,}")
    mac("ProvEdges", s["edges"], "{:,}")
    mac("ProvNodesPerBatch", s["nodes_per_batch"], "{:.0f}")
    mac("ProvEdgesPerBatch", s["edges_per_batch"], "{:.0f}")
    mac("ProvBuild", s["build_s"], "{:.1f}")
    mac("ProvLoad", s["load_s"], "{:.1f}")
    mac("ProvCovValue", _pct(s["lineage_coverage_value_weighted"]), "{:.1f}")
    mac("ProvCovContrib", _pct(s["lineage_coverage_contributions"]), "{:.1f}")
    for k, v in s["latency_ms"].items():
        nm = {"explain": "Explain", "rules_used": "Rules", "impact_factor": "Impact"}[k]
        mac(f"Lat{nm}Pfifty", v["p50"], "{:.1f}")
        mac(f"Lat{nm}Pninetyfive", v["p95"], "{:.1f}")
    vs = s.get("value_share_by_tier", {})
    for t, v in vs.items():
        mac(f"ShareTier{t.capitalize()}", _pct(v), "{:.1f}")
    cq = pd.read_csv(os.path.join(RES, "e3", "competency_questions.csv"))
    sym = {"yes": r"\checkmark", "partial": "(partial)", "no": "--"}
    lines = []
    for r in cq.itertuples():
        lat = "" if pd.isna(r.C_latency_ms_p50) else f"{r.C_latency_ms_p50:.1f}"
        lines.append(" & ".join([r.cq, r.question, sym[r.A], sym[r.B], sym[r.C], lat]) + r"\\")
    with open(os.path.join(TAB, "e3_cq.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    for rep in ("A", "B", "C"):
        mac(f"CqYes{rep}", str(int((cq[rep] == "yes").sum())))
        mac(f"CqPartial{rep}", str(int((cq[rep] == "partial").sum())))
    imp = pd.read_csv(os.path.join(RES, "e3", "factor_revision_impact.csv"))
    grid = imp[imp.factor.str.startswith("fac:grid:")]
    mac("ImpGridMin", _pct(grid.affected_share.min()), "{:.0f}")
    mac("ImpGridMax", _pct(grid.affected_share.max()), "{:.0f}")
    mac("ImpAllMax", _pct(imp.affected_share.max()), "{:.0f}")
    mac("ImpAllMin", _pct(imp.affected_share.min()), "{:.1f}")
    mac("ImpExact", "all" if imp.exact_match.all() else "not all")
    mac("ImpMissed", str(int(imp.missed.sum())))
    mac("ImpSuperfluous", str(int(imp.superfluous.sum())))
    mac("ImpQueryMedian", imp.impact_query_ms.median(), "{:.0f}")
    mac("ImpNFactors", str(len(imp)))


def e5(mac, RES, TAB, FIG, COL):
    p = os.path.join(RES, "e5", "t1_t2_intensity.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    models = ["default", "group_median", "ridge", "rf", "mlp", "hgb", "hgb_fe"]
    mname = {"default": "Default", "group_median": "Group median", "ridge": "Ridge", "rf": "RF", "mlp": "MLP",
             "hgb": "GBM", "hgb_fe": "GBM + FE"}
    task = {("spinning", "electricity"): "Spinning, electricity", ("fabric", "electricity"): "Fabric, electricity",
            ("wet", "electricity"): "Wet proc., electricity", ("garment", "electricity"): "CMT, electricity",
            ("wet", "heat"): "Wet proc., heat (T1)", ("wet", "water"): "Wet proc., water (T2)"}
    lines = []
    for proto, pname in (("P1_unseen_facility", "P1 unseen facility"), ("P2_seen_facility", "P2 seen facility")):
        lines.append(r"\multicolumn{10}{l}{\textbf{" + pname + r"}}\\")
        for (st, q), tname in task.items():
            sub = d[(d.stage == st) & (d.quantity == q) & (d.protocol == proto)].set_index("model")
            if sub.empty:
                continue
            best = sub.loc[models, "MAPE"].idxmin()
            cells = [tname, f"{int(sub['n'].iloc[0]):,}"]
            for m in models:
                v = _pct(sub.loc[m, "MAPE"])
                cells.append((r"\textbf{%.1f}" % v) if m == best else f"{v:.1f}")
            cov = sub.loc["hgb_fe", "conformal_cov90"]
            cells.append(f"{_pct(cov):.0f}")
            lines.append(" & ".join(cells) + r"\\")
        lines.append(r"\addlinespace")
    with open(os.path.join(TAB, "e5_intensity.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    for proto, pn in (("P1_unseen_facility", "Pone"), ("P2_seen_facility", "Ptwo")):
        dd = d[d.protocol == proto].groupby("model").MAPE.mean()
        for m in models:
            mac(f"MlMean{pn}{mname[m].replace(' ', '').replace('+', 'plus')}", _pct(dd[m]))
        cov = d[(d.protocol == proto) & (d.model == "hgb_fe")].conformal_cov90
        mac(f"ConfCov{pn}Min", _pct(cov.min()), "{:.0f}")
        mac(f"ConfCov{pn}Max", _pct(cov.max()), "{:.0f}")
        r2 = d[(d.protocol == proto) & (d.model == "hgb_fe")].R2_log
        mac(f"Rsq{pn}Min", r2.min(), "{:.2f}")
        mac(f"Rsq{pn}Max", r2.max(), "{:.2f}")
    t3 = pd.read_csv(os.path.join(RES, "e5", "t3_pcf_screening.csv")).set_index("model")
    for m, nm in (("category_median", "Cat"), ("secondary_calc_M1", "Sec"), ("ridge", "Ridge"), ("rf", "Rf"), ("hgb", "Gbm")):
        mac(f"TThree{nm}Mape", _pct(t3.loc[m, "MAPE"]))
        mac(f"TThree{nm}Rsq", t3.loc[m, "R2_log"], "{:.2f}")
        mac(f"TThree{nm}Rho", t3.loc[m, "spearman"], "{:.2f}")
    t4 = pd.read_csv(os.path.join(RES, "e5", "t4_anomaly.csv")).set_index("method")
    lines = []
    nice = {"rule": "Physical range rule", "robust_z": "Robust $z$-score", "isolation_forest": "Isolation forest",
            "model_residual": "Model-based residual (Eq.~\\ref{eq:screen})", "rule+model_residual": "Range rule + model residual"}
    for m in ["rule", "robust_z", "isolation_forest", "model_residual", "rule+model_residual"]:
        r = t4.loc[m]
        f = lambda x: "--" if pd.isna(x) else f"{x:.2f}"
        lines.append(" & ".join([nice[m], f(r.AUROC), f(r.AUPRC), f(r.precision), f(r.recall), f(r.F1),
                                 f(r.recall_unit_error), f(r.recall_under_report), f(r.recall_spike),
                                 f(r.recall_stale_copy)]) + r"\\")
        key = {"rule": "Rule", "robust_z": "Robz", "isolation_forest": "Iso", "model_residual": "Res",
               "rule+model_residual": "Comb"}[m]
        mac(f"TFour{key}Auroc", r.AUROC, "{:.2f}")
        mac(f"TFour{key}Auprc", r.AUPRC, "{:.2f}")
        mac(f"TFour{key}Fone", r.F1, "{:.2f}")
        mac(f"TFour{key}Prec", r.precision, "{:.2f}")
        mac(f"TFour{key}Rec", r.recall, "{:.2f}")
    mac("TFourPrev", _pct(t4.prevalence.iloc[0]), "{:.1f}")
    mac("TFourN", int(t4.n.iloc[0]), "{:,}")
    with open(os.path.join(TAB, "e5_anomaly.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    imp = pd.read_csv(os.path.join(RES, "e5", "t1_permutation_importance.csv"))
    top = (imp.groupby(["stage", "quantity"]).apply(lambda g: g.sort_values("delta_mae_log", ascending=False)
                                                   .head(3).feature.tolist()))
    for (st, q), feats in top.items():
        mac(f"TopFeat{st.capitalize()}{q.capitalize()}", ", ".join(f.replace("_", " ") for f in feats))
    # figure: MAPE by model and protocol (dot plot)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    order = ["default", "group_median", "ridge", "rf", "mlp", "hgb", "hgb_fe"]
    tasks = list(task.items())
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
    for ax, (proto, ttl) in zip(axes, (("P1_unseen_facility", "(a) P1: unseen facility"),
                                       ("P2_seen_facility", "(b) P2: seen facility"))):
        for i, ((st, q), tname) in enumerate(tasks):
            sub = d[(d.stage == st) & (d.quantity == q) & (d.protocol == proto)].set_index("model")
            x = np.arange(len(order)) + (i - 2.5) * 0.09
            ax.plot(x, _pct(sub.loc[order, "MAPE"].to_numpy()), color=cols[i], lw=1.4, marker="o", ms=4,
                    markeredgecolor="white", markeredgewidth=0.8, label=tname)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([mname[m] for m in order], rotation=35, ha="right")
        ax.set_title(ttl, fontsize=8, loc="left", color=INK)
        ax.set_ylabel("MAPE vs. true intensity (%)" if ax is axes[0] else "")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(FIG, "res_e5_ml.pdf"))
    plt.close(fig)


def e7(mac, RES, TAB, FIG):
    """Repeated measurements (rounds with shuffled size order): medians and ranges."""
    p = os.path.join(RES, "e7", "scalability_runs.jsonl")
    if not os.path.exists(p):
        return
    runs = pd.DataFrame([json.loads(x) for x in open(p, encoding="utf-8") if x.strip()])
    num = runs.drop(columns=["started"]).groupby("n_batches")
    d = num.median().reset_index()
    lo, hi, cnt = num.min(), num.max(), num.size()
    d.to_csv(os.path.join(RES, "e7", "scalability_median.csv"), index=False)
    lines = []
    f = lambda x, fmt="{:.1f}": "--" if pd.isna(x) else fmt.format(x)
    for r in d.itertuples():
        lines.append(" & ".join([f"{r.n_batches:,}", f"{r.events:,.0f}", f"{r.observations:,.0f}", f(r.generate_s),
                                 f(r.M2_total_s), f(r.M3_total_s), f(r.compliance_s),
                                 f(getattr(r, "prov_build_s", np.nan)), f(getattr(r, "prov_nodes", np.nan), "{:,.0f}"),
                                 f(r.event_ingest_per_s, "{:,.0f}"), f(r.obs_ingest_per_s, "{:,.0f}"),
                                 f(getattr(r, "explain_p50_ms", np.nan))]) + r"\\")
    with open(os.path.join(TAB, "e7_scalability.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    big, first = d.iloc[-1], d.iloc[0]
    mac("ScaleRounds", int(cnt.min()), "{}")
    mac("ScaleMaxN", big.n_batches, "{:,.0f}")
    mac("ScaleMaxEvents", big.events, "{:,.0f}")
    mac("ScaleMaxObs", big.observations, "{:,.0f}")
    mac("ScaleMaxMThree", big.M3_total_s / 60, "{:.1f}")
    mac("ScaleMaxMThreeLo", lo.M3_total_s.iloc[-1] / 60, "{:.1f}")
    mac("ScaleMaxMThreeHi", hi.M3_total_s.iloc[-1] / 60, "{:.1f}")
    mac("ScaleMaxMTwo", big.M2_total_s, "{:.0f}")
    mac("ScaleMaxComp", big.compliance_s, "{:.0f}")
    mac("ScaleMinMThree", first.M3_total_s, "{:.0f}")
    mac("ScaleMinN", first.n_batches, "{:,.0f}")
    mac("ScaleMaxTrainShare", 100 * (big.M3_screen_s + big.M3_gapfill_s) / big.M3_total_s, "{:.0f}")
    mac("ScaleMinTrainShare", 100 * (first.M3_screen_s + first.M3_gapfill_s) / first.M3_total_s, "{:.0f}")
    mac("RatioMThreeMTwoMin", first.M3_total_s / first.M2_total_s, "{:.0f}")
    mac("RatioMThreeMTwoMax", big.M3_total_s / big.M2_total_s, "{:.1f}")
    # variability between rounds: largest max/min ratio of total analytics time over sizes
    ratio = pd.concat([hi[c] / lo[c] for c in ("M2_total_s", "M3_total_s", "compliance_s")], axis=1).max(axis=1)
    mac("ScaleVarMax", ratio.max(), "{:.1f}")
    mac("ScaleVarMedian", ratio.median(), "{:.1f}")
    # outlying rounds of the single-threaded steps (generation, provenance build): max / median over rounds
    for col, name in (("prov_build_s", "Prov"),):
        rel = (hi[col] / d.set_index("n_batches")[col]).dropna()
        if len(rel):
            mac(f"{name}OutlierRatio", rel.max(), "{:.0f}")
            mac(f"{name}OutlierN", rel.idxmax(), "{:,}")
    # comparison with the preliminary single-pass run (same code and seeds, run earlier on the same machine)
    sp = os.path.join(RES, "e7", "scalability_single_pass.csv")
    if os.path.exists(sp):
        s0 = pd.read_csv(sp).set_index("n_batches")
        dm = d.set_index("n_batches")
        common = dm.index.intersection(s0.index)
        rr = pd.concat([dm.loc[common, c] / s0.loc[common, c] for c in ("M2_total_s", "M3_total_s", "compliance_s")])
        mac("PrelimRatioMax", rr.max(), "{:.1f}")
        mac("PrelimRatioMin", rr.min(), "{:.1f}")
    # log-log slope of median runtimes; per-passport time at the smallest and largest size
    for k in ("M2", "M3"):
        sl = np.polyfit(np.log(d.n_batches), np.log(d[f"{k}_total_s"]), 1)[0]
        mac(f"Slope{k}", sl, "{:.2f}")
        mac(f"PerPassport{k}Min", 1000 * first[f"{k}_total_s"] / first.n_batches, "{:.1f}")
        mac(f"PerPassport{k}Max", 1000 * big[f"{k}_total_s"] / big.n_batches, "{:.1f}")
    sl = np.polyfit(np.log(d.n_batches), np.log(d.compliance_s), 1)[0]
    mac("SlopeComp", sl, "{:.2f}")
    pv = d.dropna(subset=["prov_build_s"])
    if len(pv):
        mac("SlopeProv", np.polyfit(np.log(pv.n_batches), np.log(pv.prov_build_s), 1)[0], "{:.2f}")
        mac("ProvMaxN", pv.n_batches.iloc[-1], "{:,}")
        mac("ProvMaxBuild", pv.prov_build_s.iloc[-1], "{:.0f}")
        mac("ProvMaxNodes", pv.prov_nodes.iloc[-1], "{:,.0f}")
        mac("ExplainMaxNPfifty", pv.explain_p50_ms.iloc[-1], "{:.1f}")
        mac("ExplainPfiftyMinAll", pv.explain_p50_ms.min(), "{:.1f}")
        mac("ExplainPfiftyMaxAll", pv.explain_p50_ms.max(), "{:.1f}")
    mac("EventIngestRate", d.event_ingest_per_s.median(), "{:,.0f}")
    mac("ObsIngestRate", d.obs_ingest_per_s.median(), "{:,.0f}")
    s1 = d[d.n_batches == 2000]
    if len(s1):
        mac("SOneEvents", s1.events.iat[0], "{:,.0f}")
        mac("SOneObs", s1.observations.iat[0], "{:,.0f}")
    s4 = d[d.n_batches == 5000]
    if len(s4):
        mac("SFourEvents", s4.events.iat[0], "{:,.0f}")
        mac("SFourObs", s4.observations.iat[0], "{:,.0f}")
    pa = os.path.join(RES, "e7", "api_latency.csv")
    if os.path.exists(pa):
        a = pd.read_csv(pa)
        lines = []
        for r in a.itertuples():
            lines.append(" & ".join([r.endpoint, str(r.concurrency), f"{r.p50_ms:.2f}", f"{r.p95_ms:.2f}",
                                     f"{r.p99_ms:.2f}", f"{r.throughput_rps:,.0f}"]) + r"\\")
        with open(os.path.join(TAB, "e7_api.tex"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        one = a[a.concurrency == 1]
        mac("ApiPfiftyMin", one.p50_ms.min(), "{:.1f}")
        mac("ApiPfiftyMax", one.p50_ms.max(), "{:.1f}")
        mac("ApiThrOneMin", one.throughput_rps.min(), "{:,.0f}")
        mac("ApiThrOneMax", one.throughput_rps.max(), "{:,.0f}")
        mac("ApiThrMax", a.throughput_rps.max(), "{:,.0f}")
        mac("ApiThrConcMax", a[a.concurrency > 1].throughput_rps.max(), "{:,.0f}")
        ex = a[a.endpoint == "GET provenance (explain)"]
        if len(ex):
            mac("ApiExplainThrOne", ex[ex.concurrency == 1].throughput_rps.iat[0], "{:,.0f}")
            mac("ApiExplainThrBest", ex.throughput_rps.max(), "{:,.0f}")
            mac("ApiExplainThrBestC", ex.loc[ex.throughput_rps.idxmax(), "concurrency"], "{}")
        mac("ApiPninetyfiveMaxThirtyTwo", a[a.concurrency == 32].p95_ms.max(), "{:.0f}")
        srch = a[(a.endpoint == "GET search") & (a.concurrency == 32)]
        if len(srch):
            mac("ApiSearchPninetyfiveThirtyTwo", srch.p95_ms.iat[0], "{:.0f}")
        oth = a[(a.endpoint != "GET search") & (a.concurrency == 32)]
        mac("ApiOtherPninetyfiveMaxThirtyTwo", oth.p95_ms.max(), "{:.0f}")
    # figure: medians with min-max range over rounds
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 2.8))
    series = [("M3_total_s", "Analytics M3 (incl. ML)", "#2a78d6"), ("M2_total_s", "Analytics M2", "#eb6834"),
              ("compliance_s", "Compliance rules", "#1baf7a"), ("generate_s", "Scenario generation", "#8a8984"),
              ("prov_build_s", "Provenance graph build", "#4a3aa7")]
    for col, lab, c in series:
        ok = d[col].notna().to_numpy()
        x = d.n_batches.to_numpy()[ok]
        ax.fill_between(x, lo[col].to_numpy()[ok], hi[col].to_numpy()[ok], color=c, alpha=0.15, lw=0)
        ax.plot(x, d[col].to_numpy()[ok], color=c, lw=2, marker="o", ms=4, markeredgecolor="white", label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of batch passports")
    ax.set_ylabel("wall-clock time (s)")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "res_e7_scalability.pdf"))
    plt.close(fig)


def e8e9(mac, RES, TAB, FIG):
    p = os.path.join(RES, "e8_e9", "e8_e9.json")
    if not os.path.exists(p):
        return
    s = json.load(open(p))
    e8 = s["e8"]
    a = e8["a_intensity_mean_of_ratios"]
    mac("AggIntMedian", _pct(a["median_abs_err"]))
    mac("AggIntMax", _pct(a["max_abs_err"]))
    b = e8["b_org_renewable_share"]
    mac("AggRenMedian", b["median_abs_pp"])
    mac("AggRenMax", b["max_abs_pp"])
    mac("AggRenShareGt", _pct(b["orgs_diff_gt_5pp"]), "{:.0f}")
    mac("AggRenOrgs", str(b["orgs"]))
    c = e8["c_aware_global_cf"]
    mac("AggAwaMedian", _pct(c["median_abs_err"]))
    mac("AggAwaPninety", _pct(c["p90_abs_err"]))
    mac("AggAwaGtFifty", _pct(c["share_gt_50pct"]), "{:.0f}")
    d = e8["d_uncertainty_aggregation"]
    mac("AggUncSharedCov", _pct(d["shared"]["coverage"]), "{:.0f}")
    mac("AggUncIndepCov", _pct(d["independent"]["coverage"]), "{:.0f}")
    mac("AggUncSharedWid", _pct(d["shared"]["median_rel_width"]))
    mac("AggUncIndepWid", _pct(d["independent"]["median_rel_width"]))
    mac("AggUncGroups", str(d["shared"]["groups"]))
    ev = e8["e_model_level_vs_batch"]
    for k, nm in (("gwp_per_unit", "Gwp"), ("energy_per_unit", "Ene"), ("water_per_unit", "Wat")):
        mac(f"ModelLevel{nm}Median", _pct(ev[k]["median"]))
        mac(f"ModelLevel{nm}Pninety", _pct(ev[k]["p90"]))
        mac(f"ModelLevel{nm}Gt", _pct(ev[k]["share_gt_20pct"]), "{:.0f}")
    e9 = s["e9"]
    t = pd.DataFrame(e9["table"])
    lines = []
    for r in t.itertuples():
        lines.append(" & ".join([r.normalisation, r.aggregation, f"{r.kendall_tau_median:.2f}", f"{r.kendall_tau_p05:.2f}",
                                 f"{r.top10_overlap_median:.1f}", f"{_pct(r.pairs_reversed_ge10pct):.0f}",
                                 f"{r.rank_range_90_median:.0f}", f"{r.rank_range_90_max:.0f}"]) + r"\\")
    with open(os.path.join(TAB, "e9_composite.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    mac("CompVariants", str(int(t.variants.iloc[0])))
    mac("CompTauMin", t.kendall_tau_median.min(), "{:.2f}")
    mac("CompTauMax", t.kendall_tau_median.max(), "{:.2f}")
    mac("CompRevMin", _pct(t.pairs_reversed_ge10pct.min()), "{:.0f}")
    mac("CompRevMax", _pct(t.pairs_reversed_ge10pct.max()), "{:.0f}")
    mac("CompRangeMax", t.rank_range_90_max.max(), "{:.0f}")
    mac("CompRangeMedMax", t.rank_range_90_median.max(), "{:.0f}")
    mac("CompCrossTauMin", e9["cross_method_tau_min"], "{:.2f}")


def run(MACROS, mac, RES, TAB, FIG, COL, LABEL):
    e3(mac, RES, TAB)
    e5(mac, RES, TAB, FIG, COL)
    e7(mac, RES, TAB, FIG)
    e8e9(mac, RES, TAB, FIG)
