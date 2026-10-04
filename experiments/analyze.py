"""Aggregate experiment outputs into manuscript tables (LaTeX), figures (PDF) and
number macros, so that no reported value is transcribed by hand.

Outputs: manuscript/tables/*.tex, manuscript/figures/res_*.pdf,
         manuscript/sections/results_macros.tex
"""
from __future__ import annotations

import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
MS = os.path.join(HERE, "..", "..", "manuscript")
TAB = os.path.join(MS, "tables")
FIG = os.path.join(MS, "figures")
os.makedirs(TAB, exist_ok=True)

# validated categorical palette (fixed order; colour follows the method, never its rank)
COL = {"M3": "#2a78d6", "M2": "#eb6834", "M2c": "#1baf7a", "M1": "#eda100", "M3-noML": "#e87ba4",
       "M3-noscreen": "#008300", "M3-nocal": "#4a3aa7", "M0": "#8a8984"}
LABEL = {"M0": "M0 naive", "M1": "M1 secondary only", "M2": "M2 primary+default (ecoinvent pedigree)",
         "M2c": "M2c primary+default (Ciroth pedigree)", "M3": "M3 proposed", "M3-noML": "M3 w/o ML imputation",
         "M3-noscreen": "M3 w/o model screening", "M3-nocal": "M3 w/o empirical calibration"}
ORDER = ["M0", "M1", "M2", "M2c", "M3", "M3-noML", "M3-noscreen", "M3-nocal"]
SHORT = {"M2": "M2 (ecoinvent pedigree)", "M2c": "M2c (Ciroth pedigree)", "M3-noML": "M3 w/o ML imputation", "M3": "M3 proposed"}
IND = {"gwp_per_unit": "CLI-01 carbon footprint", "energy_per_unit": "ENE-01 manufacturing energy",
       "water_per_unit": "WAT-02 blue water consumption", "aware_per_unit": "WAT-03 scarcity-weighted water"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42})
MACROS: dict[str, str] = {}


def mac(name, value, fmt="{:.1f}"):
    MACROS[name] = fmt.format(value) if not isinstance(value, str) else value


def pct(x):
    return 100 * x


# ---------------------------------------------------------------------------
def e1():
    rows, cov, dqr = [], [], []
    for f in sorted(glob.glob(os.path.join(RES, "e1", "*_summary.json"))):
        s = json.load(open(f))
        for k, m in s["methods"].items():
            for ind in IND:
                rows.append(dict(seed=s["seed"], method=k, ind=ind, **{x: m[ind].get(x, np.nan) for x in
                                 ("MAPE", "MdAPE", "bias", "p90APE", "spearman", "coverage95", "rel_width95",
                                  "rel_interval_score")}))
            rows.append(dict(seed=s["seed"], method=k, ind="recycled", MAPE=m["recycled_content"]["MAE_pp"],
                             MdAPE=m["recycled_content_declared"]["MAE_pp"], bias=m["recycled_content"]["bias_pp"],
                             p90APE=m["recycled_content_declared"]["bias_pp"],
                             spearman=m["recycled_content"]["computable"]))
            if "coverage_diagnostics" in m:
                cov.append(dict(seed=s["seed"], method=k, **{a: b for a, b in m["coverage_diagnostics"].items()
                                                              if a != "n_clean"}))
                dqr.append(dict(seed=s["seed"], method=k, **m["dq"]))
            if k == "M3":
                mac("SCOneMissing", pct(s["realised_missing_share"]), "{:.0f}")
        mac("NSeeds", str(len(glob.glob(os.path.join(RES, "e1", "*_summary.json")))))
    df = pd.DataFrame(rows)
    g = df.groupby(["ind", "method"]).agg(["mean", "std"])
    # --- table ------------------------------------------------------------------
    lines = []
    for ind, name in IND.items():
        lines.append(r"\multicolumn{7}{l}{\textbf{" + name + r"}}\\")
        for k in ORDER:
            r = g.loc[(ind, k)]
            def ms(c, scale=100, d=1):
                v, sd = r[(c, "mean")], r[(c, "std")]
                if np.isnan(v):
                    return "--"
                return f"{v*scale:.{d}f} $\\pm$ {sd*scale:.{d}f}"
            cells = [LABEL[k], ms("MAPE"), ms("MdAPE"), ms("bias"), ms("coverage95"), ms("rel_width95"),
                     ms("rel_interval_score")]
            if k == "M3":
                cells = [r"\textbf{" + c + "}" if i == 0 else c for i, c in enumerate(cells)]
            lines.append(" & ".join(cells) + r"\\")
        lines.append(r"\addlinespace")
    with open(os.path.join(TAB, "e1_accuracy.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    for ind in IND:
        short = {"gwp_per_unit": "Gwp", "energy_per_unit": "Ene", "water_per_unit": "Wat", "aware_per_unit": "Awa"}[ind]
        for k in ORDER:
            kk = k.replace("-", "").replace("noML", "NoML").replace("noscreen", "NoScr").replace("nocal", "NoCal")
            for met, nm in (("MAPE", "Mape"), ("coverage95", "Cov"), ("rel_width95", "Wid"), ("rel_interval_score", "Is"),
                            ("bias", "Bias"), ("MdAPE", "Mdape")):
                v = g.loc[(ind, k), (met, "mean")]
                if not np.isnan(v):
                    mac(f"E{short}{met and nm}{kk}", pct(v) if met != "rel_interval_score" else v,
                        "{:.1f}" if met != "rel_interval_score" else "{:.2f}")
    rc = df[df.ind == "recycled"].groupby("method")[["MAPE", "MdAPE", "bias", "p90APE", "spearman"]].mean().loc["M3"]
    mac("RecEvidMAE", rc["MAPE"], "{:.2f}")
    mac("RecDeclMAE", rc["MdAPE"], "{:.2f}")
    mac("RecEvidBias", rc["bias"], "{:+.2f}")
    mac("RecDeclBias", rc["p90APE"], "{:+.2f}")
    mac("RecComputable", pct(rc["spearman"]), "{:.0f}")
    df = df[df.ind != "recycled"]
    # relative improvements
    for ind, short in (("gwp_per_unit", "Gwp"), ("energy_per_unit", "Ene"), ("water_per_unit", "Wat")):
        a, b = g.loc[(ind, "M2"), ("MAPE", "mean")], g.loc[(ind, "M3"), ("MAPE", "mean")]
        mac(f"E{short}RelImp", pct(1 - b / a), "{:.0f}")
    cv = pd.DataFrame(cov).groupby("method").mean(numeric_only=True)
    for k in ("M3", "M2c", "M3-nocal"):
        kk = k.replace("-", "").replace("nocal", "NoCal")
        mac(f"CovClean{kk}", pct(cv.loc[k, "clean"]))
        mac(f"CovUnev{kk}", pct(cv.loc[k, "unevidenced_re"]))
        mac(f"CovFraud{kk}", pct(cv.loc[k, "fraud"]))
    dq = pd.DataFrame(dqr).groupby("method").agg(["mean", "std"])
    for k in ("M3", "M2c"):
        kk = k
        for c, nm in (("rho_dq_ape", "RhoDq"), ("rho_primary_ape", "RhoPrim"), ("rho_width_ape", "RhoWidth")):
            mac(f"{nm}{kk}", dq.loc[k, (c, "mean")], "{:.2f}")
    # --- figure: accuracy bars and calibration-sharpness ------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(width_ratios=[1.1, 0.85, 1.25]))
    meths = ["M1", "M2", "M3-noML", "M3-noscreen", "M3-nocal", "M3"]
    for ax, ind, ttl in ((axes[0], "gwp_per_unit", "(a) CLI-01: MAPE"),
                         (axes[1], "energy_per_unit", "(b) ENE-01: MAPE")):
        vals = [pct(g.loc[(ind, k), ("MAPE", "mean")]) for k in meths]
        sds = [pct(g.loc[(ind, k), ("MAPE", "std")]) for k in meths]
        y = np.arange(len(meths))
        ax.barh(y, vals, xerr=sds, color=[COL[k] for k in meths], height=0.62, edgecolor="white", linewidth=1.0,
                error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        for yi, v, sd in zip(y, vals, sds):
            ax.text(v + sd + max(vals) * 0.04, yi, f"{v:.1f}", va="center", fontsize=7, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[k].replace(" (ecoinvent pedigree)", "").replace(" (Ciroth pedigree)", "") for k in meths]
                           if ax is axes[0] else [""] * len(meths))
        ax.set_xlabel("MAPE (%)")
        ax.set_title(ttl, fontsize=8, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.38)
    ax = axes[2]
    for k in ("M1", "M2", "M2c", "M3-noML", "M3-nocal", "M3"):
        w = pct(g.loc[("gwp_per_unit", k), ("rel_width95", "mean")])
        c = pct(g.loc[("gwp_per_unit", k), ("coverage95", "mean")])
        ax.scatter(w, c, s=38, color=COL[k], edgecolor="white", linewidth=1.5, zorder=3)
        dx, dy, ha = {"M2c": (4, -2.5, "left"), "M3-noML": (4, -4.5, "left"), "M3-nocal": (-4, 1.0, "right"),
                      "M3": (4, -3.2, "left"), "M1": (4, 0.8, "left"), "M2": (4, 0.8, "left")}[k]
        ax.text(w + dx, c + dy, k, fontsize=7, color=INK, ha=ha)
    ax.axhline(95, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(3, 96.3, "nominal 95 %", fontsize=6.5, color=INK2, ha="left")
    ax.set_xlabel("95 % interval width (% of median)")
    ax.set_ylabel("coverage (%)")
    ax.set_title("(c) CLI-01: coverage vs. width", fontsize=8, loc="left", color=INK)
    ax.set_xlim(0, 110)
    ax.set_ylim(25, 100)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(FIG, "res_e1_accuracy.pdf"))
    plt.close(fig)
    return g


def e1b():
    rows = []
    for f in glob.glob(os.path.join(RES, "e1b", "*_summary.json")):
        s = json.load(open(f))
        for k, m in s["methods"].items():
            rows.append(dict(mech=s["mechanism"], r=s["missing_rate"], seed=s["seed"], method=k,
                             realised=s["realised_missing_share"], MAPE=m["gwp_per_unit"]["MAPE"],
                             E_MAPE=m["energy_per_unit"]["MAPE"], cov=m["gwp_per_unit"]["coverage95"],
                             IS=m["gwp_per_unit"]["rel_interval_score"]))
    if not rows:
        return
    df = pd.DataFrame(rows).groupby(["mech", "r", "method"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), sharex=True, sharey="row")
    meths = ["M2", "M2c", "M3-noML", "M3"]
    for j, mech in enumerate(("MCAR", "MNAR")):
        d = df[df.mech == mech]
        for i, (col, lab) in enumerate((("MAPE", "CLI-01 MAPE (%)"), ("cov", "CLI-01 coverage of 95 % interval (%)"))):
            ax = axes[i, j]
            for k in meths:
                dd = d[d.method == k].sort_values("r")
                if k == "M2c" and col == "MAPE":
                    continue   # identical point estimates to M2
                ax.plot(pct(dd.realised), pct(dd[col]), color=COL[k], lw=2, marker="o", ms=4,
                        markeredgecolor="white", markeredgewidth=1.0, label=SHORT[k])
            if col == "cov":
                ax.axhline(95, color=INK2, lw=0.8, ls=(0, (3, 2)))
            ax.set_ylabel(lab if j == 0 else "")
            if i == 0:
                ax.set_title(f"{'(a)' if j == 0 else '(b)'} {mech} non-disclosure", fontsize=8, loc="left", color=INK)
            if i == 1:
                ax.set_xlabel("realised share of missing activity records (%)")
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(FIG, "res_e1b_sensitivity.pdf"))
    plt.close(fig)
    for mech in ("MCAR", "MNAR"):
        d = df[(df.mech == mech)]
        hi = d.r.max()
        for k in ("M2", "M3"):
            v = d[(d.r == hi) & (d.method == k)]
            mac(f"Sens{mech}{k.replace('-', '')}MapeHi", pct(v.MAPE.iat[0]))
            mac(f"Sens{mech}{k.replace('-', '')}CovHi", pct(v["cov"].iat[0]))
            v0 = d[(d.r == 0.0) & (d.method == k)]
            mac(f"Sens{mech}{k.replace('-', '')}MapeZero", pct(v0.MAPE.iat[0]))
        mac(f"Sens{mech}RealHi", pct(d[d.r == hi].realised.mean()), "{:.0f}")
        mac(f"Sens{mech}RealZero", pct(d[d.r == 0.0].realised.mean()), "{:.0f}")
    df.to_csv(os.path.join(RES, "e1b_summary.csv"), index=False)


def e4():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "e1", "*_summary.json"))):
        s = json.load(open(f))
        for rid, r in s["compliance"]["rules"].items():
            for mode in ("flagged", "violation_only"):
                d = r[mode]
                rows.append(dict(seed=s["seed"], rule=rid, mode=mode, precision=d["precision"], recall=d["recall"],
                                 f1=d["f1"], prevalence=d["prevalence"], unverifiable=r["unverifiable"]))
        mac("CompBatchesPerSec", s["compliance"]["batches_per_s"], "{:,.0f}")
    df = pd.DataFrame(rows).groupby(["rule", "mode"]).mean(numeric_only=True)
    from tsa.compliance import RULES
    desc = {r.rule_id: r.description for r in RULES}
    lines = []
    for rid in sorted(desc):
        a, b = df.loc[(rid, "violation_only")], df.loc[(rid, "flagged")]
        f = lambda x: "--" if np.isnan(x) else f"{x:.2f}"
        lines.append(" & ".join([rid, desc[rid], f"{a.prevalence:.0f}", f(a.precision), f(a.recall), f(a.f1),
                                 f(b.precision), f(b.recall), f(b.f1), f"{b.unverifiable:.0f}"]) + r"\\")
    with open(os.path.join(TAB, "e4_compliance.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    for rid in ("R03", "R08", "R09", "R10"):
        mac(f"Rec{rid}", df.loc[(rid, "violation_only"), "recall"], "{:.2f}")
        mac(f"Prec{rid}", df.loc[(rid, "violation_only"), "precision"], "{:.2f}")


def e6():
    p = os.path.join(RES, "e6", "interop.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    w, s = d["with_extensions"], d["standard_only"]
    mac("EpcisDocs", str(w["documents"]))
    mac("EpcisEvents", w["events"], "{:,}")
    mac("EpcisObj", w["event_types"]["ObjectEvent"], "{:,}")
    mac("EpcisTrans", w["event_types"]["TransformationEvent"], "{:,}")
    mac("EpcisErrExt", str(w["schema_errors"]))
    mac("EpcisErrStd", str(s["schema_errors"]))
    mac("EpcisReportsExt", w["sensor_reports"], "{:,}")
    mac("EpcisReportsStd", s["sensor_reports"], "{:,}")
    mac("EpcisMisclass", pct(s["misclassified_share"]))
    mac("EpcisMbExt", w["bytes"] / 1e6)
    mac("EpcisMbStd", s["bytes"] / 1e6)
    mac("EpcisExportRate", w["events"] / w["export_s"], "{:,.0f}")
    mac("EpcisValidateRate", w["events"] / w["validate_s"], "{:,.0f}")
    pp = d["passport_jsonld"]
    mac("PassportKB", pp["bytes_mean"] / 1000)
    mac("PassportGenMs", pp["generation_s"] / pp["n"] * 1000)
    ing = d["ingestion"]
    mac("IngMapped", ing["records_mapped"], "{:,}")
    mac("IngQuar", ing["records_quarantined"], "{:,}")
    mac("IngRate", ing["records_per_s"], "{:,.0f}")


def write_macros():
    with open(os.path.join(MS, "sections", "results_macros.tex"), "w", encoding="utf-8") as f:
        f.write("% Auto-generated by code/experiments/analyze.py -- do not edit by hand\n")
        words = dict(zip("0123456789", ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")))
        for k, v in sorted(MACROS.items()):
            name = "".join(words.get(ch, ch) for ch in k if ch.isalnum())
            f.write(f"\\newcommand{{\\{name}}}{{{v}}}\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(HERE, ".."))
    e1()
    e1b()
    e4()
    e6()
    for mod in ("analyze_more",):
        try:
            __import__(mod).run(MACROS, mac, RES, TAB, FIG, COL, LABEL)
        except ModuleNotFoundError:
            pass
    write_macros()
    print(len(MACROS), "macros")
