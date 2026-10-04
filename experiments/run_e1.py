"""E1/E2/E4: indicator accuracy, uncertainty calibration, data-quality informativeness
and compliance-rule performance on one synthetic scenario.

usage: python run_e1.py --seed 1 [--n 2000] [--missing 0.25] [--mech MAR] [--methods M0,M1,...]
                        [--mode full|summary] [--out results/e1]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import RESULTS, EVAL_INDICATORS, batch_flags, accuracy_metrics, compliance_truth, prf, save_json  # noqa: E402
from tsa.synth import generate, ScenarioConfig  # noqa: E402
from tsa.engine import Engine, METHODS, observed_view, canonicalise  # noqa: E402
from tsa.compliance import ComplianceEngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--missing", type=float, default=0.25)
    ap.add_argument("--mech", default="MAR")
    ap.add_argument("--anomaly", type=float, default=0.03)
    ap.add_argument("--methods", default="M0,M1,M2,M2c,M3,M3-noML,M3-noscreen,M3-nocal")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--out", default=os.path.join(RESULTS, "e1"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.mech}_m{a.missing:.2f}_a{a.anomaly:.2f}_s{a.seed}"
    t0 = time.perf_counter()
    cfg = ScenarioConfig(n_batches=a.n, seed=a.seed, missing_rate=a.missing, missing_mechanism=a.mech,
                         anomaly_rate=a.anomaly)
    sc = generate(cfg)
    t_gen = time.perf_counter() - t0
    v = observed_view(sc)
    truth = sc.truth
    flags = batch_flags(sc)
    miss_share = float((sc.observations[sc.observations.quantity.isin(["electricity", "heat", "water"]) &
                                        (sc.observations.source_system != "audit")].data_type == "missing").mean())
    summary = dict(tag=tag, seed=a.seed, n=a.n, missing_rate=a.missing, mechanism=a.mech, anomaly_rate=a.anomaly,
                   realised_missing_share=miss_share, t_generate=t_gen, methods={})
    for k in a.methods.split(","):
        r = Engine(v, METHODS[k], n_samples=a.samples, seed=a.seed).run()
        met = {c: accuracy_metrics(r.indicators, truth, c) for c in EVAL_INDICATORS}
        # recycled content: evidence-based vs declared
        I = r.indicators.set_index("batch_id")
        tr = truth.set_index("batch_id").recycled_content_true
        for col in ("recycled_content", "recycled_content_declared"):
            e = (I[col] - tr).dropna()
            met[col] = dict(MAE_pp=float(e.abs().mean() * 100), bias_pp=float(e.mean() * 100),
                            computable=float(I[col].notna().mean()))
        met["timings"] = r.timings
        met["name"] = METHODS[k].name
        # DQ informativeness (E2)
        if "gwp_per_unit_lo" in I:
            ape = (I.gwp_per_unit / truth.set_index("batch_id").gwp_per_unit - 1).abs()
            relw = (I.gwp_per_unit_hi - I.gwp_per_unit_lo) / I.gwp_per_unit_med
            from scipy.stats import spearmanr
            met["dq"] = dict(rho_dq_ape=float(spearmanr(I.dq_score_gwp, ape).statistic),
                             rho_primary_ape=float(spearmanr(I.primary_share_gwp, ape).statistic),
                             rho_width_ape=float(spearmanr(relw, ape).statistic))
            cov = ((truth.set_index("batch_id").gwp_per_unit >= I.gwp_per_unit_lo) &
                   (truth.set_index("batch_id").gwp_per_unit <= I.gwp_per_unit_hi))
            f = flags.reindex(I.index)
            clean = ~f.fraud.astype(bool) & ~f.unevidenced_re.astype(bool)
            met["coverage_diagnostics"] = dict(all=float(cov.mean()), clean=float(cov[clean].mean()),
                                               fraud=float(cov[f.fraud.astype(bool)].mean()) if f.fraud.any() else None,
                                               unevidenced_re=float(cov[f.unevidenced_re.astype(bool)].mean()),
                                               n_clean=int(clean.sum()))
        summary["methods"][k] = met
        if a.mode == "full":
            keep = ["batch_id"] + [c for c in I.reset_index().columns if c.startswith(tuple(EVAL_INDICATORS)) or
                                   c.startswith(("dq_", "primary_", "gwp_share_", "gwp_")) or
                                   c in ("recycled_content", "recycled_content_declared", "mci", "chem_intensity",
                                         "waste_ratio", "waste_recovery_rate", "renewable_elec_share",
                                         "renewable_elec_share_location", "certified_material_ratio",
                                         "mono_material_share")]
            r.indicators[sorted(set(keep), key=keep.index)].to_csv(os.path.join(a.out, f"{tag}_{k}.csv"), index=False)
        print(k, {c: round(met[c]["MAPE"], 3) for c in EVAL_INDICATORS}, round(r.timings["total"], 1), "s", flush=True)
    if a.mode == "full":
        truth.merge(flags.reset_index(), on="batch_id").to_csv(os.path.join(a.out, f"{tag}_truth.csv"), index=False)
        # compliance (E4)
        t = time.perf_counter()
        ce = ComplianceEngine(v)
        R, timing = ce.run()
        t_c = time.perf_counter() - t
        lab = compliance_truth(sc)
        rules = {}
        for rid in lab.columns:
            flagged = lab.index.isin(R[R.rule_id == rid].batch_id)
            viol = lab.index.isin(R[(R.rule_id == rid) & (R.outcome == "violation")].batch_id)
            rules[rid] = dict(flagged=prf(flagged, lab[rid].to_numpy()), violation_only=prf(viol, lab[rid].to_numpy()),
                              unverifiable=int(((R.rule_id == rid) & (R.outcome == "unverifiable")).sum()),
                              time_s=timing[rid])
        summary["compliance"] = dict(rules=rules, total_time_s=t_c, batches_per_s=a.n / t_c)
        R.to_csv(os.path.join(a.out, f"{tag}_compliance.csv"), index=False)
    summary["t_total"] = time.perf_counter() - t0
    save_json(summary, os.path.join(a.out, f"{tag}_summary.json"))
    print("done", tag, round(summary["t_total"], 1), "s")


if __name__ == "__main__":
    main()
