"""E5: machine-learning tasks.

T1  activity-intensity estimation (electricity by stage, process heat) and
T2  wet-processing water intensity: models trained on OBSERVED measured data,
    evaluated against ground-truth intensities under two protocols:
      P1 unseen facility (GroupKFold by facility)  - supplier without history
      P2 seen facility   (KFold over lots)         - lot-level gaps, facility history available
T3  screening estimate of product carbon footprint (kg CO2e/kg) from static
    passport attributes (GroupKFold by product variant).
T4  anomaly detection in reported activity data (threshold tuned on a separate
    validation scenario, evaluated on the test scenario).
"""
from __future__ import annotations

import json
import os
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from common import RESULTS, save_json  # noqa: E402
from tsa import params as P  # noqa: E402
from tsa.synth import generate, ScenarioConfig  # noqa: E402
from tsa.engine import Engine, METHODS, observed_view, canonicalise, ACT  # noqa: E402
from tsa.ml import (lot_features, encode, make_model, IntensityImputer, screening_scores,  # noqa: E402
                    plausibility_violation, isolation_scores, CAT_COLS, NUM_COLS)
from tsa.factors import build_factor_db  # noqa: E402

OUT = os.path.join(RESULTS, "e5")
os.makedirs(OUT, exist_ok=True)
DEFAULT_INT = {("spinning", "electricity"): P.DEFAULTS["spin_kwh_per_kg"],
               ("fabric", "electricity"): 0.6 * P.DEFAULTS["weave_kwh_per_kg"] + 0.4 * P.DEFAULTS["knit_kwh_per_kg"],
               ("wet", "electricity"): P.DEFAULTS["wet_elec_kwh_per_kg"],
               ("garment", "electricity"): P.DEFAULTS["garment_kwh_per_kg"],
               ("wet", "heat"): P.DEFAULTS["wet_heat_mj_per_kg"], ("wet", "water"): P.DEFAULTS["wet_water_l_per_kg"] / 1000}


def lot_table(sc, v):
    O = canonicalise(v.observations)
    mass = O[O.quantity == "mass_in"].dropna(subset=["value_c"]).drop_duplicates("lot_id").set_index("lot_id").value_c
    S = sc.stages.set_index("lot_id")
    return O, mass, S


def intensity_dataset(sc, v, st, q):
    O, mass, S = lot_table(sc, v)
    m = O[(O.stage == st) & (O.quantity == q) & (O.data_type == "measured") & O.value_c.notna()]
    m = m.sort_values("source_system").drop_duplicates("lot_id")          # one measured value per lot
    m = m[m.lot_id.isin(mass.index)]
    lots = m.assign(mass_in=mass.loc[m.lot_id].to_numpy())
    X = lot_features(lots.assign(stage=st), v.batch_master, v.facility_master)
    inten = lots.value_c.to_numpy() / lots.mass_in.to_numpy()
    y_obs = np.log(inten)
    true_int = S.loc[lots.lot_id, q].to_numpy() / S.loc[lots.lot_id, "mass_in"].to_numpy()
    # unsupervised screening, identical to the pipeline (Sect. 7.2): range rule or robust residual z > 4
    flag = plausibility_violation(np.full(len(lots), st), np.full(len(lots), q), inten)
    flag |= screening_scores(X, y_obs, lots.facility_id.to_numpy()) > 4.0
    return (X.reset_index(drop=True), y_obs, np.log(true_int), lots.facility_id.to_numpy(), lots.batch_id.to_numpy(),
            flag)


def metrics(y_true_log, y_pred_log):
    t, p = np.exp(y_true_log), np.exp(y_pred_log)
    return dict(MAE=float(np.mean(np.abs(p - t))), RMSE=float(np.sqrt(np.mean((p - t) ** 2))),
                MAPE=float(np.mean(np.abs(p / t - 1))), R2_log=float(r2_score(y_true_log, y_pred_log)))


def t1_t2(sc, v, seed=0):
    global screened
    screened = []
    res, imp_rows = [], []
    for st, q in ACT:
        X, y, yt, fac, _, flag = intensity_dataset(sc, v, st, q)
        n = len(y)
        screened.append(dict(stage=st, quantity=q, n=n, rejected=int(flag.sum())))
        for proto in ("P1_unseen_facility", "P2_seen_facility"):
            split = GroupKFold(5).split(X, y, fac) if proto.startswith("P1") else \
                KFold(5, shuffle=True, random_state=seed).split(X)
            preds = {k: np.zeros(n) for k in ("default", "group_median", "ridge", "rf", "hgb", "mlp", "hgb_fe")}
            cover, width = [], []
            for fold, (tr, te) in enumerate(split):
                tr = tr[~flag[tr]]                      # train on screened data; test on all lots vs. truth
                A_tr, A_te = encode(X.iloc[tr]), encode(X.iloc[te])
                preds["default"][te] = np.log(DEFAULT_INT[(st, q)])
                grp = X.category.astype(str) + "|" + X.depth.astype(str)
                gm = pd.Series(y[tr]).groupby(grp.iloc[tr].to_numpy()).median()
                preds["group_median"][te] = grp.iloc[te].map(gm).fillna(np.median(y[tr])).to_numpy()
                for k in ("ridge", "rf", "hgb", "mlp"):
                    preds[k][te] = make_model(k, seed).fit(A_tr, y[tr]).predict(A_te)
                im = IntensityImputer(seed=seed).fit(X.iloc[tr].reset_index(drop=True), y[tr], fac[tr])
                mu, seen = im.predict(X.iloc[te].reset_index(drop=True), fac[te])
                preds["hgb_fe"][te] = mu
                # conformal 90 % intervals from the residual pools (seen/unseen), evaluated against truth
                lo_s, hi_s = np.quantile(im.resid_, [0.05, 0.95])
                lo_u, hi_u = np.quantile(im.resid_unseen_, [0.05, 0.95])
                lo = mu + np.where(seen, lo_s, lo_u)
                hi = mu + np.where(seen, hi_s, hi_u)
                cover.append(((yt[te] >= lo) & (yt[te] <= hi)).mean())
                width.append(np.median(np.exp(hi - lo)))
                if proto.startswith("P1") and st == "wet" and q in ("heat", "water") or (proto.startswith("P1") and
                                                                                       (st, q) == ("spinning", "electricity")):
                    m = make_model("hgb", seed).fit(A_tr, y[tr])
                    base = np.mean(np.abs(m.predict(A_te) - yt[te]))
                    rng = np.random.default_rng(seed + fold)
                    for col in CAT_COLS + NUM_COLS:
                        Xp = X.iloc[te].copy()
                        Xp[col] = rng.permutation(Xp[col].to_numpy())
                        inc = np.mean(np.abs(m.predict(encode(Xp)) - yt[te])) - base
                        imp_rows.append(dict(stage=st, quantity=q, fold=fold, feature=col, delta_mae_log=inc))
            for k, p_ in preds.items():
                row = dict(stage=st, quantity=q, protocol=proto, model=k, n=n, **metrics(yt, p_))
                row["vs_observed_MAPE"] = float(np.mean(np.abs(np.exp(p_ - y) - 1)))
                if k == "hgb_fe":
                    row["conformal_cov90"] = float(np.mean(cover))
                    row["conformal_width_ratio"] = float(np.mean(width))
                res.append(row)
            print(st, q, proto, {k: round(metrics(yt, p_)["MAPE"], 3) for k, p_ in preds.items()}, flush=True)
    return pd.DataFrame(res), pd.DataFrame(imp_rows)


def t3(sc, v, seed=0):
    B = v.batch_master.set_index("batch_id")
    fm = v.facility_master.set_index("facility_id")
    comp = v.components.pivot_table(index="batch_id", columns="material", values="share", aggfunc="sum").reindex(B.index).fillna(0)
    X = pd.DataFrame(index=B.index)
    for c in ("category", "depth", "finish", "route"):
        X[c] = B[c]
    for st in ("spinning", "fabric", "wet", "garment"):
        X[f"cty_{st}"] = fm.loc[B[f"fac_{st}"], "country"].to_numpy()
    X["gsm"], X["ne"], X["unit_mass"], X["log_units"] = B.gsm, B["ne"], B.unit_mass, np.log(B.units)
    for m in P.MATERIALS:
        X[f"share_{m}"] = comp.get(m, 0.0)
    A = pd.get_dummies(X, columns=["category", "depth", "finish", "route"] + [f"cty_{s}" for s in
                                                                            ("spinning", "fabric", "wet", "garment")]).astype(float).to_numpy()
    y = np.log(sc.truth.set_index("batch_id").loc[B.index, "gwp_per_kg"].to_numpy())
    groups = B.variant_id.to_numpy()
    m1 = Engine(v, METHODS["M1"], n_samples=0).run().indicators.set_index("batch_id").loc[B.index, "gwp_per_kg"].to_numpy()
    preds = {k: np.zeros(len(y)) for k in ("category_median", "secondary_calc_M1", "ridge", "rf", "hgb")}
    for tr, te in GroupKFold(5).split(A, y, groups):
        cm = pd.Series(y[tr]).groupby(X.category.iloc[tr].to_numpy()).median()
        preds["category_median"][te] = X.category.iloc[te].map(cm).to_numpy()
        preds["secondary_calc_M1"][te] = np.log(m1[te])
        for k in ("ridge", "rf", "hgb"):
            preds[k][te] = make_model(k, seed).fit(A[tr], y[tr]).predict(A[te])
    rows = []
    for k, p_ in preds.items():
        rows.append(dict(model=k, **metrics(y, p_), spearman=float(spearmanr(p_, y).statistic)))
    print("T3", {r["model"]: round(r["MAPE"], 3) for r in rows}, flush=True)
    return pd.DataFrame(rows)


def anomaly_frame(sc, v):
    O = canonicalise(v.observations)
    lab = sc.obs_labels.set_index("obs_id")
    a = O[O.quantity.isin(["electricity", "heat", "water"]) & O.value_c.notna() & (O.source_system != "audit")].copy()
    mass = O[O.quantity == "mass_in"].dropna(subset=["value_c"]).drop_duplicates("lot_id").set_index("lot_id").value_c
    a = a[a.lot_id.isin(mass.index)]
    a["mass_in"] = mass.loc[a.lot_id].to_numpy()
    a["inten"] = a.value_c / a.mass_in
    a["y"] = lab.loc[a.obs_id, "anomaly"].to_numpy().astype(bool)
    a["atype"] = lab.loc[a.obs_id, "anomaly_type"].to_numpy()
    scores = {k: np.zeros(len(a)) for k in ("rule", "robust_z", "isolation_forest", "model_residual")}
    for (st, q), g in a.groupby(["stage", "quantity"]):
        ix = a.index.get_indexer(g.index)
        li = np.log(g.inten.to_numpy())
        scores["rule"][ix] = plausibility_violation(g.stage.to_numpy(), g.quantity.to_numpy(), g.inten.to_numpy()).astype(float)
        med = np.median(li)
        scores["robust_z"][ix] = np.abs(li - med) / (1.4826 * np.median(np.abs(li - med)) + 1e-9)
        X = lot_features(g.assign(stage=st), v.batch_master, v.facility_master)
        scores["isolation_forest"][ix] = isolation_scores(X, li)
        scores["model_residual"][ix] = screening_scores(X, li, g.facility_id.to_numpy())
    return a, scores


def t4(sc_val, v_val, sc_test, v_test):
    av, sv = anomaly_frame(sc_val, v_val)
    at, stt = anomaly_frame(sc_test, v_test)
    rows = []
    for k in ("rule", "robust_z", "isolation_forest", "model_residual", "rule+model_residual"):
        if k == "rule+model_residual":
            s_val = np.maximum(sv["rule"] * 1e6, sv["model_residual"])
            s_test = np.maximum(stt["rule"] * 1e6, stt["model_residual"])
        else:
            s_val, s_test = sv[k], stt[k]
        # threshold maximising F1 on the validation scenario
        cands = np.unique(np.quantile(s_val, np.linspace(0.5, 0.999, 300)))
        f1s = []
        for c in cands:
            p = s_val >= c
            tp = (p & av.y.to_numpy()).sum()
            f1s.append(2 * tp / (p.sum() + av.y.sum()) if p.sum() + av.y.sum() else 0)
        thr = cands[int(np.argmax(f1s))] if k != "rule" else 0.5
        pred = s_test >= thr
        y = at.y.to_numpy()
        tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        row = dict(method=k, threshold=float(thr), AUROC=float(roc_auc_score(y, s_test)),
                   AUPRC=float(average_precision_score(y, s_test)), precision=float(prec), recall=float(rec),
                   F1=float(2 * prec * rec / max(prec + rec, 1e-12)), prevalence=float(y.mean()), n=int(len(y)))
        for t in ("unit_error", "under_report", "spike", "stale_copy"):
            m = at.atype.to_numpy() == t
            row[f"recall_{t}"] = float(pred[m].mean()) if m.any() else None
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df[["method", "AUROC", "AUPRC", "precision", "recall", "F1"]].round(3).to_string(), flush=True)
    return df


if __name__ == "__main__":
    t0 = time.perf_counter()
    sc = generate(ScenarioConfig(n_batches=5000, seed=11))
    v = observed_view(sc)
    r12, imp = t1_t2(sc, v)
    r12.to_csv(os.path.join(OUT, "t1_t2_intensity.csv"), index=False)
    pd.DataFrame(screened).to_csv(os.path.join(OUT, "t1_t2_screened.csv"), index=False)
    imp.groupby(["stage", "quantity", "feature"]).delta_mae_log.mean().reset_index().to_csv(
        os.path.join(OUT, "t1_permutation_importance.csv"), index=False)
    t3(sc, v).to_csv(os.path.join(OUT, "t3_pcf_screening.csv"), index=False)
    sc_val = generate(ScenarioConfig(n_batches=3000, seed=12))
    t4(sc_val, observed_view(sc_val), sc, v).to_csv(os.path.join(OUT, "t4_anomaly.csv"), index=False)
    save_json(dict(runtime_s=time.perf_counter() - t0), os.path.join(OUT, "meta.json"))
    print("done", round(time.perf_counter() - t0, 1))
