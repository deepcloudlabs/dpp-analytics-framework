"""Machine-learning components.

* Feature construction from passport/PLM attributes of process lots.
* ``IntensityImputer``: estimation of missing activity intensities (e.g. kWh/kg,
  MJ/kg, m3/kg) with a cross-fitted gradient-boosting model, a shrunken
  facility effect learned from the facility's own verified history, and a
  split-conformal residual pool used to propagate estimation uncertainty.
* ``screening_scores``: plausibility screening of reported activity data by
  physical range rules and model-based robust residual scores.

All models are trained on OBSERVED passport data only (never on ground truth).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, KFold

CAT_COLS = ["stage", "route", "category", "depth", "finish", "country"]
NUM_COLS = ["gsm", "ne", "log_mass", "unit_mass", "log_units", "dye_reactive", "dye_disperse", "dye_acid",
            "dye_vat_indigo", "n_dye"]
CATEGORIES = {
    "stage": ["spinning", "fabric", "wet", "garment"],
    "route": ["knit", "woven"],
    "category": ["tshirt", "jeans", "sweater", "jacket", "dress"],
    "depth": ["light", "medium", "dark"],
    "finish": ["none", "softener", "enzyme_wash", "wr_fluorinated", "wr_fluorine_free"],
    "country": ["CN", "IN", "BD", "PK", "VN", "TR", "IT", "PT", "DE"],
}

# physical plausibility bounds on canonical intensities (per kg of lot input)
PLAUSIBLE = {
    ("spinning", "electricity"): (0.3, 20.0), ("fabric", "electricity"): (0.1, 20.0),
    ("wet", "electricity"): (0.1, 10.0), ("garment", "electricity"): (0.05, 15.0),
    ("wet", "heat"): (2.0, 200.0), ("wet", "water"): (0.005, 0.5),
}


def lot_features(lots: pd.DataFrame, batch_master: pd.DataFrame, facility_master: pd.DataFrame) -> pd.DataFrame:
    """Design matrix for process lots. ``lots`` needs batch_id, stage, facility_id, mass_in."""
    bm = batch_master.set_index("batch_id")
    fm = facility_master.set_index("facility_id")
    X = pd.DataFrame(index=lots.index)
    X["stage"] = lots.stage.values
    b = bm.loc[lots.batch_id.values]
    for c in ("route", "category", "depth", "finish"):
        X[c] = b[c].values
    X["country"] = fm.loc[lots.facility_id.values, "country"].values
    X["gsm"] = b["gsm"].values
    X["ne"] = b["ne"].values
    X["log_mass"] = np.log(np.maximum(lots.mass_in.values.astype(float), 1e-6))
    X["unit_mass"] = b["unit_mass"].values
    X["log_units"] = np.log(b["units"].values.astype(float))
    dyes = b["dye_classes"].values
    for d in ("reactive", "disperse", "acid", "vat_indigo"):
        X[f"dye_{d}"] = [float(d in s.split("|")) for s in dyes]
    X["n_dye"] = [len(s.split("|")) for s in dyes]
    return X


def encode(X: pd.DataFrame) -> np.ndarray:
    parts = [X[NUM_COLS].to_numpy(float)]
    for c in CAT_COLS:
        cats = CATEGORIES[c]
        v = X[c].to_numpy()
        parts.append(np.stack([(v == k).astype(float) for k in cats], axis=1))
    return np.hstack(parts)


def make_model(name: str, seed: int = 0, loss: str = "squared_error"):
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                                             min_samples_leaf=20, l2_regularization=1.0, loss=loss, random_state=seed)
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=800,
                                                            early_stopping=True, random_state=seed))
    raise ValueError(name)


class IntensityImputer:
    """Log-intensity regression + shrunken facility effect + conformal residuals."""

    def __init__(self, model: str = "hgb", k_shrink: float = 3.0, n_folds: int = 5, seed: int = 0):
        self.model_name, self.k, self.n_folds, self.seed = model, k_shrink, n_folds, seed

    def fit(self, X: pd.DataFrame, y: np.ndarray, fac: np.ndarray):
        A = encode(X)
        n = len(y)
        oof = np.zeros(n)
        n_groups = len(np.unique(fac))
        splitter = GroupKFold(n_splits=min(self.n_folds, n_groups))
        for tr, te in splitter.split(A, y, groups=fac):
            m = make_model(self.model_name, self.seed).fit(A[tr], y[tr])
            oof[te] = m.predict(A[te])
        r = y - oof
        df = pd.DataFrame(dict(fac=fac, r=r))
        s = df.groupby("fac").r.agg(["sum", "count"])
        self.fac_effect_ = (s["sum"] / (s["count"] + self.k)).to_dict()
        # leave-one-out facility effect for conformal residuals
        sums = df.fac.map(s["sum"]).to_numpy()
        cnts = df.fac.map(s["count"]).to_numpy()
        loo = (sums - r) / (cnts - 1 + self.k)
        self.resid_ = r - loo
        self.resid_unseen_ = r
        self.model_ = make_model(self.model_name, self.seed).fit(A, y)
        return self

    def predict(self, X: pd.DataFrame, fac: np.ndarray):
        mu = self.model_.predict(encode(X))
        eff = np.array([self.fac_effect_.get(f, 0.0) for f in fac])
        seen = np.array([f in self.fac_effect_ for f in fac])
        return mu + eff, seen

    def interval(self, alpha: float = 0.05, seen: bool = True):
        r = self.resid_ if seen else self.resid_unseen_
        return np.quantile(r, [alpha / 2, 1 - alpha / 2])


def screening_scores(X: pd.DataFrame, y: np.ndarray, fac: np.ndarray, seed: int = 0, k: float = 3.0) -> np.ndarray:
    """Robust residual score |r - b_f| / (1.4826 MAD) from a cross-fitted
    absolute-error gradient-boosting model with leave-one-out facility effect.
    ``y`` is the log-intensity of reported (any tier) values."""
    A = encode(X)
    oof = np.zeros(len(y))
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(A):
        m = make_model("hgb", seed, loss="absolute_error").fit(A[tr], y[tr])
        oof[te] = m.predict(A[te])
    r = y - oof
    df = pd.DataFrame(dict(fac=fac, r=r))
    med = df.groupby("fac").r.transform(lambda s: (s.sum() - s) / (len(s) - 1 + k) if len(s) > 1 else 0.0)
    rr = r - med.to_numpy()
    mad = np.median(np.abs(rr - np.median(rr))) * 1.4826 + 1e-9
    return np.abs(rr - np.median(rr)) / mad


def plausibility_violation(stage: np.ndarray, quantity: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    lo = np.array([PLAUSIBLE.get((s, q), (0, np.inf))[0] for s, q in zip(stage, quantity)])
    hi = np.array([PLAUSIBLE.get((s, q), (0, np.inf))[1] for s, q in zip(stage, quantity)])
    return (intensity < lo) | (intensity > hi)


def isolation_scores(X: pd.DataFrame, y: np.ndarray, seed: int = 0) -> np.ndarray:
    A = np.hstack([encode(X), y[:, None]])
    iso = IsolationForest(n_estimators=300, random_state=seed).fit(A)
    return -iso.score_samples(A)
