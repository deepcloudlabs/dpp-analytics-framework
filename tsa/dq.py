"""Data-quality and trust model.

Uncertainty assignment schemes:

* ``ecoinvent``  - established pedigree approach as operationalised in the
  ecoinvent v3 data quality guideline (Weidema et al. 2013, Table 10.5):
  pedigree scores map to variances of the ln-transformed data, which are
  added to a basic-uncertainty variance (Table 10.3).
* ``ciroth``     - the same pedigree scores with the empirically based
  factors of Ciroth et al. (2016, Table 6), expressed as GSD; each factor
  contributes a variance (ln U)^2. Missing score-5 factors are replaced by
  the score-4 value (assumption).
* ``empirical``  - the proposed scheme: dispersion and bias of each data
  tier are estimated from the platform's own independent verification data
  and model residuals (implemented in ``engine.py``).

The observation-level data-quality score and the contribution-weighted
indicator-level score are also defined here.
"""
from __future__ import annotations

import numpy as np

# Weidema et al. (2013) ecoinvent v3 guideline, Table 10.5: variance of ln-transformed data
ECOINVENT_VAR = {
    "reliability":   (0.0, 0.0006, 0.002, 0.008, 0.04),
    "completeness":  (0.0, 0.0001, 0.0006, 0.002, 0.008),
    "temporal":      (0.0, 0.0002, 0.002, 0.008, 0.04),
    "geographical":  (0.0, 2.5e-5, 0.0001, 0.0006, 0.002),
    "technological": (0.0, 0.0006, 0.008, 0.04, 0.12),
}
# Ciroth et al. (2016), Table 6: tentative empirical factors expressed as GSD (score 5 n.a. -> score 4)
CIROTH_GSD = {
    "reliability":   (1.0, 1.54, 1.61, 1.69, 1.69),
    "completeness":  (1.0, 1.03, 1.04, 1.08, 1.08),
    "temporal":      (1.0, 1.03, 1.10, 1.19, 1.29),
    "geographical":  (1.0, 1.04, 1.08, 1.11, 1.11),
    "technological": (1.0, 1.18, 1.65, 2.08, 2.80),
}
# Weidema et al. (2013) Table 10.3: basic uncertainty (variance of ln-transformed data)
BASIC_VAR = {"energy": 0.0006, "emission_factor": 0.0006, "water": 0.0006, "transport": 0.12, "mass": 0.0006}

# Pedigree scores (reliability, completeness, temporal, geographical, technological)
# assigned to data tiers - author assumption documented in the manuscript.
TIER_PEDIGREE = {
    "measured":   (1, 1, 1, 1, 1),
    "calculated": (2, 2, 1, 1, 1),
    "estimated":  (4, 3, 1, 1, 1),
    "default":    (2, 3, 3, 2, 3),
    "imputed":    (4, 3, 1, 1, 2),
    "generic_factor": (2, 2, 2, 2, 2),
}

SCHEME = {"current": "ecoinvent"}


def pedigree_sigma(tier: str, basic: str = "energy", scheme: str | None = None) -> float:
    """Log-scale standard deviation implied by pedigree scores under a scheme."""
    scheme = scheme or SCHEME["current"]
    scores = TIER_PEDIGREE[tier]
    var = BASIC_VAR[basic]
    for (name, _), sc in zip(ECOINVENT_VAR.items(), scores):
        if scheme == "ecoinvent":
            var += ECOINVENT_VAR[name][sc - 1]
        elif scheme == "ciroth":
            var += np.log(CIROTH_GSD[name][sc - 1]) ** 2
        else:
            raise ValueError(scheme)
    return float(np.sqrt(var))


# Observation-level data-quality scores (1 = best, 5 = worst)
TIER_SCORE = {"measured": 1, "calculated": 2, "estimated": 3, "imputed": 3, "default": 4, "zero": 5, "missing": 5}
VERIF_SCORE = {"third_party": 1, "second_party": 2, "self_declared": 3, "none": 4, "platform": 3}
SPEC_SCORE = {"specific": 1, "country": 2, "generic": 4, "none": 1}


def row_dq(tier: np.ndarray, verification: np.ndarray, specificity: np.ndarray) -> np.ndarray:
    """Data-quality score of one contribution: mean of tier, verification and
    factor-specificity scores (1 best .. 5 worst)."""
    t = np.array([TIER_SCORE.get(x, 5) for x in tier], float)
    v = np.array([VERIF_SCORE.get(x, 4) for x in verification], float)
    s = np.array([SPEC_SCORE.get(x, 4) for x in specificity], float)
    return (t + v + s) / 3.0
