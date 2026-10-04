"""Reference factor database used by the analytics engine.

Factors are external knowledge (emission factors, characterisation factors,
secondary/default process data). Each factor is versioned and carries a
dataset reference and an uncertainty (log-scale standard deviation), so that
it can appear as a leaf entity in the analytical provenance graph.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import params as P


@dataclass(frozen=True)
class Factor:
    factor_id: str
    value: float
    unit: str
    dataset: str
    version: str
    sigma: float          # log-scale standard deviation (lognormal); 0 = treated as exact
    specificity: str      # 'specific' | 'country' | 'generic'


def build_factor_db(default_sigma: dict | None = None) -> dict[str, Factor]:
    """Return the factor table. ``default_sigma`` overrides the uncertainty of
    generic (secondary) factors, e.g. with empirically calibrated values."""
    ds = default_sigma or {}
    F: dict[str, Factor] = {}
    for c, v in P.COUNTRIES.items():
        F[f"grid:{c}"] = Factor(f"grid:{c}", v["grid_ef"], "kgCO2e/kWh", "grid-intensity", "2024", 0.10, "country")
        mix = sum(w * P.FUEL_EF[f] for f, w in v["fuel"].items()) / sum(v["fuel"].values())
        F[f"fuelmix:{c}"] = Factor(f"fuelmix:{c}", mix, "kgCO2e/MJ", "industrial-heat-mix", "assumption-1",
                                   ds.get("fuelmix", 0.35), "country")
        F[f"aware_na:{c}"] = Factor(f"aware_na:{c}", v["aware_na"], "m3eq/m3", "AWARE-non-agri", "2.0", 0.0, "country")
        F[f"aware_un:{c}"] = Factor(f"aware_un:{c}", v["aware_un"], "m3eq/m3", "AWARE-unspecified", "2.0", 0.0, "country")
    for f, v in P.FUEL_EF.items():
        F[f"fuel:{f}"] = Factor(f"fuel:{f}", v, "kgCO2e/MJ", "IPCC-2006-stationary", "2006", 0.05, "specific")
    for m, v in P.TRANSPORT_EF.items():
        F[f"transport:{m}"] = Factor(f"transport:{m}", v, "kgCO2e/tkm", "transport-ef", "2024", 0.20, "generic")
    for m, v in P.MATERIALS.items():
        F[f"fibre_gwp:{m}"] = Factor(f"fibre_gwp:{m}", v["gwp"], "kgCO2e/kg", "fibre-lci", "v1",
                                     ds.get("fibre_gwp", 0.30), "generic")
        F[f"fibre_water:{m}"] = Factor(f"fibre_water:{m}", v["water"] / 1000.0, "m3/kg", "fibre-lci", "v1",
                                       ds.get("fibre_water", 0.35), "generic")
    D = P.DEFAULTS
    for k, unit in (("spin_kwh_per_kg", "kWh/kg"), ("weave_kwh_per_kg", "kWh/kg"), ("knit_kwh_per_kg", "kWh/kg"),
                    ("wet_elec_kwh_per_kg", "kWh/kg"), ("wet_heat_mj_per_kg", "MJ/kg"),
                    ("wet_water_l_per_kg", "m3/kg"), ("garment_kwh_per_kg", "kWh/kg")):
        val = D[k] / 1000.0 if k == "wet_water_l_per_kg" else D[k]
        F[f"default:{k}"] = Factor(f"default:{k}", val, unit, "generic-process-data", "v1",
                                   ds.get("process", 0.40), "generic")
    F["default:consumption_ratio"] = Factor("default:consumption_ratio", 0.10, "1", "generic-process-data", "v1",
                                            ds.get("consumption_ratio", 0.40), "generic")
    return F
