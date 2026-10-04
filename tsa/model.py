"""Canonical data model: vocabularies, units and identifiers.

The prototype stores the DPP core and lifecycle events as relational tables
(pandas DataFrames / SQLite). The column conventions defined here are the
canonical schema that source adapters map into and that the analytics engine
consumes. Every observation carries the metadata tuple

    value + unit + method + source + timestamp + provenance + uncertainty
    + verification status

that the manuscript proposes as the minimum analytical record.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------
DATA_TYPES = ("measured", "calculated", "estimated", "default", "imputed", "missing")
"""measured   - primary metered/lab value for the object itself
calculated - primary value derived from measured data by a documented rule
             (e.g. allocation of a metered facility total to a lot)
estimated  - value declared by the data provider without measurement
default    - secondary (generic) value from a reference database
imputed    - value predicted by a statistical/ML model inside the platform
missing    - no value available"""

VERIFICATION = ("third_party", "second_party", "self_declared", "none")

SOURCE_SYSTEMS = ("ERP", "MES", "PLM", "IoT", "supplier_portal", "logistics",
                  "certification", "laboratory", "lca_database", "recycling_system")

LIFECYCLE_EVENTS = (
    "raw_material_sourcing", "spinning", "weaving", "knitting", "dyeing_finishing",
    "cutting_assembly", "quality_inspection", "shipment", "receiving", "sale",
    "repair", "reuse", "collection", "sorting", "recycling", "disposal",
)

# Canonical quantities and their canonical units
QUANTITIES = {
    "electricity": "kWh",
    "heat": "MJ",
    "water": "m3",
    "chemicals": "kg",
    "mass_in": "kg",
    "mass_out": "kg",
    "waste": "kg",
    "waste_recovered": "kg",
    "renewable_share": "1",
    "transport_work": "tkm",
    "concentration": "mg/kg",
}

# ---------------------------------------------------------------------------
# Units: conversion to canonical units. Energy may be declared in heat or
# electricity units; conversion between kWh and MJ uses 3.6 MJ/kWh.
# UN/ECE Recommendation 20 codes are used in EPCIS exports.
# ---------------------------------------------------------------------------
UNIT_TABLE = {
    # unit: (dimension, factor to canonical unit of that dimension)
    "kWh": ("energy_kwh", 1.0), "Wh": ("energy_kwh", 1e-3), "MWh": ("energy_kwh", 1e3),
    "MJ": ("energy_mj", 1.0), "GJ": ("energy_mj", 1e3), "kJ": ("energy_mj", 1e-3),
    "m3": ("volume", 1.0), "L": ("volume", 1e-3), "l": ("volume", 1e-3),
    "kg": ("mass", 1.0), "g": ("mass", 1e-3), "t": ("mass", 1e3),
    "tkm": ("transport", 1.0),
    "mg/kg": ("concentration", 1.0), "ppm": ("concentration", 1.0), "%w/w": ("concentration", 1e4),
    "1": ("ratio", 1.0), "%": ("ratio", 0.01),
}
REC20 = {"kWh": "KWH", "Wh": "WHR", "MWh": "MWH", "MJ": "3B", "GJ": "GV", "m3": "MTQ", "L": "LTR",
         "kg": "KGM", "g": "GRM", "t": "TNE"}
KWH_TO_MJ = 3.6


class UnitError(ValueError):
    pass


def to_canonical(value: float, unit: str, quantity: str) -> float:
    """Convert ``value`` in ``unit`` into the canonical unit of ``quantity``."""
    if unit not in UNIT_TABLE:
        raise UnitError(f"unknown unit '{unit}'")
    dim, f = UNIT_TABLE[unit]
    target = QUANTITIES[quantity]
    tdim, tf = UNIT_TABLE[target]
    v = value * f
    if dim == tdim:
        return v / tf
    if dim == "energy_kwh" and tdim == "energy_mj":
        return v * KWH_TO_MJ / tf
    if dim == "energy_mj" and tdim == "energy_kwh":
        return v / KWH_TO_MJ / tf
    raise UnitError(f"cannot convert {unit} to {target} for {quantity}")


# ---------------------------------------------------------------------------
# Identifiers (GS1-style keys resolved under a reserved example domain).
# ---------------------------------------------------------------------------
RESOLVER = "https://id.example.org"


def gs1_check_digit(body: str) -> str:
    """Mod-10 check digit used by GTIN/GLN keys."""
    total = 0
    for i, ch in enumerate(reversed(body)):
        total += int(ch) * (3 if i % 2 == 0 else 1)
    return str((10 - total % 10) % 10)


def gtin14(company_prefix: int, item_ref: int) -> str:
    body = f"0{company_prefix:07d}{item_ref:05d}"[:13]
    return body + gs1_check_digit(body)


def gln13(company_prefix: int, loc_ref: int) -> str:
    body = f"{company_prefix:07d}{loc_ref:05d}"[:12]
    return body + gs1_check_digit(body)


def digital_link(gtin: str, lot: str | None = None, serial: str | None = None) -> str:
    uri = f"{RESOLVER}/01/{gtin}"
    if lot:
        uri += f"/10/{lot}"
    if serial:
        uri += f"/21/{serial}"
    return uri


def record_hash(*parts) -> str:
    """Content hash used as an integrity reference in provenance records."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return h[:16]


@dataclass(frozen=True)
class RuleRef:
    """Identifier of a versioned calculation or compliance rule."""
    rule_id: str
    version: str
    description: str

    @property
    def iri(self) -> str:
        return f"urn:tsa:rule:{self.rule_id}:{self.version}"
