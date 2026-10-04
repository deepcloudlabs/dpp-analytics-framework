"""Interoperability: EPCIS 2.0 event export/import and JSON-LD passport documents.

* Lifecycle events are exported as GS1 EPCIS 2.0 JSON-LD documents
  (ObjectEvent for sourcing, shipment and collection; TransformationEvent for
  spinning, fabric formation, wet processing, cut-make-trim and recycling).
  Activity observations are carried in ``sensorElementList`` (measurement
  types Energy, Volume, Mass with UN/ECE Rec. 20 units). Analytical metadata
  that EPCIS does not standardise (data tier, verification status, source
  system, uncertainty) are carried as namespaced extension keys.
* Documents are validated against the official EPCIS 2.0 JSON Schema
  (https://ref.gs1.org/standards/epcis/2.0.1/epcis-json-schema.json), which
  must be obtained from GS1 and passed by path.
* ``from_epcis`` re-imports observations to measure round-trip fidelity with
  and without the extension profile.
* The passport document is JSON-LD with an explicit @context; every
  indicator result carries the analytical record
  value + unit + method + source + timestamp + provenance + uncertainty + verification.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .model import REC20, gtin14, digital_link, RESOLVER

EPCIS_CONTEXT = "https://ref.gs1.org/standards/epcis/epcis-context.jsonld"
TSA_NS = "https://example.org/tsa#"
MATERIAL_GTIN = {}
STAGE_EVENT = {"spinning": "spinning", "fabric": None, "wet": "dyeing_finishing", "garment": "cutting_assembly"}
SENSOR_MAP = {  # canonical quantity -> (EPCIS measurementType, Rec20 uom, canonical unit, extension flow tag)
    "electricity": ("Energy", "KWH", "kWh", "electricity"),
    "heat": ("Energy", "3B", "MJ", "fuel"),
    "water": ("Volume", "MTQ", "m3", "water_withdrawal"),
    "wastewater": ("Volume", "MTQ", "m3", "wastewater_discharge"),
    "chemicals": ("Mass", "KGM", "kg", "chemical_input"),
    "waste": ("Mass", "KGM", "kg", "solid_waste"),
    "waste_recovered": ("Mass", "KGM", "kg", "solid_waste_recovered"),
}


def _iso(t) -> str:
    return pd.Timestamp(t).tz_localize("UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z") if pd.Timestamp(t).tzinfo is None \
        else pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _gln_uri(gln: str) -> str:
    return f"{RESOLVER}/414/{gln}"


def _class_uri(lot_id: str, material: str | None = None) -> str:
    key = material or "intermediate"
    if key not in MATERIAL_GTIN:
        MATERIAL_GTIN[key] = gtin14(6000000, len(MATERIAL_GTIN) + 1)
    return digital_link(MATERIAL_GTIN[key], lot_id.replace("-", ""))


# ---------------------------------------------------------------------------
def to_epcis(view, batch_ids, canon_obs: pd.DataFrame, extensions: bool = True) -> dict:
    """Export the lifecycle events of ``batch_ids`` as one EPCIS 2.0 document."""
    ev = view.events[view.events.batch_id.isin(batch_ids)]
    fm = view.facility_master.set_index("facility_id")
    O = canon_obs[canon_obs.batch_id.isin(batch_ids) & canon_obs.value_c.notna()]
    by_lot = {k: g for k, g in O.groupby("lot_id")}
    flm = view.fibre_lot_master.set_index("lot_id")
    certs = view.certificates.groupby("holder_facility").cert_id.apply(list).to_dict()
    events = []
    for r in ev.itertuples():
        loc = _gln_uri(fm.at[r.facility_id, "gln"]) if r.facility_id in fm.index else f"{RESOLVER}/414/unknown"
        base = {"eventTime": _iso(r.time), "eventTimeZoneOffset": "+00:00",
                "eventID": f"urn:uuid:{_uuid(r.event_id)}", "readPoint": {"id": loc}, "bizLocation": {"id": loc}}
        if extensions:
            base["tsa:eventRef"] = r.event_id
            base["tsa:batch"] = r.batch_id
        cert = certs.get(r.facility_id)
        if cert:
            base["certificationInfo"] = [f"https://cert.example.org/{c}" for c in cert]
        outs = json.loads(r.outputs) if r.outputs else []
        ins = json.loads(r.inputs) if r.inputs else []
        if r.event_type == "raw_material_sourcing":
            lot = outs[0]
            mass = _mass(by_lot.get(lot), "mass_out")
            e = {"type": "ObjectEvent", "action": "ADD", "bizStep": "commissioning", **base,
                 "quantityList": [{"epcClass": _class_uri(lot, flm.at[lot, "material"] if lot in flm.index else None),
                                   **({"quantity": round(mass, 6), "uom": "KGM"} if mass is not None else {})}]}
            if extensions:
                e["tsa:material"] = flm.at[lot, "material"] if lot in flm.index else None
            e.update(_sensor(by_lot.get(lot), extensions, fibre=True))
        elif r.event_type in ("spinning", "weaving", "knitting", "dyeing_finishing", "cutting_assembly"):
            lot = outs[0]
            g = by_lot.get(lot)
            e = {"type": "TransformationEvent", **base,
                 "inputQuantityList": [{"epcClass": _class_uri(i)} for i in ins],
                 "outputQuantityList": [{"epcClass": _class_uri(lot),
                                         **({"quantity": round(_mass(g, "mass_out"), 6), "uom": "KGM"}
                                            if _mass(g, "mass_out") is not None else {})}]}
            if extensions:
                e["tsa:processType"] = r.event_type
                e["tsa:lot"] = lot
            e.update(_sensor(g, extensions))
        elif r.event_type == "shipment":
            leg = ins[0]
            g = by_lot.get(leg)
            e = {"type": "ObjectEvent", "action": "OBSERVE", "bizStep": "shipping", "disposition": "in_transit", **base,
                 "quantityList": [{"epcClass": _class_uri(leg)}]}
            if extensions and g is not None and len(g):
                row = g.iloc[0]
                e["tsa:leg"] = leg
                e["tsa:transportMode"] = row.get("mode")
                e["tsa:tkmMain"] = None if pd.isna(row.get("tkm_main")) else float(row.get("tkm_main"))
                e["tsa:tkmRoad"] = None if pd.isna(row.get("tkm_road")) else float(row.get("tkm_road"))
        elif r.event_type == "collection":
            e = {"type": "ObjectEvent", "action": "OBSERVE", "bizStep": "collecting", **base,
                 "quantityList": [{"epcClass": _class_uri(r.batch_id + "-EOL")}]}
        elif r.event_type in ("sorting", "recycling"):
            e = {"type": "ObjectEvent", "action": "OBSERVE", "bizStep": "inspecting" if r.event_type == "sorting" else "other",
                 **base, "quantityList": [{"epcClass": _class_uri(r.batch_id + "-EOL")}]}
            if extensions:
                e["tsa:processType"] = r.event_type
        else:
            continue
        events.append(e)
    ctx = [EPCIS_CONTEXT] + ([{"tsa": TSA_NS}] if extensions else [])
    return {"@context": ctx, "type": "EPCISDocument", "schemaVersion": "2.0",
            "creationDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "epcisBody": {"eventList": events}}


def _uuid(s: str) -> str:
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "tsa:" + s))


def _mass(g, q):
    if g is None:
        return None
    m = g[g.quantity == q]
    return float(m.value_c.iat[0]) if len(m) else None


def _sensor(g, extensions, fibre=False) -> dict:
    if g is None or not len(g):
        return {}
    reports = []
    for row in g.itertuples():
        if fibre and row.quantity in ("fibre_gwp_intensity", "fibre_water_intensity"):
            if not extensions:
                continue
            rep = {"type": f"{TSA_NS}{'CarbonIntensity' if 'gwp' in row.quantity else 'WaterIntensity'}",
                   "value": float(row.value_c)}
        elif row.quantity in SENSOR_MAP:
            mt, uom, _, flow = SENSOR_MAP[row.quantity]
            rep = {"type": mt, "value": float(row.value_c), "uom": uom}
            if extensions:
                rep["tsa:flow"] = flow
        else:
            continue
        if extensions:
            rep["tsa:dataType"] = row.data_type
            rep["tsa:verification"] = row.verification
            rep["tsa:sourceSystem"] = row.source_system
            rep["tsa:observationRef"] = row.obs_id
        if row.data_type == "calculated":
            rep["dataProcessingMethod"] = "https://example.org/tsa/method/allocation-by-mass"
        reports.append(rep)
    if not reports:
        return {}
    return {"sensorElementList": [{"sensorMetadata": {"time": _iso(g.timestamp.iat[0])}, "sensorReport": reports}]}


# ---------------------------------------------------------------------------
def load_validator(schema_path: str):
    import jsonschema
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    cls = jsonschema.validators.validator_for(schema)
    return cls(schema, format_checker=cls.FORMAT_CHECKER)


def validate(doc: dict, validator) -> list[str]:
    return [f"{'/'.join(map(str, e.absolute_path))}: {e.message[:160]}" for e in validator.iter_errors(doc)]


def from_epcis(doc: dict) -> pd.DataFrame:
    """Re-import sensor observations from an EPCIS document into canonical rows.

    Without the extension profile, electricity and fuel energy can only be
    distinguished by unit convention, and data tier / verification / source
    are lost; those fields are returned as None."""
    unit_to_q = {"KWH": ("electricity", 1.0), "3B": ("heat", 1.0), "MTQ": ("water", 1.0), "KGM": ("chemicals", 1.0)}
    flow_to_q = {v[3]: k for k, v in SENSOR_MAP.items()}
    rows = []
    for e in doc["epcisBody"]["eventList"]:
        for se in e.get("sensorElementList", []):
            for rep in se["sensorReport"]:
                flow = rep.get("tsa:flow")
                if flow:
                    q = flow_to_q[flow]
                elif rep.get("uom") in unit_to_q:
                    q = unit_to_q[rep["uom"]][0]
                elif isinstance(rep.get("type"), str) and "Intensity" in rep["type"]:
                    q = "fibre_gwp_intensity" if "Carbon" in rep["type"] else "fibre_water_intensity"
                else:
                    q = None
                rows.append(dict(obs_id=rep.get("tsa:observationRef"), quantity=q, value_c=rep.get("value"),
                                 data_type=rep.get("tsa:dataType"), verification=rep.get("tsa:verification"),
                                 source_system=rep.get("tsa:sourceSystem"),
                                 lot=e.get("tsa:lot"), event_time=e["eventTime"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
PASSPORT_CONTEXT = {
    "@version": 1.1,
    "schema": "https://schema.org/",
    "gs1": "https://gs1.org/voc/",
    "prov": "http://www.w3.org/ns/prov#",
    "dqv": "http://www.w3.org/ns/dqv#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "tsa": TSA_NS,
    "gtin": "gs1:gtin",
    "batch": "tsa:batch",
    "operator": {"@id": "tsa:operator", "@type": "@id"},
    "manufacturingFacility": {"@id": "tsa:manufacturingFacility", "@type": "@id"},
    "countryOfManufacture": "tsa:countryOfManufacture",
    "fibreComposition": {"@id": "tsa:fibreComposition", "@container": "@list"},
    "fibre": "tsa:fibre",
    "share": {"@id": "tsa:share", "@type": "xsd:decimal"},
    "claims": {"@id": "tsa:claim", "@container": "@set"},
    "claimType": "tsa:claimType",
    "claimValue": {"@id": "tsa:claimValue", "@type": "xsd:decimal"},
    "evidence": {"@id": "tsa:evidence", "@type": "@id", "@container": "@set"},
    "indicators": {"@id": "tsa:indicatorResult", "@container": "@set"},
    "indicator": {"@id": "tsa:indicator", "@type": "@id"},
    "value": {"@id": "tsa:value", "@type": "xsd:decimal"},
    "unit": "tsa:unit",
    "lower95": {"@id": "tsa:lower95", "@type": "xsd:decimal"},
    "upper95": {"@id": "tsa:upper95", "@type": "xsd:decimal"},
    "method": {"@id": "tsa:calculationMethod", "@type": "@id"},
    "boundary": "tsa:lifecycleBoundary",
    "source": "tsa:dataSourceSummary",
    "generatedAt": {"@id": "prov:generatedAtTime", "@type": "xsd:dateTime"},
    "provenance": {"@id": "prov:wasDerivedFrom", "@type": "@id"},
    "verificationStatus": "tsa:verificationStatus",
    "primaryDataShare": {"@id": "tsa:primaryDataShare", "@type": "xsd:decimal"},
    "dataQuality": {"@id": "dqv:hasQualityMeasurement"},
    "qualityValue": {"@id": "dqv:value", "@type": "xsd:decimal"},
    "qualityDimension": {"@id": "dqv:isMeasurementOf", "@type": "@id"},
}
RECORD_FIELDS = ["value", "unit", "method", "source", "generatedAt", "provenance", "lower95", "upper95",
                 "verificationStatus"]
INDICATOR_META = {
    "gwp_per_unit": ("tsa:CLI-02", "kg CO2e/unit", "cradle-to-shelf (excl. use, end-of-life)"),
    "energy_per_unit": ("tsa:ENE-01", "kWh/unit", "manufacturing gate-to-gate (spinning to garment)"),
    "water_per_unit": ("tsa:WAT-02", "m3/unit", "cradle-to-gate (fibre + wet processing)"),
    "aware_per_unit": ("tsa:WAT-03", "m3 world-eq/unit", "cradle-to-gate (fibre + wet processing)"),
}


def passport_jsonld(view, result, batch_id: str, generated_at: str) -> dict:
    d = view.dpp.set_index("batch_id").loc[batch_id]
    comp = view.components[view.components.batch_id == batch_id]
    cl = view.claims[view.claims.batch_id == batch_id]
    I = result.indicators.set_index("batch_id").loc[batch_id]
    inds = []
    for name, (iri, unit, boundary) in INDICATOR_META.items():
        lo, hi = I.get(f"{name}_lo"), I.get(f"{name}_hi")
        inds.append({"@id": f"urn:tsa:ind:{batch_id}:{name}", "@type": "tsa:IndicatorResult", "indicator": iri,
                     "value": round(float(I[name]), 6), "unit": unit, "boundary": boundary,
                     "lower95": None if lo is None or pd.isna(lo) else round(float(lo), 6),
                     "upper95": None if hi is None or pd.isna(hi) else round(float(hi), 6),
                     "method": f"urn:tsa:method:{result.method.name.split()[0]}:v1",
                     "source": "lifecycle events and observations of the product batch (see provenance)",
                     "generatedAt": generated_at, "provenance": f"urn:tsa:ind:{batch_id}:{name}:lineage",
                     "verificationStatus": "platform-calculated; not third-party verified",
                     "primaryDataShare": round(float(I["primary_share_gwp"]), 4) if name == "gwp_per_unit" else None,
                     "dataQuality": {"@type": "dqv:QualityMeasurement", "qualityDimension": "tsa:ContributionWeightedDQ",
                                     "qualityValue": round(float(I["dq_score_gwp"]), 4)} if name == "gwp_per_unit" else None})
    doc = {"@context": PASSPORT_CONTEXT, "@id": d.dpp_uri if isinstance(d.dpp_uri, str) else f"urn:tsa:dpp:{batch_id}",
           "@type": ["tsa:ProductPassport", "gs1:Product"],
           "gtin": d.gtin, "batch": d.lot, "operator": d.operator_id,
           "manufacturingFacility": f"{RESOLVER}/414/{d.manufacturer_facility}" if isinstance(d.manufacturer_facility, str) else None,
           "countryOfManufacture": d.country_of_manufacture,
           "fibreComposition": [{"fibre": m, "share": float(s)} for m, s in zip(comp.material, comp.share)],
           "claims": [{"claimType": t, "claimValue": float(v), "evidence": [f"https://cert.example.org/{e}" for e in json.loads(ev)]}
                      for t, v, ev in zip(cl.claim_type, cl.value, cl.evidence)],
           "indicators": inds}
    return _drop_none(doc)


def _drop_none(x):
    if isinstance(x, dict):
        return {k: _drop_none(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [_drop_none(v) for v in x]
    return x


def undefined_terms(doc: dict, ctx: dict = PASSPORT_CONTEXT) -> set:
    """JSON-LD term check: keys that would be dropped by expansion (neither
    keywords, context terms nor CURIEs with a declared prefix)."""
    prefixes = {k for k, v in ctx.items() if isinstance(v, str) and v.endswith(("/", "#"))}
    bad = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k.startswith("@"):
                    pass
                elif k in ctx:
                    pass
                elif ":" in k and k.split(":", 1)[0] in prefixes:
                    pass
                else:
                    bad.add(k)
                if k != "@context":
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(doc)
    return bad


def record_completeness(doc: dict) -> float:
    """Share of indicator results that carry every field of the analytical record."""
    inds = doc.get("indicators", [])
    if not inds:
        return float("nan")
    ok = [all(f in r for f in RECORD_FIELDS) for r in inds]
    return float(np.mean(ok))
