"""E6: interoperability.

(a) EPCIS 2.0 export of all lifecycle events, validated against the official
    GS1 EPCIS 2.0 JSON Schema (with and without the analytical extension profile);
(b) round-trip fidelity: recovery of quantity semantics and analytical
    metadata when re-importing observations from EPCIS documents;
(c) JSON-LD passport documents: undefined-term check and completeness of the
    analytical record for every indicator result;
(d) heterogeneous-source ingestion through declarative mappings (functional
    verification and throughput).
"""
from __future__ import annotations

import json
import os
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import RESULTS, EPCIS_SCHEMA, save_json  # noqa: E402
from tsa.synth import generate, ScenarioConfig  # noqa: E402
from tsa.engine import Engine, METHODS, observed_view, canonicalise  # noqa: E402
from tsa import interop, acquisition  # noqa: E402

OUT = os.path.join(RESULTS, "e6")
os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    t0 = time.perf_counter()
    sc = generate(ScenarioConfig(n_batches=2000, seed=31))
    v = observed_view(sc)
    O = canonicalise(v.observations)
    r = Engine(v, METHODS["M3"], n_samples=300, seed=31).run()
    val = interop.load_validator(EPCIS_SCHEMA)
    res = {}
    batches = v.dpp.batch_id.to_numpy()
    chunks = [batches[i:i + 50] for i in range(0, len(batches), 50)]
    for ext in (True, False):
        n_ev, n_err, t_exp, t_val, size, back = 0, 0, 0.0, 0.0, 0, []
        types = {}
        for ch in chunks:
            t = time.perf_counter(); doc = interop.to_epcis(v, ch, O, extensions=ext); t_exp += time.perf_counter() - t
            t = time.perf_counter(); errs = interop.validate(doc, val); t_val += time.perf_counter() - t
            n_err += len(errs)
            n_ev += len(doc["epcisBody"]["eventList"])
            for e in doc["epcisBody"]["eventList"]:
                types[e["type"]] = types.get(e["type"], 0) + 1
            size += len(json.dumps(doc))
            back.append(interop.from_epcis(doc))
        B = pd.concat(back, ignore_index=True)
        key = "with_extensions" if ext else "standard_only"
        # round-trip: compare to the original observations exported as sensor reports
        orig = O[O.value_c.notna() & O.quantity.isin(list(interop.SENSOR_MAP) +
                                                        ["fibre_gwp_intensity", "fibre_water_intensity"])]
        orig = orig[orig.source_system != "audit"] if False else orig
        res[key] = dict(documents=len(chunks), events=n_ev, event_types=types, schema_errors=n_err,
                        export_s=t_exp, validate_s=t_val, bytes=size, sensor_reports=len(B))
        if ext:
            m = B.merge(orig[["obs_id", "quantity", "value_c", "data_type", "verification", "source_system"]],
                        on="obs_id", suffixes=("", "_ref"))
            res[key].update(quantity_recovered=float((m.quantity == m.quantity_ref).mean()),
                            value_exact=float(np.isclose(m.value_c, m.value_c_ref, rtol=1e-12).mean()),
                            data_type_preserved=float((m.data_type == m.data_type_ref).mean()),
                            verification_preserved=float((m.verification == m.verification_ref).mean()),
                            source_preserved=float((m.source_system == m.source_system_ref).mean()),
                            matched=len(m))
        else:
            # without extensions observation identifiers are lost; compare multisets per quantity class
            dist_back = B.quantity.value_counts()
            exported = orig[orig.quantity.isin(interop.SENSOR_MAP)].quantity.value_counts()
            ambiguous = {"water/wastewater (MTQ)": int(exported.get("wastewater", 0)),
                         "chemicals/waste/waste_recovered (KGM)": int(exported.get("waste", 0) + exported.get("waste_recovered", 0))}
            res[key].update(recovered_quantity_counts=dist_back.to_dict(), exported_quantity_counts=exported.to_dict(),
                            misclassified_reports=sum(ambiguous.values()),
                            misclassified_share=sum(ambiguous.values()) / exported.sum(),
                            ambiguous_classes=ambiguous, data_type_preserved=0.0, verification_preserved=0.0,
                            source_preserved=0.0, fibre_primary_data_exported=0)
    # --- JSON-LD passports ---------------------------------------------------------------
    t = time.perf_counter()
    docs = [interop.passport_jsonld(v, r, b, "2026-09-25T00:00:00Z") for b in batches]
    t_pp = time.perf_counter() - t
    undefined = set().union(*[interop.undefined_terms(d) for d in docs])
    comp = np.array([interop.record_completeness(d) for d in docs])
    res["passport_jsonld"] = dict(n=len(docs), generation_s=t_pp, undefined_terms=sorted(undefined),
                                  record_completeness_mean=float(comp.mean()),
                                  bytes_mean=float(np.mean([len(json.dumps(d)) for d in docs])))
    # --- heterogeneous ingestion -----------------------------------------------------------
    src, faults = acquisition.export_sources(O, seed=31, fault_rate=0.01)
    t = time.perf_counter(); R, Q = acquisition.ingest(src); t_in = time.perf_counter() - t
    ref = O.set_index("obs_id")
    m = R.set_index("obs_id").join(ref[["value_c", "quantity", "lot_id"]], rsuffix="_ref")
    rel = (m.value_c / m.value_c_ref - 1).abs()
    res["ingestion"] = dict(records_mapped=len(R), records_quarantined=len(Q), faults_injected=len(faults),
                            quarantine_recall=len(set(Q.obs_id) & faults) / max(len(faults), 1),
                            false_quarantine=len(set(Q.obs_id) - faults), value_exact_1e9=float((rel < 1e-9).mean()),
                            quantity_correct=float((m.quantity == m.quantity_ref).mean()),
                            lot_correct=float((m.lot_id == m.lot_id_ref).mean()), ingest_s=t_in,
                            records_per_s=(len(R) + len(Q)) / t_in, by_source=R.source.value_counts().to_dict(),
                            quarantine_reasons=Q.reason.value_counts().to_dict())
    res["runtime_s"] = time.perf_counter() - t0
    save_json(res, os.path.join(OUT, "interop.json"))
    print(json.dumps(res, indent=1, default=float))
