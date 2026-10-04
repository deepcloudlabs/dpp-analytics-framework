# Textile sustainability analytics

Research prototype accompanying the manuscript *From passport data to sustainability intelligence: a standards-aligned, provenance-aware analytics framework for textile Digital Product Passports*.

**All data produced by this code are synthetic.** The scenario generator encodes documented assumptions (see `tsa/params.py` and manuscript Appendix A); results characterise method behaviour under those assumptions and are not industrial evidence.

## Layout

| Path | Content |
|---|---|
| `tsa/params.py` | Scenario parameters with source tags (literature, datasets, or `assumption`) |
| `tsa/synth.py` | Ground-truth value-chain generator and data-degradation model (observed DPP layer, anomalies, faults) |
| `tsa/model.py` | Vocabularies, units (UN/ECE Rec. 20 codes), identifiers (GS1-style keys under `id.example.org`) |
| `tsa/factors.py` | Versioned factor database |
| `tsa/dq.py` | Pedigree schemes (ecoinvent v3 guideline, Ciroth et al. 2016) and data-quality scores |
| `tsa/ml.py` | Features, intensity imputer (GBM + facility effect + conformal residuals), screening scores |
| `tsa/engine.py` | Analytics engine: methods M0–M3 and ablations; contribution table; Monte Carlo; indicators (incl. MCI) |
| `tsa/provenance.py` | Analytical provenance graph (PROV-O semantics), SQLite store, explain/impact queries, PROV JSON-LD export |
| `tsa/compliance.py` | Versioned rules R01–R10 (violation vs. unverifiable) |
| `tsa/interop.py` | EPCIS 2.0 export/import, JSON-LD passports, analytical-record completeness |
| `tsa/acquisition.py` | Heterogeneous source extracts and declarative mapping-based ingestion |
| `tsa/store.py` | SQLite DPP store, HTTP/JSON API (illustrative; not the EN 18222 specification), load benchmark |
| `experiments/run_e1.py` … `run_e8_e9.py` | Experiments E1–E9 |
| `experiments/analyze.py`, `analyze_more.py` | Generate manuscript tables, figures and number macros |
| `results/` | Experiment outputs (CSV/JSON) and logs |

## Requirements

Python ≥ 3.12 with numpy, pandas, scipy, scikit-learn, matplotlib, networkx and jsonschema (tested with Python 3.14.3, NumPy 2.4.2, pandas 3.0.1, SciPy 1.17.1, scikit-learn 1.8.0). E6 needs the official GS1 EPCIS 2.0 JSON Schema (`epcis-json-schema.json`, https://ref.gs1.org/standards/epcis/). A local copy retrieved from ref.gs1.org on 2026-09-25 is in `external/` for convenience; check GS1's terms before redistributing it in a public repository, or remove it and set the `EPCIS_SCHEMA` environment variable to your own download.

## Reproduction

```bash
cd experiments
python launch_e1.py 6          # E1, E2, E4 (5 seeds) and the missing-data sensitivity grid
python run_e3.py               # provenance
python run_e5.py               # machine-learning tasks T1–T4
python run_e6.py               # interoperability
python run_e8_e9.py            # aggregation rules and composite scores (needs E1 seed 1)
python run_e7.py --rounds 3    # scalability: 3 rounds, size order shuffled per round (run alone)
python run_e7.py --api-only    # API latency and throughput at 20,000 passports (run alone)
python analyze.py              # tables, figures and macros for the manuscript
```

Random seeds are fixed in the scripts. `OMP_NUM_THREADS` is set to 4 by default (`tsa/__init__.py`); timings depend on hardware. On the laptop used for the paper, identical runs differed by up to several-fold (probably because of CPU power management and core scheduling), which is why E7 reports medians and ranges over rounds.
