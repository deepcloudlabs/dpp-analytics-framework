"""Data acquisition layer: heterogeneous source extracts and declarative mapping.

Four source formats are emulated from the observed passport data:

* MES      - JSON lines per process order, nested measurements, local time
             with UTC offsets, mixed energy units (kWh/MWh/Wh, MJ/GJ/kWh);
* IoT      - interval (15-min) water-meter increments in litres, to be
             aggregated over the lot processing window;
* SUPPLIER - spreadsheets with multilingual headers (EN/DE/TR) and locale
             decimal formats (1,234.5 vs 1.234,5);
* ERP      - semicolon-separated material movements with UN/ECE Rec. 20
             unit codes (KGM, TNE).

Each format is mapped to canonical observations by a declarative mapping
specification. Records that cannot be mapped (unknown unit, unparsable value,
missing key) are quarantined, never silently dropped. The evaluation compares
the mapped values with the canonical values of the originating observations.
"""
from __future__ import annotations

import csv
import io
import json
import re

import numpy as np
import pandas as pd

UNIT_ALIASES = {"kwh": "kWh", "mwh": "MWh", "wh": "Wh", "mj": "MJ", "gj": "GJ", "m3": "m3", "m³": "m3", "l": "L",
                "kg": "kg", "t": "t", "KGM": "kg", "TNE": "t", "KWH": "kWh", "MTQ": "m3", "LTR": "L"}
TO_CANON = {("electricity", "kWh"): 1, ("electricity", "MWh"): 1e3, ("electricity", "Wh"): 1e-3,
            ("heat", "MJ"): 1, ("heat", "GJ"): 1e3, ("heat", "kWh"): 3.6,
            ("water", "m3"): 1, ("water", "L"): 1e-3, ("chemicals", "kg"): 1, ("chemicals", "t"): 1e3,
            ("waste", "kg"): 1, ("waste", "t"): 1e3}

MAPPINGS = {
    "SUPPLIER": {  # header aliases -> canonical field; unit extracted from header brackets
        "record": ["Record", "Satz-ID", "Kayit No"],
        "lot": ["Lot", "Charge", "Parti No"],
        "electricity": ["Electricity", "Strom", "Elektrik"],
        "heat": ["Fuel energy", "Brennstoffenergie", "Yakit enerjisi"],
        "water": ["Water", "Wasser", "Su"],
    },
    "ERP": {"record": "DOC", "lot": "LOT", "quantity": {"CHEM": "chemicals", "WASTE": "waste"}, "value": "QTY", "unit": "UOM"},
    "MES": {"record": "rec", "lot": "order", "time": "ts", "measurements": "meas",
            "kind": {"EL": "electricity", "TH": "heat"}, "value": "val", "unit": "u"},
    "IOT": {"record": "rec", "lot": "lot", "time": "ts", "increment": "inc_l", "quantity": "water", "unit": "L"},
}


def _fmt(x: float, locale: str) -> str:
    s = f"{x:,.12g}" if abs(x) >= 1e-3 else f"{x:.12e}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".") if locale == "de" else s


def export_sources(canon: pd.DataFrame, seed: int = 0, fault_rate: float = 0.01):
    """Create source extracts from canonical observations (non-audit, reported)."""
    rng = np.random.default_rng(seed)
    O = canon[(canon.source_system != "audit") & canon.value_c.notna()].copy()
    faults = set()

    def maybe_fault(rec_id):
        if rng.random() < fault_rate:
            faults.add(rec_id)
            return rng.choice(["unit", "value", "key"])
        return None

    # --- MES: electricity/heat (non-estimated) -----------------------------------------------
    mes = O[O.quantity.isin(["electricity", "heat"]) & (O.data_type != "estimated")]
    lines = []
    for r in mes.itertuples():
        f = maybe_fault(r.obs_id)
        unit = r.unit
        val = r.value
        kind = "EL" if r.quantity == "electricity" else "TH"
        off = int(rng.choice([0, 3, 5, 6, 7, 8]))
        ts = (pd.Timestamp(r.timestamp) + pd.Timedelta(hours=off)).strftime("%Y-%m-%dT%H:%M:%S") + f"+{off:02d}:00"
        rec = {"rec": r.obs_id, "order": r.lot_id, "ts": ts, "meas": [{"kind": kind, "val": val, "u": unit}]}
        if f == "unit":
            rec["meas"][0]["u"] = unit + "h"
        elif f == "value":
            rec["meas"][0]["val"] = "n/a"
        elif f == "key":
            rec.pop("order")
        lines.append(json.dumps(rec))
    # --- IoT: water measured -> 15-min increments ----------------------------------------------
    iot = O[(O.quantity == "water") & (O.data_type == "measured")]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["rec", "lot", "ts", "inc_l"])
    for r in iot.itertuples():
        f = maybe_fault(r.obs_id)
        f = "value" if f == "unit" else f          # interval files carry no unit field
        n = int(rng.integers(16, 48))
        litres = r.value_c * 1000.0
        parts = rng.dirichlet(np.ones(n)) * litres
        parts = np.round(parts, 6)
        parts[-1] = litres - parts[:-1].sum()
        t0 = pd.Timestamp(r.timestamp)
        for k, p in enumerate(parts):
            val = "x" if (f == "value" and k == 0) else f"{p:.6f}"
            lot = "" if f == "key" else r.lot_id
            w.writerow([r.obs_id, lot, (t0 + pd.Timedelta(minutes=15 * k)).isoformat(), val])
    iot_csv = buf.getvalue()
    # --- SUPPLIER: estimated values, multilingual headers and locales ------------------------------
    sup = O[O.quantity.isin(["electricity", "heat", "water"]) & (O.data_type == "estimated")]
    sheets = []
    for fac, g in sup.groupby("facility_id"):
        lang = ["EN", "DE", "TR"][int(rng.integers(0, 3))]
        loc = "de" if lang in ("DE", "TR") else "en"
        k = {"EN": 0, "DE": 1, "TR": 2}[lang]
        M = MAPPINGS["SUPPLIER"]
        buf = io.StringIO()
        w = csv.writer(buf, delimiter=";" if loc == "de" else ",")
        w.writerow([M["record"][k], M["lot"][k], "quantity_header", "value"])
        for r in g.itertuples():
            f = maybe_fault(r.obs_id)
            hdr = f"{M[r.quantity][k]} [{r.unit}]" if f != "unit" else f"{M[r.quantity][k]} [{r.unit}x]"
            val = "?" if f == "value" else _fmt(r.value, loc)
            w.writerow([r.obs_id, "" if f == "key" else r.lot_id, hdr, val])
        sheets.append((lang, loc, buf.getvalue()))
    # --- ERP: chemicals and waste ------------------------------------------------------------------
    erp = O[O.quantity.isin(["chemicals", "waste"])]
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(["DOC", "LOT", "TYPE", "QTY", "UOM"])
    for r in erp.itertuples():
        f = maybe_fault(r.obs_id)
        uom = {"kg": "KGM", "t": "TNE"}[r.unit]
        w.writerow([r.obs_id, "" if f == "key" else r.lot_id, "CHEM" if r.quantity == "chemicals" else "WASTE",
                    "#VALUE!" if f == "value" else f"{r.value:.12g}", "KGX" if f == "unit" else uom])
    return dict(MES="\n".join(lines), IOT=iot_csv, SUPPLIER=sheets, ERP=buf.getvalue()), faults


def _num(s: str, loc: str) -> float:
    s = s.strip()
    if loc == "de":
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    return float(s)


def ingest(sources) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map all source extracts to canonical observations; return (records, quarantine)."""
    out, quar = [], []

    def emit(rec, lot, q, value, unit, src):
        u = UNIT_ALIASES.get(unit, UNIT_ALIASES.get(unit.lower() if isinstance(unit, str) else unit, unit))
        if not lot:
            quar.append((rec, src, "missing key"))
            return
        if (q, u) not in TO_CANON:
            quar.append((rec, src, f"unknown unit {unit}"))
            return
        if value is None or not np.isfinite(value):
            quar.append((rec, src, "unparsable value"))
            return
        out.append((rec, lot, q, value * TO_CANON[(q, u)], src))

    M = MAPPINGS["MES"]
    for line in sources["MES"].splitlines():
        r = json.loads(line)
        for m in r[M["measurements"]]:
            try:
                val = float(m[M["value"]])
            except (TypeError, ValueError):
                val = None
            emit(r[M["record"]], r.get(M["lot"]), M["kind"][m["kind"]], val, m[M["unit"]], "MES")
    iot = pd.read_csv(io.StringIO(sources["IOT"]), dtype=str, keep_default_na=False)
    for rec, g in iot.groupby("rec"):
        vals = pd.to_numeric(g.inc_l, errors="coerce")
        lot = g.lot.iat[0]
        val = float(vals.sum()) if vals.notna().all() else None
        emit(rec, lot, "water", val, "L", "IOT")
    S = MAPPINGS["SUPPLIER"]
    rev = {alias: q for q, aliases in S.items() if q not in ("record", "lot") for alias in aliases}
    for lang, loc, text in sources["SUPPLIER"]:
        rd = csv.reader(io.StringIO(text), delimiter=";" if loc == "de" else ",")
        next(rd)
        for rec, lot, hdr, val in rd:
            m = re.match(r"^(.*?)\s*\[(.+)\]$", hdr)
            q = rev.get(m.group(1)) if m else None
            try:
                v = _num(val, loc)
            except ValueError:
                v = None
            if q is None:
                quar.append((rec, "SUPPLIER", "unknown header"))
                continue
            emit(rec, lot, q, v, m.group(2), "SUPPLIER")
    E = MAPPINGS["ERP"]
    rd = csv.DictReader(io.StringIO(sources["ERP"]), delimiter=";")
    for r in rd:
        try:
            v = float(r[E["value"]])
        except ValueError:
            v = None
        emit(r[E["record"]], r[E["lot"]], E["quantity"][r["TYPE"]], v, r[E["unit"]], "ERP")
    R = pd.DataFrame(out, columns=["obs_id", "lot_id", "quantity", "value_c", "source"])
    Q = pd.DataFrame(quar, columns=["obs_id", "source", "reason"])
    return R, Q
