"""Persistence and API layer of the prototype.

* ``DppStore``: SQLite database holding the DPP core (passports, components,
  claims), an append-only lifecycle event store, observations with their
  metadata, analytical records (indicator results with uncertainty, data
  quality and provenance reference) and the analytical provenance graph.
* ``serve``: a minimal HTTP/JSON API (standard library) exposing passport
  retrieval, event history, indicator results, provenance lineage (explain),
  attribute search and event append. The endpoints are ILLUSTRATIVE of the
  lifecycle-management and searchability functions named in EN 18222; they do
  not implement the API specification of that standard.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS passport(batch_id TEXT PRIMARY KEY, dpp_uri TEXT, gtin TEXT, lot TEXT, category TEXT,
    brand TEXT, operator_id TEXT, manufacturer_facility TEXT, country TEXT, doc TEXT, status TEXT, version INTEGER);
CREATE TABLE IF NOT EXISTS event(seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE, batch_id TEXT,
    event_type TEXT, facility_id TEXT, time TEXT, inputs TEXT, outputs TEXT, record_hash TEXT, recorded_at REAL);
CREATE TABLE IF NOT EXISTS observation(obs_id TEXT PRIMARY KEY, batch_id TEXT, lot_id TEXT, stage TEXT,
    facility_id TEXT, quantity TEXT, value REAL, unit TEXT, data_type TEXT, verification TEXT, source_system TEXT,
    timestamp TEXT);
CREATE TABLE IF NOT EXISTS indicator(node_id TEXT PRIMARY KEY, batch_id TEXT, name TEXT, value REAL, unit TEXT,
    lo REAL, hi REAL, method TEXT, dq REAL, primary_share REAL, generated_at TEXT);
"""
INDEXES = """
CREATE INDEX IF NOT EXISTS ev_batch ON event(batch_id);
CREATE INDEX IF NOT EXISTS ob_batch ON observation(batch_id);
CREATE INDEX IF NOT EXISTS ind_batch ON indicator(batch_id);
CREATE INDEX IF NOT EXISTS pp_cat ON passport(category, country);
"""


class DppStore:
    def __init__(self, path: str):
        self.path = path
        con = sqlite3.connect(path)
        con.executescript(SCHEMA)
        con.close()

    def connect(self):
        con = sqlite3.connect(self.path, check_same_thread=False)
        con.execute("PRAGMA query_only=0")
        return con

    # --- bulk loading (timed) ---------------------------------------------------------
    def load_passports(self, view, docs: dict) -> float:
        d = view.dpp
        rows = [(b, u, g, l, c, br, op, mf, co, json.dumps(docs.get(b, {})), "active", 1) for b, u, g, l, c, br, op, mf, co in
                zip(d.batch_id, d.dpp_uri, d.gtin, d.lot, d.category, d.brand, d.operator_id, d.manufacturer_facility,
                    d.country_of_manufacture)]
        return self._insert("INSERT OR REPLACE INTO passport VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    def append_events(self, ev: pd.DataFrame, batch_size: int = 5000) -> float:
        now = time.time()
        rows = [(e, b, t, f, str(tm), i, o, h, now) for e, b, t, f, tm, i, o, h in
                zip(ev.event_id, ev.batch_id, ev.event_type, ev.facility_id, ev.time, ev.inputs, ev.outputs, ev.record_hash)]
        return self._insert("INSERT INTO event(event_id,batch_id,event_type,facility_id,time,inputs,outputs,record_hash,"
                            "recorded_at) VALUES (?,?,?,?,?,?,?,?,?)", rows, batch_size)

    def load_observations(self, O: pd.DataFrame, batch_size: int = 20000) -> float:
        rows = list(zip(O.obs_id, O.batch_id, O.lot_id, O.stage, O.facility_id, O.quantity,
                        [None if pd.isna(x) else float(x) for x in O.value], O.unit, O.data_type, O.verification,
                        O.source_system, O.timestamp.astype(str)))
        return self._insert("INSERT OR REPLACE INTO observation VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows, batch_size)

    def load_indicators(self, result, generated_at: str) -> float:
        I = result.indicators
        rows = []
        units = {"gwp_per_unit": "kg CO2e/unit", "energy_per_unit": "kWh/unit", "water_per_unit": "m3/unit",
                 "aware_per_unit": "m3 world-eq/unit"}
        for name, unit in units.items():
            lo = I.get(f"{name}_lo", pd.Series(np.nan, index=I.index))
            hi = I.get(f"{name}_hi", pd.Series(np.nan, index=I.index))
            rows += [(f"ind:{b}:{name}", b, name, float(v), unit, float(l), float(h), result.method.name, float(dq),
                      float(ps), generated_at) for b, v, l, h, dq, ps in
                     zip(I.batch_id, I[name], lo, hi, I.dq_score_gwp, I.primary_share_gwp)]
        return self._insert("INSERT OR REPLACE INTO indicator VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)

    def attach_provenance(self, N: pd.DataFrame, E: pd.DataFrame) -> float:
        from .provenance import ProvenanceStore
        ps = ProvenanceStore(self.path)
        t = ps.load(N, E)
        ps.con.close()
        return t

    def _insert(self, sql, rows, batch_size=10000) -> float:
        con = self.connect()
        t = time.perf_counter()
        for i in range(0, len(rows), batch_size):
            con.executemany(sql, rows[i:i + batch_size])
            con.commit()
        dt = time.perf_counter() - t
        con.executescript(INDEXES)
        con.close()
        return dt


# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    db_path = None
    local = threading.local()

    def log_message(self, *a):
        pass

    def _con(self):
        if not hasattr(self.local, "con"):
            self.local.con = sqlite3.connect(self.db_path, check_same_thread=False)
        return self.local.con

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        parts = [p for p in u.path.split("/") if p]
        con = self._con()
        try:
            if len(parts) == 2 and parts[0] == "dpp":
                r = con.execute("SELECT doc FROM passport WHERE batch_id=?", (parts[1],)).fetchone()
                return self._send(200, json.loads(r[0])) if r else self._send(404, {"error": "not found"})
            if len(parts) == 3 and parts[0] == "dpp" and parts[2] == "events":
                rows = con.execute("SELECT event_id,event_type,facility_id,time,inputs,outputs,record_hash FROM event "
                                   "WHERE batch_id=? ORDER BY seq", (parts[1],)).fetchall()
                return self._send(200, [dict(zip(("event_id", "type", "facility", "time", "inputs", "outputs", "hash"), r))
                                        for r in rows])
            if len(parts) == 3 and parts[0] == "dpp" and parts[2] == "indicators":
                rows = con.execute("SELECT node_id,name,value,unit,lo,hi,method,dq,primary_share,generated_at FROM indicator "
                                   "WHERE batch_id=?", (parts[1],)).fetchall()
                keys = ("id", "name", "value", "unit", "lower95", "upper95", "method", "dq", "primary_share", "generated_at")
                return self._send(200, [dict(zip(keys, r)) for r in rows])
            if len(parts) == 3 and parts[0] == "indicator" and parts[2] == "explain":
                q = """WITH RECURSIVE lin(node) AS (SELECT ? UNION SELECT e.dst FROM lin CROSS JOIN edges e ON e.src = lin.node)
                       SELECT n.node_id, n.kind, n.label, n.attrs FROM lin CROSS JOIN nodes n ON n.node_id = lin.node"""
                rows = con.execute(q, (parts[1],)).fetchall()
                return self._send(200, [dict(zip(("id", "kind", "label", "attrs"), r)) for r in rows])
            if parts == ["search"]:
                qs = parse_qs(u.query)
                cat, cty = qs.get("category", [None])[0], qs.get("country", [None])[0]
                maxg = float(qs.get("max_gwp", ["1e9"])[0])
                rows = con.execute("SELECT p.batch_id, p.dpp_uri, i.value FROM passport p JOIN indicator i ON "
                                   "i.batch_id = p.batch_id AND i.name='gwp_per_unit' WHERE (? IS NULL OR p.category=?) "
                                   "AND (? IS NULL OR p.country=?) AND i.value <= ? LIMIT 100",
                                   (cat, cat, cty, cty, maxg)).fetchall()
                return self._send(200, [dict(zip(("batch_id", "dpp", "gwp_per_unit"), r)) for r in rows])
            return self._send(404, {"error": "unknown endpoint"})
        except Exception as ex:  # pragma: no cover
            return self._send(500, {"error": str(ex)})

    def do_POST(self):
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if len(parts) == 3 and parts[0] == "dpp" and parts[2] == "events":
            con = self._con()
            con.execute("INSERT INTO event(event_id,batch_id,event_type,facility_id,time,inputs,outputs,record_hash,"
                        "recorded_at) VALUES (?,?,?,?,?,?,?,?,?)",
                        (body["event_id"], parts[1], body["event_type"], body.get("facility_id"), body.get("time"),
                         json.dumps(body.get("inputs", [])), json.dumps(body.get("outputs", [])), body.get("hash"), time.time()))
            con.commit()
            return self._send(201, {"status": "appended"})
        return self._send(404, {"error": "unknown endpoint"})


def serve(db_path: str, port: int = 0):
    handler = type("H", (_Handler,), {"db_path": db_path, "local": threading.local()})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv


def benchmark(port: int, paths: list[str], concurrency: int = 1, n_requests: int = 1000) -> dict:
    """Closed-loop load: ``concurrency`` clients with persistent connections."""
    import http.client
    from concurrent.futures import ThreadPoolExecutor
    lat = []
    lock = threading.Lock()
    per = n_requests // concurrency

    def client(k):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        local = []
        for i in range(per):
            p = paths[(k * per + i) % len(paths)]
            t = time.perf_counter()
            c.request("GET", p)
            r = c.getresponse()
            r.read()
            local.append(time.perf_counter() - t)
            if r.status != 200:
                raise RuntimeError(f"{p}: {r.status}")
        c.close()
        with lock:
            lat.extend(local)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as ex:
        list(ex.map(client, range(concurrency)))
    wall = time.perf_counter() - t0
    a = np.array(lat) * 1000
    return dict(concurrency=concurrency, n=len(a), p50_ms=float(np.percentile(a, 50)), p95_ms=float(np.percentile(a, 95)),
                p99_ms=float(np.percentile(a, 99)), mean_ms=float(a.mean()), throughput_rps=len(a) / wall)
