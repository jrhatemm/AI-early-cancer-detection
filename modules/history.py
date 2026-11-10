"""Anonymised scan history in SQLite (stores a hash of the filename, never the name)."""
import hashlib
import sqlite3

from config import settings as cfg

SCHEMA = """CREATE TABLE IF NOT EXISTS scans(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, source_hash TEXT, cancer TEXT, model TEXT,
    trained INTEGER, level TEXT, score REAL, confidence REAL, regions INTEGER)"""


def _conn():
    cfg.HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(cfg.HISTORY_DB)
    c.row_factory = sqlite3.Row
    c.execute(SCHEMA)
    return c


def log_scan(result):
    a = result["assessment"]
    h = hashlib.sha256(str(result["source"]).encode()).hexdigest()[:12]
    with _conn() as c:
        c.execute("INSERT INTO scans(ts,source_hash,cancer,model,trained,level,score,confidence,regions)"
                  " VALUES (?,?,?,?,?,?,?,?,?)",
                  (result["timestamp"], h, result["cancer"], result["model"], int(result["trained"]),
                   a["level"], a["score"], a["confidence"], len(result["regions"])))


def recent(limit=20):
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,))]


def stats():
    with _conn() as c:
        rows = c.execute("SELECT cancer, level, COUNT(*) n FROM scans GROUP BY cancer, level").fetchall()
    out = {}
    for r in rows:
        out.setdefault(r["cancer"], {})[r["level"]] = r["n"]
    return out
