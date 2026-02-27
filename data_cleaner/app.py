"""
data_cleaner/app.py

API
---
GET /api/cards/count              → { "count": N }
GET /api/card/<idx>/image         → JPEG image bytes (from cards.lmdb)
GET /api/card/<idx>/info          → card metadata from unified_card_database.db
GET /api/duplicates               → list-of-lists of duplicate LMDB indices
GET /api/sets                     → all sets with canonical_card_count
GET /api/set/<set_id>             → single set metadata row
GET /api/set/<set_id>/cards       → canonical cards belonging to that set
"""

from __future__ import annotations

import io
import json
import sqlite3
import lmdb
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, send_file, jsonify, send_from_directory, abort

# ── Paths (resolved relative to this file) ────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
CARDS_LMDB  = ROOT / "data/cards/cards.lmdb"
CARDS_INDEX = ROOT / "data/cards/cards_index.json"
CARD_DB     = ROOT / "unified_card_database.db"
DUPES_JSON       = ROOT / "data/cards/duplicate_groups.json"
CANONICAL_MAP    = ROOT / "data/cards/canonical_map.json"
CANONICAL_INDEX  = ROOT / "data/cards/canonical_index.json"

app = Flask(__name__, static_folder="static", static_url_path="")

# ── LMDB ──────────────────────────────────────────────────────────────────────
_env: lmdb.Environment | None = None

def get_env() -> lmdb.Environment:
    global _env
    if _env is None:
        _env = lmdb.open(str(CARDS_LMDB), readonly=True, lock=False,
                         readahead=False, meminit=False)
    return _env

def get_count() -> int:
    with get_env().begin() as txn:
        return int(txn.get(b"__len__").decode())

# ── cards_index: list[str]  — position i → product_id string ─────────────────
_cards_index: list[str] | None = None

def get_cards_index() -> list[str]:
    global _cards_index
    if _cards_index is None:
        with open(CARDS_INDEX) as f:
            _cards_index = json.load(f)
    return _cards_index

# ── SQLite ─────────────────────────────────────────────────────────────────────
_db_con: sqlite3.Connection | None = None

def get_db() -> sqlite3.Connection:
    global _db_con
    if _db_con is None:
        _db_con = sqlite3.connect(str(CARD_DB), check_same_thread=False)
        _db_con.row_factory = sqlite3.Row
    return _db_con

# Columns we actually want to surface in the API
_INFO_COLS = (
    "product_id", "name", "set_name", "set_code", "number", "rarity",
    "type", "card_type", "full_type", "color", "converted_cost",
    "power", "toughness", "description", "flavor_text",
    "market_price", "mid_price", "low_price",
    "image_url", "duplicate", "score",
)

def _lookup_card(product_id: str) -> dict | None:
    cur = get_db().cursor()
    cols = ", ".join(_INFO_COLS)
    cur.execute(f"SELECT {cols} FROM cards_1 WHERE product_id = ?", (product_id,))
    row = cur.fetchone()
    if row is None:
        return None
    return dict(row)

# ── Duplicate groups ──────────────────────────────────────────────────────────
_dupes_cache: list[list[int]] | None = None

def get_duplicates() -> list[list[int]]:
    global _dupes_cache
    if _dupes_cache is None:
        if not DUPES_JSON.exists():
            _dupes_cache = []
        else:
            with open(DUPES_JSON) as f:
                _dupes_cache = json.load(f)
    return _dupes_cache

# ── Canonical map ──────────────────────────────────────────────────────────────
_canonical_cache: dict | None = None

def get_canonical_map() -> dict:
    global _canonical_cache
    if _canonical_cache is None:
        if not CANONICAL_MAP.exists():
            _canonical_cache = {"resolved": [], "unresolved": get_duplicates(),
                                "stats": {"total_groups": len(get_duplicates()),
                                          "resolved": 0, "unresolved": len(get_duplicates())}}
        else:
            with open(CANONICAL_MAP) as f:
                _canonical_cache = json.load(f)
    return _canonical_cache


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/api/cards/count")
def api_count():
    return jsonify(count=get_count())


@app.get("/api/card/<int:idx>/image")
def api_card_image(idx: int):
    n = get_count()
    if idx < 0 or idx >= n:
        abort(404, description=f"Card index {idx} out of range [0, {n})")
    with get_env().begin() as txn:
        data = txn.get(str(idx).encode())
    if data is None:
        abort(404, description=f"Card {idx} not found in LMDB")
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


@app.get("/api/duplicates")
def api_duplicates():
    cm = get_canonical_map()
    return jsonify(
        unresolved=cm["unresolved"],
        resolved=cm["resolved"],
        stats=cm["stats"],
    )


@app.get("/api/card/<int:idx>/info")
def api_card_info(idx: int):
    index = get_cards_index()
    n = get_count()
    if idx < 0 or idx >= n:
        abort(404, description=f"Card index {idx} out of range [0, {n})")
    if idx >= len(index):
        abort(404, description=f"Card index {idx} not in cards_index.json")
    product_id = index[idx]
    info = _lookup_card(product_id)
    if info is None:
        return jsonify(index=idx, product_id=product_id, found=False)
    return jsonify(index=idx, found=True, **info)


@app.get("/api/card/by-product-id/<product_id>")
def api_card_by_product_id(product_id: str):
    index = get_cards_index()
    try:
        idx = index.index(product_id)
    except ValueError:
        abort(404, description=f"product_id {product_id!r} not found in cards_index.json")
    return jsonify(index=idx, product_id=product_id)


# ── Canonical card index ──────────────────────────────────────────────────────
# Canonical = one representative lmdb index per unique image, with excluded
# cards removed.  Built once from canonical_map.json on first access.

_canonical_meta: dict[int, dict] | None = None   # lmdb_idx → card metadata


def _build_canonical_meta() -> dict[int, dict]:
    index = get_cards_index()

    # Load canonical index from pre-built JSON (produced by build_canonical_index.py)
    if CANONICAL_INDEX.exists():
        with open(CANONICAL_INDEX) as f:
            canonical: set[int] = set(json.load(f))
    else:
        # Fallback: compute on the fly from canonical_map (slower)
        cm  = get_canonical_map()
        n   = get_count()
        excluded_set: set[int] = {e["lmdb_idx"] for e in cm.get("excluded", [])}
        all_grouped:  set[int] = set()
        canonical = set()
        for grp in cm.get("resolved", []):
            all_grouped.update(grp["indices"])
            ci = grp["canonical_idx"]
            if ci not in excluded_set:
                canonical.add(ci)
        for grp in cm.get("unresolved", []):
            all_grouped.update(grp)
            for idx in grp:
                if idx not in excluded_set:
                    canonical.add(idx)
                    break
        for i in range(n):
            if i not in all_grouped and i not in excluded_set:
                canonical.add(i)
        canonical -= excluded_set

    # Batch-lookup card metadata (SQLite variable limit is 999)
    product_ids = [index[i] for i in canonical if i < len(index)]
    if not product_ids:
        return {}

    db      = get_db()
    pid_map: dict[str, dict] = {}
    chunk   = 900
    for start in range(0, len(product_ids), chunk):
        batch = product_ids[start : start + chunk]
        phs   = ",".join("?" * len(batch))
        rows  = db.execute(
            f"SELECT product_id, name, set_id, set_code, number, rarity, card_type "
            f"FROM cards_1 WHERE product_id IN ({phs})",
            batch,
        ).fetchall()
        for r in rows:
            pid_map[r["product_id"]] = dict(r)

    meta: dict[int, dict] = {}
    for i in canonical:
        if i >= len(index):
            continue
        pid = index[i]
        m   = pid_map.get(pid, {})
        meta[i] = {
            "product_id": pid,
            "name":       m.get("name"),
            "set_id":     m.get("set_id"),
            "set_code":   m.get("set_code"),
            "number":     m.get("number"),
            "rarity":     m.get("rarity"),
            "card_type":  m.get("card_type"),
        }
    return meta


def get_canonical_meta() -> dict[int, dict]:
    global _canonical_meta
    if _canonical_meta is None:
        _canonical_meta = _build_canonical_meta()
    return _canonical_meta


def _sort_key(number: str | None):
    """Sort card numbers numerically by front face (before ' // ')."""
    front = (number or "").split("//")[0].strip()
    try:
        return (0, int(front), "")
    except ValueError:
        return (1, 0, front)


# ── Sets API ──────────────────────────────────────────────────────────────────

@app.get("/api/sets")
def api_sets():
    meta = get_canonical_meta()
    db   = get_db()

    counts: dict[str, int] = {}
    for m in meta.values():
        sid = m.get("set_id")
        if sid:
            counts[sid] = counts.get(sid, 0) + 1

    if not counts:
        return jsonify([])

    phs  = ",".join("?" * len(counts))
    rows = db.execute(
        f"SELECT id, name, set_name, set_code, set_id, series, "
        f"       total_cards, release_date "
        f"FROM sets WHERE set_id IN ({phs})",
        list(counts.keys()),
    ).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        d["canonical_card_count"] = counts.get(row["set_id"], 0)
        result.append(d)

    result.sort(key=lambda s: s.get("name") or "")
    return jsonify(result)


@app.get("/api/set/<set_id>")
def api_set_info(set_id: str):
    db  = get_db()
    row = db.execute(
        "SELECT id, name, set_name, set_code, set_id, series, "
        "       total_cards, release_date "
        "FROM sets WHERE set_id = ?",
        (set_id,),
    ).fetchone()
    if row is None:
        abort(404, description=f"Set {set_id!r} not found")
    meta   = get_canonical_meta()
    d      = dict(row)
    d["canonical_card_count"] = sum(
        1 for m in meta.values() if m.get("set_id") == set_id
    )
    return jsonify(d)


@app.get("/api/set/<set_id>/cards")
def api_set_cards(set_id: str):
    meta  = get_canonical_meta()
    cards = [
        {"lmdb_idx": idx, **m}
        for idx, m in meta.items()
        if m.get("set_id") == set_id
    ]
    cards.sort(key=lambda c: _sort_key(c.get("number")))
    return jsonify(cards)


# ── SPA fallback — serve index.html for any non-API route ────────────────────

@app.get("/", defaults={"path": ""})
@app.get("/<path:path>")
def spa(path: str):
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    print(f"Cards LMDB : {CARDS_LMDB.resolve()}")
    print(f"Card DB    : {CARD_DB.resolve()}")
    print(f"Cards      : {get_count():,}")
    print(f"Index size : {len(get_cards_index()):,}")
    print("Open       : http://localhost:5000")
    app.run(debug=True, port=5000)
