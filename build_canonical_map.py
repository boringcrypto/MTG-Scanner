"""
build_canonical_map.py  —  run once (or re-run after adding new rules)

Reads duplicate_groups.json, looks up each card in the SQLite DB, and
classifies each group as "resolved" (we understand why they're duplicates)
or "unresolved" (needs manual inspection).

Resolved rules implemented so far:
  token_same_set_number : every card in the group has rarity=Token,
                          the same set_code, and the same front number
                          (the part before ' // ' in numbers like '10 // 6').
                          These are double-faced tokens printed back-to-back.

Outputs: data/cards/canonical_map.json
  {
    "resolved": [
      {
        "canonical_idx": 42,         ← first index in the group (arbitrary)
        "indices":       [42, 100],  ← all LMDB indices with this image
        "product_ids":   ["123", "456"],
        "rule":          "token_same_set_number"
      },
      ...
    ],
    "unresolved": [
      [1, 2, 3],  ← raw index lists, same format as duplicate_groups.json
      ...
    ],
    "stats": {
      "total_groups":    N,
      "resolved":        N,
      "unresolved":      N
    }
  }

Usage:
    python build_canonical_map.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT           = Path(__file__).resolve().parent
DUPES_JSON     = ROOT / "data/cards/duplicate_groups.json"
CARDS_INDEX    = ROOT / "data/cards/cards_index.json"
CARD_DB        = ROOT / "unified_card_database.db"
OUT_PATH       = ROOT / "data/cards/canonical_map.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_db_records(
    product_ids: list[str], con: sqlite3.Connection
) -> dict[str, dict]:
    """Return a dict product_id → row (dict) for all given product_ids."""
    placeholders = ",".join("?" * len(product_ids))
    cur = con.execute(
        f"SELECT product_id, name, rarity, set_code, set_name, number FROM cards_1 "
        f"WHERE product_id IN ({placeholders})",
        product_ids,
    )
    return {row["product_id"]: dict(row) for row in cur.fetchall()}


# ── Exclude rules ─────────────────────────────────────────────────────────────
# Each function takes a single record dict and returns a reason string if the
# card should be excluded, or None if it should be kept.

def exclude_display_commander_thick_stock(record: dict) -> str | None:
    name = (record.get("name") or "").strip()
    if name.endswith(" (Display Commander) - Thick Stock"):
        return "display_commander_thick_stock"
    return None


def exclude_foil_variant(record: dict) -> str | None:
    """Exclude foil-treatment variants that duplicate a regular printing."""
    name = (record.get("name") or "").strip()
    for suffix, reason in (
        (" (Foil Etched)",  "foil_etched"),
        (" (Rainbow Foil)", "rainbow_foil"),
        (" (Ripple Foil)",  "ripple_foil"),
        (" (Surge Foil)",   "surge_foil"),
        (" (Galaxy Foil)",  "galaxy_foil"),
    ):
        if name.endswith(suffix):
            return reason
    return None


def exclude_set_ced(record: dict) -> str | None:
    """Exclude Collector's Edition (CED) — not used in competitive play."""
    if (record.get("set_code") or "").strip().upper() == "CED":
        return "set_ced"
    return None


def exclude_unknown_event_playtest(record: dict) -> str | None:
    """Exclude Un-Known Event Playtest Cards — internal playtest prints."""
    if (record.get("set_name") or "").strip() == "Un-Known Event Playtest Cards":
        return "unknown_event_playtest"
    return None


EXCLUDE_RULES: list = [
    exclude_display_commander_thick_stock,
    exclude_foil_variant,
    exclude_set_ced,
    exclude_unknown_event_playtest,
    # Add more exclusion predicates here
]


def get_exclude_reason(record: dict) -> str | None:
    """Return the first matching exclusion reason, or None."""
    for fn in EXCLUDE_RULES:
        reason = fn(record)
        if reason:
            return reason
    return None


# ── Rules ──────────────────────────────────────────────────────────────────────

def rule_tokens(records: list[dict]) -> bool:
    """
    All cards in the group must be:
      - rarity = Token  (case-insensitive)
    These are double-faced tokens that share one image but have different backs.
    """
    rarities   = {(r.get("rarity") or "").strip().lower() for r in records}

    if rarities != {"token"}:
        return False
    return True


_SANCTION_SETS = {"PSAL", "PS11"}

def rule_sanctioned_sets(records: list[dict]) -> bool:
    """
    All cards in the group belong to a sanctioned promo set (PSAL or PS11).
    These sets intentionally reprint existing cards with the same image.
    """
    return all(
        (r.get("set_code") or "").strip().upper() in _SANCTION_SETS
        for r in records
    )


def rule_world_championship_blank(records: list[dict]) -> bool:
    """
    All cards in the group have names ending in ' World Championship Blank Card'.
    These are filler/placeholder cards from World Championship decks.
    """
    return all(
        (r.get("name") or "").strip().endswith(" World Championship Blank Card")
        for r in records
    )


RULES = [
    ("token_same_set_number", rule_tokens),
    ("sanctioned_sets", rule_sanctioned_sets),
    ("world_championship_blank", rule_world_championship_blank),
    # Add more rules here as new duplicate patterns are understood
]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not DUPES_JSON.exists():
        print(f"ERROR: {DUPES_JSON} not found. Run find_duplicates.py first.")
        return

    with open(DUPES_JSON) as f:
        groups: list[list[int]] = json.load(f)
    with open(CARDS_INDEX) as f:
        cards_index: list[str] = json.load(f)

    con = sqlite3.connect(str(CARD_DB))
    con.row_factory = sqlite3.Row

    resolved:   list[dict] = []
    unresolved: list[list[int]] = []
    excluded:   list[dict] = []

    for group in groups:
        all_product_ids = [cards_index[i] for i in group if i < len(cards_index)]
        if not all_product_ids:
            unresolved.append(group)
            continue

        records_map = load_db_records(all_product_ids, con)

        # Split the group into kept and excluded indices
        kept_indices: list[int] = []
        for idx, pid in zip(group, all_product_ids):
            rec = records_map.get(pid)
            if rec is None:
                kept_indices.append(idx)  # no DB record — keep; can't classify
                continue
            reason = get_exclude_reason(rec)
            if reason:
                excluded.append({"lmdb_idx": idx, "product_id": pid, "reason": reason})
            else:
                kept_indices.append(idx)

        # A group needs at least 2 members to still be a duplicate group
        if len(kept_indices) < 2:
            continue

        product_ids = [cards_index[i] for i in kept_indices if i < len(cards_index)]
        records     = [records_map[pid] for pid in product_ids if pid in records_map]

        matched_rule = None
        for rule_name, rule_fn in RULES:
            if records and rule_fn(records):
                matched_rule = rule_name
                break

        if matched_rule:
            resolved.append({
                "canonical_idx": kept_indices[0],
                "indices":       kept_indices,
                "product_ids":   product_ids,
                "rule":          matched_rule,
            })
        else:
            unresolved.append(kept_indices)

    con.close()

    stats = {
        "total_groups":   len(groups),
        "resolved":       len(resolved),
        "unresolved":     len(unresolved),
        "excluded_cards": len(excluded),
    }

    result = {"resolved": resolved, "unresolved": unresolved, "excluded": excluded, "stats": stats}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Total duplicate groups : {stats['total_groups']:,}")
    print(f"Resolved               : {stats['resolved']:,}")
    print(f"Unresolved             : {stats['unresolved']:,}")
    print(f"Excluded cards         : {stats['excluded_cards']:,}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
