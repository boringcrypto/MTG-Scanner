"""
build_canonical_index.py  —  run after build_canonical_map.py

Reads canonical_map.json and cards_224.lmdb (for total count), then
produces a flat sorted list of canonical LMDB indices and writes it to
data/cards/canonical_index.json.

This is the single source of truth for "which cards have unique images
and are not excluded".  All analysis scripts and the web app load this
file rather than recomputing the list themselves.

Rules:
  - resolved duplicate groups  → canonical_idx only
  - unresolved duplicate groups → first non-excluded index in the group
  - singletons (not in any group, not excluded) → kept as-is
  - excluded indices → removed entirely

Usage:
    python build_canonical_index.py
"""

from __future__ import annotations

import json
import lmdb
from pathlib import Path

ROOT            = Path(__file__).resolve().parent
CANONICAL_MAP   = ROOT / "data/cards/canonical_map.json"
CARDS_224       = ROOT / "data/cards/cards_224.lmdb"
OUT_PATH        = ROOT / "data/cards/canonical_index.json"


def main():
    if not CANONICAL_MAP.exists():
        print(f"ERROR: {CANONICAL_MAP} not found.  Run build_canonical_map.py first.")
        return

    # Total card count from LMDB
    env = lmdb.open(str(CARDS_224), readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        n_total = int(txn.get(b"__len__").decode())
    env.close()
    print(f"Total cards in DB : {n_total:,}")

    with open(CANONICAL_MAP) as f:
        cm = json.load(f)

    excluded_set: set[int] = {e["lmdb_idx"] for e in cm.get("excluded", [])}
    all_grouped:  set[int] = set()
    canonical:    set[int] = set()

    for grp in cm.get("resolved", []):
        all_grouped.update(grp["indices"])
        ci = grp["canonical_idx"]
        if ci not in excluded_set:
            canonical.add(ci)

    for grp in cm.get("unresolved", []):
        all_grouped.update(grp)
        for idx in grp:          # first non-excluded wins
            if idx not in excluded_set:
                canonical.add(idx)
                break

    for i in range(n_total):     # singletons
        if i not in all_grouped and i not in excluded_set:
            canonical.add(i)

    canonical -= excluded_set    # safety pass

    result = sorted(canonical)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f)

    n_excluded   = len(excluded_set)
    n_deduped    = n_total - len(excluded_set) - len(result) + len(
        excluded_set & set(range(n_total))  # rough count of dupes removed
    )
    print(f"Canonical images  : {len(result):,}")
    print(f"Excluded cards    : {n_excluded:,}")
    print(f"Deduplicated      : {n_total - n_excluded - len(result):,}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
