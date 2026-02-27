"""
find_duplicates.py  —  one-off script

Finds all groups of exactly identical images in cards_224.lmdb by hashing
the raw stored bytes (MD5). No model, no embeddings, no threshold needed.

Outputs:  data/cards/duplicate_groups.json
    A list of lists, each inner list contains the LMDB indices (0-based) of
    cards that share the exact same image (size >= 2).

Usage:
    python find_duplicates.py
    python find_duplicates.py --out data/cards/duplicate_groups.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lmdb
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

CARDS_224 = "data/cards/cards_224.lmdb"
OUT_PATH  = "data/cards/duplicate_groups.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=OUT_PATH)
    ap.add_argument("--db",  type=str, default=CARDS_224)
    args = ap.parse_args()

    env = lmdb.open(args.db, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        N = int(txn.get(b"__len__").decode())
    print(f"Cards: {N:,}")

    hash_to_indices: dict[str, list[int]] = defaultdict(list)

    with env.begin() as txn:
        for i in tqdm(range(N), desc="Hashing", unit="img"):
            data = txn.get(str(i).encode())
            h = hashlib.md5(data).hexdigest()
            hash_to_indices[h].append(i)
    env.close()

    groups = sorted(
        [indices for indices in hash_to_indices.values() if len(indices) >= 2],
        key=lambda g: (-len(g), g[0]),
    )

    total_dupes = sum(len(g) for g in groups)
    print(f"\nDuplicate groups : {len(groups):,}")
    print(f"Cards in groups  : {total_dupes:,} / {N:,}  ({total_dupes/N:.1%})")

    print("\nTop 10 largest groups:")
    for g in groups[:10]:
        print(f"  size={len(g)}  indices={g[:8]}{'…' if len(g) > 8 else ''}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(groups, f)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

