"""
build_phash_neighbors.py
------------------------
Compute DCT-based perceptual-hash (pHash) for every card in cards_224.lmdb
and save a neighbour map: for each card, all other cards within HAMMING_THRESHOLD
bits, capped at MAX_NEIGHBORS (sorted by distance ascending).

Output
------
  data/cards/phash_neighbors.npy   – dict  {lmdb_id: np.ndarray of neighbor lmdb_ids}
                                    load with: np.load(..., allow_pickle=True).item()

Usage
-----
    python build_phash_neighbors.py
    python build_phash_neighbors.py --threshold 10 --max-neighbors 64
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import imagehash
import lmdb
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from card_loader import _CARDS_224, _CANONICAL_INDEX, Card224Loader

# ── Config ────────────────────────────────────────────────────────────────────
HAMMING_THRESHOLD = 6    # greyscale pHash threshold (first pass)
RGB_THRESHOLD     = 6    # per-channel RGB pHash average threshold (second pass)
DHASH_THRESHOLD   = 10   # blurred dHash threshold (third pass)
MAX_NEIGHBORS     = 32   # cap per card to avoid basic-land clusters exploding
OUTPUT_PATH       = Path("data/cards/phash_neighbors.npy")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_all_raw(lmdb_path: str, ids: list) -> list[bytes]:
    """Read raw JPEG bytes for each lmdb_id in one sequential LMDB scan."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False,
                    readahead=True, meminit=False)
    out = []
    with env.begin(buffers=True) as txn:
        for lmdb_id in tqdm(ids, desc="  reading LMDB", leave=False):
            raw = txn.get(str(lmdb_id).encode())
            out.append(bytes(raw))
    env.close()
    return out


def _compute_phashes(raw_jpegs: list[bytes], workers: int = 8) -> np.ndarray:
    """
    Compute 64-bit greyscale pHash for each JPEG. Returns (N, 8) uint8.
    """
    def _one(raw: bytes) -> np.ndarray:
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        return _phash_bgr(bgr)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        hashes = list(tqdm(ex.map(_one, raw_jpegs),
                           total=len(raw_jpegs), desc="  computing pHash", leave=False))
    return np.stack(hashes)                   # (N, 8) uint8


def _compute_rgb_phashes(raw_jpegs: list[bytes], workers: int = 8) -> np.ndarray:
    """
    Compute per-channel R/G/B pHash for each JPEG.
    Returns (N, 24) uint8: [R_8bytes | G_8bytes | B_8bytes] per row.
    """
    def _one(raw: bytes) -> np.ndarray:
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        r, g, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        parts = []
        for ch in (r, g, b):
            h = imagehash.phash(Image.fromarray(ch))
            parts.append(np.packbits(h.hash.flatten().astype(np.uint8)))
        return np.concatenate(parts)          # (24,) uint8

    with ThreadPoolExecutor(max_workers=workers) as ex:
        hashes = list(tqdm(ex.map(_one, raw_jpegs),
                           total=len(raw_jpegs), desc="  computing RGB pHash", leave=False))
    return np.stack(hashes)                   # (N, 24) uint8


def _compute_dhashes(raw_jpegs: list[bytes], workers: int = 8) -> np.ndarray:
    """
    Compute 64-bit blurred dHash for each JPEG. Returns (N, 8) uint8.
    """
    def _one(raw: bytes) -> np.ndarray:
        bgr     = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        return _dhash_bgr(bgr)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        hashes = list(tqdm(ex.map(_one, raw_jpegs),
                           total=len(raw_jpegs), desc="  computing dHash", leave=False))
    return np.stack(hashes)                   # (N, 8) uint8


def _build_neighbors(packed: np.ndarray, rgb_packed: np.ndarray, d_packed: np.ndarray,
                     ids: list, threshold: int, rgb_threshold: int, d_threshold: int,
                     max_nb: int, chunk: int = 2048) -> dict:
    """
    Three-pass GPU neighbor search:
      Pass 1: greyscale pHash Hamming ≤ threshold
      Pass 2: mean R/G/B pHash Hamming ≤ rgb_threshold
      Pass 3: blurred dHash Hamming ≤ d_threshold
    """
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N        = len(ids)
    ids_arr  = np.array(ids)
    result   = {}

    g_all    = torch.from_numpy(np.unpackbits(packed,     axis=1).astype(np.float16)).to(device)  # (N, 64)
    rgb_all  = torch.from_numpy(np.unpackbits(rgb_packed, axis=1).astype(np.float16)).to(device)  # (N, 192)
    d_all    = torch.from_numpy(np.unpackbits(d_packed,   axis=1).astype(np.float16)).to(device)  # (N, 64)

    g_bits   = g_all.sum(1)
    rgb_bits = rgb_all.sum(1)
    d_bits   = d_all.sum(1)

    for start in tqdm(range(0, N, chunk), desc="  building neighbors", leave=False):
        g_block   = g_all[start:start + chunk]
        rgb_block = rgb_all[start:start + chunk]
        d_block   = d_all[start:start + chunk]
        c         = g_block.size(0)

        g_dists   = (g_bits[start:start+c].unsqueeze(1) + g_bits.unsqueeze(0)
                     - 2 * (g_block @ g_all.T)).short().cpu().numpy()          # (C, N)
        rgb_dists = ((rgb_bits[start:start+c].unsqueeze(1) + rgb_bits.unsqueeze(0)
                      - 2 * (rgb_block @ rgb_all.T)) / 3.0).cpu().numpy()     # (C, N)
        d_dists   = (d_bits[start:start+c].unsqueeze(1) + d_bits.unsqueeze(0)
                     - 2 * (d_block @ d_all.T)).short().cpu().numpy()          # (C, N)

        for j in range(c):
            global_i = start + j
            g_row    = g_dists[j].copy()
            rgb_row  = rgb_dists[j].copy()
            d_row    = d_dists[j].copy()
            g_row[global_i] = rgb_row[global_i] = d_row[global_i] = 9999

            nb_pos = np.where(
                (g_row <= threshold) & (rgb_row <= rgb_threshold) & (d_row <= d_threshold)
            )[0]
            if len(nb_pos) > max_nb:
                nb_pos = nb_pos[np.argsort(g_row[nb_pos])[:max_nb]]
            result[ids_arr[global_i].item()] = ids_arr[nb_pos]

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def build(threshold: int = HAMMING_THRESHOLD,
          rgb_threshold: int = RGB_THRESHOLD,
          d_threshold: int = DHASH_THRESHOLD,
          max_neighbors: int = MAX_NEIGHBORS,
          output: Path = OUTPUT_PATH) -> dict:
    loader = Card224Loader()
    ids    = loader.all_indices

    print(f"Cards : {len(ids):,}")
    print(f"Thresholds : grey≤{threshold}  rgb≤{rgb_threshold}  dhash≤{d_threshold}  max_nb={max_neighbors}")

    t0         = time.perf_counter()
    raw        = _read_all_raw(_CARDS_224, ids)
    packed     = _compute_phashes(raw)
    rgb_packed = _compute_rgb_phashes(raw)
    d_packed   = _compute_dhashes(raw)
    print(f"  hashing done ({time.perf_counter()-t0:.1f}s)")

    t1        = time.perf_counter()
    neighbors = _build_neighbors(packed, rgb_packed, d_packed, ids,
                                 threshold, rgb_threshold, d_threshold, max_neighbors)
    print(f"  Neighbor build done ({time.perf_counter()-t1:.1f}s)")

    # Stats
    counts = np.array([len(v) for v in neighbors.values()])
    print(f"  Neighbor counts — mean={counts.mean():.1f}  "
          f"median={int(np.median(counts))}  "
          f"max={counts.max()}  "
          f"cards_with_0={int((counts == 0).sum())}")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, neighbors)
    print(f"Saved → {output}  ({output.stat().st_size / 1024:.0f} KB)")

    return neighbors


def _load_image(lmdb_path: str, lmdb_id: int) -> np.ndarray:
    """Load a single card as BGR uint8 from LMDB."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(buffers=True) as txn:
        raw = bytes(txn.get(str(lmdb_id).encode()))
    env.close()
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)


def _normalize_luminance(bgr: np.ndarray) -> np.ndarray:
    """
    Normalize luminance via CLAHE on the L channel (LAB space).
    Makes the same card under different exposures/saturations produce
    near-identical greyscale structure for pHash.
    Returns BGR uint8.
    """
    lab     = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab     = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _phash_bgr(bgr: np.ndarray) -> np.ndarray:
    """Compute 64-bit pHash for a BGR image. Returns (8,) uint8 packed bits."""
    rgb_pil = Image.fromarray(cv2.cvtColor(_normalize_luminance(bgr), cv2.COLOR_BGR2RGB))
    h = imagehash.phash(rgb_pil)
    return np.packbits(h.hash.flatten().astype(np.uint8))


def _dhash_bgr(bgr: np.ndarray) -> np.ndarray:
    """Compute 64-bit dHash (blurred) for a BGR image. Returns (8,) uint8 packed bits."""
    blurred = cv2.GaussianBlur(bgr, (3, 3), 1)
    rgb_pil = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    h = imagehash.dhash(rgb_pil)
    return np.packbits(h.hash.flatten().astype(np.uint8))


def _rgb_phash_dist(bgr_a: np.ndarray, bgr_b: np.ndarray) -> float:
    """Average pHash Hamming distance across R, G, B channels independently."""
    rgb_a = cv2.split(cv2.cvtColor(bgr_a, cv2.COLOR_BGR2RGB))
    rgb_b = cv2.split(cv2.cvtColor(bgr_b, cv2.COLOR_BGR2RGB))
    total = 0
    for ca, cb in zip(rgb_a, rgb_b):
        ha = np.packbits(imagehash.phash(Image.fromarray(ca)).hash.flatten().astype(np.uint8))
        hb = np.packbits(imagehash.phash(Image.fromarray(cb)).hash.flatten().astype(np.uint8))
        total += _hamming(ha, hb)
    return total / 3.0


def _phash_view(bgr: np.ndarray) -> np.ndarray:
    """Return the image that pHash actually processes: normalized luminance → grayscale BGR."""
    norm = _normalize_luminance(bgr)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two (8,) uint8 packed-bit arrays."""
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())


def _make_composite(images: list[np.ndarray], labels: list[str],
                    thumb: int = 224, max_cols: int = 8) -> np.ndarray:
    """
    Tile images into a grid with a small label beneath each.
    All images are resized to thumb×thumb.
    """
    n        = len(images)
    n_cols   = min(n, max_cols)
    n_rows   = -(-n // n_cols)                    # ceil div
    label_h  = 20
    cell_h   = thumb + label_h
    canvas   = np.zeros((n_rows * cell_h, n_cols * thumb, 3), dtype=np.uint8)

    for i, (img, lbl) in enumerate(zip(images, labels)):
        r, c   = divmod(i, n_cols)
        thumb_img = cv2.resize(img, (thumb, thumb), interpolation=cv2.INTER_AREA)
        y0 = r * cell_h
        canvas[y0:y0 + thumb, c * thumb:(c + 1) * thumb] = thumb_img
        # label row
        cv2.putText(canvas, lbl,
                    (c * thumb + 2, y0 + thumb + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1,
                    cv2.LINE_AA)
    return canvas


def test_interactive(neighbors: dict, threshold: int) -> None:
    """
    Interactively show a random card and its phash-neighbors.
    Press Enter for another, Q/Esc to quit.
    """
    # Only cards that actually have neighbors
    with_nb = [lmdb_id for lmdb_id, nb in neighbors.items() if len(nb) > 0]
    if not with_nb:
        print("No cards have neighbors at this threshold — try a larger --threshold.")
        return

    print(f"{len(with_nb):,} cards have ≥1 neighbor.  Press Enter for next, Q/Esc to quit.")
    rng = np.random.default_rng()

    while True:
        anchor_id  = int(rng.choice(with_nb))
        nb_ids     = [int(x) for x in neighbors[anchor_id].tolist()]

        anchor_img  = _load_image(_CARDS_224, anchor_id)
        anchor_hash = _phash_bgr(anchor_img)
        anchor_dhash = _dhash_bgr(anchor_img)

        # Compute distances for each neighbor and sort by pHash distance
        nb_data = []
        for nb_id in nb_ids:
            img     = _load_image(_CARDS_224, nb_id)
            pd      = _hamming(anchor_hash, _phash_bgr(img))
            dd      = _hamming(anchor_dhash, _dhash_bgr(img))
            rd      = _rgb_phash_dist(anchor_img, img)
            nb_data.append((pd, dd, rd, nb_id, img))
        nb_data.sort(key=lambda x: x[2])  # sort by rgb phash distance
        nb_data = [x for x in nb_data if x[0] <= 6 and x[2] <= 6]  # p≤6 AND rgb≤6
        if not nb_data:
            continue

        images, labels = [anchor_img], [f"ANCHOR {anchor_id}"]
        for pd, dd, rd, nb_id, img in nb_data:
            images.append(img)
            labels.append(f"{nb_id} p={pd} d={dd} rgb={rd:.1f}")

        composite = _make_composite(images, labels, max_cols=7)
        title     = f"anchor={anchor_id}  neighbors={len(nb_ids)}  threshold≤{threshold}bits"
        cv2.imshow(title, composite)
        cv2.moveWindow(title, 0, 0)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (13, 10):          # Enter
                cv2.destroyAllWindows()
                break
            if key in (ord('q'), 27):    # Q or Esc
                cv2.destroyAllWindows()
                return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold",     type=int, default=HAMMING_THRESHOLD,
                    help="Greyscale pHash threshold (default: %(default)s)")
    ap.add_argument("--rgb-threshold", type=int, default=RGB_THRESHOLD,
                    help="RGB per-channel pHash avg threshold (default: %(default)s)")
    ap.add_argument("--dhash-threshold", type=int, default=DHASH_THRESHOLD,
                    help="Blurred dHash threshold (default: %(default)s)")
    ap.add_argument("--max-neighbors", type=int, default=MAX_NEIGHBORS,
                    help="Cap on neighbors per card (default: %(default)s)")
    ap.add_argument("--output",        type=Path, default=OUTPUT_PATH)
    ap.add_argument("--test",          action="store_true",
                    help="Interactively show a random card and its phash-neighbors")
    args = ap.parse_args()

    if args.test:
        if args.output.exists():
            print(f"Loading {args.output} …")
            neighbors = np.load(args.output, allow_pickle=True).item()
        else:
            neighbors = build(args.threshold, args.rgb_threshold, args.dhash_threshold,
                              args.max_neighbors, args.output)
        test_interactive(neighbors, args.threshold)
    else:
        build(args.threshold, args.rgb_threshold, args.dhash_threshold,
              args.max_neighbors, args.output)
