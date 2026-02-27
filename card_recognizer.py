"""
card_recognizer.py
------------------
Nearest-neighbour card recognition using 384-bit ViT perceptual hashes.

Pipeline
--------
1. (Once) Resize every card in cards.lmdb to 224×224 (stretched) and write
   cards_224.lmdb — fast JPEG re-encode, done once.
2. (Once) Hash all cards with ViT-Small in batches of 512; save card_hash_index.npy.
3. At query time: hash the query BGR image, find nearest neighbours by Hamming
   distance via vectorised numpy XOR — sub-millisecond for 100k cards.

Usage
-----
    rec = CardRecognizer()              # builds index on first run (~few minutes)
    rec = CardRecognizer(force_rebuild=True)

    matches = rec.recognize_bgr(bgr)   # list of Match, sorted by Hamming distance
    print(matches[0])

CLI quick-test:
    python card_recognizer.py --query path/to/card.jpg --top 5 --show
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch
from tqdm import tqdm

from card_hasher import CardHasher, HASH_BITS, IMAGE_SIZE
from lmdb_writer import LmdbWriter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CARDS_LMDB     = "data/cards/cards.lmdb"
CARDS_224_LMDB = "data/cards/cards_224.lmdb"   # one-time resized cache (224×224, native ViT-Small resolution)
INDEX_CACHE    = "data/cards/card_hash_index.npy"  # packed uint8, shape (N, 48)
JPEG_QUALITY   = 75
BATCH_SIZE     = 512   # safe for ViT-Small fp32 on 8 GB VRAM (~300 MB activations)


# ---------------------------------------------------------------------------
# Data class for results
# ---------------------------------------------------------------------------

@dataclass
class Match:
    idx:      int    # card index in cards_256.lmdb
    distance: int    # Hamming distance (0 = identical, 384 = opposite)

    @property
    def similarity(self) -> float:
        """Fraction of matching bits (1.0 = perfect match)."""
        return 1.0 - self.distance / HASH_BITS

    def __repr__(self) -> str:
        return f"Match(idx={self.idx}, hamming={self.distance}, similarity={self.similarity:.1%})"


# ---------------------------------------------------------------------------
# Resized-lmdb helpers
# ---------------------------------------------------------------------------

def _build_resized_lmdb(src: str = CARDS_LMDB, dst: str = CARDS_224_LMDB) -> int:
    """
    Read every card from *src*, stretch-resize to 224×224, JPEG-encode, and
    write to *dst*.  Returns the number of cards written.
    """
    src_env = lmdb.open(src, readonly=True, lock=False,
                        readahead=False, meminit=False)
    with src_env.begin() as txn:
        n = int(txn.get(b"__len__").decode())

    print(f"Building {dst}: resizing {n:,} cards to {IMAGE_SIZE}×{IMAGE_SIZE} …")

    WRITE_BATCH = 500
    write_buf: dict = {}

    with LmdbWriter(dst) as writer:
        with src_env.begin() as txn:
            for idx in tqdm(range(n), unit="card"):
                data = txn.get(str(idx).encode())
                if data is None:
                    continue
                bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                bgr = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
                ok, buf = cv2.imencode(".jpg", bgr,
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    continue
                write_buf[str(idx).encode()] = buf.tobytes()
                if len(write_buf) >= WRITE_BATCH:
                    writer.put_batch(write_buf)
                    write_buf.clear()

        if write_buf:
            writer.put_batch(write_buf)
        writer.put(b"__len__", str(n).encode())

    print(f"  Done — {n:,} cards written to {dst}")
    return n


# ---------------------------------------------------------------------------
# Recognizer
# ---------------------------------------------------------------------------

class CardRecognizer:
    """
    Loads or builds a precomputed hash index, then answers nearest-neighbour
    queries in sub-millisecond time (pure numpy XOR).

    On first run two one-time build steps are executed:
      1. cards_224.lmdb  — all cards resized to 224×224 (stretch, no padding)
      2. card_hash_index.npy  — (N, 48) packed uint8 hash matrix

    Parameters
    ----------
    cards_lmdb    : source cards.lmdb
    cards_224     : destination for resized images
    index_cache   : path for the .npy hash matrix
    force_rebuild : ignore existing caches and recompute everything
    device        : 'cuda' or 'cpu'
    dtype         : torch.bfloat16 (default on Ampere+) or torch.float32
    batch_size    : images per ViT forward pass (256 is safe for 8 GB VRAM)
    """

    def __init__(
        self,
        cards_lmdb:    str        = CARDS_LMDB,
        cards_224:     str        = CARDS_224_LMDB,
        index_cache:   str        = INDEX_CACHE,
        force_rebuild: bool       = False,
        device:        str | None = None,
        dtype=None,
        batch_size:    int        = BATCH_SIZE,
    ):
        self.cards_lmdb  = cards_lmdb
        self.cards_224   = cards_224
        self.index_cache = Path(index_cache)
        self.hasher      = CardHasher(pretrained=True, device=device, dtype=dtype)

        # ── Step 1: ensure 224×224 lmdb exists ──────────────────────────
        if not Path(cards_224).exists():
            _build_resized_lmdb(cards_lmdb, cards_224)

        # ── Step 2: ensure hash index exists ────────────────────────────
        if force_rebuild or not self.index_cache.exists():
            self._packed = self._build_index(self.cards_224, batch_size)
        else:
            print(f"Loading hash index from {self.index_cache} …")
            self._packed = np.load(str(self.index_cache))   # (N, 48) uint8
            print(f"  {len(self._packed):,} cards indexed.")

        # Expand to full bit matrix for vectorised Hamming
        # Shape: (N, 384) uint8
        self._bits: np.ndarray = np.unpackbits(
            self._packed, axis=1, count=HASH_BITS, bitorder="big"
        ).astype(np.uint8)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self, cards_224: str, batch_size: int) -> np.ndarray:
        """
        Hash all cards in cards_224.lmdb in batches and return a packed
        uint8 matrix of shape (N, 48).
        """
        env = lmdb.open(cards_224, readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin() as txn:
            n = int(txn.get(b"__len__").decode())

        print(f"Building hash index for {n:,} cards  (batch={batch_size}) …")
        packed_rows: list[np.ndarray] = []

        with env.begin() as txn:
            batch_bgrs: list[np.ndarray] = []
            for idx in tqdm(range(n), unit="card"):
                data = txn.get(str(idx).encode())
                if data is None:
                    # flush current batch first if any
                    if batch_bgrs:
                        bits = self.hasher.hash_batch_bgr(batch_bgrs)
                        for b in bits:
                            packed_rows.append(np.packbits(b, bitorder="big"))
                        batch_bgrs = []
                    packed_rows.append(np.zeros(HASH_BITS // 8, dtype=np.uint8))
                    continue

                bgr = cv2.imdecode(
                    np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                batch_bgrs.append(bgr)

                if len(batch_bgrs) == batch_size:
                    bits = self.hasher.hash_batch_bgr(batch_bgrs)   # (B, 384)
                    for b in bits:
                        packed_rows.append(np.packbits(b, bitorder="big"))
                    batch_bgrs = []

            # flush remainder
            if batch_bgrs:
                bits = self.hasher.hash_batch_bgr(batch_bgrs)
                for b in bits:
                    packed_rows.append(np.packbits(b, bitorder="big"))

        packed = np.stack(packed_rows, axis=0)   # (N, 48)
        np.save(str(self.index_cache), packed)
        print(f"Index saved to {self.index_cache}  ({packed.nbytes / 1024**2:.1f} MB)")
        return packed

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize_bgr(self, bgr: np.ndarray, top_k: int = 5) -> list[Match]:
        """
        Find the *top_k* closest cards by Hamming distance.

        Parameters
        ----------
        bgr   : OpenCV BGR uint8 array (any size — will be resized internally)
        top_k : number of results to return

        Returns
        -------
        List of Match objects sorted by ascending Hamming distance.
        """
        query_bits = self.hasher.hash_bgr(bgr)   # (384,) uint8

        # Vectorised Hamming: XOR bit-by-bit then sum
        diffs     = np.bitwise_xor(self._bits, query_bits[np.newaxis, :])  # (N, 384)
        distances = diffs.sum(axis=1)                                       # (N,)

        k = min(top_k, len(distances))
        top_idx = np.argpartition(distances, k - 1)[:k]
        top_idx = top_idx[np.argsort(distances[top_idx])]

        return [Match(idx=int(i), distance=int(distances[i])) for i in top_idx]

    def get_card_image(self, idx: int) -> np.ndarray:
        """Fetch the 224×224 BGR card image from cards_224.lmdb."""
        env = lmdb.open(self.cards_224, readonly=True, lock=False)
        with env.begin() as txn:
            data = txn.get(str(idx).encode())
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognise an MTG card by hash similarity.")
    parser.add_argument("--query",   default=None,
                        help="Path to query card image (omit to pick a random card)")
    parser.add_argument("--top",     type=int, default=5)
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild resized lmdb and hash index")
    parser.add_argument("--show",    action="store_true",
                        help="Display query vs best match side-by-side")
    args = parser.parse_args()

    rec = CardRecognizer(force_rebuild=args.rebuild)

    if args.query is None:
        env = lmdb.open(CARDS_224_LMDB, readonly=True, lock=False)
        with env.begin() as txn:
            n = int(txn.get(b"__len__").decode())
        rand_idx = random.randrange(n)
        print(f"No --query supplied; using random card from lmdb (idx={rand_idx})")
        query_bgr = rec.get_card_image(rand_idx)
    else:
        query_bgr = cv2.imread(args.query)
        if query_bgr is None:
            raise FileNotFoundError(f"Cannot load image: {args.query}")

    t0 = time.perf_counter()
    matches = rec.recognize_bgr(query_bgr, top_k=args.top)
    elapsed = time.perf_counter() - t0

    print(f"\nTop {args.top} matches  (query time: {elapsed*1000:.2f} ms)")
    print("-" * 40)
    for rank, m in enumerate(matches, 1):
        print(f"  #{rank}  {m}")

    if args.show and matches:
        THUMB_W, THUMB_H = 224, 314   # display size for each card
        LABEL_H = 28
        FONT    = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.55
        FONT_THICK = 1
        PAD     = 4   # pixels between cards

        def make_card(img: np.ndarray, label: str, border_color) -> np.ndarray:
            card = cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
            # coloured border
            card = cv2.copyMakeBorder(card, 3, 0, 3, 3,
                                      cv2.BORDER_CONSTANT, value=border_color)
            # label strip below
            strip = np.zeros((LABEL_H, card.shape[1], 3), dtype=np.uint8)
            tw, th = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)[0]
            tx = max(0, (strip.shape[1] - tw) // 2)
            cv2.putText(strip, label, (tx, LABEL_H - 7),
                        FONT, FONT_SCALE, (220, 220, 220), FONT_THICK, cv2.LINE_AA)
            return np.vstack([card, strip])

        # Query tile (white border)
        q_tile = make_card(query_bgr, "Query", (220, 220, 220))

        # Match tiles
        tiles = [q_tile]
        for rank, m in enumerate(matches, 1):
            img   = rec.get_card_image(m.idx)
            # Best match = green border, rest = yellow
            color = (0, 200, 0) if rank == 1 else (0, 200, 220)
            label = f"#{rank}  d={m.distance}  {m.similarity:.1%}"
            tiles.append(make_card(img, label, color))

        # Pad all tiles to the same height then hstack
        max_h = max(t.shape[0] for t in tiles)
        padded = []
        for t in tiles:
            pad = max_h - t.shape[0]
            if pad:
                t = cv2.copyMakeBorder(t, 0, pad, 0, 0,
                                       cv2.BORDER_CONSTANT, value=(30, 30, 30))
            # add PAD-pixel gap on the right
            t = cv2.copyMakeBorder(t, 0, 0, 0, PAD,
                                   cv2.BORDER_CONSTANT, value=(30, 30, 30))
            padded.append(t)

        composite = np.hstack(padded)
        cv2.imshow(f"Query vs Top-{len(matches)} matches", composite)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


