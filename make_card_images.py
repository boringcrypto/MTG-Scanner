"""
make_card_images.py
-------------------
Pre-processes raw MTG card JPEGs into a single LMDB for fast random access:
  1. Keeps only portrait-orientation images (h > w).
  2. Resizes so height ≤ MAX_HEIGHT (640 px), preserving aspect ratio.
  3. Re-encodes as JPEG and writes into cards.lmdb.

LMDB layout:
  key b"__len__"        → ascii int (total number of cards)
  key b"0", b"1", ...  → JPEG-encoded bytes of each card

Run once before generating the training set:
    python make_card_images.py
    python make_card_images.py --src C:\\card_images --out cards.lmdb --height 640
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import lmdb
import numpy as np
from tqdm import tqdm

MAX_HEIGHT = 640
DEFAULT_SRC = "C:\\card_images"
DEFAULT_OUT = "cards.lmdb"
JPEG_QUALITY = 75
LMDB_INIT_SIZE = 250 * 1024 ** 2   # start at 250 MB
LMDB_GROW_SIZE = 250 * 1024 ** 2   # grow by 250 MB whenever the db is full


def _put_batch(out_path: str, env_holder: list, batch: dict) -> None:
    """Write *batch* to LMDB, auto-growing by LMDB_GROW_SIZE on MapFullError."""
    for attempt in range(1, 10):
        try:
            with env_holder[0].begin(write=True) as txn:
                for k, v in batch.items():
                    txn.put(k, v)
            return
        except lmdb.MapFullError:
            cur = env_holder[0].info()["map_size"]
            new = cur + LMDB_GROW_SIZE
            print(f"\nLMDB full at {cur // 1024**2} MB — growing to {new // 1024**2} MB (attempt {attempt})")
            env_holder[0].close()
            env_holder[0] = lmdb.open(out_path, map_size=new)
    raise RuntimeError("LMDB growth did not resolve MapFullError after 1000 attempts — disk may be full")


def encode(src_path: Path, max_height: int):
    """Load, filter, resize, JPEG-encode one card. Returns bytes or None."""
    img = cv2.imread(str(src_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if h <= w:
        return None  # landscape — skip
    if h > max_height:
        scale = max_height / h
        img = cv2.resize(img, (max(1, round(w * scale)), max_height),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


def main():
    parser = argparse.ArgumentParser(description="Pre-process MTG card images into LMDB.")
    parser.add_argument("--src",     default=DEFAULT_SRC, help="Source directory of raw JPEGs")
    parser.add_argument("--out",     default=DEFAULT_OUT, help="Output LMDB path")
    parser.add_argument("--height",  type=int, default=MAX_HEIGHT, help="Maximum output height in pixels")
    parser.add_argument("--workers", type=int, default=8, help="Parallel encode threads")
    args = parser.parse_args()

    src_dir = Path(args.src)
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
    src_paths = [p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    print(f"Found {len(src_paths)} images in {src_dir}")

    def job(src_path):
        return encode(src_path, args.height)

    BATCH = 500
    env_holder = [lmdb.open(args.out, map_size=LMDB_INIT_SIZE)]
    idx = 0
    batch = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for data in tqdm(ex.map(job, src_paths), total=len(src_paths), desc="encoding"):
            if data is None:
                continue
            batch[str(idx).encode()] = data
            idx += 1
            if len(batch) >= BATCH:
                _put_batch(args.out, env_holder, batch)
                batch.clear()

    if batch:
        _put_batch(args.out, env_holder, batch)

    _put_batch(args.out, env_holder, {b"__len__": str(idx).encode()})
    env_holder[0].close()

    print(f"Done — {idx} cards written to {args.out}")


if __name__ == "__main__":
    main()
