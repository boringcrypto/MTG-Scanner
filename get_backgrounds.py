"""
get_backgrounds.py

Downloads background images from multiple sources and stores them directly
in a single LMDB file for fast random access during training.

LMDB layout  (backgrounds.lmdb):
    key b"__len__"        → ascii int   (total images stored)
    key b"0", b"1", …    → JPEG bytes  (short side scaled to TARGET_SHORT px)

A small JSON sidecar (backgrounds_index.json) records (source, url) pairs
for deduplication across runs — it is NOT needed at training time.

Usage:
    python get_backgrounds.py --dtd 5000
    python get_backgrounds.py --openimages 50000
    python get_backgrounds.py --summary
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import cv2
import lmdb
import numpy as np
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORK_DIR        = Path(__file__).parent
LMDB_PATH       = WORK_DIR / "backgrounds.lmdb"
INDEX_FILE      = WORK_DIR / "backgrounds_index.json"
LMDB_MAP_SIZE   = 20 * 1024 ** 3   # 20 GB sparse
JPEG_QUALITY    = 75
TARGET_SHORT    = 640               # short side target; long side ≥ 640 → crop room
REQUEST_TIMEOUT = 30
RETRY_ATTEMPTS  = 3
RETRY_DELAY     = 2
BATCH           = 200               # LMDB write batch size


# ---------------------------------------------------------------------------
# LMDB + index helpers
# ---------------------------------------------------------------------------

def _open_lmdb(write: bool = False):
    return lmdb.open(str(LMDB_PATH), map_size=LMDB_MAP_SIZE,
                     readonly=not write, lock=write,
                     readahead=False, meminit=False)


def _load_index() -> dict:
    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            return json.load(f)
    return {"count": 0, "seen": {}}   # seen: {source: set-of-urls}


def _save_index(index: dict) -> None:
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f)


def _is_seen(index: dict, source: str, url: str) -> bool:
    return url in index["seen"].get(source, {})


def _mark_seen(index: dict, source: str, url: str) -> None:
    index["seen"].setdefault(source, {})[url] = 1


def _resize_encode(img: np.ndarray) -> Optional[bytes]:
    """Scale so short side = TARGET_SHORT, re-encode as JPEG."""
    h, w = img.shape[:2]
    short = min(h, w)
    if short != TARGET_SHORT:
        scale = TARGET_SHORT / short
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img,
                         (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=interp)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


class _LmdbWriter:
    """Context manager that batches LMDB puts and tracks the global index."""

    def __init__(self, index: dict):
        self.index = index
        self.env   = _open_lmdb(write=True)
        self.batch: dict[bytes, bytes] = {}
        self._start_count = index["count"]

    def put(self, jpeg_bytes: bytes) -> None:
        idx = self.index["count"]
        self.batch[str(idx).encode()] = jpeg_bytes
        self.index["count"] += 1
        if len(self.batch) >= BATCH:
            self._flush()

    def _flush(self) -> None:
        if not self.batch:
            return
        with self.env.begin(write=True) as txn:
            for k, v in self.batch.items():
                txn.put(k, v)
        self.batch.clear()

    def close(self) -> int:
        self._flush()
        with self.env.begin(write=True) as txn:
            txn.put(b"__len__", str(self.index["count"]).encode())
        self.env.close()
        return self.index["count"] - self._start_count


def _download_bytes(url: str) -> Optional[bytes]:
    headers = {"User-Agent": "get_backgrounds/1.0 (TCG training set builder)"}
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"  [error] Failed to download {url}: {e}")
    return None


def _bytes_to_bgr(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _download_large_file(url: str, dest: Path, label: str) -> None:
    headers = {"User-Agent": "get_backgrounds/1.0 (TCG training set builder)"}
    with requests.get(url, headers=headers, stream=True, timeout=REQUEST_TIMEOUT) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) or None
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            unit_divisor=1024, desc=label,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                f.write(chunk)
                bar.update(len(chunk))


# ---------------------------------------------------------------------------
# Source 1 – Describable Textures Dataset (DTD)
# ---------------------------------------------------------------------------

DTD_BASE = "https://www.robots.ox.ac.uk/~vgg/data/dtd"
DTD_TARBALL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"


def download_dtd(max_images: int = 500) -> int:
    """
    Download images from the Describable Textures Dataset (DTD).
    Fetches the official tarball (~600 MB) once, extracts to a temp dir,
    then encodes directly into LMDB.
    """
    import tarfile

    source   = "dtd"
    cache_tar   = WORK_DIR / "dtd-r1.0.1.tar.gz"
    extract_tmp = WORK_DIR / "_dtd_extract"
    index = _load_index()

    if not cache_tar.exists():
        print("[DTD] Downloading tarball (~600 MB) …")
        _download_large_file(DTD_TARBALL, cache_tar, "DTD download")
        print("[DTD] Download complete.")

    if not extract_tmp.exists():
        print("[DTD] Extracting tarball …")
        with tarfile.open(cache_tar, "r:gz") as tar:
            members = [m for m in tar.getmembers()
                       if m.name.endswith((".jpg", ".jpeg", ".png"))
                       and "/images/" in m.name]
            tar.extractall(path=extract_tmp, members=tqdm(members, desc="DTD extract"))
        print("[DTD] Extraction complete.")

    all_paths = list(extract_tmp.rglob("*.jpg")) + \
                list(extract_tmp.rglob("*.jpeg")) + \
                list(extract_tmp.rglob("*.png"))
    random.shuffle(all_paths)
    if max_images:
        all_paths = all_paths[:max_images]

    to_add = [p for p in all_paths if not _is_seen(index, source, f"dtd_tar::{p.name}")]
    if not to_add:
        print(f"[DTD] All images already in LMDB — nothing to do.")
        return 0

    print(f"[DTD] Writing {len(to_add)} images to LMDB …")
    writer = _LmdbWriter(index)
    added  = 0
    for src_path in tqdm(to_add, desc="DTD"):
        img = cv2.imread(str(src_path))
        if img is None:
            continue
        data = _resize_encode(img)
        if data is None:
            continue
        writer.put(data)
        _mark_seen(index, source, f"dtd_tar::{src_path.name}")
        added += 1

    writer.close()
    _save_index(index)
    print(f"[DTD] Added {added} images.")
    return added


# ---------------------------------------------------------------------------
# Source 2 – Places365 (validation set, freely downloadable)
# ---------------------------------------------------------------------------

PLACES365_VAL_FILELIST = (
    "http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"
)
PLACES365_VAL_IMAGES = "http://data.csail.mit.edu/places/places365/val_256.tar"


def download_places365(max_images: int = 500) -> int:
    """
    Download validation images from Places365-Standard (256×256).
    Fetches the val_256 tarball (~2 GB), streams images directly into LMDB.
    """
    import tarfile

    source    = "places365"
    cache_tar = WORK_DIR / "places365_val_256.tar"
    index     = _load_index()

    if not cache_tar.exists():
        print("[Places365] Downloading val_256.tar (~2 GB) …")
        _download_large_file(PLACES365_VAL_IMAGES, cache_tar, "Places365 download")
        print("[Places365] Download complete.")

    writer = _LmdbWriter(index)
    added  = 0
    with tarfile.open(cache_tar, "r") as tar:
        members = [m for m in tar.getmembers()
                   if m.name.lower().endswith((".jpg", ".jpeg"))]
        random.shuffle(members)
        members = members[:max_images]
        for member in tqdm(members, desc="Places365"):
            url_key = f"places365_tar::{member.name}"
            if _is_seen(index, source, url_key):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            img = _bytes_to_bgr(f.read())
            if img is None:
                continue
            data = _resize_encode(img)
            if data is None:
                continue
            writer.put(data)
            _mark_seen(index, source, url_key)
            added += 1

    writer.close()
    _save_index(index)
    print(f"[Places365] Added {added} images.")
    return added


# ---------------------------------------------------------------------------
# Source 3 – Open Images Dataset (via FiftyOne)
# ---------------------------------------------------------------------------

def download_openimages(max_images: int = 300) -> int:
    """
    Download random images from Open Images v7 using the FiftyOne library.
    Requires: pip install fiftyone
    """
    try:
        import fiftyone.zoo as foz
    except ImportError:
        print("[OpenImages] fiftyone not installed. Run: pip install fiftyone")
        return 0

    source = "openimages"
    index  = _load_index()

    print(f"[OpenImages] Loading {max_images} samples via FiftyOne …")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        max_samples=max_images,
        shuffle=True,
        label_types=[],
    )

    writer = _LmdbWriter(index)
    added  = 0
    for sample in tqdm(dataset, desc="OpenImages"):
        url_key = f"openimages::{sample.filepath}"
        if _is_seen(index, source, url_key):
            continue
        img = cv2.imread(sample.filepath)
        if img is None:
            continue
        data = _resize_encode(img)
        if data is None:
            continue
        writer.put(data)
        _mark_seen(index, source, url_key)
        added += 1

    writer.close()
    _save_index(index)
    print(f"[OpenImages] Added {added} images.")
    return added


# ---------------------------------------------------------------------------
# Source 4 – Unsplash (official API — requires a free Access Key)
# ---------------------------------------------------------------------------

UNSPLASH_API_URL   = "https://api.unsplash.com/photos/random"
UNSPLASH_BATCH_SIZE = 30

UNSPLASH_QUERIES = [
    "table", "wood", "fabric", "carpet", "desk",
    "floor", "concrete", "grass", "paper", "texture",
    "playmat", "board game", "indoor", "outdoor", "stone",
]


def download_unsplash(
    max_images: int = 200,
    width: int = 1024,
    height: int = 768,
    unsplash_access_key: Optional[str] = None,
) -> int:
    """
    Download random images from Unsplash using the official API.
    Register at https://unsplash.com/oauth/applications for a Client-ID.
    Set via UNSPLASH_ACCESS_KEY env var or the unsplash_access_key argument.
    """
    access_key = unsplash_access_key or os.environ.get("UNSPLASH_ACCESS_KEY", "")
    if not access_key:
        print(
            "[Unsplash] No access key found.\n"
            "  Register at https://unsplash.com/oauth/applications and set\n"
            "  UNSPLASH_ACCESS_KEY=YOUR_KEY or pass unsplash_access_key=."
        )
        return 0

    source  = "unsplash"
    index   = _load_index()
    headers = {"Authorization": f"Client-ID {access_key}", "Accept-Version": "v1"}
    queries = UNSPLASH_QUERIES * (max_images // len(UNSPLASH_QUERIES) + 1)
    random.shuffle(queries)

    writer = _LmdbWriter(index)
    added  = 0
    with tqdm(total=max_images, desc="Unsplash") as bar:
        for query in queries:
            if added >= max_images:
                break
            batch  = min(UNSPLASH_BATCH_SIZE, max_images - added)
            params = {"query": query, "count": batch,
                      "orientation": "landscape", "content_filter": "high"}
            try:
                resp = requests.get(UNSPLASH_API_URL, headers=headers,
                                    params=params, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                photos    = resp.json()
                remaining = resp.headers.get("X-Ratelimit-Remaining", "?")
                if remaining != "?" and int(remaining) < 5:
                    print(f"\n[Unsplash] Rate limit nearly exhausted. Stopping.")
                    break
            except Exception as e:
                print(f"\n[Unsplash] API error for '{query}': {e}")
                time.sleep(2)
                continue

            for photo in photos:
                if added >= max_images:
                    break
                url_key = f"unsplash::{photo.get('id', '')}"
                if _is_seen(index, source, url_key):
                    continue
                raw_url  = photo["urls"]["raw"]
                img_url  = f"{raw_url}&w={width}&h={height}&fit=crop&fm=jpg&q=85"
                raw_data = _download_bytes(img_url)
                if raw_data is None:
                    continue
                img = _bytes_to_bgr(raw_data)
                if img is None:
                    continue
                data = _resize_encode(img)
                if data is None:
                    continue
                writer.put(data)
                _mark_seen(index, source, url_key)
                added += 1
                bar.update(1)

            time.sleep(1)

    writer.close()
    _save_index(index)
    print(f"[Unsplash] Added {added} images.")
    return added


# ---------------------------------------------------------------------------
# Source 5 – COCO 2017 validation set
# ---------------------------------------------------------------------------

COCO_VAL_ANNOTATIONS = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/{filename}"


def download_coco(max_images: int = 300) -> int:
    """Download images from COCO 2017 validation set."""
    import zipfile

    source          = "coco"
    cache_zip       = WORK_DIR / "coco_annotations_val2017.zip"
    annotations_dir = WORK_DIR / "_coco_annotations"
    index           = _load_index()

    if not cache_zip.exists():
        print("[COCO] Downloading annotation zip (~240 MB) …")
        _download_large_file(COCO_VAL_ANNOTATIONS, cache_zip, "COCO annotations")

    if not annotations_dir.exists():
        print("[COCO] Extracting annotations …")
        with zipfile.ZipFile(cache_zip, "r") as z:
            z.extractall(annotations_dir)

    ann_file = annotations_dir / "annotations" / "instances_val2017.json"
    with open(ann_file) as f:
        coco_data = json.load(f)

    all_images = coco_data["images"]
    random.shuffle(all_images)

    writer = _LmdbWriter(index)
    added  = 0
    for img_info in tqdm(all_images[:max_images * 2], desc="COCO"):
        if added >= max_images:
            break
        filename = img_info["file_name"]
        url_key  = f"coco::{filename}"
        if _is_seen(index, source, url_key):
            continue
        raw_data = _download_bytes(COCO_IMAGE_URL.format(filename=filename))
        if raw_data is None:
            continue
        img = _bytes_to_bgr(raw_data)
        if img is None:
            continue
        data = _resize_encode(img)
        if data is None:
            continue
        writer.put(data)
        _mark_seen(index, source, url_key)
        added += 1

    writer.close()
    _save_index(index)
    print(f"[COCO] Added {added} images.")
    return added


# ---------------------------------------------------------------------------
# Source 6 – Synthetic backgrounds
# ---------------------------------------------------------------------------

def _make_noise(h: int, w: int) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _make_gradient(h: int, w: int) -> np.ndarray:
    angle = random.uniform(0, np.pi * 2)
    dx, dy = np.cos(angle), np.sin(angle)
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    val = np.clip(xx * dx + yy * dy, 0, 1)
    c1  = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
    c2  = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
    return (val[..., None] * c1 + (1 - val[..., None]) * c2).astype(np.uint8)


def _make_solid(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), [random.randint(0, 255) for _ in range(3)], dtype=np.uint8)


def _make_perlin_like(h: int, w: int) -> np.ndarray:
    result = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    for scale in [4, 8, 16, 32, 64]:
        small = np.random.rand(h // scale + 2, w // scale + 2).astype(np.float32)
        layer = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        result += layer * amplitude
        amplitude *= 0.5
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr  = cv2.cvtColor(cv2.merge([result, result, result]), cv2.COLOR_GRAY2BGR)
    tint = np.array([random.randint(100, 255) for _ in range(3)], dtype=np.float32) / 255.0
    return np.clip(bgr.astype(np.float32) * tint, 0, 255).astype(np.uint8)


_SYNTHETIC_GENERATORS = [_make_noise, _make_gradient, _make_solid, _make_perlin_like]


def generate_synthetic(count: int = 500, width: int = 1024, height: int = 768) -> int:
    """Generate synthetic backgrounds (noise, gradients, solids, multi-scale noise)."""
    source = "synthetic"
    index  = _load_index()

    writer = _LmdbWriter(index)
    added  = 0
    for i in tqdm(range(count), desc="Synthetic"):
        gen = random.choice(_SYNTHETIC_GENERATORS)
        img = gen(height, width)
        if random.random() < 0.3:
            k = random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (k, k), 0)
        data = _resize_encode(img)
        if data is None:
            continue
        writer.put(data)
        _mark_seen(index, source, f"synthetic::{i}::{random.random()}")
        added += 1

    writer.close()
    _save_index(index)
    print(f"[Synthetic] Generated {added} images.")
    return added


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def store_summary() -> dict:
    """Print a summary of what's in the LMDB."""
    index   = _load_index()
    summary: dict[str, int] = {}
    for src, urls in index["seen"].items():
        summary[src] = len(urls)
    total = index["count"]
    print(f"\n{'─'*40}")
    print(f"  LMDB: {LMDB_PATH}")
    print(f"{'─'*40}")
    for src, count in sorted(summary.items()):
        print(f"  {src:<15} {count:>6} images")
    print(f"{'─'*40}")
    print(f"  {'TOTAL':<15} {total:>6} images")
    print(f"{'─'*40}\n")
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download backgrounds for YOLO training.")
    parser.add_argument("--dtd",         type=int, default=0,   metavar="N", help="Download N images from DTD")
    parser.add_argument("--places365",   type=int, default=0,   metavar="N", help="Download N images from Places365")
    parser.add_argument("--openimages",  type=int, default=0,   metavar="N", help="Download N images from Open Images (requires fiftyone)")
    parser.add_argument("--unsplash",    type=int, default=0,   metavar="N", help="Download N images from Unsplash (requires --unsplash-key or UNSPLASH_ACCESS_KEY env var)")
    parser.add_argument("--unsplash-key", type=str, default="",  metavar="KEY", help="Unsplash API access key (Client-ID)")
    parser.add_argument("--coco",        type=int, default=0,   metavar="N", help="Download N images from COCO")
    parser.add_argument("--synthetic",   type=int, default=0,   metavar="N", help="Generate N synthetic backgrounds")
    parser.add_argument("--all",         type=int, default=0,   metavar="N", help="Download N images from ALL sources")
    parser.add_argument("--summary",     action="store_true",                help="Print store summary and exit")
    args = parser.parse_args()

    if args.summary:
        store_summary()
    else:
        n = args.all
        if args.dtd        or n: download_dtd(args.dtd or n)
        if args.places365  or n: download_places365(args.places365 or n)
        if args.openimages or n: download_openimages(args.openimages or n)
        if args.unsplash   or n: download_unsplash(args.unsplash or n, unsplash_access_key=args.unsplash_key or None)
        if args.coco       or n: download_coco(args.coco or n)
        if args.synthetic  or n: generate_synthetic(args.synthetic or n)
        store_summary()
