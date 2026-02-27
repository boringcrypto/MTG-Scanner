"""
make_recog_dataset.py
Generates 224×224 recognition training images for the CardRecognizer.

Pipeline for each sample:
  1.  Pick a card from cards.lmdb (recording its integer index)
  2.  Composite it onto a random background with make_sample()
      (same augmentations used for YOLO training)
  3.  Run the 640×640 composite through the pose model (last.pt)
      to find the 4 corners — simulating real inference conditions.
      Pass --no-model to skip this step and use the ground-truth corners
      directly (much faster; good for quick dataset builds).
  4.  Perspective-warp the card region to RECOG_W × RECOG_H (224 × 224)
  5.  Save to  <out>/<card_idx:06d>/<serial:04d>.jpg

The output tree is ImageFolder-compatible: each subdirectory name is the
card's integer LMDB index (= the class for the recognition model).

Modes
-----
--per-card N   (default)
    Generate N augmented composites for every card in cards.lmdb.
    Deterministic coverage; ideal for balanced training.

--total N
    Generate exactly N images by sampling cards uniformly at random.
    Useful for a quick fixed-size dataset.

Usage
-----
    python make_recog_dataset.py --per-card 5 --out recog_dataset
    python make_recog_dataset.py --total 50000 --out recog_dataset
    python make_recog_dataset.py --per-card 3 --no-model   # fast GT warp
"""

import random
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Reuse everything from make_training_set ───────────────────────────────────
from make_training_set import make_sample, warp_to_rect, SIZE
from datasets import CardDataset, BackgroundDataset

RECOG_W         = 224
RECOG_H         = 224
DEFAULT_MODEL   = "webcam_demo/last.pt"
DEFAULT_OUT     = "recog_dataset"
DEFAULT_INDEX   = "data/cards/card_hash_index.npy"
DEFAULT_C224    = "data/cards/cards_224.lmdb"


# ── Shared helpers ────────────────────────────────────────────────────────────

def _load_pose(model_path, no_model):
    """Load and fuse the pose model, or return None when no_model is True."""
    if no_model:
        return None
    from ultralytics import YOLO
    print(f"Loading pose model: {model_path}")
    pose = YOLO(str(model_path))
    pose.fuse()
    print("Pose model ready.")
    return pose


def _make_crop(card_idx, card_bgra, bg_bgr, pose, conf_min):
    """
    Composite one card, optionally run pose inference, warp to RECOG_W×RECOG_H.
    Returns (crop_bgr, composite_bgr, source_str) or (None, None, None) if the model missed.
    """
    composite, gt_label = make_sample(card_bgra, bg_bgr)

    if pose is not None:
        results = pose(composite, verbose=False)[0]
        boxes = results.boxes
        kpts  = results.keypoints
        if boxes is None or kpts is None or len(boxes) == 0:
            return None, None, None
        confs  = boxes.conf.cpu().numpy()
        best   = int(confs.argmax())
        if float(confs[best]) < conf_min:
            return None, None, None
        pts_px = kpts.xy[best].cpu().numpy()
        if (pts_px < 0).any() or (pts_px[:, 0] >= SIZE).any() or (pts_px[:, 1] >= SIZE).any():
            return None, None, None
        source = "model"
    else:
        pts_px = gt_label.reshape(4, 2) * np.array([SIZE, SIZE], dtype=np.float32)
        source = "GT"

    return warp_to_rect(composite, pts_px, RECOG_W, RECOG_H), composite, source


# ── Nearest-neighbour negatives ───────────────────────────────────────────────

def _fetch_224(lmdb_env, idx: int) -> np.ndarray:
    """Fetch one 224×224 BGR image from an open cards_224 lmdb env."""
    with lmdb_env.begin() as txn:
        data = txn.get(str(idx).encode())
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def _build_negatives(index_path: str, card_indices: list[int]) -> dict[int, int]:
    """
    For each card index in card_indices, find the nearest other card in
    Hamming space (= hardest negative for triplet loss).

    Uses faiss.IndexBinaryFlat which runs exact Hamming NN search with
    SIMD popcnt — the full 104k-card index is searched in a few seconds.

    Returns dict: card_idx → nearest_neighbour_card_idx (never self).
    """
    index_path = Path(index_path)
    if not index_path.exists():
        print(f"[negatives] WARNING: {index_path} not found — skipping negatives.")
        return {}

    print(f"[negatives] Loading hash index from {index_path} …")
    packed = np.load(str(index_path))                    # (N, 48) uint8
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    N, nbytes = packed.shape
    hash_bits = nbytes * 8                               # 384

    import faiss
    fi = faiss.IndexBinaryFlat(hash_bits)
    fi.add(packed)                                        # index all N cards

    unique     = np.array(sorted(set(card_indices)), dtype=np.int64)
    queries    = np.ascontiguousarray(packed[unique])    # (U, 48)

    # Search top-2: rank 0 = self (distance 0), rank 1 = nearest neighbour
    print(f"[negatives] Searching {len(unique)} queries against {N} cards …")
    _, I = fi.search(queries, 2)                         # (U, 2)

    # Normally I[:,0] == unique (self), but guard against hash collisions
    nn = np.where(I[:, 0] != unique, I[:, 0], I[:, 1])

    return dict(zip(unique.tolist(), nn.tolist()))


# ── Generate ──────────────────────────────────────────────────────────────────

def generate(per_card, total, model_path, cards_root, bgs_root, out_dir, no_model, conf_min, index_path):
    cards   = CardDataset(lmdb_path=cards_root)
    bgs     = BackgroundDataset(lmdb_path=bgs_root)
    n_cards = len(cards)
    out     = Path(out_dir)
    pose    = _load_pose(model_path, no_model)

    # Build job list: (card_idx, serial_within_card)
    if total is not None:
        serial_counter: dict[int, int] = {}
        jobs = []
        for _ in range(total):
            card_idx = random.randrange(n_cards)
            s = serial_counter.get(card_idx, 0)
            jobs.append((card_idx, s))
            serial_counter[card_idx] = s + 1
    else:
        jobs = [(ci, s) for ci in range(n_cards) for s in range(per_card)]
        random.shuffle(jobs)

    generated = skipped = 0
    samples: list[tuple[int, str]] = []   # (card_idx, relative_path)

    for card_idx, serial in tqdm(jobs, desc="generating"):
        crop, _, _ = _make_crop(card_idx, cards[card_idx], bgs.random(), pose, conf_min)
        if crop is None:
            skipped += 1
            continue
        card_dir = out / f"{card_idx:06d}"
        card_dir.mkdir(parents=True, exist_ok=True)
        rel_path = f"{card_idx:06d}/{serial:04d}.jpg"
        cv2.imwrite(str(out / rel_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        samples.append((card_idx, rel_path))
        generated += 1

    print(f"\nDone — {generated} images written to {out_dir}/")
    if skipped:
        print(f"Skipped {skipped} samples (model did not detect card or low confidence)")

    # ── Build triplets.json ───────────────────────────────────────────────────
    card_indices = [ci for ci, _ in samples]
    negatives    = _build_negatives(index_path, card_indices)

    triplets = [
        {"anchor": ci, "positive": rel, "negative": negatives[ci]}
        for ci, rel in samples
        if ci in negatives
    ]
    triplets_path = out / "triplets.json"
    triplets_path.write_text(json.dumps(triplets))
    print(f"Triplets manifest written to {triplets_path}  ({len(triplets)} entries)")


# ── Test / preview ────────────────────────────────────────────────────────────

def test_preview(model_path, cards_root, bgs_root, no_model, conf_min, index_path, cards_224):
    """Show anchor + positive crop + negative side by side for each sample."""
    import lmdb
    cards = CardDataset(lmdb_path=cards_root)
    bgs   = BackgroundDataset(lmdb_path=bgs_root)
    pose  = _load_pose(model_path, no_model)

    negatives = _build_negatives(index_path, list(range(len(cards))))

    c224_env = None
    if Path(cards_224).exists():
        c224_env = lmdb.open(cards_224, readonly=True, lock=False, readahead=False)
    else:
        print(f"[preview] {cards_224} not found — anchor/negative tiles will be blank")

    blank = np.zeros((RECOG_H, RECOG_W, 3), dtype=np.uint8)

    print("Press any key for next sample, 'q' to quit.")
    while True:
        card_idx = random.randrange(len(cards))
        crop, composite, source = _make_crop(card_idx, cards[card_idx], bgs.random(), pose, conf_min)
        if crop is None:
            print(f"  card {card_idx}: no detection — skipping")
            continue

        anchor   = _fetch_224(c224_env, card_idx)          if c224_env else blank.copy()
        neg_idx  = negatives.get(card_idx)
        negative = _fetch_224(c224_env, neg_idx)           if (c224_env and neg_idx is not None) else blank.copy()

        # Three 224×224 tiles side by side with colour-coded borders
        def bordered(img, color):
            return cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=color)

        row = np.hstack([
            bordered(anchor,   (200, 200, 200)),   # white  — anchor
            bordered(crop,     (0,   200,   0)),   # green  — positive
            bordered(negative, (0,   0,   200)),   # red    — negative
        ])

        labels = [
            (f"anchor  #{card_idx}",      (8,   20)),
            (f"positive ({source})",      (RECOG_W + 16, 20)),
            (f"negative #{neg_idx}",      (RECOG_W * 2 + 24, 20)),
        ]
        for text, (x, y) in labels:
            cv2.putText(row, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("make_recog_dataset preview", row)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    if c224_env:
        c224_env.close()
    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate recognition training images from synthetic composites"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--per-card", type=int, default=5, metavar="N",
                      help="Augmented copies per card (default: 5)")
    mode.add_argument("--total",    type=int, default=None, metavar="N",
                      help="Generate exactly N images via random sampling")

    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        help=f"Pose model .pt (default: {DEFAULT_MODEL})")
    parser.add_argument("--cards",    default="data/cards/cards.lmdb")
    parser.add_argument("--bgs",      default="backgrounds.lmdb")
    parser.add_argument("--out",      default=DEFAULT_OUT,
                        help=f"Output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--no-model", action="store_true",
                        help="Skip YOLO inference; use ground-truth corners (fast)")
    parser.add_argument("--conf",     type=float, default=0.40,
                        help="Min YOLO confidence to keep detection (default: 0.40)")
    parser.add_argument("--index",    default=DEFAULT_INDEX,
                        help=f"Hash index .npy for negative mining (default: {DEFAULT_INDEX})")
    parser.add_argument("--cards-224", default=DEFAULT_C224,
                        help=f"cards_224.lmdb for anchor/negative preview (default: {DEFAULT_C224})")
    parser.add_argument("--test",     action="store_true",
                        help="Preview crops interactively instead of saving")

    args = parser.parse_args()

    if args.test:
        test_preview(args.model, args.cards, args.bgs, args.no_model, args.conf,
                     args.index, args.cards_224)
    else:
        generate(
            per_card   = args.per_card,
            total      = args.total,
            model_path = args.model,
            cards_root = args.cards,
            bgs_root   = args.bgs,
            out_dir    = args.out,
            no_model   = args.no_model,
            conf_min   = args.conf,
            index_path = args.index,
        )
