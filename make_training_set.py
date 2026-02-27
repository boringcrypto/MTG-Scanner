"""
make_training_set.py
Generates a YOLO Pose dataset: 640x640 images, 1 card per image.
Card occupies 50–100% of the image area (by bounding-box diagonal).

Output layout:
    dataset/images/{train,val,test}/NNNNN.jpg
    dataset/labels/{train,val,test}/NNNNN.txt
        (YOLO Pose: 0 cx cy w h  x1 y1 2  x2 y2 2  x3 y3 2  x4 y4 2)
        keypoints are TL TR BR BL corners, visibility=2 (always visible)

Usage:
    python make_training_set.py --total 10000
"""

import random, math, argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from datasets import CardDataset, BackgroundDataset
from augmentations import (
    AugmentationPipeline,
    color_jitter, white_balance, add_noise,
    gaussian_blur, motion_blur,
    glare, shade, foil,
)
from codetiming import Timer

# Augmentations applied to the card surface before compositing
CARD_AUGMENTATIONS = AugmentationPipeline([
    (shade,  30),
    (glare,  25),
    (foil,   20),
])

# Augmentations applied to the full composite image
IMAGE_AUGMENTATIONS = AugmentationPipeline([
    (color_jitter,  100),
    (white_balance, 100),
    (gaussian_blur,  30),
    (motion_blur,    30),
    (add_noise,     100),
])


# ── Core composite ────────────────────────────────────────────────────────────

SIZE = 640

def _composite_card(bgra, bg):
    """
    Jitter, rotate, scale, place and warp a BGRA card onto a 640×640 BGR background.
    Returns (composite_bgr, obb_label_string).
    All transforms are pure coordinate math; pixels are resampled exactly once.
    """
    S  = SIZE
    ch, cw = bgra.shape[:2]

    # Step 1: Perspective jitter — nudge each corner by ±PERSP_PCT of card dims
    PERSP_PCT = 0.05
    corners = np.array([[0,0],[cw,0],[cw,ch],[0,ch]], dtype=np.float32)
    jitter   = np.random.uniform(-PERSP_PCT, PERSP_PCT, corners.shape).astype(np.float32)
    jitter  *= np.array([[cw, ch]], dtype=np.float32)
    jittered = corners + jitter   # TL TR BR BL in card pixel space, slightly skewed

    # Step 2: Rotate jittered corners around card centre
    angle = random.uniform(0, 360)
    rad   = math.radians(angle)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    ocx, ocy = cw / 2, ch / 2
    rotated = [(( x - ocx) * cos_r - (y - ocy) * sin_r,
                ( x - ocx) * sin_r + (y - ocy) * cos_r)
               for x, y in jittered]

    # Step 3: Axis-aligned bounding box of rotated corners
    xs = [p[0] for p in rotated];  ys = [p[1] for p in rotated]
    bbox_w = max(xs) - min(xs);    bbox_h = max(ys) - min(ys)

    # Step 4: Scale so bbox fits in S, card height ≥ S/2
    scale_max = S / max(bbox_w, bbox_h)
    scale_min = (S / 2) / ch
    scale = random.uniform(min(scale_min, scale_max), scale_max)

    # Step 5: Place — random position within canvas
    room_x = S - bbox_w * scale;   room_y = S - bbox_h * scale
    px = random.uniform(0, max(0.0, room_x))
    py = random.uniform(0, max(0.0, room_y))
    dst_corners = np.array(
        [((x - min(xs)) * scale + px, (y - min(ys)) * scale + py) for x, y in rotated],
        dtype=np.float32)

    # Step 6: Single warpPerspective — original card pixels → canvas
    src_corners = np.array([[0,0],[cw,0],[cw,ch],[0,ch]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped = cv2.warpPerspective(bgra, M, (S, S),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Step 7: Alpha-blend onto background — cv2.blendLinear avoids float32 image allocations.
    # Weights must be float32 single-channel; dividing the alpha channel is cheap (640×640×1).
    w1 = warped[:, :, 3].astype(np.float32) / 255.0   # (S, S) card weight
    w2 = 1.0 - w1                                      # (S, S) bg weight
    composite = cv2.blendLinear(warped[:, :, :3], bg, w1, w2)

    # shape (8,): normalised xy corners TL TR BR BL — raw corners, used by both
    # the disk writer (converted to pose format) and the streaming dataset.
    label = (dst_corners / S).flatten().astype(np.float32)
    return composite, label


def make_sample(card_bgr, bg_bgr):
    """
    Composite one card onto one background at 640×640.
    Pass card_bgr=None for a background-only (negative) sample.
    Returns (composite_bgr, label)  — label is "" when no card.
    """
    S = SIZE

    # 1. Background → random square crop → 640×640
    with Timer("bg_crop", logger=None):
        bh, bw = bg_bgr.shape[:2]
        crop = min(bh, bw)
        x0 = random.randint(0, bw - crop)
        y0 = random.randint(0, bh - crop)
        bg = cv2.resize(bg_bgr[y0:y0+crop, x0:x0+crop], (S, S))

    if card_bgr is not None:
        # 2. Card: surface effects + warp onto bg  (alpha already baked in by make_card_images.py)
        bgra = card_bgr
        with Timer("card_aug", logger=None):
            bgra[:, :, :3] = CARD_AUGMENTATIONS(bgra[:, :, :3])
        with Timer("composite", logger=None):
            bg, label = _composite_card(bgra, bg)
    else:
        label = None   # no card — negative sample

    # 3. Post-composite augmentations (whole image — card and bg get same treatment)
    with Timer("img_aug", logger=None):
        bg = IMAGE_AUGMENTATIONS(bg)

    return bg, label


# ── Dataset generation ────────────────────────────────────────────────────────

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

def generate(total, card_root, bg_root, out_dir, num_workers=14):
    cards = CardDataset(lmdb_path=card_root)
    bgs   = BackgroundDataset(lmdb_path=bg_root)

    # Build split index ranges
    counts = {}
    remaining = total
    for i, (split, frac) in enumerate(SPLITS.items()):
        n = int(total * frac) if i < len(SPLITS) - 1 else remaining
        counts[split] = n
        remaining -= n

    # Create all output directories upfront
    for split in counts:
        (Path(out_dir) / "images" / split).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "labels" / split).mkdir(parents=True, exist_ok=True)

    def make_one(args):
        """Run entirely in C/NumPy — GIL released for most of the work."""
        idx, split = args
        with Timer("load_bg", logger=None):
            bg_bgr = bgs.random()
        if bg_bgr is None:
            return

        if random.random() < 0.05:
            # Negative sample — background only, empty label
            composite, label = make_sample(None, bg_bgr)
        else:
            with Timer("load_card", logger=None):
                card = cards.random()
            composite, label = make_sample(card, bg_bgr)

        name = f"{idx:06d}"
        img_path = Path(out_dir) / "images" / split / f"{name}.jpg"
        lbl_path = Path(out_dir) / "labels" / split / f"{name}.txt"
        with Timer("save", logger=None):
            cv2.imwrite(str(img_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if label is not None:
                # Pose format: class cx cy w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
                corners = label.reshape(4, 2)   # TL TR BR BL, normalised
                xs, ys  = corners[:, 0], corners[:, 1]
                cx = float((xs.min() + xs.max()) / 2)
                cy = float((ys.min() + ys.max()) / 2)
                bw = float(xs.max() - xs.min())
                bh = float(ys.max() - ys.min())
                kpts = " ".join(
                    f"{corners[i,0]:.6f} {corners[i,1]:.6f} 2"
                    for i in range(4)
                )
                label_str = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kpts}"
            else:
                label_str = ""
            lbl_path.write_text(label_str)

    # Build the full work list: (global_idx, split)
    work = []
    idx = 0
    for split, n in counts.items():
        for _ in range(n):
            work.append((idx, split))
            idx += 1

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        list(tqdm(ex.map(make_one, work), total=len(work), desc="generating"))

    # dataset.yaml
    yaml_path = Path(out_dir) / "dataset.yaml"
    yaml_path.write_text(
        f"path: {Path(out_dir).resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 1\nnames: ['card']\n"
        f"kpt_shape: [4, 3]\n"  # 4 corners, each with x y visibility
        f"task: pose\n"
    )
    print(f"\nDone — {total} images in {out_dir}/")
    print(f"dataset.yaml written to {yaml_path}")
    print("\nTiming summary:")
    t = Timer.timers
    for name in t.keys():
        print(f"  {name:12s}  total={t[name]*1000:.0f}ms  "
              f"mean={t.mean(name)*1000:.1f}ms  "
              f"min={t.min(name)*1000:.1f}ms  "
              f"max={t.max(name)*1000:.1f}ms  "
              f"n={t.count(name)}")


# ── Geometry helpers (reusable) ──────────────────────────────────────────────

def warp_to_rect(img, pts_px, out_w, out_h):
    """
    Perspective-warp 4 corners (TL TR BR BL, pixel coords) to an upright
    out_w × out_h rectangle.
    """
    src = pts_px.astype(np.float32)
    dst = np.array([
        [0,       0       ],
        [out_w-1, 0       ],
        [out_w-1, out_h-1 ],
        [0,       out_h-1 ],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


# ── OBB visualisation helper (reusable) ──────────────────────────────────────

def show_pose(img_bgr, label, window="Pose preview"):
    """
    Draw the 4 corner keypoints onto img_bgr and show it.
    label: float32 (8,) array of normalised TL TR BR BL corners, or None for negatives.
    Press any key to advance, 'q' to quit.
    Returns False if the user pressed 'q', True otherwise.
    """
    vis = img_bgr.copy()

    if label is not None:
        S = img_bgr.shape[1]
        H = img_bgr.shape[0]
        pts = label.reshape(4, 2).copy()
        pts[:, 0] *= S
        pts[:, 1] *= H
        pts = pts.astype(np.int32)

        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        colors = [(0,0,255), (0,255,255), (255,0,0), (255,255,255)]
        labels = ["TL","TR","BR","BL"]
        for pt, color, lbl in zip(pts, colors, labels):
            cv2.circle(vis, tuple(pt), 6, color, -1)
            cv2.putText(vis, lbl, (pt[0]+6, pt[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        cv2.arrowedLine(vis, tuple(pts[0]), tuple(pts[1]),
                        (0, 255, 0), 2, tipLength=0.2)

    cv2.imshow(window, vis)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(window)
    return key != ord('q')


# ── Test mode ─────────────────────────────────────────────────────────────────

def test_preview(cards_root, bg_root):
    cards = CardDataset(lmdb_path=cards_root)
    bgs   = BackgroundDataset(lmdb_path=bg_root)
    print("Generating previews — press any key for next, 'q' to quit.")
    while True:
        card_bgr = cards.random()
        bg_bgr = bgs.random()
        composite, label = make_sample(card_bgr, bg_bgr)
        if not show_pose(composite, label):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total",    type=int, default=10000)
    parser.add_argument("--cards",    default="data/cards/cards.lmdb")
    parser.add_argument("--bgs",      default="backgrounds.lmdb")
    parser.add_argument("--out",      default="dataset")
    parser.add_argument("--workers",  type=int, default=14,
                        help="Number of threads (default: 14)")
    parser.add_argument("--test",     action="store_true",
                        help="Preview generated samples instead of saving")
    args = parser.parse_args()

    if args.test:
        test_preview(args.cards, args.bgs)
    else:
        generate(args.total, args.cards, args.bgs, args.out, args.workers)

