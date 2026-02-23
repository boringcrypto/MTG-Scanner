"""
corner_alpha.py
---------------
Removes the rounded corners from MTG card images using the official
card dimensions:  2.5" × 3.5"  with a corner radius of 1/8" (0.125").

Corner radius as a fraction of card width = 0.125 / 2.5 = 0.05
So for any image:  r = round(width * 0.05)

A filled circle of radius r is subtracted from each corner, which matches
the physical card spec exactly — no flood-fill, no colour assumptions.

Alpha Edition cards have a slightly larger corner radius (~3/16").
Pass corner_radius_fraction=0.075 for those.

Usage (standalone):
    python corner_alpha.py --input C:/card_images/12345.jpg --show

Usage (as a module):
    from corner_alpha import add_alpha
    bgra = add_alpha(bgr_image)          # returns H×W×4 uint8
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

# MTG card corner radius = 1/8" on a 2.5"-wide card
MTG_CORNER_FRACTION         = 0.125 / 2.5   # ≈ 0.05  (Standard / modern)
MTG_CORNER_FRACTION_ALPHA   = 0.1875 / 2.5  # ≈ 0.075 (Alpha / Beta edition)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def add_alpha(
    bgr: np.ndarray,
    corner_radius_fraction: float = MTG_CORNER_FRACTION,
) -> np.ndarray:
    """
    Punch out the four rounded corners of a card image and return BGRA.

    Parameters
    ----------
    bgr                    : H×W×3 uint8 BGR image (as returned by cv2.imread)
    corner_radius_fraction : radius / card_width.
                             Use MTG_CORNER_FRACTION (0.05) for Standard/modern.
                             Use MTG_CORNER_FRACTION_ALPHA (0.075) for Alpha/Beta.

    Returns
    -------
    H×W×4 uint8 BGRA — corners are transparent (alpha = 0).
    """
    if bgr is None:
        raise ValueError("bgr image is None")

    h, w = bgr.shape[:2]
    r = max(1, round(w * corner_radius_fraction))

    # Start with a fully-opaque alpha channel
    alpha = np.full((h, w), 255, dtype=np.uint8)

    # Draw filled black circles at each corner centre.
    # The circle covers the region that should be transparent.
    # Corner centres are exactly (r, r) away from each corner.
    for (cx, cy) in [(r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)]:
        cv2.circle(alpha, (cx, cy), r, 0, thickness=-1)

    # Black out the rectangular strips beyond the circle centres so the
    # full corner square (not just the arc) is transparent.
    alpha[:r,    :r   ] = np.minimum(alpha[:r,    :r   ], alpha[:r,    :r   ])  # already handled by circle
    # Zero the four corner rectangles outside the circle
    alpha[:r,    :r   ] = 0
    alpha[:r,    w-r: ] = 0
    alpha[h-r:,  :r   ] = 0
    alpha[h-r:,  w-r: ] = 0

    # Restore pixels inside the circle arc within those rectangles
    # (the cv2.circle already drew the correct disc; re-apply it on top)
    for (cx, cy) in [(r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)]:
        cv2.circle(alpha, (cx, cy), r, 255, thickness=-1)

    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


# ---------------------------------------------------------------------------
# Convenience: process a file on disk → save as PNG (preserves alpha)
# ---------------------------------------------------------------------------

def process_file(src: Path, dst: Path, alpha_edition: bool = False):
    bgr = cv2.imread(str(src))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read {src}")
    fraction = MTG_CORNER_FRACTION_ALPHA if alpha_edition else MTG_CORNER_FRACTION
    bgra = add_alpha(bgr, corner_radius_fraction=fraction)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), bgra)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha-cut MTG card corners")
    parser.add_argument("--input",  required=True, help="Source card image")
    parser.add_argument("--output", default=None,  help="Destination PNG (default: <input>_alpha.png)")
    parser.add_argument("--alpha-edition", action="store_true",
                        help="Use larger Alpha/Beta corner radius (3/16\" instead of 1/8\")")
    parser.add_argument("--show", action="store_true",
                        help="Display result in a window")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output) if args.output else src.with_name(src.stem + "_alpha.png")

    bgr = cv2.imread(str(src))
    if bgr is None:
        raise SystemExit(f"Cannot read: {src}")

    fraction = MTG_CORNER_FRACTION_ALPHA if args.alpha_edition else MTG_CORNER_FRACTION
    bgra = add_alpha(bgr, corner_radius_fraction=fraction)
    cv2.imwrite(str(dst), bgra)
    print(f"Saved → {dst}")

    if args.show:
        # Checkerboard background so transparency is visible
        h, w = bgra.shape[:2]
        checker = np.zeros((h, w, 3), dtype=np.uint8)
        tile = 16
        for r in range(0, h, tile):
            for c in range(0, w, tile):
                colour = 200 if ((r // tile + c // tile) % 2 == 0) else 128
                checker[r:r+tile, c:c+tile] = colour

        alpha_f = bgra[:, :, 3:4].astype(np.float32) / 255.0
        bgr_f   = bgra[:, :, :3].astype(np.float32)
        chk_f   = checker.astype(np.float32)
        composite = (bgr_f * alpha_f + chk_f * (1 - alpha_f)).astype(np.uint8)

        cv2.imshow("Corner alpha (checkerboard)", composite)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
