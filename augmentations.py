"""
augmentations.py
----------------
Image augmentation primitives used during synthetic dataset generation.

Each augmentation is a plain function: img -> img.

Compose them with AugmentationPipeline, setting chance (0-100) per entry:
    pipe = AugmentationPipeline([
        (color_jitter,  100),
        (white_balance, 100),
        (gaussian_blur,  30),
    ])
    img = pipe(img)
"""

import random
import math
import cv2
import numpy as np
from codetiming import Timer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _skewed(max_val: float) -> float:
    """Value in [0, max_val] heavily weighted toward 0 (square of uniform)."""
    return (random.random() ** 2) * max_val


# ── Pre-computed structures ───────────────────────────────────────────────────

# add_noise: pre-scaled 2D pools for every amplitude value (2–12).
# Each pool row is independent; rows selected via fancy indexing — no Python loop.
_NOISE_POOL_ROWS = 4096
_NOISE_POOL_COLS = 4096  # covers images up to 682 px wide * 3 channels

def _build_noise_pools(rows: int, cols: int) -> dict:
    """Return {amplitude: uint8 array (rows, cols, 2)} where [...,0]=pos, [...,1]=neg."""
    base = np.random.randint(-12, 13, (rows, cols), dtype=np.int8)
    pools = {}
    for amp in range(2, 13):
        scaled = np.round(base * (amp / 12.0)).astype(np.int8)
        pos = np.where(scaled > 0, scaled, 0).astype(np.uint8)
        neg = np.where(scaled < 0, -scaled, 0).astype(np.uint8)
        pools[amp] = (pos, neg)
    return pools

_NOISE_POOLS = _build_noise_pools(_NOISE_POOL_ROWS, _NOISE_POOL_COLS)

# motion_blur: pre-build kernels for all (length, angle_step) combinations
def _build_motion_kernels(max_length: int = 20, angle_steps: int = 24) -> dict:
    kernels = {}
    angles = [i * (360 / angle_steps) for i in range(angle_steps)]
    for length in range(2, max_length + 1):
        for angle in angles:
            k = np.zeros((length, length), dtype=np.float32)
            cv2.line(k, (0, length // 2), (length - 1, length // 2), 1.0, 1)
            M = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1.0)
            k = cv2.warpAffine(k, M, (length, length))
            s = k.sum()
            if s > 0:
                k /= s
            kernels[(length, angle)] = k
    return kernels

_MOTION_KERNELS = _build_motion_kernels()
_MOTION_ANGLES  = sorted({a for (_, a) in _MOTION_KERNELS})


# ── Colour / tone ─────────────────────────────────────────────────────────────

def color_jitter(img):
    """Random brightness, contrast, saturation and hue shift — applied via LUT."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # uint8 H,S,V

    # Hue: shift by a fixed offset (mod 180) — build a 256-entry LUT
    h_shift = int(random.uniform(-10, 10))
    h_lut = np.arange(256, dtype=np.int16)
    h_lut = ((h_lut + h_shift) % 180).clip(0, 255).astype(np.uint8)
    hsv[:, :, 0] = cv2.LUT(hsv[:, :, 0], h_lut)

    # Saturation and Value: scale by a fixed factor — build LUT per channel
    s_scale = random.uniform(0.7, 1.3)
    v_scale = random.uniform(0.6, 1.4)
    lut_s = np.clip(np.arange(256, dtype=np.float32) * s_scale, 0, 255).astype(np.uint8)
    lut_v = np.clip(np.arange(256, dtype=np.float32) * v_scale, 0, 255).astype(np.uint8)
    hsv[:, :, 1] = cv2.LUT(hsv[:, :, 1], lut_s)
    hsv[:, :, 2] = cv2.LUT(hsv[:, :, 2], lut_v)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def white_balance(img):
    """Random per-channel gain — applied via LUT, no float32 image allocation."""
    gains = np.random.uniform(0.8, 1.2, 3)
    lut_b = np.clip(np.arange(256, dtype=np.float32) * gains[0], 0, 255).astype(np.uint8)
    lut_g = np.clip(np.arange(256, dtype=np.float32) * gains[1], 0, 255).astype(np.uint8)
    lut_r = np.clip(np.arange(256, dtype=np.float32) * gains[2], 0, 255).astype(np.uint8)
    b, g, r = cv2.split(img)
    return cv2.merge([cv2.LUT(b, lut_b), cv2.LUT(g, lut_g), cv2.LUT(r, lut_r)])


def add_noise(img):
    """Additive uniform noise — rows drawn from a pre-scaled 2D pool via fancy indexing."""
    h, w = img.shape[:2]
    amplitude = int(random.uniform(2, 12))
    pos_pool, neg_pool = _NOISE_POOLS[amplitude]
    row_idx = np.random.randint(0, _NOISE_POOL_ROWS, h)
    col_off = np.random.randint(0, _NOISE_POOL_COLS - w * 3)
    # Fancy-index rows, slice columns at a random offset, reshape — all in C; no Python loop.
    pos = pos_pool[row_idx, col_off : col_off + w * 3].reshape(h, w, 3)
    neg = neg_pool[row_idx, col_off : col_off + w * 3].reshape(h, w, 3)
    return cv2.subtract(cv2.add(img, pos), neg)


# ── Blur ──────────────────────────────────────────────────────────────────────

def gaussian_blur(img):
    """Defocus blur — radius 0–6 px, small radii most common."""
    r = _skewed(6)
    if r < 0.5:
        return img
    k = int(r) * 2 + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def motion_blur(img):
    """Directional motion blur — kernel selected from pre-computed table."""
    length = int(_skewed(20))
    if length < 2:
        return img
    angle = random.choice(_MOTION_ANGLES)
    return cv2.filter2D(img, -1, _MOTION_KERNELS[(length, angle)])


# ── Surface effects (card-only) ───────────────────────────────────────────────

def glare(img):
    """Elliptical specular highlight."""
    h, w = img.shape[:2]
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(h // 4, 3 * h // 4)
    rx = int(random.uniform(0.1, 0.4) * w)
    ry = int(random.uniform(0.1, 0.4) * h)
    intensity = random.uniform(0.3, 0.8)
    Y, X = np.ogrid[:h, :w]
    dist = ((X - cx) / max(rx, 1)) ** 2 + ((Y - cy) / max(ry, 1)) ** 2
    mask = np.clip(1.0 - dist, 0, 1).astype(np.float32)[:, :, np.newaxis]
    out = img.astype(np.float32)
    out += mask * (intensity * 255)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def shade(img):
    """Directional gradient shadow across the card."""
    h, w = img.shape[:2]
    darkness = random.uniform(0.25, 0.65)
    rad      = math.radians(random.uniform(0, 360))
    # Use broadcasting instead of mgrid — avoids allocating two int64 h×w arrays
    X = np.linspace(0, 1, w, dtype=np.float32)
    Y = np.linspace(0, 1, h, dtype=np.float32)
    proj = (X - 0.5) * math.cos(rad) + (Y - 0.5)[:, np.newaxis] * math.sin(rad)
    proj -= proj.min()
    proj /= proj.max() + 1e-6
    s = (1.0 - proj * (1.0 - darkness))[:, :, np.newaxis]
    out = img.astype(np.float32)
    out *= s
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def foil(img):
    """
    Iridescent rainbow foil sheen.
    Four layered effects: hue sweep, saturation boost, tinted specular, sparkle noise.
    """
    h, w = img.shape[:2]
    out  = img.astype(np.float32)

    # 1. Rainbow hue sweep — use linspace + broadcasting instead of mgrid
    rad  = math.radians(random.uniform(0, 360))
    X    = np.linspace(0, 1, w, dtype=np.float32)
    Y    = np.linspace(0, 1, h, dtype=np.float32)
    proj = X * math.cos(rad) + Y[:, np.newaxis] * math.sin(rad)
    proj -= proj.min()
    proj /= proj.max() + 1e-6
    hue_map = ((random.uniform(0, 180) + proj * random.uniform(30, 90)) % 180).astype(np.float32)

    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    blend = random.uniform(0.25, 0.60)
    hsv[:, :, 0] = (hsv[:, :, 0] * (1 - blend) + hue_map * blend) % 180

    # 2. Saturation boost — clip in-place
    hsv[:, :, 1] *= random.uniform(1.15, 1.50)
    np.clip(hsv[:, :, 1], 0, 255, out=hsv[:, :, 1])
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # 3. Tinted specular highlight — reuse X/Y from above
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(h // 4, 3 * h // 4)
    rx = int(random.uniform(0.08, 0.30) * w)
    ry = int(random.uniform(0.08, 0.30) * h)
    Xp = np.linspace(0, w - 1, w, dtype=np.float32)
    Yp = np.linspace(0, h - 1, h, dtype=np.float32)
    dist      = ((Xp - cx) / max(rx, 1)) ** 2 + ((Yp - cy) / max(ry, 1))[:, np.newaxis] ** 2
    spec_mask = np.clip(1.0 - dist, 0, 1, dtype=np.float32)[:, :, np.newaxis]
    tint      = np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)],
                         dtype=np.float32) * (255 * random.uniform(0.15, 0.45))
    out += spec_mask * tint

    # 4. Sparkle noise
    n  = int(random.uniform(0.001, 0.005) * w * h)
    sy = np.random.randint(0, h, n)
    sx = np.random.randint(0, w, n)
    out[sy, sx] += np.random.uniform(0.3, 1.0, n).astype(np.float32)[:, np.newaxis] * 255

    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class AugmentationPipeline:
    """
    Apply a list of (fn, chance) pairs in sequence.
    `chance` is an integer 0–100; the function is skipped if a random roll exceeds it.
    """
    def __init__(self, augmentations: list[tuple]):
        self.augmentations = augmentations

    def __call__(self, img: np.ndarray) -> np.ndarray:
        for fn, chance in self.augmentations:
            if random.random() * 100 <= chance:
                with Timer(f"Augment: {fn.__name__}", logger=None):
                    img = fn(img)
        return img


