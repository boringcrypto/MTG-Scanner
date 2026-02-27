"""
card_hasher.py

Perceptual hashing for MTG cards using ViT-Small (384-dim).

Takes a 256x256 BGR image (OpenCV) and returns a 384-bit hash by:
  1. Running the image through a pre-trained ViT-Small model
  2. Extracting the CLS token embedding (384 floats)
  3. Binarising with threshold = 0 (sign function)

The sign threshold is robust for the pre-trained model because ViT's final
LayerNorm centres each feature around 0, so the median of any feature across
a large corpus is close to 0. This gives roughly balanced 0/1 bit distributions
without any calibration data.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HASH_BITS        = 384          # matches ViT-Small embed_dim exactly
IMAGE_SIZE       = 224          # native ViT-Small pre-training resolution
MODEL_NAME       = "vit_small_patch16_224"
FINETUNED_WEIGHTS = "runs/recog/last.pt"  # loaded automatically if present

# ImageNet normalisation constants (float32, channel-first friendly)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CardHasher(nn.Module):
    """
    ViT-Small backbone that outputs a 384-bit binary hash for a BGR image.

    Single image
    ------------
    hasher = CardHasher()
    bits   = hasher.hash_bgr(bgr_array)        # np.ndarray uint8 (384,)
    packed = hasher.hash_bgr_bytes(bgr_array)  # bytes, 48 bytes

    Batch (recommended for index building)
    ---------------------------------------
    bits_batch = hasher.hash_batch_bgr(list_of_bgr)  # np.ndarray uint8 (B, 384)

    dtype defaults to bfloat16 when the GPU supports it (Ampere+), otherwise
    float32.  BF16 has the same dynamic range as FP32 so it is safe for ViT
    inference, and runs at full tensor-core speed on supported hardware.
    Pass dtype=torch.float32 to force FP32.
    """

    def __init__(
        self,
        pretrained: bool = True,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if dtype is None:
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        self.dtype = dtype

        # Load ViT-Small at its native 224×224 resolution — no positional
        # embedding interpolation needed.
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,      # remove classifier head → returns CLS token (384-d)
        )
        self.backbone.eval()
        self.backbone.to(device=self.device, dtype=self.dtype)

        # Load fine-tuned weights if available
        ft_path = Path(FINETUNED_WEIGHTS)
        if ft_path.exists():
            state = torch.load(ft_path, map_location=self.device, weights_only=True)
            self.backbone.load_state_dict(state)
            print(f"[CardHasher] Loaded fine-tuned weights from {ft_path}")
        else:
            print(f"[CardHasher] No fine-tuned weights found at {ft_path}, using pretrained.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bgr_to_tensor(self, bgrs: list[np.ndarray]) -> torch.Tensor:
        """
        Convert a list of BGR uint8 arrays (each 256×256 or any size —
        will be stretched to IMAGE_SIZE×IMAGE_SIZE) to a normalised
        (B, 3, H, W) float32 tensor on self.device.
        """
        frames = []
        for bgr in bgrs:
            if bgr.shape[0] != IMAGE_SIZE or bgr.shape[1] != IMAGE_SIZE:
                bgr = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
            # BGR → RGB, uint8 → float32 [0,1]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # ImageNet normalise in HWC then transpose to CHW
            rgb = (rgb - _MEAN) / _STD
            frames.append(rgb.transpose(2, 0, 1))   # (3, H, W)
        batch = np.stack(frames, axis=0)            # (B, 3, H, W)
        return torch.from_numpy(batch).to(device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed_batch_bgr(self, bgrs: list[np.ndarray]) -> np.ndarray:
        """
        Return raw float32 CLS embeddings for a batch of BGR images.
        Shape: (B, 384)
        """
        x = self._bgr_to_tensor(bgrs)       # (B, 3, H, W)
        return self.backbone(x).float().cpu().numpy()   # (B, 384) float32

    @torch.no_grad()
    def hash_batch_bgr(self, bgrs: list[np.ndarray]) -> np.ndarray:
        """
        Compute 384-bit hashes for a batch of BGR images in one forward pass.

        Parameters
        ----------
        bgrs : list of BGR uint8 numpy arrays (any size, will be resized)

        Returns
        -------
        np.ndarray of dtype uint8, shape (B, 384), values in {0, 1}.
        """
        embeddings = self.embed_batch_bgr(bgrs)     # (B, 384) float32
        return (embeddings > 0).astype(np.uint8)

    @torch.no_grad()
    def hash_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """Hash a single BGR image. Returns uint8 array of shape (384,)."""
        return self.hash_batch_bgr([bgr])[0]

    def hash_bgr_bytes(self, bgr: np.ndarray) -> bytes:
        """Hash a single BGR image and pack into 48 bytes."""
        return np.packbits(self.hash_bgr(bgr), bitorder="big").tobytes()

    @staticmethod
    def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
        """Hamming distance between two (384,) uint8 bit arrays."""
        return int(np.count_nonzero(a != b))

    @staticmethod
    def hamming_distance_bytes(a: bytes, b: bytes) -> int:
        """Hamming distance between two 48-byte packed hashes."""
        xor = np.bitwise_xor(
            np.frombuffer(a, dtype=np.uint8),
            np.frombuffer(b, dtype=np.uint8),
        )
        return int(np.unpackbits(xor).sum())


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print(f"Loading {MODEL_NAME} …")
    hasher = CardHasher(pretrained=True)
    print(f"  device : {hasher.device}")
    print(f"  dtype  : {hasher.dtype}")
    print(f"  params : {sum(p.numel() for p in hasher.backbone.parameters()):,}")

    dummy  = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    dummy2 = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    # Single image
    t0 = time.perf_counter()
    bits = hasher.hash_bgr(dummy)
    elapsed = time.perf_counter() - t0
    packed = hasher.hash_bgr_bytes(dummy)
    print(f"\nSingle hash:")
    print(f"  bits shape : {bits.shape}  dtype={bits.dtype}")
    print(f"  packed     : {len(packed)} bytes")
    print(f"  bit mean   : {bits.mean():.3f}  (≈0.5 is ideal)")
    print(f"  latency    : {elapsed*1000:.1f} ms")

    # Batch of 256
    batch = [np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
             for _ in range(256)]
    t0 = time.perf_counter()
    batch_bits = hasher.hash_batch_bgr(batch)
    elapsed = time.perf_counter() - t0
    print(f"\nBatch-256 hash:")
    print(f"  shape    : {batch_bits.shape}")
    print(f"  latency  : {elapsed*1000:.1f} ms  ({elapsed/256*1000:.2f} ms/img)")

    print(f"\nHamming distance (same image) : {CardHasher.hamming_distance(bits, hasher.hash_bgr(dummy))}")
    print(f"Hamming distance (random img) : {CardHasher.hamming_distance(bits, hasher.hash_bgr(dummy2))}")
