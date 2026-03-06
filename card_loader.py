"""
card_loader.py  –  Thread-pooled LMDB image loader for training.

CardLoader             : abstract base class.
Card224Loader          : clean images from cards_224.lmdb.
AugmentedCard224Loader : augmented images from cards_224.lmdb.

RECOG_AUG_PIPELINE : the standard augmentation pipeline used in train_recog.py.
"""

from functools import partial
import threading
import lmdb
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Generator

from card_hasher import _MEAN, _STD
from augmentations import (
    AugmentationPipeline, color_jitter, white_balance,
    add_noise, gaussian_blur,
)

_CARDS_224        = "data/cards/cards_224.lmdb"
_CANONICAL_INDEX  = "data/cards/canonical_index.json"
_IMG_WORKERS      = 16

# ── Standard augmentation pipeline ───────────────────────────────────────────

RECOG_AUG_PIPELINE = AugmentationPipeline([
    (color_jitter,                    100),
    (add_noise,                       100),
    (white_balance,                   100),
    (partial(gaussian_blur, max_r=3), 100),
])


def prefetch_stream(getter: Callable) -> Generator:
    """
    Calls getter() once synchronously, then re-submits it async before each yield.
    Stops when getter() returns None.
    """
    with ThreadPoolExecutor(max_workers=1) as ex:
        result = getter()
        while result is not None:
            fut = ex.submit(getter)
            yield result
            result = fut.result()


# ── Base loader ───────────────────────────────────────────────────────────────

class CardLoader:
    """
    Base class: thread-pooled LMDB image loader.

    Subclasses set lmdb_path, workers, mean, std and optionally override _augment().
    All images are returned as float32 (C, H, W), mean/std normalised.
    """

    def __init__(
        self,
        lmdb_path:  str,
        workers:    int,
        index_path: str | None = None,
        mean:       np.ndarray = _MEAN,
        std:        np.ndarray = _STD,
    ):
        self.lmdb_path  = lmdb_path
        self.mean       = mean
        self.std        = std
        self._index_path = index_path
        self._executor  = ThreadPoolExecutor(max_workers=workers)
        self._tls       = threading.local()
        self._indices: list | None = None

    def _env(self) -> lmdb.Environment:
        """Return (or open) a thread-local LMDB env for this loader's path."""
        if not hasattr(self._tls, 'envs'):
            self._tls.envs = {}
        if self.lmdb_path not in self._tls.envs:
            self._tls.envs[self.lmdb_path] = lmdb.open(
                self.lmdb_path,
                readonly=True, lock=False, readahead=False, meminit=False,
            )
        return self._tls.envs[self.lmdb_path]

    def _process_image(self, bgr: np.ndarray) -> np.ndarray:
        """No-op in base class; override in subclasses."""
        return bgr

    def _fetch(self, indices: list) -> np.ndarray:
        return self._fetch_with_original(indices)[0]

    def _fetch_with_original(self, indices: list) -> tuple[np.ndarray, np.ndarray]:
        """Fetch processed + original (pre-_process_image) tensors for each index."""
        def decode_pair(idx):
            env = self._env()
            with env.begin(buffers=True) as txn:
                raw = bytes(txn.get(str(idx).encode()))
            bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            def to_tensor(b):
                rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                return ((rgb - self.mean) / self.std).transpose(2, 0, 1)
            return to_tensor(self._process_image(bgr)), to_tensor(bgr)
        pairs = list(self._executor.map(decode_pair, indices))
        return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def all_indices(self) -> list:
        """All indices from the associated index file (cached)."""
        if self._indices is None:
            if self._index_path is None:
                raise RuntimeError("No index_path set on this loader.")
            import json
            self._indices = json.loads(open(self._index_path).read())
        return self._indices

    def __len__(self) -> int:
        return len(self.all_indices)

    def fetch(self, indices: list) -> np.ndarray:
        """Blocking fetch. Returns float32 (N, 3, H, W)."""
        return self._fetch(indices)

    def fetch_async(self, indices: list) -> Future:
        """Non-blocking fetch. Returns a Future[np.ndarray]."""
        return self._executor.submit(self._fetch, indices)

    def stream(self, batch_size: int, indices: list | None = None):
        """Yield (batch_indices, images) in chunks of batch_size."""
        for idx, processed, _ in self.stream_with_original(batch_size, indices):
            yield idx, processed

    def stream_with_original(self, batch_size: int, indices: list | None = None,
                              device: torch.device | None = None):
        """
        Like stream(), but yields (batch_indices, processed_images, original_images).
        Each image is read from LMDB exactly once.
        If device is given, processed and original arrays are returned as tensors on that device.
        """
        indices = indices if indices is not None else self.all_indices
        starts = list(range(0, len(indices), batch_size))
        if not starts:
            return
        ready = self._executor.submit(self._fetch_with_original, indices[starts[0]: starts[0] + batch_size])
        for i, start in enumerate(starts):
            if i + 1 < len(starts):
                nxt = starts[i + 1]
                prefetching = self._executor.submit(self._fetch_with_original, indices[nxt: nxt + batch_size])
            else:
                prefetching = None
            processed_batch, original_batch = ready.result()
            if device is not None:
                processed_batch = torch.from_numpy(processed_batch).to(device)
                original_batch  = torch.from_numpy(original_batch).to(device)
            yield indices[start: start + batch_size], processed_batch, original_batch
            ready = prefetching


# ── Concrete loaders ──────────────────────────────────────────────────────────

class Card224Loader(CardLoader):
    """Clean images from cards_224.lmdb."""

    def __init__(self):
        super().__init__(_CARDS_224, _IMG_WORKERS, index_path=_CANONICAL_INDEX)


class AugmentedCard224Loader(CardLoader):
    """Augmented images from cards_224.lmdb."""

    def __init__(self, pipeline: AugmentationPipeline = RECOG_AUG_PIPELINE):
        super().__init__(_CARDS_224, _IMG_WORKERS, index_path=_CANONICAL_INDEX)
        self._pipeline = pipeline

    def _process_image(self, bgr: np.ndarray) -> np.ndarray:
        return self._pipeline(bgr)


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    N = 8
    clean_loader = Card224Loader()
    aug_loader   = AugmentedCard224Loader()
    indices = clean_loader.all_indices[:N]

    clean_imgs = clean_loader.fetch(indices)   # (N, 3, H, W)
    aug_imgs   = aug_loader.fetch(indices)     # (N, 3, H, W)

    # Denormalise: (3,H,W) float → (H,W,3) uint8
    def to_rgb(img: np.ndarray) -> np.ndarray:
        rgb = (img.transpose(1, 2, 0) * _STD + _MEAN).clip(0, 1)
        return (rgb * 255).astype(np.uint8)

    fig, axes = plt.subplots(2, N, figsize=(N * 2, 5))
    for i in range(N):
        axes[0, i].imshow(to_rgb(clean_imgs[i]))
        axes[0, i].axis("off")
        axes[0, i].set_title(str(indices[i]), fontsize=7)

        axes[1, i].imshow(to_rgb(aug_imgs[i]))
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("clean",     fontsize=9)
    axes[1, 0].set_ylabel("augmented", fontsize=9)
    plt.tight_layout()
    plt.show()
