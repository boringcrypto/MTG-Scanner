"""
streaming_dataset.py
--------------------
A streaming Pose dataset that generates samples on-the-fly in memory —
no files written, no JPEG encode/decode cycle between generation and training.

Plugs into Ultralytics by subclassing YOLODataset and overriding:
  - get_img_files       : returns N dummy paths so BaseDataset knows the length
  - get_labels          : returns N stub dicts (no actual annotation yet)
  - get_image_and_label : generates image + label together, bypassing deepcopy/stub
  - cache_labels        : no-op, prevents .cache file writes

Each DataLoader worker independently calls make_sample(), so generation is
fully parallel across workers.

Labels use 4 corner keypoints (TL TR BR BL) so the model predicts exact
quad corners, not just a rotated rectangle.
"""

import os
import random
import numpy as np
from ultralytics.utils.instance import Instances

from ultralytics.data.dataset import YOLODataset

from datasets import CardDataset, BackgroundDataset
from make_training_set import make_sample


# ── Streaming dataset ─────────────────────────────────────────────────────────

class StreamingPoseDataset(YOLODataset):
    """
    Generates 640×640 Pose samples in memory on every __getitem__ call.
    Predicts 4 corner keypoints (TL TR BR BL) for exact quad extraction.

    Args:
        epoch_size:  Number of samples to present per epoch.
        card_root:   Path to cards.lmdb
        bg_root:     Path to backgrounds.lmdb
        **kwargs:    Forwarded to YOLODataset / BaseDataset (imgsz, augment, hyp, …)
    """

    def __init__(self, epoch_size: int, card_root: str, bg_root: str, **kwargs):
        self.epoch_size = epoch_size
        self.card_root  = card_root
        self.bg_root    = bg_root
        self._cards = None
        self._bgs   = None
        print(f"[StreamingPoseDataset.__init__] pid={os.getpid()} epoch_size={epoch_size} — calling super().__init__")
        super().__init__(img_path=".", **kwargs)
        print(f"[StreamingPoseDataset.__init__] pid={os.getpid()} super().__init__ done")

    def _ensure_datasets(self):
        """Open LMDB handles lazily — called inside each worker process."""
        if self._cards is None:
            print(f"[StreamingPoseDataset._ensure_datasets] pid={os.getpid()} opening LMDB")
            self._cards = CardDataset(lmdb_path=self.card_root)
        if self._bgs is None:
            self._bgs = BackgroundDataset(lmdb_path=self.bg_root)
            print(f"[StreamingPoseDataset._ensure_datasets] pid={os.getpid()} LMDB open")

    # ── Ultralytics hook overrides ────────────────────────────────────────────

    def get_img_files(self, img_path) -> list[str]:
        """Return dummy paths — only the count matters for __len__."""
        print(f"[StreamingPoseDataset.get_img_files] pid={os.getpid()} returning {self.epoch_size} dummy paths")
        return [f"__stream_{i}__" for i in range(self.epoch_size)]

    def get_labels(self) -> list[dict]:
        """Return minimal stub dicts — real data is generated in get_image_and_label."""
        print(f"[StreamingPoseDataset.get_labels] pid={os.getpid()} building {self.epoch_size} stubs")
        stubs = [
            {
                "im_file": f,
                "shape": (640, 640),
                "cls": np.zeros((0, 1), dtype=np.float32),
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "segments": [],
                "keypoints": np.zeros((0, 4, 3), dtype=np.float32),
                "normalized": True,
                "bbox_format": "xywh",
            }
            for f in self.im_files
        ]
        print(f"[StreamingPoseDataset.get_labels] pid={os.getpid()} done")
        return stubs

    def cache_labels(self, path=None) -> dict:
        print(f"[StreamingPoseDataset.cache_labels] pid={os.getpid()} called (no-op, path={path})")
        labels = self.get_labels()
        n = len(labels)
        return {
            "labels":  labels,
            "results": (n, 0, 0, 0, n),
            "msgs":    [],
            "version": 0,
            "hash":    "",
        }

    def get_image_and_label(self, index: int) -> dict:
        """
        Generate image + label together in one call.

        Overrides BaseDataset.get_image_and_label so we never touch the stub
        in self.labels[index].  Returns the same dict structure as the real
        YOLODataset pose pipeline: xywh bbox + (N, 4, 3) keypoints.

        label_arr: float32 (8,) normalised TL TR BR BL corners from make_sample.
        """
        self._ensure_datasets()

        bg = self._bgs.random()
        if random.random() < 0.05:
            img, label_arr = make_sample(None, bg)
        else:
            card = self._cards.random()
            img, label_arr = make_sample(card, bg)

        h, w = img.shape[:2]

        if label_arr is not None:
            # Corners: (4, 2) normalised, TL TR BR BL
            corners = label_arr.reshape(4, 2).astype(np.float32)
            xs, ys  = corners[:, 0], corners[:, 1]
            cx = float((xs.min() + xs.max()) / 2)
            cy = float((ys.min() + ys.max()) / 2)
            bw = float(xs.max() - xs.min())
            bh = float(ys.max() - ys.min())
            xywh = np.array([[cx, cy, bw, bh]], dtype=np.float32)  # (1, 4)
            cls  = np.zeros((1, 1), dtype=np.float32)
            # Keypoints: (1, 4, 3) — x, y, visibility=2 (visible)
            kpts = np.ones((1, 4, 3), dtype=np.float32) * 2
            kpts[0, :, :2] = corners
            instances = Instances(xywh, np.zeros((1, 0, 2), dtype=np.float32), kpts, bbox_format="xywh", normalized=True)
        else:
            instances = Instances(
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 0, 2), dtype=np.float32),
                np.zeros((0, 4, 3), dtype=np.float32),
                bbox_format="xywh", normalized=True,
            )
            cls = np.zeros((0, 1), dtype=np.float32)

        return {
            "im_file":      self.im_files[index],
            "ori_shape":    (h, w),
            "resized_shape":(h, w),
            "ratio_pad":    (1.0, 1.0),
            "img":          img,
            "cls":          cls,
            "instances":    instances,
        }
