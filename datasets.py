import random
import cv2
import lmdb
import numpy as np
from torch.utils.data import Dataset
from corner_alpha import add_alpha
from codetiming import Timer


class CardDataset(Dataset):
    """
    Reads pre-processed portrait cards from a single LMDB file.
    Each value is a JPEG-encoded BGR image; add_alpha() is applied on load.
    """
    def __init__(self, lmdb_path="data/cards/cards.lmdb"):
        self._lmdb_path = lmdb_path
        self.env = None  # opened lazily per worker
        _env = lmdb.open(lmdb_path, readonly=True, lock=False,
                         readahead=False, meminit=False)
        with _env.begin() as txn:
            self._len = int(txn.get(b"__len__").decode())
        _env.close()

    def _open_env(self):
        if self.env is None:
            self.env = lmdb.open(self._lmdb_path, readonly=True,
                                 lock=False, readahead=False, meminit=False)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        self._open_env()
        with Timer("card_fetch", logger=None):
            with self.env.begin() as txn:
                data = txn.get(str(idx).encode())
        with Timer("card_imread", logger=None):
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        with Timer("card_add_alpha", logger=None):
            return add_alpha(img)

    def random(self):
        return self[random.randrange(self._len)]


class BackgroundDataset(Dataset):
    """
    Reads pre-processed background images from a single LMDB file.
    Each value is a JPEG-encoded BGR image; cropping is done at sample-generation time.
    """
    def __init__(self, lmdb_path="backgrounds.lmdb"):
        self._lmdb_path = lmdb_path
        self.env = None  # opened lazily per worker
        _env = lmdb.open(lmdb_path, readonly=True, lock=False,
                         readahead=False, meminit=False)
        with _env.begin() as txn:
            self._len = int(txn.get(b"__len__").decode())
        _env.close()

    def _open_env(self):
        if self.env is None:
            self.env = lmdb.open(self._lmdb_path, readonly=True,
                                 lock=False, readahead=False, meminit=False)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        self._open_env()
        with self.env.begin() as txn:
            data = txn.get(str(idx).encode())
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

    def random(self):
        return self[random.randrange(self._len)]


# ── Quick visual check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    ds = CardDataset()
    img = ds.random()
    if img is not None:
        cv2.imshow("Random portrait card", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
