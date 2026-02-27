"""
lmdb_writer.py
--------------
Shared utility: an auto-growing LMDB writer used by make_card_images.py,
get_backgrounds.py, and card_recognizer.py.
"""

from __future__ import annotations

import lmdb

_DEFAULT_INIT = 256 * 1024 ** 2   # 256 MB
_DEFAULT_GROW = 256 * 1024 ** 2   # 256 MB per growth step


class LmdbWriter:
    """
    Wraps an lmdb environment and handles batched writes with automatic
    map-size growth.

    Usage
    -----
    with LmdbWriter("out.lmdb") as w:
        w.put_batch({b"0": data0, b"1": data1, ...})
        w.put(b"__len__", b"42")

    Any extra keyword arguments are forwarded to every lmdb.open() call
    (e.g. lock=True, readahead=False, meminit=False).
    """

    def __init__(
        self,
        path:         str,
        initial_size: int = _DEFAULT_INIT,
        grow_size:    int = _DEFAULT_GROW,
        **open_kwargs,
    ):
        self.path        = path
        self.grow_size   = grow_size
        self._map_size   = initial_size
        self._open_kwargs = open_kwargs
        self.env         = lmdb.open(path, map_size=initial_size, **open_kwargs)

    # ------------------------------------------------------------------

    def put_batch(self, batch: dict[bytes, bytes]) -> None:
        """
        Write *batch* to the lmdb.  Automatically grows the map on
        MapFullError and retries up to 10 times.
        """
        for attempt in range(1, 10):
            try:
                with self.env.begin(write=True) as txn:
                    for k, v in batch.items():
                        txn.put(k, v)
                return
            except lmdb.MapFullError:
                self._map_size += self.grow_size
                print(
                    f"\nLMDB '{self.path}' full — growing to "
                    f"{self._map_size // 1024 ** 2} MB (attempt {attempt})"
                )
                self.env.close()
                self.env = lmdb.open(self.path, map_size=self._map_size, **self._open_kwargs)
        raise RuntimeError(
            f"LMDB '{self.path}' could not grow after 1000 attempts — disk may be full"
        )

    def put(self, key: bytes, value: bytes) -> None:
        """Write a single key/value pair (e.g. the final __len__ record)."""
        self.put_batch({key: value})

    # ------------------------------------------------------------------

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> "LmdbWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()
