"""
train_recog.py  –  NN-cluster repulsion training.

Each epoch:
  1. Sample SAMPLES_PER_EPOCH random cards from the canonical set.
  2. Embed all of them (no-grad pass) → working pool of (index, embedding) pairs.
  3. Repeatedly:
       a. Take the first image in the pool as an anchor.
       b. Find its 31 nearest neighbours within the remaining pool (by L2 distance).
       c. Remove all 32 from the pool.
       d. Forward pass on the 32 with grad; compute pairwise L2 distances.
       e. Loss = mean relu(MARGIN - dist) over all unique pairs  →  backprop.
  4. Save checkpoint.  Repeat next epoch.
"""

import json
import lmdb
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

import timm
from card_hasher import MODEL_NAME, IMAGE_SIZE, _MEAN, _STD

# ── Config ────────────────────────────────────────────────────────────────────

CARDS_224         = "data/cards/cards_224.lmdb"
CANONICAL_INDEX   = "data/cards/canonical_index.json"
CHECKPOINT_DIR    = "runs/recog"
CLUSTER_SIZE      = 128        # anchor + 127 nearest neighbours
MARGIN_SLACK      = 0.2        # added on top of current min pairwise distance
LR                = 3e-5
EPOCHS            = 100
EMBED_BATCH       = 1024       # images per no-grad forward pass
IMG_WORKERS       = 16          # parallel image-decode threads


# ── Image helpers ─────────────────────────────────────────────────────────────

def _open_lmdb(path: str):
    return lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)


# Thread-local LMDB env cache — one persistent env per thread, avoids open/close overhead
_tls = threading.local()

def _tls_env(path: str) -> lmdb.Environment:
    if not hasattr(_tls, 'envs'):
        _tls.envs = {}
    if path not in _tls.envs:
        _tls.envs[path] = _open_lmdb(path)
    return _tls.envs[path]


def _decode_one(args) -> np.ndarray:
    """Decode a single image; called from worker threads."""
    lmdb_path, idx = args
    env = _tls_env(lmdb_path)
    with env.begin(buffers=True) as txn:
        raw = bytes(txn.get(str(idx).encode()))
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        bgr = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - _MEAN) / _STD).transpose(2, 0, 1)  # (3, H, W)


_executor = ThreadPoolExecutor(max_workers=IMG_WORKERS)


def _fetch_images(lmdb_path: str, indices: list) -> np.ndarray:
    """
    Load + preprocess images in parallel.
    Returns float32 (N, 3, H, W).
    """
    args = [(lmdb_path, idx) for idx in indices]
    return np.stack(list(_executor.map(_decode_one, args)))


# ── Embedding ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def embed_all(model: nn.Module, lmdb_path: str, indices: list,
              device: torch.device) -> np.ndarray:
    """
    Embed every index in batches.
    Returns L2-normalised float32 array of shape (N, D).

    Prefetch: while the GPU processes batch N, batch N+1 is decoded in parallel.
    """
    model.eval()
    starts = list(range(0, len(indices), EMBED_BATCH))
    if not starts:
        return np.empty((0,), dtype=np.float32)

    def _submit(start):
        batch = indices[start: start + EMBED_BATCH]
        return _executor.submit(_fetch_images, lmdb_path, batch)

    out = []
    prefetch = _submit(starts[0])

    for i, start in enumerate(tqdm(starts, desc="  embedding", leave=False)):
        imgs_np = prefetch.result()
        # Start fetching next batch immediately
        if i + 1 < len(starts):
            prefetch = _submit(starts[i + 1])

        x = torch.from_numpy(imgs_np).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            e = model(x).float()
        e = F.normalize(e, dim=1).cpu().numpy()
        out.append(e)

    return np.vstack(out)                               # (N, D)


# ── Loss ──────────────────────────────────────────────────────────────────────

def repulsion_loss(embs: torch.Tensor, margin: float) -> torch.Tensor:
    """
    Push every pair of embeddings at least `margin` apart.

    embs  : (K, D)  –  L2-normalised  (K = CLUSTER_SIZE)
    loss  : mean relu(margin - ||a - b||₂)  over all K*(K-1)/2 unique pairs
    """
    # torch.cdist is numerically stable (avoids sqrt-of-zero gradient NaN)
    dists    = torch.cdist(embs, embs, p=2)                          # (K, K)
    mask     = torch.triu(torch.ones(embs.size(0), embs.size(0),
                                     device=embs.device, dtype=torch.bool), diagonal=1)
    pen      = F.relu(margin - dists[mask])                          # (K*(K-1)/2,)
    violated = pen[pen > 0]
    # Mean over violated pairs only — avoids dilution from already-satisfied pairs.
    # Falls back to zero (with grad) if nothing is violated.
    return violated.mean() if violated.numel() > 0 else pen.sum()


# ── Dynamic margin ───────────────────────────────────────────────────────────

def min_pairwise_dist(embs: np.ndarray, device: torch.device,
                      row_batch: int = 512) -> float:
    """
    Minimum nearest-neighbour L2 distance across all embeddings.
    Uses torch.cdist in row batches on the same device as training.
    """
    N = len(embs)
    t = torch.from_numpy(embs).to(device)
    min_d = torch.full((N,), float("inf"), device=device)

    for start in range(0, N, row_batch):
        end  = min(start + row_batch, N)
        rows = t[start:end]
        d    = torch.cdist(rows, t, p=2)
        for local_i in range(end - start):
            d[local_i, start + local_i] = float("inf")
        min_d[start:end] = d.min(dim=1).values

    return float(min_d.min().item())



def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    canonical: list = json.loads(Path(CANONICAL_INDEX).read_text())
    print(f"Canonical set: {len(canonical):,} images")

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model = model.to(device=device, dtype=torch.float32)
    print("Model loaded.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    out_dir = Path(CHECKPOINT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()

    for epoch in range(1, EPOCHS + 1):

        # ── Step 1: use full canonical set ─────────────────────────────────────
        epoch_indices: list = list(canonical)
        n = len(epoch_indices)
        print(f"\nEpoch {epoch}/{EPOCHS}  –  {n:,} canonical images")

        # ── Step 2: embed (no grad) ────────────────────────────────────────────
        embs_np = embed_all(model, CARDS_224, epoch_indices, device)     # (N, D)

        # ── Step 2b: set margin = min pairwise dist + slack ───────────────────
        epoch_min = min_pairwise_dist(embs_np, device)
        margin = epoch_min + MARGIN_SLACK
        print(f"  min_dist={epoch_min:.4f}  margin={margin:.4f}")

        # ── Step 3: build mutable pool ────────────────────────────────────────
        # Keep embeddings in a pre-allocated GPU tensor; delete by swap-to-tail
        # so removal is O(CLUSTER_SIZE) with no reallocation.
        pool_embs_gpu = torch.from_numpy(embs_np).to(device)   # (N, D) – mutated in-place
        pool_idx_arr  = np.array(epoch_indices, dtype=np.int64) # (N,)   – mutated in-place
        active        = n

        total_loss  = 0.0
        total_viol  = 0
        total_pairs = 0
        n_batches   = 0

        expected_batches = n // CLUSTER_SIZE
        pbar = tqdm(total=expected_batches, desc=f"  Epoch {epoch}")

        model.train()

        def _next_cluster():
            """GPU nearest-neighbour search; returns (lmdb_ids, positions)."""
            anchor_pos = int(rng.integers(active))
            with torch.no_grad():
                dists = torch.cdist(
                    pool_embs_gpu[anchor_pos:anchor_pos + 1],
                    pool_embs_gpu[:active],
                ).squeeze(0)                                   # (active,)
            pos      = torch.topk(dists, CLUSTER_SIZE, largest=False).indices.cpu().tolist()
            lmdb_ids = [int(pool_idx_arr[p]) for p in pos]
            return lmdb_ids, pos

        # Bootstrap: compute first cluster and start its fetch before the loop
        cluster_lmdb, cluster_pos = _next_cluster()
        prefetch = _executor.submit(_fetch_images, CARDS_224, cluster_lmdb)

        # ── Step 4: consume pool ───────────────────────────────────────────────
        while True:
            # Collect images fetched during last GPU step (or bootstrap)
            imgs_np = prefetch.result()

            # Remove current cluster from pool (swap-to-tail, O(CLUSTER_SIZE))
            for p in sorted(cluster_pos, reverse=True):
                last = active - 1
                pool_embs_gpu[p] = pool_embs_gpu[last]
                pool_idx_arr[p]  = pool_idx_arr[last]
                active -= 1

            # If pool is big enough, compute NEXT cluster and prefetch its images
            # NOW — this runs in parallel with the GPU step below
            if active >= CLUSTER_SIZE:
                cluster_lmdb, cluster_pos = _next_cluster()
                prefetch = _executor.submit(_fetch_images, CARDS_224, cluster_lmdb)
            else:
                prefetch = None

            # ── Forward pass with grad ─────────────────────────────────────────
            x = torch.from_numpy(imgs_np).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                e = model(x).float()
            e = F.normalize(e, dim=1)                  # (K, D)

            loss = repulsion_loss(e, margin)
            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # ── Stats ──────────────────────────────────────────────────────────
            with torch.no_grad():
                K    = e.size(0)
                dm   = torch.cdist(e, e, p=2)
                mask = torch.triu(torch.ones(K, K, device=device, dtype=torch.bool), diagonal=1)
                pd   = dm[mask]
                viol = int((pd < margin).sum().item())
                np_  = int(pd.numel())

            total_loss  += loss.item()
            total_viol  += viol
            total_pairs += np_
            n_batches   += 1

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             viol=f"{viol}/{np_}",
                             pool=active)

            if prefetch is None:
                break  # pool exhausted, no next cluster

        pbar.close()

        avg_loss  = total_loss  / max(n_batches, 1)
        viol_rate = total_viol  / max(total_pairs, 1)
        print(f"  avg_loss={avg_loss:.4f}  violated={viol_rate:.1%}"
              f"  batches={n_batches}  leftover={active}")

        ckpt = out_dir / f"epoch_{epoch:03d}.pt"
        torch.save(model.state_dict(), ckpt)
        torch.save(model.state_dict(), out_dir / "last.pt")
        print(f"  Saved → {ckpt}")


if __name__ == "__main__":
    train()
