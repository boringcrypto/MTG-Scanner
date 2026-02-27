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
MARGIN_SLACK      = 0.1        # added on top of current min pairwise distance
LR                = 3e-5
EPOCHS            = 100
EMBED_BATCH       = 1024       # images per no-grad forward pass
IMG_WORKERS       = 8          # parallel image-decode threads


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
    """
    model.eval()
    out = []
    for start in tqdm(range(0, len(indices), EMBED_BATCH),
                      desc="  embedding", leave=False):
        batch = indices[start: start + EMBED_BATCH]
        x = torch.from_numpy(_fetch_images(lmdb_path, batch)).to(device)
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
        pool_idx  = list(epoch_indices)
        pool_embs = list(embs_np)
        active    = n

        total_loss  = 0.0
        total_viol  = 0
        total_pairs = 0
        n_batches   = 0

        expected_batches = n // CLUSTER_SIZE
        pbar = tqdm(total=expected_batches, desc=f"  Epoch {epoch}")

        model.train()

        # ── Step 4: consume pool ───────────────────────────────────────────────
        while len(pool_idx) >= CLUSTER_SIZE:

            # Anchor = random pick from pool
            anchor_pos = rng.integers(len(pool_idx))
            anchor_emb = np.asarray(pool_embs[anchor_pos])  # (D,)
            pool_mat   = np.stack(pool_embs)                # (M, D)

            # L2 distances from anchor to every pool member (self = 0, closest)
            diff     = pool_mat - anchor_emb           # (M, D)
            dists_np = (diff * diff).sum(axis=1) ** 0.5  # (M,)

            # Grab the CLUSTER_SIZE positions with smallest distance (includes anchor)
            cluster_pos = np.argpartition(dists_np, CLUSTER_SIZE - 1)[:CLUSTER_SIZE].tolist()

            # Gather LMDB indices for this cluster
            cluster_lmdb = [pool_idx[p] for p in cluster_pos]

            # Remove cluster from pool
            for p in sorted(cluster_pos, reverse=True):
                del pool_idx[p]
                del pool_embs[p]

            # ── Forward pass with grad ─────────────────────────────────────────
            x = torch.from_numpy(_fetch_images(CARDS_224, cluster_lmdb)).to(device)
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
                             pool=len(pool_idx))

        pbar.close()

        avg_loss  = total_loss  / max(n_batches, 1)
        viol_rate = total_viol  / max(total_pairs, 1)
        print(f"  avg_loss={avg_loss:.4f}  violated={viol_rate:.1%}"
              f"  batches={n_batches}  leftover={len(pool_idx)}")

        ckpt = out_dir / f"epoch_{epoch:03d}.pt"
        torch.save(model.state_dict(), ckpt)
        torch.save(model.state_dict(), out_dir / "last.pt")
        print(f"  Saved → {ckpt}")


if __name__ == "__main__":
    train()
