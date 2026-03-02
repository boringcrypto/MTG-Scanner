"""
train_recog.py  –  NN-cluster repulsion training.

Each epoch:
  1. Full embed_all → (N, D) snapshot; compute min pairwise dist → margin.
  2. Keep all N embeddings in a GPU tensor (no pool removal).
  3. Repeatedly:
       a. Pick the anchor with the globally lowest stale NN distance.
       b. Verify its true NN distance; if ≥ margin → trigger a full refresh.
       c. Forward pass on anchor + 127 nearest neighbours with grad.
       d. Loss = mean relu(margin - dist) over all violated pairs → backprop.
       e. Patch the 128 updated embeddings back into the GPU tensor (free).
       f. Every FULL_REFRESH_EVERY steps: full embed_all to correct drift.
  4. Epoch ends when a full refresh confirms all pairs satisfy margin.
  5. Save checkpoint.  Repeat next epoch.
"""

from functools import partial
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
from analyse_embeddings import embedding_report, invariance_report
from augmentations import (
    AugmentationPipeline, color_jitter, white_balance,
    add_noise, gaussian_blur, motion_blur,
)

# ── Augmentation pipeline (applied on uint8 BGR before normalisation) ──────────
_AUG_PIPELINE = AugmentationPipeline([
    (color_jitter,  100),
    (add_noise,     100),
    (white_balance, 100),
    (partial(gaussian_blur, max_r=3), 100),
])

# ── Config ────────────────────────────────────────────────────────────────────

CARDS_224         = "data/cards/cards_224.lmdb"
CANONICAL_INDEX   = "data/cards/canonical_index.json"
CHECKPOINT_DIR    = "runs/recog"
CLUSTER_SIZE      = 64        # anchor + 127 nearest neighbours
MARGIN_SLACK      = 0.1        # added on top of current min pairwise distance
LAMBDA_INV        = 0.1        # weight of invariance loss (aug → clean target)
LR                = 3e-5
EPOCHS            = 500
EMBED_BATCH          = CLUSTER_SIZE    # images per no-grad forward pass
IMG_WORKERS          = 16      # parallel image-decode threads
REFRESH_STEPS        = 10      # partial re-embed rounds per epoch


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
    lmdb_path, idx, augment = args
    env = _tls_env(lmdb_path)
    with env.begin(buffers=True) as txn:
        raw = bytes(txn.get(str(idx).encode()))
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        bgr = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE))
    if augment:
        bgr = _AUG_PIPELINE(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - _MEAN) / _STD).transpose(2, 0, 1)  # (3, H, W)


_executor = ThreadPoolExecutor(max_workers=IMG_WORKERS)


def _fetch_images(lmdb_path: str, indices: list, augment: bool = False) -> np.ndarray:
    """
    Load + preprocess images in parallel.
    Returns float32 (N, 3, H, W).
    """
    args = [(lmdb_path, idx, augment) for idx in indices]
    return np.stack(list(_executor.map(_decode_one, args)))


# ── Embedding ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def embed_all(model: nn.Module, lmdb_path: str, indices: list,
              device: torch.device, augment: bool = False) -> np.ndarray:
    """
    Embed every index in batches (no grad).
    Returns L2-normalised float32 array of shape (N, D).
    Used for partial re-embeds during the inner loop.

    Prefetch: while the GPU processes batch N, batch N+1 is decoded in parallel.
    """
    was_training = model.training
    model.eval()
    starts = list(range(0, len(indices), EMBED_BATCH))
    if not starts:
        if was_training:
            model.train()
        return np.empty((0,), dtype=np.float32)

    def _submit(start):
        batch = indices[start: start + EMBED_BATCH]
        return _executor.submit(_fetch_images, lmdb_path, batch, augment)

    out = []
    prefetch = _submit(starts[0])

    for i, start in enumerate(tqdm(starts, desc="  embedding", leave=False)):
        imgs_np = prefetch.result()
        # Start fetching next batch immediately
        if i + 1 < len(starts):
            prefetch = _submit(starts[i + 1])

        x = torch.from_numpy(imgs_np).to(device)
        e = model(x)
        e = F.normalize(e, dim=1).cpu().numpy()
        out.append(e)

    if was_training:
        model.train()
    return np.vstack(out)                               # (N, D)


def embed_all_with_inv(
        model: nn.Module, lmdb_path: str, indices: list,
        device: torch.device, optimizer: torch.optim.Optimizer,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combined augmented + clean forward pass with a per-batch invariance update.

    For every batch:
      1. Fetch augmented and clean images in parallel.
      2. Forward augmented with grad; forward clean with no_grad (fixed target).
      3. Invariance loss = MSE(aug_emb, clean_emb) — train aug to match clean.
      4. Zero-grad → backward → step.
      5. Accumulate both embedding arrays for pool / clean_targets.

    Returns (aug_embs_np, clean_embs_np), each (N, D) float32.
    """
    model.train()
    starts = list(range(0, len(indices), EMBED_BATCH))
    if not starts:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty

    def _submit_both(start):
        batch = indices[start: start + EMBED_BATCH]
        return (
            _executor.submit(_fetch_images, lmdb_path, batch, True),
            _executor.submit(_fetch_images, lmdb_path, batch, False),
        )

    aug_out    = []
    clean_out  = []
    batch_losses: list[float] = []
    card_dists: list[np.ndarray] = []   # per-card L2(aug, clean) for every batch
    prefetch = _submit_both(starts[0])

    for i, start in enumerate(tqdm(starts, desc="  inv-embed", leave=False)):
        fut_aug, fut_clean = prefetch
        if i + 1 < len(starts):
            prefetch = _submit_both(starts[i + 1])

        x_aug   = torch.from_numpy(fut_aug.result()).to(device)
        x_clean = torch.from_numpy(fut_clean.result()).to(device)

        e_aug = F.normalize(model(x_aug), dim=1)            # grad flows here
        with torch.no_grad():
            e_clean = F.normalize(model(x_clean), dim=1)   # fixed target

        diff      = e_aug - e_clean                         # (K, D)
        per_card  = diff.pow(2).sum(dim=1)                  # (K,) squared L2
        inv_loss  = per_card.mean()
        optimizer.zero_grad()
        inv_loss.backward()
        optimizer.step()

        batch_losses.append(inv_loss.item())
        card_dists.append(per_card.detach().sqrt().cpu().numpy())  # L2 distance

        aug_out.append(e_aug.detach().cpu().numpy())
        clean_out.append(e_clean.cpu().numpy())

    all_dists = np.concatenate(card_dists)   # (N,)
    all_aug   = np.vstack(aug_out)
    all_clean = np.vstack(clean_out)
    bl        = np.array(batch_losses)
    p95       = float(np.percentile(all_dists, 95))
    print(f"  inv-embed  loss avg={bl.mean():.6f}  "
          f"dist mean={all_dists.mean():.4f}  p95={p95:.4f}  max={all_dists.max():.4f}")
    return all_aug, all_clean, all_dists, batch_losses          # (N,D), (N,D), (N,), list


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
                      row_batch: int = 512) -> tuple[float, int, int, torch.Tensor]:
    """
    Minimum nearest-neighbour L2 distance across all embeddings.
    Returns (min_dist, row_index i, col_index j, min_d_tensor).
    min_d_tensor[k] = distance from k to its nearest neighbour (GPU, shape (N,)).
    Used as a lower-bound priority queue for anchor selection during training.
    """
    N = len(embs)
    t = torch.from_numpy(embs).to(device)
    min_d   = torch.full((N,), float("inf"), device=device)
    min_col = torch.zeros(N, dtype=torch.long, device=device)

    for start in range(0, N, row_batch):
        end  = min(start + row_batch, N)
        rows = t[start:end]
        d    = torch.cdist(rows, t, p=2)
        for local_i in range(end - start):
            d[local_i, start + local_i] = float("inf")
        result = d.min(dim=1)
        min_d[start:end]   = result.values
        min_col[start:end] = result.indices

    i = int(min_d.argmin().item())
    j = int(min_col[i].item())
    return float(min_d[i].item()), i, j, min_d


@torch.no_grad()
def _refresh_min_d(pool_embs: torch.Tensor, min_d: torch.Tensor,
                   row_indices: torch.Tensor, row_batch: int = 512) -> None:
    """
    Update min_d[row_indices] in-place.
    Batches cdist to cap VRAM at row_batch × N × 4 bytes (~200 MB at 512 rows, 100k pool).
    """
    device = pool_embs.device
    for start in range(0, len(row_indices), row_batch):
        end = min(start + row_batch, len(row_indices))
        idx = row_indices[start:end]
        d   = torch.cdist(pool_embs[idx], pool_embs)          # (batch, N)
        d[torch.arange(end - start, device=device), idx] = float('inf')
        min_d[idx] = d.min(dim=1).values


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    canonical: list = json.loads(Path(CANONICAL_INDEX).read_text())
    print(f"Canonical set: {len(canonical):,} images")

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model = model.to(device=device, dtype=torch.float32)

    bn_layers = [name for name, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    if bn_layers:
        print(f"✅ Found {len(bn_layers)} Batch Normalization layers!")
        print("First 3 BN layers found:", bn_layers[:3])
    else:
        print("❌ No Batch Normalization layers found. Checking for LayerNorm...")
        ln_layers = [name for name, m in model.named_modules() if isinstance(m, nn.LayerNorm)]
        print(f"Found {len(ln_layers)} LayerNorm layers instead.")

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

        # ── Step 2: combined inv-embed pass — augmented pool + clean targets + inv grad ──
        embs_np, clean_np, inv_dists, inv_losses = embed_all_with_inv(
            model, CARDS_224, epoch_indices, device, optimizer,
        )

        # ── Step 2b: set margin = min pairwise dist + slack ───────────────────
        epoch_min, close_i, close_j, min_d_gpu = min_pairwise_dist(embs_np, device)
        margin = epoch_min + MARGIN_SLACK
        print(f"  min_dist={epoch_min:.4f}  margin={margin:.4f}"
              f"  closest=({epoch_indices[close_i]}, {epoch_indices[close_j]})")
        # ── Write per-epoch embedding report (describes state after previous epoch) ──
        report_epoch = epoch - 1
        report_stem  = out_dir / f"epoch_{report_epoch:03d}"
        report_text, report_data = embedding_report(
            embs_np, epoch_indices, device,
            nn_dists=min_d_gpu.cpu().numpy(),
            label=f"Epoch {report_epoch} (state before epoch {epoch} training)  "
                  f"– {n:,} canonical images",
        )
        report_stem.with_suffix(".txt").write_text(report_text)
        report_stem.with_name(report_stem.name + "_report.json").write_text(
            json.dumps(report_data, indent=2)
        )
        print(f"  Report  → {report_stem}.txt / _report.json")
        # ── Write invariance report ─────────────────────────────────────────────────────
        inv_text, inv_data = invariance_report(
            inv_dists, inv_losses, epoch_indices,
            label=f"Epoch {report_epoch} invariance pass"
                  f" — aug→clean distances  ({n:,} canonical images)",
        )
        report_stem.with_name(report_stem.name + "_inv.txt").write_text(inv_text)
        report_stem.with_name(report_stem.name + "_inv_report.json").write_text(
            json.dumps(inv_data, indent=2)
        )
        print(f"  Inv     → {report_stem}_inv.txt / _inv_report.json")
        # ── Step 3: persistent GPU pool ────────────────────────────────────────────────
        pool_embs_gpu     = torch.from_numpy(embs_np).to(device)    # (N, D) augmented
        pool_idx_arr      = np.array(epoch_indices, dtype=np.int64)  # (N,) – fixed
        # Clean embeddings frozen for the epoch — invariance targets, never mutated.
        clean_targets_gpu = torch.from_numpy(clean_np).to(device)   # (N, D) clean

        total_loss  = 0.0
        total_viol  = 0
        total_pairs = 0
        n_batches   = 0

        model.train()

        def _next_cluster():
            """
            Pick the anchor with the lowest stale NN distance (hardest pair),
            verify its true NN distance, and return None if this round is done.
            """
            anchor_pos = int(min_d_gpu.argmin().item())
            with torch.no_grad():
                dists = torch.cdist(
                    pool_embs_gpu[anchor_pos:anchor_pos + 1],
                    pool_embs_gpu,
                ).squeeze(0)                                   # (N,)
            dists[anchor_pos] = float('inf')                   # exclude self
            min_d_gpu[anchor_pos] = dists.min()
            if float(min_d_gpu[anchor_pos].item()) >= margin:
                return None                                    # round done
            dists[anchor_pos] = 0.0                            # restore so topk includes anchor
            pos      = torch.topk(dists, CLUSTER_SIZE, largest=False).indices.cpu().tolist()
            lmdb_ids = [int(pool_idx_arr[p]) for p in pos]
            return lmdb_ids, pos

        _debug_first_round = True  # set True temporarily to diagnose first-step behaviour

        # ── Step 4: REFRESH_STEPS rounds ──────────────────────────────────────
        for refresh_i in range(REFRESH_STEPS):

            first_result = _next_cluster()
            if first_result is None:
                print(f"  Converged after {refresh_i} refresh(es) — stopping epoch.")
                break

            cluster_lmdb, cluster_pos = first_result
            prefetch = _executor.submit(_fetch_images, CARDS_224, cluster_lmdb, True)
            touched_pos: set = set()
            pbar = tqdm(desc=f"  Epoch {epoch} round {refresh_i + 1}/{REFRESH_STEPS}",
                        unit="step")

            # ── Inner loop: drain all violations in current embedding snapshot ─
            while True:
                imgs_np = prefetch.result()

                # Submit next prefetch in parallel with GPU work
                next_result = _next_cluster()
                if next_result is not None:
                    cluster_lmdb_next, cluster_pos_next = next_result
                    prefetch_next = _executor.submit(_fetch_images, CARDS_224,
                                                     cluster_lmdb_next, True)

                # ── Forward + backward ─────────────────────────────────────────
                x = torch.from_numpy(imgs_np).to(device)
                e = model(x)
                e = F.normalize(e, dim=1)                      # (K, D)

                if _debug_first_round:
                    _debug_first_round = False
                    with torch.no_grad():
                        dm_dbg = torch.cdist(e, e, p=2)
                        c_pos_list = cluster_pos
                        # close_i / close_j are pool-array positions from min_pairwise_dist
                        pos_in_cluster_i = c_pos_list.index(close_i) if close_i in c_pos_list else None
                        pos_in_cluster_j = c_pos_list.index(close_j) if close_j in c_pos_list else None
                        print(f"\n  [DEBUG] === First step diagnostics ===")
                        print(f"  [DEBUG] min_pairwise_dist reported:")
                        print(f"           pool pos {close_i} = lmdb id {epoch_indices[close_i]}")
                        print(f"           pool pos {close_j} = lmdb id {epoch_indices[close_j]}")
                        print(f"           dist = {epoch_min:.6f}")
                        print(f"  [DEBUG] _next_cluster selected anchor pool pos: {c_pos_list[0]}"
                              f" = lmdb id {epoch_indices[c_pos_list[0]]}")
                        print(f"  [DEBUG] Closest pair in cluster at indices: {pos_in_cluster_i}, {pos_in_cluster_j}")
                        if pos_in_cluster_i is not None and pos_in_cluster_j is not None:
                            d_pair = dm_dbg[pos_in_cluster_i, pos_in_cluster_j].item()
                            print(f"  [DEBUG] Recomputed dist in forward pass: {d_pair:.6f}")
                        else:
                            print(f"  [DEBUG] *** One or both closest cards NOT in this cluster ***")
                        print(f"  [DEBUG] Pool dist (direct vectors): {(pool_embs_gpu[close_i] - pool_embs_gpu[close_j]).norm().item():.6f}")
                        print(f"  [DEBUG] min_d_gpu[close_i]: {min_d_gpu[close_i].item():.6f}")
                        print(f"  [DEBUG] margin={margin:.6f},  loss={repulsion_loss(e, margin).item():.6f}")

                cluster_pos_t = torch.tensor(cluster_pos, dtype=torch.long, device=device)
                # Invariance: pull augmented embeddings toward frozen clean targets.
                inv_loss = (e - clean_targets_gpu[cluster_pos_t]).pow(2).sum(dim=1).mean()
                loss = repulsion_loss(e, margin) + LAMBDA_INV * inv_loss
                if loss.item() > 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # ── Patch pool + update min_d + stats ─────────────────────────
                with torch.no_grad():
                    K     = e.size(0)
                    e_det = e.detach()

                    pool_embs_gpu[cluster_pos_t] = e_det
                    _refresh_min_d(pool_embs_gpu, min_d_gpu, cluster_pos_t)

                    dm   = torch.cdist(e_det, e_det, p=2)
                    mask = torch.triu(torch.ones(K, K, device=device,
                                                 dtype=torch.bool), diagonal=1)
                    pd   = dm[mask]
                    viol = int((pd < margin).sum().item())
                    np_  = int(pd.numel())

                touched_pos.update(cluster_pos)
                total_loss  += loss.item()
                total_viol  += viol
                total_pairs += np_
                n_batches   += 1

                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 viol=f"{viol}/{np_}",
                                 min_d=f"{float(min_d_gpu.min().item()):.4f}")

                if next_result is None:
                    break
                cluster_lmdb, cluster_pos = cluster_lmdb_next, cluster_pos_next
                prefetch = prefetch_next

            pbar.close()

            # ── Partial re-embed: only cards trained this round ────────────────
            touched_list     = sorted(touched_pos)
            touched_lmdb_ids = [int(pool_idx_arr[p]) for p in touched_list]
            print(f"  [round {refresh_i + 1}] re-embedding"
                  f" {len(touched_list):,} touched cards…")
            fresh_np = embed_all(model, CARDS_224, touched_lmdb_ids, device, augment=True)

            touched_t = torch.tensor(touched_list, dtype=torch.long, device=device)
            pool_embs_gpu[touched_t] = torch.from_numpy(fresh_np).to(device)

            # Update min_d for re-embedded rows (batched to cap VRAM)
            _refresh_min_d(pool_embs_gpu, min_d_gpu, touched_t)

            # Report current min but keep margin fixed for the epoch.
            # Resetting margin = new_min + slack here would cause an infinite
            # loop: partial re-embed snaps the hard pair back to its original
            # distance, margin resets to the same violated value, and the next
            # round repeats identically forever.
            new_min = float(min_d_gpu.min().item())
            print(f"  [round {refresh_i + 1}] min_dist={new_min:.4f}"
                  f"  margin={margin:.4f} (epoch-fixed)")

        avg_loss  = total_loss  / max(n_batches, 1)
        viol_rate = total_viol  / max(total_pairs, 1)
        print(f"  avg_loss={avg_loss:.4f}  violated={viol_rate:.1%}"
              f"  batches={n_batches}")

        ckpt = out_dir / f"epoch_{epoch:03d}.pt"
        torch.save(model.state_dict(), ckpt)
        torch.save(model.state_dict(), out_dir / "last.pt")
        print(f"  Saved → {ckpt}")


if __name__ == "__main__":
    train()
