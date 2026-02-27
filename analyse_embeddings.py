"""
analyse_embeddings.py

Analyses the embedding space of a ViT-S model over cards_224.lmdb.
Reports nearest-neighbour distance statistics, clustering metrics, and
produces histograms so you can judge how well-spread the embeddings are.

Usage
-----
    # Fresh ImageNet-pretrained weights
    python analyse_embeddings.py

    # Fine-tuned weights
    python analyse_embeddings.py --weights runs/recog/last.pt

    # Limit to a random subset
    python analyse_embeddings.py --sample 5000

    # Save plots instead of showing them
    python analyse_embeddings.py --save-plots
"""
from __future__ import annotations
print("Starting analyser...")

import argparse
import json
import lmdb
import cv2
import numpy as np
print("Importing pyorch")
import torch
print("Importing timm")
import timm
from pathlib import Path
print("Importing tqdm")
from tqdm import tqdm

print("Importing card_hasher")
from card_hasher import MODEL_NAME, IMAGE_SIZE, _MEAN, _STD
print("Imports done.")

# ── Default paths ─────────────────────────────────────────────────────────────
CARDS_224        = "data/cards/cards_224.lmdb"
CANONICAL_INDEX  = "data/cards/canonical_index.json"   # produced by build_canonical_index.py
EMBED_BATCH      = 256   # images per forward pass


# ── Embedding ─────────────────────────────────────────────────────────────────

def load_model(weights: str | None, device: torch.device) -> torch.nn.Module:
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    if weights:
        p = Path(weights)
        if p.exists():
            state = torch.load(str(p), map_location=device, weights_only=True)
            model.load_state_dict(state)
            print(f"Loaded fine-tuned weights: {p}")
        else:
            print(f"WARNING: weights file not found ({p}), using ImageNet pretrained.")
    else:
        print("Using ImageNet pretrained weights (no fine-tuning).")
    model.eval()
    model.to(device=device, dtype=torch.float32)
    return model


def embed_cards(model: torch.nn.Module, cards_224: str,
                sample_indices: list[int], device: torch.device) -> np.ndarray:
    """Return L2-normalised embeddings, shape (N, D)."""
    env = lmdb.open(cards_224, readonly=True, lock=False,
                    readahead=False, meminit=False)
    all_embs: list[np.ndarray] = []
    buf: list[np.ndarray] = []

    def _flush():
        imgs = []
        for bgr in buf:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            imgs.append(((rgb - _MEAN) / _STD).transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(imgs)).to(device)
        with torch.no_grad():
            emb = model(x).float().cpu().numpy()
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        all_embs.append(emb)
        buf.clear()

    for idx in tqdm(sample_indices, desc="Embedding", unit="img"):
        with env.begin() as txn:
            data = txn.get(str(idx).encode())
        buf.append(cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR))
        if len(buf) == EMBED_BATCH:
            _flush()
    if buf:
        _flush()
    env.close()
    return np.vstack(all_embs).astype(np.float32)


# ── Distance computation (batched to avoid OOM) ───────────────────────────────

def nearest_neighbour_distances(embs: np.ndarray,
                                device: torch.device,
                                row_batch: int = 512) -> np.ndarray:
    """
    For each embedding, find its nearest-neighbour L2 distance (self excluded).
    Processed in row batches to keep VRAM usage bounded.
    Returns array of shape (N,).
    """
    N = len(embs)
    t = torch.from_numpy(embs).to(device)
    nn_dists = torch.full((N,), float("inf"), device=device)

    for start in tqdm(range(0, N, row_batch), desc="NN distances", unit="batch"):
        end  = min(start + row_batch, N)
        rows = t[start:end]                           # (B, D)
        d    = torch.cdist(rows, t, p=2)              # (B, N)
        # Mask self
        for local_i in range(end - start):
            d[local_i, start + local_i] = float("inf")
        nn_dists[start:end] = d.min(dim=1).values

    return nn_dists.cpu().numpy()


def knn_distances(embs: np.ndarray, k: int,
                  device: torch.device, row_batch: int = 512) -> np.ndarray:
    """
    For each embedding, find the mean distance to its k nearest neighbours.
    Returns shape (N,).
    """
    N  = len(embs)
    t  = torch.from_numpy(embs).to(device)
    out = np.zeros(N, dtype=np.float32)
    k_eff = min(k + 1, N)   # +1 because self will be included, then removed

    for start in tqdm(range(0, N, row_batch), desc=f"k={k} distances", unit="batch", leave=False):
        end  = min(start + row_batch, N)
        rows = t[start:end]
        d    = torch.cdist(rows, t, p=2)
        for local_i in range(end - start):
            d[local_i, start + local_i] = float("inf")
        topk = d.topk(k, largest=False, dim=1).values     # (B, k)
        out[start:end] = topk.mean(dim=1).cpu().numpy()

    return out


# ── Text histogram ─────────────────────────────────────────────────────────────

def text_histogram(values: np.ndarray, n_bins: int = 20,
                   width: int = 50, title: str = "") -> None:
    lo, hi  = values.min(), values.max()
    bins    = np.linspace(lo, hi, n_bins + 1)
    counts, edges = np.histogram(values, bins=bins)
    max_c = counts.max()
    if title:
        print(f"\n  {title}")
    print(f"  {'dist':>8}  {'count':>7}  {'bar'}")
    print(f"  {'':-<8}  {'':-<7}  {'':-<{width}}")
    for i, c in enumerate(counts):
        bar = "█" * int(c / max_c * width)
        mid = (edges[i] + edges[i + 1]) / 2
        print(f"  {mid:8.4f}  {c:7d}  {bar}")


# ── Stats helper ───────────────────────────────────────────────────────────────

def print_stats(label: str, values: np.ndarray) -> None:
    ps = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  n       = {len(values):,}")
    print(f"  min     = {values.min():.6f}")
    print(f"  p1      = {ps[0]:.6f}")
    print(f"  p5      = {ps[1]:.6f}")
    print(f"  p25     = {ps[2]:.6f}")
    print(f"  median  = {ps[3]:.6f}")
    print(f"  mean    = {values.mean():.6f}  (±{values.std():.6f})")
    print(f"  p75     = {ps[4]:.6f}")
    print(f"  p95     = {ps[5]:.6f}")
    print(f"  p99     = {ps[6]:.6f}")
    print(f"  max     = {values.max():.6f}")


def print_threshold_table(nn_dists: np.ndarray) -> None:
    N = len(nn_dists)
    print(f"\n{'─'*60}")
    print("  Fraction of cards with NN distance BELOW threshold")
    print(f"{'─'*60}")
    print(f"  {'threshold':>10}  {'count':>8}  {'fraction':>10}")
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                   0.60, 0.70, 0.80, 1.00, 1.20, 1.40]:
        n_below = int((nn_dists < thresh).sum())
        print(f"  {thresh:10.2f}  {n_below:8,}  {n_below/N:10.2%}")


def pca_variance(embs: np.ndarray, n_components: int = 50) -> None:
    """Print cumulative variance explained by top PCA components."""
    # Zero-mean
    mu   = embs.mean(axis=0)
    X    = embs - mu
    # Covariance via SVD on (N, D) — cheaper than (D, D) covariance when N < D
    k    = min(n_components, X.shape[0], X.shape[1])
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s    = s[:k]
    var  = s ** 2
    total = var.sum()   # approximate (only top-k)
    total_full_approx = (X ** 2).sum()   # Frobenius norm = total variance
    cum  = np.cumsum(var) / total_full_approx

    print(f"\n{'─'*60}")
    print("  PCA: cumulative variance explained by top-k components")
    print(f"{'─'*60}")
    check_at = [1, 2, 5, 10, 20, 50]
    for c in check_at:
        if c <= k:
            print(f"  top-{c:3d} components: {cum[c-1]:.2%}")
    print(f"  (embed dim = {embs.shape[1]})")


# ── Image helpers ───────────────────────────────────────────────────────────────

def fetch_image(db_path: str, card_idx: int) -> np.ndarray:
    """Return the 224×224 BGR image for card_idx from LMDB."""
    env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        data = txn.get(str(card_idx).encode())
    env.close()
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def show_closest_pairs(db_path: str,
                       pairs: list[tuple[int, int, float]],
                       save: bool = False,
                       save_path: str = "closest_pairs.png") -> None:
    """
    Display (or save) a grid of the top-N closest pairs.
    Each row = one pair: [card_a | card_b], labelled with card indices and distance.
    """
    import matplotlib.pyplot as plt

    n    = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(5, n * 2.6))
    fig.suptitle("Top closest pairs  (left ↔ right)", fontsize=12, y=1.01)

    if n == 1:
        axes = [axes]   # make iterable

    for row, (card_a, card_b, dist) in enumerate(pairs):
        for col, card_idx in enumerate([card_a, card_b]):
            bgr   = fetch_image(db_path, card_idx)
            rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ax    = axes[row][col]
            ax.imshow(rgb)
            ax.axis("off")
            label = f"card {card_idx}"
            if col == 0:
                label += f"\nd={dist:.4f}"
            ax.set_title(label, fontsize=7, pad=2)

    plt.tight_layout()
    if save:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Pair images saved to {save_path}")
    else:
        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Analyse ViT-S embedding space over cards_224.lmdb")
    ap.add_argument("--weights",          type=str, default=None,
                    help="Path to fine-tuned weights (.pt). Omit for ImageNet pretrained.")
    ap.add_argument("--sample",           type=int, default=None,
                    help="Number of cards to sample randomly from the canonical set.")
    ap.add_argument("--db",               type=str, default=CARDS_224,
                    help=f"Path to cards_224.lmdb (default: {CARDS_224}).")
    ap.add_argument("--canonical-index",  type=str, default=CANONICAL_INDEX,
                    help=f"Path to canonical_index.json (default: {CANONICAL_INDEX}).")
    ap.add_argument("--all-cards",        action="store_true",
                    help="Ignore canonical index and use every card in the DB.")
    ap.add_argument("--save-plots",       action="store_true",
                    help="Save matplotlib plots to disk instead of showing them.")
    ap.add_argument("--no-plots",         action="store_true",
                    help="Skip matplotlib entirely (text output only).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Dataset size ─────────────────────────────────────────────────────────
    env = lmdb.open(args.db, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        N_CARDS = int(txn.get(b"__len__").decode())
    env.close()
    print(f"Cards in db : {N_CARDS:,}")

    # ── Canonical index ───────────────────────────────────────────────────────
    if args.all_cards:
        base_indices = list(range(N_CARDS))
        print(f"Mode        : all cards  ({N_CARDS:,})")
    else:
        cidx_path = Path(args.canonical_index)
        if not cidx_path.exists():
            print(f"WARNING: canonical_index.json not found at {cidx_path}.")
            print("         Run build_canonical_index.py first, or pass --all-cards.")
            return
        with open(cidx_path) as f:
            base_indices = json.load(f)
        print(f"Mode        : canonical  ({len(base_indices):,} unique images, "
              f"{N_CARDS - len(base_indices):,} excluded/deduplicated)")

    if args.sample is not None and args.sample < len(base_indices):
        sample_indices = np.random.choice(base_indices, size=args.sample, replace=False).tolist()
        print(f"Sampling    : {args.sample:,} cards")
    else:
        sample_indices = base_indices
        print(f"Sampling    : all {len(sample_indices):,} canonical cards")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(args.weights, device)

    # ── Embed ─────────────────────────────────────────────────────────────────
    embs = embed_cards(model, args.db, sample_indices, device)
    print(f"\nEmbedding shape : {embs.shape}  (L2-normalised unit vectors)")
    print(f"Embed dim       : {embs.shape[1]}")

    # ── Global norms (sanity check — should all be ~1.0) ──────────────────────
    norms = np.linalg.norm(embs, axis=1)
    print(f"Norm range      : {norms.min():.6f} – {norms.max():.6f}  (all should be 1.0)")

    # ── Nearest-neighbour distances ───────────────────────────────────────────
    nn_dists = nearest_neighbour_distances(embs, device)

    print_stats("Nearest-Neighbour L2 distance  (lower = more clustered)", nn_dists)
    text_histogram(nn_dists, title="NN distance distribution")
    print_threshold_table(nn_dists)

    # Cosine similarity from L2: sim = 1 - d^2/2  (unit vectors)
    nn_cos = 1.0 - (nn_dists ** 2) / 2.0
    print_stats("Nearest-Neighbour cosine similarity  (higher = more clustered)", nn_cos)

    # ── k-NN mean distances for k = 1, 5, 10 ────────────────────────────────
    for k in [5, 10]:
        if len(embs) > k:
            mk = knn_distances(embs, k, device)
            print_stats(f"Mean distance to {k}-nearest neighbours", mk)

    # ── PCA –– dimensionality / collapse check ───────────────────────────────
    n_pca = min(50, len(embs) - 1)
    pca_variance(embs, n_components=n_pca)

    # ── Top-10 closest pairs ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Top-10 closest pairs  (smallest NN distances)")
    print(f"{'─'*60}")
    nn_sorted    = np.argsort(nn_dists)[:10]
    t            = torch.from_numpy(embs).to(device)
    closest_pairs: list[tuple[int, int, float]] = []
    for rank, idx in enumerate(nn_sorted, 1):
        row   = t[idx].unsqueeze(0)
        drow  = torch.cdist(row, t, p=2).squeeze(0)
        drow[idx] = float("inf")
        nn_idx = int(drow.argmin().item())
        card_a = sample_indices[idx]
        card_b = sample_indices[nn_idx]
        dist   = float(nn_dists[idx])
        closest_pairs.append((card_a, card_b, dist))
        print(f"  #{rank:2d}  card {card_a:5d} ↔ card {card_b:5d}  d={dist:.6f}")

    # ── Matplotlib plots ──────────────────────────────────────────────────────
    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            weights_label = Path(args.weights).stem if args.weights else "ImageNet pretrained"
            fig.suptitle(f"Embedding analysis — {weights_label}  (n={len(embs):,})", fontsize=13)

            # 1. NN distance histogram ────────────────────────────────────────
            ax = axes[0]
            ax.hist(nn_dists, bins=80, color="steelblue", edgecolor="none", alpha=0.85)
            ax.axvline(np.median(nn_dists), color="red", linestyle="--",
                       label=f"median={np.median(nn_dists):.3f}")
            ax.axvline(np.mean(nn_dists), color="orange", linestyle="--",
                       label=f"mean={np.mean(nn_dists):.3f}")
            ax.set_xlabel("NN L2 distance")
            ax.set_ylabel("count")
            ax.set_title("Nearest-neighbour distance")
            ax.legend(fontsize=8)

            # 2. CDF of NN distances ──────────────────────────────────────────
            ax = axes[1]
            sorted_d = np.sort(nn_dists)
            cdf      = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
            ax.plot(sorted_d, cdf, color="steelblue")
            for thresh in [0.2, 0.5, 1.0]:
                frac = float((nn_dists < thresh).mean())
                ax.axvline(thresh, color="gray", linestyle=":", linewidth=0.8)
                ax.text(thresh, 0.05, f"{frac:.0%}<{thresh}", fontsize=7,
                        ha="center", rotation=90, va="bottom", color="gray")
            ax.set_xlabel("NN L2 distance")
            ax.set_ylabel("CDF")
            ax.set_title("CDF of NN distances")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            # 3. PCA scree plot ───────────────────────────────────────────────
            ax = axes[2]
            mu = embs.mean(axis=0)
            X  = embs - mu
            k  = min(50, X.shape[0] - 1, X.shape[1])
            _, s, _ = np.linalg.svd(X, full_matrices=False)
            s = s[:k]
            var_explained = (s ** 2) / (X ** 2).sum()
            cum_var = np.cumsum(var_explained)
            ax.bar(range(1, k + 1), var_explained * 100, color="steelblue",
                   alpha=0.7, label="individual")
            ax2 = ax.twinx()
            ax2.plot(range(1, k + 1), cum_var * 100, color="red",
                     linewidth=1.5, label="cumulative")
            ax2.set_ylabel("Cumulative variance (%)", color="red")
            ax2.tick_params(axis="y", colors="red")
            ax.set_xlabel("PCA component")
            ax.set_ylabel("Variance explained (%)")
            ax.set_title("PCA scree (top 50 components)")

            plt.tight_layout()

            if args.save_plots:
                out = Path("embedding_analysis.png")
                plt.savefig(str(out), dpi=150)
                print(f"\nPlot saved to {out}")
            else:
                plt.show()

            # ── Pair images ──────────────────────────────────────────────
            show_closest_pairs(
                args.db,
                closest_pairs,
                save=args.save_plots,
                save_path="closest_pairs.png",
            )

        except ImportError:
            print("\n(matplotlib not available — skipping plots)")

    print("\nDone.")


if __name__ == "__main__":
    main()
