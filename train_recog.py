"""
train_recog.py  –  SimCLR with greedy nearest-neighbour batching.

Each epoch:
  1. Embed all cards (clean, no grad) → (N, D) pool.
  2. Greedy NN batching: pick random seed, claim 64 nearest, repeat until all assigned.
  3. Sort batches by mean intra-cluster distance; keep tightest BATCH_KEEP fraction.
  4. For each kept batch: fetch aug + clean images, NT-Xent loss, step.
  5. Save checkpoint.
"""

import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from card_loader import Card224Loader, AugmentedCard224Loader, CompositeAugCard224Loader
from card_model import CardEmbedder, CHECKPOINT_DIR
from val_recog import EmbeddingValidator


# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
BATCH_KEEP  = 0.25    # train on the tightest 25% of clusters each epoch
EMBED_BATCH = 256     # larger batch size for no-grad embedding pass
TEMPERATURE   = 0.07
MARGIN        = 0.05    # only repel negatives within this cosine distance of the positive
LR            = 3e-5
LR_MIN        = LR * 0.01  # cosine decay floor
WARMUP_EPOCHS = 10         # linearly ramp LR from LR/10 → LR over first N epochs
EPOCHS        = 100
# Swap to AugmentedCard224Loader for fast synthetic augmentation
AUG_LOADER  = CompositeAugCard224Loader


# ── Loss ──────────────────────────────────────────────────────────────────────

def nt_xent_loss(aug_embs: torch.Tensor, clean_embs: torch.Tensor,
                 temperature: float, margin: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decoupled contrastive loss with semi-hard negative masking.

      inv_loss : -mean(sim(z_i, z_pos)) / τ           (pull positives together)
      sep_loss : mean(log Σ_{j: semi-hard} exp(sim/τ)) (push semi-hard negatives apart)

    Semi-hard: negatives within `margin` cosine units of the positive.
    Negatives already more than `margin` below the positive are masked out (-inf)
    so they contribute no gradient, preventing over-repulsion of dissimilar cards.
    """
    K    = aug_embs.size(0)
    N    = 2 * K
    z    = torch.cat([aug_embs, clean_embs], dim=0)    # (2K, D)
    sim  = torch.mm(z, z.T) / temperature              # (2K, 2K)

    rows   = torch.arange(N, device=z.device)
    labels = torch.cat([
        torch.arange(K, N, device=z.device),           # aug_i   → clean_i
        torch.arange(K,    device=z.device),            # clean_i → aug_i
    ])

    pos_sim  = sim[rows, labels]                       # (2K,) temperature-scaled
    inv_loss = -pos_sim.mean()

    # Exclude self and positive pair
    sim_neg               = sim.clone()
    sim_neg[rows, rows]   = float('-inf')
    sim_neg[rows, labels] = float('-inf')

    # Mask out negatives already more than `margin` below the positive
    # threshold is in temperature-scaled space: (pos_sim - margin/τ)
    threshold = (pos_sim - margin / temperature).unsqueeze(1)  # (2K, 1)
    sim_neg[sim_neg < threshold] = float('-inf')

    sep_per_row = torch.logsumexp(sim_neg, dim=1)    # (2K,)  may be -inf where no semi-hard neg
    active      = sep_per_row.isfinite()
    sep_loss    = sep_per_row[active].mean() if active.any() else sim.new_tensor(0.0)

    return inv_loss + sep_loss, inv_loss, sep_loss


# ── Embedding helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def embed_all(model: CardEmbedder, indices: list, device: torch.device,
              batch_size: int = 256) -> np.ndarray:
    """Embed all cards (clean images, no grad). Returns (N, D) float32 array."""
    from card_loader import Card224Loader
    loader = Card224Loader()
    model.eval()
    out   = []
    total = -(-len(indices) // batch_size)
    for _, imgs_np in tqdm(loader.stream(batch_size, indices),
                           desc="  embedding", leave=False, total=total):
        out.append(model(torch.from_numpy(imgs_np).to(device)).cpu().numpy())
    return np.vstack(out)


# ── Greedy batching ───────────────────────────────────────────────────────────

def build_batches(embs: np.ndarray, batch_size: int,
                  device: torch.device) -> list[tuple[list[int], float]]:
    """
    Greedy nearest-neighbour batching over L2-normalised embeddings.

    Uses dot-product similarity (= cosine sim for unit vectors) on GPU.
    Iterates in a random seed order so batch composition varies each epoch.
    Returns a list of (position_indices, mean_sim_to_seed) sorted by
    mean_sim descending — tightest clusters first.
    """
    t          = torch.from_numpy(embs).to(device)          # (N, D)  GPU
    unassigned = torch.ones(len(embs), dtype=torch.bool, device=device)
    batches: list[tuple[list[int], float]] = []

    for seed in torch.randperm(len(embs), device=device).tolist():
        if not unassigned[seed]:
            continue
        sim = t @ t[seed]                                   # (N,)  dot product
        sim[~unassigned] = -float('inf')
        k       = int(unassigned.sum().item())
        nearest = torch.topk(sim, min(batch_size, k), largest=True).indices
        mean_s  = sim[nearest].mean().item()
        unassigned[nearest] = False
        batches.append((nearest.tolist(), mean_s))

    batches.sort(key=lambda b: b[1], reverse=True)         # highest sim first
    return batches


# ── Trainer ───────────────────────────────────────────────────────────────────

class CardEmbeddingTrainer:

    def __init__(self, resume: bool = False):
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aug_loader   = AUG_LOADER()
        self.clean_loader = Card224Loader()
        self.model        = CardEmbedder.load("<latest>" if resume else None,
                                              device=self.device).train()
        self.optimizer    = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.start_epoch  = self._last_epoch() + 1 if resume else 1
        self.scheduler    = self._make_scheduler()
        self.indices      = self.aug_loader.all_indices    # list of lmdb ids
        self.validator    = EmbeddingValidator(
            self.aug_loader, self.clean_loader, self.model, self.indices, self.device
        )

    def _make_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Warm-up then cosine decay. Fast-forwards to start_epoch when resuming."""
        def lr_lambda(step: int) -> float:
            if step < WARMUP_EPOCHS:
                return (step + 1) / WARMUP_EPOCHS          # 0.1 → 1.0
            progress = (step - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
            return LR_MIN / LR + 0.5 * (1 - LR_MIN / LR) * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda, last_epoch=self.start_epoch - 2
        )

    @staticmethod
    def _last_epoch() -> int:
        """Return the epoch number of the highest epoch_NNN.pt in CHECKPOINT_DIR, or 0."""
        pts = sorted(CHECKPOINT_DIR.glob("epoch_*.pt"))
        if not pts:
            return 0
        try:
            return int(pts[-1].stem.split("_")[1])
        except (IndexError, ValueError):
            return 0

    def _step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def _embed_all(self, model_epoch: int) -> np.ndarray:
        """Embed all cards (clean, no grad); cache result to disk."""
        out_path = CHECKPOINT_DIR / f"embeddings_epoch_{model_epoch:03d}.npy"
        if out_path.exists():
            return np.load(out_path)
        embs = embed_all(self.model, self.indices, self.device, EMBED_BATCH)
        self.model.train()
        np.save(out_path, embs)
        return embs

    def train(self):
        if self.start_epoch > 1:
            print(f"Resuming from epoch {self.start_epoch}")
        for epoch in range(self.start_epoch, self.start_epoch + EPOCHS):
            embs    = self._embed_all(model_epoch=epoch - 1)
            m       = self.validator.validate(embs, epoch)
            batches = build_batches(embs, BATCH_SIZE, self.device)
            keep_n  = max(1, int(len(batches) * BATCH_KEEP))
            batches = batches[:keep_n]

            self.model.train()
            total_loss, inv_sum, sep_sum, n = 0.0, 0.0, 0.0, 0
            pbar = tqdm(batches, desc=f"epoch {epoch:4d}", leave=False)
            for pos_list, _ in pbar:
                lmdb_ids = [self.indices[p] for p in pos_list]
                x_aug    = torch.from_numpy(self.aug_loader.fetch(lmdb_ids)).to(self.device)
                x_clean  = torch.from_numpy(self.clean_loader.fetch(lmdb_ids)).to(self.device)
                loss, inv, sep = nt_xent_loss(self.model(x_aug), self.model(x_clean), TEMPERATURE, MARGIN)
                self._step(loss)
                total_loss += loss.item(); inv_sum += inv.item(); sep_sum += sep.item(); n += 1
                pbar.set_postfix(loss=f"{total_loss/n:.3f}",
                                 inv=f"{inv_sum/n:.3f}",
                                 sep=f"{sep_sum/n:.3f}")
            cur_lr = self.optimizer.param_groups[0]['lr']
            print(f"epoch {epoch:4d}  loss {total_loss/n:.4f}"
                  f"  inv {inv_sum/n:.4f}  sep {sep_sum/n:.4f}"
                  f"  lr {cur_lr:.2e}")
            self.model.save(epoch)
            self.scheduler.step()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="Resume training from runs/recog/last.pt")
    args = ap.parse_args()
    CardEmbeddingTrainer(resume=args.resume).train()
