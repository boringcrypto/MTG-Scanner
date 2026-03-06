"""
train_recog.py  –  SimCLR with greedy nearest-neighbour batching.

Each epoch:
  1. Embed all cards (clean, no grad) → (N, D) pool.
  2. Greedy NN batching: pick random seed, claim 64 nearest, repeat until all assigned.
  3. Sort batches by mean intra-cluster distance; keep tightest BATCH_KEEP fraction.
  4. For each kept batch: fetch aug + clean images, NT-Xent loss, step.
  5. Save checkpoint.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from card_loader import Card224Loader, AugmentedCard224Loader
from card_model import CardEmbedder


# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
BATCH_KEEP  = 0.70    # train on the tightest 70% of clusters each epoch
EMBED_BATCH = 256     # larger batch size for no-grad embedding pass
TEMPERATURE = 0.07
LR          = 3e-5
EPOCHS      = 500


# ── Loss ──────────────────────────────────────────────────────────────────────

def nt_xent_loss(aug_embs: torch.Tensor, clean_embs: torch.Tensor,
                 temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    NT-Xent (SimCLR) contrastive loss, decomposed into:
      inv_loss : -sim(z_i, z_pos) / τ             (alignment — invariance to augmentation)
      sep_loss : log Σ_{j≠i} exp(sim(z_i,z_j)/τ)  (separation — spread of negatives)
      total    : inv_loss + sep_loss               (= cross_entropy)

    aug_embs, clean_embs : (K, D)  L2-normalised
    """
    K = aug_embs.size(0)
    z = torch.cat([aug_embs, clean_embs], dim=0)    # (2K, D)
    sim = torch.mm(z, z.T) / temperature             # (2K, 2K)  cosine × 1/τ
    sim.fill_diagonal_(float('-inf'))                # exclude self-similarity
    labels = torch.cat([
        torch.arange(K, 2 * K, device=z.device),    # aug_i   → clean_i
        torch.arange(K,         device=z.device),    # clean_i → aug_i
    ])
    pos_sim  = sim[torch.arange(2 * K, device=z.device), labels]  # (2K,)
    inv_loss = -pos_sim.mean()
    sep_loss = torch.logsumexp(sim, dim=1).mean()
    return inv_loss + sep_loss, inv_loss, sep_loss


# ── Greedy batching ───────────────────────────────────────────────────────────

def build_batches(embs: np.ndarray, batch_size: int) -> list[tuple[list[int], float]]:
    """
    Greedy nearest-neighbour batching over L2-normalised embeddings.

    Iterates in a random seed order so batch composition varies each epoch.
    Returns a list of (position_indices, mean_dist_from_seed) sorted by
    mean_dist ascending — tightest clusters first.
    """
    t          = torch.from_numpy(embs)              # CPU, (N, D)
    unassigned = torch.ones(len(embs), dtype=torch.bool)
    batches: list[tuple[list[int], float]] = []

    for seed in torch.randperm(len(embs)).tolist():
        if not unassigned[seed]:
            continue
        dists = torch.cdist(t[seed:seed + 1], t).squeeze(0)   # (N,)
        dists[~unassigned] = float('inf')
        k       = int(unassigned.sum().item())
        nearest = torch.topk(dists, min(batch_size, k), largest=False).indices
        mean_d  = dists[nearest].mean().item()
        unassigned[nearest] = False
        batches.append((nearest.tolist(), mean_d))

    batches.sort(key=lambda b: b[1])
    return batches


# ── Trainer ───────────────────────────────────────────────────────────────────

class CardEmbeddingTrainer:

    def __init__(self):
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aug_loader   = AugmentedCard224Loader()
        self.clean_loader = Card224Loader()
        self.model        = CardEmbedder().to(device=self.device, dtype=torch.float32)
        self.optimizer    = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.indices      = self.aug_loader.all_indices    # list of lmdb ids

    def _step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def _embed_all(self) -> np.ndarray:
        """Embed all cards (clean, no grad). Returns (N, D) float32 array."""
        self.model.eval()
        out = []
        for _, imgs_np in tqdm(self.clean_loader.stream(EMBED_BATCH, self.indices),
                               desc="  embedding", leave=False):
            out.append(self.model(torch.from_numpy(imgs_np).to(self.device)).cpu().numpy())
        self.model.train()
        return np.vstack(out)

    def train(self):
        for epoch in range(1, EPOCHS + 1):
            embs    = self._embed_all()
            batches = build_batches(embs, BATCH_SIZE)
            keep_n  = max(1, int(len(batches) * BATCH_KEEP))
            batches = batches[:keep_n]

            self.model.train()
            total_loss, inv_sum, sep_sum, n = 0.0, 0.0, 0.0, 0
            pbar = tqdm(batches, desc=f"epoch {epoch:4d}", leave=False)
            for pos_list, _ in pbar:
                lmdb_ids = [self.indices[p] for p in pos_list]
                x_aug    = torch.from_numpy(self.aug_loader.fetch(lmdb_ids)).to(self.device)
                x_clean  = torch.from_numpy(self.clean_loader.fetch(lmdb_ids)).to(self.device)
                loss, inv, sep = nt_xent_loss(self.model(x_aug), self.model(x_clean), TEMPERATURE)
                self._step(loss)
                total_loss += loss.item(); inv_sum += inv.item(); sep_sum += sep.item(); n += 1
                pbar.set_postfix(loss=f"{total_loss/n:.3f}",
                                 inv=f"{inv_sum/n:.3f}",
                                 sep=f"{sep_sum/n:.3f}")
            print(f"epoch {epoch:4d}  loss {total_loss/n:.4f}"
                  f"  inv {inv_sum/n:.4f}  sep {sep_sum/n:.4f}")
            self.model.save(epoch)


if __name__ == "__main__":
    CardEmbeddingTrainer().train()
