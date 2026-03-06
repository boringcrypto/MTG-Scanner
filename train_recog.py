"""
train_recog.py  –  SimCLR training with NT-Xent loss.

Each epoch:
  1. Stream all cards in batches of BATCH_SIZE, each yielding an aug + clean view.
  2. Forward both views → 2K L2-normalised embeddings.
  3. NT-Xent loss: (aug_i, clean_i) is the positive pair; all other 2(K-1) are negatives.
  4. Backprop, step, repeat.  Save checkpoint.
"""

import torch
import torch.nn.functional as F

from card_loader import AugmentedCard224Loader
from card_model import CardEmbedder


# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
TEMPERATURE = 0.07
LR          = 3e-5
EPOCHS      = 500


# ── Loss ──────────────────────────────────────────────────────────────────────

def nt_xent_loss(aug_embs: torch.Tensor, clean_embs: torch.Tensor,
                 temperature: float) -> torch.Tensor:
    """
    NT-Xent (SimCLR) contrastive loss.

    aug_embs, clean_embs : (K, D)  L2-normalised
    Positive pair for aug_i  is clean_i  (index i+K in the concatenated tensor).
    Positive pair for clean_i is aug_i   (index i   in the concatenated tensor).
    All other 2(K-1) pairs within the batch are treated as negatives.
    """
    K = aug_embs.size(0)
    z = torch.cat([aug_embs, clean_embs], dim=0)    # (2K, D)
    sim = torch.mm(z, z.T) / temperature             # (2K, 2K)  cosine × 1/τ
    sim.fill_diagonal_(float('-inf'))                # exclude self-similarity
    labels = torch.cat([
        torch.arange(K, 2 * K, device=z.device),    # aug_i   → clean_i
        torch.arange(K,         device=z.device),    # clean_i → aug_i
    ])
    return F.cross_entropy(sim, labels)


# ── Trainer ───────────────────────────────────────────────────────────────────

class CardEmbeddingTrainer:

    def __init__(self):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loader    = AugmentedCard224Loader()
        self.model     = CardEmbedder().to(device=self.device, dtype=torch.float32)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)

    def _step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        for epoch in range(1, EPOCHS + 1):
            self.model.train()
            for _, x_aug, x_clean in self.loader.stream_with_original(BATCH_SIZE,
                                                                       device=self.device):
                e_aug   = self.model(x_aug)
                e_clean = self.model(x_clean)
                self._step(nt_xent_loss(e_aug, e_clean, TEMPERATURE))
            self.model.save(epoch)


if __name__ == "__main__":
    CardEmbeddingTrainer().train()
