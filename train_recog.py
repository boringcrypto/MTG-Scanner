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
from tqdm import tqdm

from card_loader import AugmentedCard224Loader
from card_model import CardEmbedder


# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
TEMPERATURE = 0.07
LR          = 3e-5
EPOCHS      = 500


# ── Loss ──────────────────────────────────────────────────────────────────────

def nt_xent_loss(aug_embs: torch.Tensor, clean_embs: torch.Tensor,
                 temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    NT-Xent (SimCLR) contrastive loss, decomposed into:
      inv_loss : -sim(z_i, z_pos) / τ            (alignment — invariance to augmentation)
      sep_loss : log Σ_{j≠i} exp(sim(z_i,z_j)/τ) (separation — spread of negatives)
      total    : inv_loss + sep_loss              (= cross_entropy)

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
            total_loss, inv_loss_sum, sep_loss_sum, n_batches = 0.0, 0.0, 0.0, 0
            pbar = tqdm(self.loader.stream_with_original(BATCH_SIZE, device=self.device),
                        desc=f"epoch {epoch:4d}", leave=False)
            for _, x_aug, x_clean in pbar:
                e_aug   = self.model(x_aug)
                e_clean = self.model(x_clean)
                loss, inv, sep = nt_xent_loss(e_aug, e_clean, TEMPERATURE)
                self._step(loss)
                total_loss    += loss.item()
                inv_loss_sum  += inv.item()
                sep_loss_sum  += sep.item()
                n_batches     += 1
                pbar.set_postfix(loss=f"{total_loss / n_batches:.3f}",
                                 inv=f"{inv_loss_sum / n_batches:.3f}",
                                 sep=f"{sep_loss_sum / n_batches:.3f}")
            print(f"epoch {epoch:4d}  loss {total_loss/n_batches:.4f}"
                  f"  inv {inv_loss_sum/n_batches:.4f}  sep {sep_loss_sum/n_batches:.4f}")
            self.model.save(epoch)


if __name__ == "__main__":
    CardEmbeddingTrainer().train()
