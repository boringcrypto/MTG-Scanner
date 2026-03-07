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
from pathlib import Path
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
# phash neighbor supervision
PHASH_NEIGHBORS_PATH = Path("data/cards/phash_neighbors.npy")
PHASH_SIM_LOW   = 0.50   # pull phash-NN pairs that are further apart than this
PHASH_SIM_HIGH  = 0.75   # push phash-NN pairs that are closer than this
PHASH_WEIGHT    = 0.5    # scale of phash band loss relative to main loss
# Swap to AugmentedCard224Loader for fast synthetic augmentation
AUG_LOADER  = CompositeAugCard224Loader


# ── Loss ──────────────────────────────────────────────────────────────────────

def nt_xent_loss(aug_embs: torch.Tensor, clean_embs: torch.Tensor,
                 temperature: float, margin: float,
                 phash_nn_mask: torch.Tensor | None = None,
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Three-tier contrastive loss:

      inv_loss   : aug/clean pull  (-mean sim of positive pairs)
      sep_loss   : semi-hard repulsion of unrelated negatives (phash-NN excluded)
      phash_loss : band loss on phash-NN pairs\n                   pull if sim < PHASH_SIM_LOW, push if sim > PHASH_SIM_HIGH

    phash_nn_mask : (K, K) bool tensor, True where cards i and j are phash-neighbors.
    """
    K    = aug_embs.size(0)
    N    = 2 * K
    z    = torch.cat([aug_embs, clean_embs], dim=0)    # (2K, D)
    sim  = torch.mm(z, z.T) / temperature              # (2K, 2K)

    rows   = torch.arange(N, device=z.device)
    labels = torch.cat([
        torch.arange(K, N, device=z.device),
        torch.arange(K,    device=z.device),
    ])

    pos_sim  = sim[rows, labels]
    inv_loss = -pos_sim.mean()

    sim_neg               = sim.clone()
    sim_neg[rows, rows]   = float('-inf')
    sim_neg[rows, labels] = float('-inf')

    # Exclude phash-NN pairs from sep denominator so they are never pushed as hard negatives
    if phash_nn_mask is not None:
        sim_neg[phash_nn_mask.repeat(2, 2)] = float('-inf')

    # Semi-hard masking: only repel negatives within margin of the positive
    threshold = (pos_sim - margin / temperature).unsqueeze(1)
    sim_neg[sim_neg < threshold] = float('-inf')

    sep_per_row = torch.logsumexp(sim_neg, dim=1)
    active      = sep_per_row.isfinite()
    sep_loss    = sep_per_row[active].mean() if active.any() else sim.new_tensor(0.0)

    # phash band loss on clean-clean similarity (temperature-UNscaled)
    phash_loss = sim.new_tensor(0.0)
    if phash_nn_mask is not None and phash_nn_mask.any():
        sim_cc = torch.mm(clean_embs, clean_embs.T)   # (K, K) cosine sim, no temp
        pull_mask = phash_nn_mask & (sim_cc < PHASH_SIM_LOW)
        if pull_mask.any():
            phash_loss = phash_loss - sim_cc[pull_mask].mean()
        push_mask = phash_nn_mask & (sim_cc > PHASH_SIM_HIGH)
        if push_mask.any():
            phash_loss = phash_loss + sim_cc[push_mask].mean()

    total = inv_loss + sep_loss + PHASH_WEIGHT * phash_loss
    return total, inv_loss, sep_loss, phash_loss


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

def build_batches(embs: np.ndarray, batch_size: int, device: torch.device,
                  phash_neighbors: dict | None = None,
                  id_to_pos: dict | None = None,
                  indices: list | None = None) -> list[tuple[list[int], float]]:
    """
    Greedy NN batching. If phash_neighbors is provided, the seed's phash-neighbors
    are injected into the batch first (ensuring they always co-occur for band loss),
    then remaining slots are filled with embedding-NN.
    """
    t          = torch.from_numpy(embs).to(device)
    unassigned = torch.ones(len(embs), dtype=torch.bool, device=device)
    batches: list[tuple[list[int], float]] = []

    for seed in torch.randperm(len(embs), device=device).tolist():
        if not unassigned[seed]:
            continue

        # Inject phash-neighbors of this seed (if available and still unassigned)
        reserved: list[int] = []
        if phash_neighbors is not None and id_to_pos is not None and indices is not None:
            seed_id  = indices[seed]
            nb_ids   = phash_neighbors.get(seed_id, [])
            for nb_id in nb_ids:
                nb_pos = id_to_pos.get(int(nb_id))
                if nb_pos is not None and unassigned[nb_pos]:
                    reserved.append(nb_pos)
                    if len(reserved) >= batch_size - 1:
                        break

        # Fill remaining slots with embedding-NN
        sim = t @ t[seed]
        sim[~unassigned] = -float('inf')
        for r in reserved:
            sim[r] = float('inf')             # guarantee they're picked
        k       = int(unassigned.sum().item())
        nearest = torch.topk(sim, min(batch_size, k), largest=True).indices
        mean_s  = (t[nearest] @ t[seed]).mean().item()
        unassigned[nearest] = False
        batches.append((nearest.tolist(), mean_s))

    batches.sort(key=lambda b: b[1], reverse=True)
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
        self.id_to_pos    = {lmdb_id: pos for pos, lmdb_id in enumerate(self.indices)}
        self.phash_neighbors, self.phash_nb_sets = self._load_phash_neighbors()
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
        last_epoch = self.start_epoch - 2  # -1 for fresh start, N-2 when resuming
        if last_epoch >= 0:
            # PyTorch requires initial_lr pre-set when last_epoch >= 0
            for pg in self.optimizer.param_groups:
                pg.setdefault("initial_lr", pg["lr"])
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda, last_epoch=last_epoch
        )

    def _load_phash_neighbors(self) -> tuple[dict, dict]:
        """Load phash neighbor map. Returns (neighbors_dict, neighbor_sets_dict)."""
        if not PHASH_NEIGHBORS_PATH.exists():
            print(f"[warn] {PHASH_NEIGHBORS_PATH} not found — phash supervision disabled.")
            return {}, {}
        raw = np.load(PHASH_NEIGHBORS_PATH, allow_pickle=True).item()
        nb_sets = {k: set(int(x) for x in v) for k, v in raw.items()}
        n_with_nb = sum(1 for v in nb_sets.values() if v)
        print(f"[phash] loaded {len(raw):,} entries, {n_with_nb:,} with ≥1 neighbor")
        return raw, nb_sets

    def _build_phash_mask(self, lmdb_ids: list) -> torch.Tensor:
        """Build (K, K) bool mask: True where cards i and j are phash-neighbors."""
        K    = len(lmdb_ids)
        mask = torch.zeros(K, K, dtype=torch.bool, device=self.device)
        for i, lid_i in enumerate(lmdb_ids):
            nb_set = self.phash_nb_sets.get(lid_i)
            if nb_set:
                for j, lid_j in enumerate(lmdb_ids):
                    if i != j and lid_j in nb_set:
                        mask[i, j] = True
        return mask

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
            batches = build_batches(embs, BATCH_SIZE, self.device,
                                    self.phash_neighbors, self.id_to_pos, self.indices)
            keep_n  = max(1, int(len(batches) * BATCH_KEEP))
            batches = batches[:keep_n]

            self.model.train()
            total_loss, inv_sum, sep_sum, ph_sum, n = 0.0, 0.0, 0.0, 0.0, 0
            pbar = tqdm(batches, desc=f"epoch {epoch:4d}", leave=False)
            for pos_list, _ in pbar:
                lmdb_ids      = [self.indices[p] for p in pos_list]
                phash_mask    = self._build_phash_mask(lmdb_ids)
                x_aug         = torch.from_numpy(self.aug_loader.fetch(lmdb_ids)).to(self.device)
                x_clean       = torch.from_numpy(self.clean_loader.fetch(lmdb_ids)).to(self.device)
                loss, inv, sep, ph = nt_xent_loss(
                    self.model(x_aug), self.model(x_clean),
                    TEMPERATURE, MARGIN, phash_mask
                )
                self._step(loss)
                total_loss += loss.item(); inv_sum += inv.item()
                sep_sum += sep.item(); ph_sum += ph.item(); n += 1
                pbar.set_postfix(loss=f"{total_loss/n:.3f}",
                                 inv=f"{inv_sum/n:.3f}",
                                 sep=f"{sep_sum/n:.3f}",
                                 ph=f"{ph_sum/n:.3f}")
            cur_lr = self.optimizer.param_groups[0]['lr']
            print(f"epoch {epoch:4d}  loss {total_loss/n:.4f}"
                  f"  inv {inv_sum/n:.4f}  sep {sep_sum/n:.4f}"
                  f"  ph {ph_sum/n:.4f}  lr {cur_lr:.2e}")
            self.model.save(epoch)
            self.scheduler.step()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="Resume training from runs/recog/last.pt")
    args = ap.parse_args()
    CardEmbeddingTrainer(resume=args.resume).train()
