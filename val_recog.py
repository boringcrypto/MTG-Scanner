"""
val_recog.py – Retrieval validation and visualisation for CardEmbedder.

Saved to CHECKPOINT_DIR each epoch:
  val_epoch_NNN_retrievals.png  : worst SHOW_WORST + best SHOW_BEST rows
                                  each row: [aug probe | correct clean | retrieved clean]
  val_epoch_NNN_stats.png       : rank histogram (log) + cosine sim distributions
  training_log.csv              : running per-epoch metrics
  training_curve.png            : top-1 % and mean rank over all epochs so far
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from card_hasher import _MEAN, _STD
from card_model import CHECKPOINT_DIR

_MEAN_ARR = np.array(_MEAN, dtype=np.float32).reshape(1, 1, 3)   # (1,1,3) for broadcast
_STD_ARR  = np.array(_STD,  dtype=np.float32).reshape(1, 1, 3)

VAL_SIZE   = 1024
SHOW_WORST = 10
SHOW_BEST  = 5
THUMB      = 224     # image cell size (px)
BORDER     = 4       # border width (px)
LABEL_H    = 18      # pixel height of caption row below each image


def _to_pil(chw: np.ndarray) -> Image.Image:
    """Denormalize CHW float32 (ImageNet-normalised) → PIL Image."""
    hwc = chw.transpose(1, 2, 0) * _STD_ARR + _MEAN_ARR
    return Image.fromarray((hwc.clip(0.0, 1.0) * 255).astype(np.uint8))


def _bordered(img: Image.Image, color: tuple) -> Image.Image:
    """Return a new image with a solid colour border."""
    w, h = img.size
    out = Image.new("RGB", (w + 2 * BORDER, h + 2 * BORDER), color)
    out.paste(img, (BORDER, BORDER))
    return out


class EmbeddingValidator:

    def __init__(self, aug_loader, clean_loader, model, indices, device):
        self.aug_loader   = aug_loader
        self.clean_loader = clean_loader
        self.model        = model
        self.indices      = indices        # list – maps position → lmdb id
        self.device       = device
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self._log_path    = CHECKPOINT_DIR / "training_log.csv"
        self._log_fields  = ["epoch", "top1", "top5", "top10",
                              "mean_rank", "median_rank", "cos_correct", "cos_nn"]

    @torch.no_grad()
    def validate(self, pool: np.ndarray, epoch: int) -> dict:
        """Run retrieval validation; save images + logs; return metrics dict."""
        self.model.eval()
        N = len(self.indices)

        # ── Probe embeddings ─────────────────────────────────────────────────
        val_pos  = torch.randperm(N)[:VAL_SIZE].tolist()      # pool positions
        lmdb_ids = [self.indices[p] for p in val_pos]
        aug_np   = self.aug_loader.fetch(lmdb_ids)            # (V, C, H, W)
        aug_embs = self.model(torch.from_numpy(aug_np).to(self.device))   # (V, D)

        pool_gpu = torch.from_numpy(pool).to(self.device)     # (N, D)
        sim      = aug_embs @ pool_gpu.T                      # (V, N)  cosine sim

        # ── Metrics ──────────────────────────────────────────────────────────
        val_pos_t   = torch.tensor(val_pos, device=self.device)
        correct_sim = sim[torch.arange(VAL_SIZE, device=self.device), val_pos_t]  # (V,)
        ranks       = (sim >= correct_sim.unsqueeze(1)).sum(dim=1).float()         # (V,)

        top1        = (ranks <= 1).float().mean().item()
        top5        = (ranks <= 5).float().mean().item()
        top10       = (ranks <= 10).float().mean().item()
        mean_rank   = ranks.mean().item()
        median_rank = ranks.median().item()
        nn_sim      = sim.max(dim=1).values.mean().item()
        correct_cos = correct_sim.mean().item()

        metrics = dict(epoch=epoch, top1=top1, top5=top5, top10=top10,
                       mean_rank=mean_rank, median_rank=median_rank,
                       cos_correct=correct_cos, cos_nn=nn_sim)

        # ── Visualisations ───────────────────────────────────────────────────
        retrieved_pos = sim.argmax(dim=1).cpu().tolist()
        ranks_np      = ranks.cpu().numpy()
        correct_np_v  = correct_sim.cpu().numpy()
        nn_np         = sim.max(dim=1).values.cpu().numpy()

        self._retrieval_sheet(aug_np, val_pos, retrieved_pos, ranks_np, epoch)
        self._stats_plot(ranks_np, correct_np_v, nn_np, epoch)
        self._append_log(metrics)
        self._training_curve()

        self.model.train()
        print(f"  val   top1 {metrics['top1']:.1%}  top5 {metrics['top5']:.1%}  top10 {metrics['top10']:.1%}"
              f"  mean_rank {metrics['mean_rank']:.1f}  median_rank {metrics['median_rank']:.0f}"
              f"  cos_correct {metrics['cos_correct']:.4f}  cos_nn {metrics['cos_nn']:.4f}")
        return metrics

    # ── Retrieval sheet ──────────────────────────────────────────────────────

    def _retrieval_sheet(self, aug_np, val_pos, retrieved_pos, ranks, epoch):
        """
        Grid of SHOW_WORST worst + SHOW_BEST best retrievals.
        Columns: [aug probe | correct clean | top-1 retrieved]
        Retrieved column border: green = correct, red = wrong.
        """
        order    = np.argsort(ranks)[::-1]
        worst_i  = order[:SHOW_WORST].tolist()
        best_i   = order[::-1][:SHOW_BEST].tolist()
        rows_i   = worst_i + best_i
        row_tags = ["W"] * SHOW_WORST + ["B"] * SHOW_BEST

        def fetch_clean(positions):
            return self.clean_loader.fetch([self.indices[p] for p in positions])

        correct_imgs  = fetch_clean([val_pos[i]       for i in rows_i])
        retrieved_imgs = fetch_clean([retrieved_pos[i] for i in rows_i])

        n_rows  = len(rows_i)
        cell    = THUMB + 2 * BORDER
        row_h   = cell + LABEL_H
        sheet   = Image.new("RGB", (3 * cell, n_rows * row_h), (30, 30, 30))
        draw    = ImageDraw.Draw(sheet)

        col_labels = ["aug (probe)", "correct", "retrieved"]

        for r, (i, tag) in enumerate(zip(rows_i, row_tags)):
            is_ok   = (retrieved_pos[i] == val_pos[i])
            rank_v  = int(ranks[i])
            border3 = (0, 180, 0) if is_ok else (200, 50, 50)
            y0      = r * row_h

            for c, (img_np, border) in enumerate([
                (aug_np[i],          (80, 80, 80)),
                (correct_imgs[r],    (80, 80, 80)),
                (retrieved_imgs[r],  border3),
            ]):
                pil    = _bordered(_to_pil(img_np), border)
                x0     = c * cell
                sheet.paste(pil, (x0, y0))
                caption = col_labels[c] if r == 0 else ""
                if c == 2:
                    caption = f"{'✓' if is_ok else '✗'} rank {rank_v}"
                draw.text((x0 + BORDER + 2, y0 + cell + 1),
                          caption, fill=(180, 180, 180))

            # row badge (W / B) on right edge
            draw.text((3 * cell - 14, y0 + cell // 2 - 5),
                      tag, fill=(220, 180, 60) if tag == "W" else (60, 180, 60))

        out = CHECKPOINT_DIR / f"val_epoch_{epoch:03d}_retrievals.png"
        sheet.save(out)

    # ── Stats plot ───────────────────────────────────────────────────────────

    def _stats_plot(self, ranks, correct_sims, nn_sims, epoch):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        _dark(fig, axes)

        axes[0].hist(ranks, bins=60, color="#5b9bd5", edgecolor="none")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Rank")
        axes[0].set_ylabel("Count (log)")
        axes[0].set_title(f"Rank distribution  (epoch {epoch})")

        axes[1].hist(correct_sims, bins=60, alpha=0.75, color="#70ad47", label="correct pair")
        axes[1].hist(nn_sims,      bins=60, alpha=0.75, color="#ed7d31", label="nearest neighbour")
        axes[1].set_xlabel("Cosine similarity")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Similarity distributions")
        axes[1].legend(facecolor="#3d3d3d", labelcolor="white")

        plt.tight_layout()
        fig.savefig(CHECKPOINT_DIR / f"val_epoch_{epoch:03d}_stats.png",
                    dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)

    # ── Training log & curve ─────────────────────────────────────────────────

    def _append_log(self, metrics: dict):
        write_header = not self._log_path.exists()
        with open(self._log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._log_fields)
            if write_header:
                w.writeheader()
            w.writerow({k: metrics[k] for k in self._log_fields})

    def _training_curve(self):
        if not self._log_path.exists():
            return
        epochs, top1s, mean_ranks = [], [], []
        with open(self._log_path) as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]))
                top1s.append(float(row["top1"]) * 100)
                mean_ranks.append(float(row["mean_rank"]))

        fig, ax1 = plt.subplots(figsize=(10, 4))
        _dark(fig, [ax1])
        ax1.plot(epochs, top1s, color="#70ad47", label="top-1 %")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Top-1 %", color="#70ad47")
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis="y", colors="#70ad47")

        ax2 = ax1.twinx()
        ax2.set_facecolor("#2d2d2d")
        ax2.tick_params(colors="white")
        ax2.tick_params(axis="y", colors="#ed7d31")
        ax2.plot(epochs, mean_ranks, color="#ed7d31", linestyle="--", label="mean rank")
        ax2.set_ylabel("Mean rank", color="#ed7d31")

        lines  = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, facecolor="#3d3d3d", labelcolor="white")
        ax1.set_title("Training progress", color="white")

        plt.tight_layout()
        fig.savefig(CHECKPOINT_DIR / "training_curve.png",
                    dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)


# ── Matplotlib dark theme helper ─────────────────────────────────────────────

def _dark(fig, axes):
    fig.patch.set_facecolor("#1e1e1e")
    for ax in axes:
        ax.set_facecolor("#2d2d2d")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")
