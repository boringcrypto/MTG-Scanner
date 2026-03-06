from pathlib import Path
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME      = "vit_small_patch16_224"
CHECKPOINT_DIR  = Path("runs/recog")


class CardEmbedder(nn.Module):
    """ViT-Small backbone with L2-normalised output embeddings."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(x), dim=1)

    def save(self, epoch: int) -> Path:
        """Save backbone weights to CHECKPOINT_DIR/epoch_NNN.pt and last.pt."""
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt = CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt"
        torch.save(self.backbone.state_dict(), ckpt)
        torch.save(self.backbone.state_dict(), CHECKPOINT_DIR / "last.pt")
        return ckpt

    @classmethod
    def load(cls, path: str | Path | None = "<latest>",
             device: torch.device | None = None) -> "CardEmbedder":
        """Load a CardEmbedder.

        path='<latest>'  load CHECKPOINT_DIR/last.pt (default)
        path=None        ImageNet pretrained weights
        path=<file>      load a specific checkpoint

        If device is given the model is moved there and set to eval mode.
        """
        if path == "<latest>":
            p = CHECKPOINT_DIR / "last.pt"
            path = p if p.exists() else None

        if path is None:
            model = cls(pretrained=True)
        else:
            p = Path(path)
            model = cls(pretrained=False)
            model.backbone.load_state_dict(torch.load(p, map_location="cpu"))

        if device is not None:
            return model.eval().to(device=device, dtype=torch.float32)
        return model
