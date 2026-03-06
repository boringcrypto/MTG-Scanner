"""
webcam_demo/app.py
Minimal Flask server for live MTG card corner detection (pose model) +
card recognition via CardRecognizer.

Usage:
    python webcam_demo/app.py
    python webcam_demo/app.py --model path/to/best.pt
    # then open http://localhost:5000 in your browser
"""

import sys
from pathlib import Path

# Allow importing card_hasher / card_recognizer from the parent directory
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import base64
import argparse
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO

# -- Pose model ---------------------------------------------------------------

def _find_latest_model():
    local = Path(__file__).parent / "last.pt"
    if local.exists():
        return str(local)
    return None

p = argparse.ArgumentParser()
p.add_argument("--model",    default=None, help="Path to pose best.pt")
p.add_argument("--test-aug", action="store_true",
               help="Replace webcam crop with a random augmented LMDB image for testing")
args, _ = p.parse_known_args()
TEST_AUG = args.test_aug

model_path = args.model or _find_latest_model()
if model_path is None:
    sys.exit("No model found. Pass --model path/to/best.pt or train first.")
print(f"Loading pose model: {model_path}")
model = YOLO(str(model_path))
model.fuse()

CORNER_LABELS = ["TL", "TR", "BR", "BL"]
CORNER_COLORS = [
    (0,   0,   255),
    (0,   255, 255),
    (255, 0,   0  ),
    (255, 255, 255),
]

# -- Card recognizer ----------------------------------------------------------

import json
import torch
import lmdb
from dataclasses import dataclass
from make_training_set import warp_to_rect
from card_model import CardEmbedder
from card_hasher import _MEAN, _STD
from card_loader import AugmentedCard224Loader
from train_recog import embed_all

DEMO_DIR = Path(__file__).resolve().parent

@dataclass
class EmbedMatch:
    idx:        int    # lmdb id in cards_224.lmdb
    cosine_sim: float

    @property
    def similarity(self) -> float:
        return self.cosine_sim

    @property
    def distance(self) -> float:
        """Cosine distance (0 = identical)."""
        return round(1.0 - self.cosine_sim, 4)


class EmbedRecognizer:
    """Nearest-neighbour card recognition using a trained CardEmbedder + precomputed embeddings."""

    def __init__(self, weights: Path, embeddings: Path, cards_224: Path,
                 canonical_index: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = CardEmbedder.load(weights, device=self.device)
        with open(canonical_index) as f:
            self.indices = json.load(f)                            # position → lmdb_id
        if embeddings.exists():
            pool = np.load(embeddings).astype(np.float32)
        else:
            print(f"{embeddings.name} not found — computing embeddings now (this may take a while) ...")
            pool = embed_all(self.model, self.indices, self.device)
            np.save(embeddings, pool)
            print(f"Saved to {embeddings}")
        self.pool   = torch.from_numpy(pool).to(self.device)       # (N, D)
        with open(canonical_index) as f:
            self.indices = json.load(f)                            # position → lmdb_id
        self._cards_224 = str(cards_224)
        self._env: lmdb.Environment | None = None
        self._mean = _MEAN.reshape(1, 1, 3)
        self._std  = _STD.reshape(1, 1, 3)

    def _lmdb_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(self._cards_224, readonly=True,
                                  lock=False, readahead=False, meminit=False)
        return self._env

    def _preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 → normalised (1, 3, 224, 224) float32 tensor."""
        if bgr.shape[:2] != (224, 224):
            bgr = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = ((rgb - self._mean) / self._std).transpose(2, 0, 1)
        return torch.from_numpy(chw).unsqueeze(0).to(device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def recognize_bgr(self, bgr: np.ndarray, top_k: int = 5) -> list[EmbedMatch]:
        emb  = self.model(self._preprocess(bgr))                   # (1, D)
        sims = (emb @ self.pool.T).squeeze(0)                      # (N,)
        topk = torch.topk(sims, top_k)
        return [
            EmbedMatch(idx=self.indices[int(pos)], cosine_sim=round(float(sim), 4))
            for sim, pos in zip(topk.values.cpu(), topk.indices.cpu())
        ]

    def get_card_image(self, lmdb_id: int) -> np.ndarray:
        env = self._lmdb_env()
        with env.begin(buffers=True) as txn:
            raw = bytes(txn.get(str(lmdb_id).encode()))
        return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)


print("Loading EmbedRecognizer ...")
recognizer = EmbedRecognizer(
    weights         = DEMO_DIR / "recog.pt",
    embeddings      = DEMO_DIR / "embeddings.npy",
    cards_224       = REPO / "data/cards/cards_224.lmdb",
    canonical_index = REPO / "data/cards/canonical_index.json",
)
print("EmbedRecognizer ready.")

_aug_loader: AugmentedCard224Loader | None = None
if TEST_AUG:
    print("TEST_AUG mode: recognition will use random augmented LMDB images.")
    _aug_loader = AugmentedCard224Loader()


def _chw_to_bgr(chw: np.ndarray) -> np.ndarray:
    """Denormalise CHW float32 (ImageNet-normalised) → BGR uint8."""
    hwc = (chw.transpose(1, 2, 0) * _STD.reshape(1,1,3) + _MEAN.reshape(1,1,3)).clip(0, 1)
    rgb = (hwc * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# Standard card display size (5:7 ratio) and recogniser input
DISPLAY_W, DISPLAY_H = 280, 392
RECOG_W,   RECOG_H   = 224, 224

# -- Helpers ------------------------------------------------------------------

def _warp_card(img, pts_px, out_w, out_h):
    """Perspective-warp 4 corners (TL TR BR BL) to an upright rectangle."""
    return warp_to_rect(img, pts_px, out_w, out_h)


def _encode_b64(img, quality=85):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii")

# -- App ----------------------------------------------------------------------

STATIC = Path(__file__).parent / "static"
app = Flask(__name__, static_folder=str(STATIC))


@app.get("/")
def index():
    return send_from_directory(str(STATIC), "index.html")


@app.get("/model")
def model_info():
    return jsonify({"path": model_path})


@app.post("/detect")
def detect():
    """
    Accepts a JPEG in the request body.
    Returns JSON with detections + annotated JPEG (base64).
    """
    data = request.get_data()
    arr  = np.frombuffer(data, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "could not decode image"}), 400

    h, w = img.shape[:2]
    results = model(img, verbose=False)[0]

    detections = []
    annotated  = img.copy()
    boxes = results.boxes
    kpts  = results.keypoints

    if boxes is not None and kpts is not None:
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].cpu())
            box  = boxes.xywhn[i].cpu().numpy().tolist()
            pts_px = kpts.xy[i].cpu().numpy()
            corners_norm = [[float(x / w), float(y / h)] for x, y in pts_px]
            detections.append({"box": box, "corners": corners_norm, "conf": conf})

            poly = pts_px.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            for j, (px, py) in enumerate(pts_px):
                cx_i, cy_i = int(px), int(py)
                cv2.circle(annotated, (cx_i, cy_i), 7, CORNER_COLORS[j], -1)
                cv2.putText(annotated, CORNER_LABELS[j], (cx_i + 8, cy_i - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CORNER_COLORS[j], 2, cv2.LINE_AA)
            tl = pts_px[0].astype(int)
            cv2.putText(annotated, f"{conf*100:.0f}%", (tl[0], tl[1] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    return jsonify({"detections": detections, "annotated": _encode_b64(annotated)})


@app.post("/recognize")
def recognize():
    """
    Accepts a JPEG in the request body.
    Returns JSON:
    {
      "card":    "<base64 JPEG>",   -- perspective-corrected upright card
      "matches": [
        { "rank": 1, "idx": 42, "distance": 12, "similarity": 0.97,
          "thumb": "<base64 JPEG>" },
        ...
      ]
    }
    Returns null card + empty matches if no card is detected.
    """
    data = request.get_data()
    arr  = np.frombuffer(data, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "could not decode image"}), 400

    h, w = img.shape[:2]
    results = model(img, verbose=False)[0]
    boxes = results.boxes
    kpts  = results.keypoints

    if boxes is None or kpts is None or len(boxes) == 0:
        return jsonify({"card": None, "matches": []})

    # Pick highest-confidence detection
    confs  = boxes.conf.cpu().numpy()
    best   = int(confs.argmax())
    pts_px = kpts.xy[best].cpu().numpy()  # (4,2) TL TR BR BL

    # Ensure all corners are inside the frame
    if (pts_px < 0).any() or (pts_px[:, 0] >= w).any() or (pts_px[:, 1] >= h).any():
        return jsonify({"card": None, "matches": []})

    # Warp for recogniser (native ViT-Small resolution, stretched)
    card_recog = _warp_card(img, pts_px, RECOG_W, RECOG_H)

    if TEST_AUG and _aug_loader is not None:
        rand_id    = recognizer.indices[int(np.random.randint(len(recognizer.indices)))]
        aug_chw    = _aug_loader.fetch([rand_id])[0]   # (3, 224, 224)
        card_recog = _chw_to_bgr(aug_chw)
        print(f"  [test-aug] using lmdb_id={rand_id}")

    top5 = recognizer.recognize_bgr(card_recog, top_k=5)

    matches = []
    for rank, m in enumerate(top5, 1):
        thumb_bgr = recognizer.get_card_image(m.idx)
        matches.append({
            "rank":       rank,
            "idx":        m.idx,
            "distance":   m.distance,
            "similarity": round(m.similarity, 4),
            "thumb":      _encode_b64(thumb_bgr, quality=80),
        })

    return jsonify({
        "card":    _encode_b64(card_recog),
        "matches": matches,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
