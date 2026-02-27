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
p.add_argument("--model", default=None, help="Path to pose best.pt")
args, _ = p.parse_known_args()

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

from card_recognizer import CardRecognizer
from make_training_set import warp_to_rect

print("Loading CardRecognizer (may build index on first run) ...")
recognizer = CardRecognizer(
    cards_lmdb  = str(REPO / "data/cards/cards.lmdb"),
    cards_224   = str(REPO / "data/cards/cards_224.lmdb"),
    index_cache = str(REPO / "data/cards/card_hash_index.npy"),
)
print("CardRecognizer ready.")

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
