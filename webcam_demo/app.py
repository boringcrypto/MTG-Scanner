"""
webcam_demo/app.py
Minimal Flask server for live MTG card corner detection (pose model).

Usage:
    python webcam_demo/app.py
    python webcam_demo/app.py --model path/to/best.pt
    # then open http://localhost:5000 in your browser
"""

import sys
import base64
import argparse
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from pathlib import Path

# ── Model ─────────────────────────────────────────────────────────────────────

def _find_latest_model():
    """Find last.pt from the highest-numbered training run under runs/."""
    import re
    root = Path(__file__).parent.parent / "runs"
    best_num  = -1
    best_path = None
    for weights in root.rglob("last.pt"):
        run_dir = weights.parent.parent          # e.g. runs/pose/train6
        m = re.search(r'(\d+)$', run_dir.name)  # extract trailing number
        num = int(m.group(1)) if m else 0
        if num > best_num:
            best_num  = num
            best_path = weights
    return str(best_path) if best_path else None

p = argparse.ArgumentParser()
p.add_argument("--model", default=None, help="Path to pose best.pt")
args, _ = p.parse_known_args()

model_path = args.model or _find_latest_model()
if model_path is None:
    sys.exit("No model found. Pass --model path/to/best.pt or train first.")
print(f"Loading model: {model_path}")
model = YOLO(str(model_path))
model.fuse()

CORNER_LABELS = ["TL", "TR", "BR", "BL"]
CORNER_COLORS = [
    (0,   0,   255),   # TL — red
    (0,   255, 255),   # TR — yellow
    (255, 0,   0  ),   # BR — blue
    (255, 255, 255),   # BL — white
]

# ── App ───────────────────────────────────────────────────────────────────────

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
    Accepts a JPEG image as raw bytes in the request body.
    Returns JSON:
      {
        "detections": [
          {
            "box":     [cx, cy, w, h],        # normalised 0-1
            "corners": [[x,y], ...],           # TL TR BR BL, normalised 0-1
            "conf":    0.97
          }, ...
        ],
        "annotated": "<base64 JPEG>"
      }
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
            box  = boxes.xywhn[i].cpu().numpy().tolist()      # cx cy w h normalised

            # keypoints: (4, 2) pixel coords
            pts_px = kpts.xy[i].cpu().numpy()                 # (4, 2)
            corners_norm = [[float(x / w), float(y / h)] for x, y in pts_px]
            detections.append({"box": box, "corners": corners_norm, "conf": conf})

            # Draw card outline
            poly = pts_px.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [poly], isClosed=True,
                          color=(0, 255, 0), thickness=2)

            # Draw labeled corner dots
            for j, (px, py) in enumerate(pts_px):
                cx_i, cy_i = int(px), int(py)
                color = CORNER_COLORS[j]
                label = CORNER_LABELS[j]
                cv2.circle(annotated, (cx_i, cy_i), 7, color, -1)
                cv2.putText(annotated, label,
                            (cx_i + 8, cy_i - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            # Confidence near TL corner
            tl = pts_px[0].astype(int)
            cv2.putText(annotated, f"{conf*100:.0f}%",
                        (tl[0], tl[1] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    _, buf  = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_img = base64.b64encode(buf).decode("ascii")

    return jsonify({"detections": detections, "annotated": b64_img})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
