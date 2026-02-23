"""
export_onnx.py
Export the YOLO26 pose model to ONNX for browser-side inference.

Usage:
    python export_onnx.py
    python export_onnx.py --model runs/.../weights/best.pt
    python export_onnx.py --model webcam_demo/last.pt --out webcam_demo/static/model.onnx

The resulting model.onnx is served as a static file by Flask and loaded
by browser_demo.html for fully client-side inference.

# Output tensor shape:  [1, 300, 18]
#   300 = max detections (NMS already applied by YOLO26 ONNX export)
#   18  = x1 y1 x2 y2 (xyxy, model-pixel space)
#         + conf + cls
#         + 4 kpts × 3 (x, y, visibility)  — coordinates in model-pixel space
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ── Args ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--model", default=None,
               help="Path to .pt weights (default: webcam_demo/last.pt)")
p.add_argument("--out", default=None,
               help="Destination for model.onnx (default: same folder as .pt)")
p.add_argument("--imgsz", type=int, default=640)
args = p.parse_args()

model_path = Path(args.model or "webcam_demo/last.pt")
if not model_path.exists():
    sys.exit(f"Model not found: {model_path}")

# ── Export ────────────────────────────────────────────────────────────────────
print(f"Loading {model_path} …")
model = YOLO(str(model_path))

export_path = model.export(
    format   = "onnx",
    imgsz    = args.imgsz,
    simplify = True,   # fold constant ops; reduces graph size
    opset    = 17,
    dynamic  = False,  # fixed batch=1 is fine for browser inference
)
export_path = Path(export_path)
print(f"Exported → {export_path}")

# ── Move to desired location ──────────────────────────────────────────────────
dest = Path(args.out) if args.out else Path("webcam_demo/static/model.onnx")
dest.parent.mkdir(parents=True, exist_ok=True)
if dest != export_path:
    export_path.rename(dest)
    print(f"Moved   → {dest}")
else:
    print(f"Saved   → {dest}")

# ── Inspect output shape ──────────────────────────────────────────────────────
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
    inp  = sess.get_inputs()[0]
    print(f"\nInput  : name={inp.name!r}  shape={inp.shape}  dtype={inp.type}")
    dummy = np.zeros([1, 3, args.imgsz, args.imgsz], dtype=np.float32)
    out   = sess.run(None, {inp.name: dummy})
    for i, o in enumerate(sess.get_outputs()):
        print(f"Output[{i}]: name={o.name!r}  shape={out[i].shape}  dtype={out[i].dtype}")
    print("\nDone. Copy/rename to webcam_demo/static/model.onnx if not already there.")
except ImportError:
    print("\nonnxruntime not installed — skipping shape check.")
    print("Install with:  pip install onnxruntime")
