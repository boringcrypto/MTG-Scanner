"""
debug_dataset.py
----------------
Runs train.py's normal on-disk training path (no --stream) but intercepts
get_image_and_label on the real dataset after it is built, prints the first
few samples, then exits.

Usage:
    python debug_dataset.py
"""

import sys
import numpy as np
from ultralytics import YOLO

DATASET_YAML  = "dataset/dataset.yaml"
MODEL         = "yolo26n-obb.pt"
INTERCEPT_N   = 3   # how many samples to capture before stopping

# ── Pretty-printer ────────────────────────────────────────────────────────────

def fmt_val(v):
    if isinstance(v, np.ndarray):
        return (f"ndarray shape={v.shape} dtype={v.dtype} "
                f"min={float(v.min()):.4f} max={float(v.max()):.4f}")
    try:
        from ultralytics.utils.instance import Instances
        if isinstance(v, Instances):
            bboxes = v.bboxes
            segs   = v.segments
            return (
                f"Instances bboxes shape={bboxes.shape} dtype={bboxes.dtype} "
                f"min={float(bboxes.min()):.4f} max={float(bboxes.max()):.4f} | "
                f"segments={segs.shape if segs is not None else None} | "
                f"format={v._bboxes.format} normalized={v.normalized}"
            )
    except Exception:
        pass
    return repr(v)


def print_sample(idx, result):
    print(f"\n{'='*60}")
    print(f"  Sample index: {idx}  im_file: {result.get('im_file', '?')}")
    print(f"{'='*60}")
    for k, v in result.items():
        print(f"  {k:20s}: {fmt_val(v)}")


# ── Callback: patch dataset after it's built, capture N samples then stop ─────

class _Done(Exception):
    pass

def on_train_start(trainer):
    ds = trainer.train_loader.dataset
    orig = ds.__class__.get_image_and_label

    def patched(self, index):
        result = orig(self, index)
        print_sample(result)
        return result

    ds.__class__.get_image_and_label = patched
    print(f"\n[debug] Hooked get_image_and_label on {ds.__class__.__name__} "
          f"(len={len(ds)})\n")


# ── Run training (will abort after first few samples) ────────────────────────

model = YOLO(MODEL)
model.add_callback("on_train_start", on_train_start)

try:
    model.train(
        data       = DATASET_YAML,
        task       = "obb",
        epochs     = 1,
        batch      = 4,
        imgsz      = 640,
        workers    = 0,   # single process so the patch stays in scope
        device     = "cpu",
        plots      = False,
        save       = False,
        verbose    = False,
    )
except (_Done, Exception) as e:
    print(f"\n[debug] Stopped: {e}")
    sys.exit(0)
