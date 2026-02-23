"""
browser_app.py
Minimal Flask server that serves browser_demo.html and model.onnx.
All inference runs client-side in the browser via ONNX Runtime Web.

Usage:
    python webcam_demo/browser_app.py
    # then open http://localhost:5001
"""

from flask import Flask, send_from_directory
from pathlib import Path

STATIC = Path(__file__).parent / "static"
app = Flask(__name__, static_folder=str(STATIC))

@app.get("/")
def index():
    return send_from_directory(str(STATIC), "browser_demo.html")

@app.get("/model.onnx")
def model_onnx():
    return send_from_directory(str(STATIC), "model.onnx")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
