#!/usr/bin/env python3
"""
Generates ArcFace face embeddings from image URLs using InsightFace + ONNX Runtime.
No TensorFlow required — works with Python 3.14+.

Called by starfinder: python face_embed.py <image_url>
Output: JSON {"embedding": [...512 floats...]} or {"error": "message"}

First run downloads the buffalo_l model (~300 MB, cached after that).
"""
import sys
import json
import os
import tempfile
import urllib.request


def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: face_embed.py <image_url>"}))
        sys.exit(1)

    url = sys.argv[1]

    try:
        import cv2
        import numpy as np
        from insightface.app import FaceAnalysis
    except ImportError as e:
        print(json.dumps({
            "error": f"Missing dependency: {e}. Run: pip install insightface onnxruntime opencv-python"
        }))
        sys.exit(1)

    tmp = None
    try:
        # Download image to temp file
        ext = url.split(".")[-1].split("?")[0].lower()
        if ext not in ("jpg", "jpeg", "png", "webp"):
            ext = "jpg"
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
            tmp = f.name

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Starfinder)"},
        )
        with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
            f.write(r.read())

        # Load image
        img = cv2.imread(tmp)
        if img is None:
            print(json.dumps({"error": "Could not decode image"}))
            return

        # Initialise InsightFace — redirect its stdout chatter to stderr
        import io
        _real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
        finally:
            sys.stdout = _real_stdout

        faces = app.get(img)

        if not faces:
            _real_stdout = sys.stdout
            sys.stdout = sys.stderr
            try:
                app.prepare(ctx_id=0, det_size=(320, 320))
            finally:
                sys.stdout = _real_stdout
            faces = app.get(img)

        if faces:
            # Use the largest detected face (most prominent)
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embedding = face.embedding.tolist()
            print(json.dumps({"embedding": embedding}))
        else:
            print(json.dumps({"error": "No face detected in image"}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


if __name__ == "__main__":
    main()
