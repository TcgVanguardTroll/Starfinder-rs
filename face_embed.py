#!/usr/bin/env python3
"""
Generates ArcFace face embeddings from image URLs.
Called by starfinder: python face_embed.py <image_url>
Output: JSON {"embedding": [...512 floats...]} or {"error": "message"}

First run downloads the ArcFace model weights (~100 MB, cached after that).
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
        from deepface import DeepFace
    except ImportError:
        print(json.dumps({
            "error": "deepface not installed. Run:  pip install deepface tf-keras"
        }))
        sys.exit(1)

    tmp = None
    try:
        # Download image to a temp file — DeepFace works best with file paths
        ext = url.split(".")[-1].split("?")[0].lower()
        if ext not in ("jpg", "jpeg", "png", "webp", "gif"):
            ext = "jpg"
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
            tmp = f.name

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Starfinder)"},
        )
        with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
            f.write(r.read())

        # RetinaFace is the most accurate detector; OpenCV is faster fallback
        result = None
        for backend in ("retinaface", "opencv"):
            try:
                result = DeepFace.represent(
                    img_path=tmp,
                    model_name="ArcFace",
                    enforce_detection=True,
                    detector_backend=backend,
                )
                if result:
                    break
            except Exception:
                pass

        # Last attempt with enforce_detection=False (still crops, just lenient)
        if not result:
            result = DeepFace.represent(
                img_path=tmp,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="opencv",
            )

        if result and result[0].get("embedding"):
            print(json.dumps({"embedding": result[0]["embedding"]}))
        else:
            print(json.dumps({"error": "No face detected in image"}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


if __name__ == "__main__":
    main()
