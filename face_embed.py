#!/usr/bin/env python3
"""
Generates ArcFace face embeddings from image URLs using InsightFace + ONNX Runtime.
No TensorFlow required — works with Python 3.14+.

Usage:  python face_embed.py <url1> [url2 url3 ...]

The model is loaded ONCE per invocation, so passing many URLs at once is far
cheaper than one call per image (model load dominates per-call cost). Uses the
CUDA execution provider automatically when available, else CPU.

Output: a JSON array, one entry per input URL, in order:
    [{"embedding": [...512 floats...]}, {"error": "No face detected"}, ...]

First run downloads the buffalo_l model (~300 MB, cached after that).
"""
import sys
import json
import os
import glob
import tempfile
import urllib.request


def _register_cuda_dlls():
    """Add the pip-installed NVIDIA CUDA/cuDNN bin dirs to the DLL search path
    so onnxruntime-gpu can create CUDA sessions without a system CUDA toolkit.
    Must run before onnxruntime is imported."""
    try:
        import nvidia
        base = list(nvidia.__path__)[0]
        for d in glob.glob(os.path.join(base, "*", "bin")):
            if os.path.isdir(d):
                os.add_dll_directory(d)
    except Exception:
        pass


def download(url):
    ext = url.split(".")[-1].split("?")[0].lower()
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "jpg"
    fd, path = tempfile.mkstemp(suffix=f".{ext}")
    os.close(fd)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Luminary)"})
    with urllib.request.urlopen(req, timeout=30) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def main():
    urls = sys.argv[1:]
    if not urls:
        print(json.dumps([{"error": "Usage: face_embed.py <url> [url ...]"}]))
        sys.exit(1)

    _register_cuda_dlls()
    try:
        import cv2
        from insightface.app import FaceAnalysis
        import onnxruntime as ort
    except ImportError as e:
        err = {"error": f"Missing dependency: {e}. Run: pip install insightface onnxruntime opencv-python"}
        print(json.dumps([err] * len(urls)))
        sys.exit(1)

    # Prefer GPU (CUDA) when the installed onnxruntime exposes it; else CPU.
    available = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]

    # Load the model ONCE — this is the expensive part we amortise across URLs.
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0, det_size=(640, 640))
    finally:
        sys.stdout = _real_stdout

    results = []
    for url in urls:
        tmp = None
        try:
            tmp = download(url)
            img = cv2.imread(tmp)
            if img is None:
                results.append({"error": "Could not decode image"})
                continue
            faces = app.get(img)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                results.append({"embedding": face.embedding.tolist()})
            else:
                results.append({"error": "No face detected"})
        except Exception as e:
            results.append({"error": str(e)})
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    print(json.dumps(results))


if __name__ == "__main__":
    main()
