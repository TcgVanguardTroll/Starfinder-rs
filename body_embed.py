#!/usr/bin/env python3
"""
Generates body-shape embeddings from image URLs using MediaPipe Pose (Tasks API).

Where face_embed.py captures the *face*, this captures *build/silhouette*: it
detects 33 body landmarks and derives scale- and position-invariant proportion
ratios (shoulder/hip breadth, torso vs leg length, etc.). Two people with the
same proportions get similar vectors regardless of image size, crop, or distance.

Usage:  python body_embed.py <url1> [url2 ...]
Output: a JSON array, one entry per URL, in order:
    [{"body": [...ratios...]}, {"error": "No pose detected"}, ...]

Downloads the ~5 MB pose model on first run (cached in ~/.luminary).
"""
import sys
import os
import json
import math
import tempfile
import urllib.request

os.environ.setdefault("GLOG_minloglevel", "3")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def model_path():
    cache = os.path.join(os.path.expanduser("~"), ".luminary")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "pose_landmarker_lite.task")
    if not os.path.exists(path):
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


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


def d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midp(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def build_vector(lm):
    """Scale-invariant body-proportion ratios from 33 pose landmarks."""
    p = lambda i: (lm[i].x, lm[i].y)
    sh_w = d(p(11), p(12))           # shoulder breadth
    hip_w = d(p(23), p(24))          # hip breadth
    sh_mid, hip_mid = midp(p(11), p(12)), midp(p(23), p(24))
    torso = d(sh_mid, hip_mid)
    knee_mid = midp(p(25), p(26))
    ank_mid = midp(p(27), p(28))
    thigh = d(hip_mid, knee_mid)
    shin = d(knee_mid, ank_mid)
    leg = thigh + shin

    if torso < 1e-4 or sh_w < 1e-4 or hip_w < 1e-4:
        return None
    return [
        sh_w / torso,                          # shoulder breadth
        hip_w / torso,                         # hip breadth
        sh_w / hip_w,                          # >1 inverted-triangle, <1 pear/hourglass
        leg / torso,                           # leg length vs torso
        thigh / shin if shin > 1e-4 else 1.0,  # thigh:shin
    ]


def main():
    urls = sys.argv[1:]
    if not urls:
        print(json.dumps([{"error": "Usage: body_embed.py <url> [url ...]"}]))
        sys.exit(1)

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as e:
        print(json.dumps([{"error": f"Missing dependency: {e}. Run: pip install mediapipe"}] * len(urls)))
        sys.exit(1)

    try:
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path()),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        landmarker = vision.PoseLandmarker.create_from_options(opts)
    except Exception as e:
        print(json.dumps([{"error": f"Pose model init failed: {e}"}] * len(urls)))
        sys.exit(1)

    results = []
    for url in urls:
        tmp = None
        try:
            tmp = download(url)
            image = mp.Image.create_from_file(tmp)
            res = landmarker.detect(image)
            if not res.pose_landmarks:
                results.append({"error": "No pose detected"})
                continue
            vec = build_vector(res.pose_landmarks[0])
            results.append({"body": vec} if vec else {"error": "Degenerate pose"})
        except Exception as e:
            results.append({"error": str(e)})
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    print(json.dumps(results))


if __name__ == "__main__":
    main()
