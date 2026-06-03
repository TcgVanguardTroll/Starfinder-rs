#!/usr/bin/env python3
"""Motion-descriptor sidecar: turn a short clip into soft-tissue *dynamics*
numbers — never pixels leave this process.

For a contiguous window of sampled frames we run MediaPipe Pose (Tasks API, same
model as body_embed.py) to locate the torso (a scale + bust/glute ROIs), then
measure per-frame vertical optical-flow magnitude inside each ROI versus the
skeletal (pelvis) motion. The ratio of tissue motion to skeletal motion is the
"jiggle" signal a still can't show; its amplitude + dominant frequency (FFT) are
the descriptor.

Output (stdout): a JSON list, one object per input path, e.g.
  {"path": "...", "ok": true, "frames": 174, "fps": 9.9, "torso_px": 263.0,
   "bust_amp": 0.041, "bust_freq": 2.3, "glute_amp": 0.038, "glute_freq": 2.1,
   "skel_amp": 0.018, "jiggle_bust": 2.28, "jiggle_glute": 2.11}

MVP/first-cut: optical flow captures all motion in the ROI (a co-star or camera
pan contaminates it), so prefer single-subject, stable-camera clips and treat
the numbers as indicative until identity-gating + camera-motion compensation
land. Scale-invariant (ROI flow / torso px) so body size alone doesn't inflate it.
"""
import json
import os
import sys
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
# Normalised-landmark indices (avoids the legacy mp.solutions import).
L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24

TARGET_FPS = 10.0   # effective sampling rate (oscillation up to ~5 Hz is plenty)
MAX_FRAMES = 240    # sampled frames scanned (~24s at 10 fps)
PROC_WIDTH = 480    # downscale width for speed; ratios are scale-free anyway
START_FRAC = 0.10   # skip intros/titles
CUT_THRESH = 0.12   # pelvis jump (in torso lengths) above which a frame pair is
#                     treated as a scene cut / re-detection and dropped


def _cached(url):
    cache = os.path.join(os.path.expanduser("~"), ".luminary")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, url.rsplit("/", 1)[-1])
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def _dominant_freq(signal, fps):
    """Dominant non-DC frequency (Hz) of a 1-D signal via rFFT."""
    s = np.asarray(signal, dtype=np.float64)
    s = s - s.mean()
    if len(s) < 8 or not np.any(s):
        return 0.0
    power = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    freqs = np.fft.rfftfreq(len(s), d=1.0 / fps)
    power[0] = 0.0  # drop DC
    return float(freqs[int(np.argmax(power))])


def _roi_mean_vflow(flow, box, w, h):
    """Mean absolute vertical flow inside a pixel box, clamped to the frame."""
    x0, y0, x1, y1 = box
    x0 = max(0, min(w - 1, int(x0)))
    x1 = max(0, min(w, int(x1)))
    y0 = max(0, min(h - 1, int(y0)))
    y1 = max(0, min(h, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return float(np.mean(np.abs(flow[y0:y1, x0:x1, 1])))


def analyse(path, landmarker):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"path": path, "ok": False, "error": "cannot open"}
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, round(src_fps / TARGET_FPS))
    eff_fps = src_fps / stride
    start = int(total * START_FRAC) if total > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    # Series collected only over *continuous* frame pairs (cuts/teleports dropped).
    # Store RAW normalised flows (no destructive subtraction) + the rigid
    # reference, so downstream can form ratios without losing dynamic range.
    bust_sig, glute_sig, rigid_sig, skel_sig, torso_samples = [], [], [], [], []
    prev_gray = None
    prev_pelvis = None
    scanned = 0
    raw = 0
    while scanned < MAX_FRAMES:
        if not cap.grab():
            break
        raw += 1
        if raw % stride != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        scanned += 1
        h0, w0 = frame.shape[:2]
        scale = PROC_WIDTH / float(w0)
        frame = cv2.resize(frame, (PROC_WIDTH, max(1, int(h0 * scale))))
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if res.pose_landmarks:
            lm = res.pose_landmarks[0]
            sh = ((lm[L_SH].x + lm[R_SH].x) / 2 * w, (lm[L_SH].y + lm[R_SH].y) / 2 * h)
            hp = ((lm[L_HIP].x + lm[R_HIP].x) / 2 * w, (lm[L_HIP].y + lm[R_HIP].y) / 2 * h)
            sw = abs(lm[L_SH].x - lm[R_SH].x) * w
            torso = max(1.0, abs(hp[1] - sh[1]))
            half = max(8.0, sw / 2)
            pelvis = hp[1] / torso
            if prev_gray is not None and prev_pelvis is not None:
                skel_dy = abs(pelvis - prev_pelvis)
                # Drop scene cuts / re-detections on another person: a continuous
                # shot moves the pelvis only a little between sampled frames.
                if skel_dy <= CUT_THRESH:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    # Rigid reference: a thin band at the shoulder line moves with
                    # the camera + body but carries little soft tissue, so ROI flow
                    # ABOVE it is the jiggle. Better than whole-frame subtraction,
                    # which over-counts background when the subject fills the frame.
                    rigid_box = (sh[0] - half, sh[1] - 0.08 * torso,
                                 sh[0] + half, sh[1] + 0.08 * torso)
                    rigid = _roi_mean_vflow(flow, rigid_box, w, h)
                    rigid = (rigid / torso) if rigid is not None else 0.0
                    bust = (sh[0] - half, sh[1] + 0.10 * torso, sh[0] + half, sh[1] + 0.55 * torso)
                    glute = (hp[0] - half, hp[1] - 0.15 * torso, hp[0] + half, hp[1] + 0.45 * torso)
                    b = _roi_mean_vflow(flow, bust, w, h)
                    g = _roi_mean_vflow(flow, glute, w, h)
                    if b is not None and g is not None:
                        bust_sig.append(b / torso)
                        glute_sig.append(g / torso)
                        rigid_sig.append(rigid)  # shoulder-band (rigid+camera) motion
                        skel_sig.append(skel_dy)
                        torso_samples.append(torso)
            prev_pelvis = pelvis
        prev_gray = gray
    cap.release()

    n = len(bust_sig)
    if n < 16:
        return {"path": os.path.basename(path), "ok": False,
                "error": f"only {n} continuous frames (cuts/camera too heavy)"}
    bust_sig = np.asarray(bust_sig)
    glute_sig = np.asarray(glute_sig)
    bust_m = float(np.mean(bust_sig))
    glute_m = float(np.mean(glute_sig))
    rigid_m = float(np.mean(rigid_sig)) + 1e-6  # shoulder-band rigid+camera motion
    return {
        "path": os.path.basename(path),
        "ok": True,
        "frames": n,
        "fps": round(eff_fps, 1),
        "torso_px": round(float(np.mean(torso_samples)), 1),
        "bust_flow": round(bust_m, 4),     # raw normalised ROI vertical flow
        "glute_flow": round(glute_m, 4),
        "rigid_flow": round(rigid_m, 4),   # the shoulder-band reference
        "bust_freq": round(_dominant_freq(bust_sig, eff_fps), 2),
        "glute_freq": round(_dominant_freq(glute_sig, eff_fps), 2),
        "skel_amp": round(float(np.mean(skel_sig)), 4),
        # jiggle = tissue motion / rigid-body motion (>1 = tissue oscillates more
        # than the frame moves). Ratio keeps dynamic range (no destructive sub).
        "jiggle_bust": round(bust_m / rigid_m, 2),
        "jiggle_glute": round(glute_m / rigid_m, 2),
    }


def make_landmarker():
    """A single-pose PoseLandmarker (Tasks API, IMAGE mode). Reused per batch."""
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_cached(POSE_MODEL_URL)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def main():
    paths = sys.argv[1:]
    if not paths:
        print(json.dumps([{"error": "Usage: motion_embed.py <video> [video ...]"}]))
        sys.exit(1)
    landmarker = make_landmarker()
    out = []
    for path in paths:
        try:
            out.append(analyse(path, landmarker))
        except Exception as e:  # noqa: BLE001 - report, never crash the batch
            out.append({"path": os.path.basename(path), "ok": False, "error": repr(e)})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
