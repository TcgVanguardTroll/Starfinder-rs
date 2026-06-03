#!/usr/bin/env python3
"""Known-height figure scan: use a performer's KNOWN height as the scale ruler to
turn video silhouette into ABSOLUTE measurements (cm/inches), then a rough
physics weight estimate (elliptical cross-sections -> volume -> mass).

Per full-body, upright, frontal frame: pixel height (nose->ankle, /0.84 to
recover full stature) + known height => cm-per-pixel. Then silhouette widths at
shoulder/waist/hip convert to cm; circumference via an ellipse with an assumed
depth:width ratio; volume via stacked elliptical disks -> weight at ~1.0 kg/L.

Local, numbers-only, read-only. Usage: figure_physics.py <video> <height_cm>
"""
import json
import math
import os
import statistics as st
import sys

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import body_embed as be  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

NOSE, L_ANK, R_ANK = 0, 27, 28
NOSE_ANKLE_FRAC = 0.84   # nose->ankle vertical span as a fraction of full height
DEPTH_RATIO = 0.72       # assumed body depth:width at the torso (anthropometric)
TRUNK_MASS_FRAC = 0.46   # trunk share of body mass (to extrapolate full weight)
SAMPLES = int(sys.argv[3]) if len(sys.argv) > 3 else 150  # frames sampled per clip
MIN_FULL_BODY = 3        # minimum scale-anchor frames to report
PROC_WIDTH = 720


def models():
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=be._cached(be.POSE_MODEL_URL)),
            running_mode=vision.RunningMode.IMAGE, num_poses=1))
    seg = vision.ImageSegmenter.create_from_options(
        vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=be._cached(be.SEG_MODEL_URL)),
            output_confidence_masks=True))
    return pose, seg


def ellipse_circ(a, b):
    if a + b <= 0:
        return 0.0
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


def main():
    path, height_cm = sys.argv[1], float(sys.argv[2])
    pose, seg = models()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(json.dumps({"ok": False, "error": "cannot open"}))
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, total // SAMPLES) if total else 30
    waist_cm, hip_cm, shoulder_cm, trunk_cm = [], [], [], []
    funnel = {"sampled": 0, "pose": 0, "nose+ankles_visible": 0, "upright": 0, "frontal": 0}
    i = taken = full_body = 0
    while taken < SAMPLES:
        if not cap.grab():
            break
        i += 1
        if i % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        taken += 1
        h0, w0 = frame.shape[:2]
        fr = cv2.resize(frame, (PROC_WIDTH, max(1, int(h0 * PROC_WIDTH / w0))))
        hpx, wpx = fr.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        funnel["sampled"] += 1
        res = pose.detect(image)
        if not res.pose_landmarks:
            continue
        funnel["pose"] += 1
        lm = res.pose_landmarks[0]
        # Scale needs head-to-toe + upright + frontal (shoulders don't collapse).
        if be._min_visibility(lm, (NOSE, L_ANK, R_ANK)) < be.MIN_VISIBILITY:
            continue
        funnel["nose+ankles_visible"] += 1
        if not be.is_upright(lm):
            continue
        funnel["upright"] += 1
        if be.classify_view(lm) not in ("front", "rear"):
            continue
        funnel["frontal"] += 1
        ankle_y = max(lm[L_ANK].y, lm[R_ANK].y)
        span = (ankle_y - lm[NOSE].y) * hpx
        if span <= 1:
            continue
        cmpp = height_cm / (span / NOSE_ANKLE_FRAC)   # cm per pixel
        sres = seg.segment(image)
        if not sres.confidence_masks:
            continue
        mask = sres.confidence_masks[0].numpy_view()
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        sh_y = (lm[11].y + lm[12].y) / 2
        hip_y = (lm[23].y + lm[24].y) / 2
        torso = hip_y - sh_y
        if torso < 1e-3:
            continue
        mid = int(((lm[23].x + lm[24].x) / 2) * wpx)
        to_cm = lambda frac: frac * wpx * cmpp  # silhouette frac -> cm  # noqa: E731

        def band(y0, y1, fn):
            vals = [be.width_at(mask, float(y), mid) for y in np.linspace(y0, y1, 7)]
            vals = [v for v in vals if v > 1e-4]
            return fn(vals) if vals else None

        sh = be.width_at(mask, sh_y, mid)
        wa = band(sh_y + 0.45 * torso, sh_y + 0.78 * torso, min)
        hi = band(hip_y - 0.10 * torso, hip_y + 0.30 * torso, max)
        if not (sh and wa and hi):
            continue
        full_body += 1
        shoulder_cm.append(to_cm(sh))
        waist_cm.append(to_cm(wa))
        hip_cm.append(to_cm(hi))
        trunk_cm.append(torso * hpx * cmpp)
    cap.release()

    if full_body < MIN_FULL_BODY:
        print(json.dumps({"ok": False, "full_body_frames": full_body, "funnel": funnel,
                          "error": "too few full-body frames to calibrate scale"}))
        return

    sw, ww, hw = st.median(shoulder_cm), st.median(waist_cm), st.median(hip_cm)
    trunk_h = st.median(trunk_cm)
    # Circumference: ellipse with assumed depth = width * DEPTH_RATIO.
    waist_circ = ellipse_circ(ww / 2, ww * DEPTH_RATIO / 2)
    hip_circ = ellipse_circ(hw / 2, hw * DEPTH_RATIO / 2)
    # Rough volume: mean elliptical cross-section area over the trunk height, then
    # extrapolate trunk -> whole body by mass fraction. ~1.0 kg/L.
    def area(w):
        return math.pi * (w / 2) * (w * DEPTH_RATIO / 2)
    trunk_vol_cm3 = st.mean([area(sw), area(ww), area(hw)]) * trunk_h
    weight_kg = (trunk_vol_cm3 / 1000.0) / TRUNK_MASS_FRAC
    out = {
        "ok": True, "full_body_frames": full_body, "known_height_cm": height_cm,
        "shoulder_width_cm": round(sw, 1), "waist_width_cm": round(ww, 1),
        "hip_width_cm": round(hw, 1),
        "WHR_width": round(ww / hw, 3),
        "est_waist_in": round(waist_circ / 2.54, 1),
        "est_hip_in": round(hip_circ / 2.54, 1),
        "est_weight_kg_rough": round(weight_kg, 1),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
