#!/usr/bin/env python3
"""Figure-scan prototype: recover a performer's waist/hip ratio from VIDEO
silhouette, robustly (median over many frames/angles), to test whether video
gives a truer figure than a single (deceptive) still.

A real figure is 3D, and circumference needs BOTH width (front view) and depth
(side view) — which only video reliably supplies for the same person. Here we
measure, scale-free:
  WHR_front = waist_width / hip_width   (front frames)
  WHR_side  = waist_depth / hip_depth   (side frames; width_at on a profile spans
                                         front->back, i.e. depth)
and compare to the performer's RECORDED waist/hip. Reuses body_embed's exact
silhouette measurement (width_at + the waist/hip anchors). Local, numbers-only.
"""
import json
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

SAMPLES_PER_CLIP = 50
PROC_WIDTH = 640


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


def scan(paths, pose, seg):
    front_whr, side_whr = [], []
    for path in paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            continue
        try:  # release the capture even if a frame raises (handle leak otherwise)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            stride = max(1, total // SAMPLES_PER_CLIP) if total else 30
            i = 0
            taken = 0
            while taken < SAMPLES_PER_CLIP:
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
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = pose.detect(image)
                if not res.pose_landmarks:
                    continue
                lm = res.pose_landmarks[0]
                if be._min_visibility(lm, (11, 12, 23, 24)) < be.MIN_VISIBILITY:
                    continue
                sres = seg.segment(image)
                if not sres.confidence_masks:
                    continue
                mask = sres.confidence_masks[0].numpy_view()
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                _, w = mask.shape
                sh_y = (lm[11].y + lm[12].y) / 2
                hip_y = (lm[23].y + lm[24].y) / 2
                torso = hip_y - sh_y
                if torso < 1e-4:
                    continue
                mid = int(((lm[23].x + lm[24].x) / 2) * w)

                # waist = NARROWEST width in the lower-torso band; hip = WIDEST around
                # the hips/glutes. min/max over a band finds the true waist/hip
                # regardless of the exact anchor and is far more robust to posture
                # than a single fixed y.
                def band(y0, y1, fn):
                    vals = [be.width_at(mask, float(y), mid) for y in np.linspace(y0, y1, 7)]
                    vals = [v for v in vals if v > 1e-4]
                    return fn(vals) if vals else None

                waist = band(sh_y + 0.45 * torso, sh_y + 0.78 * torso, min)
                hip = band(hip_y - 0.10 * torso, hip_y + 0.30 * torso, max)
                if not waist or not hip:
                    continue
                view = be.classify_view(lm)
                sw = abs(lm[11].x - lm[12].x)
                if view in ("front", "rear") and sw >= 0.08:
                    front_whr.append(waist / hip)
                elif view == "side":
                    side_whr.append(waist / hip)   # widths here are depths
        finally:
            cap.release()
    return front_whr, side_whr


def main():
    if len(sys.argv) < 2:
        print("Usage: figure_scan.py <video> [video ...]")
        sys.exit(1)
    pose, seg = models()
    fw, sw = scan(sys.argv[1:], pose, seg)
    out = {
        "front_frames": len(fw),
        "side_frames": len(sw),
        "WHR_front_median": round(st.median(fw), 3) if fw else None,
        "WHR_side_median": round(st.median(sw), 3) if sw else None,
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
