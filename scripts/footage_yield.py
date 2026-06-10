#!/usr/bin/env python3
"""Yield test: sample frames evenly across each clip and run the EXACT body_embed
CV (pose + segmentation -> classify_view / build_vector / build_seg_vector /
build_proj_vector), to measure how many clean front / rear / side vectors the
footage yields per clip. Decides whether the footage is worth wiring up as a
`footage` ImageSource (esp. for the scarce side->proj signal).

Local + numbers-only (no pixels leave the process, no DB writes). Usage:
  footage_yield.py <video> [video ...]
"""
import json
import os
import sys

import cv2
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import body_embed as be  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

SAMPLES_PER_CLIP = 50   # frames sampled evenly across the whole clip
PROC_WIDTH = 640        # enough resolution for the seg mask / proj


def make_models():
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=be._cached(be.POSE_MODEL_URL)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
    )
    seg = vision.ImageSegmenter.create_from_options(
        vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=be._cached(be.SEG_MODEL_URL)),
            output_confidence_masks=True,
        )
    )
    return pose, seg


def analyse(path, pose, seg):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"path": os.path.basename(path), "ok": False, "error": "cannot open"}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, total // SAMPLES_PER_CLIP) if total else 30
    tally = {"sampled": 0, "pose": 0, "front": 0, "rear": 0, "side": 0,
             "pose_vec": 0, "seg_vec": 0, "proj_vec": 0}
    idx = 0
    while tally["sampled"] < SAMPLES_PER_CLIP:
        if not cap.grab():
            break
        idx += 1
        if idx % stride != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        tally["sampled"] += 1
        h0, w0 = frame.shape[:2]
        fr = cv2.resize(frame, (PROC_WIDTH, max(1, int(h0 * PROC_WIDTH / w0))))
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = pose.detect(image)
        if not res.pose_landmarks:
            continue
        tally["pose"] += 1
        lm = res.pose_landmarks[0]
        view = be.classify_view(lm)
        if view in tally:
            tally[view] += 1
        try:
            sres = seg.segment(image)
            mask = sres.confidence_masks[0].numpy_view() if sres.confidence_masks else None
        except Exception:  # noqa: BLE001
            mask = None
        if be.build_vector(lm):
            tally["pose_vec"] += 1
        if mask is not None and be.build_seg_vector(lm, mask):
            tally["seg_vec"] += 1
        if mask is not None and be.build_proj_vector(lm, mask):
            tally["proj_vec"] += 1
    cap.release()
    return {"path": os.path.basename(path), "ok": True, **tally}


def main():
    paths = sys.argv[1:]
    if not paths:
        print(json.dumps([{"error": "Usage: footage_yield.py <video> [...]"}]))
        sys.exit(1)
    pose, seg = make_models()
    out = [analyse(p, pose, seg) for p in paths]
    print(json.dumps(out, indent=1))
    okrows = [r for r in out if r.get("ok")]
    if okrows:
        agg = {k: sum(r.get(k, 0) for r in okrows)
               for k in ("sampled", "pose", "front", "rear", "side",
                         "pose_vec", "seg_vec", "proj_vec")}
        print(f"\nTOTAL over {len(okrows)} clip(s): {agg}", flush=True)
        print(f"  usable per clip: pose {agg['pose_vec']/len(okrows):.1f}, "
              f"seg {agg['seg_vec']/len(okrows):.1f}, "
              f"PROJ(side) {agg['proj_vec']/len(okrows):.1f}", flush=True)


if __name__ == "__main__":
    main()
