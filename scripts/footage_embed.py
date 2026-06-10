#!/usr/bin/env python3
"""Footage extraction (#24): turn a performer's clips into the per-frame body
vectors the corpus stores — especially SIDE-view proj (butt shape), the signal
stills miss. Reuses body_embed's exact CV so vectors match a real ingest.

Per clip, samples frames evenly; for each pose-detected frame emits
{view, quality, pose, seg, proj} using build_vector / build_seg_vector /
build_proj_vector + classify_view. Output (stdout): JSON list, one object per
clip: {clip, frames:[{view,quality,pose?,seg?,proj?}, ...]}.

Local, numbers-only (no pixels leave the process, no DB writes). Identity gating
+ DB insert are the apply step (run post-ingest). Usage:
  footage_embed.py <video> [video ...]
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

SAMPLES_PER_CLIP = 60
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


def embed_clip(path, pose, seg):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"clip": os.path.basename(path), "ok": False, "error": "cannot open"}
    try:  # release the capture even if a frame raises (handle leak otherwise)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, total // SAMPLES_PER_CLIP) if total else 30
        frames = []
        i = taken = 0
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
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            res = pose.detect(image)
            if not res.pose_landmarks:
                continue
            lm = res.pose_landmarks[0]
            mask = None
            try:
                sres = seg.segment(image)
                if sres.confidence_masks:
                    mask = sres.confidence_masks[0].numpy_view()
            except Exception:  # noqa: BLE001
                pass
            pose_vec = be.build_vector(lm)
            seg_vec = be.build_seg_vector(lm, mask) if mask is not None else None
            proj_vec = be.build_proj_vector(lm, mask) if mask is not None else None
            if not (pose_vec or seg_vec or proj_vec):
                continue
            # Quality: more visible landmarks + having a proj/seg vector = better.
            vis = be._min_visibility(lm, (11, 12, 23, 24))
            quality = round(float(min(1.0, max(0.0, vis))), 3)
            entry = {"view": be.classify_view(lm), "quality": quality}
            if pose_vec:
                entry["pose"] = pose_vec
            if seg_vec:
                entry["seg"] = seg_vec
            if proj_vec:
                entry["proj"] = proj_vec
            frames.append(entry)
    finally:
        cap.release()
    return {"clip": os.path.basename(path), "ok": True, "frames": frames}


def main():
    paths = sys.argv[1:]
    if not paths:
        print(json.dumps([{"error": "Usage: footage_embed.py <video> [...]"}]))
        sys.exit(1)
    pose, seg = models()
    print(json.dumps([embed_clip(p, pose, seg) for p in paths]))


if __name__ == "__main__":
    main()
