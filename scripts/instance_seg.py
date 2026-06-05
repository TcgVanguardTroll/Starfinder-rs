#!/usr/bin/env python3
"""#10 Phase 1: isolate ONE performer's body from a partnered frame, so the
existing silhouette measurements (build_proj_vector / build_bust_vector) read HER
shape instead of a mask merged with her partner. Runs under Python 3.13
(ultralytics+torch+mediapipe+insightface — NOT the 3.14 sidecar env, which has no
torch).

isolate_her(frame, seed) ->
  (her_mask_bool[H,W], her_pose_landmarks_in_frame_coords)  or  None

Pipeline:
  1. YOLOv8-seg -> per-PERSON instance masks (each person separate, even touching).
  2. Pick HER: the instance whose box holds a face matching her seed (InsightFace);
     fallback to the largest person when no face is visible (rear/doggy frames).
  3. MediaPipe pose on her box crop -> her landmarks (mapped back to frame coords),
     so the measurement uses HER skeleton, not the partner's.

Standalone test:  instance_seg.py <image> [seed_name]   (saves *_her.png overlay)
"""
import os
import sys

import cv2
import numpy as np

# Kept out of the repo; falls back to the bare name (auto-download) if missing.
_YOLO_LOCAL = os.path.join(os.environ.get("LOCALAPPDATA", ""), "yolo-seg", "yolov8s-seg.pt")
YOLO_MODEL = _YOLO_LOCAL if os.path.exists(_YOLO_LOCAL) else "yolov8s-seg.pt"
MIN_DET_SCORE = 0.5
ID_THRESHOLD = 0.3


def load_models():
    from ultralytics import YOLO
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    yolo = YOLO(YOLO_MODEL)
    # MediaPipe pose (image mode) — loaded once; body_embed caches the model file.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import body_embed as be
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=be._cached(be.POSE_MODEL_URL)),
            running_mode=vision.RunningMode.IMAGE, num_poses=1))
    face = None
    try:
        from insightface.app import FaceAnalysis
        _r = sys.stdout
        sys.stdout = sys.stderr
        try:
            face = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            face.prepare(ctx_id=0, det_size=(640, 640))
        finally:
            sys.stdout = _r
    except Exception:  # noqa: BLE001
        face = None
    return yolo, pose, face


def person_instances(yolo, frame):
    """List of (bool_mask[H,W], (x1,y1,x2,y2), conf) for each detected person."""
    res = yolo.predict(frame, classes=[0], conf=0.40, verbose=False)[0]
    if res.masks is None:
        return []
    h, w = frame.shape[:2]
    out = []
    masks = res.masks.data.cpu().numpy()          # [N, mh, mw], 0..1
    boxes = res.boxes.xyxy.cpu().numpy()          # [N, 4]
    confs = res.boxes.conf.cpu().numpy()          # [N]
    for m, box, c in zip(masks, boxes, confs):
        full = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
        out.append((full, box.astype(int), float(c)))
    return out


def _norm(v):
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / n if n else a


def pick_her(frame, persons, seed, face_app):
    """Index of the performer's instance. Prefer the one whose box holds a
    seed-matching face; else the largest person (her face may be hidden)."""
    if not persons:
        return None
    if seed is not None and face_app is not None:
        seed = _norm(seed)
        best_i, best_sim = None, ID_THRESHOLD
        for i, (_, (x1, y1, x2, y2), _) in enumerate(persons):
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            faces = face_app.get(crop)
            if not faces:
                continue
            f = max(faces, key=lambda z: (z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1]))
            if getattr(f, "det_score", 1.0) < MIN_DET_SCORE:
                continue
            sim = float(np.dot(_norm(f.embedding), seed))
            if sim > best_sim:
                best_i, best_sim = i, sim
        if best_i is not None:
            return best_i
    # Fallback: largest person by mask area.
    return max(range(len(persons)), key=lambda i: int(persons[i][0].sum()))


def her_pose(pose, frame, box):
    """MediaPipe pose on her box crop; landmarks remapped to whole-frame [0,1]."""
    import mediapipe as mp
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    res = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks[0]
    cw, ch = (x2 - x1), (y2 - y1)
    # Remap each normalized-in-crop landmark to normalized-in-frame.
    for p in lm:
        p.x = (x1 + p.x * cw) / w
        p.y = (y1 + p.y * ch) / h
    return lm


def isolate_her(frame, seed, models):
    yolo, pose, face_app = models
    persons = person_instances(yolo, frame)
    idx = pick_her(frame, persons, seed, face_app)
    if idx is None:
        return None
    her_mask, box, _ = persons[idx]
    lm = her_pose(pose, frame, box)
    if lm is None:
        return None
    return her_mask, lm


def main():
    img = cv2.imread(sys.argv[1])
    if img is None:
        print("could not read", sys.argv[1])
        return
    models = load_models()
    res = isolate_her(img, None, models)
    if res is None:
        print("no person isolated")
        return
    mask, lm = res
    n_people = len(person_instances(models[0], img))
    overlay = img.copy()
    overlay[mask] = (0.5 * overlay[mask] + np.array([0, 180, 0])).astype(np.uint8)
    out = os.path.splitext(sys.argv[1])[0] + "_her.png"
    cv2.imwrite(out, overlay)
    print(f"people detected: {n_people}; her mask px: {int(mask.sum())}; overlay -> {out}")


if __name__ == "__main__":
    main()
