#!/usr/bin/env python3
"""Footage apply step (#24): identity-gate a performer's clips and land the
per-frame body vectors into the `images` corpus — especially the SIDE-view proj
(butt shape) that stills miss. The companion to footage_embed.py: same CV, plus
the face gate + DB insert that footage_embed deliberately leaves out.

Sampling is SEEK-based (cap.set on a timestamp), not grab-every-frame: cost is
proportional to the number of samples, NOT clip length, so a 23-hour, 19 GB
compilation costs the same as a 2-minute clip (it just gets sparser temporal
coverage). ~1 frame / SAMPLE_EVERY_S, clamped to [MIN_SAMPLES, MAX_SAMPLES].

Pipeline, per seeded performer folder under ROOT:
  1. For each clip, seek-sample frames; run pose+seg (body) AND InsightFace (face).
  2. Clip gate: trust a clip only if >= MIN_CLIP_MATCHES of its sampled frames
     hold a face matching the performer's seed (a co-star/compilation can't pass).
  3. Per-frame, within an accepted clip:
       - a frame WITH a face is kept only if that face matches the seed
         (drops a co-star's front shot, or another performer in a compilation);
       - a faceless side/rear frame (the proj goldmine) is gallery-trusted.
  4. Insert kept frames into `images` (source='footage') as raw LE-f32 blobs,
     matching embedder::embedding_to_blob. INSERT OR IGNORE on (performer, url)
     => idempotent + resumable.

Seeds come from scripts/seeds.json (extract_seeds.py). View strings (front/rear/
side) come straight from body_embed.classify_view, which the aggregator keys on.

Usage:  apply_footage.py [--every N] [--max M] [performer name ...]
        (default: every seeded folder; --every secs between samples, --max cap)
Run AFTER any `luminary ingest` finishes — both write luminary.db and SQLite
serialises writers. Re-run `luminary aggregate` afterwards to fold proj in.
"""
import json
import os
import struct
import sys

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sqlite3  # noqa: E402
import body_embed as be  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

from _paths import db_path  # cross-platform DB location
DB = db_path()
ROOT = r"D:\Gooniverse\Milfs"
SEEDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds.json")

SAMPLE_EVERY_S = 8.0      # seconds between samples — dense enough to catch the
                          # brief side/standing windows that are the whole point
MIN_SAMPLES = 24          # floor, so short clips still get enough frames
MAX_SAMPLES = 2000        # per-clip cap — a 23 h file still gets ~1 sample/43 s
                          # (decent coverage), bounded to ~7 min compute
PROC_WIDTH = 640
ID_THRESHOLD = 0.3        # cosine(face, seed) to count as the same person
MIN_DET_SCORE = 0.5       # InsightFace detector confidence floor (match face_embed)
MIN_CLIP_MATCHES = 3      # seed-matching frames needed to trust a whole clip
MIN_QUALITY = 0.5         # drop partial-body frames (shoulders/hips not clearly
                          # visible) — only confident full-body frames feed centroids
VIDEO_EXTS = (".mp4", ".ts", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv")


def blob(vec):
    """Raw little-endian f32 — byte-compatible with embedder::embedding_to_blob."""
    return struct.pack(f"<{len(vec)}f", *vec) if vec else None


def normed(v):
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / n if n else a


def longpath(p):
    """Windows extended-length prefix so >260-char paths (long scene filenames)
    can be opened by cv2/os."""
    if os.name == "nt" and len(p) > 255 and not p.startswith("\\\\?\\"):
        return "\\\\?\\" + os.path.abspath(p)
    return p


def load_pose_seg():
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=be._cached(be.POSE_MODEL_URL)),
            running_mode=vision.RunningMode.IMAGE, num_poses=1))
    seg = vision.ImageSegmenter.create_from_options(
        vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=be._cached(be.SEG_MODEL_URL)),
            output_confidence_masks=True))
    return pose, seg


def load_face():
    from insightface.app import FaceAnalysis
    _real = sys.stdout
    sys.stdout = sys.stderr  # InsightFace prints model chatter to stdout
    try:
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
    finally:
        sys.stdout = _real
    return app


def largest_face(app, bgr):
    """(embedding, det_score) for the biggest confident face, else None."""
    faces = app.get(bgr)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    if getattr(f, "det_score", 1.0) < MIN_DET_SCORE:
        return None
    return f.embedding, float(f.det_score)


def sample_times(cap, every_s, max_samples):
    """Evenly spread sample timestamps (ms) across the clip's duration.
    Returns (times_ms, dur_min). Falls back to None when duration is unknown."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    if fps > 0 and nfr > 0:
        dur = nfr / fps
        n = int(dur // every_s)
        n = max(MIN_SAMPLES, min(max_samples, n))
        return [dur * (k + 0.5) / n * 1000.0 for k in range(n)], dur / 60.0
    return None, 0.0


def process_frame(frame, pose, seg, face_app, seed):
    """One frame -> body entry (or None) plus face verdict {'match','other','none'}."""
    h0, w0 = frame.shape[:2]
    if w0 == 0:
        return None, "none", None
    fr = cv2.resize(frame, (PROC_WIDTH, max(1, int(h0 * PROC_WIDTH / w0))))
    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

    verdict, id_sim = "none", None
    hit = largest_face(face_app, fr)
    if hit is not None:
        id_sim = float(np.dot(normed(hit[0]), seed))
        verdict = "match" if id_sim >= ID_THRESHOLD else "other"

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = pose.detect(image)
    if not res.pose_landmarks:
        return None, verdict, id_sim
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
    bust_vec = be.build_bust_vector(lm, mask) if mask is not None else None
    if not (pose_vec or seg_vec or proj_vec or bust_vec):
        return None, verdict, id_sim
    vis = be._min_visibility(lm, (11, 12, 23, 24))
    quality = round(float(min(1.0, max(0.0, vis))), 3)
    return {"view": be.classify_view(lm), "quality": quality,
            "pose": pose_vec, "seg": seg_vec, "proj": proj_vec, "bust": bust_vec,
            "face": verdict}, verdict, id_sim


def scan_clip(path, pose, seg, face_app, seed, every_s, max_samples):
    """Seek-sample a clip. Returns (frames, n_match, n_other, status, dur_min)."""
    cap = cv2.VideoCapture(longpath(path))
    if not cap.isOpened():
        return [], 0, 0, "open-fail", 0.0
    try:  # release the capture even if a frame raises (handle leak otherwise)
        times, dur_min = sample_times(cap, every_s, max_samples)
        frames, n_match, n_other, sidx = [], 0, 0, 0
        if times is not None:
            # Seek-based: O(samples), independent of clip length.
            for ms in times:
                cap.set(cv2.CAP_PROP_POS_MSEC, ms)
                ok, frame = cap.read()
                if not ok:
                    continue
                sidx += 1
                entry, verdict, _ = process_frame(frame, pose, seg, face_app, seed)
                if verdict == "match":
                    n_match += 1
                elif verdict == "other":
                    n_other += 1
                if entry:
                    entry["idx"] = sidx
                    frames.append(entry)
        else:
            # Unknown duration: read sequentially, every ~SAMPLE_EVERY_S*30 frames.
            stride, i = max(1, int(every_s * 30)), 0
            while len(frames) < max_samples:
                if not cap.grab():
                    break
                i += 1
                if i % stride:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                entry, verdict, _ = process_frame(frame, pose, seg, face_app, seed)
                if verdict == "match":
                    n_match += 1
                elif verdict == "other":
                    n_other += 1
                if entry:
                    entry["idx"] = i
                    frames.append(entry)
    finally:
        cap.release()
    return frames, n_match, n_other, "ok", dur_min


def main():
    args = sys.argv[1:]
    every_s, max_samples, names_in = SAMPLE_EVERY_S, MAX_SAMPLES, []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--every":
            every_s = float(args[i + 1]); i += 2; continue
        if a == "--max":
            max_samples = int(args[i + 1]); i += 2; continue
        names_in.append(a); i += 1

    seeds = json.load(open(SEEDS, encoding="utf-8"))
    names = [n for n in (names_in or sorted(seeds)) if n in seeds]
    for n in [n for n in names_in if n not in seeds]:
        print(f"! {n}: no seed in seeds.json — skipping (run extract_seeds.py)")
    if not names:
        print("nothing to do (no seeded performers selected)")
        return

    pose, seg = load_pose_seg()
    face_app = load_face()
    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")

    grand = 0
    for name in names:
        seed = normed(seeds[name])
        folder = os.path.join(ROOT, name)
        if not os.path.isdir(folder):
            print(f"! {name}: folder not found under {ROOT}")
            continue
        seen = {r[0] for r in con.execute(
            "SELECT url FROM images WHERE performer=?", (name,))}
        clips = sorted(f for f in os.listdir(folder)
                       if not f.startswith("._")
                       and f.lower().endswith(VIDEO_EXTS))
        kept_total, views = 0, {}
        print(f"\n=== {name} ({len(clips)} clip(s)) ===")
        for clip in clips:
            frames, n_match, n_other, status, dur_min = scan_clip(
                os.path.join(folder, clip), pose, seg, face_app, seed,
                every_s, max_samples)
            tag = f"[{dur_min:.0f}m]"
            if status == "open-fail":
                print(f"  ! {clip[:54]:54} {tag} could not open — skipped")
                continue
            if n_match < MIN_CLIP_MATCHES:
                print(f"  - {clip[:54]:54} {tag} rejected "
                      f"(match {n_match} < {MIN_CLIP_MATCHES}, other {n_other})")
                continue
            kept = 0
            for fr in frames:
                if fr["face"] == "other":      # someone else's face in this frame
                    continue
                if fr["quality"] < MIN_QUALITY:  # partial-body / low-confidence
                    continue
                url = f"footage://{clip}#{fr['idx']}"
                if url in seen:
                    continue
                con.execute(
                    "INSERT OR IGNORE INTO images "
                    "(performer,url,source,view,quality,pose_vec,seg_vec,face_vec,proj_vec,bust_vec) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (name, url, "footage", fr["view"], fr["quality"],
                     blob(fr["pose"]), blob(fr["seg"]), None, blob(fr["proj"]),
                     blob(fr.get("bust"))))
                seen.add(url)
                kept += 1
                views[fr["view"]] = views.get(fr["view"], 0) + 1
            con.commit()
            kept_total += kept
            print(f"  + {clip[:54]:54} {tag} accepted "
                  f"(match {n_match}, other {n_other}) -> {kept} frames")
        grand += kept_total
        print(f"  {name}: {kept_total} frames kept, views={views}")
    con.close()
    print(f"\nDONE: {grand} footage frames inserted across {len(names)} performer(s).")
    print("Next: run `luminary aggregate` to fold these (esp. side->proj) into the index.")


if __name__ == "__main__":
    main()
