#!/usr/bin/env python3
"""#16/bust: populate bust_vec for the EXISTING footage corpus without a full
re-extraction. Bust projection comes only from SIDE/profile silhouettes, so this
re-seeks each stored footage frame, runs pose+seg, and UPDATEs images.bust_vec
where build_bust_vector fires (profile frames). Then `luminary aggregate` rolls
it into body_index.bust_vec and the `bust` modality starts contributing.

DRY by default (counts only); --write performs the UPDATEs.
Usage: backfill_bust.py [--write] [performer ...]
"""
import argparse
import os
import struct
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refine_frames as rf  # sample_times, DB, ROOT
import body_embed as be  # noqa: E402
import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

PROC_WIDTH = 640


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


def blob(vec):
    return struct.pack(f"<{len(vec)}f", *vec) if vec else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()

    import sqlite3
    pose, seg = load_pose_seg()
    con = sqlite3.connect(rf.DB, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")

    where = "source='footage' AND bust_vec IS NULL"
    params = []
    if args.names:
        where += " AND performer IN (%s)" % ",".join("?" * len(args.names))
        params = args.names
    rows = con.execute(
        f"SELECT performer, url, view FROM images WHERE {where}", params).fetchall()
    # Group by (performer, clip); bust needs a profile, so most yield comes from side.
    byclip = {}
    for perf, url, view in rows:
        clip = url[len("footage://"):url.index("#")]
        byclip.setdefault((perf, clip), []).append((int(url[url.index("#") + 1:]), url))

    checked = filled = 0
    for (perf, clip), items in byclip.items():
        path = os.path.join(rf.ROOT, perf, clip)
        if len(path) > 255:
            path = "\\\\?\\" + os.path.abspath(path)
        times, cap = rf.sample_times(path)
        if times is None:
            cap.release()
            continue
        try:  # release the capture even if a frame raises (handles leak per clip otherwise)
            for sidx, url in items:
                cap.set(cv2.CAP_PROP_POS_MSEC, times[min(max(sidx - 1, 0), len(times) - 1)])
                ok, frame = cap.read()
                if not ok:
                    continue
                checked += 1
                h0, w0 = frame.shape[:2]
                fr = cv2.resize(frame, (PROC_WIDTH, max(1, int(h0 * PROC_WIDTH / w0))))
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                res = pose.detect(image)
                if not res.pose_landmarks:
                    continue
                lm = res.pose_landmarks[0]
                try:
                    sres = seg.segment(image)
                    mask = sres.confidence_masks[0].numpy_view() if sres.confidence_masks else None
                except Exception:  # noqa: BLE001
                    mask = None
                bust = be.build_bust_vector(lm, mask) if mask is not None else None
                if bust:
                    filled += 1
                    if args.write:
                        con.execute("UPDATE images SET bust_vec=? WHERE performer=? AND url=?",
                                    (blob(bust), perf, url))
            if args.write:
                con.commit()
        finally:
            cap.release()
    con.close()
    print(f"checked {checked} footage frames; bust extracted on {filled} "
          f"(profiles). {'WROTE bust_vec' if args.write else 'DRY-RUN'}.")
    if args.write:
        print("Now run `luminary aggregate` to fold bust into body_index.")


if __name__ == "__main__":
    main()
