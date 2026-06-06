#!/usr/bin/env python3
"""#10 Phase 2: re-process footage with instance segmentation so the body
measurements read HER isolated silhouette — unlocking the PARTNERED frames
(cowgirl/missionary/doggystyle/etc.) that apply_footage.py had to discard because
a co-star's face/body was present.

Per sampled frame: instance_seg.isolate_her() -> her clean YOLO mask + her pose
landmarks (already identity-gated via her seed face) -> run build_vector /
build_seg_vector / build_proj_vector / build_bust_vector on HER mask. Pose-route
is implicit: the proj/bust builders only fire on an upright profile, seg/pose on
a full body, so leaning/lying frames self-reject. Inserts as source='footageseg'
(distinct url scheme, so it ADDS to the apply_footage corpus, no PK clash).

Runs under the Python 3.13 env (ultralytics+torch+mediapipe+insightface).
DRY by default (prints per-clip yield); --write inserts. Then `luminary aggregate`.

Usage: seg_footage.py [--write] [--every S] [--max N] [performer ...]
"""
import argparse
import json
import os
import struct
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                    # scripts/ (instance_seg, refine_frames)
sys.path.insert(0, os.path.dirname(_HERE))   # repo root (body_embed)
import instance_seg as iseg  # isolate_her, load_models, person_instances
import body_embed as be  # noqa: E402
import refine_frames as rf  # sample_times, DB, ROOT  # noqa: E402

SEEDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds.json")
EVERY_S, MIN_SAMPLES, MAX_SAMPLES = 10.0, 24, 400   # seg is ~3x heavier than apply_footage
VIDEO_EXTS = (".mp4", ".ts", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv")


def blob(v):
    return struct.pack(f"<{len(v)}f", *v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--every", type=float, default=EVERY_S)
    ap.add_argument("--max", type=int, default=MAX_SAMPLES)
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()

    seeds = json.load(open(SEEDS, encoding="utf-8"))
    names = [n for n in (args.names or sorted(seeds)) if n in seeds]
    if not names:
        print("no seeded performers selected")
        return
    models = iseg.load_models()  # YOLO + pose + insightface
    import sqlite3
    con = sqlite3.connect(rf.DB, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")

    grand = 0
    for name in names:
        seed = np.asarray(seeds[name], dtype=np.float32)
        folder = os.path.join(rf.ROOT, name)
        if not os.path.isdir(folder):
            print(f"! {name}: no folder")
            continue
        seen = {r[0] for r in con.execute(
            "SELECT url FROM images WHERE performer=? AND source='footageseg'", (name,))}
        clips = sorted(f for f in os.listdir(folder)
                       if not f.startswith("._") and f.lower().endswith(VIDEO_EXTS))
        kept, views = 0, {}
        print(f"\n=== {name} ({len(clips)} clips) ===", flush=True)
        for clip in clips:
            path = os.path.join(folder, clip)
            if len(path) > 255:
                path = "\\\\?\\" + os.path.abspath(path)
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            if not (fps > 0 and nfr > 0):
                cap.release()
                continue
            dur = nfr / fps
            n = max(MIN_SAMPLES, min(args.max, int(dur // args.every)))
            ck = 0
            for k in range(n):
                cap.set(cv2.CAP_PROP_POS_MSEC, dur * (k + 0.5) / n * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    continue
                url = f"footageseg://{clip}#{k}"
                if url in seen:
                    continue
                res = iseg.isolate_her(frame, seed, models)
                if res is None:
                    continue
                mask, lm = res
                m = mask.astype(np.uint8)
                pose_vec = be.build_vector(lm)
                seg_vec = be.build_seg_vector(lm, m)
                proj_vec = be.build_proj_vector(lm, m)
                bust_vec = be.build_bust_vector(lm, m)
                if not (pose_vec or seg_vec or proj_vec or bust_vec):
                    continue
                view = be.classify_view(lm)
                q = round(float(min(1.0, max(0.0, be._min_visibility(lm, (11, 12, 23, 24))))), 3)
                if args.write:
                    con.execute(
                        "INSERT OR IGNORE INTO images "
                        "(performer,url,source,view,quality,pose_vec,seg_vec,face_vec,proj_vec,bust_vec) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (name, url, "footageseg", view, q,
                         blob(pose_vec), blob(seg_vec), None, blob(proj_vec), blob(bust_vec)))
                ck += 1
                views[view] = views.get(view, 0) + 1
            cap.release()
            if args.write:
                con.commit()
            if ck:
                print(f"  + {clip[:54]:54} -> {ck} frames", flush=True)
            kept += ck
        grand += kept
        print(f"  {name}: {kept} frames, views={views}", flush=True)
    con.close()
    print(f"\n{'WROTE' if args.write else 'DRY-RUN'}: {grand} seg-isolated footage frames.")
    if args.write:
        print("Now run `luminary aggregate`.")


if __name__ == "__main__":
    main()
