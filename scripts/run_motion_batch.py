#!/usr/bin/env python3
"""Batch motion-descriptor extraction over performer subfolders.

Walks <root>'s immediate subfolders (one performer each), samples up to --per
clips from each, runs motion_embed.analyse on them, and writes a grouped dataset
to scripts/motion_dataset.json:

  {performer: {"stats": {meas,height,weight} | null, "clips": [<descriptor>...]}}

Stats are pulled from body_index where present (the pairing #22 needs). Pixels
never leave the sidecar; only the numeric descriptors are stored. CPU-bound and
single-threaded per clip — run in the background; it will share cores with the
ingest but writes no DB rows of its own.
"""
import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import motion_embed as me  # noqa: E402

VID = {".mp4", ".ts", ".mov", ".avi", ".mkv", ".wmv", ".m4v", ".webm"}
DB = r"C:\Users\TCGVANGUARDTROLL\AppData\Local\luminary\luminary.db"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motion_dataset.json")


def stats_lookup():
    out = {}
    try:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        for name, data in con.execute("select name, data from body_index"):
            try:
                p = json.loads(data)
            except Exception:  # noqa: BLE001
                continue
            out[name.lower()] = {
                "measurements": p.get("measurements"),
                "height": p.get("height"),
                "weight": p.get("weight"),
            }
        con.close()
    except Exception as e:  # noqa: BLE001
        print(f"stats lookup failed: {e}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\Gooniverse\Milfs")
    ap.add_argument("--per", type=int, default=8, help="clips sampled per performer")
    ap.add_argument("--only", default=None, help="restrict to one subfolder (validation)")
    args = ap.parse_args()

    stats = stats_lookup()
    landmarker = me.make_landmarker()
    subs = sorted(
        d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))
    )
    if args.only:
        subs = [d for d in subs if d.lower() == args.only.lower()]
    dataset = {}
    for d in subs:
        folder = os.path.join(args.root, d)
        clips = sorted(
            f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in VID
        )[: args.per]
        descs = []
        for f in clips:
            try:
                r = me.analyse(os.path.join(folder, f), landmarker)
            except Exception as e:  # noqa: BLE001
                r = {"path": f, "ok": False, "error": repr(e)}
            descs.append(r)
            ok = "ok" if r.get("ok") else f"skip ({r.get('error', '')[:40]})"
            print(f"  {d} / {f[:48]} -> {ok}", flush=True)
        dataset[d] = {"stats": stats.get(d.lower()), "clips": descs}
        json.dump(dataset, open(OUT, "w", encoding="utf-8"), indent=1)
        good = sum(1 for c in descs if c.get("ok"))
        print(f"[{d}] {good}/{len(descs)} usable  (stats: "
              f"{'yes' if stats.get(d.lower()) else 'NO'})", flush=True)
    print(f"\nwrote {OUT}: {len(dataset)} performers", flush=True)


if __name__ == "__main__":
    main()
