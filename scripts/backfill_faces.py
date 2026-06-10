#!/usr/bin/env python3
"""Backfill face embeddings for the body_index ROSTER (not just the 35-performer
library that `luminary embed` covers). For each roster performer missing a face,
embed their StashDB profile (InsightFace ArcFace) and store it in the `candidates`
table — which is where body-search's face modality reads (get_embedding_any).
After this, the `overall` face+body blend can score the whole 1008, not just ~100.

Resumable (skips performers already embedded). DRY by default; --write stores.
Run under C:\\Python314 (insightface). Usage: backfill_faces.py [--write] [limit]
"""
import json
import os
import sys
import tempfile
import urllib.request

import cv2

from _paths import db_path  # cross-platform DB location
DB = db_path()
UA = "Mozilla/5.0 (Luminary)"
MIN_DET = 0.5


def download(url):
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def main():
    import sqlite3
    args = sys.argv[1:]
    write = "--write" in args
    limit = next((int(a) for a in args if a.isdigit()), None)

    from insightface.app import FaceAnalysis
    _r = sys.stdout
    sys.stdout = sys.stderr
    try:
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
    finally:
        sys.stdout = _r

    c = sqlite3.connect(DB, timeout=60)
    have = {n.lower() for (n,) in c.execute("SELECT name FROM candidates")}
    have |= {n.lower() for (n,) in c.execute("SELECT name FROM performers WHERE embedding IS NOT NULL")}
    todo = [(n, d) for (n, d) in c.execute("SELECT name, data FROM body_index")
            if n.lower() not in have]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} roster performers missing a face embedding")

    ok = miss = 0
    for i, (name, data) in enumerate(todo, 1):
        j = json.loads(data)
        urls = [u for u in ([j.get("face_url"), j.get("profile_image_url")]
                            + (j.get("gallery_urls") or [])[:2]) if u]
        emb = None
        for u in urls:
            tmp = None
            try:
                tmp = download(u)
                img = cv2.imread(tmp)
                if img is None:
                    continue
                faces = app.get(img)
                if not faces:
                    continue
                f = max(faces, key=lambda z: (z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1]))
                if getattr(f, "det_score", 1.0) >= MIN_DET:
                    emb = f.embedding
                    break
            except Exception:  # noqa: BLE001
                pass
            finally:
                if tmp and os.path.exists(tmp):
                    os.unlink(tmp)
        if emb is None:
            miss += 1
        else:
            ok += 1
            if write:
                c.execute("INSERT OR REPLACE INTO candidates (name, data, embedding) VALUES (?,?,?)",
                          (name, data, json.dumps([float(x) for x in emb])))
                if i % 25 == 0:
                    c.commit()
        if i % 50 == 0:
            print(f"  ...{i}/{len(todo)} ({ok} embedded, {miss} no-face)", flush=True)
    if write:
        c.commit()
    c.close()
    print(f"\n{'WROTE' if write else 'DRY-RUN'}: {ok} embedded, {miss} no-face.")


if __name__ == "__main__":
    main()
