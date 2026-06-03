#!/usr/bin/env python3
"""Backfill height (+ weight) onto the cached body_index roster.

The roster was bulk-loaded from StashDB, whose search path omits stature/mass, so
every candidate has null height/weight while the `overall` blend weights them
(height x2, weight x1.5 in recommender::feature_vector). This pass fills them:

  * height  -- StashDB findPerformer(<uuid>) by the UUID already in source_url
              (exact, one call each). StashDB returns height in cm.
  * weight  -- TPDB /performers?q=<name> -> /performers/<id> extras.weight,
              accepted ONLY on an exact (normalised) name match (StashDB has no
              weight field). Conservative: a non-match leaves weight null/neutral
              rather than risk a wrong-person value.

Network-only: writes results to scripts/backfill_stats.json. A separate apply
step (apply_backfill_stats.py) folds them into body_index.data, so the DB is not
touched while the long ingest is still writing to it.
"""
import json
import os
import re
import sqlite3
import ssl
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DB = r"C:\Users\TCGVANGUARDTROLL\AppData\Local\luminary\luminary.db"
CFG = r"C:\Users\TCGVANGUARDTROLL\AppData\Local\luminary\config.json"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backfill_stats.json")

STASHDB = "https://stashdb.org/graphql"
TPDB = "https://api.theporndb.net"
CTX = ssl.create_default_context()

cfg = json.load(open(CFG, encoding="utf-8"))
STASH_KEY = cfg["stashdb_key"]
TPDB_KEY = cfg["api_key"]

UUID_RE = re.compile(r"stashdb\.org/performers/([0-9a-fA-F-]{36})")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _req(url, *, data=None, headers=None, method="GET"):
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("User-Agent", "Luminary/0.1.0")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - best-effort, retry then give up
            last = e
    raise last


def stash_height(uuid):
    q = ("query($id: ID!){ findPerformer(id:$id){ name height "
         "cup_size band_size waist_size hip_size } }")
    body = json.dumps({"query": q, "variables": {"id": uuid}}).encode()
    r = _req(STASHDB, data=body, method="POST",
             headers={"Content-Type": "application/json", "ApiKey": STASH_KEY})
    p = (r.get("data") or {}).get("findPerformer")
    if not p:
        return None
    h = p.get("height")
    return f"{int(h)}cm" if h and 90 < int(h) < 230 else None


def tpdb_weight(name):
    """Top exact-name match's weight (raw TPDB string), else None."""
    su = f"{TPDB}/performers?q={urllib.parse.quote(name)}"
    sr = _req(su, headers={"Authorization": f"Bearer {TPDB_KEY}"})
    target = norm(name)
    for item in (sr.get("data") or [])[:5]:
        if norm(item.get("name")) != target:
            continue
        det = _req(f"{TPDB}/performers/{item['id']}",
                   headers={"Authorization": f"Bearer {TPDB_KEY}"})
        extras = (det.get("data") or {}).get("extras") or {}
        w = extras.get("weight")
        if w and any(ch.isdigit() for ch in str(w)):
            return str(w), item.get("name")
        return None
    return None


def worklist():
    con = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    rows = con.execute("select name, data from body_index").fetchall()
    con.close()
    out = []
    for name, data in rows:
        try:
            p = json.loads(data)
        except Exception:  # noqa: BLE001
            continue
        m = UUID_RE.search(p.get("source_url") or "")
        out.append((name, m.group(1) if m else None,
                    bool(p.get("height")), bool(p.get("weight"))))
    return out


def main():
    items = worklist()
    print(f"{len(items)} roster entries", flush=True)
    results = {}

    # ---- height via StashDB (by UUID) ----
    def fetch_h(it):
        name, uuid, has_h, _ = it
        if has_h or not uuid:
            return name, None
        try:
            return name, stash_height(uuid)
        except Exception as e:  # noqa: BLE001
            print(f"  [h] {name}: {e}", flush=True)
            return name, None

    done = h_ok = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed(ex.submit(fetch_h, it) for it in items):
            name, h = fut.result()
            done += 1
            if h:
                results.setdefault(name, {})["height"] = h
                h_ok += 1
            if done % 100 == 0:
                print(f"  height {done}/{len(items)} ({h_ok} filled)", flush=True)
                json.dump(results, open(OUT, "w", encoding="utf-8"))
    print(f"height done: {h_ok} filled", flush=True)

    # ---- weight via TPDB (by exact-name match) ----
    def fetch_w(it):
        name, _, _, has_w = it
        if has_w:
            return name, None
        try:
            return name, tpdb_weight(name)
        except Exception as e:  # noqa: BLE001
            print(f"  [w] {name}: {e}", flush=True)
            return name, None

    done = w_ok = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for fut in as_completed(ex.submit(fetch_w, it) for it in items):
            name, w = fut.result()
            done += 1
            if w:
                results.setdefault(name, {})["weight"] = w[0]
                w_ok += 1
            if done % 100 == 0:
                print(f"  weight {done}/{len(items)} ({w_ok} filled)", flush=True)
                json.dump(results, open(OUT, "w", encoding="utf-8"))
    print(f"weight done: {w_ok} filled", flush=True)

    json.dump(results, open(OUT, "w", encoding="utf-8"))
    print(f"wrote {OUT}: {len(results)} performers enriched", flush=True)


if __name__ == "__main__":
    sys.exit(main())
