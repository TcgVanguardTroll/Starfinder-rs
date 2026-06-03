#!/usr/bin/env python3
"""Fetch stats (height/measurements/weight) for the Milfs-folder performers that
lack them in body_index, so their footage becomes usable for the stats->motion
model (#22). Name-based: StashDB searchPerformers -> height + cup/band/waist/hip;
TPDB search -> detail -> weight. Network-only -> scripts/milfs_stats.json
(joined with motion_dataset.json downstream; no DB write).
"""
import json
import os
import re
import ssl
import time
import urllib.parse
import urllib.request

CFG = r"C:\Users\TCGVANGUARDTROLL\AppData\Local\luminary\config.json"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "milfs_stats.json")
cfg = json.load(open(CFG, encoding="utf-8"))
STASH_KEY, TPDB_KEY = cfg["stashdb_key"], cfg["api_key"]
TPDB = "https://api.theporndb.net"
CTX = ssl.create_default_context()

NAMES = [
    "Dee Siren", "Naughty Alysha", "Montse Swinger", "Seka Black",
    "Karla Insatiable", "Monique Fuentes", "Merce", "Victoria Lobov",
    "Persia Monir", "Christina Sapphire", "Lisa Ann", "Ava Addams",
]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _req(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("User-Agent", "Luminary/0.1.0")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    for delay in (0, 1, 2, 4):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            continue
    return None


def stash_stats(name):
    q = ("query($t:String!){ searchPerformers(term:$t, limit:3){ performers{"
         " name height cup_size band_size waist_size hip_size } } }")
    body = json.dumps({"query": q, "variables": {"t": name}}).encode()
    r = _req("https://stashdb.org/graphql", data=body,
             headers={"Content-Type": "application/json", "ApiKey": STASH_KEY})
    perfs = (((r or {}).get("data") or {}).get("searchPerformers") or {}).get("performers") or []
    tgt = norm(name)
    p = next((x for x in perfs if norm(x.get("name")) == tgt), perfs[0] if perfs else None)
    if not p:
        return {}
    out = {}
    h = p.get("height")
    if h and 90 < int(h) < 230:
        out["height"] = f"{int(h)}cm"
    band, cup, waist, hip = (p.get("band_size"), p.get("cup_size"),
                             p.get("waist_size"), p.get("hip_size"))
    if band and cup and waist and hip:
        out["measurements"] = f"{band}{cup}-{waist}-{hip}"
    return out


def tpdb_weight(name):
    sr = _req(f"{TPDB}/performers?q={urllib.parse.quote(name)}",
              headers={"Authorization": f"Bearer {TPDB_KEY}"})
    tgt = norm(name)
    for item in ((sr or {}).get("data") or [])[:5]:
        if norm(item.get("name")) != tgt:
            continue
        det = _req(f"{TPDB}/performers/{item['id']}",
                   headers={"Authorization": f"Bearer {TPDB_KEY}"})
        w = (((det or {}).get("data") or {}).get("extras") or {}).get("weight")
        return str(w) if w and any(c.isdigit() for c in str(w)) else None
    return None


def main():
    out = {}
    for name in NAMES:
        rec = stash_stats(name)
        w = tpdb_weight(name)
        if w:
            rec["weight"] = w
        out[name] = rec
        print(f"  {name}: {rec}", flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        time.sleep(0.3)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
