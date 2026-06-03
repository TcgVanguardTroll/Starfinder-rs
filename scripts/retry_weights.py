#!/usr/bin/env python3
"""Retry TPDB weight lookups for roster names still missing a weight.

The bulk pass hit TPDB 429s at the tail. This re-fetches only the gaps, single
-threaded with a delay + exponential backoff on 429, and merges into
backfill_stats.json in place.
"""
import json
import os
import re
import sqlite3
import ssl
import time
import urllib.parse
import urllib.request

DB = r"C:\Users\TCGVANGUARDTROLL\AppData\Local\luminary\luminary.db"
CFG = r"C:\Users\TCGVANGUARDTROLL\AppData\Local\luminary\config.json"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backfill_stats.json")
TPDB = "https://api.theporndb.net"
CTX = ssl.create_default_context()

TPDB_KEY = json.load(open(CFG, encoding="utf-8"))["api_key"]
results = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def get(url):
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Luminary/0.1.0")
    req.add_header("Authorization", f"Bearer {TPDB_KEY}")
    delay = 1.0
    for _ in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(delay)
                delay = min(delay * 2, 16)
                continue
            raise
    return None


def weight_for(name):
    sr = get(f"{TPDB}/performers?q={urllib.parse.quote(name)}")
    if not sr:
        return None
    t = norm(name)
    for item in (sr.get("data") or [])[:5]:
        if norm(item.get("name")) != t:
            continue
        det = get(f"{TPDB}/performers/{item['id']}")
        if not det:
            return None
        w = ((det.get("data") or {}).get("extras") or {}).get("weight")
        return str(w) if w and any(c.isdigit() for c in str(w)) else None
    return None


con = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
names = [r[0] for r in con.execute("select name from body_index").fetchall()]
con.close()

missing = [n for n in names if not results.get(n, {}).get("weight")]
print(f"{len(missing)} names missing weight; retrying...", flush=True)
ok = 0
for i, name in enumerate(missing, 1):
    try:
        w = weight_for(name)
    except Exception as e:  # noqa: BLE001
        print(f"  {name}: {e}", flush=True)
        w = None
    if w:
        results.setdefault(name, {})["weight"] = w
        ok += 1
    time.sleep(0.4)
    if i % 25 == 0:
        print(f"  {i}/{len(missing)} ({ok} recovered)", flush=True)
        json.dump(results, open(OUT, "w", encoding="utf-8"))

json.dump(results, open(OUT, "w", encoding="utf-8"))
print(f"recovered {ok} weights; backfill_stats.json now has "
      f"{sum(1 for v in results.values() if v.get('weight'))} weights, "
      f"{sum(1 for v in results.values() if v.get('height'))} heights", flush=True)
