#!/usr/bin/env python3
"""Fold scripts/backfill_stats.json into body_index.data (height/weight).

Row-by-row autocommit with a 30s busy timeout and a retry-on-locked loop, so it
interleaves safely with a concurrent ingest writer (SQLite serialises writers;
each update here is tiny). Only fills fields that are currently empty.
"""
import json
import os
import sqlite3
import time

DB = r"C:\Users\TCGVANGUARDTROLL\AppData\Local\luminary\luminary.db"
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backfill_stats.json")


def with_retry(fn, *a, tries=8):
    for i in range(tries):
        try:
            return fn(*a)
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                time.sleep(0.5 * (i + 1))
                continue
            raise
    raise RuntimeError("gave up after retries (db stayed locked)")


def main():
    enrich = json.load(open(SRC, encoding="utf-8"))
    con = sqlite3.connect(DB, timeout=30)
    con.execute("PRAGMA busy_timeout=30000")
    cur = con.cursor()
    h_set = w_set = skipped = 0
    for name, fields in enrich.items():
        row = with_retry(
            lambda n=name: cur.execute(
                "select data from body_index where name=?", (n,)
            ).fetchone()
        )
        if not row:
            continue
        p = json.loads(row[0])
        changed = False
        if fields.get("height") and not p.get("height"):
            p["height"] = fields["height"]
            h_set += 1
            changed = True
        if fields.get("weight") and not p.get("weight"):
            p["weight"] = fields["weight"]
            w_set += 1
            changed = True
        if changed:
            with_retry(
                lambda n=name, d=json.dumps(p): (
                    cur.execute("update body_index set data=? where name=?", (d, n)),
                    con.commit(),
                )
            )
        else:
            skipped += 1
    con.close()
    print(f"applied: height+{h_set}, weight+{w_set}, skipped {skipped}")


if __name__ == "__main__":
    main()
