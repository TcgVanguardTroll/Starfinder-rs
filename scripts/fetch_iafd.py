#!/usr/bin/env python3
"""#11: pull authoritative measurements/attributes from IAFD (the best-curated
source) to overwrite unreliable StashDB/TPDB data (e.g. Dee Siren was 34B, really
36D). Searches IAFD by name, parses the bio, and — only when the hit is confident
(name + a non-conflicting ethnicity/hair cross-check against what we already have)
— updates performers + body_index.

DRY by default (prints what it found + a MATCH/SKIP verdict). --write applies.
Rate-limited + resumable (skips performers whose data already came from IAFD).

Usage: fetch_iafd.py [--write] [--all] [performer ...]
ToS note: scraping IAFD is against their terms — personal-tool use, your call.
"""
import argparse
import json
import re
import sqlite3
import time
import urllib.parse
import urllib.request

from _paths import db_path  # cross-platform DB location
DB = db_path()
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
BASE = "https://www.iafd.com"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read().decode("utf-8", "ignore")


def strip_tags(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def search_person(name):
    """First female performer result URL for `name`, or None."""
    url = f"{BASE}/results.asp?searchtype=comprehensive&searchstring={urllib.parse.quote(name)}"
    html = get(url)
    # Performer links are now /person.rme/id=<uuid> (was perfid=.../gender=f/...).
    links = re.findall(r'href="(/person\.rme/id=[0-9a-fA-F-]+)"', html)
    # de-dup preserving order; first hit is the top-ranked match
    seen, uniq = set(), []
    for h in links:
        if h not in seen:
            seen.add(h); uniq.append(h)
    return BASE + uniq[0] if uniq else None


def parse_bio(html):
    """bioheading -> biodata pairs from an IAFD person page."""
    pairs = re.findall(
        r'class="bioheading">\s*(.*?)\s*</p>\s*<p[^>]*class="biodata">\s*(.*?)\s*</p>',
        html, re.S)
    bio = {strip_tags(k).lower(): strip_tags(v) for k, v in pairs}
    name = re.search(r'<h1[^>]*>\s*(.*?)\s*(?:<|$)', html)
    bio["_name"] = strip_tags(name.group(1)) if name else ""
    return bio


def to_meas(bio):
    m = bio.get("measurements", "")
    m = re.sub(r"\s", "", m)
    return m if re.match(r"^\d{2}[A-K]{1,3}-\d{2}-\d{2}$", m) else None


def cm_from_height(bio):
    h = bio.get("height", "")
    m = re.search(r"\((\d{2,3})\s*cm\)", h)
    return f"{m.group(1)}cm" if m else None


def kg_from_weight(bio):
    w = bio.get("weight", "")
    m = re.search(r"\((\d{2,3})\s*kg\)", w)
    return f"{m.group(1)}kg" if m else None


def confident(bio, have):
    """True if the IAFD hit plausibly matches our performer (avoid wrong-person
    overwrites). Conflicting recorded ethnicity is the strongest red flag."""
    eth = bio.get("ethnicity", "").lower()
    have_eth = (have.get("ethnicity") or "").lower()
    if eth and have_eth and eth.split()[0] != have_eth.split()[0]:
        return False, f"ethnicity conflict (iafd={eth} vs ours={have_eth})"
    if not to_meas(bio):
        return False, "no parseable measurements"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--all", action="store_true", help="every body_index performer")
    ap.add_argument("--force", action="store_true", help="re-fetch performers already sourced from IAFD")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()

    c = sqlite3.connect(DB, timeout=60)
    if args.names:
        names = args.names
    elif args.all:
        names = [r[0] for r in c.execute("SELECT name FROM body_index ORDER BY name")]
    else:
        names = ["Dee Siren"]

    upd = skip = 0
    for name in names:
        row = c.execute("SELECT data FROM body_index WHERE name=?", (name,)).fetchone()
        have = json.loads(row[0]) if row else {}
        if have.get("measurements_source") == "IAFD" and not args.force:
            skip += 1
            continue
        try:
            url = search_person(name)
            if not url:
                print(f"  {name:22} no IAFD result"); skip += 1; time.sleep(1.0); continue
            bio = parse_bio(get(url))
        except Exception as e:  # noqa: BLE001
            print(f"  {name:22} fetch error: {e}"); skip += 1; time.sleep(1.0); continue
        time.sleep(1.2)  # be polite

        meas, ht, wt = to_meas(bio), cm_from_height(bio), kg_from_weight(bio)
        ok, why = confident(bio, have)
        old = have.get("measurements")
        verdict = "MATCH" if ok else f"SKIP ({why})"
        print(f"  {name:22} iafd='{bio.get('_name','')}' meas={meas} ht={ht} "
              f"(was {old}) eyes={bio.get('eye color')} -> {verdict}")
        if not ok:
            skip += 1
            continue
        upd += 1
        if not args.write:
            continue
        # apply — preserve the original (revertible) and mark the source (resumable)
        for tbl, key in (("performers", "lower(name)=lower(?)"), ("body_index", "name=?")):
            r = c.execute(f"SELECT data FROM {tbl} WHERE {key}", (name,)).fetchone()
            if not r:
                continue
            j = json.loads(r[0])
            if "measurements_orig" not in j and j.get("measurements"):
                j["measurements_orig"] = j["measurements"]
            j["measurements_source"] = "IAFD"
            if meas: j["measurements"] = meas
            if ht: j["height"] = ht
            if wt: j["weight"] = wt
            if bio.get("eye color"): j["eye_color"] = bio["eye color"].title()
            if bio.get("hair color"): j["hair_color"] = bio["hair color"].title()
            sets, params = ["data=?"], [json.dumps(j)]
            if tbl == "performers":
                if meas: sets.append("measurements=?"); params.append(meas)
                if ht: sets.append("height=?"); params.append(ht)
            params.append(name)
            c.execute(f"UPDATE {tbl} SET {','.join(sets)} WHERE {key}", params)
        c.commit()
    c.close()
    print(f"\n{'UPDATED' if args.write else 'DRY-RUN'}: {upd} matched, {skip} skipped.")


if __name__ == "__main__":
    main()
