#!/usr/bin/env python3
"""Download the authors' CrowS-Pairs CSV and sample by bias category."""
import argparse
import csv
import hashlib
import io
import json
import random
import time
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/8aaac11c485473159ec9328a65253a5be9a479dc/data/crows_pairs_anonymized.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=2)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    for attempt in range(5):
        try:
            with urlopen(URL, timeout=45) as response:
                content = response.read().decode("utf-8-sig")
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    rows = []
    for row in csv.DictReader(io.StringIO(content)):
        if not row.get("sent_more") or not row.get("sent_less"):
            continue
        rows.append({"id": f"crows:{row['']}", "sent_more": row["sent_more"],
                     "sent_less": row["sent_less"], "stereo_antistereo": row["stereo_antistereo"],
                     "bias_type": row["bias_type"]})
    if not args.full:
        rng = random.Random(42)
        groups = {}
        for row in rows:
            groups.setdefault(row["bias_type"], []).append(row)
        rows = [r for key, group in sorted(groups.items()) for r in rng.sample(group, min(args.per_type, len(group)))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    args.output.write_text(payload)
    print(f"wrote {len(rows)} rows, types={sorted({r['bias_type'] for r in rows})}, sha256={hashlib.sha256(payload.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
