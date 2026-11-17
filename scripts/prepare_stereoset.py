#!/usr/bin/env python3
"""Sample the official StereoSet intrasentence dev split."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/moinnadeem/StereoSet/ead7d086a64a192a1eca88e0dd2fd163de375218/data/dev.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=4)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    for attempt in range(5):
        try:
            with urlopen(URL, timeout=60) as response:
                data = json.load(response)
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    rows = []
    for item in data["data"]["intrasentence"]:
        choices = {s["gold_label"]: s["sentence"] for s in item["sentences"]}
        if not all(k in choices for k in ("stereotype", "anti-stereotype", "unrelated")):
            continue
        rows.append({"id": item["id"], "bias_type": item["bias_type"],
                     "context": item["context"], "target": item["target"],
                     "stereotype": choices["stereotype"], "anti_stereotype": choices["anti-stereotype"],
                     "unrelated": choices["unrelated"]})
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
