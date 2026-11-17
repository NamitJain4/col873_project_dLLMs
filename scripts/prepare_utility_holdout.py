#!/usr/bin/env python3
"""Create fictional routine-email holdout records disjoint from canary training."""
import hashlib
import json
import random
from pathlib import Path

from prepare_synthetic_mia import background


def main():
    root = Path("data/processed/synthetic_mia")
    train = [json.loads(s) for s in (root / "train.jsonl").read_bytes().splitlines() if s]
    excluded = {r["text"] for r in train}
    rng = random.Random(2026)
    rows = []
    while len(rows) < 128:
        row = background(len(rows), rng)
        if row["text"] in excluded:
            continue
        excluded.add(row["text"])
        row["id"] = f"utility:{len(rows):04d}"
        rows.append(row)
    output = root / "utility_holdout.jsonl"
    blob = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()
    output.write_bytes(blob)
    print(f"{output}: {len(rows)} rows; sha256={hashlib.sha256(blob).hexdigest()}")


if __name__ == "__main__":
    main()
