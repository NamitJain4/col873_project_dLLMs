#!/usr/bin/env python3
"""Create a deterministic, domain-stratified BOLD generation pilot."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from urllib.request import urlopen

BASE = "https://huggingface.co/datasets/AmazonScience/bold/resolve/be5f5a99b386a7c4fa7ea905685ee2d2c98301eb/"
FILES = ["gender_prompt_wiki.json", "political_ideology_prompt_wiki.json",
         "profession_prompt_wiki.json", "race_prompt_wiki.json",
         "religious_ideology_prompt_wiki.json"]


def get_lines(filename):
    url = BASE + filename
    for attempt in range(5):
        try:
            with urlopen(url, timeout=45) as response:
                return response.read().decode("utf-8").split("\n")
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=8)
    ap.add_argument("--full", action="store_true", help="include every prompt, including multiple prompts per source row")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.per_domain < 1:
        ap.error("--per-domain must be positive")
    rng = random.Random(42)
    pools = {}
    counts = {}
    for filename in FILES:
        lines = get_lines(filename)
        for lineno, line in enumerate(lines):
            row = json.loads(line)
            if not row.get("prompts"):
                continue
            domain = row["domain"]
            pool = pools.setdefault(domain, [])
            prompts = enumerate(row["prompts"]) if args.full else [(0, row["prompts"][0])]
            for prompt_idx, prompt in prompts:
                counts[domain] = counts.get(domain, 0) + 1
                record = {"id": f"{filename}:{lineno}:{prompt_idx}" if args.full else f"{filename}:{lineno}",
                          "source_file": filename, "source_row": lineno,
                          "domain": domain, "category": row["category"], "name": row["name"],
                          "prompt": prompt}
                if args.full:
                    record["prompt_index"] = prompt_idx
                if args.full:
                    pool.append(record)
                elif len(pool) < args.per_domain:
                    pool.append(record)
                else:
                    replacement = rng.randrange(counts[domain])
                    if replacement < args.per_domain:
                        pool[replacement] = record
        print(f"scanned {filename}: {len(lines)} rows", flush=True)
    if not args.full and any(len(pool) < args.per_domain for pool in pools.values()):
        raise RuntimeError(f"BOLD domain too small: {counts}")
    records = sorted((r for pool in pools.values() for r in pool), key=lambda x: x["id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    args.output.write_text(payload)
    print(f"wrote {len(records)} rows; domains={counts}; sha256={hashlib.sha256(payload.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
