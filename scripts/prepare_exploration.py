#!/usr/bin/env python3
"""Create fixed, balanced subsets for generation interventions."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def select(source, key, per_group, output, skip_per_group=0):
    groups = defaultdict(list)
    for line in source.read_bytes().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if not row.get("prompt", row.get("context", "")).strip():
            continue
        groups[key(row)].append(row)
    chosen = []
    for group, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda r: hashlib.sha256(f"explore-v1:{r['id']}".encode()).digest())
        if len(ordered) < skip_per_group + per_group:
            raise ValueError(f"insufficient rows in {group}")
        chosen.extend(ordered[skip_per_group:skip_per_group + per_group])
    chosen.sort(key=lambda r: r["id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in chosen)
    output.write_text(payload)
    print(output, len(chosen), hashlib.sha256(payload.encode()).hexdigest())


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data/processed"
    select(data / "bold_full.jsonl", lambda r: (r["domain"],), 12,
           data / "bold_explore_60.jsonl")
    select(data / "bold_full.jsonl", lambda r: (r["domain"],), 120,
           data / "bold_replicate_600.jsonl", skip_per_group=12)
    select(data / "bbq_full.jsonl",
           lambda r: (r["category"], r["context_condition"], r["question_polarity"]), 3,
           data / "bbq_explore_132.jsonl")
    select(data / "bbq_full.jsonl",
           lambda r: (r["category"], r["context_condition"], r["question_polarity"]), 30,
           data / "bbq_replicate_1320.jsonl", skip_per_group=3)


if __name__ == "__main__":
    main()
