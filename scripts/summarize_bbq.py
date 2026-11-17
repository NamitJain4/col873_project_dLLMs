#!/usr/bin/env python3
"""Report BBQ accuracy and invalid responses by context condition."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    print("file\tcondition\tn\tvalid\taccuracy\tinvalid_rate")
    for path in args.files:
        rows = [json.loads(s) for s in path.read_bytes().splitlines() if s]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"duplicate results in {path}")
        groups = defaultdict(list)
        for row in rows:
            groups[row["context_condition"]].append(row)
        for condition, group in sorted(groups.items()):
            n = len(group)
            valid = sum(r["prediction"] is not None for r in group)
            correct = sum(r["correct"] for r in group)
            print(f"{path}\t{condition}\t{n}\t{valid}\t{correct/n:.4f}\t{(n-valid)/n:.4f}")


if __name__ == "__main__":
    main()
