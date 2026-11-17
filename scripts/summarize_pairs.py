#!/usr/bin/env python3
"""Summarize preliminary CrowS-Pairs and StereoSet preference results."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    print("file\ttype\tn\tstereotype_preference\tmeaningful_preference")
    for path in args.files:
        rows = [json.loads(s) for s in path.read_bytes().splitlines() if s]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"duplicate results in {path}")
        groups = defaultdict(list)
        for row in rows:
            groups[row["bias_type"]].append(row)
        for key, group in sorted(groups.items()):
            if "prefers_stereotype" in group[0]:
                stereotype = sum(r["prefers_stereotype"] for r in group) / len(group)
                meaningful = sum(sum(r["prefers_meaningful"]) for r in group) / (2 * len(group))
            else:
                stereotype = sum(r["prefers_more"] if r["stereo_antistereo"] == "stereo" else not r["prefers_more"] for r in group) / len(group)
                meaningful = float("nan")
            print(f"{path}\t{key}\t{len(group)}\t{stereotype:.4f}\t{meaningful:.4f}")


if __name__ == "__main__":
    main()
