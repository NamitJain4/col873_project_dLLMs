#!/usr/bin/env python3
"""Require complete, correctly aligned Dream results before interpretation."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    items = [json.loads(line) for line in path.read_bytes().splitlines() if line]
    ids = [r["id"] for r in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate ids in {path}")
    return items


def check(path, source, expected_method, expected_n):
    source_blob = source.read_bytes()
    expected_ids = {r["id"] for r in (json.loads(s) for s in source_blob.splitlines() if s)}
    items = rows(path)
    if (len(items) != expected_n or len(expected_ids) != expected_n or
            {r["id"] for r in items} != expected_ids):
        raise ValueError(f"incomplete result: {path}")
    digest = hashlib.sha256(source_blob).hexdigest()
    if any(r["model"] != "dream" or r["data_sha256"] != digest or
           r["score_method"] != expected_method for r in items):
        raise ValueError(f"provenance or score alignment mismatch: {path}")
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", action="store_true")
    parser.add_argument("--mia", action="store_true")
    args = parser.parse_args()
    if not args.pairs and not args.mia:
        parser.error("select --pairs or --mia")
    out = ROOT / "results/exploration"
    data = ROOT / "data/processed"
    if args.pairs:
        method = "mean_masked_token_logprob_4x_30pct_dream_aligned_v2"
        for dataset, n in (("crows", 1508), ("stereoset", 2106)):
            items = check(out / f"{dataset}_dream_aligned_v2.jsonl",
                          data / f"{dataset}_full.jsonl", method, n)
            for item in items:
                scores = ([item["score_more"], item["score_less"]] if dataset == "crows" else
                          [item["score_stereotype"], item["score_anti_stereotype"], item["score_unrelated"]])
                if any(not math.isfinite(v) for v in scores):
                    raise ValueError(f"nonfinite score: {dataset} {item['id']}")
            print(dataset, n, "valid")
    if args.mia:
        method = "mean_8x_30pct_masked_response_logprob_dream_aligned_v2"
        for mode in ("base", "full", "goldfish"):
            items = check(out / f"mia_{mode}_dream_aligned_v2.jsonl",
                          data / "synthetic_mia/candidates.jsonl", method, 192)
            if sum(r["member"] for r in items) != 96:
                raise ValueError(f"candidate balance changed: {mode}")
            if any(not math.isfinite(r["response_score"]) for r in items):
                raise ValueError(f"nonfinite MIA score: {mode}")
            utility = check(out / f"utility_{mode}_dream_aligned_v2.jsonl",
                            data / "synthetic_mia/utility_holdout.jsonl", method, 128)
            if any(not math.isfinite(r["response_score"]) for r in utility):
                raise ValueError(f"nonfinite utility score: {mode}")
            print(mode, "192 MIA and 128 utility rows valid")


if __name__ == "__main__":
    main()
