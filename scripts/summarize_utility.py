#!/usr/bin/env python3
"""Validate and summarize paired held-out utility scores within each model."""
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data/processed/synthetic_mia/utility_holdout.jsonl"
blob = source.read_bytes()
digest = hashlib.sha256(blob).hexdigest()
expected = {json.loads(s)["id"] for s in blob.splitlines() if s}
if len(expected) != 128:
    raise ValueError("unexpected utility holdout size")

print("model\tmode\tn\tmean_response_score\tdelta_from_full\tdelta_ci_low\tdelta_ci_high")
for model in ("qwen", "llama", "llada", "dream"):
    scores = {}
    for mode in ("base", "full", "goldfish"):
        path = ROOT / "results/utility" / f"{mode}_{model}.jsonl"
        rows = [json.loads(s) for s in path.read_bytes().splitlines() if s]
        ids = [r["id"] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError(f"missing, extra, or duplicate utility ids: {path}")
        if any(r["model"] != model or r["mode"] != mode or
               r["data_sha256"] != digest or not math.isfinite(r["response_score"]) for r in rows):
            raise ValueError(f"utility provenance or score mismatch: {path}")
        if any((r["adapter_sha256"] is None) != (mode == "base") for r in rows):
            raise ValueError(f"utility adapter mismatch: {path}")
        scores[mode] = {r["id"]: r["response_score"] for r in rows}
    for mode in ("base", "full", "goldfish"):
        values = [scores[mode][i] for i in sorted(expected)]
        if mode == "full":
            print(f"{model}\t{mode}\t128\t{mean(values):.6f}\t0\t0\t0")
            continue
        deltas = [scores[mode][i] - scores["full"][i] for i in sorted(expected)]
        rng = random.Random(42)
        draws = sorted(mean(deltas[rng.randrange(len(deltas))] for _ in deltas) for _ in range(1000))
        print(f"{model}\t{mode}\t128\t{mean(values):.6f}\t{mean(deltas):.6f}\t"
              f"{draws[24]:.6f}\t{draws[974]:.6f}")
