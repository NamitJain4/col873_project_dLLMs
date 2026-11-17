#!/usr/bin/env python3
"""Validate controlled MIA outputs before reporting extraction or AUROC."""
import hashlib
import json
import math
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data/processed/synthetic_mia/candidates.jsonl"
blob = source.read_bytes()
digest = hashlib.sha256(blob).hexdigest()
candidates = {r["id"]: r for r in (json.loads(s) for s in blob.splitlines() if s)}
if len(candidates) != 192 or sum(r["member"] for r in candidates.values()) != 96:
    raise ValueError("unexpected candidate membership balance")
ap = argparse.ArgumentParser()
ap.add_argument("--modes", nargs="+", choices=("base", "full", "goldfish"), default=("base", "full"))
args = ap.parse_args()
for model in ("qwen", "llama", "llada", "dream"):
    for mode in args.modes:
        path = ROOT / "results/mia" / f"{mode}_{model}.jsonl"
        rows = [json.loads(s) for s in path.read_bytes().splitlines() if s]
        ids = [r["id"] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(candidates):
            raise ValueError(f"missing, extra, or duplicate ids: {path}")
        if any(r["model"] != model or r["data_sha256"] != digest for r in rows):
            raise ValueError(f"model or data hash mismatch: {path}")
        if any((r["adapter_sha256"] is None) != (mode == "base") for r in rows):
            raise ValueError(f"adapter mode mismatch: {path}")
        for row in rows:
            source_row = candidates[row["id"]]
            if (row["member"] != source_row["member"] or
                    row["repetition_group"] != source_row["repetition_group"] or
                    row["canary"] != source_row["canary"] or
                    row["exact_extraction"] != (row["canary"] in row["generated"]) or
                    not math.isfinite(row["response_score"])):
                raise ValueError(f"candidate label or score mismatch: {path} {row['id']}")
        print(f"{model}\t{mode}\t{len(rows)}\tvalid")
