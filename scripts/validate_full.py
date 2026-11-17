#!/usr/bin/env python3
"""Validate completeness, provenance, and basic value integrity of full outputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen", "llama", "llada", "dream", "llama31")
DATASETS = ("bbq_full", "bold_full", "crows_full", "stereoset_full")


def output_path(dataset, model):
    stem = f"bbq_{dataset}" if dataset.startswith("bbq_") else dataset
    return ROOT / "results" / f"{stem}_{model}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = ap.parse_args()
    for dataset in DATASETS:
        source = ROOT / "data" / "processed" / f"{dataset}.jsonl"
        blob = source.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        expected = {json.loads(line)["id"] for line in blob.splitlines() if line}
        if len(expected) != sum(bool(line) for line in blob.splitlines()):
            raise ValueError(f"duplicate input ids: {dataset}")
        for model in args.models:
            path = output_path(dataset, model)
            records = [json.loads(line) for line in path.read_bytes().splitlines() if line]
            ids = [r["id"] for r in records]
            if len(ids) != len(set(ids)) or set(ids) != expected:
                raise ValueError(f"missing, extra, or duplicate ids: {path}")
            if any(r["model"] != model or r["data_sha256"] != digest for r in records):
                raise ValueError(f"model or input hash mismatch: {path}")
            revisions = {r["model_revision"] for r in records}
            if len(revisions) != 1:
                raise ValueError(f"mixed checkpoint revisions: {path}")
            if dataset == "bbq_full" and any(r["prediction"] not in (None, 0, 1, 2) for r in records):
                raise ValueError(f"invalid BBQ choice: {path}")
            if dataset in ("crows_full", "stereoset_full"):
                keys = ("score_more", "score_less") if dataset == "crows_full" else (
                    "score_stereotype", "score_anti_stereotype", "score_unrelated")
                if any(not math.isfinite(r[key]) for r in records for key in keys):
                    raise ValueError(f"nonfinite pair score: {path}")
            empty = sum(not r["continuation"].strip() for r in records) if dataset == "bold_full" else 0
            print(f"{model}\t{dataset}\t{len(records)}\tvalid\tempty_continuations={empty}")


if __name__ == "__main__":
    main()
