#!/usr/bin/env python3
"""Check durable output counts and input hashes for pilot and full runs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen", "llama", "llama31", "llada", "dream")
DATASETS = ("bbq_pilot", "bbq_balanced", "bold_pilot", "crows_pilot", "stereoset_pilot",
            "bbq_full", "bold_full", "crows_full", "stereoset_full")


def result_path(name, model):
    filename = f"bbq_{name}_{model}.jsonl" if name.startswith("bbq_") else f"{name}_{model}.jsonl"
    return ROOT / "results" / filename


def main():
    print("model\tdataset\trows\texpected\tstatus")
    for model in MODELS:
        for name in DATASETS:
            source = ROOT / "data" / "processed" / f"{name}.jsonl"
            output = result_path(name, model)
            if not source.exists():
                print(f"{model}\t{name}\t0\t?\tmissing input")
                continue
            expected = sum(bool(line) for line in source.read_bytes().splitlines())
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if not output.exists():
                print(f"{model}\t{name}\t0\t{expected}\tpending")
                continue
            try:
                rows = [json.loads(s) for s in output.read_bytes().splitlines() if s]
                ids = {r["id"] for r in rows}
                valid = len(ids) == len(rows) and all(r["model"] == model and r["data_sha256"] == digest for r in rows)
            except (KeyError, ValueError, json.JSONDecodeError):
                valid = False
                rows = []
            status = "invalid" if not valid else "complete" if len(rows) == expected else "partial"
            print(f"{model}\t{name}\t{len(rows)}\t{expected}\t{status}")


if __name__ == "__main__":
    main()
