#!/usr/bin/env python3
"""Generate and save BBQ predictions, one durable record at a time."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dllm_eval.bbq import make_prompt, parse_choice
from dllm_eval.models import Generator, MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()
    if args.new_tokens < 1 or args.steps < 1:
        ap.error("generation lengths must be positive")
    data_bytes = args.data.read_bytes()
    data_sha256 = hashlib.sha256(data_bytes).hexdigest()
    rows = [json.loads(line) for line in data_bytes.splitlines() if line]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate benchmark ids")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if args.output.exists():
        for line in args.output.read_bytes().splitlines():
            if line:
                item = json.loads(line)
                if item["model"] != args.model or item["data_sha256"] != data_sha256:
                    raise ValueError("existing output belongs to a different run or dataset")
                completed[item["id"]] = item
    remaining = [r for r in rows if r["id"] not in completed]
    print(f"model={args.model} dataset_rows={len(rows)} completed={len(completed)} remaining={len(remaining)} sha256={data_sha256}", flush=True)
    if not remaining:
        return
    torch_seed = 42
    import torch
    torch.manual_seed(torch_seed)
    generator = Generator(args.model)
    with args.output.open("a") as sink:
        for n, row in enumerate(remaining, 1):
            prompt = make_prompt(row)
            start = time.monotonic()
            response = generator.generate(prompt, args.new_tokens, args.steps)
            pred = parse_choice(response)
            record = {"id": row["id"], "model": args.model, "model_repo": generator.repo,
                      "model_revision": generator.revision,
                      "data_sha256": data_sha256, "category": row["category"],
                      "context_condition": row["context_condition"], "label": row["label"],
                      "prediction": pred, "correct": pred == row["label"], "response": response,
                      "prompt": prompt, "latency_s": round(time.monotonic() - start, 3),
                      "new_tokens": args.new_tokens, "steps": args.steps, "seed": torch_seed}
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            print(f"{n}/{len(remaining)} {row['id']} pred={pred} label={row['label']} time={record['latency_s']}s", flush=True)


if __name__ == "__main__":
    main()
