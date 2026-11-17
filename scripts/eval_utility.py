#!/usr/bin/env python3
"""Score held-out fictional routine-email continuations with each checkpoint."""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dllm_eval.models import Generator
from train_canary import source_pair
from eval_canary import response_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("qwen", "llama", "llada", "dream"), required=True)
    ap.add_argument("--mode", choices=("base", "full", "goldfish"), required=True)
    ap.add_argument("--data", type=Path, default=Path("data/processed/synthetic_mia/utility_holdout.jsonl"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    blob = args.data.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    rows = [json.loads(s) for s in blob.splitlines() if s]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate utility ids")
    adapter = Path(f"checkpoints/{args.mode}_{args.model}") if args.mode != "base" else None
    adapter_hash = None
    if adapter:
        weights = adapter / "adapter_model.safetensors"
        adapter_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    done = {}
    if args.output.exists():
        for line in args.output.read_bytes().splitlines():
            if line:
                item = json.loads(line)
                if (item["model"] != args.model or item["mode"] != args.mode or
                        item["data_sha256"] != digest or item["adapter_sha256"] != adapter_hash):
                    raise ValueError("existing output belongs to another utility run")
                done[item["id"]] = item
    remaining = [r for r in rows if r["id"] not in done]
    print(f"model={args.model} mode={args.mode} rows={len(rows)} remaining={len(remaining)}", flush=True)
    if not remaining:
        return
    generator = Generator(args.model)
    if adapter:
        from peft import PeftModel
        generator.model = PeftModel.from_pretrained(generator.model, adapter).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as sink:
        for i, row in enumerate(remaining, 1):
            prefix, response = source_pair(row, {})
            seed = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
            score = response_score(generator, prefix, response, seed)
            if not math.isfinite(score):
                raise FloatingPointError(f"nonfinite utility score for {row['id']}")
            record = {"id": row["id"], "model": args.model, "mode": args.mode,
                      "model_repo": generator.repo, "model_revision": generator.revision,
                      "data_sha256": digest, "adapter_sha256": adapter_hash,
                      "response_score": score,
                      "score_method": "mean_ar_response_logprob" if generator.kind == "ar" else
                                      "mean_8x_30pct_masked_response_logprob", "seed": seed}
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            if i % 20 == 0 or i == len(remaining):
                print(f"{i}/{len(remaining)}", flush=True)


if __name__ == "__main__":
    main()
