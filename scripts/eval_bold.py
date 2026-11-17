#!/usr/bin/env python3
"""Save BOLD continuations for later sentiment/regard scoring."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dllm_eval.models import Generator, MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--chat-continuation", action="store_true",
                    help="ask instruction checkpoints to continue the prefix in a chat message")
    args = ap.parse_args()
    blob = args.data.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    rows = [json.loads(line) for line in blob.splitlines() if line]
    done = {}
    if args.output.exists():
        for line in args.output.read_bytes().splitlines():
            if line:
                item = json.loads(line)
                mode = "chat_continuation" if args.chat_continuation else "raw_prefix"
                if (item["model"] != args.model or item["data_sha256"] != digest or
                        item.get("prompt_mode", "raw_prefix") != mode):
                    raise ValueError("existing results belong to another run")
                done[item["id"]] = item
    remaining = [r for r in rows if r["id"] not in done]
    print(f"model={args.model} rows={len(rows)} remaining={len(remaining)}", flush=True)
    if not remaining:
        return
    import torch
    torch.manual_seed(42)
    generator = Generator(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as sink:
        for i, row in enumerate(remaining, 1):
            start = time.monotonic()
            model_input = ("Continue this passage naturally. Write only the continuation.\n\n"
                           "Passage: " + row["prompt"]) if args.chat_continuation else row["prompt"]
            skip_reason = "empty_source_prompt" if not row["prompt"].strip() else None
            continuation = ("" if skip_reason else
                            generator.generate(model_input, new_tokens=48, steps=24,
                                               chat=args.chat_continuation))
            result = {**row, "model": args.model, "model_repo": generator.repo,
                      "model_revision": generator.revision, "data_sha256": digest,
                      "prompt_mode": "chat_continuation" if args.chat_continuation else "raw_prefix",
                      "model_input": model_input,
                      "skip_reason": skip_reason,
                      "continuation": continuation, "latency_s": round(time.monotonic() - start, 3),
                      "new_tokens": 48, "steps": 24, "seed": 42}
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            print(f"{i}/{len(remaining)} {row['id']} {result['latency_s']}s", flush=True)


if __name__ == "__main__":
    main()
