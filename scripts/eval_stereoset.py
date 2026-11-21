#!/usr/bin/env python3
"""Pilot StereoSet preference and language-modeling checks."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dllm_eval.models import Generator, MODELS
from eval_crows import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    blob = args.data.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    expected_method = ("mean_ar_token_logprob" if MODELS[args.model][1] == "ar" else
                       f"mean_masked_token_logprob_4x_30pct_{MODELS[args.model][1]}_aligned_v2")
    rows = [json.loads(s) for s in blob.splitlines() if s]
    done = {}
    if args.output.exists():
        for s in args.output.read_bytes().splitlines():
            if s:
                item = json.loads(s)
                if (item["model"] != args.model or item["data_sha256"] != digest or
                        item["score_method"] != expected_method):
                    raise ValueError("existing results belong to another run")
                done[item["id"]] = item
    remaining = [r for r in rows if r["id"] not in done]
    print(f"model={args.model} rows={len(rows)} remaining={len(remaining)}", flush=True)
    if not remaining:
        return
    generator = Generator(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as sink:
        for i, row in enumerate(remaining, 1):
            seed = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
            s = score(generator, row["stereotype"], seed)
            a = score(generator, row["anti_stereotype"], seed)
            u = score(generator, row["unrelated"], seed)
            record = {**row, "model": args.model, "model_repo": generator.repo,
                      "model_revision": generator.revision, "data_sha256": digest,
                      "score_stereotype": s, "score_anti_stereotype": a, "score_unrelated": u,
                      "prefers_stereotype": s > a, "prefers_meaningful": (s > u, a > u),
                      "score_method": "mean_ar_token_logprob" if generator.kind == "ar" else
                                      f"mean_masked_token_logprob_4x_30pct_{generator.kind}_aligned_v2",
                      "seed": seed}
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            print(f"{i}/{len(remaining)} {row['id']} prefers_stereotype={record['prefers_stereotype']}", flush=True)


if __name__ == "__main__":
    import fcntl
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
