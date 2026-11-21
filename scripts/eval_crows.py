#!/usr/bin/env python3
"""Pilot pair preference: AR token likelihood / diffusion masked-token score."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dllm_eval.models import Generator, MODELS
from dllm_eval.scoring import masked_prediction_positions


def score(generator, text, seed, masks=4, ratio=0.3):
    import torch
    import torch.nn.functional as F
    tokenizer = generator.tokenizer
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to("cuda")
    if tokenizer.bos_token_id is not None and ids[0, 0].item() != tokenizer.bos_token_id:
        ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]], device="cuda", dtype=ids.dtype), ids], dim=1)
    with torch.inference_mode():
        if generator.kind == "ar":
            logits = generator.model(ids).logits[:, :-1, :].float()
            target = ids[:, 1:]
            return F.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).mean().item()
        mask_id = getattr(generator.model.config, "mask_token_id", None)
        if mask_id is None:
            raise RuntimeError("diffusion checkpoint has no mask_token_id")
        eligible = [i for i, token in enumerate(ids[0].tolist()) if token not in tokenizer.all_special_ids]
        if not eligible:
            raise ValueError("empty text after tokenization")
        rng = torch.Generator(device="cpu").manual_seed(seed)
        scores = []
        for _ in range(masks):
            count = max(1, round(len(eligible) * ratio))
            perm = torch.randperm(len(eligible), generator=rng)[:count].tolist()
            positions = torch.tensor([eligible[i] for i in perm], device="cuda")
            masked = ids.clone()
            masked[0, positions] = mask_id
            prediction_positions = masked_prediction_positions(generator.kind, positions)
            logits = generator.model(masked).logits[0, prediction_positions, :].float()
            targets = ids[0, positions]
            scores.append(F.log_softmax(logits, dim=-1).gather(1, targets[:, None]).mean().item())
        return sum(scores) / len(scores)


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
            more = score(generator, row["sent_more"], seed)
            less = score(generator, row["sent_less"], seed)
            record = {**row, "model": args.model, "model_repo": generator.repo,
                      "model_revision": generator.revision, "data_sha256": digest,
                      "score_more": more, "score_less": less, "prefers_more": more > less,
                      "score_method": "mean_ar_token_logprob" if generator.kind == "ar" else
                                      f"mean_masked_token_logprob_4x_30pct_{generator.kind}_aligned_v2",
                      "seed": seed}
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            print(f"{i}/{len(remaining)} {row['id']} prefers_more={record['prefers_more']}", flush=True)


if __name__ == "__main__":
    # A separate pairs job may overlap a resumable all_full job. Lock before
    # reading existing rows so only one writer can append to this output.
    import fcntl
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
