#!/usr/bin/env python3
"""Measure exact canary extraction and model-specific response likelihood."""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dllm_eval.models import Generator, MODELS
from dllm_eval.scoring import masked_prediction_positions
from train_canary import INSTRUCTION, encode


def response_score(generator, prompt, response, seed, masks=8):
    import torch
    import torch.nn.functional as F
    tokens, prompt_len = encode(generator.tokenizer, prompt, response)
    ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        if generator.kind == "ar":
            logits = generator.model(input_ids=ids).logits[:, prompt_len - 1:-1].float()
            targets = ids[:, prompt_len:]
            return F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).mean().item()
        mask_id = getattr(generator.model.config, "mask_token_id", None)
        if mask_id is None:
            raise RuntimeError("diffusion model has no mask token")
        eligible = list(range(prompt_len, ids.shape[1]))
        rng = torch.Generator(device="cpu").manual_seed(seed)
        values = []
        for _ in range(masks):
            count = max(1, round(len(eligible) * 0.3))
            chosen = torch.randperm(len(eligible), generator=rng)[:count]
            positions = torch.tensor([eligible[i] for i in chosen.tolist()], device="cuda")
            noisy = ids.clone()
            noisy[0, positions] = mask_id
            prediction_positions = masked_prediction_positions(generator.kind, positions)
            logits = generator.model(input_ids=noisy).logits[0, prediction_positions].float()
            targets = ids[0, positions]
            values.append(F.log_softmax(logits, dim=-1).gather(1, targets[:, None]).mean().item())
        return sum(values) / len(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("qwen", "llama", "llada", "dream"), required=True)
    ap.add_argument("--data", type=Path, default=Path("data/processed/synthetic_mia/candidates.jsonl"))
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=0)
    args = ap.parse_args()
    if args.max_examples < 0:
        ap.error("--max-examples cannot be negative")
    blob = args.data.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    expected_method = ("mean_ar_response_logprob" if MODELS[args.model][1] == "ar" else
                       f"mean_8x_30pct_masked_response_logprob_{MODELS[args.model][1]}_aligned_v2")
    rows = [json.loads(s) for s in blob.splitlines() if s]
    if args.max_examples:
        rows = rows[:args.max_examples]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate candidate ids")
    adapter_hash = None
    if args.adapter:
        weights = next(args.adapter.glob("adapter_model.*"), None)
        if weights is None:
            raise FileNotFoundError(f"adapter weights missing in {args.adapter}")
        adapter_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    done = {}
    if args.output.exists():
        for s in args.output.read_bytes().splitlines():
            if s:
                r = json.loads(s)
                if (r["model"] != args.model or r["data_sha256"] != digest or
                        r["adapter_sha256"] != adapter_hash or r["score_method"] != expected_method):
                    raise ValueError("output belongs to another candidate set, model, or adapter")
                done[r["id"]] = r
    remaining = [r for r in rows if r["id"] not in done]
    print(f"model={args.model} adapter={args.adapter} candidates={len(rows)} remaining={len(remaining)}", flush=True)
    if not remaining:
        return
    import torch
    torch.manual_seed(42)
    generator = Generator(args.model)
    if args.adapter:
        from peft import PeftModel
        generator.model = PeftModel.from_pretrained(generator.model, args.adapter).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as sink:
        for i, row in enumerate(remaining, 1):
            user_prompt = INSTRUCTION + row["prompt"]
            response = row["text"][len(row["prompt"]):]
            seed = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
            score = response_score(generator, row["prompt"], response, seed)
            generated = generator.generate(user_prompt, new_tokens=48, steps=24, chat=True)
            if not math.isfinite(score):
                raise FloatingPointError(f"nonfinite response score for {row['id']}")
            record = {"id": row["id"], "model": args.model, "model_repo": generator.repo,
                      "model_revision": generator.revision, "data_sha256": digest,
                      "adapter_sha256": adapter_hash, "member": row["member"],
                      "repetition_group": row["repetition_group"], "canary": row["canary"],
                      "generated": generated, "exact_extraction": row["canary"] in generated,
                      "response_score": score,
                      "score_method": "mean_ar_response_logprob" if generator.kind == "ar" else
                                      f"mean_8x_30pct_masked_response_logprob_{generator.kind}_aligned_v2", "seed": seed}
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            if i % 20 == 0 or i == len(remaining):
                print(f"{i}/{len(remaining)} exact={record['exact_extraction']}", flush=True)


if __name__ == "__main__":
    main()
