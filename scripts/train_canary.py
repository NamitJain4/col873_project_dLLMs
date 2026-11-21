#!/usr/bin/env python3
"""Matched LoRA fine-tuning on fictional email canaries for AR and masked dLLMs."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dllm_eval.models import MODELS
from dllm_eval.scoring import masked_prediction_positions

INSTRUCTION = "Continue this fictional email exactly from its prefix. Write only the continuation.\n\n"


def source_pair(row, candidates):
    text = row["text"]
    if row["id"].startswith("synth:"):
        candidate = candidates[row["id"]]
        if not candidate["member"] or candidate["text"] != text:
            raise ValueError("training member/candidate mismatch")
        prefix = candidate["prompt"]
    else:
        body_start = text.index("\nThe ") + 1
        prefix = text[:body_start]
    if not text.startswith(prefix):
        raise ValueError("prefix mismatch")
    return prefix, text[len(prefix):]


def encode(tokenizer, prefix, response):
    messages = [{"role": "user", "content": INSTRUCTION + prefix}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": response}])
    if full[:len(prompt)] != prompt or len(full) > 256 or len(full) <= len(prompt):
        raise ValueError("chat template prefix or sequence length invalid")
    return full, len(prompt)


def retained_targets(row_id, answer_len, seed, drop_fraction):
    if drop_fraction == 0:
        return [True] * answer_len
    digest = hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    keep = [rng.random() >= drop_fraction for _ in range(answer_len)]
    if not any(keep):
        keep[0] = True
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("qwen", "llama", "llada", "dream"), required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed/synthetic_mia"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=0, help="0 trains on the full fixed corpus")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accumulation", type=int, default=8)
    ap.add_argument("--loss-drop-fraction", type=float, default=0.0,
                    help="stable Goldfish-style omission of response-token loss")
    args = ap.parse_args()
    if (args.max_examples < 0 or args.rank < 1 or args.lr <= 0 or args.accumulation < 1 or
            not 0 <= args.loss_drop_fraction < 1):
        ap.error("invalid training hyperparameters")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite adapter: {args.output}")

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("training requires a GPU")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    repo, kind, revision = MODELS[args.model]
    remote = kind != "ar"
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=remote)
    candidates = {r["id"]: r for r in (json.loads(s) for s in (args.data_dir / "candidates.jsonl").read_bytes().splitlines() if s)}
    train_path = args.data_dir / "train.jsonl"
    train_blob = train_path.read_bytes()
    train = [json.loads(s) for s in train_blob.splitlines() if s]
    if args.max_examples:
        train = train[:args.max_examples]
    encoded = []
    for row in train:
        tokens, prompt_len = encode(tokenizer, *source_pair(row, candidates))
        keep = retained_targets(row["id"], len(tokens) - prompt_len,
                                args.seed, args.loss_drop_fraction)
        encoded.append((tokens, prompt_len, keep))
    print(f"model={args.model} examples={len(encoded)} max_length={max(len(x) for x, _, _ in encoded)} "
          f"loss_drop_fraction={args.loss_drop_fraction}", flush=True)

    cls = AutoModelForCausalLM if kind == "ar" else AutoModel
    model = cls.from_pretrained(repo, revision=revision, trust_remote_code=remote,
                                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
                                            target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA selected no parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    mask_id = getattr(model.config, "mask_token_id", None)
    if kind != "ar" and mask_id is None:
        raise RuntimeError("diffusion model has no mask token")
    losses = []
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    for i, (tokens, prompt_len, keep_flags) in enumerate(encoded, 1):
        ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
        if kind == "ar":
            labels = ids.clone()
            labels[:, :prompt_len] = -100
            if args.loss_drop_fraction:
                dropped = torch.tensor([not x for x in keep_flags], device="cuda", dtype=torch.bool)
                response_positions = torch.arange(prompt_len, ids.shape[1], device="cuda")
                labels[0, response_positions[dropped]] = -100
            loss = model(input_ids=ids, labels=labels).loss.float()
        else:
            eligible = torch.tensor([j for j, keep in enumerate(keep_flags) if keep], device="cuda") + prompt_len
            p_mask = 0.05 + 0.9 * torch.rand((), device="cuda")
            mask = torch.rand(len(eligible), device="cuda") < p_mask
            if not mask.any():
                mask[torch.randint(len(eligible), (1,), device="cuda")] = True
            positions = eligible[mask]
            noisy = ids.clone()
            noisy[0, positions] = mask_id
            prediction_positions = masked_prediction_positions(kind, positions)
            logits = model(input_ids=noisy).logits[0, prediction_positions].float()
            loss = F.cross_entropy(logits, ids[0, positions], reduction="sum") / (p_mask * len(eligible))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at example {i}")
        losses.append(loss.item())
        (loss / args.accumulation).backward()
        if i % args.accumulation == 0 or i == len(encoded):
            gradients = [p.grad for p in trainable if p.grad is not None]
            if (not gradients or any(not torch.isfinite(g).all() for g in gradients) or
                    not any(g.abs().sum() > 0 for g in gradients)):
                raise FloatingPointError(f"missing or invalid LoRA gradient at example {i}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        if i % 50 == 0 or i == len(encoded):
            print(f"{i}/{len(encoded)} steps={steps} recent_loss={sum(losses[-50:])/len(losses[-50:]):.4f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    manifest = {"model": args.model, "model_repo": repo, "model_revision": revision,
                "kind": kind, "train_sha256": hashlib.sha256(train_blob).hexdigest(),
                "examples": len(encoded), "optimizer_steps": steps,
                "rank": args.rank, "lr": args.lr, "accumulation": args.accumulation,
                "loss_drop_fraction": args.loss_drop_fraction,
                "seed": args.seed, "mean_loss": sum(losses) / len(losses),
                "last_50_loss": sum(losses[-50:]) / len(losses[-50:]),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "instruction": INSTRUCTION,
                "objective": "response-only causal CE" if kind == "ar" else
                             f"response-only random-mask CE weighted by 1/p ({kind} aligned v2)"}
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
