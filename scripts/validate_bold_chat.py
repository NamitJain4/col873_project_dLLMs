#!/usr/bin/env python3
"""Validate the full instructed BOLD condition against the pinned prompt extract."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data/processed/bold_full.jsonl"
blob = source.read_bytes()
digest = hashlib.sha256(blob).hexdigest()
expected = {json.loads(line)["id"] for line in blob.splitlines() if line}
if len(expected) != sum(bool(line) for line in blob.splitlines()):
    raise ValueError("duplicate source ids")
for model in ("qwen", "llama", "llada", "dream", "llama31"):
    path = ROOT / "results" / f"bold_chat_full_{model}.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError(f"missing, extra, or duplicate ids: {path}")
    if any(r["model"] != model or r["data_sha256"] != digest or
           r["prompt_mode"] != "chat_continuation" for r in rows):
        raise ValueError(f"model, input hash, or prompt mode mismatch: {path}")
    print(f"{model}\t{len(rows)}\tvalid\tempty={sum(not r['continuation'].strip() for r in rows)}")
