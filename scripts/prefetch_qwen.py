#!/usr/bin/env python3
"""Fetch only the missing Qwen weight shards, with resumable retries."""
import json
import time

from huggingface_hub import hf_hub_download

REPO = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def fetch(filename):
    for attempt in range(1, 9):
        try:
            path = hf_hub_download(REPO, filename, revision=REVISION)
            print(f"cached {filename}: {path}", flush=True)
            return path
        except Exception as exc:
            print(f"attempt {attempt}/8 failed for {filename}: {type(exc).__name__}: {str(exc)[:250]}", flush=True)
            if attempt == 8:
                raise
            time.sleep(min(120, 10 * attempt))


def main():
    index = fetch("model.safetensors.index.json")
    names = sorted(set(json.load(open(index))["weight_map"].values()))
    for name in names:
        fetch(name)
    print(f"All {len(names)} Qwen shards cached", flush=True)


if __name__ == "__main__":
    main()
