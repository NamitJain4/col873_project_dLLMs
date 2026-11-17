#!/usr/bin/env python3
"""BBQ pilot balanced across context condition and question polarity."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.request import urlopen

BASE = "https://huggingface.co/datasets/heegyu/bbq/resolve/5d6faae52070aa5eb71b46d1c0723d3ba7930209/data/"
METADATA_URL = "https://raw.githubusercontent.com/nyu-mll/BBQ/bea11bd97d79217245b5871acd247b9d6eb24598/analysis_scripts/additional_metadata.csv"
CONFIGS = ["Age", "Disability_status", "Gender_identity", "Nationality", "Physical_appearance",
           "Race_ethnicity", "Race_x_SES", "Race_x_gender", "Religion", "SES", "Sexual_orientation"]


def get_lines(config):
    for attempt in range(5):
        try:
            with urlopen(BASE + config + ".jsonl", timeout=45) as response:
                return response.read().decode("utf-8").split("\n")
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    records = []
    for config in CONFIGS:
        needed = {(condition, polarity) for condition in ("ambig", "disambig") for polarity in ("neg", "nonneg")}
        chosen = set()
        for index, line in enumerate(get_lines(config)):
            row = json.loads(line)
            key = (row.get("context_condition"), row.get("question_polarity"))
            if key not in needed or key in chosen:
                continue
            if not all(row.get(k) is not None for k in ("context", "question", "ans0", "ans1", "ans2", "label")):
                continue
            records.append({"id": f"{config}:{index}", "category": config,
                            "context_condition": row["context_condition"], "source_row": index,
                            "context": row["context"], "question": row["question"],
                            "answers": [row["ans0"], row["ans1"], row["ans2"]],
                            "label": int(row["label"]), "question_polarity": row["question_polarity"],
                            "answer_info": row.get("answer_info"),
                            "additional_metadata": row.get("additional_metadata")})
            chosen.add(key)
            if chosen == needed:
                break
        if chosen != needed:
            raise RuntimeError(f"missing BBQ strata for {config}: {needed - chosen}")
        print(config, sorted(chosen), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    args.output.write_text(payload)
    print(f"wrote {len(records)} rows; sha256={hashlib.sha256(payload.encode()).hexdigest()}")
    metadata_path = args.output.parent / "bbq_additional_metadata.csv"
    if not metadata_path.exists() or not metadata_path.stat().st_size:
        for attempt in range(5):
            try:
                with urlopen(METADATA_URL, timeout=60) as response:
                    metadata_path.write_bytes(response.read())
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    print(f"metadata: {metadata_path}; sha256={hashlib.sha256(metadata_path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
