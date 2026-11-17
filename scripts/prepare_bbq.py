#!/usr/bin/env python3
"""Download a reproducible BBQ extract from the HF dataset viewer API."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

BASE = "https://datasets-server.huggingface.co"
REPO = "heegyu/bbq"
OFFICIAL_REVISION = "bea11bd97d79217245b5871acd247b9d6eb24598"
OFFICIAL_BASE = f"https://raw.githubusercontent.com/nyu-mll/BBQ/{OFFICIAL_REVISION}/data/"
OFFICIAL_CONFIGS = ("Age", "Disability_status", "Gender_identity", "Nationality",
                    "Physical_appearance", "Race_ethnicity", "Race_x_SES",
                    "Race_x_gender", "Religion", "SES", "Sexual_orientation")


def get(path, query):
    url = BASE + path + "?" + urlencode(query)
    for attempt in range(5):
        try:
            with urlopen(url, timeout=45) as response:
                return json.load(response)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=1)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.per_stratum < 1:
        ap.error("--per-stratum must be positive")
    configs = (sorted(OFFICIAL_CONFIGS) if args.full else
               sorted({x["config"] for x in get("/splits", {"dataset": REPO})["splits"] if x["split"] == "test"}))
    records = []
    for config in configs:
        if args.full:
            for attempt in range(5):
                try:
                    with urlopen(OFFICIAL_BASE + config + ".jsonl", timeout=120) as response:
                        lines = response.read().decode("utf-8").split("\n")
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            chosen = {"ambig": 0, "disambig": 0}
            for row_idx, line in enumerate(lines):
                row = json.loads(line)
                condition = row["context_condition"]
                if condition not in chosen:
                    continue
                if row["category"] != config or row["example_id"] != row_idx:
                    raise ValueError(f"Unexpected BBQ row order in {config} at {row_idx}")
                records.append({"id": f"{config}:{row_idx}", "category": config,
                                "context_condition": condition, "source_row": row_idx,
                                "context": row["context"], "question": row["question"],
                                "answers": [row["ans0"], row["ans1"], row["ans2"]],
                                "label": int(row["label"]), "question_polarity": row["question_polarity"],
                                "answer_info": row["answer_info"],
                                "additional_metadata": row["additional_metadata"]})
                chosen[condition] += 1
            print(config, chosen, flush=True)
            continue
        chosen = {"ambig": 0, "disambig": 0}
        offset = 0
        while True:
            page = get("/rows", {"dataset": REPO, "config": config, "split": "test", "offset": offset, "length": 100})
            rows = page["rows"]
            for item in rows:
                row = item["row"]
                condition = row.get("context_condition")
                if condition not in chosen or (not args.full and chosen[condition] >= args.per_stratum):
                    continue
                if not all(row.get(k) is not None for k in ("context", "question", "ans0", "ans1", "ans2", "label")):
                    continue
                records.append({"id": f"{config}:{item['row_idx']}", "category": config,
                                "context_condition": condition, "source_row": item["row_idx"],
                                "context": row["context"], "question": row["question"],
                                "answers": [row["ans0"], row["ans1"], row["ans2"]],
                                "label": int(row["label"]), "question_polarity": row.get("question_polarity"),
                                "answer_info": row.get("answer_info"),
                                "additional_metadata": row.get("additional_metadata")})
                chosen[condition] += 1
            offset += len(rows)
            if not rows or (not args.full and all(n >= args.per_stratum for n in chosen.values())):
                break
        print(config, chosen, flush=True)
        if not args.full and any(n < args.per_stratum for n in chosen.values()):
            raise RuntimeError(f"Insufficient rows for {config}: {chosen}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in records)
    args.output.write_text(payload)
    print(f"wrote {len(records)} rows to {args.output}; sha256={hashlib.sha256(payload.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
