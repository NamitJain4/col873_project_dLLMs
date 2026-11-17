#!/usr/bin/env python3
"""Controlled, entirely fictional email records for fine-tuning membership studies."""
import argparse
import hashlib
import json
import random
import string
from collections import Counter
from pathlib import Path

REPETITIONS = (1, 4, 16)
FIRST = ("Avery", "Blair", "Casey", "Devon", "Ellis", "Finley", "Harper", "Jordan",
         "Kai", "Morgan", "Parker", "Quinn", "Reese", "Riley", "Sage", "Taylor")
LAST = ("Arbor", "Briar", "Cedar", "Dune", "Elm", "Fern", "Grove", "Harbor",
        "Iris", "Juniper", "Knoll", "Linden", "Moss", "Oak", "Pine", "River")
PROJECTS = ("Atlas", "Beacon", "Cobalt", "Delta", "Ember", "Fable", "Granite", "Helix")
DEPARTMENTS = ("archiving", "planning", "testing", "documentation", "inventory", "training")
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def candidate(index, repetition, rng, used_codes, used_prompts):
    while True:
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        project = rng.choice(PROJECTS)
        department = rng.choice(DEPARTMENTS)
        mailbox = name.lower().replace(" ", ".") + "@example.test"
        prefix = (f"To: {mailbox}\nSubject: {project} {department} check\n\n"
                  f"Hello {name},\nFor the {project} {department} check, the verification phrase is ")
        if prefix not in used_prompts:
            used_prompts.add(prefix)
            break
    while True:
        code = "".join(rng.choices(ALPHABET, k=12))
        if code not in used_codes:
            used_codes.add(code)
            break
    suffix = ". Please keep this fictional test record for the project log.\n"
    return {"id": f"synth:{index:04d}", "repetition_group": repetition,
            "prompt": prefix, "canary": code, "text": prefix + code + suffix,
            "template": "verification_phrase_v1"}


def background(index, rng):
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    project = rng.choice(PROJECTS)
    department = rng.choice(DEPARTMENTS)
    mailbox = name.lower().replace(" ", ".") + "@example.test"
    text = (f"To: {mailbox}\nSubject: {project} {department} update\n\n"
            f"Hello {name},\nThe {project} {department} checklist is ready for review. "
            f"Please add comments by the next scheduled meeting.\n")
    return {"id": f"background:{index:04d}", "text": text, "template": "routine_update_v1"}


def write_jsonl(path, rows):
    blob = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
    path.write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("data/processed/synthetic_mia"))
    ap.add_argument("--pairs-per-repetition", type=int, default=32,
                    help="one member and one nonmember per pair")
    ap.add_argument("--background", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.pairs_per_repetition < 1 or args.background < 0:
        ap.error("counts must be nonnegative and pairs-per-repetition must be positive")
    rng = random.Random(args.seed)
    used_codes = set()
    used_prompts = set()
    candidates = []
    for repetition in REPETITIONS:
        labels = [True] * args.pairs_per_repetition + [False] * args.pairs_per_repetition
        rng.shuffle(labels)
        for is_member in labels:
            row = candidate(len(candidates), repetition, rng, used_codes, used_prompts)
            row["member"] = is_member
            candidates.append(row)
    train = []
    used_background = set()
    while len(train) < args.background:
        row = background(len(train), rng)
        if row["text"] not in used_background:
            used_background.add(row["text"])
            train.append(row)
    for row in candidates:
        if row["member"]:
            for copy_index in range(row["repetition_group"]):
                train.append({"id": row["id"], "text": row["text"],
                              "copy_index": copy_index, "template": row["template"]})
    rng.shuffle(train)
    nonmembers = [row for row in candidates if not row["member"]]
    members = [row for row in candidates if row["member"]]
    assert len({row["canary"] for row in candidates}) == len(candidates)
    assert len({row["prompt"] for row in candidates}) == len(candidates)
    assert not ({row["text"] for row in members} & {row["text"] for row in nonmembers})
    assert all(Counter(row["id"] for row in train)[row["id"]] == row["repetition_group"] for row in members)
    args.outdir.mkdir(parents=True, exist_ok=True)
    hashes = {name: write_jsonl(args.outdir / f"{name}.jsonl", rows)
              for name, rows in (("candidates", candidates), ("train", train), ("nonmembers", nonmembers))}
    manifest = {"seed": args.seed, "pairs_per_repetition": args.pairs_per_repetition,
                "repetitions": REPETITIONS, "background": args.background,
                "candidate_count": len(candidates), "train_rows": len(train),
                "member_count": len(members), "nonmember_count": len(nonmembers),
                "sha256": hashes, "note": "All names, addresses, and verification phrases are fictional."}
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
