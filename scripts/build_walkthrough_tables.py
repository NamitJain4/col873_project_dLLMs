#!/usr/bin/env python3
"""Audit existing outputs and write small, shareable notebook inputs.

This script performs no model inference and submits no PBS jobs. The raw JSONL
files stay ignored by Git; only aggregate TSVs are written to results/summary.
"""
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "summary"
MODELS = ("llama", "llada", "qwen", "dream", "llama31")
source_cache = {}
inventory = []
inventory_paths = set()


def source_info(relative):
    path = DATA / relative
    if path not in source_cache:
        blob = path.read_bytes()
        rows = [json.loads(line) for line in blob.splitlines() if line]
        ids = [r["id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate source ids: {path}")
        source_cache[path] = (rows, set(ids), hashlib.sha256(blob).hexdigest())
    return source_cache[path]


def output_rows(phase, benchmark, model, condition, relative, source_relative,
                status="valid", aligned=False):
    source, expected, digest = source_info(source_relative)
    path = RESULTS / relative
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line]
    ids = [r["id"] for r in rows]
    if len(ids) != len(expected) or len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError(f"incomplete or duplicate output: {path}")
    if any(r["model"] != model or r["data_sha256"] != digest for r in rows):
        raise ValueError(f"model or source hash mismatch: {path}")
    if aligned and any("aligned_v2" not in r.get("score_method", "") for r in rows):
        raise ValueError(f"Dream score alignment provenance missing: {path}")
    if phase == "probe" and any(r["condition"] != condition for r in rows):
        raise ValueError(f"probe condition mismatch: {path}")
    if path not in inventory_paths:
        inventory.append((phase, benchmark, model, condition, len(expected), len(rows),
                          status, digest, str(path.relative_to(ROOT))))
        inventory_paths.add(path)
    return source, rows


def write_tsv(name, headers, rows):
    path = SUMMARY / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)
    print(path.relative_to(ROOT), len(rows))


def bold_metrics(source, rows, wanted_ids=None):
    valid = {r["id"] for r in source if r["prompt"].strip()}
    if wanted_ids is not None:
        valid &= wanted_ids
    subset = [r["continuation"] for r in rows if r["id"] in valid]
    if len(subset) != len(valid):
        raise ValueError("BOLD subset is incomplete")
    lengths = [len(text.split()) for text in subset]
    n = len(subset)
    return (n, sum(not text.strip() for text in subset) / n,
            sum(not any(ch.isalnum() for ch in text) for text in subset) / n,
            sum(length >= 20 for length in lengths) / n,
            statistics.median(lengths))


def main():
    bold_coverage = []
    bold_matched = []
    sample600 = {r["id"] for r in source_info("bold_replicate_600.jsonl")[0]}
    full = (("bbq", "bbq_bbq_full_{model}.jsonl", "bbq_full.jsonl"),
            ("bold_raw", "bold_full_{model}.jsonl", "bold_full.jsonl"),
            ("crows", "crows_full_{model}.jsonl", "crows_full.jsonl"),
            ("stereoset", "stereoset_full_{model}.jsonl", "stereoset_full.jsonl"),
            ("bold_chat", "bold_chat_full_{model}.jsonl", "bold_full.jsonl"))
    for benchmark, pattern, source_path in full:
        for model in MODELS:
            status = "invalid_dream_alignment" if (model == "dream" and
                     benchmark in ("crows", "stereoset")) else (
                     "legacy_fixed_span_decode" if benchmark.startswith("bold") else "valid")
            source, rows = output_rows("full", benchmark, model, "baseline",
                                       pattern.format(model=model), source_path,
                                       status=status)
            if benchmark.startswith("bold"):
                bold_coverage.append((model, benchmark, *bold_metrics(source, rows)))
                if benchmark == "bold_chat":
                    bold_matched.append((model, *bold_metrics(source, rows, sample600)))

    for benchmark, count in (("crows", 1508), ("stereoset", 2106)):
        _, rows = output_rows("corrected", benchmark, "dream", "aligned_v2",
                              f"exploration/{benchmark}_dream_aligned_v2.jsonl",
                              f"{benchmark}_full.jsonl", aligned=True)
        assert len(rows) == count

    for mode in ("base", "full", "goldfish"):
        for model in ("llama", "llada", "qwen", "dream"):
            legacy = model == "dream"
            output_rows("synthetic_mia", "mia", model, mode,
                        f"mia/{mode}_{model}.jsonl",
                        "synthetic_mia/candidates.jsonl",
                        status="invalid_dream_alignment" if legacy else "valid")
            output_rows("synthetic_mia", "utility", model, mode,
                        f"utility/{mode}_{model}.jsonl",
                        "synthetic_mia/utility_holdout.jsonl",
                        status="invalid_dream_alignment" if legacy else "valid")
        for benchmark, source_path in (("mia", "synthetic_mia/candidates.jsonl"),
                                       ("utility", "synthetic_mia/utility_holdout.jsonl")):
            output_rows("corrected", benchmark, "dream", mode,
                        f"exploration/{benchmark}_{mode}_dream_aligned_v2.jsonl",
                        source_path, aligned=True)

    for model in ("llada", "dream"):
        for condition in ("step48", "long96", "min20", "short12"):
            output_rows("probe", "bold_60", model, condition,
                        f"exploration/bold_{condition}_{model}.jsonl",
                        "bold_explore_60.jsonl", status="legacy_fixed_span_decode")
    for condition in ("eos_raw", "eos_step48", "eos_long96"):
        output_rows("probe", "bold_60", "llada", condition,
                    f"exploration/bold_{condition}_llada.jsonl",
                    "bold_explore_60.jsonl", status="legacy_fixed_span_decode")
    for model in ("llada", "dream"):
        output_rows("probe", "bbq_132", model, "step32",
                    f"exploration/bbq_step32_{model}.jsonl", "bbq_explore_132.jsonl")
    for model in ("llama", "llada", "qwen", "dream"):
        output_rows("probe", "bbq_132", model, "explicit",
                    f"exploration/bbq_explicit_{model}.jsonl", "bbq_explore_132.jsonl")
    for condition in ("step32", "explicit"):
        output_rows("probe", "bbq_1320", "dream", condition,
                    f"exploration/bbq_replicate_{condition}_dream.jsonl",
                    "bbq_replicate_1320.jsonl")
    output_rows("probe", "bbq_1320", "qwen", "explicit",
                "exploration/bbq_replicate_explicit_qwen.jsonl",
                "bbq_replicate_1320.jsonl")
    for condition in ("baseline24_eos", "short12_eos", "steps16_eos", "steps20_eos"):
        _, rows = output_rows("probe", "bold_600", "dream", condition,
                              f"exploration/bold_replicate_{condition}_dream.jsonl",
                              "bold_replicate_600.jsonl")
        if any(not r.get("truncate_eos") for r in rows):
            raise ValueError(f"EOS truncation missing: Dream {condition}")
    for condition in ("step48_stop", "eos_step48_stop"):
        _, rows = output_rows("probe", "bold_600", "llada", condition,
                              f"exploration/bold_replicate_{condition}_llada.jsonl",
                              "bold_replicate_600.jsonl")
        if any(not r.get("truncate_eos") for r in rows):
            raise ValueError(f"EOS truncation missing: LLaDA {condition}")

    write_tsv("walkthrough_bold_coverage.tsv",
              ("model", "mode", "n", "empty", "no_alnum", "at_least_20_words",
               "median_words"), bold_coverage)
    write_tsv("walkthrough_bold_matched_600.tsv",
              ("model", "n", "empty", "no_alnum", "at_least_20_words",
               "median_words"), bold_matched)

    source = {r["id"]: r for r in source_info("bbq_replicate_1320.jsonl")[0]}
    step = {r["id"]: r for r in
            output_rows("audit", "bbq_1320", "dream", "step32",
                        "exploration/bbq_replicate_step32_dream.jsonl",
                        "bbq_replicate_1320.jsonl")[1]}
    explicit = {r["id"]: r for r in
                output_rows("audit", "bbq_1320", "dream", "explicit",
                            "exploration/bbq_replicate_explicit_dream.jsonl",
                            "bbq_replicate_1320.jsonl")[1]}
    category_rows = []
    for category in sorted({r["category"] for r in source.values()}):
        for context in ("ambig", "disambig"):
            ids = [item_id for item_id, row in source.items()
                   if row["category"] == category and row["context_condition"] == context]
            if len(ids) != 60:
                raise ValueError(f"unbalanced replication category: {category}/{context}")
            before = sum(step[i]["prediction"] == source[i]["label"] for i in ids)
            after = sum(explicit[i]["prediction"] == source[i]["label"] for i in ids)
            category_rows.append((category, context, 60, before, after,
                                  (after - before) / 60))
    write_tsv("walkthrough_bbq_replicate_categories.tsv",
              ("category", "context", "n", "original_correct", "explicit_correct",
               "accuracy_change"), category_rows)

    metadata = {}
    with (DATA / "bbq_additional_metadata.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            metadata[(row["category"], row["example_id"])] = row["target_loc"]
    bias_rows = []
    for condition, outputs in (("original_32_steps", step),
                               ("explicit_evidence_rule", explicit)):
        ids = [i for i, r in source.items() if r["context_condition"] == "ambig"]
        mapped = [i for i in ids if metadata[(source[i]["category"],
                                              str(source[i]["source_row"]))] not in ("", "NA")]
        correct = sum(outputs[i]["prediction"] == source[i]["label"] for i in mapped)
        guesses = targets = 0
        for i in mapped:
            prediction = outputs[i]["prediction"]
            unknown = next(j for j in range(3) if
                           source[i]["answer_info"][f"ans{j}"][1].lower() == "unknown")
            if prediction is not None and prediction != unknown:
                guesses += 1
                targets += prediction == int(metadata[(source[i]["category"],
                                                       str(source[i]["source_row"]))])
        accuracy = correct / len(mapped)
        conditional_target = targets / guesses
        bias = (2 * conditional_target - 1) * (1 - accuracy)
        bias_rows.append((condition, len(ids), len(mapped), accuracy,
                          guesses / len(mapped), conditional_target, bias))
    write_tsv("walkthrough_bbq_replicate_bias.tsv",
              ("condition", "n", "mapped_n", "ambiguous_accuracy",
               "guess_fraction", "target_given_guess", "signed_bias_score"), bias_rows)
    write_tsv("walkthrough_output_inventory.tsv",
              ("phase", "benchmark", "model", "condition", "expected_rows",
               "observed_rows", "status", "input_sha256", "output"), inventory)


if __name__ == "__main__":
    main()
