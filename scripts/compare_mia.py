#!/usr/bin/env python3
"""Paired bootstrap of Goldfish versus full-training leakage on fixed candidates."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def auc(member_scores, nonmember_scores):
    wins = member_scores[:, None] > nonmember_scores[None, :]
    ties = member_scores[:, None] == nonmember_scores[None, :]
    return float(wins.mean() + 0.5 * ties.mean())


print("model\tgroup\tmembers\texact_full\texact_goldfish\texact_diff\texact_diff_ci_low\texact_diff_ci_high\tauc_full\tauc_goldfish\tauc_diff\tauc_diff_ci_low\tauc_diff_ci_high")
for model in ("qwen", "llama", "llada", "dream"):
    runs = {}
    for mode in ("full", "goldfish"):
        path = ROOT / "results/mia" / f"{mode}_{model}.jsonl"
        rows = [json.loads(s) for s in path.read_bytes().splitlines() if s]
        runs[mode] = {r["id"]: r for r in rows}
        if len(runs[mode]) != len(rows):
            raise ValueError(f"duplicate ids: {path}")
    if set(runs["full"]) != set(runs["goldfish"]):
        raise ValueError(f"candidate mismatch for {model}")
    for group in ("ALL", 1, 4, 16):
        ids = sorted(i for i, r in runs["full"].items() if group == "ALL" or r["repetition_group"] == group)
        member_ids = [i for i in ids if runs["full"][i]["member"]]
        nonmember_ids = [i for i in ids if not runs["full"][i]["member"]]
        if any(runs["full"][i]["member"] != runs["goldfish"][i]["member"] or
               runs["full"][i]["canary"] != runs["goldfish"][i]["canary"] for i in ids):
            raise ValueError(f"candidate labels changed for {model}")
        full_m = np.array([runs["full"][i]["response_score"] for i in member_ids])
        gold_m = np.array([runs["goldfish"][i]["response_score"] for i in member_ids])
        full_n = np.array([runs["full"][i]["response_score"] for i in nonmember_ids])
        gold_n = np.array([runs["goldfish"][i]["response_score"] for i in nonmember_ids])
        extract_full = np.array([runs["full"][i]["exact_extraction"] for i in member_ids], dtype=float)
        extract_gold = np.array([runs["goldfish"][i]["exact_extraction"] for i in member_ids], dtype=float)
        rng = np.random.default_rng(42)
        extract_diffs, auc_diffs = [], []
        for _ in range(1000):
            mi = rng.integers(len(member_ids), size=len(member_ids))
            ni = rng.integers(len(nonmember_ids), size=len(nonmember_ids))
            extract_diffs.append(float((extract_gold[mi] - extract_full[mi]).mean()))
            auc_diffs.append(auc(gold_m[mi], gold_n[ni]) - auc(full_m[mi], full_n[ni]))
        el, eh = np.quantile(extract_diffs, [0.025, 0.975])
        al, ah = np.quantile(auc_diffs, [0.025, 0.975])
        f_auc, g_auc = auc(full_m, full_n), auc(gold_m, gold_n)
        print(f"{model}\t{group}\t{len(member_ids)}\t{extract_full.mean():.4f}\t{extract_gold.mean():.4f}\t"
              f"{(extract_gold-extract_full).mean():.4f}\t{el:.4f}\t{eh:.4f}\t"
              f"{f_auc:.4f}\t{g_auc:.4f}\t{g_auc-f_auc:.4f}\t{al:.4f}\t{ah:.4f}")
