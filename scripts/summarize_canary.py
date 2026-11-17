#!/usr/bin/env python3
"""Aggregate controlled canary extraction and response-score MIA AUROC."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    import numpy as np

    print("file\tgroup\tn\tmembers\texact_member\texact_nonmember\tauc_response_score\tauc_ci_low\tauc_ci_high")
    for path in args.files:
        rows = [json.loads(s) for s in path.read_bytes().splitlines() if s]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"duplicate ids: {path}")
        for group in ("ALL", 1, 4, 16):
            subset = rows if group == "ALL" else [r for r in rows if r["repetition_group"] == group]
            if not subset:
                continue
            members = [r for r in subset if r["member"]]
            nonmembers = [r for r in subset if not r["member"]]
            auc = roc_auc_score([r["member"] for r in subset], [r["response_score"] for r in subset]) if members and nonmembers else float("nan")
            if members and nonmembers:
                m = np.array([r["response_score"] for r in members])
                n = np.array([r["response_score"] for r in nonmembers])
                rng = np.random.default_rng(42)
                draws = []
                for _ in range(1000):
                    sampled_m = rng.choice(m, len(m), replace=True)
                    sampled_n = rng.choice(n, len(n), replace=True)
                    wins = sampled_m[:, None] > sampled_n[None, :]
                    ties = sampled_m[:, None] == sampled_n[None, :]
                    draws.append(float(wins.mean() + 0.5 * ties.mean()))
                ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
            else:
                ci_low = ci_high = float("nan")
            print(f"{path}\t{group}\t{len(subset)}\t{len(members)}\t"
                  f"{sum(r['exact_extraction'] for r in members)/len(members) if members else float('nan'):.4f}\t"
                  f"{sum(r['exact_extraction'] for r in nonmembers)/len(nonmembers) if nonmembers else float('nan'):.4f}\t"
                  f"{auc:.4f}\t{ci_low:.4f}\t{ci_high:.4f}")


if __name__ == "__main__":
    main()
