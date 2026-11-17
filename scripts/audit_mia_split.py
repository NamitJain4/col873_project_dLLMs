#!/usr/bin/env python3
"""Check the synthetic MIA split and a blind text-only attack."""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/synthetic_mia/candidates.jsonl"))
    ap.add_argument("--no-classifier", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(s) for s in args.data.read_bytes().splitlines() if s]
    labels = [int(r["member"]) for r in rows]
    texts = [r["text"] for r in rows]
    counts = Counter((r["repetition_group"], r["member"]) for r in rows)
    if len(set(texts)) != len(texts) or len({r["canary"] for r in rows}) != len(rows):
        raise ValueError("candidate text or canary collision")
    print("candidate_count", len(rows), "member_count", sum(labels), "strata", dict(sorted(counts.items())))
    for membership in (0, 1):
        lengths = [len(text) for text, label in zip(texts, labels) if label == membership]
        print("member", membership, "mean_char_length", round(statistics.mean(lengths), 2))
    if not args.no_classifier:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline

        classifier = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                                                   max_features=20000), LogisticRegression(max_iter=1000))
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        scores = cross_val_predict(classifier, texts, labels, cv=folds, method="predict_proba")[:, 1]
        print("blind_text_only_5fold_auroc", round(roc_auc_score(labels, scores), 4))


if __name__ == "__main__":
    main()
