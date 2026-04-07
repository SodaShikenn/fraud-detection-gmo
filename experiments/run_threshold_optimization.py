"""
Threshold Optimisation for Precision Under Recall Constraints

As described in Section 3.1 of the report, GMO requires high precision
(all detected frauds should be true frauds) while maintaining different
minimum recall levels: 10%, 30%, 50%.

Usage:
    python -m experiments.run_threshold_optimization --data data/processed.csv
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.models import load_data, default_params
from src.validation import rolling_splits
from src.evaluation import choose_threshold_precision_at_min_recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--models", nargs="+", default=["lgbm", "xgb", "rf", "cat"])
    parser.add_argument("--min-recalls", nargs="+", type=float, default=[0.10, 0.30, 0.50])
    args = parser.parse_args()

    X_ml, X_cat, y, cat_idx = load_data(args.data)
    splits = rolling_splits(len(X_ml), train_frac=0.5, val_frac=0.1, test_frac=0.1, step_frac=0.05)

    pw = (y == 0).sum() / max((y == 1).sum(), 1)
    all_params = default_params(pw)

    for min_rec in args.min_recalls:
        print(f"\n{'=' * 70}")
        print(f"Minimum Recall = {min_rec * 100:.0f}%")
        print(f"{'=' * 70}")
        print(f"{'Model':<10} {'Precision':>10} {'Recall':>10} {'Threshold':>10}")
        print("-" * 40)

        for name in args.models:
            precs, recs = [], []
            try:
                from src.models import _train
                for tr, va, te in splits:
                    Xtr, ytr = X_ml.iloc[tr], y.iloc[tr]
                    Xva, yva = X_ml.iloc[va], y.iloc[va]
                    Xte, yte = X_ml.iloc[te], y.iloc[te]

                    cat_tr = X_cat.iloc[tr] if X_cat is not None and name == "cat" else None
                    cat_va = X_cat.iloc[va] if X_cat is not None and name == "cat" else None
                    cat_te = X_cat.iloc[te] if X_cat is not None and name == "cat" else None

                    model, pred_fn = _train(
                        name, all_params[name], Xtr, ytr, Xva, yva,
                        cat_tr, cat_va, cat_idx,
                    )
                    vp = pred_fn(cat_va if name == "cat" else Xva)[:, 1]
                    tp = pred_fn(cat_te if name == "cat" else Xte)[:, 1]

                    thr, prec, rec = choose_threshold_precision_at_min_recall(yva.values, vp, min_rec)
                    preds = (tp >= thr).astype(int)
                    from sklearn.metrics import precision_score, recall_score
                    precs.append(precision_score(yte, preds, zero_division=0))
                    recs.append(recall_score(yte, preds, zero_division=0))

                p, r = np.mean(precs), np.mean(recs)
                print(f"{name.upper():<10} {p:>9.1%} {r:>9.1%}")
            except Exception as e:
                print(f"{name.upper():<10} Error: {e}")


if __name__ == "__main__":
    main()
