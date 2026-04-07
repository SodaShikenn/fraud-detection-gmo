"""
Model Comparison with Walk-Forward Validation

Evaluate LightGBM, XGBoost, CatBoost, and Random Forest under both
rolling and expanding window strategies.

Usage:
    python -m experiments.run_model_comparison --data data/processed.csv
    python -m experiments.run_model_comparison --data data/processed.csv --strategy expanding
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import load_data, evaluate_on_splits
from src.validation import rolling_splits, expanding_splits
from src.evaluation import choose_threshold_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--models", nargs="+", default=["lgbm", "xgb", "rf", "cat"])
    parser.add_argument("--strategy", default="rolling", choices=["rolling", "expanding"])
    args = parser.parse_args()

    X_ml, X_cat, y, cat_idx = load_data(args.data)

    split_fn = rolling_splits if args.strategy == "rolling" else expanding_splits
    splits = split_fn(len(X_ml), train_frac=0.5, val_frac=0.1, test_frac=0.1, step_frac=0.05)
    print(f"Strategy: {args.strategy}, {len(splits)} folds\n")

    results = {}
    for name in args.models:
        try:
            res = evaluate_on_splits(
                name, splits, X_ml, y, choose_threshold_f1,
                X_cat=X_cat if name == "cat" else None,
                cat_idx=cat_idx if name == "cat" else None,
            )
            results[name] = res
        except Exception as e:
            print(f"Error ({name}): {e}")

    print(f"\n{'=' * 70}")
    print(f"COMPARISON ({args.strategy} window)")
    print(f"{'=' * 70}")
    print(f"{'Model':<10} {'AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name.upper():<10} {r.auc.mean():>10.4f} {r.f1.mean():>10.4f} "
              f"{r.precision.mean():>10.4f} {r.recall.mean():>10.4f}")


if __name__ == "__main__":
    main()
