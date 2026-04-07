"""
Recursive Feature Elimination

Run RFE across 7 rolling splits and select robust features (>50% folds).
Implements Algorithm 1 and Section 2.5.2 of the report.

Usage:
    python -m experiments.run_rfe --data data/processed.csv
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.validation import rolling_splits
from src.rfe import run_rfe


def load_data(path):
    df = pd.read_csv(path).sort_values("f3").reset_index(drop=True)
    y = df["f16"].astype(int)
    X = df.drop(columns=[c for c in ["f16", "f14", "f3"] if c in df.columns])
    cat_cols = ["f6", "f9", "f10", "f11", "f12", "f13"]
    for c in cat_cols:
        if c in X.columns:
            if X[c].dtype == "object":
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))
            X[c] = X[c].astype("category")
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--min-features", type=int, default=1)
    args = parser.parse_args()

    X, y = load_data(args.data)
    splits = rolling_splits(len(X), train_frac=0.5, val_frac=0.1, test_frac=0.1, step_frac=0.05)
    print(f"Data: {X.shape}, {len(splits)} rolling splits\n")

    results, robust = run_rfe(
        X, y, splits,
        n_estimators=2000, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", random_state=42, n_jobs=-1,
        step=args.step, min_features=args.min_features,
    )

    print(f"\n{'=' * 60}")
    for _, row in results.iterrows():
        print(f"Split {row['split']}: {row['best_n_features']} features, "
              f"val AUC={row['best_val_auc']:.4f}", end="")
        if "test_auc" in row and pd.notna(row.get("test_auc")):
            print(f", test AUC={row['test_auc']:.4f}")
        else:
            print()

    print(f"\nAvg best features: {results['best_n_features'].mean():.1f}")
    print(f"\nRobust features ({len(robust)}):")
    for f in robust:
        print(f"  {f}")

    results.drop(columns=["features"], errors="ignore").to_csv("rfe_results.csv", index=False)


if __name__ == "__main__":
    main()
