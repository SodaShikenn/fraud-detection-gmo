"""
SHAP Feature Importance Analysis

Generate SHAP summary plots to understand model decision-making.
Key findings from the report:
  - f7 (deposit history) is the most important feature
  - f7xf5 (no-deposit x amount interaction) has very high importance
  - fraud_ratio_store_7d provides temporal risk signals per store

Usage:
    python -m experiments.run_shap_analysis --data data/processed.csv
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from src.validation import temporal_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="shap_summary.png")
    args = parser.parse_args()

    df = pd.read_csv(args.data).sort_values("f3").reset_index(drop=True)
    y = df["f16"].astype(int)
    days = df["f3"].values
    X = df.drop(columns=[c for c in ["f16", "f14", "f3"] if c in df.columns])

    cat_cols = ["f6", "f9", "f10", "f11", "f12", "f13"]
    for c in cat_cols:
        if c in X.columns:
            if X[c].dtype == "object":
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))
            X[c] = X[c].astype("category")

    # Temporal split: Train Day 0-325, Test Day 326-396
    tr_idx, te_idx = temporal_split(len(X), train_end_day=325, days=days)
    X_train, y_train = X.iloc[tr_idx], y.iloc[tr_idx]
    X_test, y_test = X.iloc[te_idx], y.iloc[te_idx]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train fraud rate: {y_train.mean():.4f}, Test fraud rate: {y_test.mean():.4f}")

    model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train)

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_test, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        print(f"SHAP plot saved: {args.output}")
    except ImportError:
        print("shap not installed. Falling back to LightGBM feature importance.")
        imp = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.booster_.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False).head(20)

        plt.figure(figsize=(10, 8))
        plt.barh(imp.feature[::-1], imp.importance[::-1])
        plt.xlabel("Gain")
        plt.title("Top 20 Feature Importance (Gain)")
        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        print(f"Importance plot saved: {args.output}")


if __name__ == "__main__":
    main()
