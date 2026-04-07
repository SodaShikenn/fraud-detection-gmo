"""
Feature Engineering Pipeline

Build 65 engineered features from raw transaction data.

Usage:
    python -m experiments.run_preprocessing --input data/DSS2025.csv --output data/processed.csv
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.preprocessing import build_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Raw data: {df.shape}")

    df = build_features(df)
    print(f"After feature engineering: {df.shape}")

    y = df["f16"].astype(int)
    print(f"Fraud rate: {y.mean():.4f} ({y.sum()}/{len(y)})")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
