# Fraud Detection for E-Commerce Payments

Machine learning pipeline for detecting fraudulent transactions in e-commerce payment data. Developed at the **University of Tokyo, Data Scientist Seminar (DSS) 2025** in collaboration with **GMO Payment Gateway**.

## Problem

GMO Payment Gateway processes ~16 trillion yen annually across 156,575+ merchants. Detecting fraudulent transactions early prevents financial losses, but false positives create friction for legitimate customers. The key challenge from GMO: **all detected frauds should be true frauds** (high precision), while maintaining acceptable recall.

**Core challenges:**
- **Class imbalance** -- ~10% fraud rate (~6,800 fraud out of ~65,000 transactions)
- **Temporal distribution shift** -- fraud rate drops from 13% to 5% between train/test periods
- **Data leakage prevention** -- all features must use only past data

## Dataset

> The dataset is confidential and not included in this repository.

- ~65,000 anonymised e-commerce transactions over 397 days
- 16 raw features (masked): temporal, amount, device OS, store, geographic, email domain, address
- Binary target: legitimate (0) vs. fraud (1)

## Project Structure

```
src/
  preprocessing.py    -- Feature engineering (65 features in 5 categories)
  validation.py       -- Rolling & expanding walk-forward splits
  models.py           -- LightGBM, XGBoost, CatBoost, Random Forest
  rfe.py              -- Recursive Feature Elimination (Algorithm 1)
  evaluation.py       -- Threshold optimisation, metrics

experiments/
  run_preprocessing.py          -- Raw data -> 65 features
  run_model_comparison.py       -- 4-model comparison (rolling/expanding)
  run_rfe.py                    -- Feature selection -> 47 robust features
  run_threshold_optimization.py -- Precision under minimum recall constraints
  run_shap_analysis.py          -- SHAP importance analysis
```

## Feature Engineering (65 features)

Starting from 16 raw features, we engineer 65 additional features in five categories:

| Category | Count | Key Features |
|---|---|---|
| **Base Statistics** | 5 | `purchase_count_cumulative`, `past_mean_amount`, `past_std_amount`, `mu_personal`, `sigma_personal` |
| **Feature Structure** | 10 | Category frequency counts (`f9_count`, `f12_count`, ...) and co-occurrence (`f6xf9_count`, `f6xf9xf12_count`, ...) |
| **Rule-based** | 7 | Amount digit patterns (`f5_end_with_00`), category changes (`f6_changed`), **`f7xf5`** (no-deposit x amount) |
| **Feature Enhancement** | 10 | `f5_x_{count}` = amount / log(frequency) for each count feature |
| **Statistical Model** | 33 | Fraud ratios per group per window, log-normal LLR for amounts/intervals, relative amount scores |

All features are computed using only past data via expanding windows, cumulative counts, and shifted aggregations.

## Validation Strategy

Two walk-forward strategies, both with `[Train=50%, Val=10%, Test=10%, Step=5%]`:

- **Rolling window** -- fixed-size training window slides forward (7 folds). Prioritises adaptation to recent fraud trends.
- **Expanding window** -- training window grows from start (7 folds). Prioritises stability and long-term patterns.

Additionally, a simple temporal split (Train: Day 0-325, Test: Day 326-396) reveals the distribution shift: 13% fraud in training, 5% in testing.

## Feature Selection (RFE)

Recursive Feature Elimination using LightGBM gain importance:
1. Start with all features
2. Train LightGBM, remove lowest-importance feature, record validation AUC
3. Repeat until 1 feature remains
4. Select feature set with highest validation AUC

Applied across 7 rolling folds; **robust features** (selected in >= 4/7 folds) yield **47 final features**. Key robust features include `f7` (deposit history), `f9`/`f10`/`f11`/`f12` (categorical), `f7xf5` (interaction), and `fraud_ratio_store_7d`.

## Model Results

| Model | Test AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| **LightGBM** | **~0.94** | **~0.54** | - | **~0.64** |
| CatBoost | ~0.94 | ~0.52 | Higher | Lower |
| XGBoost | ~0.93 | ~0.50 | - | - |
| Random Forest | ~0.91 | ~0.48 | - | - |

**LightGBM with rolling window** was selected as the final model due to the best F1/recall balance.

### Precision Under Recall Constraints

| Min Recall | Best Model | Precision | Recall |
|---|---|---|---|
| 10% | XGBoost | 74% | 13% |
| 30% | XGBoost | 60% | 33% |
| 50% | Random Forest | 51% | 52% |

## SHAP Analysis

Key findings from SHAP feature importance:
1. **`f7` (deposit history)** -- most important feature; no deposit history strongly indicates fraud
2. **`f7xf5` (no-deposit x amount)** -- interaction term with very high importance; "new user making high-amount purchase" is the strongest fraud signal
3. **`fraud_ratio_store_7d`** -- stores with recent fraud activity provide temporal risk signals

## Usage

```bash
pip install -r requirements.txt

# 1. Feature engineering
python -m experiments.run_preprocessing --input data/raw.csv --output data/processed.csv

# 2. Model comparison (rolling or expanding window)
python -m experiments.run_model_comparison --data data/processed.csv --strategy rolling

# 3. Feature selection via RFE
python -m experiments.run_rfe --data data/processed.csv

# 4. Precision-recall threshold analysis
python -m experiments.run_threshold_optimization --data data/processed.csv

# 5. SHAP analysis
python -m experiments.run_shap_analysis --data data/processed.csv
```

## License

Developed for academic purposes at the University of Tokyo.
