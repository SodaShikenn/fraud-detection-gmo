# Fraud Detection for E-Commerce Payments

Machine learning pipeline for detecting fraudulent transactions in e-commerce payment data. Developed in the [Advanced Course](https://dss.i.u-tokyo.ac.jp/advance/) of the **University of Tokyo [Data Science School (DSS)](https://dss.i.u-tokyo.ac.jp/)**, 2025, in collaboration with **GMO Payment Gateway**. The Advanced Course involves group-based analysis of real business data provided by partner companies, covering problem discovery, analysis proposal, implementation, and client feedback.

> [Japanese README (main)](README.md)

## Background

GMO Payment Gateway is a Tokyo Stock Exchange Prime-listed payment infrastructure company processing over 16 trillion yen annually (FY2024/09) across 156,575+ merchants. In recent years, online payment fraud amounts have reached new record highs every year, making advanced fraud detection an urgent industry-wide challenge.

Unpaid transactions fall into two patterns: (A) **Malicious fraud** -- no intent to pay from the start, using fake addresses or targeting high-value resalable goods; and (B) **At-risk transactions** -- initial intent to pay exists, but payments eventually stop. Both require operational countermeasures. Fraud patterns are diverse and their timing unpredictable; manual review and experience-based knowledge alone are insufficient, making data-driven fraud detection essential.

Key requirement from GMO: **all detected frauds should be true frauds** (high precision), while maintaining acceptable recall.

**Core technical challenges:**

- **Class imbalance** -- ~10% fraud rate (~6,800 fraud out of ~65,000 transactions). Accuracy is misleadingly high from majority-class prediction alone, so precision and recall are the primary metrics
- **Temporal distribution shift (prior shift)** -- fraud rate changes over time (13% in training period -> 5% in test period)
- **Data leakage prevention** -- all features must use only past data, as in production deployment

## Dataset

> The dataset is confidential and not included in this repository.

~65,000 anonymised e-commerce transactions over 397 days. All fields are masked.

| Column | Content | Description |
|---|---|---|
| f3 | Days from purchase date | Days from reference date (temporal ordering) |
| f4 | Purchase month | Month (0-11) |
| f5 | Purchase amount | Transaction amount (integer) |
| f6 | Device OS | OS type ID |
| f7 | Deposit history | 0=has deposit history (higher trust), 1=none |
| f9 | Store | Store ID |
| f10 | Postal code | Full postal code |
| f11 | Postal code prefix | First 3 digits (regional prefix) |
| f12 | Email domain | Email domain ID |
| f13 | Prefecture code | Prefecture |
| f14 | Address | Used as user identifier |
| **f16** | **Fraud status** | **Target: 0=legitimate, 1=fraud** |

### Temporal Distribution Shift

| Period | Days | Fraud Rate | Count |
|---|---|---|---|
| Early | Day 0-99 | 13.5% | 13,941 |
| Mid | Day 100-199 | 12.5% | 13,544 |
| Late | Day 200-299 | 13.3% | 13,278 |
| Recent | Day 300-396 | **6.0%** | 24,276 |

The fraud rate drops sharply from 13.5% to 6.0%. Naive random splitting risks leaking future data into training.

## Project Structure

```text
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

Starting from only 16 raw features, we engineer 65 additional features in five categories. All features are computed using only past data via expanding windows, cumulative counts, and shifted aggregations -- **completely preventing data leakage**.

### 1. Base Statistics (5)

Compute each user's (address) normal purchase behaviour from past transactions, then quantify deviations.

| Feature | Description |
|---|---|
| `purchase_count_cumulative` | Cumulative purchase count per user |
| `past_mean_amount` | Historical mean purchase amount |
| `past_std_amount` | Historical std of purchase amount |
| `mu_personal` | Log-normal mu (per user) |
| `sigma_personal` | Log-normal sigma (per user) |

### 2. Feature Structure (10)

Frequency counts of categorical values and their co-occurrences.

- Single frequency: `f6_count`, `f9_count`, `f10_count`, `f11_count`, `f12_count`
- Pairwise co-occurrence: `f6xf9_count`, `f6xf12_count`, `f9xf12_count`
- Triple co-occurrence: `f6xf9xf12_count`
- Frequency product: `f9_count_x_f12_count`

### 3. Rule-based (7)

Domain-knowledge-driven features capturing amount digit patterns and category changes.

| Feature | Description |
|---|---|
| `f5_end_with_5`, `f5_end_with_0`, `f5_end_with_00` | Amount digit patterns (round numbers) |
| `f6_changed`, `f9_changed`, `f12_changed` | Change flags for OS, store, email from previous transaction |
| **`f7xf5`** | **Returns purchase amount when f7=0 (no deposit history), else 0. Isolates "high-amount transactions by users with no payment track record"** |

### 4. Feature Enhancement (10)

Dividing amount by log-frequency highlights "unusually high amount for the observed frequency".

`f5_x_{count_col}` = f5 / log1p(count) for every count-type feature.

### 5. Statistical Model (33)

Probability and anomaly-based features:

| Subgroup | Count | Description |
|---|---|---|
| Fraud ratios (4 groups x 3 windows) | 20 | Per-prefecture, store, email domain, and OS fraud rates over past 7d/30d/all-time, with data counts |
| Global amount LLR | 4 | `log_f5`, `f5_loglik_global_y0`, `f5_loglik_global_y1`, `f5_amount_llr_global` |
| Personal amount LLR | 2 | `f5_loglik_personal`, `f5_amount_llr_personal` |
| Inter-transaction interval LLR | 5 | `delta_days`, `log_delta`, `delta_days_loglik_y0`, `delta_days_loglik_y1`, `delta_days_llr` |
| Deviation from past mean | 2 | `amount_ratio_to_past_mean`, `amount_std_score` |

LLR (log-likelihood ratio) quantifies whether a transaction amount or interval is closer to the "legitimate" or "fraud" distribution, assuming log-normal distributions estimated via expanding windows.

## Handling Class Imbalance

All models use **class weight adjustment** (`class_weight="balanced"`). By assigning higher weight to fraudulent transactions, the loss function increases the minority class's influence, preventing degradation of fraud detection performance (especially recall). Unlike oversampling or undersampling, this preserves the original data size and distribution.

## Validation Strategy

Random splitting is inappropriate as it leaks future data. To preserve temporal causality, we employ two walk-forward strategies (`[Train=50%, Val=10%, Test=10%, Step=5%]`, 7 folds):

- **Rolling Window** -- fixed-size training window slides forward. Prioritises adaptation to recent fraud trends. Robust against short-term concept drift.
- **Expanding Window** -- training window grows from the start. Prioritises stability and long-term patterns. More stable when data is sparse.

**Rolling Window showed more stable and reliable performance** across evaluations and was selected as the final strategy.

A simple temporal split (Train: Day 0-325, Test: Day 326-396) confirms the distribution shift: 13% fraud rate (5,842 cases) in training vs. 5% (957 cases) in testing.

## Feature Selection (RFE)

Recursive Feature Elimination using LightGBM gain importance (Algorithm 1):

1. Initialise feature set with all features
2. Train LightGBM on current feature set
3. Record validation AUC; obtain feature importances (gain)
4. Remove the single lowest-importance feature
5. Repeat until 1 feature remains
6. Output the feature set achieving maximum validation AUC

Applied across 7 rolling folds, the optimal feature count **varied from 12 to 70**, reflecting temporal distribution changes. This highlights the importance of fold-independent feature selection.

Final selection: **47 robust features** (selected in >= 4/7 folds). The 10 features selected in all 7 folds:

| Feature | Selected | Description |
|---|---|---|
| `f7` | 7/7 | Deposit history |
| `f9` | 7/7 | Store |
| `f10` | 7/7 | Postal code |
| `f11` | 7/7 | Postal prefix |
| `f12` | 7/7 | Email domain |
| `f7xf5` | 7/7 | No-deposit x amount |
| `f5_x_f6_count` | 7/7 | Amount / log(OS frequency) |
| `fraud_ratio_store_7d` | 7/7 | Store fraud rate (past 7 days) |
| `f5_x_f6xf12_count` | 7/7 | Amount / log(OS x email co-occurrence) |
| `fraud_ratio_email_all` | 7/7 | Email domain fraud rate (all-time) |

## Model Results

Four models compared under rolling window validation:

| Model | Test AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| **LightGBM** | **~0.94** | **~0.54** | - | **~0.64** |
| CatBoost | ~0.94 | ~0.52 | Higher | Lower |
| XGBoost | ~0.93 | ~0.50 | - | - |
| Random Forest | ~0.91 | ~0.48 | - | - |

CatBoost slightly outperformed on AUC and precision, but **LightGBM** was selected as the final model for its superior F1 and recall balance.

### Precision Under Recall Constraints

Following GMO's requirement, precision is maximised under different minimum recall floors:

| Min Recall | Best Model | Precision | Recall |
|---|---|---|---|
| 10% | XGBoost | 74% +/- 9% | 13% +/- 3% |
| 30% | XGBoost | 60% +/- 4% | 33% +/- 4% |
| 50% | Random Forest | 51% +/- 5% | 52% +/- 2% |

For example, at minimum 10% recall, XGBoost achieves 74% precision -- 3 out of 4 flagged transactions are truly fraudulent, making it well-suited for integration with GMO's existing review workflow.

## SHAP Analysis

SHAP (SHapley Additive exPlanations) values quantitatively explain model decision-making.

### Key Findings

1. **`f7` (deposit history)** -- highest overall contribution. `f7=0` (has deposit history) pushes toward fraud; `f7=1` (no history) pushes toward legitimate. Seemingly counterintuitive, but coherent when combined with the interaction term.

2. **`f7xf5` (no-deposit x amount)** -- the most important finding of this study. Feature design:
   - `f7=1` (no deposit history): `f7xf5 = purchase amount`
   - `f7=0` (has deposit history): `f7xf5 = 0`

   This explicitly isolates "**high-amount transactions with no payment track record**". SHAP confirms extremely strong fraud-direction contribution in this case. This demonstrates that "high amount = always fraud" is too simplistic; proper feature design can capture the underlying fraud structure.

3. **`fraud_ratio_store_7d` (store fraud rate, past 7 days)** -- transactions at stores with recent fraud clusters push toward fraud; low-fraud-rate stores push toward legitimate. Temporal risk signals at the store level serve as effective auxiliary features.

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
