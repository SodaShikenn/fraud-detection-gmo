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

~65,000 anonymised e-commerce transactions over 397 days. All columns are masked (f3, f4, ...) and consist of temporal, numerical, categorical, and binary features, plus the target variable (fraud flag).

### Temporal Distribution Shift

The figure below shows transaction volume (bar chart) and fraud rate (line chart) over time. A substantial shift in fraud rate between the training period (red) and test period (green) is clearly visible.

<p align="center">
  <img src="docs/figures/temporal_split.png" width="700" alt="Temporal Split: Transaction Volume and Fraud Dynamics"/>
</p>

| Period | Days | Fraud Rate | Count |
|---|---|---|---|
| Early | Day 0-99 | 13.5% | 13,941 |
| Mid | Day 100-199 | 12.5% | 13,544 |
| Late | Day 200-299 | 13.3% | 13,278 |
| Recent | Day 300-396 | **6.0%** | 24,276 |

The fraud rate drops sharply from 13.5% to 6.0%. With such a prior shift, naive random splitting risks leaking future data into training, leading to overly optimistic evaluation. A validation strategy that strictly preserves temporal causality is therefore essential.

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

Compute each user's "normal state" from past transactions, then quantify deviations. Specifically, we calculate per-user cumulative transaction count, historical mean and standard deviation of the numerical feature, and log-normal distribution parameters (mu, sigma) estimated via expanding windows. This models each user's historical transaction patterns at the individual level.

### 2. Feature Structure (10)

Frequency counts of categorical features and their co-occurrences, computed via cumcount. Single frequency (5), pairwise co-occurrence (3), triple co-occurrence (1), and frequency product (1). Fraudulent transactions tend to involve rare category combinations, and frequency information captures this signal.

### 3. Rule-based (7)

Domain-knowledge-driven features: numerical value digit patterns (3 features for round-number detection), category change flags from the previous transaction (3), and a **binary x numerical interaction term** (1).

The interaction term is a key design in this study. It explicitly isolates "high-value transactions by users with a specific attribute": when the binary feature = 1, the interaction term returns the numerical feature's value; when = 0, it returns 0. This ensures the numerical magnitude only contributes to fraud detection conditionally.

### 4. Feature Enhancement (10)

Dividing the numerical feature by log-frequency highlights "unusually high values for the observed frequency". For example, if a very rare category combination has a high numerical value, this feature takes a large value. Computed for every count-type feature.

### 5. Statistical Model (33)

Probability and anomaly-based features:

| Subgroup | Count | Description |
|---|---|---|
| Per-category fraud ratios (4 groups x 3 windows) | 20 | Per-category fraud rates over past 7d/30d/all-time, with data counts |
| Global LLR | 4 | Log-likelihood ratio for numerical feature (population distribution) |
| Personal LLR | 2 | Log-likelihood ratio for numerical feature (per-user distribution) |
| Inter-transaction interval LLR | 5 | Log-likelihood ratio for transaction intervals |
| Deviation from past mean | 2 | Ratio and standardised score vs. user's historical mean |

**LLR (log-likelihood ratio)** quantifies whether an observed value is closer to the "legitimate" or "fraud" distribution. Assuming log-normal distributions with parameters estimated via expanding windows: LLR > 0 indicates the observation is closer to the fraud distribution, LLR < 0 indicates closer to the legitimate distribution. The same approach is applied to both the numerical feature and inter-transaction intervals, capturing temporal anomaly patterns.

## Handling Class Imbalance

Fraudulent transactions account for only ~10% of all transactions. In most learning algorithms, the majority class dominates the loss function, leading to insufficient learning of the minority class. To address this, all models use **class weight adjustment** (`class_weight="balanced"`). By assigning higher weight to fraudulent transactions, the loss function increases the minority class's influence, preventing degradation of fraud detection performance (especially recall). Unlike oversampling or undersampling, this preserves the original data size and distribution.

## Validation Strategy

Fraud detection is inherently time-dependent: models are trained on past data and must predict future data. Random splitting is inappropriate as it leaks future data. To preserve temporal causality, we employ two walk-forward strategies (`[Train=50%, Val=10%, Test=10%, Step=5%]`, 7 folds).

### Rolling Window

A fixed-size training window slides forward, discarding older data. Prioritises adaptation to recent fraud trends and is robust against short-term concept drift.

<p align="center">
  <img src="docs/figures/rolling_window.png" width="600" alt="Rolling Window Validation"/>
</p>

- Advantages: fast adaptation to recent changes, eliminates influence of outdated patterns
- Limitations: discards long-term historical information, may fluctuate when data is sparse

### Expanding Window

The training window grows from the start, accumulating all past data. Prioritises stability and long-term patterns, more robust when data is sparse.

<p align="center">
  <img src="docs/figures/expanding_window.png" width="600" alt="Expanding Window Validation"/>
</p>

- Advantages: stable long-term pattern learning, more training data
- Limitations: slower adaptation to distribution shifts

### Evaluation Results

Both strategies were evaluated with the same 4 models (LightGBM, XGBoost, CatBoost, Random Forest). **Rolling Window showed more stable and reliable performance** and was selected as the final strategy.

<p align="center">
  <img src="docs/figures/table_rolling.png" width="600" alt="Rolling Window Results"/>
</p>
<p align="center">Average test performance under rolling window validation</p>

<p align="center">
  <img src="docs/figures/table_expanding.png" width="600" alt="Expanding Window Results"/>
</p>
<p align="center">Average test performance under expanding window validation</p>

A simple temporal split (Train: Day 0-325, Test: Day 326-396) confirms the distribution shift: 13% fraud rate (5,842 cases) in training vs. 5% (957 cases) in testing.

## Feature Selection (RFE)

To improve generalisation and model interpretability by removing redundant features, we apply Recursive Feature Elimination using LightGBM gain importance.

**Algorithm:**

1. Initialise feature set with all features
2. Train LightGBM on current feature set
3. Record validation AUC; obtain feature importances (gain)
4. Remove the single lowest-importance feature
5. Repeat until 1 feature remains
6. Output the feature set achieving maximum validation AUC

Applied across 7 rolling folds, the optimal feature count **varied from 12 to 70** (see figure below). This reflects temporal distribution changes and highlights the importance of fold-independent feature selection.

<p align="center">
  <img src="docs/figures/rfe_validation_auc.png" width="600" alt="RFE: Validation AUC vs Feature Count"/>
</p>
<p align="center">Validation AUC vs. remaining feature count. Red dots indicate maximum AUC per fold</p>

Final selection: **47 robust features** (selected in >= 4/7 folds). The 10 features selected in all 7 folds consist of raw categorical features (5), the interaction term (1), feature enhancements (2), and per-category fraud ratios (2).

## Model Results

Four models compared under rolling window validation. LightGBM achieves AUC ~0.94, F1 ~0.54, and recall ~0.64.

| Model | Test AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| **LightGBM** | **~0.94** | **~0.54** | ~0.48 | **~0.64** |
| CatBoost | ~0.95 | ~0.55 | ~0.50 | ~0.62 |
| XGBoost | ~0.94 | ~0.53 | ~0.49 | ~0.61 |
| Random Forest | ~0.92 | ~0.53 | ~0.46 | ~0.66 |

CatBoost slightly outperformed on AUC and precision, but **LightGBM with Rolling Window** was selected as the final model for its F1/recall balance and stability across folds.

### Precision Under Recall Constraints

Following GMO's requirement ("all detected frauds should be true frauds"), detection thresholds are tuned to maximise precision under different minimum recall floors. Note that 100% precision is trivially achievable at 0% recall, hence the minimum recall constraint is essential for practical use.

| Min Recall | Best Model | Precision | Recall |
|---|---|---|---|
| 10% | XGBoost | 74% +/- 9% | 13% +/- 3% |
| 30% | XGBoost | 60% +/- 4% | 33% +/- 4% |
| 50% | Random Forest | 51% +/- 5% | 52% +/- 2% |

For example, at minimum 10% recall, XGBoost achieves 74% precision -- 3 out of 4 flagged transactions are truly fraudulent, while detecting 13% of all fraud. This is well-suited for integration with GMO's existing review workflow as a high-precision additional screening layer.

## SHAP Analysis

SHAP (SHapley Additive exPlanations) values quantitatively explain how the model uses each feature for its decisions. The figure below is a SHAP summary plot (stacked view) across all test data, simultaneously visualising the magnitude and direction of each feature's impact on the model output (fraud probability).

<p align="center">
  <img src="docs/figures/shap_summary.png" width="500" alt="SHAP Summary Plot"/>
</p>
<p align="center">SHAP summary plot. X-axis: SHAP value (contribution to model output). Colour: feature value magnitude</p>

### Key Findings

1. **Binary feature** -- highest overall contribution. Users with one particular value of the binary feature show a tendency toward fraud. The direction seems counterintuitive on its own, but becomes coherent when considered alongside the interaction term below.

2. **Binary x numerical interaction term** -- the most important finding of this study. This feature explicitly isolates "high-value transactions by users with a specific attribute":
   - Binary feature = 1: interaction term = numerical feature value
   - Binary feature = 0: interaction term = 0

   SHAP confirms extremely strong fraud-direction contribution for this conditional high-value case. This demonstrates that "high value = always fraud" is too simplistic; proper conditional feature design can capture the underlying fraud structure. When the binary feature = 0, the interaction term is always 0, suppressing the numerical feature's influence on the model's decision.

3. **Per-category fraud rate (past 7 days)** -- transactions in categories with recent fraud clusters push toward fraud; low-fraud-rate categories push toward legitimate. Temporal risk signals at the category level serve as effective auxiliary features. Since this feature computes historical fraud ratios via expanding windows, it captures temporal risk dynamics without data leakage.

### Discussion

The analysis reveals that in this fraud detection model, the **interaction between a binary feature and a numerical feature** is far more important than any single feature's magnitude. The high importance of the interaction term suggests that "attribute-based feature design" is effective for improving fraud detection performance. The key insight is that fraud structures that cannot be captured by simple rules (e.g., high value = fraud) can be extracted through appropriate feature engineering combined with machine learning models.

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
