"""
Recursive Feature Elimination (RFE) with LightGBM

Implements Algorithm 1 from the report:
  1. Initialise feature set with all features
  2. Until only 1 feature remains:
     a. Train LightGBM on current features
     b. Get feature importance (gain)
     c. Compute validation AUC
     d. Remove the least important feature
  3. Return the feature set with highest validation AUC

Applied across 7 rolling splits; robust features are those selected
in >50% (>=4) of the folds, yielding the final 47 features.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import roc_auc_score

from src.evaluation import choose_threshold_f1


class RFE_LGBM:
    """Single-fold RFE using LightGBM gain importance."""

    def __init__(
        self,
        n_estimators=2000, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", random_state=42, n_jobs=-1,
        step=1, min_features=1, verbose=True,
    ):
        self.lgbm_kw = dict(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, class_weight=class_weight,
            random_state=random_state, n_jobs=n_jobs, verbose=-1,
        )
        self.step = step
        self.min_features = min_features
        self.verbose = verbose

        self.best_features_: Optional[List[str]] = None
        self.best_score_ = -np.inf
        self.best_n_features_ = 0
        self.history_: List[dict] = []

    def fit(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
        feats = list(X_train.columns)
        it = 0

        while len(feats) >= self.min_features:
            it += 1
            m = lgb.LGBMClassifier(**self.lgbm_kw)
            m.fit(X_train[feats], y_train,
                  eval_set=[(X_val[feats], y_val)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(0)])

            vp = m.predict_proba(X_val[feats])[:, 1]
            auc = roc_auc_score(y_val, vp)

            if auc > self.best_score_:
                self.best_score_ = auc
                self.best_features_ = feats.copy()
                self.best_n_features_ = len(feats)

            self.history_.append({"iteration": it, "n_features": len(feats), "val_auc": auc})

            if self.verbose and it % 5 == 0:
                print(f"  iter {it}: {len(feats):>3} feats, AUC={auc:.5f}")

            if len(feats) <= self.min_features:
                break

            imp = pd.Series(m.booster_.feature_importance(importance_type="gain"), index=feats)
            n_rm = min(max(self.step, 2) if len(feats) > 50 else self.step,
                       len(feats) - self.min_features)
            drop = imp.nsmallest(n_rm).index.tolist()
            feats = [f for f in feats if f not in drop]

        # Test evaluation with best feature set
        self.test_metrics_ = {}
        if X_test is not None and y_test is not None:
            m = lgb.LGBMClassifier(**self.lgbm_kw)
            m.fit(X_train[self.best_features_], y_train,
                  eval_set=[(X_val[self.best_features_], y_val)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            tp = m.predict_proba(X_test[self.best_features_])[:, 1]
            thr, _ = choose_threshold_f1(y_val, m.predict_proba(X_val[self.best_features_])[:, 1])
            pred = (tp >= thr).astype(int)
            from sklearn.metrics import f1_score, precision_score, recall_score
            self.test_metrics_ = {
                "test_auc": roc_auc_score(y_test, tp),
                "test_f1": f1_score(y_test, pred),
                "test_precision": precision_score(y_test, pred, zero_division=0),
                "test_recall": recall_score(y_test, pred, zero_division=0),
            }
        return self


def run_rfe(X, y, splits, **rfe_kw) -> Tuple[pd.DataFrame, List[str]]:
    """
    Run RFE across rolling splits and select robust features.

    Robust features = selected in >50% of folds (as described in Sec 2.5.2).

    Returns (results_df, robust_features).
    """
    rows = []
    all_selected = []

    for i, (tr, va, te) in enumerate(splits, 1):
        print(f"\n--- RFE Split {i}/{len(splits)} ---")
        rfe = RFE_LGBM(**rfe_kw)
        rfe.fit(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], X.iloc[te], y.iloc[te])
        all_selected.extend(rfe.best_features_)
        rows.append({
            "split": i,
            "best_n_features": rfe.best_n_features_,
            "best_val_auc": rfe.best_score_,
            **rfe.test_metrics_,
            "features": rfe.best_features_,
        })

    results = pd.DataFrame(rows)

    # Robust feature selection: >50% of folds
    counts = Counter(all_selected)
    threshold = len(splits) / 2
    robust = sorted([f for f, c in counts.items() if c > threshold],
                    key=lambda f: -counts[f])

    print(f"\nRobust features (>{len(splits)//2} folds): {len(robust)}")
    return results, robust
