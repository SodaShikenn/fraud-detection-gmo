"""
Model Training and Walk-Forward Evaluation

Supports four model families evaluated in the report:
  - LightGBM (final model selected)
  - XGBoost
  - CatBoost
  - Random Forest

All models use class weight adjustment to handle the ~10% fraud rate imbalance.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Callable, Optional, Dict, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

from src.evaluation import compute_metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
CAT_COLS = ["f6", "f9", "f10", "f11", "f12", "f13"]


def load_data(
    path: str,
    target: str = "f16",
    drop_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[int]]:
    """
    Load processed CSV and prepare for model training.

    Returns (X_ml, X_cat, y, cat_idx) where:
      - X_ml has label-encoded categoricals (LightGBM, XGBoost, RF)
      - X_cat has string categoricals (CatBoost)
    """
    if drop_cols is None:
        drop_cols = [target, "f14", "f3"]
    if cat_cols is None:
        cat_cols = CAT_COLS

    df = pd.read_csv(path).sort_values("f3").reset_index(drop=True)
    y = df[target].astype(int)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    existing_cats = [c for c in cat_cols if c in X.columns]

    X_cat = X.copy()
    for c in existing_cats:
        X_cat[c] = X_cat[c].astype(str).astype("category")
    cat_idx = [X_cat.columns.get_loc(c) for c in existing_cats]

    X_ml = X.copy()
    for c in existing_cats:
        X_ml[c] = LabelEncoder().fit_transform(X_ml[c].astype(str))

    print(f"Loaded {path}: {X_ml.shape}, positive rate={y.mean():.4f}")
    return X_ml, X_cat, y, cat_idx


# ---------------------------------------------------------------------------
# Hyperparameters (with class weight / balanced handling)
# ---------------------------------------------------------------------------
def default_params(pos_weight: float = 1.0) -> Dict[str, Dict[str, Any]]:
    return {
        "lgbm": dict(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        ),
        "xgb": dict(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            gamma=1.0, scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
        "rf": dict(
            n_estimators=400, min_samples_leaf=100, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "cat": dict(
            iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=10,
            loss_function="Logloss", eval_metric="AUC",
            class_weights=[1.0, float(pos_weight)],
            random_seed=42, verbose=False,
        ),
    }


# ---------------------------------------------------------------------------
# Train a single model
# ---------------------------------------------------------------------------
def _train(name, params, Xtr, ytr, Xva, yva, Xtr_cat=None, Xva_cat=None, cat_idx=None):
    if name == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        return m, m.predict_proba

    if name == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(**params)
        m.fit(Xtr, ytr)
        return m, m.predict_proba

    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(**params)
        m.fit(Xtr, ytr)
        return m, m.predict_proba

    if name == "cat":
        from catboost import CatBoostClassifier, Pool
        m = CatBoostClassifier(**params)
        m.fit(Pool(Xtr_cat or Xtr, ytr, cat_features=cat_idx))

        def _pred(X):
            return m.predict_proba(Pool(X, cat_features=cat_idx))
        return m, _pred

    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Evaluate across walk-forward splits
# ---------------------------------------------------------------------------
def evaluate_on_splits(
    model_name: str,
    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    X_ml: pd.DataFrame,
    y: pd.Series,
    threshold_fn: Callable,
    X_cat: Optional[pd.DataFrame] = None,
    cat_idx: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Walk-forward evaluation: train on each fold, optimise threshold on
    validation, report metrics on test.
    """
    pw = (y == 0).sum() / max((y == 1).sum(), 1)
    params = default_params(pw)[model_name]
    rows = []

    for step, (tr, va, te) in enumerate(splits, 1):
        Xtr, ytr = X_ml.iloc[tr], y.iloc[tr]
        Xva, yva = X_ml.iloc[va], y.iloc[va]
        Xte, yte = X_ml.iloc[te], y.iloc[te]
        cat_tr = X_cat.iloc[tr] if X_cat is not None else None
        cat_va = X_cat.iloc[va] if X_cat is not None else None
        cat_te = X_cat.iloc[te] if X_cat is not None else None

        model, pred_fn = _train(
            model_name, params, Xtr, ytr, Xva, yva,
            cat_tr, cat_va, cat_idx,
        )
        vp = pred_fn(cat_va if model_name == "cat" else Xva)[:, 1]
        tp = pred_fn(cat_te if model_name == "cat" else Xte)[:, 1]

        thr, _ = threshold_fn(yva.values, vp)
        m = compute_metrics(yte.values, tp, thr)
        rows.append({"step": step, "threshold": thr,
                      "val_auc": roc_auc_score(yva, vp), **m})

    res = pd.DataFrame(rows)
    print(f"\n[{model_name.upper()}] {len(splits)} folds | "
          f"AUC {res.auc.mean():.4f} | F1 {res.f1.mean():.4f} | "
          f"Prec {res.precision.mean():.4f} | Rec {res.recall.mean():.4f}")
    return res
