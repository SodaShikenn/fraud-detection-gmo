"""
Evaluation Metrics and Threshold Optimisation

Provides:
  - F1-based threshold selection
  - Precision-constrained recall maximisation (minimum recall thresholds)
  - Standard classification metrics for fraud detection
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    confusion_matrix,
)
from typing import Tuple, Dict


# ---------------------------------------------------------------------------
# Threshold selection strategies
# ---------------------------------------------------------------------------

def choose_threshold_f1(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """Find threshold maximising F1 on the precision-recall curve."""
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1 = np.divide(2 * prec * rec, prec + rec,
                    out=np.zeros_like(prec), where=(prec + rec) != 0)
    best = np.argmax(f1)
    return (thr[best] if best < len(thr) else thr[-1]), f1[best]


def choose_threshold_precision_at_min_recall(
    y_true: np.ndarray, y_proba: np.ndarray, min_recall: float = 0.10
) -> Tuple[float, float, float]:
    """
    Maximise precision subject to a minimum recall constraint.

    As described in Section 3.1 of the report, GMO requires that detected
    frauds should be true frauds (high precision), while maintaining at
    least a minimum fraction of all frauds detected (recall floor).

    Returns (threshold, precision, recall).
    """
    best_thr, best_prec, best_rec = 0.5, 0.0, 0.0
    for thr in np.arange(0.01, 0.99, 0.01):
        pred = (y_proba >= thr).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        if rec >= min_recall and prec > best_prec:
            best_prec = prec
            best_rec = rec
            best_thr = thr
    return best_thr, best_prec, best_rec


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_proba, threshold=0.5) -> Dict[str, float]:
    """Compute classification metrics at a given threshold."""
    pred = (y_proba >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "threshold": threshold,
    }


def print_report(y_true, y_proba, threshold, name="Model") -> Dict[str, float]:
    """Print evaluation report with confusion matrix."""
    m = compute_metrics(y_true, y_proba, threshold)
    pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    print(f"\n{'=' * 60}")
    print(f"{name}  (threshold={threshold:.3f})")
    print(f"{'=' * 60}")
    print(f"  ROC-AUC:   {m['auc']:.4f}")
    print(f"  PR-AUC:    {m['pr_auc']:.4f}")
    print(f"  F1:        {m['f1']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall:    {m['recall']:.4f}")
    print(f"  Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Detected:  {tp}/{tp + fn} frauds ({m['recall'] * 100:.1f}%)")
    return m
