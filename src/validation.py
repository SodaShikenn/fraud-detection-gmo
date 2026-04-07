"""
Walk-Forward Time-Series Validation

Implements two splitting strategies described in the report:
  - Rolling window:   fixed-size training window slides forward
  - Expanding window:  training window grows from the start

Both prevent data leakage by strictly using only past data for training.
"""

import numpy as np
from typing import List, Tuple


def rolling_splits(
    n: int,
    train_frac: float = 0.5,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    step_frac: float = 0.05,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Rolling window strategy: training window has fixed size and slides forward.

    Example with 7 splits:
      Step 1: Train [0..50%],     Val [50..60%], Test [60..70%]
      Step 2: Train [5..55%],     Val [55..65%], Test [65..75%]
      ...
    """
    train_sz = int(n * train_frac)
    val_sz = int(n * val_frac)
    test_sz = int(n * test_frac)
    step_sz = int(n * step_frac)

    splits = []
    start = 0
    while start + train_sz + val_sz + test_sz <= n:
        tr = np.arange(start, start + train_sz)
        va = np.arange(start + train_sz, start + train_sz + val_sz)
        te = np.arange(start + train_sz + val_sz, start + train_sz + val_sz + test_sz)
        splits.append((tr, va, te))
        start += step_sz
    return splits


def expanding_splits(
    n: int,
    train_frac: float = 0.5,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    step_frac: float = 0.05,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Expanding window strategy: training starts at index 0 and grows each step.

    Example:
      Step 1: Train [0..50%],     Val [50..60%], Test [60..70%]
      Step 2: Train [0..55%],     Val [55..65%], Test [65..75%]
      ...
    """
    train_sz = int(n * train_frac)
    val_sz = int(n * val_frac)
    test_sz = int(n * test_frac)
    step_sz = int(n * step_frac)

    splits = []
    start = 0
    while start + train_sz + val_sz + test_sz <= n:
        tr = np.arange(0, start + train_sz)
        va = np.arange(start + train_sz, start + train_sz + val_sz)
        te = np.arange(start + train_sz + val_sz, start + train_sz + val_sz + test_sz)
        splits.append((tr, va, te))
        start += step_sz
    return splits


def temporal_split(
    n: int,
    train_end_day: int = 325,
    days: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple temporal split: Train on Day 0-325, Test on Day 326-396.

    Parameters
    ----------
    n : int
        Total number of samples.
    train_end_day : int
        Last day (inclusive) in the training set.
    days : array
        Array of day values (column f3) for each sample.

    Returns
    -------
    (train_idx, test_idx)
    """
    if days is None:
        raise ValueError("days array is required for temporal_split")
    train_idx = np.where(days <= train_end_day)[0]
    test_idx = np.where(days > train_end_day)[0]
    return train_idx, test_idx
