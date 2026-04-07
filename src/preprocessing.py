"""
Feature Engineering Pipeline (65 features)

Implements the five feature categories described in the report:
  1. Base Statistics        (5 features)  - per-user historical amount statistics
  2. Feature Structure     (10 features)  - category frequency & co-occurrence
  3. Rule-based            (7 features)   - domain-knowledge patterns
  4. Feature Enhancement   (10 features)  - amount x frequency interactions
  5. Statistical Model     (33 features)  - fraud ratios, log-likelihood ratios,
                                            interval anomaly, relative amounts

All features respect temporal ordering: only past data (excluding the current
row) is used, preventing information leakage.
"""

import numpy as np
import pandas as pd
from math import sqrt, pi


# ============================================================================
# Column aliases (anonymised / masked dataset)
# ============================================================================
COL_DAY = "f3"          # temporal feature (ordinal)
COL_MONTH = "f4"        # temporal feature (cyclic)
COL_AMOUNT = "f5"       # numerical feature
COL_OS = "f6"           # categorical feature A
COL_DEPOSIT = "f7"      # binary feature
COL_STORE = "f9"        # categorical feature B
COL_ZIP = "f10"         # categorical feature C
COL_ZIP3 = "f11"        # categorical feature D (coarser granularity of C)
COL_EMAIL = "f12"       # categorical feature E
COL_PREF = "f13"        # categorical feature F
COL_ADDR = "f14"        # user identifier
COL_LABEL = "f16"       # target (0/1)


# ============================================================================
# Helper: lognormal log-pdf (vectorised)
# ============================================================================
def _lognormal_logpdf(x, mu, sigma):
    """Vectorised log-pdf of LogNormal(mu, sigma)."""
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x) & (x > 0) & np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)
    if not ok.any():
        return out
    logx = np.log(x[ok])
    s = np.maximum(sigma[ok], 1e-6)
    z = (logx - mu[ok]) / s
    out[ok] = -0.5 * z ** 2 - np.log(s * sqrt(2 * pi)) - logx
    return out


# ============================================================================
# Helper: rolling fraud ratio per group (uses only past data)
# ============================================================================
def _fraud_ratio_rolling(df, group_col, label_col, day_col, days=None):
    """
    Compute historical fraud ratio per group using only strictly-past rows.

    Returns (ratio_series, count_series).
    """
    ratio = pd.Series(np.nan, index=df.index, dtype=float)
    count = pd.Series(0, index=df.index, dtype=int)

    for gval in df[group_col].unique():
        mask = df[group_col] == gval
        gidx = df.index[mask]
        gdata = df.loc[gidx]

        for idx in gidx:
            cur_day = df.at[idx, day_col]
            past = gdata.index < idx
            if days is not None:
                past = past & (gdata[day_col] > cur_day - days) & (gdata[day_col] < cur_day)
            else:
                past = past & (gdata[day_col] < cur_day)

            subset = gdata[past]
            if len(subset) > 0:
                ratio[idx] = subset[label_col].mean()
                count[idx] = len(subset)
            else:
                ratio[idx] = 0.0
                count[idx] = 0

    return ratio, count


# ============================================================================
# 1. Base Statistics (5 features)
# ============================================================================
def _base_statistics(df):
    """
    Per-user cumulative purchase statistics (using only past transactions).

    Features:
      - purchase_count_cumulative
      - past_mean_amount
      - past_std_amount
      - mu_personal    (log-normal mu of amount)
      - sigma_personal (log-normal sigma of amount)
    """
    df["purchase_count_cumulative"] = df.groupby(COL_ADDR).cumcount() + 1

    g = df.groupby(COL_ADDR)
    df["past_mean_amount"] = (
        g[COL_AMOUNT].expanding().mean().shift(1).reset_index(level=0, drop=True)
    )
    df["past_std_amount"] = (
        g[COL_AMOUNT].expanding().std(ddof=1).shift(1).reset_index(level=0, drop=True)
    )

    df["log_f5"] = np.where(df[COL_AMOUNT] > 0, np.log(df[COL_AMOUNT]), np.nan)
    gl = df.groupby(COL_ADDR)["log_f5"]
    df["mu_personal"] = gl.expanding().mean().shift(1).reset_index(level=0, drop=True)
    df["sigma_personal"] = gl.expanding().std(ddof=1).shift(1).reset_index(level=0, drop=True)

    return df


# ============================================================================
# 2. Feature Structure (10 features)
# ============================================================================
def _feature_structure(df):
    """
    Category frequency counts and co-occurrence counts.

    Features:
      - f6_count, f9_count, f10_count, f11_count, f12_count
      - f6xf9_count, f6xf12_count, f9xf12_count, f6xf9xf12_count
      - f9_count_x_f12_count
    """
    for col in [COL_OS, COL_STORE, COL_ZIP, COL_ZIP3, COL_EMAIL]:
        df[f"{col}_count"] = df.groupby(col).cumcount() + 1

    for c1, c2 in [(COL_OS, COL_STORE), (COL_OS, COL_EMAIL), (COL_STORE, COL_EMAIL)]:
        key = df[c1].astype(str) + "_" + df[c2].astype(str)
        df[f"{c1}x{c2}_count"] = df.groupby(key).cumcount() + 1

    key3 = df[COL_OS].astype(str) + "_" + df[COL_STORE].astype(str) + "_" + df[COL_EMAIL].astype(str)
    df[f"{COL_OS}x{COL_STORE}x{COL_EMAIL}_count"] = df.groupby(key3).cumcount() + 1

    df[f"{COL_STORE}_count_x_{COL_EMAIL}_count"] = df[f"{COL_STORE}_count"] * df[f"{COL_EMAIL}_count"]

    return df


# ============================================================================
# 3. Rule-based (7 features)
# ============================================================================
def _rule_based(df):
    """
    Domain-knowledge features.

    Features:
      - f5_end_with_5, f5_end_with_0, f5_end_with_00
      - f6_changed, f9_changed, f12_changed
      - f7xf5
    """
    amt_str = df[COL_AMOUNT].astype(str)
    df["f5_end_with_5"] = amt_str.str.endswith("5").astype(int)
    df["f5_end_with_0"] = amt_str.str.endswith("0").astype(int)
    df["f5_end_with_00"] = amt_str.str.endswith("00").astype(int)

    for col in [COL_OS, COL_STORE, COL_EMAIL]:
        prev = df.groupby(COL_ADDR)[col].shift(1)
        df[f"{col}_changed"] = ((df[col] != prev) & prev.notna()).astype(int)

    df["f7xf5"] = (df[COL_DEPOSIT] == 0).astype(int) * df[COL_AMOUNT]

    return df


# ============================================================================
# 4. Feature Enhancement (10 features)
# ============================================================================
def _feature_enhancement(df):
    """
    Amount scaled by log of frequency counts.

    Features: f5_x_{count_col} = f5 / log1p(count) for each count column.
    """
    count_cols = [c for c in df.columns if c.endswith("_count")]
    for col in count_cols:
        df[f"f5_x_{col}"] = df[COL_AMOUNT] / np.log1p(df[col])
    return df


# ============================================================================
# 5. Statistical Model (33 features)
# ============================================================================
def _statistical_model(df):
    """
    Fraud ratios, log-likelihood ratio scores, and temporal anomaly features.

    Sub-groups:
      a) Fraud ratio/count per group per window   (20 features)
      b) Global amount LLR                        (3 features + log_f5)
      c) Personal amount LLR                      (2 features)
      d) Time-interval LLR                        (5 features)
      e) Amount relative to personal history       (2 features)
    """
    # --- (a) Fraud ratios for 4 groups x {7d, 30d, all} -----------------------
    groups = [
        (COL_PREF, "prefecture"),
        (COL_STORE, "store"),
        (COL_EMAIL, "email"),
        (COL_OS, "os"),
    ]
    for col, name in groups:
        for days in [7, 30, None]:
            suffix = f"{name}_{'all' if days is None else f'{days}d'}"
            r, c = _fraud_ratio_rolling(df, col, COL_LABEL, COL_DAY, days=days)
            df[f"fraud_ratio_{suffix}"] = r
            if days is not None:
                df[f"fraud_count_{suffix}"] = c

    # --- (b) Global amount log-likelihood ratio --------------------------------
    # log_f5 already created in _base_statistics
    stream_y0 = df["log_f5"].where(df[COL_LABEL] == 0)
    stream_y1 = df["log_f5"].where(df[COL_LABEL] == 1)

    mu0 = stream_y0.expanding(1).mean().shift(1).ffill().fillna(0)
    sig0 = stream_y0.expanding(2).std().shift(1).ffill().fillna(1)
    mu1 = stream_y1.expanding(1).mean().shift(1).ffill().fillna(0)
    sig1 = stream_y1.expanding(2).std().shift(1).ffill().fillna(1)

    df["f5_loglik_global_y0"] = _lognormal_logpdf(df[COL_AMOUNT].values, mu0.values, sig0.values)
    df["f5_loglik_global_y1"] = _lognormal_logpdf(df[COL_AMOUNT].values, mu1.values, sig1.values)
    df["f5_amount_llr_global"] = (df["f5_loglik_global_y1"] - df["f5_loglik_global_y0"]).fillna(0)

    # --- (c) Personal amount LLR -----------------------------------------------
    df["mu_personal"] = df["mu_personal"].fillna(mu0)
    df["sigma_personal"] = df["sigma_personal"].fillna(sig0)

    df["f5_loglik_personal"] = _lognormal_logpdf(
        df[COL_AMOUNT].values, df["mu_personal"].values, df["sigma_personal"].values,
    )
    df["f5_amount_llr_personal"] = (df["f5_loglik_global_y1"] - df["f5_loglik_personal"]).fillna(0)

    # --- (d) Time-interval features --------------------------------------------
    df["delta_days"] = df.groupby(COL_ADDR)[COL_DAY].diff()
    df["log_delta"] = np.where(df["delta_days"] > 0, np.log(df["delta_days"]), np.nan)

    dt_y0 = df["log_delta"].where(df[COL_LABEL] == 0)
    dt_y1 = df["log_delta"].where(df[COL_LABEL] == 1)

    mu_dt0 = dt_y0.expanding(1).mean().shift(1).ffill().fillna(0)
    sig_dt0 = dt_y0.expanding(2).std().shift(1).ffill().fillna(1)
    mu_dt1 = dt_y1.expanding(1).mean().shift(1).ffill().fillna(0)
    sig_dt1 = dt_y1.expanding(2).std().shift(1).ffill().fillna(1)

    df["delta_days_loglik_y0"] = _lognormal_logpdf(df["delta_days"].values, mu_dt0.values, sig_dt0.values)
    df["delta_days_loglik_y1"] = _lognormal_logpdf(df["delta_days"].values, mu_dt1.values, sig_dt1.values)
    df["delta_days_llr"] = (df["delta_days_loglik_y1"] - df["delta_days_loglik_y0"]).fillna(0)

    # --- (e) Amount relative to personal history --------------------------------
    df["amount_ratio_to_past_mean"] = (df[COL_AMOUNT] / df["past_mean_amount"]).fillna(-1)
    df["amount_std_score"] = (
        (df[COL_AMOUNT] - df["past_mean_amount"]) / df["past_std_amount"]
    ).fillna(0)

    return df


# ============================================================================
# Main entry point
# ============================================================================
def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build all 65 engineered features from raw transaction data.

    Parameters
    ----------
    df_raw : DataFrame
        Raw CSV with columns f1-f16 (or subset).

    Returns
    -------
    DataFrame with original columns plus 65 engineered features.
    """
    df = df_raw.copy()

    # Clean and sort temporally
    drop_cols = ["f1", "f2", "f8", "f15"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.dropna()
    df = df.sort_values([COL_DAY, COL_ADDR]).reset_index(drop=True)

    print("Building base statistics (5 features)...")
    df = _base_statistics(df)

    print("Building feature structure (10 features)...")
    df = _feature_structure(df)

    print("Building rule-based features (7 features)...")
    df = _rule_based(df)

    print("Building feature enhancement (10 features)...")
    df = _feature_enhancement(df)

    print("Building statistical model features (33 features)...")
    df = _statistical_model(df)

    # Fill remaining NaNs
    df["delta_days"] = df["delta_days"].fillna(-1)
    df["log_delta"] = df["log_delta"].fillna(0)
    df["past_mean_amount"] = df["past_mean_amount"].fillna(-1)
    df["past_std_amount"] = df["past_std_amount"].fillna(-1)

    for col in ["f5_loglik_personal", "delta_days_loglik_y0", "delta_days_loglik_y1",
                "delta_days_llr", "f5_amount_llr_personal", "f5_amount_llr_global"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df
