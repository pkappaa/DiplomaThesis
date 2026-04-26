"""
Data loading, weekly aggregation, expanding-window normalisation, and
sequence construction for the LSTM beat-median classifier.

Pipeline
--------
  load_data()
    → aggregate_to_weekly()          daily (date, ticker, 9 feats)
                                     → weekly (week_end, ticker, 9 feats)
    → expanding_zscore(..., 'week_end')
    → build_sequences()              → X (N, 52, 9), y (N,), metadata

Weekly aggregation rules (per spec)
------------------------------------
  Returns / momentum  ret_1d, ret_5d, ret_21d, spy_ret_1d  → mean
  Volatility          vol_21d, spy_vol_21d                  → max
  Oscillators / rank  rsi_14, rank, zscore                  → last (Friday value)

Sequence layout
---------------
For prediction week T (week_end = Friday F_T):
  X = the 52 feature weeks with week_end STRICTLY BEFORE F_T  → shape (52, 9)
  y = 1 iff asset beat cross-sectional median return during week T

No-lookahead guarantee: bisect_left finds the position of F_T in the sorted
feature-week index; slicing [pos-52 : pos] excludes week T's own aggregated
features entirely.
"""

import bisect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROCESSED = Path("data/processed")

FEAT_COLS = [
    "ret_1d", "ret_5d", "ret_21d", "vol_21d",
    "rsi_14", "rank", "zscore",
    "spy_ret_1d", "spy_vol_21d",
]

# Aggregation rules for collapsing ~5 daily rows into one weekly row
WEEKLY_AGG: dict[str, str] = {
    "ret_1d":     "mean",
    "ret_5d":     "mean",
    "ret_21d":    "mean",
    "vol_21d":    "max",
    "rsi_14":     "last",
    "rank":       "last",
    "zscore":     "mean",
    "spy_ret_1d": "mean",
    "spy_vol_21d":"max",
}


# ── public loaders ────────────────────────────────────────────────────────────

def load_data():
    """Return (daily_long, weekly_labels, splits_dict)."""
    daily  = pd.read_csv(PROCESSED / "daily_returns.csv",  parse_dates=["date"])
    labels = pd.read_csv(PROCESSED / "weekly_labels.csv",  parse_dates=["week_end"])
    with open(PROCESSED / "splits.json") as fh:
        splits = json.load(fh)
    return daily, labels, splits


# ── weekly aggregation ────────────────────────────────────────────────────────

def aggregate_to_weekly(daily_long: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse daily long-format features into one row per (W-FRI week, ticker).

    Each trading day is assigned to the Friday of its calendar week via
    dayofweek arithmetic: offset = (4 - dayofweek) % 7.  This produces the
    same Friday dates as pandas W-FRI resampling used in preprocess.py, so
    the resulting week_end column aligns exactly with weekly_labels.week_end.
    """
    df = daily_long.copy()
    offset = (4 - df["date"].dt.dayofweek) % 7
    df["week_end"] = (df["date"] + pd.to_timedelta(offset, unit="D")).dt.normalize()

    weekly = (
        df.sort_values(["week_end", "ticker", "date"])          # sort so 'last' = Friday
        .groupby(["week_end", "ticker"], sort=True)
        .agg(WEEKLY_AGG)
        .reset_index()
    )
    return weekly   # columns: week_end, ticker, ret_1d, …


# ── normalisation ─────────────────────────────────────────────────────────────

def expanding_zscore(df_long: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Apply expanding-window cross-sectional z-score to every FEAT_COLS column.

    For each period T, features are normalised using running mean/std
    accumulated from ALL (period ≤ T, asset) observations seen so far.
    Supports both daily data (date_col='date') and weekly data
    (date_col='week_end').  Returns a DataFrame with identical shape and
    columns, feature values replaced by their z-scored counterparts.
    """
    df_long = df_long.sort_values([date_col, "ticker"]).reset_index(drop=True)
    dates = sorted(df_long[date_col].unique())

    acc_n   = {f: 0    for f in FEAT_COLS}
    acc_sum = {f: 0.0  for f in FEAT_COLS}
    acc_sq  = {f: 0.0  for f in FEAT_COLS}

    parts = []
    for date in dates:
        mask   = df_long[date_col] == date
        day_df = df_long[mask].copy()

        for f in FEAT_COLS:
            orig = df_long.loc[mask, f].values.astype(np.float64)
            n    = acc_n[f]

            if n < 2:
                mu  = float(np.nanmean(orig))
                sig = float(np.nanstd(orig)) + 1e-8
            else:
                mu  = acc_sum[f] / n
                var = acc_sq[f] / n - mu ** 2
                sig = float(np.sqrt(max(var, 0.0))) + 1e-8

            day_df[f] = (orig - mu) / sig

            valid = orig[~np.isnan(orig)]
            acc_n[f]   += len(valid)
            acc_sum[f] += float(np.sum(valid))
            acc_sq[f]  += float(np.sum(valid ** 2))

        parts.append(day_df)

    return pd.concat(parts, ignore_index=True)


# ── sequence construction ─────────────────────────────────────────────────────

def build_sequences(
    weekly_norm: pd.DataFrame,
    weekly_labels: pd.DataFrame,
    lookback: int = 52,
):
    """
    Build training tensors from weekly-normalised features and weekly labels.

    Parameters
    ----------
    weekly_norm   : output of expanding_zscore on weekly-aggregated data.
                    Must have columns: week_end, ticker, and FEAT_COLS.
    weekly_labels : long format with week_end (Friday), ticker, label.
    lookback      : number of weekly timesteps per sequence (default 52).

    Returns
    -------
    X          : FloatTensor (n_samples, lookback, n_features)  — 52 weekly steps
    y          : FloatTensor (n_samples,)
    week_dates : list[pd.Timestamp]  — label week_end per sample
    asset_names: list[str]           — ticker per sample
    """
    # Wide format: index = week_end (Friday), MultiIndex columns = (feature, ticker)
    weekly_wide = weekly_norm.pivot(
        index="week_end", columns="ticker", values=FEAT_COLS
    ).sort_index()
    all_weeks = list(weekly_wide.index)   # sorted pd.Timestamps (Fridays)

    assets = sorted(weekly_labels["ticker"].unique())

    # Pre-extract per-asset arrays in FEAT_COLS order for O(1) slicing later
    asset_arrs: dict[str, np.ndarray] = {}
    for asset in assets:
        arr = np.column_stack(
            [weekly_wide[(f, asset)].values for f in FEAT_COLS]
        ).astype(np.float32)
        asset_arrs[asset] = np.nan_to_num(arr, nan=0.0)

    # Fast O(1) label lookup
    labels_dict = (
        weekly_labels.set_index(["week_end", "ticker"])["label"].to_dict()
    )

    week_dates_unique = sorted(weekly_labels["week_end"].unique())

    X_list, y_list, wk_list, asset_list = [], [], [], []

    for week_end in week_dates_unique:
        week_end_ts = pd.Timestamp(week_end)

        # bisect_left returns the position of week_end_ts in all_weeks.
        # Slice [pos-lookback : pos] contains exactly the lookback feature
        # weeks that end STRICTLY BEFORE week_end_ts — no lookahead.
        prior_idx = bisect.bisect_left(all_weeks, week_end_ts)
        if prior_idx < lookback:
            continue

        start_idx = prior_idx - lookback

        for asset in assets:
            label = labels_dict.get((week_end_ts, asset))
            if label is None:
                continue

            feat_mat = asset_arrs[asset][start_idx:prior_idx]  # (lookback, F)
            X_list.append(feat_mat)
            y_list.append(float(label))
            wk_list.append(week_end_ts)
            asset_list.append(asset)

    X = torch.from_numpy(np.array(X_list, dtype=np.float32))
    y = torch.from_numpy(np.array(y_list,  dtype=np.float32))
    return X, y, wk_list, asset_list
