"""
Data loading, daily label computation, expanding-window normalisation, and
multi-task sequence construction for the LSTM beat-median classifier.

Fischer & Krauss (2018) daily sliding-window approach:
  - Features: daily, 11 features per asset (unchanged)
  - Sequences: X[i] = daily_features[i : i+240], shape (240, 209)
  - Labels:    daily beat-median binary, y[i] = label for day i+240, shape (19,)
  - Slide step: 1 trading day

Pipeline
--------
  load_data()
    → expanding_zscore(daily, date_col='date')
    → build_sequences()     → X (n_days, 240, 209), y (n_days, 19), ...

Sequence layout
---------------
For prediction day T (the day after the 240-day feature window):
  X = daily_features[T-240 : T]   shape (240, 209)   [19 assets × 11 features]
  y = beat-median labels for day T  shape (19,)

No-lookahead: X ends at day T-1; labels are for day T.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROCESSED = Path("data/processed")

FEAT_COLS = [
    "ret_1d", "ret_5d", "ret_21d", "vol_21d",
    "rsi_14", "momentum_12_1", "reversal_1m",
    "rank", "zscore",
    "spy_ret_1d", "spy_vol_21d",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _compute_daily_labels(daily_long: pd.DataFrame) -> pd.DataFrame:
    """
    For each trading day, label each asset 1 if its ret_1d exceeds the
    cross-sectional median across all 19 assets that day.
    """
    wide = daily_long.pivot(index="date", columns="ticker", values="ret_1d")
    med  = wide.median(axis=1)
    labs = wide.gt(med, axis=0).astype(float)
    long = labs.stack().reset_index()
    long.columns = ["date", "ticker", "label"]
    return long


# ── public loaders ────────────────────────────────────────────────────────────

def load_data():
    """Return (daily_long, daily_labels, splits_dict)."""
    daily = pd.read_csv(PROCESSED / "daily_returns.csv", parse_dates=["date"])
    with open(PROCESSED / "splits.json") as fh:
        splits = json.load(fh)
    daily_labels = _compute_daily_labels(daily)
    return daily, daily_labels, splits


# ── normalisation ─────────────────────────────────────────────────────────────

def expanding_zscore(df_long: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Apply expanding-window cross-sectional z-score to every FEAT_COLS column.

    For each period T, features are normalised using running mean/std
    accumulated from ALL (period ≤ T, asset) observations seen so far.
    Supports both daily data (date_col='date') and weekly data
    (date_col='week_end').
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
                sig = max(float(np.nanstd(orig)), 1e-8)
            else:
                mu  = acc_sum[f] / n
                var = acc_sq[f] / n - mu ** 2
                sig = max(float(np.sqrt(max(var, 0.0))), 1e-8)

            day_df[f] = np.clip((orig - mu) / sig, -10.0, 10.0)

            valid = orig[~np.isnan(orig)]
            acc_n[f]   += len(valid)
            acc_sum[f] += float(np.sum(valid))
            acc_sq[f]  += float(np.sum(valid ** 2))

        parts.append(day_df)

    return pd.concat(parts, ignore_index=True)


# ── sequence construction ─────────────────────────────────────────────────────

def build_sequences(
    daily_norm: pd.DataFrame,
    daily_labels: pd.DataFrame,
    lookback: int = 240,
    slide_step: int = 1,
):
    """
    Build multi-task training tensors from daily-normalised features and labels.
    Fischer & Krauss (2018) daily sliding-window approach.

    Parameters
    ----------
    daily_norm   : output of expanding_zscore on daily data.
                   Must have columns: date, ticker, and FEAT_COLS.
    daily_labels : long format with date, ticker, label (daily beat-median).
    lookback     : number of daily timesteps per sequence (default 240 ≈ 1 year).
    slide_step   : stride between consecutive sequences in trading days (default 1).

    Returns
    -------
    X          : FloatTensor (n_samples, lookback, n_assets * n_features)
                 Shape: (n_days, 240, 209)  [19 assets × 11 features per timestep]
                 Each timestep concatenates all assets' features in alphabetical order.
                 Feature block order: [asset0_f0..f10, asset1_f0..f10, ...]
    y          : FloatTensor (n_samples, n_assets)  shape (n_days, 19)
                 NaN where label is missing.
    date_list  : list[pd.Timestamp]  — label date per sample (day after window end).
    asset_names: list[str]           — tickers in consistent alphabetical order.
    """
    daily_wide = daily_norm.pivot(
        index="date", columns="ticker", values=FEAT_COLS
    ).sort_index()
    all_dates = list(daily_wide.index)
    n_days    = len(all_dates)

    assets = sorted(daily_labels["ticker"].unique())

    # Pre-extract per-asset feature arrays: shape (n_total_days, n_feats)
    asset_arrs: dict[str, np.ndarray] = {}
    for asset in assets:
        arr = np.column_stack(
            [daily_wide[(f, asset)].values for f in FEAT_COLS]
        ).astype(np.float32)
        asset_arrs[asset] = np.nan_to_num(arr, nan=0.0)

    labels_dict = (
        daily_labels.set_index(["date", "ticker"])["label"].to_dict()
    )

    X_list, y_list, date_list = [], [], []

    for i in range(lookback, n_days, slide_step):
        label_date_ts = pd.Timestamp(all_dates[i])
        start_idx     = i - lookback

        # Concatenate all assets' feature blocks per timestep → (lookback, n_assets * n_feats)
        feat_blocks = [asset_arrs[a][start_idx:i] for a in assets]
        feat_mat    = np.concatenate(feat_blocks, axis=1).astype(np.float32)

        y_row = [
            float(labels_dict[(label_date_ts, a)])
            if (label_date_ts, a) in labels_dict
            else float("nan")
            for a in assets
        ]
        if all(np.isnan(v) for v in y_row):
            continue

        X_list.append(feat_mat)
        y_list.append(y_row)
        date_list.append(label_date_ts)

    X = torch.from_numpy(np.array(X_list, dtype=np.float32))
    y = torch.from_numpy(np.array(y_list,  dtype=np.float32))

    _print_sanity_checks(X, y, date_list, assets, all_dates)

    return X, y, date_list, assets


def save_sequences(
    X: torch.Tensor,
    y: torch.Tensor,
    date_list: list,
    asset_names: list,
    path: Path = PROCESSED / "sequences.pt",
) -> None:
    """Persist build_sequences output for downstream RL/predict use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": X, "y": y, "dates": date_list, "assets": asset_names}, path)
    print(f"Saved sequences -> {path}  (X={tuple(X.shape)}, y={tuple(y.shape)})")


def _print_sanity_checks(
    X: torch.Tensor,
    y: torch.Tensor,
    date_list: list,
    assets: list,
    all_dates: list,
) -> None:
    sep = "=" * 62
    print(f"\n{sep}\nBUILD_SEQUENCES SANITY CHECK\n{sep}")

    n    = len(date_list)
    y_np = y.numpy()
    x_np = X.numpy()

    # [1] Total samples
    print(f"\n[1] Total samples: {n}")
    print(f"    X shape : {tuple(X.shape)}")
    print(f"    y shape : {tuple(y.shape)}")

    # [2] Label distribution per asset (target 40–60 %)
    print(f"\n[2] Label distribution per asset (target 40–60%)")
    for i, asset in enumerate(assets):
        col   = y_np[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue
        pos  = int(valid.sum())
        frac = pos / len(valid)
        flag = "  ← CHECK" if not (0.40 <= frac <= 0.60) else ""
        print(f"    {asset}: {pos}/{len(valid)} ({frac:.1%}){flag}")

    # [3] Date alignment: X[-1] date → y date must be next trading day
    print(f"\n[3] Date alignment  (X[-1] date  ->  label date = next trading day)")
    date_to_idx = {d: k for k, d in enumerate(all_dates)}
    for idx in [0, n // 2, n - 1]:
        label_dt  = date_list[idx]
        pos       = date_to_idx[label_dt]
        feat_last = all_dates[pos - 1]
        print(f"    [{idx:6d}]  X[-1]={pd.Timestamp(feat_last).date()}  "
              f"->  label={label_dt.date()}")

    # [4] Feature value range (should be roughly –3 to +3 after z-score)
    print(f"\n[4] Feature value range after expanding z-score:")
    print(f"    min={x_np.min():.4f}  max={x_np.max():.4f}  "
          f"mean={x_np.mean():.4f}  std={x_np.std():.4f}")

    # [5] No NaNs check
    n_nan_x = int(np.isnan(x_np).sum())
    n_nan_y = int(np.isnan(y.numpy()).sum())
    print(f"\n[5] NaN check:")
    print(f"    X NaNs: {n_nan_x}  (expect 0)")
    print(f"    y NaNs: {n_nan_y}  (expect 0)")

    print(f"{sep}\n")


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    seq_path = PROCESSED / "sequences.pt"
    if seq_path.exists():
        seq_path.unlink()
        print(f"Deleted old {seq_path}")

    print("Loading processed data...")
    daily, daily_labels, _ = load_data()

    print("Applying expanding z-score normalisation...")
    daily_norm = expanding_zscore(daily, date_col="date")

    print("Building 240-step daily sequences (this may take a few minutes)...")
    X, y, date_list, asset_names = build_sequences(daily_norm, daily_labels, lookback=240)

    save_sequences(X, y, date_list, asset_names)
    print("\nDone — sequences.pt ready for training.")
