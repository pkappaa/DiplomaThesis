"""
Preprocess raw OHLCV CSVs into three ML-ready artefacts:

  processed/daily_returns.csv  – long-format panel: one row per (date, ticker)
                                  with time-series + cross-sectional features
  processed/weekly_labels.csv  – binary outperformance labels (weekly frequency)
  processed/splits.json        – canonical train / val / test date boundaries

Design philosophy
-----------------
The ultimate task is a CROSS-SECTIONAL RANKING problem: each week, identify
which of the 9 SPDR sector ETFs will outperform the median and overweight them.
The feature set is designed with two complementary views:

  Time-series features  (ret_5d, ret_21d, vol_21d, rsi_14)
      Capture each asset's own momentum and risk regime over the past weeks.
      These are computed on ret.shift(1) — the return series shifted one day
      into the past — so that the feature value at date T only incorporates
      prices up through T-1.  This is the core no-lookahead guarantee.

  Cross-sectional features  (rank, zscore)
      Capture where each asset sits relative to its peers *on the same day*.
      They use the current day's 1-day return, which is already "closed" and
      therefore not a lookahead with respect to the *future* weekly label.

  Market features  (spy_ret_1d, spy_vol_21d)
      Macro context: overall market direction and volatility regime.  Shared
      across all 9 assets on a given day; act as a regime-aware conditioning
      signal for both the LSTM and the RL observation space.

Binary label motivation
-----------------------
Predicting *relative* performance (above/below median) rather than raw returns
makes the task stationary across bull and bear markets.  A model that learns
sector rotation signals can still generate alpha even when all sectors fall —
it just overweights the ones that fall least.  The binary label is also more
robust to extreme outlier returns than a continuous regression target.

No-lookahead guarantee
----------------------
All rolling windows operate on ret.shift(1).  At date T the model input is
built entirely from prices ≤ T-1 (rolling features) or the *closed* price at
T (1-day / cross-sectional features, which are point-in-time by definition).
The weekly label for Friday T uses returns from Friday T-1 to Friday T, which
is unknown until Friday T closes.  Models must therefore treat it as the target,
never as an input feature.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve the project root (one level above this file) so config is importable
# regardless of the working directory from which the script is launched.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))   # for sibling import of download
from config import ASSETS, BENCHMARK, TRAIN_END, VAL_START, VAL_END, TEST_START

RAW_DIR       = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"

# Hard-coded split boundaries (mirrors config; duplicated here for clarity)
_SPLITS = {
    "train": {"start": "2010-01-01", "end": TRAIN_END},
    "val":   {"start": VAL_START,    "end": VAL_END},
    "test":  {"start": TEST_START,   "end": "2024-12-31"},
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_close(ticker: str) -> pd.Series:
    """Load the adjusted Close series for one ticker from its raw CSV."""
    path = RAW_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file missing: {path}\n"
            "Run download.py (or the __main__ block below) first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # yfinance sometimes writes a 'Price' or 'Ticker' header row; skip it.
    df = df[pd.to_datetime(df.index, errors="coerce").notna()]
    df.index = pd.to_datetime(df.index)
    return df["Close"].rename(ticker).astype(float)


def _compute_rsi(ret: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index on an *already-shifted* return series.

    Uses the formula:  RSI = 100 × avg_gain / (avg_gain + avg_loss)
    which is algebraically equivalent to the classic  100 − 100/(1+RS)  but
    avoids divide-by-zero when avg_loss = 0 (all-gain window → RSI = 100)
    or when both are zero (flat window → RSI = NaN, caught by dropna later).

    A simple rolling mean is used instead of Wilder's EWM for exact
    reproducibility across libraries; the EWM variant can be swapped in by
    replacing .rolling(...).mean() with .ewm(alpha=1/window, adjust=False).mean().
    """
    gains    = ret.clip(lower=0)                                 # daily up-moves
    losses   = (-ret).clip(lower=0)                              # daily down-moves (positive)
    avg_gain = gains.rolling(window, min_periods=window).mean()
    avg_loss = losses.rolling(window, min_periods=window).mean()
    denom    = (avg_gain + avg_loss).replace(0, np.nan)          # NaN on flat windows
    return 100.0 * avg_gain / denom


# ── output 1: daily feature panel ─────────────────────────────────────────────

def build_daily_features() -> pd.DataFrame:
    """
    Construct a long-format (date × ticker) feature DataFrame.

    Column layout
    -------------
    date         : trading day (index in intermediate DataFrame; column in CSV)
    ticker       : asset identifier (one of the 11 SPDR ETFs)
    ret_1d       : log return on the *current* day  (point-in-time, no lookahead)
    ret_5d       : 5-day cumulative log return computed from ret.shift(1)
                   → sum of returns from T-5 to T-1
    ret_21d      : 21-day cumulative log return (same convention)
    vol_21d      : 21-day realised daily volatility (std of log returns T-21..T-1)
                   Annualise downstream with × sqrt(252)
    rsi_14       : 14-day RSI in [0, 100] from lagged returns
    rank         : cross-sectional rank of ret_1d within the 11-asset universe
                   on each day; 1 = worst performer, 11 = best
    zscore       : (ret_1d − universe_mean) / universe_std on each day
                   measures relative strength in std-dev units
    spy_ret_1d   : SPY log return (market direction signal, same for all assets)
    spy_vol_21d  : 21-day rolling SPY log-return volatility
                   (VIX-like proxy for market fear / risk regime)
    """
    # ── load prices ───────────────────────────────────────────────────────────
    closes    = pd.concat([_load_close(t) for t in ASSETS], axis=1).sort_index()
    spy_close = _load_close(BENCHMARK).sort_index()

    # Intersect trading-day calendars: SPY may have marginally different coverage
    # than sector ETFs on half-days or data-provider gaps.
    common_idx = closes.index.intersection(spy_close.index)
    closes     = closes.loc[common_idx]
    spy_close  = spy_close.loc[common_idx]

    # ── log returns (current day) ─────────────────────────────────────────────
    # ret[t] = log(Close_t / Close_{t-1}): the return that printed at today's close
    ret     = np.log(closes    / closes.shift(1))
    spy_ret = np.log(spy_close / spy_close.shift(1))

    # ── lagged return series ──────────────────────────────────────────────────
    # All rolling features are computed on ret_lag, which at date T contains
    # returns for {T-1, T-2, …} and NaN at T.  This enforces no-lookahead:
    # the rolling window at T only ever sees information from before T.
    ret_lag = ret.shift(1)
    spy_lag = spy_ret.shift(1)

    # ── time-series features (all built from ret_lag) ─────────────────────────
    # Momentum: cumulative log-return over the past N days (= log price ratio)
    ret_5d  = ret_lag.rolling(5,  min_periods=5).sum()
    ret_21d = ret_lag.rolling(21, min_periods=21).sum()

    # Realised volatility: dispersion of daily log returns over past 21 days.
    # A wider window (e.g., 63d) is more stable but slower to react to regimes.
    vol_21d = ret_lag.rolling(21, min_periods=21).std()

    # RSI captures momentum saturation: high RSI → overbought, low → oversold.
    # Used to detect potential mean-reversion in sector rotation.
    rsi_14 = _compute_rsi(ret_lag, window=14)

    # SPY volatility as a market fear index (VIX proxy from realised vol).
    # High SPY vol periods correlate with broad risk-off rotation, which the
    # RL agent and LSTM should learn to navigate differently.
    spy_vol_21d = spy_lag.rolling(21, min_periods=21).std()

    # ── cross-sectional features (use current ret; no time lookahead) ─────────
    # Cross-sectional rank: measures relative performance within the universe.
    # method="first" assigns distinct integer ranks 1–11, matching the spec.
    rank = ret.rank(axis=1, method="first")

    # Z-score: normalises each asset's return by the universe's cross-sectional
    # mean and std each day, making the signal scale-invariant across regimes.
    cs_mean = ret.mean(axis=1)                        # daily cross-sectional mean
    cs_std  = ret.std(axis=1).replace(0, np.nan)      # avoid ÷0 on perfectly flat days
    zscore  = ret.sub(cs_mean, axis=0).div(cs_std, axis=0)

    # ── assemble long-format panel ────────────────────────────────────────────
    # One row per (date, ticker); ML frameworks reshape this to (T, N, F) tensors.
    frames = []
    for ticker in ASSETS:
        df_t = pd.DataFrame(
            {
                "ticker":      ticker,
                "ret_1d":      ret[ticker],
                "ret_5d":      ret_5d[ticker],
                "ret_21d":     ret_21d[ticker],
                "vol_21d":     vol_21d[ticker],
                "rsi_14":      rsi_14[ticker],
                "rank":        rank[ticker],
                "zscore":      zscore[ticker],
                "spy_ret_1d":  spy_ret,          # market signal: identical across all assets
                "spy_vol_21d": spy_vol_21d,
            },
            index=closes.index,
        )
        frames.append(df_t)

    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = (
        panel.reset_index()
             .sort_values(["date", "ticker"])
             .reset_index(drop=True)
    )

    # Drop warm-up rows: rolling(21) + shift(1) means the first ~22 rows per
    # ticker are NaN.  Also drops early rows for XLC (launched Jun 2018) and
    # XLRE (launched Oct 2015), which have NaN until they start trading.
    panel = panel.dropna()

    effective_start = panel["date"].min().date()
    effective_end   = panel["date"].max().date()
    print(f"  Effective date range : {effective_start} → {effective_end}")
    print(f"  (rows before {effective_start} dropped; XLC live from Jun 2018)")

    return panel


# ── output 2: weekly labels ───────────────────────────────────────────────────

def build_weekly_labels() -> pd.DataFrame:
    """
    Resample daily log returns to Friday-to-Friday weeks and assign binary labels.

    Label definition
    ----------------
    For each Friday T, label_i = 1  iff  weekly_return_i > median(weekly_returns)
    across all 11 assets that week, else 0.

    With 11 assets and a strict '>' comparison, exactly 5 assets will exceed
    the median (the median itself is the 6th-highest value), giving an expected
    label mean of 5/11 ≈ 0.454 — comfortably within the [0.40, 0.60] sanity band.

    Why binary?  Continuous weekly returns are noisy, fat-tailed, and
    non-stationary.  A relative binary label is robust to volatility regimes:
    whether markets rally 2% or crash 5%, we always ask the same question —
    "which sectors held up relatively better?"  This stationarity is critical
    for generalisation from the training period to live deployment.

    Timing note: the label for week T (Friday T) is determined only after
    Friday T's market close.  It must therefore be treated as the prediction
    TARGET for features computed at the end of the prior week (Thursday T-1
    or Friday T-1 before open).
    """
    closes  = pd.concat([_load_close(t) for t in ASSETS], axis=1).sort_index()
    ret     = np.log(closes / closes.shift(1))

    # 'W-FRI': each bin spans Mon–Fri of a given week.
    # sum(daily log-returns) = log(Close_Fri / Close_Fri_prev) = weekly log-return.
    # min_count=1 ensures holiday-shortened weeks (e.g. Thanksgiving) still produce
    # a valid number rather than silently returning 0.
    weekly = ret.resample("W-FRI").sum(min_count=1)

    # Cross-sectional median outperformance label
    weekly_median = weekly.median(axis=1)
    labels = weekly.gt(weekly_median, axis=0).astype(int)

    # ── sanity check ──────────────────────────────────────────────────────────
    # A mean far from 0.5 would indicate a systematic error: e.g., returns
    # aligned so that the same assets always beat the median (data leak) or
    # severe data gaps causing the median to be anchored to zeros.
    mean_label = float(labels.values.mean())
    assert 0.40 <= mean_label <= 0.60, (
        f"Label sanity FAILED: mean={mean_label:.4f} not in [0.40, 0.60].\n"
        "Possible causes: systematic data gaps, median anchored to zero, or "
        "asymmetric missing values across tickers."
    )
    print(f"  Label mean : {mean_label:.4f}  (sanity check passed ✓)")

    # ── long format ────────────────────────────────────────────────────────────
    weekly_long = weekly.stack().rename("weekly_return")
    labels_long = labels.stack().rename("label")
    result = pd.concat([weekly_long, labels_long], axis=1).reset_index()
    result.columns = ["week_end", "ticker", "weekly_return", "label"]
    result = result.sort_values(["week_end", "ticker"]).reset_index(drop=True)
    return result


# ── output 3: date splits ─────────────────────────────────────────────────────

def write_splits() -> dict:
    """
    Persist canonical train / val / test date boundaries to splits.json.

    Storing splits as a JSON file (rather than hard-coding them in every
    training script) gives a single source of truth that all downstream
    modules — LSTM, RL, benchmarks, evaluation — can read and respect,
    preventing accidental data leakage through inconsistent date handling.
    """
    out = PROCESSED_DIR / "splits.json"
    with open(out, "w") as f:
        json.dump(_SPLITS, f, indent=2)
    print(f"  Written : {out}")
    return _SPLITS


# ── pipeline orchestration ────────────────────────────────────────────────────

def run() -> None:
    """Build all three processed outputs and print verification summaries."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(" Building  processed/daily_returns.csv")
    print(f"{'='*60}")
    daily = build_daily_features()
    daily.to_csv(PROCESSED_DIR / "daily_returns.csv", index=False)
    print(f"  Shape  : {daily.shape}  (rows × columns)")
    print(daily.head(2).to_string(index=False))

    print(f"\n{'='*60}")
    print(" Building  processed/weekly_labels.csv")
    print(f"{'='*60}")
    weekly = build_weekly_labels()
    weekly.to_csv(PROCESSED_DIR / "weekly_labels.csv", index=False)
    print(f"  Shape  : {weekly.shape}")
    print(weekly.head(2).to_string(index=False))

    print(f"\n{'='*60}")
    print(" Writing   processed/splits.json")
    print(f"{'='*60}")
    splits = write_splits()
    print(json.dumps(splits, indent=2))

    print("\n Preprocessing complete.\n")


# ── end-to-end entry point (download → preprocess) ────────────────────────────

if __name__ == "__main__":
    # Import download from the same package directory so this script can be run
    # as a single command that handles the full data pipeline:
    #   python data/preprocess.py        (from project root)
    #   python preprocess.py             (from data/ directory)
    from download import download
    download()
    run()
