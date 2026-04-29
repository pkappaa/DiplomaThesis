"""
Convert LSTM weekly beat-median probabilities into monthly portfolio weights,
run a daily-return backtest, and print performance vs the equal-weight baseline.

Output
------
data/processed/lstm_weights.csv
    index = monthly rebalancing dates (first trading day of each month)
    columns = 9 ETF tickers
    values = portfolio weights (sum to 1, all ≥ 0)

Run from the PortofolioAllocation/ directory:
    python -m lstm.allocation
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ASSETS, TRANSACTION_COST

PROCESSED = Path("data/processed")

VAL_START  = "2022-01-01"
VAL_END    = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END   = "2024-12-31"


# ── weight construction ───────────────────────────────────────────────────────

def probs_to_weights(probs_row: pd.Series, method: str = "score_weighted") -> pd.Series:
    """
    Convert a Series of P(beat median) values into portfolio weights.

    score_weighted : weights proportional to raw probabilities (soft ranking).
    top_k          : equal weight across the 5 highest-probability assets.
    """
    if method == "score_weighted":
        total = probs_row.sum()
        if total <= 0:
            return pd.Series(1.0 / len(probs_row), index=probs_row.index)
        return probs_row / total

    if method == "top_k":
        k = min(5, len(probs_row))   # top 5 of 19 assets
        w = pd.Series(0.0, index=probs_row.index)
        w[probs_row.nlargest(k).index] = 1.0 / k
        return w

    raise ValueError(f"Unknown method: '{method}'. Choose 'score_weighted' or 'top_k'.")


# ── rebalancing schedule ──────────────────────────────────────────────────────

def _first_trading_days(index: pd.DatetimeIndex, start: str, end: str) -> list:
    """First available trading day of each calendar month in [start, end]."""
    idx = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    result, seen = [], set()
    for d in idx:
        ym = (d.year, d.month)
        if ym not in seen:
            result.append(d)
            seen.add(ym)
    return result


def build_monthly_weights(
    probs_df: pd.DataFrame,
    ret_index: pd.DatetimeIndex,
    method: str = "score_weighted",
) -> pd.DataFrame:
    """
    For each monthly rebalancing date R, select the latest weekly probabilities
    strictly before R (no lookahead) and compute weights with `method`.

    Returns a DataFrame indexed by rebalancing dates (first trading day of each
    month) with a weight column per asset.
    """
    period_start = probs_df.index.min().strftime("%Y-%m-%d")
    period_end   = probs_df.index.max().strftime("%Y-%m-%d")
    rebal_dates  = _first_trading_days(ret_index, period_start, period_end)

    rows = {}
    for rdate in rebal_dates:
        avail = probs_df.loc[probs_df.index < rdate, ASSETS]
        if avail.empty:
            continue  # no probs yet (only the very first month can be empty)
        mean_probs = avail.tail(21).mean()
        rows[rdate] = probs_to_weights(mean_probs, method)

    weights = pd.DataFrame(rows).T
    weights.index.name = "date"
    return weights


# ── backtest ──────────────────────────────────────────────────────────────────

def _backtest(daily_ret: pd.DataFrame, weights_df: pd.DataFrame) -> pd.Series:
    """
    Simulate a monthly-rebalanced long-only portfolio using daily log-returns.
    Applies a one-way transaction cost on each rebalancing day.

    weights_df.index = rebalancing dates (first trading day of each month).
    Returns a daily Series of portfolio log-returns.
    """
    rebal_dates = sorted(weights_df.index)
    all_rets    = []
    prev_w      = np.zeros(len(ASSETS))

    for i, rdate in enumerate(rebal_dates):
        # Snap to nearest available trading day on or after rdate
        pos = daily_ret.index.searchsorted(rdate)
        if pos >= len(daily_ret):
            continue
        rdate_actual = daily_ret.index[pos]

        w  = weights_df.loc[weights_df.index[i], ASSETS].values.astype(float)
        tc = TRANSACTION_COST * np.abs(w - prev_w).sum()

        next_rdate = (
            rebal_dates[i + 1]
            if i + 1 < len(rebal_dates)
            else daily_ret.index[-1] + pd.Timedelta(days=1)
        )
        period = daily_ret.loc[
            (daily_ret.index >= rdate_actual) & (daily_ret.index < next_rdate),
            ASSETS,
        ]
        if period.empty:
            continue

        rets          = period @ w
        rets.iloc[0] -= tc
        all_rets.append(rets)

        # Drift weights through the period to estimate positions at next rebal
        growth    = np.exp(period.fillna(0).values.sum(axis=0))
        drifted   = w * growth
        total     = drifted.sum()
        prev_w    = drifted / total if total > 0 else w

    return pd.concat(all_rets) if all_rets else pd.Series(dtype=float)


# ── evaluation ────────────────────────────────────────────────────────────────

def _evaluate(rets: pd.Series, label: str):
    if rets.empty:
        print(f"\n=== {label} — no data ===")
        return
    ann_ret = rets.mean() * 252
    ann_vol = rets.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum_ret = float(np.exp(rets.sum()) - 1)
    print(f"\n=== {label} ===")
    print(f"  Annualized return : {ann_ret:.4%}")
    print(f"  Annualized vol    : {ann_vol:.4%}")
    print(f"  Sharpe ratio      : {sharpe:.4f}")
    print(f"  Cumulative return : {cum_ret:.4%}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    probs_path = PROCESSED / "lstm_probabilities.csv"
    if not probs_path.exists():
        print("lstm_probabilities.csv not found — run lstm/predict.py first.")
        return

    probs_df = pd.read_csv(probs_path, parse_dates=["date"], index_col="date")

    # Daily log-returns wide format
    daily    = pd.read_csv(PROCESSED / "daily_returns.csv", parse_dates=["date"])
    ret_wide = daily.pivot(index="date", columns="ticker", values="ret_1d")[ASSETS]
    ret_wide = ret_wide.sort_index()

    period_start = probs_df.index.min().strftime("%Y-%m-%d")
    period_end   = probs_df.index.max().strftime("%Y-%m-%d")
    ret_period   = ret_wide.loc[period_start:period_end]

    primary_weights = None
    for method in ["score_weighted", "top_k"]:
        weights = build_monthly_weights(probs_df, ret_wide.index, method)
        rets    = _backtest(ret_period, weights)

        val_rets      = rets.loc[VAL_START:VAL_END]
        test_rets     = rets.loc[TEST_START:TEST_END]
        combined_rets = rets.loc[VAL_START:TEST_END]

        print(f"\n{'=' * 62}")
        print(f"LSTM  method={method}")
        _evaluate(val_rets,      "val (2022-2023)")
        _evaluate(test_rets,     "test (2024)")
        _evaluate(combined_rets, "val+test 2022-2024  <- headline")

        if method == "score_weighted":
            primary_weights = weights

    # Save primary (score_weighted) monthly weights — full 2013-2024 history
    if primary_weights is not None:
        primary_weights.to_csv(PROCESSED / "lstm_weights.csv")
        print(f"\nSaved lstm_weights.csv ({len(primary_weights)} months, "
              f"{period_start} – {period_end})")

    # ── equal-weight baseline ─────────────────────────────────────────────────
    eq_path = PROCESSED / "equal_weight_returns.csv"
    if eq_path.exists():
        eq_slice = pd.read_csv(eq_path, index_col=0, parse_dates=True).squeeze()
        eq_slice = eq_slice.loc[period_start:period_end]
    else:
        eq_w     = np.full(len(ASSETS), 1.0 / len(ASSETS))
        eq_slice = ret_period @ eq_w

    print(f"\n{'=' * 62}")
    print("Equal Weight Baseline")
    _evaluate(eq_slice.loc[VAL_START:VAL_END],  "val (2022-2023)")
    _evaluate(eq_slice.loc[TEST_START:TEST_END], "test (2024)")
    _evaluate(eq_slice.loc[VAL_START:TEST_END],  "val+test 2022-2024")


if __name__ == "__main__":
    main()
