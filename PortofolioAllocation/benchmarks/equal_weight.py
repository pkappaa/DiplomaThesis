"""1/N equal-weight portfolio — simplest baseline."""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ASSETS, TRANSACTION_COST

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
STRATEGY = "equal_weight"
N_ASSETS = len(ASSETS)


def _load_data():
    panel   = pd.read_csv(DATA_DIR / "daily_returns.csv", parse_dates=["date"])
    returns = panel.pivot(index="date", columns="ticker", values="ret_1d")[ASSETS]
    with open(DATA_DIR / "splits.json") as f:
        splits = json.load(f)
    return returns, splits


def _monthly_rebal_dates(index: pd.DatetimeIndex, start: str, end: str) -> list:
    """First trading day of each calendar month within [start, end]."""
    idx = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    result, current_ym = [], None
    for d in idx:
        ym = (d.year, d.month)
        if ym != current_ym:
            result.append(d)
            current_ym = ym
    return result


def backtest(returns: pd.DataFrame, splits: dict) -> pd.Series:
    """
    Simulate 1/N portfolio over val+test with monthly rebalancing.
    TC reflects drift from the previous rebalancing back to equal weight.
    """
    target_w    = np.ones(N_ASSETS) / N_ASSETS
    val_start   = splits["val"]["start"]
    test_end    = splits["test"]["end"]
    returns     = returns.loc[:test_end]
    rebal_dates = _monthly_rebal_dates(returns.index, val_start, test_end)

    all_rets  = []
    current_w = np.zeros(N_ASSETS)  # investor starts from cash

    for i, rdate in enumerate(rebal_dates):
        new_w = target_w.copy()
        tc    = TRANSACTION_COST * np.abs(new_w - current_w).sum()

        next_date = (rebal_dates[i + 1] if i + 1 < len(rebal_dates)
                     else returns.index[-1] + pd.Timedelta(days=1))
        period = returns.loc[(returns.index >= rdate) & (returns.index < next_date)]
        if period.empty:
            continue

        rets          = (period @ new_w).copy()
        rets.iloc[0] -= tc
        all_rets.append(rets)

        # Forward-drift weights to next rebalancing
        growth    = np.exp(period.fillna(0).values.sum(axis=0))
        drifted_w = new_w * growth
        current_w = drifted_w / drifted_w.sum()

    result      = pd.concat(all_rets)
    result.name = STRATEGY
    return result


if __name__ == "__main__":
    returns, splits = _load_data()
    rets = backtest(returns, splits)

    out_path = DATA_DIR / f"{STRATEGY}_returns.csv"
    rets.to_csv(out_path)

    final_w = np.ones(N_ASSETS) / N_ASSETS
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol

    print(f"\n=== {STRATEGY} ===")
    print("Final weights:")
    for ticker, w in zip(ASSETS, final_w):
        print(f"  {ticker}: {w:.4f}")
    print(f"Annualized return : {ann_ret:.4%}")
    print(f"Annualized vol    : {ann_vol:.4%}")
    print(f"Sharpe ratio      : {sharpe:.4f}")
    print(f"Saved -> {out_path}")
