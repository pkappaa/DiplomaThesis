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


def _period_stats(rets: pd.Series, start: str, end: str) -> tuple:
    """Return (ann_ret, ann_vol, sharpe) for a sub-period slice."""
    r = rets[(rets.index >= pd.Timestamp(start)) & (rets.index <= pd.Timestamp(end))]
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 1e-12 else float("nan")
    return ann_ret, ann_vol, sharpe


def _print_period(label: str, ann_ret: float, ann_vol: float, sharpe: float) -> None:
    print(f"  {label:<18} ret={ann_ret:+.2%}  vol={ann_vol:.2%}  Sharpe={sharpe:.3f}")


if __name__ == "__main__":
    returns, splits = _load_data()
    rets = backtest(returns, splits)

    out_path = DATA_DIR / f"{STRATEGY}_returns.csv"
    rets.to_csv(out_path)

    val_stats  = _period_stats(rets, splits["val"]["start"],  splits["val"]["end"])
    test_stats = _period_stats(rets, splits["test"]["start"], splits["test"]["end"])
    comb_stats = _period_stats(rets, splits["val"]["start"],  splits["test"]["end"])

    final_w = np.ones(N_ASSETS) / N_ASSETS

    print(f"\n=== {STRATEGY}  ({N_ASSETS} assets) ===")
    print("Final weights:")
    for ticker, w in zip(ASSETS, final_w):
        print(f"  {ticker}: {w:.4f}")
    print()
    _print_period(
        f"Val  ({splits['val']['start'][:7]} to {splits['val']['end'][:7]})",
        *val_stats,
    )
    _print_period(
        f"Test ({splits['test']['start'][:7]} to {splits['test']['end'][:7]})",
        *test_stats,
    )
    _print_period(
        f"Val+Test combined",
        *comb_stats,
    )
    print(f"\nSaved -> {out_path}")
