"""Mean-Variance Optimization — maximize Sharpe ratio, Ledoit-Wolf, expanding window."""

import json
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ASSETS, TRANSACTION_COST, RANDOM_SEED

DATA_DIR   = Path(__file__).parent.parent / "data" / "processed"
STRATEGY   = "markowitz"
N_ASSETS   = len(ASSETS)
N_RESTARTS = 5


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


def _max_sharpe_weights(hist: pd.DataFrame) -> np.ndarray:
    """
    Maximize Sharpe ratio with Ledoit-Wolf shrinkage covariance.
    Uses N_RESTARTS Dirichlet starting points via SLSQP.
    Falls back to equal weight on failure.
    """
    data = hist.dropna()
    if len(data) < N_ASSETS + 1:
        return np.ones(N_ASSETS) / N_ASSETS

    mu = data.mean().values * 252
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sigma = LedoitWolf().fit(data.values).covariance_ * 252

    def neg_sharpe(w):
        port_vol = float(np.sqrt(w @ sigma @ w))
        if port_vol < 1e-10:
            return 1e10
        return -float(w @ mu) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds      = [(0.0, 1.0)] * N_ASSETS
    rng         = np.random.default_rng(RANDOM_SEED)

    best_val, best_w = np.inf, None
    for _ in range(N_RESTARTS):
        w0  = rng.dirichlet(np.ones(N_ASSETS))
        res = minimize(neg_sharpe, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-10, "maxiter": 1000})
        if res.success and res.fun < best_val:
            best_val, best_w = res.fun, res.x.copy()

    if best_w is None:
        print(f"WARNING [{STRATEGY}]: all restarts failed. Using equal weights.")
        return np.ones(N_ASSETS) / N_ASSETS

    w     = np.clip(best_w, 0.0, 1.0)
    total = w.sum()
    return w / total if total > 1e-10 else np.ones(N_ASSETS) / N_ASSETS


def backtest(returns: pd.DataFrame, splits: dict) -> pd.Series:
    """Max-Sharpe MVO backtest; expanding window; 10bps TC on monthly rebalance."""
    val_start   = splits["val"]["start"]
    test_end    = splits["test"]["end"]
    returns     = returns.loc[:test_end]
    rebal_dates = _monthly_rebal_dates(returns.index, val_start, test_end)

    all_rets  = []
    current_w = np.zeros(N_ASSETS)

    for i, rdate in enumerate(rebal_dates):
        hist  = returns[returns.index < rdate]
        new_w = _max_sharpe_weights(hist)
        tc    = TRANSACTION_COST * np.abs(new_w - current_w).sum()

        next_date = (rebal_dates[i + 1] if i + 1 < len(rebal_dates)
                     else returns.index[-1] + pd.Timedelta(days=1))
        period = returns.loc[(returns.index >= rdate) & (returns.index < next_date)]
        if period.empty:
            continue

        rets          = (period @ new_w).copy()
        rets.iloc[0] -= tc
        all_rets.append(rets)

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

    last_rdate = _monthly_rebal_dates(
        returns.index, splits["val"]["start"], splits["test"]["end"]
    )[-1]
    final_w = _max_sharpe_weights(returns[returns.index < last_rdate])

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
