"""Mean-Variance Optimization (Markowitz, 1952)."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def mvo_weights(expected_returns: np.ndarray, cov_matrix: np.ndarray,
                risk_aversion: float = 1.0) -> np.ndarray:
    n = len(expected_returns)

    def neg_utility(w):
        ret = w @ expected_returns
        var = w @ cov_matrix @ w
        return -(ret - 0.5 * risk_aversion * var)

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n
    result = minimize(neg_utility, w0, bounds=bounds, constraints=constraints)
    return result.x


def rolling_backtest(log_returns: pd.DataFrame, estimation_window: int = 252,
                     rebalance_freq: int = 21) -> pd.Series:
    """Walk-forward backtest with rolling estimation window."""
    portfolio_returns = []
    dates = []

    for t in range(estimation_window, len(log_returns), rebalance_freq):
        window = log_returns.iloc[t - estimation_window:t]
        mu     = window.mean().values * 252
        sigma  = window.cov().values  * 252
        w      = mvo_weights(mu, sigma)

        horizon = log_returns.iloc[t:t + rebalance_freq]
        period_ret = (horizon @ w)
        portfolio_returns.append(period_ret)
        dates.extend(horizon.index.tolist())

    result = pd.concat(portfolio_returns)
    result.name = "markowitz"
    return result
