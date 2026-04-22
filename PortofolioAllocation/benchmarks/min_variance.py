"""Minimum Variance Portfolio."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def min_var_weights(cov_matrix: np.ndarray) -> np.ndarray:
    n = cov_matrix.shape[0]

    def portfolio_variance(w):
        return w @ cov_matrix @ w

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(0, 1)] * n
    result = minimize(portfolio_variance, np.ones(n) / n,
                      bounds=bounds, constraints=constraints)
    return result.x


def rolling_backtest(log_returns: pd.DataFrame, estimation_window: int = 252,
                     rebalance_freq: int = 21) -> pd.Series:
    portfolio_returns = []

    for t in range(estimation_window, len(log_returns), rebalance_freq):
        window = log_returns.iloc[t - estimation_window:t]
        sigma  = window.cov().values * 252
        w      = min_var_weights(sigma)
        horizon = log_returns.iloc[t:t + rebalance_freq]
        portfolio_returns.append(horizon @ w)

    result = pd.concat(portfolio_returns)
    result.name = "min_variance"
    return result
