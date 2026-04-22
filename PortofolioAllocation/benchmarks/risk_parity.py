"""Risk Parity (Equal Risk Contribution) portfolio."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def risk_parity_weights(cov_matrix: np.ndarray) -> np.ndarray:
    n = cov_matrix.shape[0]
    target = np.ones(n) / n  # equal risk contribution

    def risk_concentration(w):
        sigma = np.sqrt(w @ cov_matrix @ w)
        mrc   = (cov_matrix @ w) / sigma   # marginal risk contribution
        rc    = w * mrc                     # risk contribution
        return np.sum((rc - target * sigma) ** 2)

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(1e-6, 1)] * n
    result = minimize(risk_concentration, np.ones(n) / n,
                      bounds=bounds, constraints=constraints)
    return result.x


def rolling_backtest(log_returns: pd.DataFrame, estimation_window: int = 252,
                     rebalance_freq: int = 21) -> pd.Series:
    portfolio_returns = []

    for t in range(estimation_window, len(log_returns), rebalance_freq):
        window = log_returns.iloc[t - estimation_window:t]
        sigma  = window.cov().values * 252
        w      = risk_parity_weights(sigma)
        horizon = log_returns.iloc[t:t + rebalance_freq]
        portfolio_returns.append(horizon @ w)

    result = pd.concat(portfolio_returns)
    result.name = "risk_parity"
    return result
