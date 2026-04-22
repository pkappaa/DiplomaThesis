"""1/N equal-weight portfolio — simplest baseline."""

import numpy as np
import pandas as pd


def get_weights(n_assets: int) -> np.ndarray:
    return np.ones(n_assets) / n_assets


def backtest(log_returns: pd.DataFrame) -> pd.Series:
    w = get_weights(log_returns.shape[1])
    portfolio_returns = log_returns @ w
    portfolio_returns.name = "equal_weight"
    return portfolio_returns
