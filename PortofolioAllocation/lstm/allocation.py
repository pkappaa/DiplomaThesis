"""Simple allocation rules that use LSTM predictions to build portfolios."""

import numpy as np
import pandas as pd


def rank_based_weights(predictions: pd.DataFrame) -> pd.DataFrame:
    """Long assets ranked highest by predicted return, short (or zero) the rest."""
    ranks  = predictions.rank(axis=1, ascending=True)
    weights = ranks.div(ranks.sum(axis=1), axis=0)
    return weights


def softmax_weights(predictions: pd.DataFrame, temperature: float = 1.0) -> pd.DataFrame:
    """Softmax over predicted returns — smooth long-only allocation."""
    scaled = predictions / temperature
    exp    = np.exp(scaled.subtract(scaled.max(axis=1), axis=0))
    return exp.div(exp.sum(axis=1), axis=0)


def backtest(log_returns: pd.DataFrame, predictions: pd.DataFrame,
             method: str = "rank") -> pd.Series:
    aligned_returns = log_returns.loc[predictions.index]

    if method == "rank":
        weights = rank_based_weights(predictions)
    elif method == "softmax":
        weights = softmax_weights(predictions)
    else:
        raise ValueError(f"Unknown method: {method}")

    portfolio_returns = (aligned_returns * weights).sum(axis=1)
    portfolio_returns.name = f"lstm_{method}"
    return portfolio_returns
