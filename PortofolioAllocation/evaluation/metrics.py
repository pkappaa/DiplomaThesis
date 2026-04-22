"""Performance metrics for portfolio evaluation."""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = 252) -> float:
    excess = returns - risk_free / periods
    return np.sqrt(periods) * excess.mean() / excess.std() if excess.std() > 0 else np.nan


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = 252) -> float:
    excess    = returns - risk_free / periods
    downside  = excess[excess < 0].std()
    return np.sqrt(periods) * excess.mean() / downside if downside > 0 else np.nan


def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    peak       = cumulative.cummax()
    drawdown   = (cumulative - peak) / peak
    return drawdown.min()


def calmar_ratio(returns: pd.Series, periods: int = 252) -> float:
    ann_return = returns.mean() * periods
    mdd        = abs(max_drawdown(returns))
    return ann_return / mdd if mdd > 0 else np.nan


def annualized_return(returns: pd.Series, periods: int = 252) -> float:
    return returns.mean() * periods


def annualized_volatility(returns: pd.Series, periods: int = 252) -> float:
    return returns.std() * np.sqrt(periods)


def summarize(returns: pd.Series) -> dict:
    return {
        "Ann. Return":     annualized_return(returns),
        "Ann. Volatility": annualized_volatility(returns),
        "Sharpe":          sharpe_ratio(returns),
        "Sortino":         sortino_ratio(returns),
        "Max Drawdown":    max_drawdown(returns),
        "Calmar":          calmar_ratio(returns),
    }
