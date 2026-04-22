"""Clean raw prices and engineer features: log returns, rolling volatility, normalisation."""

import pandas as pd
import numpy as np
from pathlib import Path
from config import TRAIN_END, VAL_END

RAW_DIR       = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"


def load_prices() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "prices.csv", index_col=0, parse_dates=True)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def compute_rolling_volatility(log_returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return log_returns.rolling(window).std().dropna()


def split(df: pd.DataFrame):
    train = df.loc[:TRAIN_END]
    val   = df.loc[TRAIN_END:VAL_END].iloc[1:]
    test  = df.loc[VAL_END:].iloc[1:]
    return train, val, test


def run():
    PROCESSED_DIR.mkdir(exist_ok=True)
    prices      = load_prices()
    log_returns = compute_log_returns(prices)
    volatility  = compute_rolling_volatility(log_returns)

    log_returns.to_csv(PROCESSED_DIR / "log_returns.csv")
    volatility.to_csv(PROCESSED_DIR  / "volatility.csv")

    for name, df in [("log_returns", log_returns), ("volatility", volatility)]:
        train, val, test = split(df)
        train.to_csv(PROCESSED_DIR / f"{name}_train.csv")
        val.to_csv(PROCESSED_DIR   / f"{name}_val.csv")
        test.to_csv(PROCESSED_DIR  / f"{name}_test.csv")

    print("Preprocessing complete.")


if __name__ == "__main__":
    run()
