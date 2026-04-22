"""Download raw price data for the configured assets and date range."""

import yfinance as yf
import pandas as pd
from pathlib import Path
from config import ASSETS, DATE_START, DATE_END

RAW_DIR = Path(__file__).parent / "raw"


def download():
    RAW_DIR.mkdir(exist_ok=True)
    data = yf.download(ASSETS, start=DATE_START, end=DATE_END, auto_adjust=True)
    prices = data["Close"]
    prices.to_csv(RAW_DIR / "prices.csv")
    print(f"Saved {prices.shape} price matrix to {RAW_DIR / 'prices.csv'}")
    return prices


if __name__ == "__main__":
    download()
