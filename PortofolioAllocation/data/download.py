"""
Download raw daily OHLCV data for the 19 multi-asset ETFs and the SPY benchmark.

Strategy
--------
Each ticker is downloaded individually rather than in a batch so that:
  - A single ticker failure (e.g., API timeout) does not abort the whole run.
  - Each raw CSV is a self-contained, inspectable file (date × OHLCV).
  - Partial re-downloads (e.g., extending the date range) are easy to do.

Prices are split- and dividend-adjusted (auto_adjust=True) so that log-return
time series are economically meaningful without gap artefacts around corporate
actions.
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Allow running as `python data/download.py` from the project root, or as
# `python download.py` from inside the data/ directory.
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ASSETS, BENCHMARK, DATE_START, DATE_END

# Approximate number of NYSE trading days from DATE_START to DATE_END.
# Used only for the missing-days assertion; a tolerance of 5 days covers
# data-provider gaps without masking genuine ticker data problems.
_MAX_MISSING_TRADING_DAYS = 150

RAW_DIR = Path(__file__).parent / "raw"


# ── helpers ───────────────────────────────────────────────────────────────────

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance ≥ 0.2 returns a MultiIndex (field, ticker) even for single-ticker
    downloads.  Drop the ticker level so column names are plain strings:
    Open, High, Low, Close, Volume.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _download_one(ticker: str) -> pd.DataFrame | None:
    """
    Fetch OHLCV for a single ticker from yfinance.
    
    Returns None if yfinance returns no data or raises an exception, so the
    caller can continue to the next ticker without crashing the whole pipeline.
    Missing Close values are forward-filled because a handful of NaNs from
    data-provider gaps would otherwise propagate into every downstream feature
    for that ticker.
    """
    try:
        df = yf.download(
            ticker,
            start=DATE_START,
            end=DATE_END,
            auto_adjust=True,   # adjusts for splits and dividends
            progress=False,     # suppress per-ticker progress bars
        )
    except Exception as exc:
        print(f"  [ERROR] {ticker}: {exc}")
        return None

    if df.empty:
        print(f"  [WARN]  {ticker}: yfinance returned an empty DataFrame")
        return None

    df = _flatten_columns(df)

    # Warn about and patch any gaps in the Close series.
    # Forward-filling is appropriate here: the last known price is the best
    # available estimate when a market is unexpectedly closed or data is absent.
    n_missing = int(df["Close"].isna().sum())
    if n_missing:
        print(f"  [WARN]  {ticker}: {n_missing} missing Close value(s) – forward-filled")
        df["Close"] = df["Close"].ffill()

    return df


# ── main download function ────────────────────────────────────────────────────

def _expected_trading_days() -> int:
    """
    Estimate the number of NYSE trading days between DATE_START and DATE_END.
    Uses pandas business-day count as an approximation (slightly over-counts by
    ~9 days/year for US holidays, but is good enough for a < 5-gap assertion).
    """
    bdays = pd.bdate_range(start=DATE_START, end=DATE_END)
    return len(bdays)


def download() -> dict[str, int]:
    """
    Download all ASSETS + BENCHMARK tickers and save them as individual CSVs
    under data/raw/<TICKER>.csv.

    After saving, asserts that each successfully downloaded ticker has fewer
    than _MAX_MISSING_TRADING_DAYS gaps relative to the full expected calendar,
    so bad data is caught early before preprocessing.

    Returns a summary dict {ticker: row_count} (row_count = 0 for failures).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Always include the benchmark so that market-feature engineering in
    # preprocess.py can load SPY without a separate download step.
    all_tickers = ASSETS + [BENCHMARK]
    results: dict[str, int] = {}

    print(f"\n{'='*60}")
    print(f" Downloading {len(all_tickers)} tickers  |  {DATE_START} → {DATE_END}")
    print(f"{'='*60}\n")

    expected_days = _expected_trading_days()

    for ticker in all_tickers:
        df = _download_one(ticker)
        if df is not None:
            out_path = RAW_DIR / f"{ticker}.csv"
            df.to_csv(out_path)
            results[ticker] = len(df)
            missing = max(0, expected_days - len(df))
            print(f"  [OK]   {ticker:6s}  {len(df):4d} rows  missing≈{missing:3d}  →  {out_path.name}")
        else:
            results[ticker] = 0

    # ── missing-days assertion ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f" {'Ticker':<8} {'Rows':>6} {'Missing':>8} {'Status':<10}")
    print(f" {'------':<8} {'----':>6} {'-------':>8} {'------':<10}")
    assertion_failures = []
    for ticker in all_tickers:
        rows    = results[ticker]
        missing = max(0, expected_days - rows) if rows > 0 else expected_days
        if rows == 0:
            status = "FAILED"
        elif missing >= _MAX_MISSING_TRADING_DAYS:
            status = f"GAPS({missing})"
            assertion_failures.append((ticker, missing))
        else:
            status = "OK"
        print(f" {ticker:<8} {rows:>6} {missing:>8} {status:<10}")

    n_ok   = sum(1 for v in results.values() if v > 0)
    failed = [t for t, v in results.items() if v == 0]
    print(f"{'─'*60}")
    print(f" {n_ok}/{len(all_tickers)} tickers downloaded successfully")
    if failed:
        print(f" FAILED tickers: {failed}")
        print(" Re-run or check your internet connection / yfinance version.")
    if assertion_failures:
        msgs = ", ".join(f"{t}({m} missing)" for t, m in assertion_failures)
        raise AssertionError(
            f"Tickers with ≥ {_MAX_MISSING_TRADING_DAYS} missing trading days: {msgs}\n"
            "Check yfinance data or consider adjusting DATE_START."
        )
    print(f" All tickers passed the < {_MAX_MISSING_TRADING_DAYS} missing-days check ✓")
    print(f"{'─'*60}\n")

    return results


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    download()
