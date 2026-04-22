"""Run all strategies on the test period and produce a unified comparison."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from evaluation.metrics import summarize

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR   = Path("evaluation/results")


def load_test_returns() -> pd.DataFrame:
    """Collect return series from every strategy — edit as you add strategies."""
    from benchmarks.equal_weight  import backtest as ew_bt
    from benchmarks.markowitz     import rolling_backtest as mvo_bt
    from benchmarks.min_variance  import rolling_backtest as mv_bt
    from benchmarks.risk_parity   import rolling_backtest as rp_bt
    from lstm.allocation          import backtest as lstm_bt
    from lstm.predict             import predict  as lstm_predict
    from rl.evaluate              import evaluate as rl_eval
    from rl_with_lstm.evaluate    import evaluate as rl_lstm_eval

    log_returns = pd.read_csv(PROCESSED_DIR / "log_returns_test.csv",
                              index_col=0, parse_dates=True)
    predictions = lstm_predict("test")

    series = [
        ew_bt(log_returns),
        mvo_bt(log_returns),
        mv_bt(log_returns),
        rp_bt(log_returns),
        lstm_bt(log_returns, predictions, method="rank"),
        rl_eval("test"),
        rl_lstm_eval("test"),
    ]
    return pd.concat(series, axis=1)


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_returns = load_test_returns()

    # Metrics table
    table = pd.DataFrame({name: summarize(all_returns[name])
                          for name in all_returns.columns}).T
    table = table.round(4)
    print(table.to_string())
    table.to_csv(RESULTS_DIR / "metrics_comparison.csv")

    # Cumulative return plot
    cumulative = (1 + all_returns).cumprod()
    cumulative.plot(figsize=(12, 6), title="Cumulative Returns — Test Period")
    plt.ylabel("Portfolio Value (start = 1)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cumulative_returns.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run()
