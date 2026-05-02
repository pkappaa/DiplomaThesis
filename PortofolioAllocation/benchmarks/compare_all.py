"""Run all 4 benchmark strategies and print a comparison table."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from equal_weight import backtest as ew_backtest, _load_data
from markowitz   import backtest as mz_backtest
from min_variance import backtest as mv_backtest
from risk_parity  import backtest as rp_backtest


def period_stats(rets: pd.Series, start: str, end: str) -> tuple:
    r = rets[(rets.index >= pd.Timestamp(start)) & (rets.index <= pd.Timestamp(end))]
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 1e-12 else float("nan")
    return ann_ret, ann_vol, sharpe


def main():
    returns, splits = _load_data()

    val_start  = splits["val"]["start"]
    val_end    = splits["val"]["end"]
    test_start = splits["test"]["start"]
    test_end   = splits["test"]["end"]

    strategies = [
        ("equal_weight", ew_backtest),
        ("markowitz",    mz_backtest),
        ("min_variance", mv_backtest),
        ("risk_parity",  rp_backtest),
    ]

    rows = []
    for name, fn in strategies:
        print(f"Running {name} ...", flush=True)
        rets = fn(returns, splits)
        rets.to_csv(
            Path(__file__).parent.parent / "data" / "processed" / f"{name}_returns.csv"
        )
        vr, vv, vs = period_stats(rets, val_start,  val_end)
        tr, tv, ts = period_stats(rets, test_start, test_end)
        cr, cv, cs = period_stats(rets, val_start,  test_end)
        rows.append((name, vr, vv, vs, tr, tv, ts, cr, cv, cs))

    print()
    val_label  = f"Val  {val_start[:7]}-{val_end[:7]}"
    test_label = f"Test {test_start[:7]}-{test_end[:7]}"
    comb_label = f"Val+Test combined"

    # ── header ────────────────────────────────────────────────────────────────
    col_w = 16
    h0 = f"{'Strategy':<16}"
    h1 = (f"{'Ret':>8}{'Vol':>8}{'Sharpe':>8}  "
          f"{'Ret':>8}{'Vol':>8}{'Sharpe':>8}  "
          f"{'Ret':>8}{'Vol':>8}{'Sharpe':>8}")
    sep = "-" * len(h0 + h1)

    print(sep)
    print(f"{'Strategy':<16}  "
          f"{val_label:^26}  "
          f"{test_label:^26}  "
          f"{comb_label:^26}")
    print(f"{'':16}  "
          f"{'Ret':>8}{'Vol':>8}{'Sharpe':>8}  "
          f"{'Ret':>8}{'Vol':>8}{'Sharpe':>8}  "
          f"{'Ret':>8}{'Vol':>8}{'Sharpe':>8}")
    print(sep)

    for name, vr, vv, vs, tr, tv, ts, cr, cv, cs in rows:
        print(f"{name:<16}  "
              f"{vr:>+8.2%}{vv:>8.2%}{vs:>8.3f}  "
              f"{tr:>+8.2%}{tv:>8.2%}{ts:>8.3f}  "
              f"{cr:>+8.2%}{cv:>8.2%}{cs:>8.3f}")

    print(sep)
    print()


if __name__ == "__main__":
    main()
