"""
Evaluate the trained PPO agent on validation and test periods.

Run from the PortofolioAllocation/ directory:
    python -m rl.evaluate
or
    python rl/evaluate.py

Outputs
-------
rl/results/rl_weights.csv  — monthly portfolio weights (val + test)
rl/results/rl_returns.csv  — monthly net returns (val + test)
rl/results/comparison.csv  — metrics table vs benchmarks
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from rl.environment import PortfolioEnv
from config import ASSETS

# ── paths ──────────────────────────────────────────────────────────────────────
_RL_DIR     = Path(__file__).parent
CKPT_DIR    = _RL_DIR / "checkpoints"
RESULTS_DIR = _RL_DIR / "results"

# Evaluation date ranges (RL-specific splits)
VAL_START   = "2022-01-01"
VAL_END     = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2024-12-31"

# ── reference benchmark results (pre-computed on the same RL periods) ──────────
# Val = 2022-01 to 2023-12,  Test = 2024-01 to 2024-12
BENCHMARKS = {
    "Equal Weight (19)":      {"val": -0.243, "test": 1.082},
    "Risk Parity (19)":       {"val": -0.135, "test": 1.343},
    "LSTM score_weighted":    {"val":  0.099, "test": 1.150},
}


# ── metrics helpers (monthly frequency) ───────────────────────────────────────

def _sharpe(rets: np.ndarray, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio for monthly returns."""
    excess = rets - rf / 12
    std    = excess.std()
    return float(excess.mean() / std * np.sqrt(12)) if std > 0 else np.nan


def _sortino(rets: np.ndarray, rf: float = 0.0) -> float:
    excess   = rets - rf / 12
    down_std = excess[excess < 0].std()
    return float(excess.mean() / down_std * np.sqrt(12)) if down_std > 0 else np.nan


def _max_drawdown(rets: np.ndarray) -> float:
    cum  = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / peak
    return float(dd.min())


def _calmar(rets: np.ndarray) -> float:
    ann_ret = rets.mean() * 12
    mdd     = abs(_max_drawdown(rets))
    return float(ann_ret / mdd) if mdd > 0 else np.nan


def _gross_sharpe(gross_rets: np.ndarray) -> float:
    std = gross_rets.std()
    return float(gross_rets.mean() / std * np.sqrt(12)) if std > 0 else np.nan


def compute_metrics(
    net_rets: np.ndarray, gross_rets: np.ndarray, turnovers: np.ndarray
) -> Dict[str, float]:
    return {
        "Ann Return":    float(net_rets.mean() * 12),
        "Ann Vol":       float(net_rets.std() * np.sqrt(12)),
        "Sharpe (Net)":  _sharpe(net_rets),
        "Sharpe (Gross)":_gross_sharpe(gross_rets),
        "Sortino":       _sortino(net_rets),
        "Max Drawdown":  _max_drawdown(net_rets),
        "Calmar":        _calmar(net_rets),
        "Avg Turnover":  float(turnovers.mean()),
    }


# ── episode runner ─────────────────────────────────────────────────────────────

def run_episode(
    model: PPO, env: PortfolioEnv
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Roll out one deterministic episode.

    Returns
    -------
    net_rets, gross_rets, turnovers : (T,) arrays
    weights_matrix                  : (T, 19) array
    dates                           : list of str
    """
    obs, _ = env.reset()
    net_rets, gross_rets, turnovers, weights_list, dates = [], [], [], [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        net_rets.append(info["return"])
        gross_rets.append(info["gross_return"])
        turnovers.append(info["turnover"])
        weights_list.append(info["weights"].copy())
        dates.append(info["date"])
        done = terminated or truncated

    return (
        np.array(net_rets),
        np.array(gross_rets),
        np.array(turnovers),
        np.array(weights_list),
        dates,
    )


# ── main evaluation ────────────────────────────────────────────────────────────

def evaluate() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = CKPT_DIR / "best_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            "Run  python -m rl.train  first."
        )
    model = PPO.load(str(model_path))
    print(f"Loaded model: {model_path}")

    # ── validation period ──────────────────────────────────────────────────
    val_env  = PortfolioEnv(VAL_START, VAL_END)
    val_net, val_gross, val_to, val_w, val_dates = run_episode(model, val_env)
    val_m = compute_metrics(val_net, val_gross, val_to)

    # ── test period ────────────────────────────────────────────────────────
    test_env  = PortfolioEnv(TEST_START, TEST_END)
    test_net, test_gross, test_to, test_w, test_dates = run_episode(model, test_env)
    test_m = compute_metrics(test_net, test_gross, test_to)

    # ── combined period (2022-01 to 2024-12) ──────────────────────────────
    comb_env  = PortfolioEnv("2022-01-01", "2024-12-31")
    comb_net, comb_gross, comb_to, _, _ = run_episode(model, comb_env)
    comb_m = compute_metrics(comb_net, comb_gross, comb_to)

    # ── print metrics ──────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  PPO AGENT — Performance Metrics")
    print("=" * 66)

    def _fmt_row(label, m):
        print(
            f"  {label:<22} "
            f"Ann Ret={m['Ann Return']:+.3f}  "
            f"Vol={m['Ann Vol']:.3f}  "
            f"Sharpe(net)={m['Sharpe (Net)']:+.3f}  "
            f"Sortino={m['Sortino']:+.3f}  "
            f"MDD={m['Max Drawdown']:.3f}  "
            f"Calmar={m['Calmar']:+.3f}  "
            f"Turnover={m['Avg Turnover']:.3f}"
        )

    _fmt_row("Validation (2022-2023)", val_m)
    _fmt_row("Test        (2024)    ", test_m)
    _fmt_row("Combined  (2022-2024)", comb_m)

    # ── comparison table ───────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  COMPARISON TABLE  (Sharpe ratio, monthly, rf=0)")
    print("  Val = 2022-01 to 2023-12  |  Test = 2024-01 to 2024-12")
    print("-" * 66)
    header = f"  {'Strategy':<24} {'Val Sharpe':>12} {'Test Sharpe':>12} {'Combined':>10}"
    print(header)
    print("-" * 66)

    rows = []
    for name, bm in BENCHMARKS.items():
        combined_ref = (bm["val"] + bm["test"]) / 2
        print(f"  {name:<24} {bm['val']:>12.3f} {bm['test']:>12.3f} {combined_ref:>10.3f}")
        rows.append({"Strategy": name,
                     "Val Sharpe": bm["val"],
                     "Test Sharpe": bm["test"],
                     "Combined Sharpe": combined_ref})

    rl_val  = val_m["Sharpe (Net)"]
    rl_test = test_m["Sharpe (Net)"]
    rl_comb = comb_m["Sharpe (Net)"]
    print(f"  {'RL Agent (Phase 1)':<24} {rl_val:>12.3f} {rl_test:>12.3f} {rl_comb:>10.3f}")
    rows.append({"Strategy": "RL Agent (Phase 1)",
                 "Val Sharpe": rl_val,
                 "Test Sharpe": rl_test,
                 "Combined Sharpe": rl_comb})
    print("=" * 66)

    # ── save results ───────────────────────────────────────────────────────

    # Weights
    all_dates   = val_dates   + test_dates
    all_weights = np.vstack([val_w, test_w])
    weights_df  = pd.DataFrame(all_weights, index=all_dates, columns=ASSETS)
    weights_df.index.name = "date"
    weights_df.to_csv(RESULTS_DIR / "rl_weights.csv")

    # Returns
    all_net    = np.concatenate([val_net,   test_net])
    all_gross  = np.concatenate([val_gross, test_gross])
    returns_df = pd.DataFrame({
        "net_return":   all_net,
        "gross_return": all_gross,
    }, index=all_dates)
    returns_df.index.name = "date"
    returns_df.to_csv(RESULTS_DIR / "rl_returns.csv")

    # Comparison table
    comp_df = pd.DataFrame(rows).set_index("Strategy")
    comp_df.to_csv(RESULTS_DIR / "comparison.csv")

    print(f"\n  Saved:")
    print(f"    {RESULTS_DIR / 'rl_weights.csv'}")
    print(f"    {RESULTS_DIR / 'rl_returns.csv'}")
    print(f"    {RESULTS_DIR / 'comparison.csv'}")


if __name__ == "__main__":
    evaluate()
