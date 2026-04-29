# Global configuration for all modules

# ── universe & benchmark ──────────────────────────────────────────────────────
# 19 multi-asset ETFs spanning US sectors, US style, international equity,
# fixed income, and real assets.  The broadened universe enables cross-asset
# rotation signals that a sector-only model cannot capture.
ASSETS = [
    # US sectors (9 SPDR Select Sector ETFs)
    "XLK",   # Information Technology
    "XLF",   # Financials
    "XLV",   # Health Care
    "XLE",   # Energy
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLB",   # Materials
    "XLU",   # Utilities
    # US style
    "QQQ",   # Nasdaq-100 (tech/growth)
    "IWM",   # Russell 2000 (small-cap)
    # International equity
    "EFA",   # MSCI EAFE (developed ex-US)
    "EEM",   # MSCI Emerging Markets
    # Fixed income
    "TLT",   # 20+ Year Treasury
    "IEF",   # 7-10 Year Treasury
    "HYG",   # High Yield Corporate
    # Real assets
    "GLD",   # Gold
    "DBC",   # Commodities
    "VNQ",   # US REITs
]

N_ASSETS = 19

# SPY is used as the market benchmark for performance attribution and as a
# macro feature (market return, market volatility proxy) in the feature set.
BENCHMARK = "SPY"

# ── date range ────────────────────────────────────────────────────────────────
DATE_START = "2010-01-01"
DATE_END   = "2024-12-31"

# ── train / val / test splits ─────────────────────────────────────────────────
# Train  2010–2021 : multiple full market cycles;
#                    .
# Val    2022–H1 2023 : post-COVID rate-hike regime for hyperparameter tuning
#                        and early stopping; no information leaks into test.
# Test   H2 2023–2024 : held-out, never used during model development –
#                        the only unbiased estimate of live performance.
TRAIN_END  = "2021-12-31"
VAL_START  = "2022-01-01"
VAL_END    = "2023-06-30"
TEST_START = "2023-07-01"

RANDOM_SEED = 42

TRANSACTION_COST = 0.001  # 0.1% per trade (one-way); applied on rebalance

LSTM_CONFIG = {
    "target": "log_return",    #"log_return" or "volatility"
    "lookback": 20,#
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "epochs": 100,
    "batch_size": 64,
    "lr": 1e-3,
}

RL_CONFIG = {
    "algorithm": "PPO",       # "PPO" or "SAC"
    "total_timesteps": 500_000,
    "learning_rate": 3e-4,
}
