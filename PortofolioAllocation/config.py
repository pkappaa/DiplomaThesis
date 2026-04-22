# Global configuration for all modules

ASSETS = [
    # e.g. "AAPL", "MSFT", "GOOGL", ...
]

DATE_START = "2015-01-01"
DATE_END   = "2024-12-31"

TRAIN_END  = "2021-12-31"
VAL_END    = "2022-12-31"
# test period: VAL_END -> DATE_END

RANDOM_SEED = 42

TRANSACTION_COST = 0.001  # 0.1% per trade

LSTM_CONFIG = {
    "target": "log_return",   # "log_return" or "volatility"
    "lookback": 20,
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
