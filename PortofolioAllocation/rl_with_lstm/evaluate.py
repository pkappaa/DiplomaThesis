"""Evaluate the RL+LSTM agent on the test split."""

import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, SAC
from rl_with_lstm.environment import PortfolioEnvWithPredictions
from config import RL_CONFIG

PROCESSED_DIR = Path("data/processed")
MODEL_DIR     = Path("rl_with_lstm/checkpoints")


def evaluate(split: str = "test") -> pd.Series:
    log_returns = pd.read_csv(PROCESSED_DIR / f"log_returns_{split}.csv",
                              index_col=0, parse_dates=True)
    predictions = pd.read_csv(PROCESSED_DIR / f"lstm_predictions_{split}.csv",
                              index_col=0, parse_dates=True)

    algo  = RL_CONFIG["algorithm"]
    cls   = PPO if algo == "PPO" else SAC
    agent = cls.load(MODEL_DIR / "rl_lstm_agent")

    env    = PortfolioEnvWithPredictions(log_returns, predictions)
    obs, _ = env.reset()
    rewards = []

    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        if done:
            break

    index  = log_returns.index[env.lookback:env.lookback + len(rewards)]
    series = pd.Series(rewards, index=index, name="rl_lstm_agent")
    return series


if __name__ == "__main__":
    returns = evaluate()
    print(returns.describe())
