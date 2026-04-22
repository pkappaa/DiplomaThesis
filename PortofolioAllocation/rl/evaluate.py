"""Run the trained RL agent on the test split and collect portfolio returns."""

import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, SAC
from rl.environment import PortfolioEnv
from config import RL_CONFIG

PROCESSED_DIR = Path("data/processed")
MODEL_DIR     = Path("rl/checkpoints")


def evaluate(split: str = "test") -> pd.Series:
    log_returns = pd.read_csv(PROCESSED_DIR / f"log_returns_{split}.csv",
                              index_col=0, parse_dates=True)

    algo  = RL_CONFIG["algorithm"]
    cls   = PPO if algo == "PPO" else SAC
    agent = cls.load(MODEL_DIR / "rl_agent")

    env  = PortfolioEnv(log_returns)
    obs, _ = env.reset()
    rewards = []

    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        if done:
            break

    index = log_returns.index[env.lookback:env.lookback + len(rewards)]
    series = pd.Series(rewards, index=index, name="rl_agent")
    return series


if __name__ == "__main__":
    returns = evaluate()
    print(returns.describe())
