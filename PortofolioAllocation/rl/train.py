"""Train the RL agent on the training split."""

import pandas as pd
from pathlib import Path
from config import RL_CONFIG, RANDOM_SEED
from rl.agent import build_agent

PROCESSED_DIR = Path("data/processed")
MODEL_DIR     = Path("rl/checkpoints")


def train():
    MODEL_DIR.mkdir(exist_ok=True)
    log_returns = pd.read_csv(PROCESSED_DIR / "log_returns_train.csv",
                              index_col=0, parse_dates=True)

    agent = build_agent(log_returns)
    agent.learn(total_timesteps=RL_CONFIG["total_timesteps"])
    agent.save(MODEL_DIR / "rl_agent")
    print("RL agent saved.")


if __name__ == "__main__":
    train()
