"""RL agent setup using Stable-Baselines3."""

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.environment import PortfolioEnv
from config import RL_CONFIG
import pandas as pd


def make_env(log_returns: pd.DataFrame):
    return DummyVecEnv([lambda: PortfolioEnv(log_returns)])


def build_agent(log_returns: pd.DataFrame):
    env = make_env(log_returns)
    algo = RL_CONFIG["algorithm"]

    if algo == "PPO":
        return PPO("MlpPolicy", env, learning_rate=RL_CONFIG["learning_rate"], verbose=1)
    elif algo == "SAC":
        return SAC("MlpPolicy", env, learning_rate=RL_CONFIG["learning_rate"], verbose=1)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
