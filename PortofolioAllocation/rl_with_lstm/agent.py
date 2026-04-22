"""RL agent setup for the LSTM-augmented environment."""

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_with_lstm.environment import PortfolioEnvWithPredictions
from config import RL_CONFIG
import pandas as pd


def make_env(log_returns: pd.DataFrame, predictions: pd.DataFrame):
    return DummyVecEnv([lambda: PortfolioEnvWithPredictions(log_returns, predictions)])


def build_agent(log_returns: pd.DataFrame, predictions: pd.DataFrame):
    env  = make_env(log_returns, predictions)
    algo = RL_CONFIG["algorithm"]

    if algo == "PPO":
        return PPO("MlpPolicy", env, learning_rate=RL_CONFIG["learning_rate"], verbose=1)
    elif algo == "SAC":
        return SAC("MlpPolicy", env, learning_rate=RL_CONFIG["learning_rate"], verbose=1)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
