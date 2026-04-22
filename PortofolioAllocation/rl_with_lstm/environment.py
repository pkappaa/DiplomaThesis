"""Portfolio Gym env augmented with LSTM predictions in the state vector."""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from config import TRANSACTION_COST


class PortfolioEnvWithPredictions(gym.Env):
    """
    State:  lookback window of log returns  +  LSTM predictions for the next step
    Action: portfolio weights vector
    Reward: portfolio log return minus transaction cost
    """

    def __init__(self, log_returns: pd.DataFrame, predictions: pd.DataFrame,
                 lookback: int = 20):
        super().__init__()
        # Align on shared index
        shared      = log_returns.index.intersection(predictions.index)
        self.returns     = log_returns.loc[shared].values.astype(np.float32)
        self.predictions = predictions.loc[shared].values.astype(np.float32)
        self.dates       = shared
        self.n_assets    = log_returns.shape[1]
        self.lookback    = lookback

        obs_dim = self.n_assets * lookback + self.n_assets  # returns window + predictions
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(0.0, 1.0, shape=(self.n_assets,), dtype=np.float32)

        self.reset()

    def _obs(self):
        window = self.returns[self.t - self.lookback:self.t].flatten()
        preds  = self.predictions[self.t]
        return np.concatenate([window, preds])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t       = self.lookback
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        return self._obs(), {}

    def step(self, action: np.ndarray):
        w = np.clip(action, 0, None)
        total = w.sum()
        w = w / total if total > 0 else np.ones(self.n_assets) / self.n_assets

        turnover = np.abs(w - self.weights).sum()
        day_ret  = self.returns[self.t] @ w
        reward   = day_ret - TRANSACTION_COST * turnover

        self.weights = w
        self.t      += 1
        done = self.t >= len(self.returns)
        return self._obs(), float(reward), done, False, {}
