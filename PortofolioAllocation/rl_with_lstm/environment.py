"""
Monthly-rebalancing portfolio gymnasium environment — Phase 2 (RL + LSTM).

Observation space : Box(-inf, inf, (116,), float32)
  [0:19]   monthly_returns   — sum of daily log returns in previous month, z-scored
  [19:38]  rolling_vol_21d   — vol_21d at end of previous month, z-scored
  [38:57]  return_ranks      — cross-sectional rank of monthly returns, scaled [-1,1]
  [57:76]  current_weights   — portfolio weights from last step
  [76]     spy_monthly_return — SPY log return last month, z-scored
  [77]     market_vol        — 21-day vol of equal-weight portfolio, z-scored
  [78:97]  lstm_probs_mean   — mean of LSTM daily probabilities over 21 days before t
  [97:116] lstm_probs_std    — std  of LSTM daily probabilities over 21 days before t

Action space : Box(0, 1, (19,), float32)
  SB3 outputs values squashed to [0,1]; softmax is applied INSIDE step() to
  produce valid portfolio weights (sum=1, all>0).

NO LOOKAHEAD: LSTM features at rebalancing date t use only the 21 trading days
strictly before t (same guarantee as the price-based features).
"""

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from config import ASSETS, TRANSACTION_COST

_BASE          = Path(__file__).parent.parent
_RETURNS_PATH  = _BASE / "data" / "processed" / "daily_returns.csv"
_LSTM_PATH     = _BASE / "data" / "processed" / "lstm_probabilities.csv"

N_OBS          = 116   # 78 base + 19 lstm_mean + 19 lstm_std
_NEUTRAL_PROB  = 1.0 / 19  # 0.0526...


class PortfolioEnv(gym.Env):
    """
    Monthly-rebalancing portfolio environment with LSTM probability features.

    Parameters
    ----------
    start_date   : first rebalancing month (inclusive), e.g. "2022-01-01"
    end_date     : last  rebalancing month (inclusive), e.g. "2023-12-31"
    assets       : ordered list of tickers; defaults to ASSETS from config
    returns_path : path to daily_returns.csv
    lstm_path    : path to lstm_probabilities.csv
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        start_date: str = "2022-01-01",
        end_date: str = "2023-12-31",
        assets: Optional[List[str]] = None,
        returns_path: Optional[str] = None,
        lstm_path: Optional[str] = None,
    ):
        super().__init__()

        self.assets      = list(assets or ASSETS)
        self.n_assets    = len(self.assets)
        self.start_date  = pd.Timestamp(start_date)
        self.end_date    = pd.Timestamp(end_date)

        self._build(Path(returns_path) if returns_path else _RETURNS_PATH)
        self._build_lstm(Path(lstm_path) if lstm_path else _LSTM_PATH)
        self._filter_episode_dates()

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(N_OBS,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            0.0, 1.0, shape=(self.n_assets,), dtype=np.float32
        )

        self.current_step    = 0
        self.current_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._neg_ep_returns: List[float] = []

    # ── data pipeline ──────────────────────────────────────────────────────────

    def _build(self, returns_path: Path) -> None:
        """Load CSV, compute monthly features, expanding normalisation stats."""
        panel = pd.read_csv(returns_path, parse_dates=["date"])

        ret_w = panel.pivot(index="date", columns="ticker", values="ret_1d")[self.assets]
        vol_w = panel.pivot(index="date", columns="ticker", values="vol_21d")[self.assets]

        spy_df = (panel[panel["ticker"] == self.assets[0]]
                  .set_index("date")[["spy_ret_1d"]])

        ew_ret   = ret_w.mean(axis=1)
        ew_vol21 = ew_ret.rolling(21, min_periods=5).std()

        all_dates = pd.DatetimeIndex(sorted(ret_w.index.unique()))

        self._all_reb = self._first_of_months(all_dates)
        n = len(self._all_reb)

        prev_ret   = np.zeros((n, self.n_assets), dtype=np.float32)
        prev_vol   = np.zeros((n, self.n_assets), dtype=np.float32)
        prev_ranks = np.zeros((n, self.n_assets), dtype=np.float32)
        spy_ret_m  = np.zeros(n, dtype=np.float32)
        mkt_vol_m  = np.zeros(n, dtype=np.float32)
        fwd_ret    = np.zeros((n, self.n_assets), dtype=np.float32)

        self._prev_ends    = np.empty(n, dtype=object)
        self._reward_start = np.empty(n, dtype=object)
        self._reward_end   = np.empty(n, dtype=object)

        for i, reb in enumerate(self._all_reb):
            if i > 0:
                prev_reb  = self._all_reb[i - 1]
                prev_mask = (all_dates >= prev_reb) & (all_dates < reb)
            else:
                one_mo_ago = reb - pd.DateOffset(months=1)
                prev_mask  = (all_dates >= one_mo_ago) & (all_dates < reb)

            prev_dates = all_dates[prev_mask]

            if len(prev_dates) > 0:
                prev_ret[i]  = ret_w.loc[prev_dates].sum().values.astype(np.float32)
                prev_vol[i]  = vol_w.loc[prev_dates[-1]].values.astype(np.float32)
                spy_ret_m[i] = float(
                    spy_df.loc[spy_df.index.intersection(prev_dates), "spy_ret_1d"].sum()
                )
                ev = ew_vol21.reindex(prev_dates).iloc[-1]
                mkt_vol_m[i] = 0.0 if np.isnan(ev) else float(ev)
                self._prev_ends[i] = prev_dates[-1]
            else:
                self._prev_ends[i] = None

            if len(prev_dates) > 0 and self.n_assets > 1:
                r = pd.Series(prev_ret[i]).rank(method="average").values
                prev_ranks[i] = (2.0 * (r - 1.0) / (self.n_assets - 1.0) - 1.0).astype(
                    np.float32
                )

            if i + 1 < n:
                next_reb  = self._all_reb[i + 1]
                curr_mask = (all_dates >= reb) & (all_dates < next_reb)
            else:
                curr_mask = all_dates >= reb

            curr_dates = all_dates[curr_mask]
            if len(curr_dates) > 0:
                fwd_ret[i]            = ret_w.loc[curr_dates].sum().values.astype(np.float32)
                self._reward_start[i] = curr_dates[0]
                self._reward_end[i]   = curr_dates[-1]
            else:
                self._reward_start[i] = reb
                self._reward_end[i]   = reb

        idx = pd.DatetimeIndex(self._all_reb)
        self._prev_ret   = pd.DataFrame(prev_ret,   index=idx, columns=self.assets)
        self._prev_vol   = pd.DataFrame(prev_vol,   index=idx, columns=self.assets)
        self._prev_ranks = pd.DataFrame(prev_ranks, index=idx, columns=self.assets)
        self._spy_ret_m  = pd.Series(spy_ret_m,     index=idx, name="spy")
        self._mkt_vol_m  = pd.Series(mkt_vol_m,     index=idx, name="mktv")
        self._fwd_ret    = pd.DataFrame(fwd_ret,    index=idx, columns=self.assets)

        _kw = dict(min_periods=2)
        self._norm = {
            "ret_mu": self._prev_ret.expanding(**_kw).mean().shift(1).fillna(0.0),
            "ret_sd": (self._prev_ret.expanding(**_kw).std().shift(1)
                       .fillna(1.0).replace(0.0, 1.0)),
            "vol_mu": self._prev_vol.expanding(**_kw).mean().shift(1).fillna(0.0),
            "vol_sd": (self._prev_vol.expanding(**_kw).std().shift(1)
                       .fillna(1.0).replace(0.0, 1.0)),
            "spy_mu": self._spy_ret_m.expanding(**_kw).mean().shift(1).fillna(0.0),
            "spy_sd": (self._spy_ret_m.expanding(**_kw).std().shift(1)
                       .fillna(1.0).replace(0.0, 1.0)),
            "mv_mu":  self._mkt_vol_m.expanding(**_kw).mean().shift(1).fillna(0.0),
            "mv_sd":  (self._mkt_vol_m.expanding(**_kw).std().shift(1)
                       .fillna(1.0).replace(0.0, 1.0)),
        }

    def _build_lstm(self, lstm_path: Path) -> None:
        """
        Precompute per-rebalancing-date LSTM mean/std features.

        For each rebalancing date t: take the 21 trading days strictly before t
        from lstm_probabilities.csv.  If no days are available (before LSTM
        coverage starts), fill with neutral prior (1/19 for mean, 0.0 for std).
        """
        lstm_df = pd.read_csv(lstm_path, parse_dates=["date"]).set_index("date")
        lstm_df = lstm_df[self.assets]  # reorder columns to match ASSETS order

        n = len(self._all_reb)
        lstm_mean = np.full((n, self.n_assets), _NEUTRAL_PROB, dtype=np.float32)
        lstm_std  = np.zeros((n, self.n_assets), dtype=np.float32)

        for i, reb in enumerate(self._all_reb):
            reb_ts = pd.Timestamp(reb)
            avail  = lstm_df.loc[lstm_df.index < reb_ts]
            if len(avail) >= 1:
                window       = avail.iloc[-21:]
                lstm_mean[i] = window.mean(axis=0).values.astype(np.float32)
                if len(window) >= 2:
                    lstm_std[i] = window.std(axis=0).values.astype(np.float32)

        idx = pd.DatetimeIndex(self._all_reb)
        self._lstm_mean = pd.DataFrame(lstm_mean, index=idx, columns=self.assets)
        self._lstm_std  = pd.DataFrame(lstm_std,  index=idx, columns=self.assets)

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _first_of_months(dates: pd.DatetimeIndex) -> np.ndarray:
        periods = dates.to_period("M")
        result  = [dates[periods == p][0] for p in sorted(periods.unique())]
        return np.array(result)

    def _filter_episode_dates(self) -> None:
        reb  = pd.DatetimeIndex(self._all_reb)
        mask = (reb >= self.start_date) & (reb <= self.end_date)
        self._reb_dates = reb[mask]
        self.n_steps    = len(self._reb_dates)

    def _softmax(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        a = a - a.max()
        e = np.exp(a)
        return (e / e.sum()).astype(np.float32)

    def _get_obs(self, step_idx: int) -> np.ndarray:
        date = self._reb_dates[step_idx]

        ret = self._prev_ret.loc[date].values.astype(np.float64)
        vol = self._prev_vol.loc[date].values.astype(np.float64)
        rnk = self._prev_ranks.loc[date].values.astype(np.float32)
        spy = float(self._spy_ret_m.loc[date])
        mkv = float(self._mkt_vol_m.loc[date])

        def _z(x, mu, sd):
            return np.clip((x - mu) / np.where(sd == 0, 1.0, sd), -3.0, 3.0)

        z_ret = _z(ret,
                   self._norm["ret_mu"].loc[date].values,
                   self._norm["ret_sd"].loc[date].values).astype(np.float32)
        z_vol = _z(vol,
                   self._norm["vol_mu"].loc[date].values,
                   self._norm["vol_sd"].loc[date].values).astype(np.float32)

        spy_mu = float(self._norm["spy_mu"].loc[date])
        spy_sd = float(self._norm["spy_sd"].loc[date])
        z_spy  = float(np.clip((spy - spy_mu) / (spy_sd if spy_sd != 0 else 1.0), -3.0, 3.0))

        mv_mu = float(self._norm["mv_mu"].loc[date])
        mv_sd = float(self._norm["mv_sd"].loc[date])
        z_mkv = float(np.clip((mkv - mv_mu) / (mv_sd if mv_sd != 0 else 1.0), -3.0, 3.0))

        lstm_m = self._lstm_mean.loc[date].values.astype(np.float32)
        lstm_s = self._lstm_std.loc[date].values.astype(np.float32)

        return np.concatenate([
            z_ret,                                        # (19,) monthly returns
            z_vol,                                        # (19,) rolling vol
            rnk,                                          # (19,) return ranks
            self.current_weights,                         # (19,) current weights
            np.array([z_spy, z_mkv], dtype=np.float32),  # (2,)  market features
            lstm_m,                                       # (19,) LSTM prob mean
            lstm_s,                                       # (19,) LSTM prob std
        ]).astype(np.float32)

    # ── gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step    = 0
        self.current_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._neg_ep_returns = []
        return self._get_obs(0), {}

    def step(self, action: np.ndarray):
        weights  = self._softmax(action)
        reb_date = self._reb_dates[self.current_step]

        fwd       = self._fwd_ret.loc[reb_date].values
        port_ret  = float(weights @ fwd)
        turnover  = float(np.abs(weights - self.current_weights).sum())
        cost      = TRANSACTION_COST * turnover
        net_ret   = port_ret - cost

        if net_ret < 0:
            self._neg_ep_returns.append(net_ret)
        n_neg = len(self._neg_ep_returns)
        downside_std = float(np.std(self._neg_ep_returns)) if n_neg >= 3 else 0.0

        reward = net_ret - 0.1 * downside_std

        self.current_weights = weights
        self.current_step   += 1
        terminated = self.current_step >= self.n_steps

        obs = self._get_obs(min(self.current_step, self.n_steps - 1))

        info = {
            "weights":      weights,
            "return":       net_ret,
            "gross_return": port_ret,
            "turnover":     turnover,
            "date":         str(reb_date.date()),
        }
        return obs, float(reward), terminated, False, info
