"""
Train a PPO agent for monthly portfolio allocation using Stable-Baselines3.

Run from the PortofolioAllocation/ directory:
    python -m rl.train
or
    python rl/train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from rl.environment import PortfolioEnv

# ── paths ──────────────────────────────────────────────────────────────────────
_RL_DIR      = Path(__file__).parent
CKPT_DIR     = _RL_DIR / "checkpoints"
LOGS_DIR     = _RL_DIR / "logs"
RESULTS_DIR  = _RL_DIR / "results"

# ── PPO hyperparameters ────────────────────────────────────────────────────────
N_STEPS_PER_EPISODE  = 24         # months per episode (2022-01 to 2023-12)
N_EPISODES_PER_ROLL  = 20         # episodes per rollout
N_STEPS              = N_STEPS_PER_EPISODE * N_EPISODES_PER_ROLL  # 480
TOTAL_TIMESTEPS      = 500 * N_STEPS                               # 240 000
SEED                 = 42

PPO_KWARGS = dict(
    learning_rate  = 3e-4,
    n_steps        = N_STEPS,
    batch_size     = N_STEPS,          # full rollout buffer
    n_epochs       = 10,
    gamma          = 0.99,
    gae_lambda     = 0.95,
    clip_range     = 0.2,
    ent_coef       = 0.01,
    max_grad_norm  = 0.5,
    seed           = SEED,
)

POLICY_KWARGS = dict(
    net_arch      = dict(pi=[256, 128], vf=[256, 128]),
    activation_fn = nn.GELU,
)

TRAIN_START = "2022-01-01"
TRAIN_END   = "2023-12-31"


# ── progress callback ──────────────────────────────────────────────────────────

class _ProgressCB(BaseCallback):
    """Print key metrics every `log_freq` rollout updates."""

    def __init__(self, log_freq: int = 50, n_steps: int = N_STEPS):
        super().__init__(verbose=0)
        self._log_freq  = log_freq
        self._n_steps   = n_steps
        self._update    = 0

    def _on_rollout_start(self) -> None:
        # Called AFTER the previous update's train() completed; logger is current.
        if self._update > 0 and self._update % self._log_freq == 0:
            vals   = self.model.logger.name_to_value
            buf    = self.model.ep_info_buffer
            mean_r = np.mean([e["r"] for e in buf]) if buf else float("nan")
            print(
                f"  [update {self._update:4d}] "
                f"ep_reward={mean_r:+.4f} | "
                f"entropy={vals.get('train/entropy_loss', float('nan')):.4f} | "
                f"pg_loss={vals.get('train/policy_gradient_loss', float('nan')):.4f} | "
                f"vf_loss={vals.get('train/value_loss', float('nan')):.4f}"
            )
        self._update += 1

    def _on_step(self) -> bool:
        return True


# ── sanity checks ──────────────────────────────────────────────────────────────

def run_sanity_checks() -> None:
    """Six pre-training sanity checks; aborts with AssertionError on failure."""
    print("\n" + "=" * 60)
    print("  SANITY CHECKS")
    print("=" * 60)

    env = PortfolioEnv(start_date=TRAIN_START, end_date=TRAIN_END)

    # 1 ── episode length ────────────────────────────────────────────────────
    obs, _ = env.reset()
    steps  = 0
    done   = False
    while not done:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    assert steps == 24, f"Expected 24 steps, got {steps}"
    print(f"\n1. Episode length : {steps} steps  [OK]")

    # 2 ── observation shape ─────────────────────────────────────────────────
    obs, _ = env.reset()
    assert obs.shape == (78,), f"Expected (78,), got {obs.shape}"
    print(f"2. Obs shape      : {obs.shape}  [OK]")

    # 3 ── softmax inside step() → valid weights ─────────────────────────────
    raw = np.random.randn(19).astype(np.float32)
    _, _, _, _, info = env.step(raw)
    w = info["weights"]
    assert abs(w.sum() - 1.0) < 1e-5, f"Weights sum={w.sum():.6f}"
    assert (w > 0).all(), "Some weights <= 0"
    print(f"3. Weights sum    : {w.sum():.6f}, all positive  [OK]")

    # 4 ── no lookahead: print dates for first step ──────────────────────────
    env.reset()
    reb0        = env._reb_dates[0]
    state_end   = env._prev_ends[list(env._all_reb).index(reb0)]
    reward_s    = env._reward_start[list(env._all_reb).index(reb0)]
    reward_e    = env._reward_end[list(env._all_reb).index(reb0)]
    print(f"4. No-lookahead check (step 0):")
    print(f"   State uses data up to: {state_end}")
    print(f"   Reward computed over : {reward_s}  to  {reward_e}")
    assert pd.Timestamp(state_end) < pd.Timestamp(reward_s), (
        "LOOKAHEAD DETECTED: state overlaps with reward period!"
    )
    print(f"   No lookahead  [OK]")

    # 5 ── reward range over random episodes ─────────────────────────────────
    ep_rewards = []
    for _ in range(5):
        obs, _ = env.reset()
        ep_r   = 0.0
        done   = False
        while not done:
            action = env.action_space.sample()
            _, r, terminated, truncated, _ = env.step(action)
            ep_r += r
            done  = terminated or truncated
        ep_rewards.append(ep_r)
    print(
        f"5. Random-policy ep reward : "
        f"min={min(ep_rewards):+.4f}  "
        f"mean={np.mean(ep_rewards):+.4f}  "
        f"max={max(ep_rewards):+.4f}"
    )

    # 6 ── equal-weight baseline ──────────────────────────────────────────────
    obs, _ = env.reset()
    ew_act  = np.zeros(19, dtype=np.float32)   # softmax([0,...,0]) = 1/19 each
    ew_rets = []
    done    = False
    while not done:
        _, _, terminated, truncated, info = env.step(ew_act)
        ew_rets.append(info["return"])
        done = terminated or truncated
    ew_arr    = np.array(ew_rets)
    ew_sharpe = float(ew_arr.mean() / (ew_arr.std() + 1e-8) * np.sqrt(12))
    ew_total  = float(ew_arr.sum())
    print(
        f"6. Equal-weight baseline   : "
        f"total_reward={ew_total:+.4f}  "
        f"monthly_sharpe={ew_sharpe:+.4f}"
    )

    print("\n  All checks passed.\n")




# ── training ───────────────────────────────────────────────────────────────────

def train() -> None:
    # Directories
    for d in (CKPT_DIR, LOGS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Sanity checks
    run_sanity_checks()

    print("=" * 60)
    print("  TRAINING  — PPO  (SB3)")
    print(f"  train   : {TRAIN_START} to {TRAIN_END}  (24 monthly steps)")
    print(f"  updates : 500  (n_steps={N_STEPS}, n_epochs=10)")
    print(f"  total   : {TOTAL_TIMESTEPS:,} timesteps")
    print("=" * 60 + "\n")

    # Environments
    train_env = DummyVecEnv([lambda: PortfolioEnv(TRAIN_START, TRAIN_END)])
    eval_env  = PortfolioEnv(TRAIN_START, TRAIN_END)

    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(CKPT_DIR),
        log_path             = str(LOGS_DIR),
        eval_freq            = N_STEPS,          # every rollout (every 20 episodes)
        n_eval_episodes      = 1,
        deterministic        = True,
        verbose              = 0,
    )
    progress_cb = _ProgressCB(log_freq=50, n_steps=N_STEPS)

    # Model
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs  = POLICY_KWARGS,
        tensorboard_log= str(LOGS_DIR),
        verbose        = 1,
        **PPO_KWARGS,
    )

    # Train
    model.learn(
        total_timesteps    = TOTAL_TIMESTEPS,
        callback           = [eval_cb, progress_cb],
        tb_log_name        = "ppo_portfolio",
        reset_num_timesteps= True,
    )

    print("\n  Training complete.")
    print(f"  Best model saved to : {CKPT_DIR / 'best_model.zip'}")


if __name__ == "__main__":
    train()
