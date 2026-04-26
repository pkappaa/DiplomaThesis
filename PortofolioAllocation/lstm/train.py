"""
Expanding-window multi-task LSTM training with 3-seed ensembling.

For each prediction year (2013–2024) and each seed in [42, 123, 777]:
  - Train from scratch on expanding train set (2010 → pred_year-1)
  - Val set = last 26 weeks of that training window (early stopping only)
  - Early stopping (patience=10) on mean val AUC across 9 asset heads
  - Checkpoint: lstm/checkpoints/window_YEAR_seed_SEED.pt
  - Minimum 3 years of training data required (start from pred_year=2013)

Run from the PortofolioAllocation/ directory:
    python -m lstm.train
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lstm.dataset import (
    build_sequences,
    expanding_zscore,
    load_data,
)
from lstm.model import LSTMClassifier, MultiTaskBCE

# ── hyperparameters ───────────────────────────────────────────────────────────
LOOKBACK     = 240
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
NUM_HEADS    = 2
BATCH_SIZE   = 256
MAX_EPOCHS   = 100
PATIENCE     = 10
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SMOOTHING    = 0.1
SEEDS        = [42, 123, 777]

PREDICTION_YEARS = list(range(2013, 2025))   # 12 windows: 2013–2024
VAL_WEEKS        = 130                       # last ~26 weeks of daily sequences used as val
MIN_TRAIN_WEEKS  = 3 * 52                    # minimum train sequences before val
CKPT_DIR = Path("lstm/checkpoints")


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_loader(X: torch.Tensor, y: torch.Tensor,
                 batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(X, y), batch_size=batch_size,
                      shuffle=shuffle, drop_last=False)


def _mean_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, list[float]]:
    """Per-asset AUC and their mean; skips assets with a single class present."""
    per_asset = []
    for i in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, i])
        yt, yp = y_true[mask, i], y_pred[mask, i]
        if len(np.unique(yt)) < 2:
            continue
        per_asset.append(float(roc_auc_score(yt, yp)))
    mean = float(np.mean(per_asset)) if per_asset else 0.5
    return mean, per_asset


# ── sanity check ──────────────────────────────────────────────────────────────

def _sanity_check(
    X: torch.Tensor,
    y: torch.Tensor,
    week_arr: pd.DatetimeIndex,
    asset_names: list[str],
) -> None:
    sep = "=" * 62
    print(f"\n{sep}\nSANITY CHECK\n{sep}")

    y_np = y.numpy()  # (n_weeks, n_assets)

    print(f"\n[1] Label distribution  (total weeks={len(y_np)})")
    for i, asset in enumerate(asset_names):
        col   = y_np[:, i]
        valid = col[~np.isnan(col)]
        pos   = int(valid.sum())
        print(f"    {asset}: pos={pos}/{len(valid)} ({pos / max(len(valid), 1):.1%})")

    print(f"\n[2] Date alignment  (approx last_feat_day  →  label_day)")
    for idx in [0, len(X) // 2, len(X) - 1]:
        label_wk  = week_arr[idx]
        last_feat = label_wk - pd.tseries.offsets.BDay(1)
        print(f"    [{idx:5d}]  last_feat≈{last_feat.date()}  "
              f"→  label={label_wk.date()}")
    print(f"{sep}\n")


# ── single-window, single-seed training ──────────────────────────────────────

def train_window(
    X_tr:        torch.Tensor,
    y_tr:        torch.Tensor,
    X_val:       torch.Tensor,
    y_val:       torch.Tensor,
    n_features:  int,
    n_assets:    int,
    asset_names: list[str],
    ckpt_path:   Path,
    year:        int,
    seed:        int,
    device:      torch.device,
) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LSTMClassifier(
        input_size=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_heads=NUM_HEADS,
        n_assets=n_assets,
    ).to(device)

    loss_fn = MultiTaskBCE(smoothing=SMOOTHING)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    loader  = _make_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)

    X_val_d  = X_val.to(device)
    y_val_d  = y_val.to(device)
    y_val_np = y_val.numpy()

    best_auc   = -1.0
    no_improve = 0

    pbar = tqdm(range(1, MAX_EPOCHS + 1), desc=f"W{year} s{seed}", unit="ep")
    for epoch in pbar:
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(y_tr)

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_pred_t = model(X_val_d)
        val_pred = val_pred_t.cpu().numpy()
        val_loss = loss_fn(val_pred_t, y_val_d).item()
        val_auc, per_asset_aucs = _mean_auc(y_val_np, val_pred)

        sched.step()
        pbar.set_postfix(
            tr_loss=f"{tr_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_auc=f"{val_auc:.4f}",
        )
        if epoch % 10 == 0 and per_asset_aucs:
            asset_str = "  ".join(
                f"{asset_names[i]}={per_asset_aucs[i]:.3f}"
                for i in range(min(len(per_asset_aucs), len(asset_names)))
            )
            tqdm.write(f"  ep{epoch:3d} per-asset AUC: {asset_str}")

        # ── early stopping on mean val AUC ──────────────────────────────────
        if val_auc > best_auc:
            best_auc   = val_auc
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                tqdm.write(f"  Early stop ep{epoch}  best_auc={best_auc:.4f}")
                break

    tqdm.write(f"  W{year} s{seed}: best val AUC = {best_auc:.4f}")
    return best_auc


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computing device: {device}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    daily, labels, _ = load_data()
    daily_norm = expanding_zscore(daily, date_col="date")

    print("Building multi-task 240-step daily sequences...")
    X, y, wk_list, asset_names = build_sequences(daily_norm, labels, LOOKBACK)
    week_arr   = pd.DatetimeIndex(wk_list)
    n_features = X.shape[2]   # 9 assets × 11 features = 99
    n_assets   = y.shape[1]   # 9

    print(f"Dataset: {len(X)} days | seq_len={X.shape[1]} | features={n_features} | assets={n_assets}")
    print(f"Asset order: {asset_names}")

    _sanity_check(X, y, week_arr, asset_names)

    for pred_year in PREDICTION_YEARS:
        cutoff     = pd.Timestamp(f"{pred_year}-01-01")
        avail_mask = week_arr < cutoff
        avail_idx  = np.where(avail_mask)[0]   # integer positions into X/y/week_arr

        n_avail = len(avail_idx)
        if n_avail < MIN_TRAIN_WEEKS + VAL_WEEKS:
            print(f"\nWindow {pred_year}: only {n_avail} sequences available — "
                  f"skipping (need ≥ {MIN_TRAIN_WEEKS + VAL_WEEKS}).")
            continue

        # Val = last VAL_WEEKS sequences within the training window
        val_cutoff = week_arr[avail_idx[-VAL_WEEKS]]
        tr_mask    = week_arr < val_cutoff
        val_mask   = (week_arr >= val_cutoff) & avail_mask

        n_tr  = int(tr_mask.sum())
        n_val = int(val_mask.sum())
        print(f"\nWindow {pred_year}: train={n_tr} weeks, val={n_val} weeks "
              f"(val starts {val_cutoff.date()})")

        if n_tr < BATCH_SIZE or n_val == 0:
            print("  Skipping — insufficient data.")
            continue

        for seed in SEEDS:
            ckpt_path = CKPT_DIR / f"window_{pred_year}_seed_{seed}.pt"
            train_window(
                X[tr_mask], y[tr_mask],
                X[val_mask], y[val_mask],
                n_features, n_assets, asset_names,
                ckpt_path, pred_year, seed, device,
            )

    print("\nAll windows and seeds trained.")


if __name__ == "__main__":
    main()
