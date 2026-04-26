"""
Expanding-window LSTM training for cross-sectional weekly beat-median
classification.

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
    FEAT_COLS,
    aggregate_to_weekly,
    build_sequences,
    expanding_zscore,
    load_data,
)
from lstm.model import LSTMClassifier, SmoothedBCE

# ── hyperparameters ───────────────────────────────────────────────────────────
LOOKBACK     = 52  # FIX 1: 52 weekly steps instead of 260 daily
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
NUM_HEADS    = 2
BATCH_SIZE   = 64
MAX_EPOCHS   = 100
PATIENCE     = 10
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SMOOTHING    = 0.1

PREDICTION_YEARS = [2022, 2023, 2024]
CKPT_DIR = Path("lstm/checkpoints")


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_loader(X: torch.Tensor, y: torch.Tensor,
                 batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(X, y), batch_size=batch_size,
                      shuffle=shuffle, drop_last=False)

def _pos_weight(y: torch.Tensor) -> float:
    pos = float(y.sum())
    neg = len(y) - pos
    return neg / max(pos, 1.0)


# ── sanity check (FIX 2) ──────────────────────────────────────────────────────

def _sanity_check(
    X: torch.Tensor,
    y: torch.Tensor,
    week_arr: pd.DatetimeIndex,
    asset_arr: np.ndarray,
) -> None:
    sep = "=" * 62
    print(f"\n{sep}\nSANITY CHECK\n{sep}")

    y_np = y.numpy()

    # 1. Label distribution
    n_pos  = int(y_np.sum())
    n_neg  = len(y_np) - n_pos
    print(f"\n[1] Label distribution  (total N={len(y_np)})")
    print(f"    pos=1 : {n_pos:5d}  ({n_pos/len(y_np):.1%})")
    print(f"    neg=0 : {n_neg:5d}  ({n_neg/len(y_np):.1%})")

    for pred_year in PREDICTION_YEARS:
        vs = pd.Timestamp(f"{pred_year - 1}-10-01")
        ve = pd.Timestamp(f"{pred_year - 1}-12-31")
        m  = (week_arr >= vs) & (week_arr <= ve)
        yv = y_np[m]
        if len(yv) == 0:
            continue
        pos = int(yv.sum())
        print(f"    Window {pred_year} val (Q4 {pred_year-1}): "
              f"N={len(yv)}, pos={pos} ({pos/len(yv):.1%})")

    # 2. Feature–label correlations
    tr_mask = week_arr < pd.Timestamp("2022-01-01")
    X_last  = X[tr_mask, -1, :].numpy()   # last weekly timestep
    y_tr    = y_np[tr_mask]
    print(f"\n[2] Pearson r  (last feature week → next-week label, train only)")
    for i, feat in enumerate(FEAT_COLS):
        if y_tr.std() == 0 or X_last[:, i].std() == 0:
            r = float("nan")
        else:
            r = float(np.corrcoef(X_last[:, i], y_tr)[0, 1])
        print(f"    {feat:15s}: r = {r:+.4f}")

    # 3. Date alignment
    print(f"\n[3] Date alignment  (last_feat_week_end  →  label_week_end)")
    sample_indices = [0, len(X) // 2, len(X) - 1]
    for idx in sample_indices:
        label_wk    = week_arr[idx]
        last_feat   = label_wk - pd.Timedelta(days=7)   # previous Friday
        print(f"    [{idx:5d}]  last_feat={last_feat.date()}  "
              f"→  label={label_wk.date()}  "
              f"ticker={asset_arr[idx]}  y={int(y_np[idx])}")
    print(f"{sep}\n")


# ── single-window training ────────────────────────────────────────────────────

def train_window(
    X_tr:  torch.Tensor,
    y_tr:  torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_features: int,
    ckpt_path: Path,
    year: int,
    device: torch.device,
) -> float:
    torch.manual_seed(42)
    np.random.seed(42)

    # FIX 3: Push model to GPU
    model = LSTMClassifier(
        input_size=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_heads=NUM_HEADS,
    ).to(device)

    loss_fn  = SmoothedBCE(smoothing=SMOOTHING)
    pw       = _pos_weight(y_tr)
    opt      = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    loader   = _make_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)

    # FIX 3: Move entire validation set to GPU once (it's small enough)
    X_val_d  = X_val.to(device)
    y_val_d  = y_val.to(device)
    y_val_np = y_val.numpy()

    best_auc   = -1.0
    no_improve = 0

    pbar = tqdm(range(1, MAX_EPOCHS + 1), desc=f"Window {year}", unit="ep")
    for epoch in pbar:
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device) # FIX 3
            opt.zero_grad()
            loss = loss_fn(model(xb), yb, pos_weight=pw)
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
            
        val_loss = loss_fn(val_pred_t, y_val_d, pos_weight=pw).item()
        val_auc = roc_auc_score(y_val_np, val_pred)

        sched.step()
        pbar.set_postfix(
            tr_loss=f"{tr_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_auc=f"{val_auc:.4f}",
        )

        # ── early stopping on AUC ───────────────────────────────────────────
        if val_auc > best_auc:
            best_auc   = val_auc
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                tqdm.write(f"  Early stop epoch {epoch}  best_auc={best_auc:.4f}")
                break

    tqdm.write(f"  Window {year}: best val AUC = {best_auc:.4f}")
    return best_auc


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # FIX 3: Detect and establish device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computing device: {device}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data and aggregating to weekly frequency...")
    daily, labels, _ = load_data()
    # FIX 1: Aggregate to weekly BEFORE expanding Z-score
    weekly      = aggregate_to_weekly(daily)
    weekly_norm = expanding_zscore(weekly, date_col="week_end")

    print("Building 52-step weekly sequences...")
    X, y, wk_list, asset_list = build_sequences(weekly_norm, labels, LOOKBACK)
    week_arr  = pd.DatetimeIndex(wk_list)
    asset_arr = np.array(asset_list)
    n_features = X.shape[2]

    print(f"Dataset: {len(X)} samples | seq_len={X.shape[1]} | features={n_features}")

    # FIX 2: Sanity Check
    _sanity_check(X, y, week_arr, asset_arr)

    # ── expanding-window training ─────────────────────────────────────────────
    for pred_year in PREDICTION_YEARS:
        val_start = pd.Timestamp(f"{pred_year - 1}-10-01")
        val_end   = pd.Timestamp(f"{pred_year - 1}-12-31")

        tr_mask  = week_arr < val_start
        val_mask = (week_arr >= val_start) & (week_arr <= val_end)

        n_tr, n_val = int(tr_mask.sum()), int(val_mask.sum())
        print(f"\nWindow {pred_year}: train={n_tr} samples, val={n_val} samples")

        if n_tr < BATCH_SIZE or n_val == 0:
            print("  Skipping — insufficient data.")
            continue

        train_window(
            X[tr_mask], y[tr_mask],
            X[val_mask], y[val_mask],
            n_features,
            CKPT_DIR / f"window_{pred_year}.pt",
            pred_year,
            device
        )

    print("\nAll windows trained.")

if __name__ == "__main__":
    main()