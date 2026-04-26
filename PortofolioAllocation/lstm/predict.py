"""
Load per-window LSTM checkpoints and generate cross-sectional probability
predictions for every week in each prediction window.

Run from the PortofolioAllocation/ directory:
    python -m lstm.predict
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lstm.dataset import (
    aggregate_to_weekly,
    build_sequences, 
    expanding_zscore, 
    load_data
)
from lstm.model import LSTMClassifier

PROCESSED = Path("data/processed")
CKPT_DIR  = Path("lstm/checkpoints")

LOOKBACK         = 52  # FIX 1: Weekly steps
HIDDEN_SIZE      = 128
NUM_LAYERS       = 2
DROPOUT          = 0.3
NUM_HEADS        = 2
PREDICTION_YEARS = [2022, 2023, 2024]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_model(ckpt_path: Path, n_features: int, device: torch.device) -> LSTMClassifier:
    model = LSTMClassifier(
        input_size=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_heads=NUM_HEADS,
    )
    # FIX 3: map_location to mapped device
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model

def _print_metrics(sub: pd.DataFrame, split_name: str):
    if sub.empty:
        return
    auc = roc_auc_score(sub["label"], sub["prob"])
    acc = accuracy_score(sub["label"], (sub["prob"] >= 0.5).astype(int))
    print(f"\n{split_name.upper()}  (n={len(sub)})")
    print(f"  Overall  AUC-ROC={auc:.4f}  Accuracy={acc:.4f}  Hit-rate={acc:.4f}")
    for asset in sorted(sub["ticker"].unique()):
        a = sub[sub["ticker"] == asset]
        a_auc = roc_auc_score(a["label"], a["prob"])
        a_acc = accuracy_score(a["label"], (a["prob"] >= 0.5).astype(int))
        print(f"    {asset}  AUC={a_auc:.4f}  Acc={a_acc:.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # FIX 3: Detect and establish device for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computing device: {device}")

    print("Loading data and aggregating to weekly frequency...")
    daily, labels, splits = load_data()
    # FIX 1: Pipeline matches train.py
    weekly = aggregate_to_weekly(daily)
    weekly_norm = expanding_zscore(weekly, date_col="week_end")

    print("Building sequences...")
    X, y, wk_list, asset_list = build_sequences(weekly_norm, labels, LOOKBACK)
    week_arr  = pd.DatetimeIndex(wk_list)
    asset_arr = np.array(asset_list)
    n_features = X.shape[2]

    rows = []

    for pred_year in PREDICTION_YEARS:
        ckpt = CKPT_DIR / f"window_{pred_year}.pt"
        if not ckpt.exists():
            print(f"Checkpoint not found: {ckpt} — skipping window {pred_year}.")
            continue

        model = _load_model(ckpt, n_features, device)

        mask = (
            (week_arr >= pd.Timestamp(f"{pred_year}-01-01"))
            & (week_arr <= pd.Timestamp(f"{pred_year}-12-31"))
        )
        if not mask.any():
            continue

        # FIX 3: Cast batch to GPU
        X_pred  = X[mask].to(device) 
        y_pred  = y[mask].numpy()
        wk_pred = week_arr[mask]
        as_pred = asset_arr[mask]

        with torch.no_grad():
            probs = model(X_pred).cpu().numpy()

        for wk, asset, prob, lbl in zip(wk_pred, as_pred, probs, y_pred):
            rows.append({"week_end": wk, "ticker": asset,
                         "prob": float(prob), "label": float(lbl)})

        print(f"Window {pred_year}: {mask.sum()} predictions generated.")

    if not rows:
        print("No predictions generated — did you run lstm/train.py first?")
        return

    prob_df = pd.DataFrame(rows)

    # ── wide-format outputs ───────────────────────────────────────────────────
    probs_wide = prob_df.pivot(index="week_end", columns="ticker", values="prob")
    probs_wide.columns.name = None
    probs_wide = probs_wide[sorted(probs_wide.columns)]
    probs_wide.index.name = "week_end"

    probs_wide.to_csv(PROCESSED / "lstm_probabilities.csv")
    (probs_wide >= 0.5).astype(int).to_csv(PROCESSED / "lstm_binary.csv")
    print(
        f"\nSaved lstm_probabilities.csv  "
        f"({len(probs_wide)} weeks × {probs_wide.shape[1]} assets)"
    )

    # ── evaluation metrics ────────────────────────────────────────────────────
    val_start  = pd.Timestamp(splits["val"]["start"])
    val_end    = pd.Timestamp(splits["val"]["end"])
    test_start = pd.Timestamp(splits["test"]["start"])
    test_end   = pd.Timestamp(splits["test"]["end"])

    val_mask  = (prob_df["week_end"] >= val_start)  & (prob_df["week_end"] <= val_end)
    test_mask = (prob_df["week_end"] >= test_start) & (prob_df["week_end"] <= test_end)

    _print_metrics(prob_df[val_mask],  "val")
    _print_metrics(prob_df[test_mask], "test")

if __name__ == "__main__":
    main()