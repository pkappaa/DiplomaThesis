"""
Load per-window LSTM checkpoints (3 seeds), ensemble-average probabilities,
and generate weekly probability predictions for all 12 walk-forward windows.

Windows: 2013–2024 (one model per prediction year).
Sections saved to data/processed/lstm_probabilities.csv:
  rl_train : 2013–2021  (windows 1–9,  for RL training)
  val      : 2022–2023  (windows 10–11, LSTM evaluation)
  test     : 2024       (window 12,     final evaluation)

AUC is reported only for val and test sections.

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
    build_sequences,
    expanding_zscore,
    load_data,
)
from lstm.model import LSTMClassifier

PROCESSED = Path("data/processed")
CKPT_DIR  = Path("lstm/checkpoints")

LOOKBACK         = 240
HIDDEN_SIZE      = 128
NUM_LAYERS       = 2
DROPOUT          = 0.3
NUM_HEADS        = 2
SEEDS            = [42, 123, 777]
PREDICTION_YEARS = [2022, 2023, 2024]        # val + test only

SPLIT_MAP: dict[int, str] = {2022: "val", 2023: "val", 2024: "test"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_model(
    ckpt_path: Path, n_features: int, n_assets: int, device: torch.device
) -> LSTMClassifier:
    model = LSTMClassifier(
        input_size=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_heads=NUM_HEADS,
        n_assets=n_assets,
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    return model


def _print_metrics(result_df: pd.DataFrame, asset_names: list[str], split_name: str):
    """Print per-asset and overall AUC/accuracy for a subset of result_df."""
    if result_df.empty:
        return
    print(f"\n{split_name.upper()}  (n={len(result_df)} days)")
    aucs = []
    for asset in asset_names:
        yt   = result_df[f"label_{asset}"].values
        yp   = result_df[asset].values
        mask = ~np.isnan(yt)
        if mask.sum() < 2 or len(np.unique(yt[mask])) < 2:
            continue
        auc = roc_auc_score(yt[mask], yp[mask])
        acc = accuracy_score(yt[mask], (yp[mask] >= 0.5).astype(int))
        print(f"  {asset}  AUC={auc:.4f}  Acc={acc:.4f}")
        aucs.append(auc)
    if aucs:
        print(f"  Overall mean AUC={np.mean(aucs):.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computing device: {device}")

    print("Loading data...")
    daily, labels, _ = load_data()
    daily_norm = expanding_zscore(daily, date_col="date")

    print("Building sequences...")
    X, y, wk_list, asset_names = build_sequences(daily_norm, labels, LOOKBACK)
    week_arr   = pd.DatetimeIndex(wk_list)
    n_features = X.shape[2]
    n_assets   = y.shape[1]
    y_np       = y.numpy()

    print(f"Asset order: {asset_names}")

    all_rows: list[dict] = []

    for pred_year in PREDICTION_YEARS:
        ckpt_paths = [CKPT_DIR / f"window_{pred_year}_seed_{seed}.pt" for seed in SEEDS]
        available  = [p for p in ckpt_paths if p.exists()]
        if not available:
            print(f"No checkpoints found for window {pred_year} — skipping.")
            continue

        mask = (
            (week_arr >= pd.Timestamp(f"{pred_year}-01-01"))
            & (week_arr <= pd.Timestamp(f"{pred_year}-12-31"))
        )
        if not mask.any():
            continue

        X_pred  = X[mask].to(device)
        y_pred  = y_np[mask]
        wk_pred = week_arr[mask]

        # Ensemble: average probabilities across all available seeds
        seed_preds = []
        for ckpt in available:
            model = _load_model(ckpt, n_features, n_assets, device)
            with torch.no_grad():
                seed_preds.append(model(X_pred).cpu().numpy())
        probs = np.mean(seed_preds, axis=0)

        split = SPLIT_MAP[pred_year]
        for wi, wk in enumerate(wk_pred):
            row: dict = {"date": wk}
            for ai, asset in enumerate(asset_names):
                row[asset]            = float(probs[wi, ai])
                row[f"label_{asset}"] = float(y_pred[wi, ai])
            all_rows.append(row)

        print(f"Window {pred_year} [{split}]: {int(mask.sum())} days, "
              f"{len(available)}/{len(SEEDS)} seeds ensembled.")

    if not all_rows:
        print("No predictions generated — did you run lstm/train.py first?")
        return

    result_df = pd.DataFrame(all_rows).set_index("date").sort_index()

    # ── save probabilities: daily dates × 9 ETF tickers, 2022–2024 only ──────
    probs_out = result_df[asset_names]
    probs_out.index.name = "date"
    probs_out.to_csv(PROCESSED / "lstm_probabilities.csv")
    print(f"\nSaved lstm_probabilities.csv  ({len(probs_out)} days × {n_assets} assets)")

    # ── daily counts per section ──────────────────────────────────────────────
    print("\nDaily predictions per section:")
    for section in ["val", "test"]:
        mask_s = np.array([SPLIT_MAP.get(d.year) == section for d in result_df.index])
        print(f"  {section:6s}: {mask_s.sum()} days")

    # ── AUC for val (2022–2023) and test (2024) separately ───────────────────
    val_mask  = result_df.index.year.isin([2022, 2023])
    test_mask = result_df.index.year == 2024
    _print_metrics(result_df[val_mask],  asset_names, "val (2022-2023)")
    _print_metrics(result_df[test_mask], asset_names, "test (2024)")


if __name__ == "__main__":
    main()
