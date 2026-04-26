"""
Verify expanding z-score fix and save corrected sequences for RL input.

Run from the PortofolioAllocation/ directory:
    python -m lstm.verify_normalization
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lstm.dataset import build_sequences, expanding_zscore, load_data, save_sequences

LOOKBACK = 240


def main():
    print("Loading data...")
    daily, labels, _ = load_data()

    print("Applying expanding z-score normalisation (fixed: floor + clip)...")
    daily_norm = expanding_zscore(daily, date_col="date")

    print("Building sequences (sanity check [4] printed below)...")
    X, y, date_list, asset_names = build_sequences(daily_norm, labels, LOOKBACK)

    x_np = X.numpy()
    lo, hi = float(x_np.min()), float(x_np.max())
    print(f"\n[4 re-check] min={lo:.6f}  max={hi:.6f}")
    if lo >= -10.0 and hi <= 10.0:
        print("    PASS — all values within [-10, 10]")
    else:
        print(f"    FAIL — values outside [-10, 10]: min={lo:.4f} max={hi:.4f}")

    save_sequences(X, y, date_list, asset_names)
    print("\nDone.")


if __name__ == "__main__":
    main()
