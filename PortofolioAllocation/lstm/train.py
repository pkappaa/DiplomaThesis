"""Train the LSTM predictor."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from config import LSTM_CONFIG, RANDOM_SEED
from lstm.model import LSTMPredictor

PROCESSED_DIR = Path("data/processed")
MODEL_DIR     = Path("lstm/checkpoints")


def make_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def train():
    torch.manual_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(exist_ok=True)

    target   = LSTM_CONFIG["target"]
    lookback = LSTM_CONFIG["lookback"]

    train_df = pd.read_csv(PROCESSED_DIR / f"{target}_train.csv", index_col=0)
    val_df   = pd.read_csv(PROCESSED_DIR / f"{target}_val.csv",   index_col=0)

    X_train, y_train = make_sequences(train_df.values, lookback)
    X_val,   y_val   = make_sequences(val_df.values,   lookback)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val,   dtype=torch.float32)
    y_val   = torch.tensor(y_val,   dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=LSTM_CONFIG["batch_size"], shuffle=True)

    model   = LSTMPredictor(X_train.shape[2], LSTM_CONFIG["hidden_size"],
                            LSTM_CONFIG["num_layers"], y_train.shape[1],
                            LSTM_CONFIG["dropout"])
    opt     = torch.optim.Adam(model.parameters(), lr=LSTM_CONFIG["lr"])
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, LSTM_CONFIG["epochs"] + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_DIR / "best.pt")

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | val_loss={val_loss:.6f}")

    print(f"Training complete. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    train()
