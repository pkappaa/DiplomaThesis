"""Load trained LSTM and generate predictions on any split."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from config import LSTM_CONFIG
from lstm.model import LSTMPredictor
from lstm.train import make_sequences

PROCESSED_DIR = Path("data/processed")
MODEL_DIR     = Path("lstm/checkpoints")


def load_model(n_assets: int) -> LSTMPredictor:
    model = LSTMPredictor(n_assets, LSTM_CONFIG["hidden_size"],
                          LSTM_CONFIG["num_layers"], n_assets,
                          LSTM_CONFIG["dropout"])
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", weights_only=True))
    model.eval()
    return model


def predict(split: str = "test") -> pd.DataFrame:
    target = LSTM_CONFIG["target"]
    df     = pd.read_csv(PROCESSED_DIR / f"{target}_{split}.csv", index_col=0, parse_dates=True)
    model  = load_model(df.shape[1])

    X, _ = make_sequences(df.values, LSTM_CONFIG["lookback"])
    X    = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        preds = model(X).numpy()

    index = df.index[LSTM_CONFIG["lookback"]:]
    return pd.DataFrame(preds, index=index, columns=df.columns)


if __name__ == "__main__":
    preds = predict("test")
    preds.to_csv(PROCESSED_DIR / "lstm_predictions_test.csv")
    print(preds.head())
