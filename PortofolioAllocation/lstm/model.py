"""LSTM model definition for log return / volatility prediction."""

import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last timestep
