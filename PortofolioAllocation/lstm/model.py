"""LSTM classifier for cross-sectional weekly beat-median prediction."""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Architecture:
      LayerNorm → 2-layer LSTM → Residual(proj input mean) →
      MultiHead Self-Attention → Dropout → FC(128→64, GELU) → FC(64→1, Sigmoid)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_heads: int = 2,
    ):
        super().__init__()
        self.input_norm    = nn.LayerNorm(input_size)
        self.lstm          = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Residual: project temporal mean of input → hidden space
        self.residual_proj = nn.Linear(input_size, hidden_size)
        self.attn          = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden_size, 64)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        xn = self.input_norm(x)                                   # (B, T, F)
        h, _ = self.lstm(xn)                                      # (B, T, H)
        res = self.residual_proj(xn.mean(dim=1, keepdim=True))    # (B, 1, H)
        h = h + res                                                # broadcast → (B, T, H)
        h, _ = self.attn(h, h, h)                                 # (B, T, H)
        out = self.drop(h[:, -1])                                  # (B, H)
        out = self.act(self.fc1(out))                              # (B, 64)
        return torch.sigmoid(self.fc2(out)).squeeze(-1)            # (B,)


class SmoothedBCE(nn.Module):
    """BCE with label smoothing (0→eps/2, 1→1-eps/2) and per-class weighting."""

    def __init__(self, smoothing: float = 0.1, eps: float = 1e-7):
        super().__init__()
        self.s   = smoothing
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pos_weight: float = 1.0,
    ) -> torch.Tensor:
        t = target * (1.0 - self.s) + 0.5 * self.s               # smooth labels
        bce = -(
            t * torch.log(pred + self.eps)
            + (1.0 - t) * torch.log(1.0 - pred + self.eps)
        )
        w = torch.where(
            target > 0.5,
            pred.new_full(pred.shape, pos_weight),
            pred.new_ones(pred.shape),
        )
        return (bce * w).mean()
