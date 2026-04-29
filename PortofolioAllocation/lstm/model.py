"""Multi-task LSTM classifier for cross-sectional weekly beat-median prediction."""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Architecture:
      LayerNorm(input_size) → 2-layer LSTM(input→hidden) →
      Residual(project temporal mean of input → hidden space) →
      MultiHead Self-Attention(hidden, num_heads) → Dropout →
      FC(hidden→64, GELU) → n_assets independent heads: FC(64→1, Sigmoid)

    Forward returns: (batch, n_assets) — one outperformance probability per asset.
    """

    def __init__(
        self,
        input_size: int = 209,   # 19 assets × 11 features
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_heads: int = 2,
        n_assets: int = 19,
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
        self.drop  = nn.Dropout(dropout)
        self.fc1   = nn.Linear(hidden_size, 64)
        self.act   = nn.GELU()
        # Independent head per asset — each learns its own decision boundary
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(n_assets)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)  →  out: (B, n_assets)
        xn = self.input_norm(x)                                     # (B, T, F)
        h, _ = self.lstm(xn)                                        # (B, T, H)
        res = self.residual_proj(xn.mean(dim=1, keepdim=True))      # (B, 1, H)
        h = h + res                                                  # broadcast (B, T, H)
        h, _ = self.attn(h, h, h)                                   # (B, T, H)
        out = self.drop(h[:, -1])                                    # (B, H)
        out = self.act(self.fc1(out))                                # (B, 64)
        return torch.cat(
            [torch.sigmoid(head(out)) for head in self.heads], dim=1
        )                                                            # (B, n_assets)


class MultiTaskBCE(nn.Module):
    """
    Mean BCE across all n_assets output heads, with label smoothing.
    NaN positions in target are masked out and excluded from the mean.
    """

    def __init__(self, smoothing: float = 0.1, eps: float = 1e-7):
        super().__init__()
        self.s   = smoothing
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, n_assets)
        mask = ~torch.isnan(target)
        t = target.clone()
        t[~mask] = 0.0                                       # dummy fill before smoothing
        t_s = t * (1.0 - self.s) + 0.5 * self.s             # label smoothing
        bce = -(
            t_s * torch.log(pred + self.eps)
            + (1.0 - t_s) * torch.log(1.0 - pred + self.eps)
        )
        bce = bce * mask.float()
        n = mask.float().sum().clamp(min=1.0)
        return bce.sum() / n
