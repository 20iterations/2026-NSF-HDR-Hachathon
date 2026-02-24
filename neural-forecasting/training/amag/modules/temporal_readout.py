import torch
import torch.nn as nn
from .temporal_encoding import PositionalEncoding, TransformerBlock


class TemporalReadout(nn.Module):
    """TR module: Transformer-based readout (same architecture as TE, separate params).

    Takes spatially-interacted features z and produces final predictions.
    Supports multi-head attention and multiple layers.
    """

    def __init__(self, hidden_dim: int = 64, d_ff: int = 256, output_dim: int = 1,
                 num_heads: int = 1, num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.pe = PositionalEncoding(hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, d_ff, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T, C, D) spatially-interacted features
        Returns:
            pred: (B, T, C) LMP predictions
        """
        B, T, C, D = z.shape

        # Reshape to process all channels together: (B*C, T, D)
        z = z.permute(0, 2, 1, 3).reshape(B * C, T, D)

        # PE + Transformer layers
        r = self.pe(z)
        for layer in self.layers:
            r = layer(r)

        # Output projection to LMP
        pred = self.output_fc(r)  # (B*C, T, 1)
        pred = pred.squeeze(-1)   # (B*C, T)

        # Reshape back: (B, C, T) -> (B, T, C)
        pred = pred.reshape(B, C, T).permute(0, 2, 1)
        return pred
