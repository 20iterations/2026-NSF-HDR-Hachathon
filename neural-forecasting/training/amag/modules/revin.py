"""Reversible Instance Normalization (RevIN).

Reference: Kim et al., "Reversible Instance Normalization for Accurate
Time-Series Forecasting against Distribution Shift", ICLR 2022.

Modified: Uses context-window-only statistics to avoid future information leakage
and better handle cross-day distribution shifts.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """RevIN normalizes each input sample by its own instance statistics,
    then denormalizes predictions with the same stats. Learnable affine
    parameters adapt to the training distribution.

    Uses context window only for statistics computation.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, context_len: int = 10):
        super().__init__()
        self.eps = eps
        self.context_len = context_len
        self.affine_weight = nn.Parameter(torch.ones(1, 1, num_channels))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_channels))
        # Instance stats stored during normalize, used by denormalize
        self._mean = None
        self._std = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize LMP signal per-instance using context-window stats only.

        Args:
            x: (B, T, C) LMP values
        Returns:
            Normalized (B, T, C)
        """
        # Compute stats from context window only (prevents future leakage)
        context = x[:, :self.context_len]  # (B, context_len, C)
        self._mean = context.mean(dim=1, keepdim=True)  # (B, 1, C)
        self._std = (context.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()
        x_norm = (x - self._mean) / self._std
        return x_norm * self.affine_weight + self.affine_bias

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the normalization applied during normalize().

        Args:
            x: (B, T, C) normalized predictions
        Returns:
            Denormalized (B, T, C)
        """
        x = (x - self.affine_bias) / self.affine_weight
        return x * self._std + self._mean
