"""Dish-TS: Distribution Shift handler with dual CONETs.

Reference: Fan et al., "Dish-TS: A General Paradigm for Alleviating
Distribution Shift in Time Series Forecasting", AAAI 2023.

Replaces RevIN: instead of raw instance statistics (noisy with short context),
Dish-TS learns distribution coefficients via small networks (CONETs) that
smooth out noisy statistics. Dual CONETs handle intra-space (input) and
inter-space (input->output) distribution shifts separately.
"""

import torch
import torch.nn as nn


class CONET(nn.Module):
    """Coefficient Network: maps instance stats to distribution coefficients."""

    def __init__(self, num_channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * num_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_channels),
        )

    def forward(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Compute distribution coefficients from instance statistics.

        Args:
            mean: (B, C) per-channel means
            std: (B, C) per-channel stds
        Returns:
            coeff: (B, C) learned distribution coefficients
        """
        stats = torch.cat([mean, std], dim=-1)  # (B, 2C)
        return self.net(stats)  # (B, C)


class DishTS(nn.Module):
    """Dish-TS: Distribution Shift handler with dual CONETs.

    Uses separate input and output CONETs to learn distribution coefficients
    for normalization and denormalization respectively.
    """

    def __init__(self, num_channels: int, context_len: int = 10):
        super().__init__()
        self.context_len = context_len
        self.input_conet = CONET(num_channels)
        self.output_conet = CONET(num_channels)
        self._mean = None
        self._std = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize LMP signal using learned input coefficients.

        Args:
            x: (B, T, C) LMP values
        Returns:
            Normalized (B, T, C)
        """
        ctx = x[:, :self.context_len]  # (B, ctx, C)
        self._mean = ctx.mean(dim=1)   # (B, C)
        self._std = (ctx.var(dim=1, unbiased=False) + 1e-5).sqrt()  # (B, C)
        input_coeff = self.input_conet(self._mean, self._std)  # (B, C)
        x_norm = (x - input_coeff.unsqueeze(1)) / (self._std.unsqueeze(1) + 1e-5)
        return x_norm

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize predictions using learned output coefficients.

        Args:
            x: (B, T, C) normalized predictions
        Returns:
            Denormalized (B, T, C)
        """
        output_coeff = self.output_conet(self._mean, self._std)  # (B, C)
        return x * (self._std.unsqueeze(1) + 1e-5) + output_coeff.unsqueeze(1)
