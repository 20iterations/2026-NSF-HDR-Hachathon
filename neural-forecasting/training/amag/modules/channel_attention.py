"""iTransformer-style cross-channel attention.

Reference: Liu et al., "iTransformer: Inverted Transformers Are Effective for
Time Series Forecasting", ICLR 2024 Spotlight.

Treats channels as tokens and applies multi-head self-attention per timestep.
This provides sample-dependent spatial interaction without expensive Adaptor MLP.
"""

import math
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Cross-channel attention: channels are tokens, features are embedding dim.

    For each timestep, applies multi-head self-attention across channels.
    This complements SI's fixed adjacency with sample-dependent spatial mixing.
    """

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(self.head_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T, C, D) spatially-interacted features
        Returns:
            z_out: (B, T, C, D) with cross-channel attention applied
        """
        B, T, C, D = z.shape

        # Reshape: (B*T, C, D) — each timestep: channels are tokens
        z_flat = z.reshape(B * T, C, D)

        # Pre-norm
        z_norm = self.norm(z_flat)

        # Multi-head attention across channels
        q = self.q_proj(z_norm).view(B * T, C, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(z_norm).view(B * T, C, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(z_norm).view(B * T, C, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B * T, C, D)
        out = self.out_proj(out)

        # Residual connection
        z_out = z_flat + self.resid_drop(out)

        return z_out.reshape(B, T, C, D)
