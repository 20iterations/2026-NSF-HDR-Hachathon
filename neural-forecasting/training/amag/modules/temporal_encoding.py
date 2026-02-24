import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    """Multi-head self-attention + FFN with pre-norm architecture.

    Pre-norm (LayerNorm before attention) is more stable for deeper stacks.
    """

    def __init__(self, d_model: int, d_ff: int = 256, num_heads: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, f"d_model={d_model} not divisible by num_heads={num_heads}"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.head_dim)

        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        B, T, D = x.shape

        # Pre-norm + Multi-head self-attention
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.out_proj(attn_out)

        # Residual
        x = x + self.resid_drop(attn_out)

        # Pre-norm + FFN + Residual
        x = x + self.resid_drop(self.ffn(self.norm2(x)))

        return x


class TemporalEncoder(nn.Module):
    """TE module: FC embedding + positional encoding + stacked Transformer blocks.

    Shared across all channels.
    Supports multi-head attention and multiple layers.
    Optionally uses separate LMP and spectral feature pathways.
    """

    def __init__(self, input_dim: int = 9, hidden_dim: int = 64, d_ff: int = 256,
                 num_heads: int = 1, num_layers: int = 1,
                 dropout: float = 0.0, use_feature_pathways: bool = False):
        super().__init__()
        self.use_feature_pathways = use_feature_pathways

        if use_feature_pathways:
            half = hidden_dim // 2
            self.embed_lmp = nn.Linear(1, half)
            self.embed_spec = nn.Linear(input_dim - 1, hidden_dim - half)
            self.embed_fuse = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.embed = nn.Linear(input_dim, hidden_dim)

        self.pe = PositionalEncoding(hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, d_ff, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, F) input features
        Returns:
            H: (B, T, C, D) temporal encodings
        """
        B, T, C, F = x.shape

        # Reshape to process all channels together: (B*C, T, F)
        x = x.permute(0, 2, 1, 3).reshape(B * C, T, F)

        # Embed
        if self.use_feature_pathways:
            lmp = x[:, :, :1]        # (B*C, T, 1)
            spec = x[:, :, 1:]       # (B*C, T, F-1)
            h = torch.cat([self.embed_lmp(lmp), self.embed_spec(spec)], dim=-1)
            h = self.embed_fuse(h)
        else:
            h = self.embed(x)

        h = self.pe(h)
        for layer in self.layers:
            h = layer(h)

        # Reshape back: (B, C, T, D) -> (B, T, C, D)
        h = h.reshape(B, C, T, -1).permute(0, 2, 1, 3)
        return h
