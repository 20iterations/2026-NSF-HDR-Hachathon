import torch
import torch.nn as nn
from .modules.temporal_encoding import TemporalEncoder
from .modules.spatial_interaction import SpatialInteraction
from .modules.temporal_readout import TemporalReadout
from .modules.revin import RevIN
from .modules.dish_ts import DishTS
from .modules.channel_attention import ChannelAttention


class AMAG(nn.Module):
    """AMAG: Additive, Multiplicative and Adaptive Graph Neural Network.

    Architecture: TE (Transformer) -> SI (Add + Modulator + Adaptor) -> CA -> TR (Transformer) -> FC
    Optional: RevIN (Kim et al., ICLR 2022) for distribution shift handling.
    Optional: Dish-TS (Fan et al., AAAI 2023) — learned distribution coefficients (replaces RevIN).
    Optional: Channel attention (iTransformer-style) for sample-dependent spatial mixing.
    Optional: Session embeddings for cross-day adaptation.
    """

    def __init__(self, num_channels: int, num_features: int = 9,
                 hidden_dim: int = 64, d_ff: int = 256,
                 num_heads: int = 1, num_layers: int = 1,
                 total_len: int = 20,
                 corr_matrix: torch.Tensor | None = None,
                 dropout: float = 0.0,
                 use_adaptor: bool = True,
                 adaptor_chunk_size: int = 8,
                 use_revin: bool = False,
                 use_dish_ts: bool = False,
                 use_channel_attn: bool = False,
                 use_feature_pathways: bool = False,
                 use_session_embed: bool = False,
                 num_sessions: int = 3):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.use_revin = use_revin
        self.use_dish_ts = use_dish_ts
        self.use_channel_attn = use_channel_attn
        self.use_session_embed = use_session_embed

        if use_dish_ts:
            self.dish_ts = DishTS(num_channels)
        elif use_revin:
            self.revin = RevIN(num_channels)

        if use_session_embed:
            self.session_embed = nn.Embedding(num_sessions, hidden_dim)

        self.te = TemporalEncoder(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            d_ff=d_ff,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_feature_pathways=use_feature_pathways,
        )

        self.si = SpatialInteraction(
            num_channels=num_channels,
            hidden_dim=hidden_dim,
            total_len=total_len,
            corr_matrix=corr_matrix,
            use_adaptor=use_adaptor,
            adaptor_chunk_size=adaptor_chunk_size,
        )

        if use_channel_attn:
            self.channel_attn = ChannelAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        self.tr = TemporalReadout(
            hidden_dim=hidden_dim,
            d_ff=d_ff,
            output_dim=1,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    def _apply_input_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply input normalization (Dish-TS or RevIN) to LMP channel."""
        if self.use_dish_ts:
            lmp = x[:, :, :, 0]  # (B, T, C)
            lmp_norm = self.dish_ts.normalize(lmp)
            x = x.clone()
            x[:, :, :, 0] = lmp_norm
        elif self.use_revin:
            lmp = x[:, :, :, 0]
            lmp_norm = self.revin.normalize(lmp)
            x = x.clone()
            x[:, :, :, 0] = lmp_norm
        return x

    def _apply_output_denorm(self, pred: torch.Tensor) -> torch.Tensor:
        """Apply output denormalization (Dish-TS or RevIN)."""
        if self.use_dish_ts:
            pred = self.dish_ts.denormalize(pred)
        elif self.use_revin:
            pred = self.revin.denormalize(pred)
        return pred

    def forward(self, x: torch.Tensor, session_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, F) input with future steps masked
            session_ids: (B,) optional session IDs for session embedding
        Returns:
            pred: (B, T, C) LMP predictions for all timesteps
        """
        x = self._apply_input_norm(x)

        h = self.te(x)    # (B, T, C, D)

        if self.use_session_embed and session_ids is not None:
            se = self.session_embed(session_ids)  # (B, D)
            h = h + se[:, None, None, :]  # broadcast to (B, T, C, D)

        z = self.si(h)    # (B, T, C, D)

        if self.use_channel_attn:
            z = self.channel_attn(z)  # (B, T, C, D)

        pred = self.tr(z)  # (B, T, C)

        pred = self._apply_output_denorm(pred)

        return pred

    def get_te_hidden(self, x: torch.Tensor, session_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return TE hidden states for domain adaptation loss computation."""
        x = self._apply_input_norm(x)
        h = self.te(x)
        if self.use_session_embed and session_ids is not None:
            se = self.session_embed(session_ids)
            h = h + se[:, None, None, :]
        return h
