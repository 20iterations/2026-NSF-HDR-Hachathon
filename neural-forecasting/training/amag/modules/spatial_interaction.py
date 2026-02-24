import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class AdaptorMLP(nn.Module):
    """Sample-dependent gating: S^(u,v) = sigma(MLP([H^(u), H^(v)])).

    Paper (Appendix B.3): 4 FC layers with input feature dimensions
    64*t, 64*2, 64*4, 64.  Output is sigmoid-gated scalar in [0, 1].

    Architecture: input_dim -> 128 -> 256 -> 64 -> 1 -> Sigmoid
    with ReLU activations between hidden layers.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (*, input_dim) -> (*, 1)"""
        return self.mlp(x)


class SpatialInteraction(nn.Module):
    """SI module: Add (additive) + Modulator (multiplicative) message passing.

    Add (with Adaptor):
        a_t^(v) = sum_u S^(u,v) * A_a^(u,v) * h_t^(u)
        where S^(u,v) = sigma(MLP([H^(u), H^(v)])) in [0,1]
    Modulator:
        m_t^(v) = sum_u A_m^(u,v) * (h_t^(u) odot h_t^(v))
    Output:
        z = beta1*h + beta2*FC(a) + beta3*FC(m)
    """

    def __init__(self, num_channels: int, hidden_dim: int = 64,
                 total_len: int = 20,
                 corr_matrix: torch.Tensor | None = None,
                 use_adaptor: bool = True,
                 adaptor_chunk_size: int = 8):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.use_adaptor = use_adaptor
        self.adaptor_chunk_size = adaptor_chunk_size

        # Learnable adjacency matrices (initialized from correlation matrix)
        if corr_matrix is not None:
            self.A_a = nn.Parameter(corr_matrix.clone())
            self.A_m = nn.Parameter(corr_matrix.clone())
        else:
            self.A_a = nn.Parameter(torch.randn(num_channels, num_channels) * 0.01)
            self.A_m = nn.Parameter(torch.randn(num_channels, num_channels) * 0.01)

        # FC layers for Add and Modulator outputs
        self.fc_add = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mod = nn.Linear(hidden_dim, hidden_dim)

        # Learnable combination weights
        self.beta1 = nn.Parameter(torch.tensor(1.0))
        self.beta2 = nn.Parameter(torch.tensor(1.0))
        self.beta3 = nn.Parameter(torch.tensor(1.0))

        # Adaptor MLP for sample-dependent gating
        if use_adaptor:
            # Input: concat(H^(u)_flat, H^(v)_flat) = 2 * T * D
            adaptor_input_dim = 2 * total_len * hidden_dim
            self.adaptor = AdaptorMLP(adaptor_input_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, T, C, D) temporal encodings from TE
        Returns:
            z: (B, T, C, D) spatially-interacted features
        """
        # --- Add module ---
        if self.use_adaptor:
            a = self._add_with_adaptor(h)
        else:
            a = torch.einsum("uv,btud->btvd", self.A_a, h)
        a = self.fc_add(a)

        # --- Modulator module (no adaptor) ---
        weighted_sum = torch.einsum("uv,btud->btvd", self.A_m, h)
        m = weighted_sum * h  # element-wise with h_t^(v)
        m = self.fc_mod(m)

        # --- Combine ---
        z = self.beta1 * h + self.beta2 * a + self.beta3 * m
        return z

    def _add_with_adaptor(self, h: torch.Tensor) -> torch.Tensor:
        """Compute Add module with sample-dependent Adaptor gating.

        Uses chunked processing over target channels + gradient checkpointing
        to keep memory bounded for large channel counts (e.g. affi=239).
        """
        B, T, C, D = h.shape
        cs = self.adaptor_chunk_size

        chunks = []
        for v_start in range(0, C, cs):
            v_end = min(v_start + cs, C)
            if self.training:
                chunk = checkpoint(
                    self._compute_add_chunk, h, v_start, v_end,
                    use_reentrant=False,
                )
            else:
                chunk = self._compute_add_chunk(h, v_start, v_end)
            chunks.append(chunk)

        return torch.cat(chunks, dim=2)  # (B, T, C, D)

    def _compute_add_chunk(self, h: torch.Tensor,
                           v_start: int, v_end: int) -> torch.Tensor:
        """Compute Add messages for a chunk of target channels [v_start, v_end).

        For each target v in chunk, computes:
            a_t^(v) = sum_u S^(u,v) * A_a^(u,v) * h_t^(u)

        where S^(u,v) = sigma(MLP(concat(H^(u)_flat, H^(v)_flat)))
        """
        B, T, C, D = h.shape
        v_size = v_end - v_start
        TD = T * D

        # Flatten temporal dimension: (B, C, T*D)
        h_flat = h.permute(0, 2, 1, 3).reshape(B, C, TD)
        h_v = h_flat[:, v_start:v_end]  # (B, v_size, TD)

        # Pairwise concatenation: [H^(u), H^(v)] for all (u, v) pairs
        # h_flat: (B, C, 1, TD)  x  h_v: (B, 1, v_size, TD)
        pair = torch.cat([
            h_flat.unsqueeze(2).expand(-1, -1, v_size, -1),
            h_v.unsqueeze(1).expand(-1, C, -1, -1),
        ], dim=-1)  # (B, C, v_size, 2*TD)

        # Adaptor MLP: S^(u,v) = sigma(MLP([H^(u), H^(v)]))
        s = self.adaptor(pair.reshape(-1, 2 * TD))  # (B*C*v_size, 1)
        s = s.view(B, C, v_size)                     # (B, C, v_size)

        # Gated adjacency: S^(u,v) * A_a^(u,v)
        w = s * self.A_a[:, v_start:v_end].unsqueeze(0)  # (B, C, v_size)

        # Message passing: a_t^(v) = sum_u w(u,v) * h_t^(u)
        a_chunk = torch.einsum("buv,btud->btvd", w, h)  # (B, T, v_size, D)
        return a_chunk
