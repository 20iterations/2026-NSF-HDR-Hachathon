import math
import torch
import torch.nn as nn


class DropPath(nn.Module):
    """
    Stochastic depth (per-sample).
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class LearnedTokenizer(nn.Module):
    """
    More 'reasonable' tokenization than hard chunking:
      feats [B, F] -> tokens [B, T, D]
    via a learned linear projection.
    """

    def __init__(self, num_features: int = 768, n_tokens: int = 16, d_model: int = 256, dropout: float = 0.0):
        super().__init__()
        self.num_features = int(num_features)
        self.n_tokens = int(n_tokens)
        self.d_model = int(d_model)
        self.proj = nn.Linear(self.num_features, self.n_tokens * self.d_model)
        self.dropout = nn.Dropout(p=float(dropout))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.ndim != 2 or feats.shape[-1] != self.num_features:
            raise ValueError(f"Expected feats shape [B,{self.num_features}], got {tuple(feats.shape)}")
        x = self.proj(feats)  # [B, T*D]
        x = x.view(feats.shape[0], self.n_tokens, self.d_model)  # [B,T,D]
        return self.dropout(x)


class FeatureGate(nn.Module):
    """
    Simple gating (SE/GLU-ish) over the 768-d feature vector to stabilize and improve expressivity.
    """

    def __init__(self, num_features: int = 768, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(int(num_features))
        self.net = nn.Sequential(
            nn.Linear(int(num_features), int(hidden)),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden), int(num_features)),
            nn.Sigmoid(),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        g = self.net(self.norm(feats))
        return feats * g


class AttnBackboneV2(nn.Module):
    """
    Transformer encoder over learned tokens with CLS + positional embedding.
    """

    def __init__(
        self,
        num_features: int = 768,
        n_tokens: int = 16,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_feedforward: int = 768,
        dropout: float = 0.1,
        drop_path: float = 0.05,
        use_gate: bool = True,
        token_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.n_tokens = int(n_tokens)
        self.d_model = int(d_model)

        self.gate = FeatureGate(num_features=num_features, hidden=max(128, d_model), dropout=dropout) if use_gate else None
        self.tokenizer = LearnedTokenizer(
            num_features=num_features, n_tokens=n_tokens, d_model=d_model, dropout=token_dropout
        )

        self.cls = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.n_tokens, self.d_model))

        # TransformerEncoderLayer doesn't include stochastic depth, so we apply DropPath on the residual output.
        self.drop_path = DropPath(p=float(drop_path))
        self.enc_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=int(n_heads),
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(dropout),
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(int(n_layers))
            ]
        )
        self.norm = nn.LayerNorm(self.d_model)
        self._init_params()

    def _init_params(self):
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B,F] -> pooled [B,D]
        if self.gate is not None:
            feats = self.gate(feats)
        x = self.tokenizer(feats)  # [B,T,D]
        b = x.shape[0]
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B,1+T,D]
        x = x + self.pos[:, : x.shape[1], :].to(dtype=x.dtype, device=x.device)

        for layer in self.enc_layers:
            # layer already includes residuals internally; we add stochastic depth on its output delta.
            y = layer(x)
            x = x + self.drop_path(y - x)

        h = self.norm(x[:, 0])
        return h


class FeatureAttnPointHeadV2(nn.Module):
    def __init__(self, num_features: int = 768, num_outputs: int = 3, **backbone_kwargs):
        super().__init__()
        self.backbone = AttnBackboneV2(num_features=num_features, **backbone_kwargs)
        d = self.backbone.d_model
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(p=float(backbone_kwargs.get("dropout", 0.1))),
            nn.Linear(d, int(num_outputs)),
        )

    def forward(self, feats: torch.Tensor):
        h = self.backbone(feats)
        return self.head(h)


class FeatureAttnGaussianHeadV2(nn.Module):
    def __init__(self, num_features: int = 768, num_outputs: int = 3, min_sigma: float = 1e-3, **backbone_kwargs):
        super().__init__()
        self.backbone = AttnBackboneV2(num_features=num_features, **backbone_kwargs)
        d = self.backbone.d_model
        pdrop = float(backbone_kwargs.get("dropout", 0.1))
        self.mu_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(p=pdrop),
            nn.Linear(d, int(num_outputs)),
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(p=pdrop),
            nn.Linear(d, int(num_outputs)),
        )
        self.softplus = nn.Softplus()
        self.min_sigma = float(min_sigma)
        self._init_sigma_bias()

    def _init_sigma_bias(self):
        inv_softplus_1 = math.log(math.expm1(1.0))
        last = self.sigma_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, inv_softplus_1)

    def forward(self, feats: torch.Tensor):
        h = self.backbone(feats)
        mu = self.mu_head(h)
        sigma = self.softplus(self.sigma_head(h)) + self.min_sigma
        return mu, sigma


class BioClip2_AttnPointRegressorV2(nn.Module):
    def __init__(self, bioclip, num_features: int = 768, num_outputs: int = 3, **head_kwargs):
        super().__init__()
        self.bioclip = bioclip
        self.regressor = FeatureAttnPointHeadV2(num_features=num_features, num_outputs=num_outputs, **head_kwargs)

    def forward(self, x):
        feats = self.bioclip(x)["image_features"]
        return self.regressor(feats)


class BioClip2_AttnGaussianRegressorV2(nn.Module):
    def __init__(self, bioclip, num_features: int = 768, num_outputs: int = 3, **head_kwargs):
        super().__init__()
        self.bioclip = bioclip
        self.regressor = FeatureAttnGaussianHeadV2(num_features=num_features, num_outputs=num_outputs, **head_kwargs)

    def forward(self, x):
        feats = self.bioclip(x)["image_features"]
        return self.regressor(feats)





