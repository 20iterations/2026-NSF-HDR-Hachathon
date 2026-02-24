"""
Submission model for weights trained by:
  baselines/training/BioClip2/train_attn_point_v2.py

Expected files in submission tarball:
  - model.py (this file renamed to model.py)
  - model_attn_point_v2.pth
  - requirements.txt
"""

import os
import torch
import torch.nn as nn
from open_clip import create_model_and_transforms

SIGMA_SCALE = float(os.environ.get("SIGMA_SCALE", "0.78"))


def get_bioclip(device):
    bioclip, _, preprocess = create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2", output_dict=True, require_pretrained=True
    )
    bioclip = bioclip.to(device)
    bioclip.eval()
    for p in bioclip.parameters():
        p.requires_grad = False
    return bioclip, preprocess


class DropPath(nn.Module):
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
    def __init__(self, num_features: int = 768, n_tokens: int = 16, d_model: int = 256, dropout: float = 0.0):
        super().__init__()
        self.num_features = int(num_features)
        self.n_tokens = int(n_tokens)
        self.d_model = int(d_model)
        self.proj = nn.Linear(self.num_features, self.n_tokens * self.d_model)
        self.dropout = nn.Dropout(p=float(dropout))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.proj(feats).view(feats.shape[0], self.n_tokens, self.d_model)
        return self.dropout(x)


class FeatureGate(nn.Module):
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
        return feats * self.net(self.norm(feats))


class AttnBackboneV2(nn.Module):
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
        self.tokenizer = LearnedTokenizer(num_features=num_features, n_tokens=n_tokens, d_model=d_model, dropout=token_dropout)
        self.cls = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.n_tokens, self.d_model))
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
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if self.gate is not None:
            feats = self.gate(feats)
        x = self.tokenizer(feats)
        b = x.shape[0]
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, : x.shape[1], :].to(dtype=x.dtype, device=x.device)
        for layer in self.enc_layers:
            y = layer(x)
            x = x + self.drop_path(y - x)
        return self.norm(x[:, 0])


class FeatureAttnPointHeadV2(nn.Module):
    def __init__(self, num_features: int = 768, num_outputs: int = 3):
        super().__init__()
        self.backbone = AttnBackboneV2(num_features=num_features)
        d = self.backbone.d_model
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(d, int(num_outputs)),
        )

    def forward(self, feats: torch.Tensor):
        return self.head(self.backbone(feats))


class BioClip2_AttnPointRegressorV2(nn.Module):
    def __init__(self, bioclip):
        super().__init__()
        self.bioclip = bioclip
        self.regressor = FeatureAttnPointHeadV2()

    def forward(self, x):
        feats = self.bioclip(x)["image_features"]
        return self.regressor(feats)


def _event_gaussian_aggregate(mu_i: torch.Tensor, sigma_i: torch.Tensor, eps: float = 1e-12):
    mu_event = mu_i.mean(dim=0)
    var_mu = ((mu_i - mu_event) ** 2).mean(dim=0)
    mean_sigma2 = (sigma_i ** 2).mean(dim=0)
    sigma_event = torch.sqrt(torch.clamp(mean_sigma2 + var_mu, min=eps))
    return mu_event, sigma_event


class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transforms = None

    def load(self):
        bioclip, transforms = get_bioclip(self.device)
        self.transforms = transforms
        weights_path = os.path.join(os.path.dirname(__file__), "model_attn_point_v2.pth")
        self.model = BioClip2_AttnPointRegressorV2(bioclip=bioclip).to(self.device)
        self.model.eval()
        self.model.regressor.load_state_dict(torch.load(weights_path, map_location="cpu"))

    def predict(self, datapoints):
        images = [entry["relative_img"] for entry in datapoints]
        tensor_images = torch.stack([self.transforms(img.convert("RGB")) for img in images])
        mu_chunks = []
        dset = torch.utils.data.TensorDataset(tensor_images)
        loader = torch.utils.data.DataLoader(dset, batch_size=8, shuffle=False)
        with torch.no_grad():
            for (x,) in loader:
                mu_chunks.append(self.model(x.to(self.device)).detach().cpu())
        mu_i = torch.cat(mu_chunks, dim=0)  # [N,3]
        sigma_i = float(SIGMA_SCALE) * torch.ones_like(mu_i)     # tuned fixed per-image sigma for point model
        mu_event, sigma_event = _event_gaussian_aggregate(mu_i, sigma_i)
        return {
            "SPEI_30d": {"mu": mu_event[0].item(), "sigma": sigma_event[0].item()},
            "SPEI_1y": {"mu": mu_event[1].item(), "sigma": sigma_event[1].item()},
            "SPEI_2y": {"mu": mu_event[2].item(), "sigma": sigma_event[2].item()},
        }




