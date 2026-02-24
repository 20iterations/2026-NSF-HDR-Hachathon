"""Codabench-compatible submission model.

Self-contained AMAG model definition + inference with training normalization stats.
v3.5: Mtime-filtered snapshots, weighted ensemble (2x for best), no TTA.

Expected files alongside this script:
  - amag_{monkey}_snap1.pth .. amag_{monkey}_snap3.pth (quality-gated, mtime-filtered)
  - amag_{monkey}_best.pth, amag_{monkey}_ema_best.pth (always included in ensemble)
  - norm_stats_{monkey}.npz (training normalization stats)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---- Positional Encoding ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ---- RevIN (Kim et al., ICLR 2022) — context-only stats ----
class RevIN(nn.Module):
    def __init__(self, num_channels, eps=1e-5, context_len=10):
        super().__init__()
        self.eps = eps
        self.context_len = context_len
        self.affine_weight = nn.Parameter(torch.ones(1, 1, num_channels))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_channels))
        self._mean = None
        self._std = None

    def normalize(self, x):
        context = x[:, :self.context_len]
        self._mean = context.mean(dim=1, keepdim=True)
        self._std = (context.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()
        x_norm = (x - self._mean) / self._std
        return x_norm * self.affine_weight + self.affine_bias

    def denormalize(self, x):
        x = (x - self.affine_bias) / self.affine_weight
        return x * self._std + self._mean


# ---- Dish-TS (Fan et al., AAAI 2023) — learned distribution coefficients ----
class CONET(nn.Module):
    def __init__(self, num_channels, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * num_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_channels),
        )

    def forward(self, mean, std):
        stats = torch.cat([mean, std], dim=-1)
        return self.net(stats)


class DishTS(nn.Module):
    def __init__(self, num_channels, context_len=10):
        super().__init__()
        self.context_len = context_len
        self.input_conet = CONET(num_channels)
        self.output_conet = CONET(num_channels)
        self._mean = None
        self._std = None

    def normalize(self, x):
        ctx = x[:, :self.context_len]
        self._mean = ctx.mean(dim=1)
        self._std = (ctx.var(dim=1, unbiased=False) + 1e-5).sqrt()
        input_coeff = self.input_conet(self._mean, self._std)
        return (x - input_coeff.unsqueeze(1)) / (self._std.unsqueeze(1) + 1e-5)

    def denormalize(self, x):
        output_coeff = self.output_conet(self._mean, self._std)
        return x * (self._std.unsqueeze(1) + 1e-5) + output_coeff.unsqueeze(1)


# ---- Transformer Block (multi-head, pre-norm) ----
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff=256, num_heads=1, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.head_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model))

    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) / self.scale, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        x = x + self.resid_drop(out)
        x = x + self.resid_drop(self.ffn(self.norm2(x)))
        return x


# ---- TE (multi-layer, feature pathways) ----
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, d_ff=256,
                 num_heads=1, num_layers=1, dropout=0.0,
                 use_feature_pathways=False):
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
            TransformerBlock(hidden_dim, d_ff, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        B, T, C, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * C, T, F)
        if self.use_feature_pathways:
            lmp = x[:, :, :1]
            spec = x[:, :, 1:]
            h = torch.cat([self.embed_lmp(lmp), self.embed_spec(spec)], dim=-1)
            h = self.embed_fuse(h)
        else:
            h = self.embed(x)
        h = self.pe(h)
        for layer in self.layers:
            h = layer(h)
        return h.reshape(B, C, T, -1).permute(0, 2, 1, 3)


# ---- Adaptor MLP ----
class AdaptorMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        return self.mlp(x)


# ---- SI ----
class SpatialInteraction(nn.Module):
    def __init__(self, num_channels, hidden_dim=64, total_len=20,
                 use_adaptor=False, adaptor_chunk_size=8):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.use_adaptor = use_adaptor
        self.adaptor_chunk_size = adaptor_chunk_size
        self.A_a = nn.Parameter(torch.zeros(num_channels, num_channels))
        self.A_m = nn.Parameter(torch.zeros(num_channels, num_channels))
        self.fc_add = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mod = nn.Linear(hidden_dim, hidden_dim)
        self.beta1 = nn.Parameter(torch.tensor(1.0))
        self.beta2 = nn.Parameter(torch.tensor(1.0))
        self.beta3 = nn.Parameter(torch.tensor(1.0))
        if use_adaptor:
            adaptor_input_dim = 2 * total_len * hidden_dim
            self.adaptor = AdaptorMLP(adaptor_input_dim)

    def forward(self, h):
        if self.use_adaptor:
            a = self._add_with_adaptor(h)
        else:
            a = torch.einsum("uv,btud->btvd", self.A_a, h)
        a = self.fc_add(a)
        ws = torch.einsum("uv,btud->btvd", self.A_m, h)
        m = self.fc_mod(ws * h)
        return self.beta1 * h + self.beta2 * a + self.beta3 * m

    def _add_with_adaptor(self, h):
        B, T, C, D = h.shape
        cs = self.adaptor_chunk_size
        chunks = []
        for v_start in range(0, C, cs):
            v_end = min(v_start + cs, C)
            chunks.append(self._compute_add_chunk(h, v_start, v_end))
        return torch.cat(chunks, dim=2)

    def _compute_add_chunk(self, h, v_start, v_end):
        B, T, C, D = h.shape
        v_size = v_end - v_start
        TD = T * D
        h_flat = h.permute(0, 2, 1, 3).reshape(B, C, TD)
        h_v = h_flat[:, v_start:v_end]
        pair = torch.cat([
            h_flat.unsqueeze(2).expand(-1, -1, v_size, -1),
            h_v.unsqueeze(1).expand(-1, C, -1, -1),
        ], dim=-1)
        s = self.adaptor(pair.reshape(-1, 2 * TD)).view(B, C, v_size)
        w = s * self.A_a[:, v_start:v_end].unsqueeze(0)
        return torch.einsum("buv,btud->btvd", w, h)


# ---- Channel Attention (iTransformer-style) ----
class ChannelAttention(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(self.head_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, z):
        B, T, C, D = z.shape
        z_flat = z.reshape(B * T, C, D)
        z_norm = self.norm(z_flat)
        q = self.q_proj(z_norm).view(B * T, C, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(z_norm).view(B * T, C, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(z_norm).view(B * T, C, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) / self.scale, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B * T, C, D)
        out = self.out_proj(out)
        return (z_flat + self.resid_drop(out)).reshape(B, T, C, D)


# ---- TR (multi-layer) ----
class TemporalReadout(nn.Module):
    def __init__(self, hidden_dim=64, d_ff=256, num_heads=1, num_layers=1, dropout=0.0):
        super().__init__()
        self.pe = PositionalEncoding(hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, d_ff, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.output_fc = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        B, T, C, D = z.shape
        z = z.permute(0, 2, 1, 3).reshape(B * C, T, D)
        r = self.pe(z)
        for layer in self.layers:
            r = layer(r)
        pred = self.output_fc(r).squeeze(-1)
        return pred.reshape(B, C, T).permute(0, 2, 1)


# ---- Full AMAG ----
class AMAG(nn.Module):
    def __init__(self, num_channels, num_features=9, hidden_dim=128, d_ff=512,
                 num_heads=4, num_layers=2, total_len=20,
                 use_adaptor=False, adaptor_chunk_size=8,
                 use_revin=False, use_dish_ts=False,
                 use_channel_attn=True,
                 use_feature_pathways=True,
                 use_session_embed=False, num_sessions=3,
                 dropout=0.0):
        super().__init__()
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
        self.te = TemporalEncoder(num_features, hidden_dim, d_ff,
                                   num_heads, num_layers, dropout,
                                   use_feature_pathways)
        self.si = SpatialInteraction(num_channels, hidden_dim, total_len,
                                     use_adaptor, adaptor_chunk_size)
        if use_channel_attn:
            self.channel_attn = ChannelAttention(hidden_dim, num_heads, dropout)
        self.tr = TemporalReadout(hidden_dim, d_ff, num_heads, num_layers, dropout)

    def forward(self, x, session_ids=None):
        if self.use_dish_ts:
            lmp = x[:, :, :, 0]
            lmp_norm = self.dish_ts.normalize(lmp)
            x = x.clone()
            x[:, :, :, 0] = lmp_norm
        elif self.use_revin:
            lmp = x[:, :, :, 0]
            lmp_norm = self.revin.normalize(lmp)
            x = x.clone()
            x[:, :, :, 0] = lmp_norm
        h = self.te(x)
        if self.use_session_embed and session_ids is not None:
            se = self.session_embed(session_ids)
            h = h + se[:, None, None, :]
        z = self.si(h)
        if self.use_channel_attn:
            z = self.channel_attn(z)
        pred = self.tr(z)
        if self.use_dish_ts:
            pred = self.dish_ts.denormalize(pred)
        elif self.use_revin:
            pred = self.revin.denormalize(pred)
        return pred


# ---- Normalization (matches training pipeline exactly) ----
def normalize_data(data, mean, std):
    """Normalize to [-1, 1] using [mean-4*std, mean+4*std] range."""
    n, t, c, f = data.shape
    flat = data.reshape(n * t, c * f)
    lo = mean - 4 * std
    hi = mean + 4 * std
    denom = hi - lo
    denom = np.where(denom == 0, 1.0, denom)
    norm = 2 * (flat - lo) / denom - 1
    return norm.reshape(n, t, c, f).astype(np.float32)


def denormalize_lmp(data, mean, std, num_features=9):
    """Denormalize LMP from [-1, 1] back to raw scale."""
    mean_lmp = mean[0, ::num_features]
    std_lmp = std[0, ::num_features]
    lo = mean_lmp - 4 * std_lmp
    hi = mean_lmp + 4 * std_lmp
    denom = hi - lo
    denom = np.where(denom == 0, 1.0, denom)
    return (data + 1) * denom / 2 + lo


# ---- Submission Model ----
class Model:
    def __init__(self, monkey_name="beignet"):
        self.monkey_name = monkey_name
        if monkey_name == "affi":
            self.num_channels = 239
        elif monkey_name == "beignet":
            self.num_channels = 89
        else:
            raise ValueError(f"Unknown monkey: {monkey_name}")

        self.models = []
        self._all_paths = []  # Paths of loaded checkpoints (for weighted ensemble)
        self.model_config = {}
        self.norm_stats = None  # Per-session normalization stats from training

    @staticmethod
    def _clean_state_dict(state_dict):
        """Strip '_orig_mod.' prefix added by torch.compile."""
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.removeprefix("_orig_mod.")] = v
        return cleaned

    def _detect_config(self, state_dict):
        """Detect model configuration from checkpoint state dict."""
        # Strip torch.compile prefix if present
        state_dict = self._clean_state_dict(state_dict)
        config = {
            "use_revin": any(k.startswith("revin.") for k in state_dict),
            "use_dish_ts": any(k.startswith("dish_ts.") for k in state_dict),
            "use_channel_attn": any(k.startswith("channel_attn.") for k in state_dict),
            "use_session_embed": any(k.startswith("session_embed.") for k in state_dict),
            "use_feature_pathways": "te.embed_lmp.weight" in state_dict,
        }

        # Detect hidden_dim from TE embed output
        if config["use_feature_pathways"]:
            config["hidden_dim"] = state_dict["te.embed_fuse.weight"].shape[0]
        else:
            config["hidden_dim"] = state_dict["te.embed.weight"].shape[0]

        # Detect num_layers from transformer layer keys
        te_layer_ids = set()
        for k in state_dict:
            if k.startswith("te.layers."):
                layer_id = int(k.split(".")[2])
                te_layer_ids.add(layer_id)
        config["num_layers"] = max(te_layer_ids) + 1 if te_layer_ids else 1

        # Detect d_ff and num_heads
        if "te.layers.0.out_proj.weight" in state_dict:
            config["d_ff"] = state_dict["te.layers.0.ffn.0.weight"].shape[0]
            if config["use_channel_attn"] and "channel_attn.q_proj.weight" in state_dict:
                config["num_heads"] = 4
            else:
                config["num_heads"] = 1
        elif "te.transformer.out_proj.weight" in state_dict:
            config["num_heads"] = 4
            config["d_ff"] = state_dict["te.transformer.ffn.0.weight"].shape[0]
            config["num_layers"] = 1
        else:
            config["num_heads"] = 1
            config["d_ff"] = 256

        # Detect num_sessions
        if config["use_session_embed"]:
            config["num_sessions"] = state_dict["session_embed.weight"].shape[0]
        else:
            config["num_sessions"] = 3

        return config

    def _make_model(self, config):
        """Create a fresh AMAG model instance matching trained config."""
        chunk = 4 if self.num_channels > 200 else 8
        return AMAG(
            num_channels=self.num_channels,
            num_features=9,
            hidden_dim=config.get("hidden_dim", 64),
            d_ff=config.get("d_ff", 256),
            num_heads=config.get("num_heads", 1),
            num_layers=config.get("num_layers", 1),
            total_len=20,
            use_adaptor=False,
            adaptor_chunk_size=chunk,
            use_revin=config.get("use_revin", False),
            use_dish_ts=config.get("use_dish_ts", False),
            use_channel_attn=config.get("use_channel_attn", False),
            use_feature_pathways=config.get("use_feature_pathways", False),
            use_session_embed=config.get("use_session_embed", False),
            num_sessions=config.get("num_sessions", 3),
            dropout=0.0,  # No dropout at inference
        )

    def _load_norm_stats(self):
        """Load per-session normalization stats saved during training."""
        base = os.path.dirname(__file__)
        stats_path = os.path.join(base, f"norm_stats_{self.monkey_name}.npz")
        if os.path.exists(stats_path):
            data = np.load(stats_path)
            num_sessions = int(data["num_sessions"])
            self.norm_stats = []
            for i in range(num_sessions):
                mean = data[f"mean_{i}"]
                std = data[f"std_{i}"]
                self.norm_stats.append((mean, std))
            print(f"Loaded training norm stats ({num_sessions} sessions)")
        else:
            print(f"WARNING: norm_stats_{self.monkey_name}.npz not found, using TTA fallback")
            self.norm_stats = None

    def _collect_seed_checkpoints(self, base):
        """Collect checkpoints from seed_* subdirectories (multi-seed ensemble).

        Returns list of (path, is_best_or_ema) tuples.
        """
        all_paths = []

        # Look for seed_* directories
        seed_dirs = sorted([
            d for d in os.listdir(base)
            if d.startswith("seed_") and os.path.isdir(os.path.join(base, d))
        ])

        if not seed_dirs:
            return all_paths

        for seed_dir in seed_dirs:
            seed_path = os.path.join(base, seed_dir)
            best_path = os.path.join(seed_path, f"amag_{self.monkey_name}_best.pth")
            ema_best_path = os.path.join(seed_path, f"amag_{self.monkey_name}_ema_best.pth")

            # Collect best/ema_best
            ref_path = None
            if os.path.exists(best_path):
                all_paths.append(best_path)
                ref_path = best_path
            if os.path.exists(ema_best_path):
                all_paths.append(ema_best_path)
                if ref_path is None:
                    ref_path = ema_best_path

            # Collect snapshots (mtime-filtered, capped at 4 per seed)
            snapshot_paths = [
                os.path.join(seed_path, f"amag_{self.monkey_name}_snap{i}.pth")
                for i in range(1, 10)
            ]
            snapshots_found = [p for p in snapshot_paths if os.path.exists(p)]
            if ref_path and snapshots_found:
                ref_mtime = os.path.getmtime(ref_path)
                snapshots_found = [
                    p for p in snapshots_found
                    if abs(os.path.getmtime(p) - ref_mtime) < 7200
                ]
            snapshots_found = snapshots_found[:4]
            all_paths.extend(snapshots_found)

            # Also load norm stats from first seed dir found
            norm_path = os.path.join(seed_path, f"norm_stats_{self.monkey_name}.npz")
            if os.path.exists(norm_path) and self.norm_stats is None:
                data = np.load(norm_path)
                num_sessions = int(data["num_sessions"])
                self.norm_stats = []
                for i in range(num_sessions):
                    mean = data[f"mean_{i}"]
                    std = data[f"std_{i}"]
                    self.norm_stats.append((mean, std))
                print(f"Loaded training norm stats from {seed_dir} ({num_sessions} sessions)")

        return all_paths

    def load(self):
        base = os.path.dirname(__file__)

        # Load normalization stats from training (top-level first)
        self._load_norm_stats()

        # Collect checkpoints from multi-seed directories first
        seed_paths = self._collect_seed_checkpoints(base)

        # Also collect top-level checkpoints (single-seed or legacy)
        best_path = os.path.join(base, f"amag_{self.monkey_name}_best.pth")
        ema_best_path = os.path.join(base, f"amag_{self.monkey_name}_ema_best.pth")

        # Detect config from best model or first available checkpoint
        config_sd = None
        if os.path.exists(best_path):
            config_sd = torch.load(best_path, map_location=device, weights_only=True)
        elif os.path.exists(ema_best_path):
            config_sd = torch.load(ema_best_path, map_location=device, weights_only=True)

        # Top-level snapshots
        snapshot_paths = [
            os.path.join(base, f"amag_{self.monkey_name}_snap{i}.pth")
            for i in range(1, 10)
        ]
        snapshots_found = [p for p in snapshot_paths if os.path.exists(p)]

        ref_path = best_path if os.path.exists(best_path) else (
            ema_best_path if os.path.exists(ema_best_path) else None)
        if ref_path and snapshots_found:
            ref_mtime = os.path.getmtime(ref_path)
            snapshots_found = [
                p for p in snapshots_found
                if abs(os.path.getmtime(p) - ref_mtime) < 7200
            ]
        snapshots_found = snapshots_found[:4]

        # Combine: top-level + seed paths (no duplicates)
        all_paths = []
        if os.path.exists(best_path):
            all_paths.append(best_path)
        if os.path.exists(ema_best_path):
            all_paths.append(ema_best_path)
        all_paths.extend(snapshots_found)
        # Add seed-based checkpoints
        seen = set(os.path.abspath(p) for p in all_paths)
        for p in seed_paths:
            if os.path.abspath(p) not in seen:
                all_paths.append(p)
                seen.add(os.path.abspath(p))

        if not all_paths:
            raise FileNotFoundError(f"No checkpoints found for {self.monkey_name}")

        self._all_paths = all_paths

        # Detect config from first available checkpoint
        if config_sd is None:
            config_sd = torch.load(all_paths[0], map_location=device, weights_only=True)
        self.model_config = self._detect_config(config_sd)

        for path in all_paths:
            m = self._make_model(self.model_config)
            state_dict = torch.load(path, map_location=device, weights_only=True)
            state_dict = self._clean_state_dict(state_dict)
            m.load_state_dict(state_dict, strict=False)
            m.to(device)
            m.eval()
            self.models.append(m)

        print(f"Loaded {len(self.models)} models (multi-seed ensemble, config={self.model_config})")

    def _find_best_session_stats(self, x):
        """Find the best matching session stats for this data.

        Uses combined mean + std correlation for robust session matching.
        If norm_stats not available, falls back to context-window TTA.
        """
        if self.norm_stats is None:
            # TTA fallback: compute from context window
            context = x[:, :10]
            n, t, c, f = context.shape
            flat = context.reshape(n * t, c * f)
            mean = flat.mean(axis=0, keepdims=True)
            std = flat.std(axis=0, keepdims=True)
            return mean, std

        if len(self.norm_stats) == 1:
            return self.norm_stats[0]

        # Compute test data stats
        n, t, c, f_dim = x.shape
        flat = x.reshape(n * t, c * f_dim)
        data_mean = flat.mean(axis=0)
        data_std = flat.std(axis=0)

        # Combined matching: mean correlation + std correlation
        best_score = -2
        best_idx = 0
        for i, (s_mean, s_std) in enumerate(self.norm_stats):
            mean_corr = np.corrcoef(data_mean, s_mean.flatten())[0, 1]
            std_corr = np.corrcoef(data_std, s_std.flatten())[0, 1]
            score = 0.5 * mean_corr + 0.5 * std_corr
            if score > best_score:
                best_score = score
                best_idx = i

        print(f"  Matched to session {best_idx} (score={best_score:.4f})")
        return self.norm_stats[best_idx]

    def predict(self, x):
        """
        Args:
            x: (N, 20, C, 9) raw test data (steps 10-19 are repeated from step 9)
        Returns:
            predictions: (N, 20, C) raw-scale LMP predictions
        """
        context_len = 10
        n, t, c, f = x.shape

        if self.norm_stats is None or len(self.norm_stats) == 1:
            # Single session or TTA fallback — use original approach
            mean, std = self._find_best_session_stats(x)
            return self._predict_with_stats(x, mean, std)

        # Compute context-only test stats for matching (avoid masked future bias)
        context = x[:, :context_len]
        ctx_flat = context.reshape(n * context_len, c * f)
        data_mean = ctx_flat.mean(axis=0)
        data_std = ctx_flat.std(axis=0)

        # Score each session
        scores = []
        for i, (s_mean, s_std) in enumerate(self.norm_stats):
            mean_corr = np.corrcoef(data_mean, s_mean.flatten())[0, 1]
            std_corr = np.corrcoef(data_std, s_std.flatten())[0, 1]
            score = 0.5 * mean_corr + 0.5 * std_corr
            scores.append(score)

        # Softmax with temperature=0.02 (very sharp — near-argmax but smooth)
        scores = np.array(scores)
        exp_scores = np.exp((scores - scores.max()) / 0.02)
        weights = exp_scores / exp_scores.sum()
        print(f"  Session scores: {scores}, weights: {weights}")

        # Predict with each normalization, blend
        final_pred = np.zeros((n, t, c), dtype=np.float64)
        for i, (mean, std) in enumerate(self.norm_stats):
            if weights[i] < 0.001:
                continue
            pred_raw = self._predict_with_stats(x, mean, std)
            final_pred += weights[i] * pred_raw

        return final_pred.astype(np.float32)

    def _predict_with_stats(self, x, mean, std):
        """Run full ensemble prediction with given normalization stats."""
        context_len = 10
        n, t, c, f = x.shape

        # Normalize using training stats (same pipeline as training)
        x_norm = normalize_data(x, mean, std)

        # Mask future (same as training)
        x_norm[:, context_len:] = x_norm[:, context_len - 1: context_len]

        # Weighted ensemble: best/ema_best get 2x weight vs snapshots
        weights = []
        for p in self._all_paths:
            bname = os.path.basename(p)
            weights.append(2.0 if "best" in bname else 1.0)
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        # Single clean forward pass per model (no TTA — hurts overfit models)
        batch_size = 8 if self.num_channels > 200 else 16

        weighted_sum = np.zeros((n, t, c), dtype=np.float64)

        for model_idx, model in enumerate(self.models):
            predictions = []
            with torch.no_grad():
                for i in range(0, n, batch_size):
                    batch = torch.from_numpy(
                        x_norm[i:i + batch_size]).to(device)
                    pred_norm = model(batch)  # (B, T, C)
                    predictions.append(pred_norm.cpu().numpy())
            model_pred = np.concatenate(predictions, axis=0)
            weighted_sum += weights[model_idx] * model_pred

        pred_norm_all = weighted_sum.astype(np.float32)

        # Denormalize with same stats used for normalization
        return denormalize_lmp(pred_norm_all, mean, std, f)
