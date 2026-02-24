import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional


def compute_norm_stats(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel-feature mean and std from data.

    Args:
        data: (N, T, C, F) array
    Returns:
        mean, std: each (1, C*F) arrays
    """
    n, t, c, f = data.shape
    flat = data.reshape(n * t, c * f)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    return mean, std


def normalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize data to [-1, 1] using [mean-4*std, mean+4*std] range.

    Args:
        data: (N, T, C, F)
        mean, std: (1, C*F)
    Returns:
        Normalized data (N, T, C, F)
    """
    n, t, c, f = data.shape
    flat = data.reshape(n * t, c * f)
    lo = mean - 4 * std
    hi = mean + 4 * std
    denom = hi - lo
    denom = np.where(denom == 0, 1.0, denom)
    norm = 2 * (flat - lo) / denom - 1
    return norm.reshape(n, t, c, f)


def denormalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray,
                num_channels: int, num_features: int = 9) -> np.ndarray:
    """Denormalize data from [-1, 1] back to original scale.

    Args:
        data: (N, T, C) — LMP predictions only (feature 0)
        mean, std: (1, C*F) — full feature stats
    Returns:
        Denormalized data (N, T, C)
    """
    mean_lmp = mean[0, ::num_features]  # (C,)
    std_lmp = std[0, ::num_features]    # (C,)

    lo = mean_lmp - 4 * std_lmp
    hi = mean_lmp + 4 * std_lmp
    denom = hi - lo
    denom = np.where(denom == 0, 1.0, denom)

    return (data + 1) * denom / 2 + lo


def denormalize_torch(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                      num_features: int = 9) -> torch.Tensor:
    """Denormalize torch tensor from [-1, 1] back to original scale. LMP only."""
    mean_lmp = mean[0, ::num_features]
    std_lmp = std[0, ::num_features]
    lo = mean_lmp - 4 * std_lmp
    hi = mean_lmp + 4 * std_lmp
    denom = hi - lo
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return (data + 1) * denom / 2 + lo


class NeuralForecastDataset(Dataset):
    def __init__(self, data_norm: np.ndarray, targets_raw: np.ndarray,
                 context_len: int = 10, session_ids: np.ndarray | None = None,
                 augment: bool = False,
                 jitter_std: float = 0.02,
                 scale_std: float = 0.1,
                 channel_drop_p: float = 0.1,
                 freq_augment: bool = False):
        """
        Args:
            data_norm: (N, T, C, F) already normalized data
            targets_raw: (N, T, C) raw LMP targets
            context_len: number of context timesteps
            session_ids: (N,) integer session index per trial (for cross-session mixup)
            augment: whether to apply on-the-fly augmentation
            jitter_std: Gaussian noise std for jittering
            scale_std: per-channel random scale std
            channel_drop_p: probability of zeroing each channel
        """
        self.context_len = context_len
        self.data_norm = data_norm.astype(np.float32)
        self.targets_raw = targets_raw.astype(np.float32)
        self.session_ids = session_ids
        self.augment = augment
        self.jitter_std = jitter_std
        self.scale_std = scale_std
        self.channel_drop_p = channel_drop_p
        self.freq_augment = freq_augment

    def __len__(self):
        return len(self.data_norm)

    def __getitem__(self, idx):
        x = self.data_norm[idx].copy()  # (T, C, F)

        if self.augment:
            x = self._apply_augmentation(x)

        if self.freq_augment and self.augment:
            x = self._apply_freq_augmentation(x)

        # Mask future: replace steps after context with copy of last context step
        x[self.context_len:] = x[self.context_len - 1]

        x_tensor = torch.from_numpy(x)
        target_norm = torch.from_numpy(self.data_norm[idx, :, :, 0])  # (T, C)
        target_raw = torch.from_numpy(self.targets_raw[idx])  # (T, C)

        if self.session_ids is not None:
            session_id = torch.tensor(self.session_ids[idx], dtype=torch.long)
            return x_tensor, target_norm, target_raw, session_id

        return x_tensor, target_norm, target_raw

    def _apply_augmentation(self, x: np.ndarray) -> np.ndarray:
        """Apply jittering, scaling, and channel dropout to normalized input.

        Args:
            x: (T, C, F) normalized data
        Returns:
            Augmented (T, C, F)
        """
        T, C, F = x.shape

        # Jittering: additive Gaussian noise
        if self.jitter_std > 0:
            x = x + np.random.randn(*x.shape).astype(np.float32) * self.jitter_std

        # Scaling: per-channel multiplicative noise (simulates impedance drift)
        if self.scale_std > 0:
            scale = np.random.normal(1.0, self.scale_std, size=(1, C, 1)).astype(np.float32)
            x = x * scale

        # Channel dropout: zero out random channels entirely
        if self.channel_drop_p > 0:
            mask = (np.random.rand(1, C, 1) > self.channel_drop_p).astype(np.float32)
            x = x * mask

        return x

    def _apply_freq_augmentation(self, x: np.ndarray) -> np.ndarray:
        """Apply phase perturbation in frequency domain.

        Preserves spectral content while creating diverse training samples.
        Better than Gaussian jitter for OOD generalization.

        Args:
            x: (T, C, F) normalized data
        Returns:
            Augmented (T, C, F)
        """
        X_fft = np.fft.rfft(x, axis=0)  # FFT along time axis
        phase_noise = np.random.uniform(-0.1 * np.pi, 0.1 * np.pi, X_fft.shape)  # v3.5: restored from v3.3
        X_fft = X_fft * np.exp(1j * phase_noise)
        return np.fft.irfft(X_fft, n=x.shape[0], axis=0).astype(np.float32)


def load_all_train_data(dataset_dir: str, train_files: list[str]) -> list[np.ndarray]:
    """Load all training data files for a monkey."""
    datasets = []
    base = Path(dataset_dir) / "train"
    for fname in train_files:
        path = base / fname
        data = np.load(str(path))["arr_0"]
        datasets.append(data)
    return datasets


def prepare_datasets(dataset_dir: str, train_files: list[str],
                     context_len: int = 10, val_split: float = 0.1,
                     seed: int = 42,
                     augment: bool = False,
                     jitter_std: float = 0.02,
                     scale_std: float = 0.1,
                     channel_drop_p: float = 0.1,
                     freq_augment: bool = False):
    """Load and prepare train/val datasets with per-session normalization.

    Each session is normalized independently using its own statistics.
    All sessions are used for training with val_split held out from each.
    Cross-date data acts as natural regularization (different observation
    noise, same underlying dynamics).

    Returns:
        train_dataset, val_dataset, per_session_stats
    """
    all_data = load_all_train_data(dataset_dir, train_files)

    # Per-session normalization: normalize each session with its own stats
    all_norm = []
    all_raw_lmp = []
    per_session_stats = []

    for session_data in all_data:
        mean, std = compute_norm_stats(session_data)
        per_session_stats.append((mean, std))
        norm = normalize(session_data, mean, std)
        all_norm.append(norm)
        all_raw_lmp.append(session_data[:, :, :, 0])

    # Hold out val_split from session 0 (same-day) only.
    # Cross-date sessions go 100% into training — their val samples would get
    # wrong denormalization (session 0 stats) and corrupt model selection.
    rng = np.random.RandomState(seed)

    train_norm_parts = []
    train_raw_parts = []
    train_session_parts = []
    val_norm_parts = []
    val_raw_parts = []

    for i, (norm, raw_lmp) in enumerate(zip(all_norm, all_raw_lmp)):
        n = len(norm)
        indices = rng.permutation(n)
        if i == 0:  # Same-day: hold out val
            n_val = max(1, int(n * val_split))
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            val_norm_parts.append(norm[val_idx])
            val_raw_parts.append(raw_lmp[val_idx])
        else:  # Cross-date: 100% training
            train_idx = indices

        train_norm_parts.append(norm[train_idx])
        train_raw_parts.append(raw_lmp[train_idx])
        train_session_parts.append(np.full(len(train_idx), i, dtype=np.int64))

    train_norm = np.concatenate(train_norm_parts, axis=0)
    train_raw = np.concatenate(train_raw_parts, axis=0)
    train_sessions = np.concatenate(train_session_parts, axis=0)

    val_norm = np.concatenate(val_norm_parts, axis=0)
    val_raw = np.concatenate(val_raw_parts, axis=0)

    train_ds = NeuralForecastDataset(
        train_norm, train_raw, context_len,
        session_ids=train_sessions,
        augment=augment,
        jitter_std=jitter_std,
        scale_std=scale_std,
        channel_drop_p=channel_drop_p,
        freq_augment=freq_augment,
    )
    val_ds = NeuralForecastDataset(val_norm, val_raw, context_len)

    return train_ds, val_ds, per_session_stats
