from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class MonkeyConfig:
    name: str
    num_channels: int
    train_files: list[str]
    test_files: list[str]


AFFI = MonkeyConfig(
    name="affi",
    num_channels=239,
    train_files=[
        "train_data_affi.npz",
        "train_data_affi_2024-03-20_private.npz",
    ],
    test_files=[
        "test_data_affi_masked.npz",
        "test_data_affi_2024-03-20_private_masked.npz",
    ],
)

BEIGNET = MonkeyConfig(
    name="beignet",
    num_channels=89,
    train_files=[
        "train_data_beignet.npz",
        "train_data_beignet_2022-06-01_private.npz",
        "train_data_beignet_2022-06-02_private.npz",
    ],
    test_files=[
        "test_data_beignet_masked.npz",
        "test_data_beignet_2022-06-01_private_masked.npz",
        "test_data_beignet_2022-06-02_private_masked.npz",
    ],
)

MONKEY_CONFIGS = {"affi": AFFI, "beignet": BEIGNET}


@dataclass
class TrainConfig:
    # Data
    dataset_dir: str = "dataset"
    context_len: int = 10
    total_len: int = 20
    num_features: int = 9

    # Model
    hidden_dim: int = 128
    d_ff: int = 512
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    use_adaptor: bool = False
    use_channel_attn: bool = True
    use_feature_pathways: bool = True

    # Training
    batch_size: int = 16
    lr: float = 5e-4
    weight_decay: float = 1e-4
    lr_decay: float = 0.95
    lr_decay_every: int = 50
    epochs: int = 150
    val_split: float = 0.1
    val_every: int = 5
    seed: int = 42
    patience: int = 20
    accumulate_steps: int = 1  # Gradient accumulation: effective_batch = batch_size * this

    # Optimizer / Scheduler selection
    optimizer_type: str = "adamw"   # "adam" or "adamw"
    scheduler_type: str = "cosine"  # "step" or "cosine"

    # EMA (Exponential Moving Average)
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_start_epoch: int = 10

    # Snapshot Ensemble (CosineAnnealingWarmRestarts)
    num_snapshots: int = 3
    snapshot_cycle_len: int = 50

    # Augmentation (training only)
    aug_jitter_std: float = 0.02
    aug_scale_std: float = 0.1
    aug_channel_drop_p: float = 0.1

    # Mixup
    use_mixup: bool = True
    mixup_alpha: float = 0.3

    # OOD: RevIN (Kim et al., ICLR 2022)
    use_revin: bool = False

    # OOD: Dish-TS (Fan et al., AAAI 2023) — replaces RevIN
    use_dish_ts: bool = False

    # OOD: MMD loss (Gretton et al., JMLR 2012)
    mmd_lambda: float = 0.0

    # OOD: CORAL loss (Sun & Saenko, ECCV 2016)
    coral_lambda: float = 0.0

    # OOD: DANN (Ganin et al., JMLR 2016) — domain-adversarial training
    use_dann: bool = False
    dann_lambda: float = 0.0

    # Consistency regularization (Laine & Aila, ICLR 2017)
    use_consistency: bool = False
    consist_lambda: float = 0.0

    # Spectral loss weight
    spectral_lambda: float = 0.0

    # Loss type: "mse" or "huber"
    loss_type: str = "mse"

    # Channel-weighted loss: weight channels by std_lmp² to match raw-space eval
    use_channel_weights: bool = False
    channel_weight_max_ratio: float = 10.0  # Max ratio between highest/lowest channel weight

    # Session embeddings
    use_session_embed: bool = False
    num_sessions: int = 3

    # Warmup epochs (linear warmup from 0 to lr)
    warmup_epochs: int = 0

    # Output
    checkpoint_dir: str = "checkpoints"


def phase1_config() -> TrainConfig:
    """Paper-faithful AMAG config (Li et al., NeurIPS 2023)."""
    return TrainConfig(
        # Model: enable adaptor per paper, paper hidden_dim=64, single layer/head
        hidden_dim=64,
        d_ff=256,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
        use_adaptor=True,
        use_channel_attn=False,
        use_feature_pathways=False,
        # Optimizer: Adam (Kingma & Ba, ICLR 2015) — paper specifies Adam, not AdamW
        optimizer_type="adam",
        weight_decay=1e-5,
        # Scheduler: StepLR x0.95 every 50 epochs — paper specification
        scheduler_type="step",
        lr=5e-4,
        lr_decay=0.95,
        lr_decay_every=50,
        # Training: paper uses 1000 epochs, 500 with early stopping
        epochs=500,
        val_every=10,
        patience=50,
        # Disable all non-paper techniques
        use_ema=False,
        use_mixup=False,
        aug_jitter_std=0.0,
        aug_scale_std=0.0,
        aug_channel_drop_p=0.0,
        use_revin=False,
        mmd_lambda=0.0,
        spectral_lambda=0.0,
        loss_type="mse",
        use_session_embed=False,
        warmup_epochs=0,
    )


def phase2_config() -> TrainConfig:
    """Competition config v4.5: clamp channel weight ratio + longer training.

    v4.5: Clamp channel weight ratio to 10x (Beignet was 156.7x unclamped).
    Longer training (400 epochs, patience=50) with 5 snapshot cycles of 80 epochs.
    v4.4: Weight training loss by channel std_lmp² so high-variance channels
    (which dominate raw-space Codabench MSE) get proportionally more gradient.
    v4.3: Reduce augmentation for better same-day fit (jitter 0.01, scale 0.05,
    channel_drop 0.05, mixup 0.15). Multi-normalization weighted prediction
    in submission/model.py eliminates session matching failures.
    """
    return TrainConfig(
        # Model: paper-faithful d=64 (prevents overfitting on 630-1049 samples)
        hidden_dim=64,
        d_ff=256,
        num_heads=1,
        num_layers=1,
        dropout=0.1,
        use_adaptor=False,
        use_channel_attn=False,
        use_feature_pathways=False,
        # Optimizer: AdamW (Loshchilov & Hutter, ICLR 2019)
        optimizer_type="adamw",
        weight_decay=1e-4,
        # Scheduler: CosineAnnealingWarmRestarts (5 cycles of 80 epochs)
        scheduler_type="cosine",
        lr=5e-4,
        epochs=400,
        val_every=5,
        patience=50,
        warmup_epochs=5,
        # EMA (Polyak & Juditsky, 1992)
        use_ema=True,
        ema_decay=0.999,
        ema_start_epoch=15,
        # Snapshot ensemble — 5 snapshots from 80-epoch cycles
        num_snapshots=5,
        snapshot_cycle_len=80,
        # Augmentation — reduced for better same-day precision
        aug_jitter_std=0.01,
        aug_scale_std=0.05,
        aug_channel_drop_p=0.05,
        # Mixup (Zhang et al., ICLR 2018) — reduced alpha
        use_mixup=True,
        mixup_alpha=0.15,
        # No instance normalization — Dish-TS CONET overfit, RevIN unstable
        use_revin=False,
        use_dish_ts=False,
        # DANN disabled — destroys same-day signal for marginal cross-date gain
        use_dann=False,
        dann_lambda=0.0,
        # CORAL/consistency/MMD disabled
        coral_lambda=0.0,
        use_consistency=False,
        consist_lambda=0.0,
        mmd_lambda=0.0,
        # Spectral loss disabled
        spectral_lambda=0.0,
        # MSE loss with channel weighting to match raw-space evaluation
        loss_type="mse",
        use_channel_weights=True,
        # Session embeddings disabled
        use_session_embed=False,
        num_sessions=3,
    )
