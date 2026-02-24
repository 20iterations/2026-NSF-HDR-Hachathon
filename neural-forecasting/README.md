# Neural Forecasting — NSF HDR Scientific Modeling out of distribution 2026

This repository contains training code and submission files for the [HDR Scientific Mood (Modeling out of distribution) Challenge: Neural Forecasting](https://www.codabench.org/competitions/9806/) from team 20iterations.

## Repository Structure

```
├── submission/                        # Codabench submission (model + weights)
│   ├── model.py                       # Self-contained inference model
│   ├── requirements.txt               # Inference dependencies
│   └── checkpoints/                   # Trained weights (3-seed ensemble)
│       ├── seed_42/
│       │   ├── amag_affi_best.pth
│       │   ├── amag_affi_ema_best.pth
│       │   ├── amag_affi_snap[1-5].pth
│       │   ├── amag_beignet_best.pth
│       │   ├── amag_beignet_ema_best.pth
│       │   ├── amag_beignet_snap[1-5].pth
│       │   ├── norm_stats_affi.npz
│       │   └── norm_stats_beignet.npz
│       ├── seed_123/  (same structure)
│       └── seed_456/  (same structure)
├── training/                          # Training code
│   ├── run_train.py                   # Entry point
│   └── amag/
│       ├── config.py                  # TrainConfig + phase1/phase2 configs
│       ├── model.py                   # AMAG model definition
│       ├── train.py                   # Training loop (EMA, mixup, snapshots)
│       ├── data.py                    # Dataset, normalization, augmentation
│       ├── adjacency.py               # Correlation matrix initialization
│       ├── losses.py                  # MMD + spectral losses
│       ├── evaluate.py                # Local evaluation
│       └── modules/
│           ├── temporal_encoding.py   # TE: multi-head transformer encoder
│           ├── spatial_interaction.py # SI: additive + multiplicative GNN
│           ├── temporal_readout.py    # TR: transformer decoder
│           ├── channel_attention.py   # iTransformer cross-channel mixing
│           ├── revin.py               # Context-only instance normalization
│           ├── dann.py                # Domain adversarial network
│           └── dish_ts.py             # Distribution shift coefficients
├── .gitignore
├── pyproject.toml
└── README.md
```

## Architecture

```
Input (B, 20, C, 9)
  → Feature Pathways (LMP + Spectral split embedding)
  → TE (Transformer Encoder with positional encoding)
  → Session Embedding (additive, for cross-day adaptation)
  → SI (Additive + Multiplicative message-passing with learnable adjacency)
  → Channel Attention (iTransformer-style cross-channel mixing)
  → TR (Transformer Readout with separate parameters)
  → Output (B, 20, C)
```

The model takes 10 context timesteps and predicts the next 10 timesteps of LMP (Local Motor Potential) activity across all channels.

### OOD Extensions

| Method | Purpose |
|--------|---------|
| RevIN (context-only) | Instance normalization using context window stats |
| Channel-weighted loss | Weights training loss by per-channel variance |
| Session embeddings | Learnable per-session offsets for cross-day adaptation |
| Channel attention | Sample-dependent cross-channel spatial mixing |
| Feature pathways | Separate LMP and spectral band embedding |
| Frequency augmentation | Phase perturbation in FFT domain |
| EMA | Exponential moving average for stable generalization |
| Snapshot ensemble | Averages predictions from 5 cosine cycle checkpoints |
| Mixup | Interpolates training samples for regularization |

## Datasets

Neural recordings from two monkeys performing center-out reaching tasks:

- **Monkey Affogato (affi):** 239 µECoG electrodes, 5 motor cortex regions
- **Monkey Beignet (beignet):** 87 µECoG electrodes, M1 region only

Each `.npz` file contains data shaped `(N, 20, C, 9)`:
- **N** = number of trials
- **20** = timesteps (10 context + 10 forecast, 30ms bins)
- **C** = channels (89 for beignet, 239 for affi)
- **9** = features (LMP + 8 frequency bands)

To reproduce training, place the dataset files under `dataset/` relative to the repo root (the training script reads from `dataset/train/` by default):

```
dataset/
├── train/
│   ├── train_data_affi.npz
│   ├── train_data_affi_2024-03-20_private.npz
│   ├── train_data_beignet.npz
│   ├── train_data_beignet_2022-06-01_private.npz
│   └── train_data_beignet_2022-06-02_private.npz
└── test/
    ├── test_data_affi_masked.npz
    ├── test_data_affi_2024-03-20_private_masked.npz
    ├── test_data_beignet_masked.npz
    ├── test_data_beignet_2022-06-01_private_masked.npz
    └── test_data_beignet_2022-06-02_private_masked.npz
```

The dataset is provided by the challenge organizers and is not included in this repository.

## Installation

Requires Python >= 3.10 and a CUDA GPU.

```bash
pip install uv
uv sync
```

Or install directly:

```bash
pip install numpy scipy tqdm torch
```

## Training

```bash
cd training

# Default: competition config, 3-seed ensemble
python run_train.py beignet
python run_train.py affi

# Custom seeds or epochs
python run_train.py beignet --seeds 42 123 456 --epochs 400
```

Checkpoints are saved to `checkpoints/seed_<N>/` per seed.

## Submission

The `submission/` directory contains everything needed for Codabench evaluation:

```python
class Model:
    def __init__(self, monkey_name):  # 'beignet' or 'affi'
    def load(self):                    # Auto-detects config, loads 3-seed ensemble
    def predict(self, x):              # (N, 20, C, 9) → (N, 20, C)
```

The inference pipeline: session-matched normalization → 3-seed weighted snapshot ensemble (best + EMA + 5 snapshots per seed) → denormalization.

Pre-trained weights are included in `submission/checkpoints/`.

## References

- Li et al., ["AMAG: Additive, Multiplicative and Adaptive Graph Neural Network For Forecasting Neural Activity"](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1c70ba3591d0694a535089e1c25888d7-Abstract-Conference.html), NeurIPS 2023
- Kim et al., ["Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift"](https://openreview.net/forum?id=cGDAkQo1C0p), ICLR 2022
- Liu et al., ["iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"](https://arxiv.org/abs/2310.06625), ICLR 2024
- Zhang et al., ["mixup: Beyond Empirical Risk Minimization"](https://arxiv.org/abs/1710.09412), ICLR 2018
- Huang et al., ["Snapshot Ensembles: Train 1, Get M for Free"](https://arxiv.org/abs/1704.00109), ICLR 2017
