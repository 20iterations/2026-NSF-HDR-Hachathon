"""Training script for AMAG model.

Usage:
    python run_train.py beignet --phase paper    # Phase 1: paper-faithful reproduction
    python run_train.py affi --phase compete     # Phase 2: OOD competition config
    python run_train.py beignet                  # Default: Phase 2 (compete)
    python run_train.py beignet --seeds 42 123 456  # Multi-seed ensemble
"""
import sys
import argparse
from amag.train import train_monkey
from amag.config import TrainConfig, phase1_config, phase2_config


def main():
    parser = argparse.ArgumentParser(description="Train AMAG model")
    parser.add_argument("monkey", nargs="?", default="beignet",
                        choices=["beignet", "affi"],
                        help="Monkey dataset to train on")
    parser.add_argument("--phase", default="compete",
                        choices=["paper", "compete"],
                        help="paper = Phase 1 (faithful reproduction), "
                             "compete = Phase 2 (OOD extensions)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Train with multiple seeds for ensemble diversity")
    args = parser.parse_args()

    if args.phase == "paper":
        cfg = phase1_config()
    else:
        cfg = phase2_config()

    if args.epochs is not None:
        cfg.epochs = args.epochs

    if args.monkey == "affi":
        cfg.batch_size = 16  # 239 channels
    else:
        cfg.batch_size = 32  # 89 channels

    # Default to 3 seeds for compete phase (multi-seed ensemble)
    if args.seeds:
        seeds = args.seeds
    elif args.phase == "compete":
        seeds = [42, 123, 456]
    else:
        seeds = [cfg.seed]
    for i, seed in enumerate(seeds):
        if len(seeds) > 1:
            print(f"\n{'='*60}")
            print(f"  Seed {i+1}/{len(seeds)}: {seed}")
            print(f"{'='*60}")
            # Save to different checkpoint dirs for multi-seed
            cfg.checkpoint_dir = f"checkpoints/seed_{seed}"

        cfg.seed = seed
        train_monkey(args.monkey, cfg)


if __name__ == "__main__":
    main()
