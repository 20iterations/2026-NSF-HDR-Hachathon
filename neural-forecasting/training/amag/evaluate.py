"""Local evaluation matching Codabench scoring.py logic.

Uses the submission Model class for inference (TTA normalization + snapshot ensemble),
matching exactly what Codabench will run.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add project root so submission.model can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from submission.model import Model


def score_mse(prediction: np.ndarray, solution: np.ndarray) -> float:
    """Compute MSE matching Codabench scoring: only on steps 10-19.

    Args:
        prediction: (N, 20, C) — model output (raw scale)
        solution: (N, 20, C, F) or (N, 20, C) — ground truth (raw scale)
    Returns:
        MSE float
    """
    sol_lmp = solution[:, :, :, 0] if solution.ndim == 4 else solution

    pred_t = torch.tensor(prediction[:, 10:])
    sol_t = torch.tensor(sol_lmp[:, 10:])

    mse = torch.nn.functional.mse_loss(pred_t, sol_t, reduction="mean")
    return mse.item()


def evaluate_all(checkpoint_dir: str = "checkpoints",
                 dataset_dir: str = "dataset",
                 val_only: bool = False,
                 val_split: float = 0.1,
                 seed: int = 42):
    """Evaluate using the submission Model on train files.

    Args:
        val_only: If True, evaluate only on the held-out validation split
                  (same-day session 0 only), giving honest local metrics.
    """

    # Copy checkpoints to submission/ so the Model class finds them
    ckpt_path = Path(checkpoint_dir)
    sub_path = Path("submission")

    for monkey_name in ["beignet", "affi"]:
        print(f"\n{'='*50}")
        print(f"  {monkey_name.upper()}")
        print(f"{'='*50}")

        # Symlink/copy checkpoints to submission dir
        _link_checkpoints(ckpt_path, sub_path, monkey_name)

        # Load submission model
        model = Model(monkey_name)
        model.load()

        # Evaluate on each training file
        if monkey_name == "beignet":
            train_files = [
                "train_data_beignet.npz",
                "train_data_beignet_2022-06-01_private.npz",
                "train_data_beignet_2022-06-02_private.npz",
            ]
        else:
            train_files = [
                "train_data_affi.npz",
                "train_data_affi_2024-03-20_private.npz",
            ]

        for fidx, fname in enumerate(train_files):
            fpath = Path(dataset_dir) / "train" / fname
            if not fpath.exists():
                print(f"  {fname}: NOT FOUND")
                continue

            data = np.load(str(fpath))["arr_0"]  # (N, 20, C, 9)

            if val_only:
                # Only evaluate on held-out val split from session 0 (same-day)
                if fidx != 0:
                    print(f"  {fname}: SKIPPED (val_only mode, cross-date)")
                    continue
                rng = np.random.RandomState(seed)
                n = len(data)
                indices = rng.permutation(n)
                n_val = max(1, int(n * val_split))
                val_indices = indices[:n_val]
                data = data[val_indices]
                print(f"  {fname} (val split: {len(data)} samples)")
            else:
                label = "same-day" if "private" not in fname else "cross-date"
                print(f"  {fname}")

            pred = model.predict(data)  # (N, 20, C)
            mse = score_mse(pred, data)

            if val_only:
                print(f"    [val-only] MSE (steps 10-19): {mse:.2f}")
            else:
                print(f"    [{label}] MSE (steps 10-19): {mse:.2f}")


def _link_checkpoints(ckpt_dir: Path, sub_dir: Path, monkey_name: str):
    """Copy checkpoint files and norm stats to submission directory.

    When seed-based checkpoints exist, removes stale top-level checkpoints
    from submission/ to prevent config detection from loading old models.
    """
    import shutil

    patterns = [f"amag_{monkey_name}_snap*.pth",
                f"amag_{monkey_name}_best.pth",
                f"amag_{monkey_name}_ema_best.pth",
                f"norm_stats_{monkey_name}.npz"]

    has_seed_dirs = any(d.is_dir() for d in ckpt_dir.glob("seed_*"))

    if has_seed_dirs:
        # Remove stale top-level checkpoints for THIS monkey only
        for pattern in patterns:
            for stale in sub_dir.glob(pattern):
                stale.unlink()
        # Remove stale files for THIS monkey within existing seed dirs
        # (don't rmtree — other monkey's files may be there)
        for stale_seed in sub_dir.glob("seed_*"):
            if stale_seed.is_dir():
                for pattern in patterns:
                    for stale in stale_seed.glob(pattern):
                        stale.unlink()

    # Top-level checkpoints (only if no seed dirs — legacy single-seed mode)
    if not has_seed_dirs:
        for pattern in patterns:
            for src in ckpt_dir.glob(pattern):
                dst = sub_dir / src.name
                if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                    shutil.copy2(str(src), str(dst))

    # Seed-based checkpoints: copy into seed_* subdirs in submission/
    for seed_dir in ckpt_dir.glob("seed_*"):
        if seed_dir.is_dir():
            dst_seed_dir = sub_dir / seed_dir.name
            dst_seed_dir.mkdir(exist_ok=True)
            for pattern in patterns:
                for src in seed_dir.glob(pattern):
                    dst = dst_seed_dir / src.name
                    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                        shutil.copy2(str(src), str(dst))


if __name__ == "__main__":
    val_only = "--val-only" in sys.argv
    evaluate_all(val_only=val_only)
