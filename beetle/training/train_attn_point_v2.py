"""
train_attn_point_v2.py

Based on train_attn_point.py but with:
- v2 attention head (learned tokenization + optional gating + stochastic depth)
- keeps your existing features: image augmentation + aug_repeats + balancing
- more stable training: AdamW + cosine LR + grad clip + early stopping

This remains a point model (mu only), so it can still be scored by evaluation2_crps.py:
  python evaluation2_crps.py --model attn_point_v2 --weights model_attn_point_v2.pth ...
"""

from pathlib import Path
from datetime import datetime
import json
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

from utils import (
    get_training_args,
    get_bioclip,
    evalute_spei_r2_scores,
    extract_bioclip_features_with_meta,
    get_collate_fn,
)
from model_attn_v2 import BioClip2_AttnPointRegressorV2


def _inverse_freq_weights(values, power: float = 1.0):
    power = float(power)
    if power <= 0:
        return None
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    w = np.array([(1.0 / counts[v]) ** power for v in values], dtype=np.float64)
    return torch.from_numpy(w)

def _save_run_metadata(save_dir: Path, args, extra: dict | None = None):
    """
    Persist the exact CLI args for reproducibility.
    Writes:
      - run_attn_point_v2_last.json (overwritten each run)
      - run_attn_point_v2_<timestamp>.json (archived)
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
        "args": vars(args),
    }
    if extra:
        meta.update(extra)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    last_path = save_dir / "run_attn_point_v2_last.json"
    arch_path = save_dir / f"run_attn_point_v2_{ts}.json"
    for p in (last_path, arch_path):
        with open(p, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def train(
    model,
    dataloader,
    val_dataloader,
    lr,
    epochs,
    weight_decay,
    grad_clip_norm,
    patience,
    save_dir,
):
    reg = model.regressor
    optimizer = optim.AdamW(reg.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = 0
    patience_left = int(patience)
    save_path = Path(save_dir, "model_attn_point_v2.pth")
    history = {
        "epoch": [],
        "lr": [],
        "train_loss": [],
        "val_loss": [],
        "train_r2_mean": [],
        "val_r2_mean": [],
        "best_epoch": None,
        "best_val_loss": None,
    }

    print("begin training (attn_point_v2, AdamW+cosine, MSE on frozen BioCLIP features)")
    tbar = tqdm(range(int(epochs)), position=0, leave=True)
    for epoch in tbar:
        reg.train()
        train_loss_sum = 0.0
        train_count = 0
        preds = []
        gts = []
        inner = tqdm(dataloader, "training", position=1, leave=False)
        for feats, y in inner:
            y = y.cuda()
            optimizer.zero_grad()
            out = reg(feats.cuda())
            loss = loss_fn(out, y)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(reg.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()

            bs = y.shape[0]
            train_loss_sum += float(loss.item()) * bs
            train_count += bs
            preds.append(out.detach().cpu())
            gts.append(y.detach().cpu())
            inner.set_postfix({"loss": float(loss.item())})

        preds_np = torch.cat(preds, dim=0).numpy()
        gts_np = torch.cat(gts, dim=0).numpy()
        tr_r2 = evalute_spei_r2_scores(gts_np, preds_np)
        train_loss = train_loss_sum / max(train_count, 1)

        # ---- val ----
        reg.eval()
        val_loss_sum = 0.0
        val_count = 0
        preds = []
        gts = []
        inner = tqdm(val_dataloader, "validating", position=1, leave=False)
        with torch.no_grad():
            for feats, y in inner:
                y = y.cuda()
                out = reg(feats.cuda())
                loss = loss_fn(out, y)
                bs = y.shape[0]
                val_loss_sum += float(loss.item()) * bs
                val_count += bs
                preds.append(out.detach().cpu())
                gts.append(y.detach().cpu())
                inner.set_postfix({"loss": float(loss.item())})

        preds_np = torch.cat(preds, dim=0).numpy()
        gts_np = torch.cat(gts, dim=0).numpy()
        va_r2 = evalute_spei_r2_scores(gts_np, preds_np)
        val_loss = val_loss_sum / max(val_count, 1)

        scheduler.step()
        cur_lr = float(optimizer.param_groups[0]["lr"])

        improved = val_loss < (best_val - 1e-8)
        if improved:
            best_val = float(val_loss)
            best_epoch = int(epoch)
            patience_left = int(patience)
            torch.save(reg.state_dict(), save_path)
        else:
            patience_left -= 1

        history["epoch"].append(int(epoch))
        history["lr"].append(float(cur_lr))
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_r2_mean"].append(float(np.mean(tr_r2)))
        history["val_r2_mean"].append(float(np.mean(va_r2)))

        tbar.set_postfix(
            {
                "epoch": epoch,
                "lr": cur_lr,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "pat": int(patience_left),
                "best_epoch": int(best_epoch),
                "best_val": float(best_val),
                "train_r2_mean": float(np.mean(tr_r2)),
                "val_r2_mean": float(np.mean(va_r2)),
            }
        )

        if patience > 0 and patience_left <= 0:
            print(f"Early stopping at epoch={epoch} (best_epoch={best_epoch}, best_val={best_val})")
            break

    reg.load_state_dict(torch.load(save_path, map_location="cpu"))
    history["best_epoch"] = int(best_epoch)
    history["best_val_loss"] = float(best_val)

    # Save history + plot
    history_path = Path(save_dir, "loss_history_attn_point_v2.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history["epoch"], history["train_loss"], label="train loss", marker="o", markersize=3)
        ax.plot(history["epoch"], history["val_loss"], label="val loss", marker="o", markersize=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss")
        ax.set_title("train_attn_point_v2 loss curves")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig_path = Path(save_dir, "loss_curve_attn_point_v2.png")
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved loss curve: {fig_path}")
    except Exception as e:
        print(f"Could not save loss curve plot (matplotlib missing or error): {e}")

    print(f"DONE! saved weights: {save_path}")
    print(f"Saved loss history: {history_path}")


def main():
    args = get_training_args()
    save_dir = Path(__file__).resolve().parent
    _save_run_metadata(
        save_dir=save_dir,
        args=args,
        extra={"script": "train_attn_point_v2.py", "intended_weights": "model_attn_point_v2.pth"},
    )

    ds = load_dataset("imageomics/sentinel-beetles", token=args.hf_token)
    bioclip, transforms = get_bioclip()

    # v2 model (bioclip frozen; we train regressor only)
    model = BioClip2_AttnPointRegressorV2(bioclip=bioclip).cuda()

    # Optional image augmentation before BioCLIP preprocess
    if args.use_augment:
        try:
            import torchvision.transforms as T

            aug = T.Compose(
                [
                    T.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                ]
            )
        except Exception as e:
            raise RuntimeError("use_augment requires torchvision") from e
    else:
        aug = None

    def dset_transforms(examples):
        imgs = [img.convert("RGB") for img in examples["file_path"]]
        if aug is not None:
            imgs = [aug(img) for img in imgs]
        examples["pixel_values"] = [transforms(img) for img in imgs]
        return examples

    train_dset = ds["train"].with_transform(dset_transforms)
    val_dset = ds["validation"].with_transform(dset_transforms)

    loaders = []
    for split_i, dset in enumerate([train_dset, val_dset]):
        img_loader = DataLoader(
            dataset=dset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=get_collate_fn(["eventID", "domainID"]),
        )

        # repeats only for train split (feature-level augmentation)
        repeats = max(int(args.aug_repeats), 1) if split_i == 0 else 1
        Xs, Ys = [], []
        event_ids_all, domain_ids_all = [], []
        for _ in range(repeats):
            X, Y, meta = extract_bioclip_features_with_meta(
                dataloader=img_loader, bioclip=bioclip, meta_columns=["eventID", "domainID"]
            )
            Xs.append(X)
            Ys.append(Y)
            event_ids_all.extend(meta["eventID"])
            domain_ids_all.extend(meta["domainID"])

        X = torch.cat(Xs, dim=0)
        Y = torch.cat(Ys, dim=0)

        if split_i == 0 and args.balance_mode != "none":
            w = torch.ones(len(Y), dtype=torch.double)
            if args.balance_mode in ("event", "event_domain"):
                ew = _inverse_freq_weights(event_ids_all, power=args.balance_power)
                if ew is not None:
                    w *= ew
            if args.balance_mode in ("domain", "event_domain"):
                dw = _inverse_freq_weights(domain_ids_all, power=args.balance_power)
                if dw is not None:
                    w *= dw
            w = torch.clamp(w, min=1e-12)
            sampler = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)
            feat_loader = DataLoader(
                dataset=torch.utils.data.TensorDataset(X, Y),
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.num_workers,
            )
        else:
            feat_loader = DataLoader(
                dataset=torch.utils.data.TensorDataset(X, Y),
                batch_size=args.batch_size,
                shuffle=split_i == 0,
                num_workers=args.num_workers,
            )
        loaders.append(feat_loader)

    train_loader, val_loader = loaders
    train(
        model=model,
        dataloader=train_loader,
        val_dataloader=val_loader,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=getattr(args, "weight_decay", 1e-2),
        grad_clip_norm=getattr(args, "grad_clip_norm", 1.0),
        patience=getattr(args, "patience", 30),
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()





