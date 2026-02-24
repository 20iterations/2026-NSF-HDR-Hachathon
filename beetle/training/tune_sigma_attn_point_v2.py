from __future__ import annotations

from pathlib import Path
import sys
import math
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from utils import get_training_args, get_bioclip, get_collate_fn
from model_attn_v2 import BioClip2_AttnPointRegressorV2


def _parse_extra_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--sigma_min", type=float, default=0.4)
    p.add_argument("--sigma_max", type=float, default=2.0)
    p.add_argument("--sigma_steps", type=int, default=33)
    p.add_argument("--sigma_list", type=str, default="")
    p.add_argument("--output_json", type=str, default="sigma_tuning_attn_point_v2.json")
    ns, rest = p.parse_known_args(argv)
    return ns, rest


def compile_event_gaussian_predictions(all_gts, all_mu, all_sigma, all_events, eps=1e-12):
    all_mu = np.array(all_mu)
    all_sigma = np.array(all_sigma)
    all_gts = np.array(all_gts)
    all_events = np.array(all_events)
    mu_event, sigma_event, gts_event = [], [], []
    for e in np.unique(all_events):
        idx = np.where(all_events == e)[0]
        if idx.size == 0:
            continue
        mu_i = all_mu[idx]
        sigma_i = all_sigma[idx]
        g_i = all_gts[idx]
        mu_e = mu_i.mean(axis=0)
        var_mu = ((mu_i - mu_e) ** 2).mean(axis=0)
        sigma_e = np.sqrt(np.maximum((sigma_i**2).mean(axis=0) + var_mu, eps))
        mu_event.append(mu_e)
        sigma_event.append(sigma_e)
        gts_event.append(g_i.mean(axis=0))
    return np.stack(gts_event), np.stack(mu_event), np.stack(sigma_event)


def gaussian_crps(y, mu, sigma, eps=1e-12):
    sigma = torch.clamp(sigma, min=eps)
    z = (y - mu) / sigma
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def _collect_validation_outputs(model, dataloader):
    all_mu, all_gts, all_events = [], [], []
    with torch.inference_mode():
        for imgs, targets, eventIDs in tqdm(dataloader, desc="Collecting attn_v2 validation outputs"):
            imgs = imgs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            mu = model(imgs)
            all_mu.extend(mu.detach().cpu().numpy())
            all_gts.extend(targets.detach().cpu().numpy())
            all_events.extend(np.array(eventIDs))
    return np.array(all_gts), np.array(all_mu), np.array(all_events)


def _score_sigma(all_gts, all_mu, all_events, sigma_scale: float):
    sigma = float(sigma_scale) * np.ones_like(all_mu)
    gts_e, mu_e, sigma_e = compile_event_gaussian_predictions(all_gts, all_mu, sigma, all_events)
    y = torch.tensor(gts_e, device="cuda", dtype=torch.float32)
    m = torch.tensor(mu_e, device="cuda", dtype=torch.float32)
    s = torch.tensor(sigma_e, device="cuda", dtype=torch.float32)
    crps = gaussian_crps(y, m, s).mean(dim=0)
    crps_list = [float(crps[0].item()), float(crps[1].item()), float(crps[2].item())]
    return crps_list, float(np.mean(crps_list))


def _build_sigma_grid(extra):
    if str(extra.sigma_list).strip():
        vals = [float(x.strip()) for x in str(extra.sigma_list).split(",") if x.strip()]
        if not vals:
            raise ValueError("--sigma_list was provided but no valid values were parsed.")
        return vals
    return np.linspace(float(extra.sigma_min), float(extra.sigma_max), max(int(extra.sigma_steps), 2)).tolist()


def main():
    extra, rest_argv = _parse_extra_args(sys.argv[1:])
    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]] + rest_argv
        args = get_training_args()
    finally:
        sys.argv = old_argv

    save_dir = Path(__file__).resolve().parent
    bioclip, transforms = get_bioclip()
    model = BioClip2_AttnPointRegressorV2(bioclip=bioclip).cuda()
    w = Path(args.weights) if args.weights else (save_dir / "model_attn_point_v2.pth")
    if not w.is_absolute():
        w = save_dir / w
    model.regressor.load_state_dict(torch.load(w, map_location="cpu"))
    model.eval()

    ds = load_dataset("imageomics/sentinel-beetles", token=args.hf_token, split="validation")

    def dset_transforms(examples):
        examples["pixel_values"] = [transforms(img.convert("RGB")) for img in examples["file_path"]]
        return examples

    dset = ds.with_transform(dset_transforms)
    loader = DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=get_collate_fn(["eventID"]),
    )

    all_gts, all_mu, all_events = _collect_validation_outputs(model, loader)
    sigma_grid = _build_sigma_grid(extra)

    best = {"weights": str(w), "best_sigma_scale": 1.0, "best_mean_crps": float("inf"), "best_crps": None, "results": []}
    for s in sigma_grid:
        crps3, mean_crps = _score_sigma(all_gts, all_mu, all_events, sigma_scale=float(s))
        row = {
            "sigma_scale": float(s),
            "mean_crps": float(mean_crps),
            "crps": {"SPEI_30d": float(crps3[0]), "SPEI_1y": float(crps3[1]), "SPEI_2y": float(crps3[2])},
        }
        best["results"].append(row)
        if mean_crps < best["best_mean_crps"]:
            best["best_sigma_scale"] = float(s)
            best["best_mean_crps"] = float(mean_crps)
            best["best_crps"] = row["crps"]

    out_path = Path(extra.output_json)
    if not out_path.is_absolute():
        out_path = save_dir / out_path
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print(f"Best sigma_scale={best['best_sigma_scale']:.4f}, mean CRPS={best['best_mean_crps']:.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

