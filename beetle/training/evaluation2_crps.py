from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np
import math
from tqdm import tqdm
from datasets import load_dataset

from utils import (
    get_training_args,
    get_bioclip,
    get_collate_fn,
)
from model import BioClip2_DeepRegressor
from model_attn import BioClip2_AttnGaussianRegressor, BioClip2_AttnPointRegressor
from model_attn_v2 import BioClip2_AttnGaussianRegressorV2, BioClip2_AttnPointRegressorV2


def compile_event_gaussian_predictions(all_gts, all_mu, all_sigma, all_events, eps=1e-12):
    all_mu = np.array(all_mu)
    all_sigma = np.array(all_sigma)
    all_gts = np.array(all_gts)
    all_events = np.array(all_events)

    unique_events = np.unique(all_events)
    mu_event = []
    sigma_event = []
    gts_event = []

    for uevent in unique_events:
        indices = np.where(all_events == uevent)[0]
        if indices.size == 0:
            continue

        mu_i = all_mu[indices]  # (N_i, 3)
        sigma_i = all_sigma[indices]  # (N_i, 3)
        gts_i = all_gts[indices]  # (N_i, 3)

        mu_e = mu_i.mean(axis=0)
        var_mu = ((mu_i - mu_e) ** 2).mean(axis=0)  # population variance
        mean_sigma2 = (sigma_i ** 2).mean(axis=0)
        sigma_e = np.sqrt(np.maximum(mean_sigma2 + var_mu, eps))

        mu_event.append(mu_e)
        sigma_event.append(sigma_e)
        gts_event.append(gts_i.mean(axis=0))

    return np.stack(gts_event), np.stack(mu_event), np.stack(sigma_event)


def gaussian_crps(y, mu, sigma, eps=1e-12):
    sigma = torch.clamp(sigma, min=eps)
    z = (y - mu) / sigma
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def gaussian_crps_loss(y, mu, sigma):
    crps = gaussian_crps(y=y, mu=mu, sigma=sigma)  # [B, 3]
    per_target = crps.mean(dim=0)
    loss = per_target.mean()
    return loss, per_target


def save_crps_results(save_path, crps_scores):
    save_data = {}
    for i, tgt in enumerate(["SPEI_30d", "SPEI_1y", "SPEI_2y"]):
        save_data[tgt] = {"CRPS": float(crps_scores[i])}
    with open(save_path, "w") as f:
        import json

        json.dump(save_data, f)


def evaluate(model, dataloader):
    with torch.inference_mode():
        tbar = tqdm(dataloader, desc="Evaluating model (CRPS)")
        all_mu = []
        all_sigma = []
        all_gts = []
        all_events = []
        for imgs, targets, eventIDs in tbar:
            imgs = imgs.cuda()
            targets = targets.cuda()

            outputs = model(imgs)
            if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                mu, sigma = outputs
            else:
                mu = outputs
                sigma = torch.ones_like(mu)

            all_mu.extend(mu.detach().cpu().numpy())
            all_sigma.extend(sigma.detach().cpu().numpy())
            all_gts.extend(targets.detach().cpu().numpy())
            all_events.extend(np.array(eventIDs))

        gts_event, mu_event, sigma_event = compile_event_gaussian_predictions(
            all_gts, all_mu, all_sigma, all_events
        )

        y = torch.tensor(gts_event, device="cuda", dtype=torch.float32)
        mu = torch.tensor(mu_event, device="cuda", dtype=torch.float32)
        sigma = torch.tensor(sigma_event, device="cuda", dtype=torch.float32)
        _, per_target_crps = gaussian_crps_loss(y=y, mu=mu, sigma=sigma)

        print(f"test CRPS SPEI_30d {per_target_crps[0].item()}")
        print(f"test CRPS SPEI_1y {per_target_crps[1].item()}")
        print(f"test CRPS SPEI_2y {per_target_crps[2].item()}")

    return [x.item() for x in per_target_crps]


def test_and_save(test_dataset, save_path, batch_size, num_workers, model):
    dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=get_collate_fn(["eventID"]),
    )

    crps_scores = evaluate(dataloader=dataloader, model=model)
    save_crps_results(save_path, crps_scores)


def main():
    args = get_training_args()
    save_dir = Path(__file__).resolve().parent

    bioclip, transforms = get_bioclip()
    if getattr(args, "model", "mlp") == "attn_gauss":
        model = BioClip2_AttnGaussianRegressor(bioclip=bioclip).cuda()
    elif getattr(args, "model", "mlp") == "attn_point":
        model = BioClip2_AttnPointRegressor(bioclip=bioclip).cuda()
    elif getattr(args, "model", "mlp") == "attn_gauss_v2":
        model = BioClip2_AttnGaussianRegressorV2(bioclip=bioclip).cuda()
    elif getattr(args, "model", "mlp") == "attn_point_v2":
        model = BioClip2_AttnPointRegressorV2(bioclip=bioclip).cuda()
    else:
        model = BioClip2_DeepRegressor(bioclip).cuda()

    weights_path = args.weights
    if weights_path is None:
        weights_path = str(save_dir / "model_crps_trainsigma1.pth")
    weights_path = Path(weights_path)
    if not weights_path.is_absolute():
        weights_path = save_dir / weights_path
    model.regressor.load_state_dict(torch.load(weights_path))

    ds = load_dataset(
        "imageomics/sentinel-beetles",
        token=args.hf_token,
        split="validation",
    )

    def dset_transforms(examples):
        examples["pixel_values"] = [
            transforms(img.convert("RGB")) for img in examples["file_path"]
        ]
        return examples

    test_dset = ds.with_transform(dset_transforms)

    test_and_save(
        test_dset,
        save_dir / "results_crps.json",
        args.batch_size,
        args.num_workers,
        model,
    )


if __name__ == "__main__":
    main()


