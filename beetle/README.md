# Beetle V2 Baseline Package

This folder contains a compact V2 baseline workflow for the HDR-SMood challenge:
- train a point regressor (`mu` only),
- evaluate with event-level CRPS,
- tune `sigma_scale`,
- prepare a Codabench submission package.

## Folder Structure

- `training/train_attn_point_v2.py`: main training script
- `training/model_attn_v2.py`: V2 attention head model definition
- `training/evaluation2_crps.py`: CRPS evaluation script
- `training/tune_sigma_attn_point_v2.py`: sigma tuning script
- `training/utils.py`: shared utilities
- `submissions/model.py`: submission-time model
- `submissions/requirements.txt`: submission dependencies

## Environment

Use a Python environment with PyTorch and OpenCLIP installed.
If dataset access requires authentication, provide `--hf_token <TOKEN>`.

## Training

Run from the `training` folder:

```bash
cd training
python train_attn_point_v2.py \
  --use_augment \
  --aug_repeats 2 \
  --balance_mode event \
  --balance_power 1.0 \
  --epochs 500 \
  --batch_size 64 \
  --lr 1e-4 \
  --num_workers 4 \
  --weight_decay 1e-2 \
  --grad_clip_norm 1.0 \
  --patience 0
```

Expected output weight:
- `training/model_attn_point_v2.pth`

## Evaluation (CRPS)

```bash
cd training
python evaluation2_crps.py --model attn_point_v2 --weights model_attn_point_v2.pth
```

## Sigma Tuning

```bash
cd training
python tune_sigma_attn_point_v2.py --weights model_attn_point_v2.pth
```

This produces a JSON report with the best `sigma_scale`.

## Submission Packaging

1. Copy `training/model_attn_point_v2.pth` into `submissions/`.
2. Ensure `submissions/model.py` and `submissions/requirements.txt` are present.
3. Zip from inside `submissions/`:

```bash
cd submissions
zip -r submit_attn_point_v2.zip model.py requirements.txt model_attn_point_v2.pth
```

## Notes

- `submissions/model.py` reads `SIGMA_SCALE` from environment (default `0.78`).
- Keep training and submission model definitions aligned to avoid `state_dict` mismatch.
# Beetle V2 Baseline Package

This folder contains a compact V2 baseline workflow for the HDR-SMood challenge:
- train a point regressor (`mu` only),
- evaluate with event-level CRPS,
- tune `sigma_scale`,
- prepare a Codabench submission package.

## Folder Structure

- `training/train_attn_point_v2.py`: main training script
- `training/model_attn_v2.py`: V2 attention head model definition
- `training/evaluation2_crps.py`: CRPS evaluation script
- `training/tune_sigma_attn_point_v2.py`: sigma tuning script
- `training/utils.py`: shared utilities
- `submissions/model.py`: submission-time model
- `submissions/requirements.txt`: submission dependencies

## Environment

Use a Python environment with PyTorch and OpenCLIP installed.
If dataset access requires authentication, provide `--hf_token <TOKEN>`.

## Training

Run from the `training` folder:

```bash
cd training
python train_attn_point_v2.py \
  --use_augment \
  --aug_repeats 2 \
  --balance_mode event \
  --balance_power 1.0 \
  --epochs 500 \
  --batch_size 64 \
  --lr 1e-4 \
  --num_workers 4 \
  --weight_decay 1e-2 \
  --grad_clip_norm 1.0 \
  --patience 0
```

Expected output weight:
- `training/model_attn_point_v2.pth`

## Evaluation (CRPS)

```bash
cd training
python evaluation2_crps.py --model attn_point_v2 --weights model_attn_point_v2.pth
```

## Sigma Tuning

```bash
cd training
python tune_sigma_attn_point_v2.py --weights model_attn_point_v2.pth
```

This produces a JSON report with the best `sigma_scale`.

## Submission Packaging

1. Copy `training/model_attn_point_v2.pth` into `submissions/`.
2. Ensure `submissions/model.py` and `submissions/requirements.txt` are present.
3. Zip from inside `submissions/`:

```bash
cd submissions
zip -r submit_attn_point_v2.zip model.py requirements.txt model_attn_point_v2.pth
```

## Notes

- `submissions/model.py` reads `SIGMA_SCALE` from environment (default `0.78`).
- Keep training and submission model definitions aligned to avoid `state_dict` mismatch.
