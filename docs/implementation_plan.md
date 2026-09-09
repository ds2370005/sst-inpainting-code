# implementation_plan.md

# Implementation Plan for PyTorch SST Anomaly Inpainting

This document defines the implementation order. The goal is to produce a runnable, testable PyTorch project before connecting it to real Himawari or data-assimilation datasets.

---

## 0. General rule

Do not implement the full training pipeline at once.

Implement in this order:

```text
1. Configuration
2. Small utility functions
3. Network blocks
4. Generators
5. Discriminator
6. Losses
7. Synthetic tensor tests
8. Dataset interfaces
9. One-batch training tests
10. Full training scripts
11. Inference script
12. Real data integration
```

Every module must run with synthetic tensors before real data are used.

---

## 1. Project initialization

Create the following structure:

```text
sst_anomaly_inpainting/
├── README.md
├── requirements.txt
├── config.yaml
├── docs/
│   ├── architecture.md
│   ├── implementation_plan.md
│   └── paper_notes.md
├── src/
│   ├── datasets/
│   ├── models/
│   ├── losses/
│   ├── train/
│   ├── infer/
│   └── utils/
├── tests/
├── checkpoints/
└── outputs/
```

Recommended `requirements.txt`:

```text
torch
torchvision
numpy
scipy
pandas
xarray
netCDF4
rasterio
pyyaml
tqdm
matplotlib
pytest
```

Do not require rasterio, xarray, or netCDF4 for synthetic tests. Keep model tests independent of geospatial libraries.

---

## 2. Configuration

Create `config.yaml`.

Required fields:

```yaml
seed: 42

data:
  patch_size: 256
  time_steps: 9
  time_interval_hours: 6
  min_temp: 0.0
  max_temp: 35.0
  anomaly_range: 5.0

training:
  epochs: 30
  batch_size: 32
  num_workers: 4
  device: cuda
  mixed_precision: false

optimizer:
  lr_generator: 1.0e-4
  lr_discriminator: 1.0e-8
  beta1: 0.5
  beta2: 0.99

loss:
  lambda_ave_rec: 10.0
  lambda_ave_adv: 0.1
  lambda_ano_rec: 10.0
  lambda_ano_adv: 0.1

paths:
  checkpoint_dir: checkpoints
  output_dir: outputs
```

Create `src/utils/config.py`:

```python
load_config(path: str) -> dict
```

---

## 3. Utilities

Create `src/utils/normalization.py`.

Functions:

```python
normalize_sst(x, min_temp, max_temp)
denormalize_sst(x_norm, min_temp, max_temp)
normalize_anomaly(anomaly, anomaly_range)
denormalize_anomaly(anomaly_norm, anomaly_range)
```

Create `src/utils/metrics.py`.

Functions:

```python
masked_mse_value(pred, target, mask)
masked_rmse(pred, target, mask)
masked_mae(pred, target, mask)
masked_bias(pred, target, mask)
```

Create `src/utils/seed.py`.

Function:

```python
set_seed(seed: int) -> None
```

---

## 4. Model blocks

Create `src/models/blocks.py`.

Implement:

```python
class ReflectionConv2d(nn.Module):
    # ReflectionPad2d + Conv2d

class ConvELU(nn.Module):
    # ReflectionConv2d + ELU

class DilatedConvELU(nn.Module):
    # ReflectionConv2d with dilation + ELU

class UpsampleConvELU(nn.Module):
    # Bilinear upsampling + ConvELU
```

Requirements:

- Use reflection padding for all 2D convolution layers.
- Use ELU activation unless explicitly disabled.
- Keep tensor shape with stride 1.
- Downsample only when stride is 2.

---

## 5. Shared encoder-decoder

Create `src/models/generator.py`.

First implement:

```python
class EncoderDecoder2D(nn.Module):
    def __init__(self, in_channels: int):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

Input:

```text
[B, T, H, W]
```

Output:

```text
[B, 1, H, W]
```

Architecture must follow `architecture.md`.

Add shape assertions.

Test with:

```python
x = torch.randn(2, 9, 256, 256)
y = model(x)
assert y.shape == (2, 1, 256, 256)
```

---

## 6. Average Estimation Generator

In `src/models/generator.py`, implement:

```python
class AverageEstimationGenerator(nn.Module):
    def __init__(self, time_steps: int = 9):
        ...

    def forward(
        self,
        sst_volume: torch.Tensor,
        mask_volume: torch.Tensor,
        monthly_average: torch.Tensor,
        monthly_mask: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

Input shapes:

```text
sst_volume      [B, 1, T, H, W]
mask_volume     [B, 1, T, H, W]
monthly_average [B, 1, 1, H, W]
monthly_mask    [B, 1, 1, H, W]
```

Output:

```text
pred_weekly     [B, 1, H, W]
```

Implementation steps:

```text
1. Concatenate SST volume and mask volume.
2. Apply 1x1x1 Conv3d.
3. Concatenate monthly average and monthly mask.
4. Apply 1x1x1 Conv3d.
5. Repeat monthly feature along temporal dimension.
6. Concatenate temporal feature and monthly feature.
7. Apply another 1x1x1 Conv3d to reduce to one 3D feature stream.
8. Squeeze channel dimension.
9. Feed into EncoderDecoder2D.
10. Return tanh output.
```

Add a synthetic test.

---

## 7. Anomaly Inpainting Generator

In `src/models/generator.py`, implement:

```python
class AnomalyInpaintingGenerator(nn.Module):
    def __init__(self, time_steps: int = 9):
        ...

    def forward(
        self,
        sst_volume: torch.Tensor,
        mask_volume: torch.Tensor,
        weekly_average: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

Input shapes:

```text
sst_volume      [B, 1, T, H, W]
mask_volume     [B, 1, T, H, W]
weekly_average  [B, 1, 1, H, W]
```

Output:

```text
pred_anomaly    [B, 1, H, W]
```

Implementation steps:

```text
1. Repeat weekly average along temporal dimension.
2. Compute anomaly_volume = (sst_volume - weekly_average) * mask_volume.
3. Concatenate anomaly_volume and mask_volume.
4. Apply 1x1x1 Conv3d.
5. Squeeze channel dimension.
6. Feed into EncoderDecoder2D.
7. Return tanh output.
```

Add a synthetic test.

---

## 8. Discriminator

Create `src/models/discriminator.py`.

Implement:

```python
class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 1):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

Input:

```text
[B, 1, H, W]
```

Output:

```text
[B, 1]
```

Use:

```python
nn.utils.spectral_norm(nn.Conv2d(...))
```

Use LeakyReLU after all intermediate layers.

Do not apply sigmoid in `forward`; use logits.

Add a synthetic test:

```python
x = torch.randn(2, 1, 256, 256)
logits = D(x)
assert logits.shape == (2, 1)
```

---

## 9. Losses

Create `src/losses/reconstruction_loss.py`.

Implement:

```python
def masked_mse_loss(pred, target, mask, eps=1e-8):
    ...
```

Create `src/losses/adversarial_loss.py`.

Implement:

```python
def discriminator_loss(real_logits, fake_logits):
    ...

def generator_adversarial_loss(fake_logits):
    ...
```

Use `torch.nn.functional.binary_cross_entropy_with_logits`.

---

## 10. Synthetic integration test

Create `tests/test_synthetic_forward.py`.

Test the complete pipeline:

```text
1. Create random SST volume, masks, monthly average.
2. Run AverageEstimationGenerator.
3. Run AnomalyInpaintingGenerator.
4. Add predicted weekly average and predicted anomaly.
5. Run Discriminator.
6. Compute all losses.
7. Call backward once.
```

This test must pass before any dataset implementation.

---

## 11. Dataset interfaces

Create `src/datasets/average_dataset.py`.

Implement class skeleton first:

```python
class AverageDataset(torch.utils.data.Dataset):
    def __init__(self, index_file: str, config: dict):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        return {
            "sst_volume": ...,          # [1, T, H, W]
            "mask_volume": ...,         # [1, T, H, W]
            "monthly_average": ...,     # [1, 1, H, W]
            "monthly_mask": ...,        # [1, 1, H, W]
            "weekly_average": ...,      # [1, H, W]
            "weekly_mask": ...,         # [1, H, W]
            "assimilation_weekly": ..., # [1, H, W]
        }
```

Create `src/datasets/anomaly_dataset.py`.

```python
class AnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, index_file: str, config: dict):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        return {
            "sst_volume": ...,          # [1, T, H, W]
            "mask_volume": ...,         # [1, T, H, W]
            "target_sst": ...,          # [1, H, W]
            "target_mask": ...,         # [1, H, W]
            "weekly_average": ...,      # [1, H, W]
            "assimilation_sst": ...,    # [1, H, W]
            "random_cloud_mask": ...,   # [1, H, W]
        }
```

At first, implement a `SyntheticAverageDataset` and `SyntheticAnomalyDataset` for testing training loops without real files.

---

## 12. Training script for Average Estimation

Create `src/train/train_average.py`.

Required CLI:

```bash
python -m src.train.train_average --config config.yaml --synthetic
```

Training loop:

```text
for epoch in epochs:
    for batch in loader:
        1. Move tensors to device.
        2. pred_weekly = G_average(...)
        3. Update D_average:
             real = assimilation_weekly
             fake = pred_weekly.detach()
        4. Update G_average:
             loss_reclong
             loss_recshort
             loss_adv
             total loss
        5. Log losses.
    save checkpoint.
```

Checkpoint content:

```python
{
    "epoch": epoch,
    "generator_state_dict": G.state_dict(),
    "discriminator_state_dict": D.state_dict(),
    "optimizer_g_state_dict": opt_g.state_dict(),
    "optimizer_d_state_dict": opt_d.state_dict(),
    "config": config,
}
```

---

## 13. Training script for Anomaly Inpainting

Create `src/train/train_anomaly.py`.

Required CLI:

```bash
python -m src.train.train_anomaly --config config.yaml --average-checkpoint checkpoints/average/best.pth --synthetic
```

Training loop:

```text
for epoch in epochs:
    for batch in loader:
        1. Move tensors to device.
        2. Obtain weekly average:
             option A: use batch["weekly_average"]
             option B: use frozen G_average prediction
        3. pred_anomaly = G_anomaly(...)
        4. pred_sst = weekly_average + pred_anomaly
        5. Update D_anomaly:
             real = assimilation_sst
             fake = pred_sst.detach()
        6. Update G_anomaly:
             masked reconstruction loss
             adversarial physical model loss
        7. Log losses.
    save checkpoint.
```

Initial implementation should use option A. Option B can be added after the script is stable.

---

## 14. Inference script

Create `src/infer/reconstruct.py`.

Required CLI:

```bash
python -m src.infer.reconstruct \
  --config config.yaml \
  --average-checkpoint checkpoints/average/best.pth \
  --anomaly-checkpoint checkpoints/anomaly/best.pth \
  --input sample_input.npz \
  --output outputs/reconstructed/sample_output.npz
```

Input `.npz` should contain:

```text
sst_volume
mask_volume
monthly_average
monthly_mask
```

Output `.npz` should contain:

```text
pred_weekly
pred_anomaly
pred_sst
```

---

## 15. Real data integration

Do this only after synthetic training works.

Required preprocessing outputs:

```text
sst_volume.npy          [N, 1, T, H, W]
mask_volume.npy         [N, 1, T, H, W]
monthly_average.npy     [N, 1, 1, H, W]
monthly_mask.npy        [N, 1, 1, H, W]
weekly_average.npy      [N, 1, H, W]
weekly_mask.npy         [N, 1, H, W]
assimilation_sst.npy    [N, 1, H, W]
```

For the first real-data version, avoid complicated geospatial IO inside the Dataset. Preprocess all arrays into `.npy` or `.npz` files first.

---

## 16. Completion criteria

The implementation is acceptable when all of the following pass:

```text
1. pytest tests/test_synthetic_forward.py passes.
2. Average generator produces [B, 1, 256, 256].
3. Anomaly generator produces [B, 1, 256, 256].
4. Discriminator produces [B, 1].
5. One training step updates generator and discriminator without NaN.
6. train_average.py runs for 2 synthetic epochs.
7. train_anomaly.py runs for 2 synthetic epochs.
8. reconstruct.py runs on synthetic .npz input.
```

---

## 17. Codex working instruction

When using Codex, give it narrow tasks:

```text
Read docs/architecture.md and implement only src/models/blocks.py and src/models/generator.py. Add synthetic tests. Do not implement datasets yet.
```

Then:

```text
Read docs/architecture.md and implement src/models/discriminator.py. Add tests for discriminator shape.
```

Then:

```text
Implement losses and one synthetic backward test.
```

Then:

```text
Implement train_average.py using SyntheticAverageDataset only.
```

This staged approach is safer than asking Codex to create the entire project in one prompt.
