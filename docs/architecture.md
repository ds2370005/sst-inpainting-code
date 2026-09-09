# architecture.md

# Network Architecture and Tensor Shapes

Target method: Hirahara et al., "Cloud-Free Sea-Surface-Temperature Image Reconstruction From Anomaly Inpainting Network".

This document specifies the PyTorch architecture, tensor conventions, and shape transitions for implementing the method.

---

## 1. Global tensor convention

Use PyTorch `NCHW` and `NCTHW` conventions.

For temporal SST volumes:

```text
sst_volume   : [B, 1, T, H, W]
mask_volume  : [B, 1, T, H, W]
```

For single SST images:

```text
sst_image    : [B, 1, H, W]
mask_image   : [B, 1, H, W]
```

For single images passed through the 3D branch, add a temporal dimension:

```text
monthly_average : [B, 1, 1, H, W]
monthly_mask    : [B, 1, 1, H, W]
weekly_average  : [B, 1, 1, H, W]
```

Mask convention:

```text
1 = observed / valid ocean pixel
0 = cloud / missing / land / invalid pixel
```

Default values:

```text
T = 9
H = W = 256
```

---

## 2. Overall two-stage architecture

The complete model has two generators and two discriminators.

```text
Stage 1: Average Estimation Network

SST volume + mask volume + monthly average + monthly mask
        |
        v
G_average
        |
        v
predicted weekly average SST


Stage 2: Anomaly Inpainting Network

SST volume + mask volume + predicted weekly average SST
        |
        v
anomaly volume = SST volume - predicted weekly average
        |
        v
G_anomaly
        |
        v
predicted anomaly
        |
        v
final SST = predicted weekly average + predicted anomaly
```

Training uses a discriminator for each stage:

```text
D_average : distinguishes predicted weekly average from data-assimilation weekly average
D_anomaly : distinguishes final reconstructed SST from data-assimilation SST
```

---

## 3. Average Estimation Generator

### 3.1 Purpose

Estimate a cloud-free short-term average SST image, corresponding to the weekly average in the paper.

### 3.2 Inputs

```text
sst_volume       : [B, 1, T, H, W]
mask_volume      : [B, 1, T, H, W]
monthly_average  : [B, 1, 1, H, W]
monthly_mask     : [B, 1, 1, H, W]
```

Concatenate SST and mask within each branch:

```text
sst_branch_input     = concat(sst_volume, mask_volume, dim=1)
                     = [B, 2, T, H, W]

monthly_branch_input = concat(monthly_average, monthly_mask, dim=1)
                     = [B, 2, 1, H, W]
```

### 3.3 Branch projection

Each branch first uses a 3D convolution:

```text
Conv3d(kernel_size=1x1x1, stride=1x1x1)
```

Recommended implementation:

```python
sst_proj = Conv3d(2, 1, kernel_size=1)(sst_branch_input)
# [B, 1, T, H, W]

monthly_proj = Conv3d(2, 1, kernel_size=1)(monthly_branch_input)
# [B, 1, 1, H, W]
```

To concatenate monthly information with temporal SST features, repeat monthly projection along the temporal axis:

```python
monthly_proj = monthly_proj.repeat(1, 1, T, 1, 1)
# [B, 1, T, H, W]
```

Then concatenate the projected branches:

```python
x = torch.cat([sst_proj, monthly_proj], dim=1)
# [B, 2, T, H, W]
```

Map the concatenated branch output into a single temporal feature tensor:

```python
x = Conv3d(2, 1, kernel_size=1)(x)
# [B, 1, T, H, W]
```

Remove the singleton channel dimension and treat time as the channel dimension for the 2D encoder-decoder:

```python
x = x.squeeze(1)
# [B, T, H, W]
```

### 3.4 Encoder

Input:

```text
[B, T, 256, 256]
```

Use reflection padding before every convolution. Use ELU after every layer except the final output layer.

```text
Layer  Type                  Kernel  Stride  Dilation  Out channels  Output shape
E1     Conv2d                5x5     1       1         32            [B, 32, 256, 256]
E2     Conv2d                3x3     2       1         64            [B, 64, 128, 128]
E3     Conv2d                3x3     1       1         64            [B, 64, 128, 128]
E4     Conv2d                3x3     2       1         128           [B, 128, 64, 64]
E5     Conv2d                3x3     1       1         128           [B, 128, 64, 64]
E6     Conv2d                3x3     2       1         256           [B, 256, 32, 32]
E7     Dilated Conv2d        3x3     1       2         256           [B, 256, 32, 32]
E8     Dilated Conv2d        3x3     1       4         256           [B, 256, 32, 32]
E9     Dilated Conv2d        3x3     1       8         256           [B, 256, 32, 32]
E10    Dilated Conv2d        3x3     1       16        256           [B, 256, 32, 32]
```

### 3.5 Decoder

```text
Layer  Type                         Kernel  Stride  Out channels  Output shape
D1     Conv2d                       3x3     1       128           [B, 128, 32, 32]
D2     Bilinear Upsampling x2       -       -       -             [B, 128, 64, 64]
D3     Conv2d                       3x3     1       64            [B, 64, 64, 64]
D4     Bilinear Upsampling x2       -       -       -             [B, 64, 128, 128]
D5     Conv2d                       3x3     1       32            [B, 32, 128, 128]
D6     Bilinear Upsampling x2       -       -       -             [B, 32, 256, 256]
D7     Conv2d                       3x3     1       16            [B, 16, 256, 256]
D8     Conv2d                       3x3     1       1             [B, 1, 256, 256]
```

Final activation:

```python
pred_weekly = torch.tanh(out)
# [B, 1, H, W]
```

For consistency with stage 2, it can also be returned as:

```python
pred_weekly_5d = pred_weekly.unsqueeze(2)
# [B, 1, 1, H, W]
```

---

## 4. Anomaly Inpainting Generator

### 4.1 Purpose

Estimate the SST anomaly between the current target SST and the short-term average SST.

### 4.2 Inputs

```text
sst_volume          : [B, 1, T, H, W]
mask_volume         : [B, 1, T, H, W]
pred_weekly_average : [B, 1, 1, H, W]
```

Broadcast the weekly average over the temporal axis:

```python
weekly_rep = pred_weekly_average.repeat(1, 1, T, 1, 1)
# [B, 1, T, H, W]
```

Compute anomaly volume:

```python
anomaly_volume = (sst_volume - weekly_rep) * mask_volume
# [B, 1, T, H, W]
```

Concatenate anomaly and mask:

```python
x = torch.cat([anomaly_volume, mask_volume], dim=1)
# [B, 2, T, H, W]
```

### 4.3 Projection and encoder-decoder

Use the same projection and encoder-decoder structure as the Average Estimation Generator, but with a single input branch:

```python
x = Conv3d(2, 1, kernel_size=1)(x)
# [B, 1, T, H, W]

x = x.squeeze(1)
# [B, T, H, W]

pred_anomaly = encoder_decoder(x)
# [B, 1, H, W]
```

Final activation:

```python
pred_anomaly_norm = torch.tanh(out)
```

If anomaly is normalized by `anomaly_range = 5.0`, then physical anomaly is:

```python
pred_anomaly_kelvin = pred_anomaly_norm * anomaly_range
```

Final SST:

```python
pred_sst = pred_weekly_average.squeeze(2) + pred_anomaly
# [B, 1, H, W]
```

---

## 5. Discriminator

### 5.1 Purpose

The discriminator implements the adversarial physical model loss. It classifies data-assimilation SST images as real and generator outputs as fake.

### 5.2 Input

```text
sst_patch : [B, 1, H, W]
```

### 5.3 Architecture

Use 2D convolution layers with spectral normalization. Use LeakyReLU after every convolution except the last output layer.

Recommended architecture:

```text
Layer  Type                         Kernel  Stride  Out channels  Output shape if H=W=256
D1     SpectralNorm Conv2d          5x5     2       64            [B, 64, 128, 128]
D2     SpectralNorm Conv2d          5x5     2       128           [B, 128, 64, 64]
D3     SpectralNorm Conv2d          5x5     2       256           [B, 256, 32, 32]
D4     SpectralNorm Conv2d          5x5     2       256           [B, 256, 16, 16]
D5     SpectralNorm Conv2d          5x5     2       256           [B, 256, 8, 8]
D6     SpectralNorm Conv2d          5x5     1       1             [B, 1, 8, 8]
Pool   Global average pooling       -       -       -             [B, 1]
Out    Sigmoid or logits            -       -       -             [B, 1]
```

Implementation recommendation:

Use `BCEWithLogitsLoss` and do not apply sigmoid inside the discriminator. This is numerically more stable than applying sigmoid followed by `BCELoss`.

```python
logits = D(x)
loss = BCEWithLogitsLoss(logits, labels)
```

---

## 6. Loss tensor shapes

### 6.1 Masked MSE

Inputs:

```text
pred   : [B, 1, H, W]
target : [B, 1, H, W]
mask   : [B, 1, H, W]
```

Formula:

```python
loss = ((pred - target) ** 2 * mask).sum() / (mask.sum() + eps)
```

### 6.2 Average generator loss

```text
pred_weekly      : [B, 1, H, W]
monthly_average  : [B, 1, H, W]
monthly_mask     : [B, 1, H, W]
weekly_average   : [B, 1, H, W]
weekly_mask      : [B, 1, H, W]
```

```python
loss_reclong = masked_mse(pred_weekly, monthly_average, monthly_mask)
loss_recshort = masked_mse(pred_weekly, weekly_average, weekly_mask)
epoch_weight = 1.0 - current_epoch / max_epochs
loss_rec = epoch_weight * loss_reclong + loss_recshort
loss_adv = BCEWithLogitsLoss(D_average(pred_weekly), torch.ones_like(...))
loss_g = lambda_ave_rec * loss_rec + lambda_ave_adv * loss_adv
```

### 6.3 Anomaly generator loss

```text
pred_sst    : [B, 1, H, W]
target_sst  : [B, 1, H, W]
target_mask : [B, 1, H, W]
```

```python
loss_rec = masked_mse(pred_sst, target_sst, target_mask)
loss_adv = BCEWithLogitsLoss(D_anomaly(pred_sst), torch.ones_like(...))
loss_g = lambda_ano_rec * loss_rec + lambda_ano_adv * loss_adv
```

---

## 7. Shape validation checklist

Use assertions in every forward method.

```python
assert sst_volume.ndim == 5
assert mask_volume.ndim == 5
assert sst_volume.shape == mask_volume.shape
assert sst_volume.shape[1] == 1
assert sst_volume.shape[2] == T
assert sst_volume.shape[-1] == W
assert sst_volume.shape[-2] == H
```

Average generator output:

```python
assert pred_weekly.shape == (B, 1, H, W)
```

Anomaly generator output:

```python
assert pred_anomaly.shape == (B, 1, H, W)
assert pred_sst.shape == (B, 1, H, W)
```

Discriminator output:

```python
assert logits.shape == (B, 1)
```

---

## 8. Minimal synthetic forward test

Codex should implement this test before real data loading.

```python
B = 2
T = 9
H = W = 256

sst_volume = torch.randn(B, 1, T, H, W)
mask_volume = torch.ones(B, 1, T, H, W)
monthly = torch.randn(B, 1, 1, H, W)
monthly_mask = torch.ones(B, 1, 1, H, W)

G_avg = AverageEstimationGenerator(time_steps=T)
G_ano = AnomalyInpaintingGenerator(time_steps=T)
D = Discriminator()

pred_weekly = G_avg(sst_volume, mask_volume, monthly, monthly_mask)
assert pred_weekly.shape == (B, 1, H, W)

pred_weekly_5d = pred_weekly.unsqueeze(2)
pred_anomaly = G_ano(sst_volume, mask_volume, pred_weekly_5d)
assert pred_anomaly.shape == (B, 1, H, W)

pred_sst = pred_weekly + pred_anomaly
logits = D(pred_sst)
assert logits.shape == (B, 1)
```
