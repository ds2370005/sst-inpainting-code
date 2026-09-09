# paper_notes.md

# Notes Mapping the Paper to the PyTorch Implementation

Target paper: Hirahara, Sonogashira, and Iiyama, "Cloud-Free Sea-Surface-Temperature Image Reconstruction From Anomaly Inpainting Network," IEEE Transactions on Geoscience and Remote Sensing, 2022.

This document records how the paper's method is translated into implementation choices.

---

## 1. Core idea

The method reconstructs cloud-free SST images by combining two ideas:

```text
1. Estimate a cloud-free short-term average SST image.
2. Estimate the anomaly between the target SST and the short-term average SST.
```

Implementation mapping:

```text
Average-estimation network  -> AverageEstimationGenerator
Anomaly inpainting network  -> AnomalyInpaintingGenerator
Physical model discriminator -> Discriminator trained with data-assimilation SST as real
```

---

## 2. Two-network design

Paper description:

```text
The model first synthesizes short-term average SST images and then reconstructs target SST images with an anomaly inpainting network.
```

Implementation:

```python
pred_weekly = G_average(sst_volume, mask_volume, monthly_average, monthly_mask)
pred_anomaly = G_anomaly(sst_volume, mask_volume, pred_weekly.unsqueeze(2))
pred_sst = pred_weekly + pred_anomaly
```

Reasoning:

Direct SST reconstruction requires predicting the full SST range. Anomaly reconstruction only predicts a narrow residual around the recent average field.

---

## 3. Input SST volume

Paper description:

```text
The method uses a spatiotemporal SST volume composed of SST images taken every h hours.
```

Paper setting:

```text
T = 9
h = 6 hours
```

Implementation:

```python
sst_volume.shape = [B, 1, 9, H, W]
mask_volume.shape = [B, 1, 9, H, W]
```

The first temporal element should correspond to the current or target time `t1` unless the preprocessing pipeline documents a different order.

Recommended convention:

```text
index 0 = t1
index 1 = t1 - 6h
index 2 = t1 - 12h
...
index 8 = t1 - 48h
```

---

## 4. Two-channel SST image

Paper description:

```text
Each SST image is a two-channel image: SST and cloud mask.
```

Implementation:

Store SST and mask separately in the Dataset, then concatenate inside the generator.

```python
x = torch.cat([sst_volume, mask_volume], dim=1)
# [B, 2, T, H, W]
```

Rationale:

Keeping them separate avoids ambiguity in loss calculation and makes it easier to apply masked losses.

---

## 5. Average Estimation Network

Paper description:

```text
The average-estimation network takes a spatiotemporal SST volume and a long-term average SST image as input. It outputs a cloud-free short-term average SST image.
```

Implementation mapping:

```text
long-term average  -> monthly_average
short-term average -> weekly_average
```

Forward pass:

```python
pred_weekly = G_average(
    sst_volume,
    mask_volume,
    monthly_average,
    monthly_mask,
)
```

Output:

```text
pred_weekly: [B, 1, H, W]
```

Training target:

```text
weekly_average, masked by weekly_mask
```

Additional weak target:

```text
monthly_average, masked by monthly_mask
```

The paper uses a long-term reconstruction loss weighted by `(1 - k/kmax)`, which gradually reduces the influence of long-term average guidance.

---

## 6. Anomaly Inpainting Network

Paper description:

```text
The anomaly is the difference between the target SSTs and the short-term average SSTs.
```

Implementation:

```python
weekly_rep = weekly_average.repeat(1, 1, T, 1, 1)
anomaly_volume = (sst_volume - weekly_rep) * mask_volume
pred_anomaly = G_anomaly(anomaly_volume, mask_volume)
pred_sst = weekly_average.squeeze(2) + pred_anomaly
```

Training target:

The reconstruction loss is applied to the final SST, not only to the anomaly:

```python
loss_rec = masked_mse(pred_sst, target_sst, target_mask)
```

Rationale:

The final product is the reconstructed SST image. Computing loss on SST keeps the target physically interpretable.

---

## 7. Anomaly range

Paper setting:

```text
The anomaly range is assumed to be within +/- 5.0 K.
```

Implementation:

```python
anomaly_norm = anomaly / 5.0
anomaly_norm = anomaly_norm.clamp(-1, 1)
```

Generator output:

```python
pred_anomaly_norm = tanh(raw_output)
pred_anomaly = pred_anomaly_norm * 5.0
```

Important note:

If the SST itself is normalized to `[-1, 1]`, do not mix normalized SST and Kelvin anomaly without a clear conversion. Choose one of the following:

```text
Option A: train the anomaly network entirely in normalized anomaly units.
Option B: train it in physical Kelvin units and only normalize for neural network input/output.
```

Recommended initial implementation:

Use normalized SST for all network inputs and outputs, but keep explicit conversion functions for SST and anomaly.

---

## 8. Generator architecture

Paper description:

```text
The generator consists of a 3D convolution layer and an encoder-decoder network.
```

Implementation mapping:

```text
3D Conv 1x1x1 -> maps SST/mask pair into one temporal feature stream
2D Encoder    -> treats T as channel dimension
2D Decoder    -> reconstructs one output image
```

Why T becomes channels:

After the 3D projection:

```python
x.shape = [B, 1, T, H, W]
x = x.squeeze(1)
# [B, T, H, W]
```

The encoder-decoder is then a 2D CNN with `in_channels=T`.

This is simpler and consistent with the paper's statement that the `T x V x H` tensor is fed into the encoder-decoder.

---

## 9. Encoder architecture

Paper table mapping:

```text
Conv 5x5, stride 1, out 32
Conv 3x3, stride 2, out 64
Conv 3x3, stride 1, out 64
Conv 3x3, stride 2, out 128
Conv 3x3, stride 1, out 128
Conv 3x3, stride 2, out 256
Dilated Conv 3x3, dilation 2, out 256
Dilated Conv 3x3, dilation 4, out 256
Dilated Conv 3x3, dilation 8, out 256
Dilated Conv 3x3, dilation 16, out 256
```

Implementation note:

Use reflection padding, not zero padding.

Use ELU, not ReLU.

---

## 10. Decoder architecture

Paper table mapping:

```text
Conv 3x3, out 128
Upsample x2
Conv 3x3, out 64
Upsample x2
Conv 3x3, out 32
Upsample x2
Conv 3x3, out 16
Conv 3x3, out 1
```

Implementation note:

Use bilinear upsampling.

The final layer should output values clipped to `[-1, 1]`. In PyTorch, use:

```python
torch.tanh(out)
```

This is differentiable and more convenient than hard clipping.

---

## 11. Contextual attention omitted

Paper description:

```text
The authors omit the contextual attention module because SST images do not have clear texture like face or natural images.
```

Implementation:

Do not implement contextual attention in the baseline.

Possible future extension:

Add mask-aware attention only after the baseline reproduces the expected behavior.

---

## 12. Discriminator and adversarial physical model loss

Paper description:

```text
The discriminator recognizes data-assimilation SST images as real samples.
```

Implementation:

```python
real = assimilation_sst
fake = pred_sst.detach()

real_logits = D(real)
fake_logits = D(fake)
loss_d = BCEWithLogits(real_logits, 1) + BCEWithLogits(fake_logits, 0)
```

Generator adversarial loss:

```python
fake_logits_for_g = D(pred_sst)
loss_adv_g = BCEWithLogits(fake_logits_for_g, 1)
```

Important distinction:

The data-assimilation image is not necessarily paired with the exact satellite input. It is used to impose physical plausibility and denoising through the adversarial distribution.

---

## 13. Average generator loss

Paper formula:

```text
L_total_ave = lambda_rec_ave * ((1 - k/kmax) * L_reclong_ave + L_recshort_ave) + lambda_adv_ave * L_adv_ave
```

Implementation:

```python
loss_reclong = masked_mse(pred_weekly, monthly_average, monthly_mask)
loss_recshort = masked_mse(pred_weekly, weekly_average, weekly_mask)
epoch_weight = 1.0 - epoch / max_epochs
loss_rec = epoch_weight * loss_reclong + loss_recshort
loss_g = lambda_ave_rec * loss_rec + lambda_ave_adv * loss_adv
```

Interpretation:

The monthly average acts as coarse guidance early in training. As training proceeds, the model should rely more on the weekly target.

---

## 14. Anomaly generator loss

Paper formula:

```text
L_total_ano = lambda_rec_ano * L_rec_ano + lambda_adv_ano * L_adv_ano
```

Implementation:

```python
pred_sst = weekly_average + pred_anomaly
loss_rec = masked_mse(pred_sst, target_sst, target_mask)
loss_g = lambda_ano_rec * loss_rec + lambda_ano_adv * loss_adv
```

The reconstruction loss is masked because ground truth SST is unavailable under cloud cover.

---

## 15. Random cloud masks

Paper description:

```text
The random mask is chosen from cloud-mask patterns in the satellite training dataset.
```

Implementation:

Create a cloud mask pool:

```text
cloud_masks.npy: [M, 1, H, W]
```

During training:

```python
mrand = random choice from cloud mask pool
masked_target = target_sst * mrand
```

Initial implementation:

This can be omitted in the first synthetic version. Add it when real Dataset classes are implemented.

---

## 16. Training settings from paper

Use these as defaults:

```text
epochs = 30
batch_size = 32
optimizer = Adam
beta1 = 0.5
beta2 = 0.99
lr_generator = 1e-4
lr_discriminator = 1e-8
lambda_ano_rec = 10.0
lambda_ano_adv = 0.1
lambda_ave_rec = 10.0
lambda_ave_adv = 0.1
```

Implementation note:

The discriminator learning rate is extremely small. Keep it initially for paper fidelity, but if adversarial learning becomes ineffective, test larger values such as:

```text
1e-6
1e-5
```

Record such deviations in experiment logs.

---

## 17. Training data mapping

Paper data:

```text
Satellite SST images: Himawari-8
Training period: February 2018 to June 2019
Patch size: 256 x 256
Cloud-cover exclusion threshold: above 50% excluded
Data assimilation images: used as real samples for adversarial physical model loss
```

Implementation mapping:

For the first real implementation, prepare preprocessed arrays:

```text
sst_volume.npy
mask_volume.npy
monthly_average.npy
monthly_mask.npy
weekly_average.npy
weekly_mask.npy
assimilation_sst.npy
```

Avoid mixing geospatial preprocessing with model training code.

---

## 18. Inference mapping

Paper method:

```text
First predict weekly average SST, then reconstruct SST using anomaly inpainting.
```

Implementation:

```python
with torch.no_grad():
    pred_weekly = G_average(sst_volume, mask_volume, monthly_average, monthly_mask)
    pred_anomaly = G_anomaly(sst_volume, mask_volume, pred_weekly.unsqueeze(2))
    pred_sst = pred_weekly + pred_anomaly
```

For large images:

The paper discusses patch-based reconstruction for some baselines and weekly-average reconstruction. For the initial implementation, use patch-wise inference only when memory prevents full-frame inference.

Recommended initial strategy:

```text
1. Train and test on 256 x 256 patches.
2. Add tiled inference with overlap.
3. Add full-frame inference only if GPU memory allows.
```

---

## 19. What is paper-faithful and what is implementation choice

Paper-faithful choices:

```text
Two-stage architecture
Average-estimation network
Anomaly inpainting network
3D Conv before encoder-decoder
Encoder-decoder architecture
Dilated convolutions
ELU activations
Reflection padding
Spectral normalization in discriminator
Adversarial physical model loss
Masked reconstruction loss
T = 9
patch size = 256
anomaly range = +/- 5 K
```

Implementation choices:

```text
Use BCEWithLogitsLoss instead of sigmoid + BCELoss
Use tanh instead of hard clipping
Repeat monthly average across T before branch fusion
Use synthetic datasets before real data integration
Use npy/npz preprocessed arrays before GeoTIFF/NetCDF integration
```

These choices should be documented in README or experiment logs.

---

## 20. Possible extension notes

After reproducing the baseline, possible research extensions include:

```text
1. Add RAFT-derived optical flow to align past SST images to t1.
2. Replace 2D encoder with ConvLSTM or temporal attention.
3. Add mask-aware attention in the anomaly network.
4. Add external variables: wind, current, chlorophyll, SSH.
5. Make anomaly range seasonal or regional rather than fixed +/- 5 K.
6. Use data-assimilation fields as paired soft targets, not only adversarial real samples.
```

For RAFT integration, the cleanest initial design is:

```text
past SST frames -> RAFT or optical-flow module -> warp to t1 -> anomaly network input
```

Do not add this before the baseline is working.
