PyTorch実装仕様書

対象論文：Cloud-Free Sea-Surface-Temperature Image Reconstruction From Anomaly Inpainting Network

1. 実装目的

SST衛星画像に含まれる雲被覆による欠損と観測ノイズを補完し、雲なし・ノイズ低減済みのSST画像を復元する。

論文手法は二段階構成とする。

第1段階：Average Estimation Network
Monthly Average SST と過去9時刻のSST系列から、雲なし Weekly Average SST を推定する。

第2段階：Anomaly Inpainting Network
現在SSTと推定Weekly Average SSTとの差分、すなわちAnomalyを補完し、最終的なCloud-Free SSTを復元する。

original_text — “Our method consists of two networks: average-estimation network and anomaly inpainting network.”
source — uploaded PDF: Cloud-Free Sea-Surface-Temperature Image Reconstruction From Anomaly Inpainting Network, p. 3, upper-middle.  
note — 実装上も、平均場推定モデルと偏差補完モデルを完全に分離して作るのがよい。

⸻

2. ディレクトリ構成
```
sst_anomaly_inpainting/
│
├── README.md
├── requirements.txt
├── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── datasets/
│   │   ├── average_dataset.py
│   │   ├── anomaly_dataset.py
│   │   └── transforms.py
│   │
│   ├── models/
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   └── blocks.py
│   │
│   ├── losses/
│   │   ├── reconstruction_loss.py
│   │   └── adversarial_loss.py
│   │
│   ├── train/
│   │   ├── train_average.py
│   │   └── train_anomaly.py
│   │
│   ├── infer/
│   │   └── reconstruct.py
│   │
│   └── utils/
│       ├── normalization.py
│       ├── patching.py
│       ├── metrics.py
│       └── visualization.py
│
├── checkpoints/
│   ├── average/
│   └── anomaly/
│
└── outputs/
    ├── figures/
    └── reconstructed/
```
⸻

3. データ仕様

3.1 入力データ

各時刻のSST画像は2チャンネルで扱う。
```
channel 0: SST値
channel 1: cloud mask
```
maskは以下で統一する。
```
1 = observed / valid ocean pixel
0 = cloud / missing / land / invalid
```
original_text — “Each SST image is a two-channel image; each pixel in the first channel contains an SST, and the second channel is a cloud mask indicating whether clouds cover each pixel or not.”
source — uploaded PDF, p. 3, mid.  
note — 論文の記述に従い、SSTと雲マスクを明示的に別チャンネルで持たせる。

3.2 時系列入力

論文では9時刻を使う。
```
T = 9
h = 6 hours
time sequence = t1, t1-6h, t1-12h, ..., t1-48h
```
original_text — “The number of images for time direction, T, was set to 9; specifically, we use SST images every 6 h until two days before.”
source — uploaded PDF, p. 6, upper-middle.  
note — 初期実装ではT=9固定でよい。後でconfig化する。

3.3 パッチサイズ
```
patch_size = 256 × 256
```
original_text — “The image size for training, i.e., V × H in III-E, was set to 256 × 256.”
source — uploaded PDF, p. 6, upper-middle.  
note — GPUメモリに応じて128や512にも変更可能だが、まず論文準拠で256にする。

⸻

4. 正規化仕様

SST値は [-1, 1] に正規化する。
```
x_norm = 2 * (x - min_temp) / (max_temp - min_temp) - 1
x = (x_norm + 1) / 2 * (max_temp - min_temp) + min_temp
```
推奨初期値：
```
min_temp: 0.0
max_temp: 35.0
```
Anomalyは論文に従い ±5.0 K を想定する。
```
anom_norm = anomaly / 5.0
anom_norm = clip(anom_norm, -1, 1)
```
original_text — “we assumed that the range of SST anomalies from both average SSTs falls within ±5.0 K and carried out training and reconstruction.”
source — uploaded PDF, p. 6, upper-middle.  
note — Anomaly推定を安定化させる中核部分。SSTそのものではなく狭いレンジの偏差を推定する。

⸻

5. Average Estimation Network

5.1 入力
```
sst_volume:        [B, 1, T, H, W]
mask_volume:       [B, 1, T, H, W]
monthly_average:   [B, 1, 1, H, W]
monthly_mask:      [B, 1, 1, H, W]
```
結合後：
```
current_input = concat(sst_volume, mask_volume)
monthly_input = concat(monthly_average, monthly_mask)
```
5.2 出力
```
pred_weekly_average: [B, 1, 1, H, W]
```
5.3 Generator構造

論文のFig. 8に対応。
```
SST volume branch:
    3D Conv 1×1×1
Monthly average branch:
    3D Conv 1×1×1
concat
Encoder
Decoder
Output Weekly Average SST
```
original_text — “the generator for average SST image takes two types of SST images, i.e., the current SST images and the long-term average SST image, so we first apply the 3-D convolution layer to each input, then concatenate them, and input to the encoder–decoder network”
source — uploaded PDF, p. 5, lower-middle.  
note — 平均場推定モデルでは、時系列SSTと長期平均SSTを別branchで処理してから結合する。

⸻

6. Anomaly Inpainting Network

6.1 入力
```
sst_volume:          [B, 1, T, H, W]
mask_volume:         [B, 1, T, H, W]
pred_weekly_average: [B, 1, 1, H, W]
```
Anomalyを作る。
```
anomaly_volume = sst_volume - pred_weekly_average
```
ただし、欠損部分はmaskで0にする。
```
anomaly_volume = anomaly_volume * mask_volume
```
Generator入力：
```
concat(anomaly_volume, mask_volume)
shape = [B, 2, T, H, W]
```
6.2 出力
```
pred_anomaly: [B, 1, 1, H, W]
```
最終SST：
```
pred_sst = pred_weekly_average + pred_anomaly
```
original_text — “The anomaly inpainting network reconstructs a cloud-free SST image at time t1 by estimating the SST anomalies, which is the difference between the target SSTs (SSTs at time t1) and the short-term average SSTs.”
source — uploaded PDF, p. 3, upper-middle.  
note — この論文の本質。SSTを直接推定せず、平均からの差分を推定する。

⸻

7. Generator詳細

Average GeneratorとAnomaly Generatorは基本的に同じEncoder-Decoderを使う。

7.1 前段3D Conv
```
Conv3d(
    in_channels=2,
    out_channels=1,
    kernel_size=(1, 1, 1),
    stride=(1, 1, 1)
)
```
目的は、SST値とmask情報を明示的に融合すること。

original_text — “Before the encoding network, we insert a 3-D convolution layer (kernel: 1 × 1 × 1 and stride: 1 × 1 × 1) to explicitly map the binary masks indicating cloud cover to the SST…”
source — uploaded PDF, p. 5, mid.  
note — ここは通常の2D画像補完とは異なる点。maskを単なる補助情報ではなく、3D ConvでSST特徴へ写像する。

7.2 Encoder

Table I準拠。
```
Input: [B, 1, T, H, W]
Conv2d 5×5, stride 1, out 32
Conv2d 3×3, stride 2, out 64
Conv2d 3×3, stride 1, out 64
Conv2d 3×3, stride 2, out 128
Conv2d 3×3, stride 1, out 128
Conv2d 3×3, stride 2, out 256
Dilated Conv2d 3×3, dilation 2, out 256
Dilated Conv2d 3×3, dilation 4, out 256
Dilated Conv2d 3×3, dilation 8, out 256
Dilated Conv2d 3×3, dilation 16, out 256
```
実装上は、3D Conv後の [B, 1, T, H, W] を以下のように変形する。
```
x = x.squeeze(1)          # [B, T, H, W]
```
つまり、Tを2D CNNのチャンネル数として扱う。
```
Encoder(in_channels=T)
```
original_text — “We then input the T × V × H tensor into the encoder–decoder network and acquire the restored image as a 1 × T × V × H tensor.”
source — uploaded PDF, p. 5, mid.  
note — 実装では時系列次元Tを2D CNNのchannelとして扱うのが自然。

7.3 Decoder

Table II準拠。
```
Conv2d 3×3, out 128
Bilinear Upsampling ×2
Conv2d 3×3, out 64
Bilinear Upsampling ×2
Conv2d 3×3, out 32
Bilinear Upsampling ×2
Conv2d 3×3, out 16
Conv2d 3×3, out 1
clip to [-1, 1]
```
activation：
```
ELU
```
最終層のみactivationなし、出力をclipまたはtanh。

推奨実装：
```
out = torch.tanh(out)
```
original_text — “we use reflect padding for all convolution layers and employ the exponential linear unit (ELU) as an activation function instead of rectified linear unit (ReLU) except for the last layer…”
source — uploaded PDF, p. 5, lower-middle.  
note — PyTorchではReflectionPad2dを明示的に使う。

⸻

8. Discriminator仕様

8.1 入力
```
sst_patch: [B, 1, H, W]
```
real：
```
data assimilation SST
```
fake：
```
generator output
```
8.2 構造

Table III準拠の2D CNN。

実装例：
```
Conv2d 1 -> 64, kernel 5, stride 2
LeakyReLU
Conv2d 64 -> 128, kernel 5, stride 2
LeakyReLU
Conv2d 128 -> 256, kernel 5, stride 2
LeakyReLU
Conv2d 256 -> 256, kernel 5, stride 2
LeakyReLU
Conv2d 256 -> 256, kernel 5, stride 2
LeakyReLU
Conv2d 256 -> 1, kernel 5, stride 1
Global average / flatten
Sigmoid
```
全ConvにSpectral Normalizationを使う。
```
nn.utils.spectral_norm(nn.Conv2d(...))
```
original_text — “Our discriminator consists of 2-D convolution layers… every 2-D convolution layer is normalized by a spectral normalization.”
source — uploaded PDF, p. 5–6.  
note — GAN安定化のため、DiscriminatorにはSpectralNormを入れる。

⸻

9. Loss仕様

9.1 Average Estimation Loss

論文式：
```
L_total_ave =
λ_rec_ave * ((1 - k/kmax) * L_rec_long + L_rec_short)
+
λ_adv_ave * L_adv_ave
```
実装：
```
loss_reclong = mse_masked(pred_weekly, monthly_average, monthly_mask)
loss_recshort = mse_masked(pred_weekly, weekly_average, weekly_mask)
epoch_weight = 1.0 - current_epoch / max_epoch
loss_g = lambda_rec * (epoch_weight * loss_reclong + loss_recshort) \
       + lambda_adv * loss_adv
```
original_text — “Finally, we define the total loss function of the generator for average-estimation network as follows…”
source — uploaded PDF, p. 4, lower-right.  
note — 学習初期はMonthly Averageも参照し、後半はWeekly Averageへの一致を重視する設計。

9.2 Anomaly Inpainting Loss

論文式：
```
L_total_ano =
λ_rec_ano * L_rec_ano
+
λ_adv_ano * L_adv_ano
```
実装：
```
pred_sst = pred_weekly + pred_anomaly
loss_rec = mse_masked(pred_sst, target_sst, target_mask)
loss_g = lambda_rec * loss_rec + lambda_adv * loss_adv
```
original_text — “We define the reconstruction loss Lano rec that is calculated only in the observed pixels…”
source — uploaded PDF, p. 4, lower-right.  
note — 欠損領域には教師値がないため、観測済みpixelだけでL2 lossを計算する。

9.3 Adversarial Physical Model Loss

DiscriminatorのrealはData Assimilation SST。
```
real = data_assimilation_sst
fake = generator_output
```
BCE loss：
```
loss_d_real = BCE(D(real), 1)
loss_d_fake = BCE(D(fake.detach()), 0)
loss_d = loss_d_real + loss_d_fake
loss_adv_g = BCE(D(fake), 1)
```
original_text — “Adversarial physical model loss is obtained by setting the ground-truth image xn in (2) to a data-assimilation image.”
source — uploaded PDF, p. 3, lower-right.  
note — 物理モデルそのものを実行するのではなく、データ同化SST画像をreal sampleとして使う。

⸻

10. Dataset実装

10.1 AverageDataset

返す内容：
```
{
    "sst_volume": Tensor [1, T, H, W],
    "mask_volume": Tensor [1, T, H, W],
    "monthly_average": Tensor [1, 1, H, W],
    "monthly_mask": Tensor [1, 1, H, W],
    "weekly_average": Tensor [1, 1, H, W],
    "weekly_mask": Tensor [1, 1, H, W],
    "assimilation_weekly": Tensor [1, H, W]
}
```
10.2 AnomalyDataset

返す内容：
```
{
    "sst_volume": Tensor [1, T, H, W],
    "mask_volume": Tensor [1, T, H, W],
    "target_sst": Tensor [1, H, W],
    "target_mask": Tensor [1, H, W],
    "weekly_average": Tensor [1, H, W],
    "assimilation_sst": Tensor [1, H, W],
    "random_cloud_mask": Tensor [1, H, W]
}
```
学習時には、別画像から取った雲マスクをランダムに適用する。

original_text — “mrand is a randomly chosen cloud mask taken from other SST images in a training dataset… Specifically, we prepare cloud-mask patterns from satellite images and randomly choose mrand from them.”
source — uploaded PDF, p. 4–5.  
note — 欠損形状を矩形マスクではなく実際の雲マスクから作る点が重要。

⸻

11. 学習設定

論文準拠の初期設定：
``` YAML
patch_size: 256
time_steps: 9
time_interval_hours: 6
epochs: 30
batch_size: 32
optimizer: Adam
beta1: 0.5
beta2: 0.99
lr_generator: 1.0e-4
lr_discriminator: 1.0e-8
lambda_ano_rec: 10.0
lambda_ano_adv: 0.1
lambda_ave_rec: 10.0
lambda_ave_adv: 0.1
```
original_text — “Our generator and discriminator are trained for 30 epochs with a batch size of 32. We used the Adam optimizer… The learning rate of the generator and discriminator are 1e−4 and 1e−8.”
source — uploaded PDF, p. 6, upper-middle.  
note — Discriminatorの学習率が極端に小さい。まず論文通りにするが、学習が不安定なら調整候補になる。

⸻

12. 学習手順

12.1 Stage 1: Average Estimation
```
for epoch:
    for batch:
        1. pred_weekly = G_average(sst_volume, mask_volume, monthly_average, monthly_mask)
        2. D_average更新
            real = assimilation_weekly
            fake = pred_weekly.detach()
        3. G_average更新
            loss = reconstruction loss + adversarial physical model loss
        4. log
        5. checkpoint保存
```
保存：
```
checkpoints/average/g_average_epoch_xx.pth
checkpoints/average/d_average_epoch_xx.pth
```
12.2 Stage 2: Anomaly Inpainting
```
for epoch:
    for batch:
        1. weekly_averageを取得
           a. 教師weekly_averageを使う
           または
           b. 学習済みG_averageでpred_weeklyを生成
        2. anomaly_volume = sst_volume - weekly_average
        3. pred_anomaly = G_anomaly(anomaly_volume, mask_volume)
        4. pred_sst = weekly_average + pred_anomaly
        5. D_anomaly更新
            real = assimilation_sst
            fake = pred_sst.detach()
        6. G_anomaly更新
            loss = masked reconstruction loss + adversarial physical model loss
        7. log
        8. checkpoint保存
```
推奨：最初は教師weekly_averageでAnomaly Networkを学習し、次にpred_weeklyでfine-tuningする。

⸻

13. 推論仕様

入力：
```
9時刻SST
9時刻mask
monthly_average
monthly_mask
```
処理：
```
pred_weekly = G_average(
    sst_volume,
    mask_volume,
    monthly_average,
    monthly_mask
)
anomaly_volume = sst_volume - pred_weekly
anomaly_volume = anomaly_volume * mask_volume
pred_anomaly = G_anomaly(
    anomaly_volume,
    mask_volume
)
pred_sst = pred_weekly + pred_anomaly
```
出力：
```
cloud_free_sst.tif
cloud_free_sst.npy
visualization.png
```
original_text — “Our model first synthesizes short-term average SST images and then reconstructs target SST images with an anomaly inpainting network.”
source — uploaded PDF, p. 3, Fig. 2 caption.  
note — 推論も2段階。Average推定後にAnomaly補完を行う。

⸻

14. 評価指標

最低限：
```
RMSE
MAE
Bias
valid pixel ratio
```
RMSE：
```
rmse = sqrt(mean((pred_sst - reference_sst) ** 2))
```
参照データ候補：
```
buoy / in situ SST
data assimilation SST
cloud-free satellite observation
```
original_text — “we used buoy data of the in situ SST quality monitor (iQuam)… as the ground-truth data of SSTs.”
source — uploaded PDF, p. 8, lower-left.  
note — 論文ではiQuam buoyを評価用ground truthとしている。手元にない場合は代替評価を用意する。

⸻

15. 実装上の注意

最初から完全再現を狙わない。以下の順で作る。

1. Datasetなしで、ランダムテンソルを入力してGeneratorのshape確認
2. Average Generator単体のforward確認
3. Anomaly Generator単体のforward確認
4. Discriminatorのforward確認
5. Loss計算確認
6. 1 batchだけ学習
7. 小規模データでoverfit確認
8. 本学習
9. 推論コード作成
10. 可視化とRMSE評価

特に重要なのは、shapeを固定して確認すること。
```
B = 2
T = 9
H = W = 256
sst_volume = torch.randn(B, 1, T, H, W)
mask_volume = torch.ones(B, 1, T, H, W)
monthly = torch.randn(B, 1, 1, H, W)
monthly_mask = torch.ones(B, 1, 1, H, W)
```
⸻

16. Codexへの実装指示文

以下をCodexに渡すとよいです。
```
Implement a PyTorch project for the SST anomaly inpainting method described in the specification.
Requirements:
- Use PyTorch.
- Implement Average Estimation Generator and Anomaly Inpainting Generator.
- Implement a shared encoder-decoder architecture based on the paper:
  encoder: conv layers with ELU and dilated convolutions.
  decoder: conv layers with bilinear upsampling.
- Use ReflectionPad2d for convolution padding.
- Use tanh output in the final layer.
- Implement a SpectralNorm-based 2D discriminator.
- Implement masked MSE reconstruction loss.
- Implement adversarial physical model loss using BCE.
- Implement train_average.py and train_anomaly.py.
- Use config.yaml for hyperparameters.
- Add tensor shape assertions.
- First make the code runnable with synthetic random tensors before requiring real SST data.
```
⸻

17. 最小実装で省略してよいもの

初期版では省略可：
```
GeoTIFF入出力
緯度経度座標処理
iQuam評価
本格的なData Assimilationデータ整形
大画像1650×1650の反復復元
```
まず実装すべきもの：
```
Generator
Discriminator
Loss
Dataset interface
Training loop
Inference loop
```
⸻

