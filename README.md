# SST Anomaly Inpainting

衛星「ひまわり」の海面水温（SST: Sea Surface Temperature）画像について、雲によって観測できない部分を深層学習で補完する研究用プロジェクトです。PyTorchを用いて、データの前処理、学習用データの読み込み、2段階のモデル学習、パッチ単位の推論を実装しています。

**現在は動作可能なベースラインであり、対象論文の完全再現や復元精度の達成を示すものではありません。** 実データ学習では再構成損失を使用し、人工雲マスクの適用とデータ同化SSTによる敵対的学習は未接続です。

## 目次

- [手法と処理の流れ](#手法と処理の流れ)
- [ディレクトリ構成](#ディレクトリ構成)
- [コードを読む順番](#コードを読む順番)
- [使用技術・ライブラリ](#使用技術ライブラリ)
- [実行方法](#実行方法)
- [データ形式と学習時の扱い](#データ形式と学習時の扱い)
- [設定項目](#設定項目)
- [テスト](#テスト)
- [実装状況と関連資料](#実装状況と関連資料)

## 手法と処理の流れ

水温を「週平均」と「そこからの偏差」に分けて推定します。ここでの **Anomalyは水温の偏差**を意味します。

1. **平均推定（Average Estimation）**：時系列SST・有効画素マスク・月平均SSTから、欠損のない週平均SSTを推定します。
2. **偏差補完（Anomaly Inpainting）**：時系列SSTと推定週平均の差を使って、対象時刻の偏差を推定します。
3. **再構成**：推定週平均と偏差を、単位を合わせて加算します。

```text
ひまわりのNetCDFデータ
    ↓ 対象領域の切り出し・品質判定・雲マスク作成
観測ごとの256×256パッチ（NPZ）＋ 7日／30日平均（NPZ）
    ↓ 同じ位置の9時刻分を読み込み・正規化
時系列SST ＋ 有効画素マスク ＋ 月平均SST
    ↓ 平均推定モデル
推定週平均SST
    ↓ 時系列SST・有効画素マスクとともに偏差補完モデルへ入力
推定偏差
    ↓ 推定週平均へ換算後の偏差を加算
復元SSTパッチ
```

標準設定では6時間間隔・9時刻分（対象時刻の48時間前から対象時刻まで）を使います。推論CLIに渡す時系列NPZは、別途組み立てと正規化が必要です。

## ディレクトリ構成

```text
.
├── README.md                         # プロジェクト概要・構成・実行方法
├── requirements.txt                  # Pythonの依存ライブラリ
├── config.yaml                       # データ条件・学習条件・保存先
├── .gitignore                        # データ・学習結果・キャッシュ等の除外設定
├── src/                              # 実装コード
│   ├── himawaridata/                 # 衛星データの取得・前処理・可視化
│   │   ├── FTP-Ptree.py              # P-TreeからNetCDFをFTP取得
│   │   ├── crop_nc_to_1650.py         # 前処理CLI、領域切り出し、平均値作成、旧形式保存
│   │   ├── frame_preprocessing.py    # 観測ごとのパッチを現行NPZ形式で保存
│   │   └── visualize_crop.py         # 切り出し領域・旧形式のパッチを可視化
│   ├── datasets/                     # 保存済みデータを学習用Tensorへ変換
│   │   ├── frame_index.py            # 観測時刻の索引作成・時系列選択・座標整合確認
│   │   ├── himawari_patch_dataset.py # 実データ学習用Dataset／DataLoader、正規化
│   │   ├── loaders.py               # JSON／JSONL／CSV索引、NPY／NPZ読み込みの共通処理
│   │   ├── average_dataset.py       # 索引ファイルに基づく平均推定用Dataset・疑似データ
│   │   └── anomaly_dataset.py       # 索引ファイルに基づく偏差補完用Dataset・疑似データ
│   ├── models/                       # ニューラルネットワーク
│   │   ├── blocks.py                # 反射パディング・畳み込み・ELU等の共通部品
│   │   ├── generator.py             # 共通Encoder–Decoder、平均推定器、偏差補完器
│   │   └── discriminator.py         # Spectral Normalization付き識別器
│   ├── losses/                       # 学習の損失関数
│   │   ├── reconstruction_loss.py   # 有効画素だけで計算するmasked MSE
│   │   └── adversarial_loss.py      # 生成器・識別器の敵対的損失（BCE）
│   ├── train/                        # 学習CLI・最適化・チェックポイント保存
│   │   ├── train_average.py         # 第1段階：平均推定モデルの学習
│   │   └── train_anomaly.py         # 第2段階：平均推定器を凍結して偏差モデルを学習
│   ├── infer/
│   │   └── reconstruct.py           # 学習済みの2モデルによるパッチ再構成
│   └── utils/
│       ├── config.py                # YAML設定の読み込み
│       └── normalization.py         # 偏差からSST正規化空間へのスケール換算
├── tests/                            # 前処理・Dataset・モデル・損失・学習・推論のテスト
└── docs/                             # 設計メモ・論文との対応・実装状況の記録
    ├── specification.md             # 初期の実装仕様
    ├── architecture.md              # ネットワーク構造・Tensor形状の設計
    ├── implementation_plan.md       # 初期の実装計画
    ├── paper_notes.md               # 論文の手法と実装の対応メモ
    └── reproduction_audit.md        # 論文との差分・未実装事項の監査記録
```

`__init__.py`はPythonパッケージの初期化やクラスの公開に使います（上の一覧では省略）。`average_dataset.py`と`anomaly_dataset.py`は汎用の索引ベースの入口で、通常のひまわり実データ学習CLIは`himawari_patch_dataset.py`を使用します。

次のディレクトリは、データ準備や実行時に作成するものです。観測データ・学習済み重みはリポジトリに同梱していません。

| パスの例 | 内容 |
|---|---|
| `data_raw/Himawari_raw/` | 入力となるNetCDFデータの配置先 |
| `data/frames256/` | 前処理済みの観測パッチ・平均値 |
| `checkpoints/average/` | 平均推定モデルとoptimizer等の保存状態 |
| `checkpoints/anomaly/` | 偏差補完モデルとoptimizer等の保存状態 |
| `outputs/` | 推論結果などの出力先 |

## コードを読む順番

- **モデルを知りたい場合**：[generator.py](src/models/generator.py) → [blocks.py](src/models/blocks.py) → [discriminator.py](src/models/discriminator.py)
- **データの作り方を知りたい場合**：[crop_nc_to_1650.py](src/himawaridata/crop_nc_to_1650.py) → [frame_preprocessing.py](src/himawaridata/frame_preprocessing.py) → [frame_index.py](src/datasets/frame_index.py) → [himawari_patch_dataset.py](src/datasets/himawari_patch_dataset.py)
- **学習・推論を追いたい場合**：[train_average.py](src/train/train_average.py) → [train_anomaly.py](src/train/train_anomaly.py) → [reconstruct.py](src/infer/reconstruct.py)
- **入出力の具体例を見たい場合**：[tests/](tests/)の各機能に対応するテスト

## 使用技術・ライブラリ

モデルはCNNのEncoder–Decoderを基本とし、拡張畳み込み（dilation=2, 4, 8, 16）、反射パディング、ELU、bilinearアップサンプリング、tanh出力を使用します。入力の融合には1×1×1のConv3dを使い、その後は時間軸をチャネルとして2次元CNNに渡します。最適化はAdamです。

| ライブラリ | 主な用途 |
|---|---|
| PyTorch | モデル定義、自動微分、学習・推論、Dataset／DataLoader |
| NumPy | 配列処理、欠損値処理、NPY／NPZの入出力 |
| netCDF4 | 衛星NetCDFデータの読み込み |
| Matplotlib | 切り出し領域やSSTの可視化 |
| PyYAML | YAML設定ファイルの読み込み |
| pytest | 自動テスト |

[requirements.txt](requirements.txt)には、このほか`torchvision`、`scipy`、`pandas`、`xarray`、`rasterio`、`tqdm`が記載されていますが、現行コードでの直接使用はありません。依存バージョンは未固定です。

## 実行方法

以下のコマンドはリポジトリ直下で実行します。

### 1. 依存ライブラリのインストール

```bash
python -m pip install -r requirements.txt
```

学習の標準設定はCUDAです。CPUを指定する場合は学習・推論コマンドに`--device cpu`を追加します。

### 2. 観測データの準備・前処理

ひまわりのNetCDFファイルを`data_raw/Himawari_raw/`に用意します。[FTP-Ptree.py](src/himawaridata/FTP-Ptree.py)を使う場合は、環境変数`PTREE_FTP_USER`・`PTREE_FTP_PASS`を設定し、スクリプト内の保存先・取得期間・データバージョンを利用環境に合わせて変更してください。

まず1観測時刻だけを処理する例です。

```bash
python -m src.himawaridata.crop_nc_to_1650 \
  data_raw/Himawari_raw data/frames256 --max-frames 1 --skip-temporal-averages
```

学習用に全観測と7日／30日平均を作成する場合は、上記2オプションを外します。

```bash
python -m src.himawaridata.crop_nc_to_1650 \
  data_raw/Himawari_raw data/frames256
```

平均の作成には同じ時刻帯の30日分の履歴が必要です。履歴が足りなくても観測フレーム自体は保存されますが、必要な平均ファイルがない対象は学習サンプルになりません。

### 3. 平均推定モデルの学習

`--data-root`には`frames/himawari/`ではなく、その親の`data/frames256`を渡します。

```bash
python -m src.train.train_average \
  --config config.yaml --data-root data/frames256
```

### 4. 偏差補完モデルの学習

```bash
python -m src.train.train_anomaly \
  --config config.yaml --data-root data/frames256 \
  --average-checkpoint checkpoints/average/latest.pth
```

実データ学習では`--average-checkpoint`が必須です。平均推定器は一度ロードして`eval`モードで固定し、`no_grad`で推定した週平均を偏差の基準にします。観測週平均をそのまま基準にはしません。チェックポイントにある温度・時刻・衛星等の設定は互換性を確認しますが、重みだけの`state_dict`にはその情報がありません。疑似データによるデバッグ学習では、チェックポイントなしで観測平均を使う経路も残しています。

現在の実データにはデータ同化SSTが含まれないため、両段階とも再構成損失のみで学習し、識別器・敵対的損失は0になります。

### 5. パッチ単位の推論

**観測フレームNPZを直接渡すことはできません。** 次のキーを持つ正規化済みNPZを別途用意します。欠損入力は0埋めし、マスクは有効画素を1とします。

| キー | バッチなしの形状 | 内容 |
|---|---|---|
| `sst_volume` | `[1,T,H,W]` | 時系列SST |
| `mask_volume` | `[1,T,H,W]` | 時系列SSTの有効画素マスク |
| `monthly_average` | `[1,1,H,W]` | 月平均SST |
| `monthly_mask` | `[1,1,H,W]` | 月平均の有効画素マスク |

先頭にバッチ次元を持つ5次元配列も受け付けます。標準設定は`T=9`、`H=W=256`です。

```bash
python -m src.infer.reconstruct \
  --config config.yaml \
  --average-checkpoint checkpoints/average/latest.pth \
  --anomaly-checkpoint checkpoints/anomaly/latest.pth \
  --input data/inference_input.npz \
  --output outputs/reconstruction.npz
```

出力NPZには`pred_weekly`、`pred_anomaly`、`pred_sst`を保存します。学習と同じ設定を使用してください。

```text
pred_sst = pred_weekly + pred_anomaly × 2 × anomaly_range / (max_temp - min_temp)
SST（℃） = (pred_sst + 1) × (max_temp - min_temp) / 2 + min_temp
```

`pred_weekly`と`pred_sst`はSST正規化空間の値、`pred_anomaly`は[-1,1]の偏差出力です。偏差を℃にするには`anomaly_range`を掛けます。

### 旧形式のデータを使う場合

時系列ボリュームを保存する旧形式も明示的に選択できます。

```bash
python -m src.himawaridata.crop_nc_to_1650 data_raw/Himawari_raw \
  data/legacy_patches256 --format legacy-volumes
```

学習コマンドの`--data-root`を`data/legacy_patches256`へ変更します。偏差学習では同様に`--average-checkpoint`が必要です。`visualize_crop.py`は旧形式を想定しています。

## データ形式と学習時の扱い

現行の出力形式は、観測フレームと平均値を分けて保存します。

```text
data/frames256/
├── frames/himawari/YYYYmmddHHMMSS_rXX_cXX.npz
└── averages/himawari/YYYYmmddHHMMSS_rXX_cXX.npz
```

- **観測NPZ（format_version=2）**：`sst`（float32）と`cloud_mask`（uint8）、各`[256,256]`。観測時刻、衛星名、緯度・経度、元ファイル、区画位置、品質閾値、雲率、有効率等も保存します。
- **平均NPZ**：`weekly_average`（`[1,H,W]`）と`monthly_average`（`[1,1,H,W]`）、時刻・座標。観測SSTそのものは含みません。
- **保存単位**：SSTは℃、欠損値はNaNです。保存時の`cloud_mask`は雲・欠損海域が1、晴天海域・陸地が0です。学習時は反転とSSTの有限値判定により、有効画素が1のマスクへ変換します。
- **パッチ作成**：1650×1650領域の中央1536×1536を、256×256の固定6×6グリッドに分割します。外周部分は含みません。
- **時系列選択**：同一区画の`[-48,-42,-36,-30,-24,-18,-12,-6,0]`時間の観測を、古い順に読みます。結合したボリュームはディスクに保存せず読み込み時に組み立てます。
- **学習対象の選別**：必要な観測・平均ファイルがない対象、対象時刻の雲率が閾値を超えるもの、有効な対象SSTがないものは除外します。雲率の分母は全パッチ画素です。過去の観測は全面欠損でも履歴として保持します。
- **検証・正規化**：メタデータ、配列形状、座標の整合性を確認し、不整合はエラーにします。設定した温度範囲で正規化し、無効入力を0埋めします。

観測パッチは雲の多さによらずすべて保存し、現行形式では雲マスクの複製プールを作りません。互換性のある既存フレームはスキップし、`--overwrite`で再生成できます。入力や処理条件を変更した場合は新しい出力ディレクトリを使用してください。

## 設定項目

[config.yaml](config.yaml)で主な条件を管理します。

| 項目 | 標準値 | 意味 |
|---|---|---|
| `data.patch_size` | 256 | パッチの一辺の画素数 |
| `data.time_steps` / `time_interval_hours` | 9 / 6 | 入力時刻数・時間間隔 |
| `data.max_cloud_fraction` | 0.5 | 学習対象時刻の雲率上限 |
| `data.min_temp` / `max_temp` | 0 / 35 | SST正規化に使う温度範囲（℃） |
| `data.anomaly_range` | 5 | 偏差出力のスケール（±5℃） |
| `training.epochs` / `batch_size` | 30 / 32 | エポック数・バッチサイズ |
| `training.device` / `mixed_precision` | cuda / false | 実行デバイス・混合精度の有効化 |
| `optimizer.lr_generator` / `lr_discriminator` | 1e-4 / 1e-8 | 生成器・識別器の学習率 |
| `loss.lambda_ave_rec` / `lambda_ano_rec` | 各10.0 | 各段階の再構成損失の重み |
| `loss.lambda_ave_adv` / `lambda_ano_adv` | 各0.1 | 同化データがある場合の敵対的損失の重み |
| `paths.checkpoint_dir` / `output_dir` | checkpoints / outputs | 保存先の設定 |

## テスト

モデルの入出力形状を確認する例：

```bash
python -m pytest tests/test_generators_forward.py
```

テスト全体を実行する場合：

```bash
python -m pytest
```

| テストファイル | 確認する内容 |
|---|---|
| `test_generators_forward.py` / `test_discriminator_forward.py` | 生成器・識別器の順伝播と入出力形状 |
| `test_losses.py` | 再構成損失・敵対的損失 |
| `test_datasets.py` | 汎用の索引ベースDataset |
| `test_himawari_real_dataset.py` / `test_frame_dataset.py` | NPZ読み込み、正規化、マスク、時系列選択 |
| `test_himawari_preprocessing.py` / `test_frame_preprocessing.py` | 切り出し・平均作成・観測フレーム保存 |
| `test_train_average.py` / `test_train_anomaly.py` | 学習処理・チェックポイント・2段階の接続 |
| `test_reconstruct.py` | 推論結果と偏差の単位換算 |

これらは実装の動作確認であり、実観測に対する復元精度の評価ではありません。

## 実装状況と関連資料

**実装済み**：ひまわりの前処理、時系列Dataset、平均推定器・偏差補完器・識別器、損失関数、2段階学習、チェックポイント保存、正規化済みパッチの推論、自動テスト。

**未実装・未検証の主な事項**：

- 別観測から得た人工雲マスクを入力に適用する実データ学習。
- データ同化SST用ローダーの接続と、実データでの敵対的学習。
- 観測フレームを直接受け付ける推論アダプター、全域の反復パッチ復元・合成。
- 学習・検証・評価の期間分割、ブイとの照合、RMSE／MAE／bias等の精度評価。
- GCOM-Cへの対応（現在はひまわりのみ）。
- GPUでの実学習・AMPの検証。現状はautocastのみでGradScalerはなく、混合精度は既定で無効。
- 学習再開用CLI。optimizerの状態は保存しますが、再開処理は未実装。

詳細は[再現監査](docs/reproduction_audit.md)を参照してください。ただし、監査記録には平均推定器の接続前の記述が残っています。**現在は、凍結した平均推定器を偏差学習で使用する処理が実装済みです。** その他の`docs/`も初期設計を含むため、現在の動作は本READMEと実装コードを合わせて確認してください。

本プロジェクトが対象とする手法は、Hiraharaらの「Cloud-Free Sea-Surface-Temperature Image Reconstruction From Anomaly Inpainting Network」です。論文との対応メモは[paper_notes.md](docs/paper_notes.md)にまとめています。
