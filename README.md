# SST Anomaly Inpainting

ひまわりの海面水温（SST）画像の雲による欠損を補完する、PyTorchによる研究用実装です。

**現在の確認対象は「9時刻の観測平均＋最近傍補間を基準SSTとするアノマリーGAN」です。** 平均推定ネットワークを省略し、JCOPEの毎時SSTを識別器の実画像に使います。全域方式も実装していますが、データ読み込み時の計算負荷が大きいため、現在は256×256パッチ方式で動作と学習を確認しています。

平原論文を参考にした変更手法であり、論文の完全再現や欠損領域での精度を保証するものではありません。RAFTは未実装です。

## 方式を選ぶ設定ファイル

| 設定 | 基準SST | 学習 | 推論 |
|---|---|---|---|
| `config_temporal_mean_patch.yaml` | パッチ内の9時刻平均＋最近傍補間 | アノマリーのみ | パッチ復元。現在の確認用 |
| `config.yaml` | 平均推定ネットワークの推定週平均 | 平均推定→アノマリー | 従来のパッチ復元 |

`main.py`の既定値は`config.yaml`です。**新方式は必ず`--config`で指定してください。** コマンドラインのエポック数などによる上書きを含め、実際に使用した設定が実験フォルダの`config.yaml`へ保存されます。サーバーへコードと設定を転送した後、その内容を確認してください。

## 現在のネットワークと入出力

```text
同じ位置・6時間間隔のSST 9枚 ＋ 各時刻の観測マスク
    │
    ├─ 有効な観測だけを画素ごとに平均 → 平均画像1枚
    │      └─ 残った欠損だけ最近傍補間 → 基準SST 1枚
    │
    └─ 各時刻のSST − 基準SST → 偏差9枚 ＋ 観測マスク
                                  ↓
                         アノマリー生成器
                                  ↓
                         対象時刻の推定偏差1枚
                                  ↓ 基準SSTを加算
                            復元SST 1枚
                                  ↓ 学習時のみ
               全体識別器：JCOPEの毎時SSTと生成画像を識別
```

- 入力時刻は対象時刻の48時間前から対象時刻まで、6時間間隔の9枚です。**この平均は週平均ではありません。**
- 元の9枚をそれぞれ補間するのではなく、平均後の1枚だけを補間します。
- 有効な時刻だけを平均し、9時刻すべてで欠損した位置を最近傍の有効な平均値で埋めます。入力全体に有効観測がなければエラーにします。
- 新方式では追加の人工雲や9時刻共通の雲マスクを適用しません。
- マスクは1が有効、0が欠損です。SST入力は0～35℃を−1～1へ写像し、欠損は0とマスクで表現します。偏差は±5℃を±1へ写像し、範囲外はクリップします。

### 生成器

学習時の入力はSSTとマスクがそれぞれ`[B,1,9,256,256]`、基準SSTが`[B,1,1,256,256]`、出力偏差が`[B,1,256,256]`です。

偏差とマスクを1×1×1のConv3dで2→1チャネルへ変換し、9時刻を2次元ネットワークの9チャネルとして扱います。

| 部分 | 構成 |
|---|---|
| エンコーダ | 出力チャネル32→64→64→128→128→256。3回のstride 2で空間サイズを1/8へ |
| 拡張畳み込み | 256チャネル、dilation 2・4・8・16 |
| デコーダ | 畳み込み各2層で128→64→32→16チャネル。途中に3回のバイリニア2倍拡大 |
| 最終出力 | 3×3畳み込みで1チャネル、−1～1にクリップ |
| 活性化・境界 | 主にELU、エンコーダ最終層と出力層は線形。反射パディング |

生成器は平均推定・アノマリーで共通の層構成を持ちますが、別の重みです。新方式では平均推定生成器は学習・使用しません。

### 識別器と損失

新設定では全体識別器のみを使用します。5×5・stride 2の畳み込み6層（64→128→256→256→256→512）と4×4の最終層で、SSTパッチ1枚から1値のlogitを出します。中間層はLeakyReLU、全畳み込みにSpectral Normalizationを使用します。

- 再構成損失：観測画素だけの画像L2ノルムをバッチ平均。画素MSEではありません。
- Gの敵対的損失：`mean(log(1 − D(fake)))`。
- Dの損失：`−mean(log D(real)) − mean(log(1 − D(fake)))`。
- Gの合計：`10 × 再構成損失 + 0.1 × 敵対的損失`。
- 実装はlogitと`logsigmoid`を使用します。Gの敵対的損失が負になることは異常ではありません。
- JCOPEは画素単位で一致する教師ではなく、独立に抽出した実画像です。アノマリー段階の識別器には偏差ではなく**復元SST**を渡します。

局所PatchGANと`physical_alpha`に対応する旧実験コードは残っていますが、同梱の新設定では使用しません。古い設定に`physical_alpha`が残るとλより優先されるため注意してください。

## フォルダ構成

```text
.
├── main.py                         # 前処理・学習・復元の実行管理
├── config*.yaml                    # 方式別の設定
├── requirements.txt
├── download_himawari_sst.py         # ひまわりFTP取得
├── check_himawari_download.py       # 取得状況・ファイル確認
├── check_jcope_download.py          # JCOPEの読み出し確認
├── src/
│   ├── preprocessing/
│   │   ├── Himawari_preprocessing.py # 全域SST・マスク保存
│   │   ├── JCOPE_preprocessing.py    # 毎時JCOPEを衛星格子へバイリニア補間
│   │   ├── hourly_assimilation.py    # 毎時・週平均の実画像プール作成
│   │   ├── full_domain_input.py      # 指定時刻までの全域9枚を読む
│   │   ├── pipeline.py              # 前処理の統合入口
│   │   ├── sampling.py              # 学習パッチの索引作成
│   │   ├── common.py                # NPY・JSONの入出力
│   │   └── GPV_preprocessing.py     # 旧GPV経路
│   ├── datasets/                   # 統合Datasetと互換ローダー
│   │   └── unified_dataset.py      # パッチ読み出し・独立JCOPE抽出
│   ├── models/
│   │   ├── generator.py            # 平均推定・アノマリー生成器
│   │   ├── discriminator.py        # 全体識別器・旧局所識別器
│   │   ├── temporal_baseline.py     # 9時刻平均・最近傍補間・進捗表示
│   │   └── blocks.py               # 畳み込み・ELUなど
│   ├── losses/                     # 再構成・敵対的損失
│   ├── train/                      # train_average.py / train_anomaly.py
│   ├── infer/                      # reconstruct.py / full_domain.py
│   ├── utils/                      # 設定・正規化・チェックポイント
│   ├── himawaridata/               # 既存の取得・前処理関連コード
│   └── gcomc/                      # GCOM-C関連コード（新方式では使用しない）
├── visualize/                      # 画像・ヒストグラム・損失曲線・評価補助
├── tests/                          # 合成データ等の自動テスト
├── docs/                           # 設計・実験履歴。旧方式の説明も含む
├── olds/                           # 旧実装
├── data/                           # ローカルデータ用。サーバーは外部パスを指定
└── outputs/                        # 実験設定・ログ・重み・復元結果
```

## データと保存先

サーバーで使用しているパスの例です。パス名だけでディスクは決まりません。保存前に`df -h /data2/itoh/datas`などで実際のマウントと空き容量を確認してください。

| 保存先 | 内容 |
|---|---|
| `/data2/itoh/himawari/data_raw` | ひまわりの元NetCDF |
| `/data2/itoh/JCOPE-T` | JCOPEの元NetCDF |
| `/data2/itoh/datas/unified_sst` | 全域ひまわりSST・マスク・索引 |
| `/data2/itoh/datas/JCOPE-T_aligned` | 衛星格子に合わせた毎時JCOPE |
| `/data2/itoh/datas/JCOPE-T_physical_pool` | JCOPEの毎時・週平均とパッチ索引 |
| `/workspace/sst-inpainting-code/outputs` | 学習・復元結果 |

`unified_sst`は256×256パッチだけを保存する形式ではありません。

```text
unified_sst/
├── dataset.json              # 全域フレームと学習時刻・位置・履歴の索引
├── run.json                  # 前処理の完了状態・設定
├── lat.npy / lon.npy         # 共通格子
└── himawari/YYYYMMDDHHMMSS/
    ├── sst.npy               # 全域の℃。無効画素はNaN
    ├── valid_mask.npy        # 1=有効観測
    ├── cloud_mask.npy        # 1=品質判定後の海上欠損
    ├── land_mask.npy         # 1=陸
    └── metadata.json
```

`cloud_mask`は独立した雲分類モデルの出力ではありません。品質判定で除外された海上画素も含みます。学習時に索引の位置でパッチを切り出します。

## 環境準備

Python 3.10のサーバー環境で運用しています。GPU学習には環境に対応したPyTorchが必要です。

```bash
python3 -m pip install -r requirements.txt
```

`eccodes`は旧GPVのGRIB処理用です。ひまわり・JCOPEのみの前処理ではGRIBを読みません。

## 前処理

前処理済みデータがある場合は再実行不要です。以下は新しい保存先に作成する場合の例です。

```bash
python3 -m src.preprocessing.pipeline \
  --config config_temporal_mean_patch.yaml \
  --himawari-input /data2/itoh/himawari/data_raw \
  --output /data2/itoh/datas/unified_sst \
  --bounds 17 50 117 150
```

JCOPEを衛星の緯度・経度格子に合わせ、学習用プールを作ります。「1km→2km」という倍率だけでなく、座標を一致させる処理です。

```bash
python3 -m src.preprocessing.JCOPE_preprocessing \
  --input /data2/itoh/JCOPE-T \
  --satellite-root /data2/itoh/datas/unified_sst \
  --output /data2/itoh/datas/JCOPE-T_aligned \
  --start 2024-08-01 --end 2025-09-30

python3 -m src.preprocessing.hourly_assimilation \
  --source /data2/itoh/datas/JCOPE-T_aligned \
  --satellite-root /data2/itoh/datas/unified_sst \
  --output /data2/itoh/datas/JCOPE-T_physical_pool
```

新方式のアノマリー学習は毎時プールを使用します。現行のプール作成処理は互換性のため週平均も作成し、配列を複製するため保存容量に注意してください。

## パッチ方式の学習（現在の確認経路）

すべてリポジトリ直下で実行します。`--run-dir`は毎回、新規または空のフォルダを指定してください。

### 1バッチの動作確認

```bash
python3 main.py \
  --config config_temporal_mean_patch.yaml \
  --data-root /data2/itoh/datas/unified_sst \
  --assimilation-root /data2/itoh/datas/JCOPE-T_physical_pool \
  --run-dir outputs/temporal_mean_patch_smoke_001 \
  --start-at anomaly --stop-after anomaly --smoke
```

`--smoke`は1エポック分の全件学習ではなく、バッチサイズ1・1更新の動作確認です。

### 10エポック学習

```bash
mkdir -p logs
nohup python3 -u main.py --config config_temporal_mean_patch.yaml --data-root /data2/itoh/datas/unified_sst --assimilation-root /data2/itoh/datas/JCOPE-T_physical_pool --run-dir outputs/training_temporal_mean_patch_epoch10_001 --start-at anomaly --stop-after anomaly --epochs 10 --batch-size 32 --num-workers 0 > logs/training_temporal_mean_patch_epoch10_001.log 2>&1 < /dev/null &
echo "学習PID: $!"
```

設定された学習対象月は2025年・2026年それぞれの4～6月です。対象時刻を月で選び、入力履歴は前月に及ぶ場合があります。JCOPEの実画像は独立抽出であり、衛星と同期間であることを要求しません。

### 進捗確認

```bash
tail -f logs/training_temporal_mean_patch_epoch10_001.log
pgrep -af 'main.py|src.train.train_anomaly'
nvidia-smi
```

`training.baseline_log_interval`で平均・補間ログの間隔を設定します。1なら毎バッチ、100なら各エポックの最初と100バッチごと、0なら非表示です。パッチ方式の学習で平均の開始・完了、最近傍補間の開始・完了、処理時間、補間画素数、最大補間距離を出します。平均の時間にはCPUへの転送を含みます。

`samples=32`は32サンプルの意味です。各サンプルの9枚を独立に平均し、32枚の基準画像を作ります。32サンプルを混ぜて平均するわけではありません。

## パッチ復元

学習後、例えば前回保存した正規化済み入力NPZを再利用して同じ地点を復元します。入力NPZには`sst_volume`、`mask_volume`、互換性のため`monthly_average`、`monthly_mask`が必要です。新方式では月平均は生成器へ使用しません。

```bash
python3 main.py \
  --config outputs/training_temporal_mean_patch_epoch10_001/config.yaml \
  --run-dir outputs/temporal_mean_patch_preview_001 \
  --start-at reconstruct --stop-after reconstruct \
  --anomaly-checkpoint outputs/training_temporal_mean_patch_epoch10_001/checkpoints/anomaly/latest.pth \
  --inference-input outputs/training_epoch10_preview_001/outputs/preview_input.npz
```

以前の入力ファイルが存在し、正規化・時刻・位置が比較条件に一致していることが前提です。平均推定モデルのチェックポイントは不要です。

```text
outputs/実験名/
├── config.yaml
├── run.json / timings.csv
├── logs/anomaly.log
├── checkpoints/anomaly/latest.pth
└── outputs/reconstruction.npz       # 復元を実行した実験のみ
```

パッチ復元NPZの`pred_weekly`は互換キー名で、新方式では「9時刻の基準SST」を意味します。`pred_weekly`・`pred_sst`は正規化SST、`pred_anomaly`は偏差単位値です。℃換算はそれぞれ`(x+1)*17.5`、偏差は`x*5`です（設定0～35℃・偏差5℃の場合）。追加の`baseline`も正規化値です。

既存の`visualize/visualize_epochsmoke.py`には週平均という旧表示が残っています。また、旧`preview_input.json`と再作成した`dataset.json`のサンプル番号が一致しないと追加可視化で停止する場合があります。表示名や対応確認を省略して精度を判断しないでください。

## 全域方式（実装済み・現在は後回し）

`config_temporal_mean.yaml`の`baseline_scope: full_domain`を使います。平均・補間を全域で行った後、学習ではパッチに切り出します。**現行Datasetはサンプルごとに全域9枚を再処理するためCPU負荷が高く、基準画像の事前保存・再利用は未実装です。**

基準画像だけの確認は学習済み重みなしで実行できます。

```bash
python3 -m src.infer.full_domain \
  --config config_temporal_mean.yaml \
  --data-root /data2/itoh/datas/unified_sst \
  --target-time 2025-04-30T00:00:00 \
  --output outputs/temporal_mean_full_check_001 --baseline-only
```

全域復元は同じ全域基準方式で学習した`--anomaly-checkpoint`を指定し、`--baseline-only`を外します。パッチ基準の重みを同じ方式とみなして使用しません。全域生成器入力は8の倍数へ反射パディングし、出力を元サイズへ切り戻します。1650×1650の場合は1656×1656へ拡張します。実データでのGPUメモリ使用量は未確認です。

出力は`baseline_sst.npy`、`observed_sst.npy`、`observation_count.npy`、`interpolated_mask.npy`、`distance_pixels.npy`、`ocean_mask.npy`、`lat.npy`、`lon.npy`、`metadata.json`です。復元時は`pred_anomaly.npy`と`reconstructed_sst.npy`も保存します。**全域出力の水温・偏差は℃**で、陸域はNaNです。

## 制約と評価上の注意

- 旧来の月平均・週平均に基づくサンプル選択条件と、ローダー内の平均計算が残っています。新方式専用の索引再設計は未実施です。
- パッチ方式はパッチ外の観測を補間に利用できません。現在は全海域の学習パッチを索引で選択し、補間関数へ海陸マスクを渡していません。全域方式は陸域を除外しますが、最近傍距離は海岸線を迂回しません。
- 新方式は人工雲を追加しません。観測画素の再構成誤差は、本来の雲下の復元精度ではありません。
- 学習／検証／テストの自動分割と独立検証はメイン処理に組み込まれていません。`visualize/`の評価補助だけで独立評価済みとはなりません。
- 月や日時の近いサンプルには入力履歴の重複があります。評価分割は履歴・基準画像の作成期間も考慮する必要があります。
- データ・損失・基準画像の方式を変更した場合は再学習が必要です。生成器を読み込めることと、比較条件が一致することは別です。
- 過去の`docs/`には平均推定やGPV、局所識別器など別実験の説明があります。現在の実行条件は保存済み設定とコードを確認してください。

## テスト

小規模な合成データで平均、補間、マスク、勾配、学習・推論の経路を検証します。実データの復元品質を保証するテストではありません。

```bash
OMP_NUM_THREADS=1 python3 -m pytest -q tests/test_temporal_baseline.py tests/test_train_anomaly.py tests/test_full_domain_temporal.py tests/test_main_pipeline.py
```

## 参考と実装の位置付け

参考論文：Hirahara et al., *Cloud-Free Sea-Surface-Temperature Image Reconstruction From Anomaly Inpainting Network*, IEEE TGRS, DOI: [10.1109/TGRS.2021.3111649](https://doi.org/10.1109/TGRS.2021.3111649)。

生成器・全体識別器は論文の表I～IIIを参考にしています。現在の観測9時刻平均＋最近傍補間、追加人工雲を使用しない学習、データ期間・データ製品は原著の設定とは異なります。旧方式や変更履歴は`docs/paper_revision.md`、`docs/hirahara_global_gan.md`を参照してください。
