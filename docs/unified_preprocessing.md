# ひまわり・GPVの一括前処理と共通ローダー

## 構成

- `src/preprocessing/Himawari_preprocessing.py`: NetCDFの水温、品質・陸地フラグを読み、℃・雲マスク・陸地マスク・有効マスクを保存。
- `src/preprocessing/GPV_preprocessing.py`: Psbs/FD01/水深1mを抽出、ひまわりの座標へ双線形補間、日別・連続7日平均を保存。
- `src/preprocessing/sampling.py`: 再現可能なランダムクロップ位置の索引を作成。
- `src/preprocessing/pipeline.py`: 上記を順番に実行する入口。
- `src/datasets/unified_dataset.py`: 時系列・平均・GPVを遅延読み込みしてPyTorchの同じバッチにまとめるローダー。学習コマンドから自動選択。
- `olds/`: 従来の固定グリッド前処理、GPV前処理、旧NPZ専用ローダーと補助コード。取得用FTPコード・モデル・汎用Dataset・検査用可視化は現位置に残す。

保存はNPY＋JSONです。全領域を1時刻ずつ保存し、読み込み時にはメモリマップで必要なパッチだけを読むため、時系列ごとに同じ画素を重複保存しません。入力ファイルは変更しません。

## サーバーで実行

リポジトリをサーバーへ反映してから実行します。本PCではサーバーのデータ処理は行っていません。

```bash
cd /workspace/sst-inpainting-code
python3 -m pip install -r requirements.txt

python3 -m src.preprocessing.pipeline \
  --himawari-input /workspace/sst-inpainting-code/data_raw/Himawari_raw \
  --gpv-input /datasets/jmagpv/jpn \
  --output /workspace/sst-inpainting-code/data/unified_sst \
  --config config.yaml \
  --patches-per-frame 8
```

出力先は空である必要があります。失敗時の部分生成物を混ぜて再開する機能はありません。修正後は新しい出力先を指定してください。

最初の読み出し確認には、別の出力先で `--max-frames 1 --max-archives 1` を追加できます。この場合は30日履歴・7日GPVが揃わず、`training_ready=false` が正常です。

ひまわりとGPVの前処理モジュールは独立したPython関数 `preprocess(...)` としても呼べます。一括実行のCLIはpipelineに集約しています。

## 出力とサンプリング条件

- `himawari/<時刻>/`: 全対象海域の `sst.npy`（℃、NaN欠損）、`cloud_mask.npy`、`land_mask.npy`、`valid_mask.npy`、メタデータ。
- `gpv/daily/<日付>/` と `gpv/weekly/<日付>/`: 共通格子の水温、有効マスク、時間区間・予報種別・元アーカイブ等のメタデータ。
- `lat.npy`, `lon.npy`: 共通座標。
- `dataset.json`: パラメータ、ファイル索引、ランダムクロップ位置、抽出数、GPV欠落日。
- `run.json`: 実行状態、学習可能か、サンプル数。未完了データはローダーが拒否。

対象領域の既定は北緯17〜50度、東経117〜150度（上端を含まない）。ひまわり実ファイルの座標で範囲を選びます。GPV範囲外はNaNで、外挿しません。GPVの日付はUTC日平均区間の開始日です。FD02等や別の深さは使用しません。全GRIBフィールドを巡回して実際の変数番号・深さを検査します。

ひまわりの各対象時刻で、雲被覆率50%以下（config設定）のランダム256角パッチを最大8個選びます。乱数シードはconfigのseedで、索引に保存されます。毎エポック位置を再抽選する方式ではありません。9時刻（6時間間隔）、マスク、平均は必ず同じ切り取り位置です。

現実装では**全画素が海域と判定できるパッチだけ**を選びます。GPVも日別・週平均それぞれ全画素有効なパッチだけを独立したプールへ登録します。これは陸地・欠損の0埋めを識別器が水温として学習しないための実装上の制約です。原著が同じ陸域除外条件だったとは断定しません。8個という個数も本実装の設定です。

ひまわりは30日間の同時刻観測ファイルがすべてある対象だけを使い、週平均は7日中3回以上、月平均は30日中5回以上の有効観測がある画素で算出します。平均はローダーが同じクロップから計算します。これは従来コードの履歴条件を引き継ぐもので、原著の完全な仕様と断定しません。

GPVは連続7日・各画素7回すべて有効なときだけ週平均を作ります。日付欠落は最初〜最後のアーカイブの範囲内で記録し、欠落をまたぐ平均は作りません。GPVは日平均予報水温であり、原著の毎時同化画像とは異なる実験条件です。

雲マスクはL3Cの品質・欠損・陸地フラグから作る「雲判定後に有効SSTがない海域」で、独立した雲分類ではありません。未知の陸地フラグは有効海域として扱いません。偏差学習時には別の学習画像からこの雲マスクをランダムに選び、入力へ追加適用します。元の正解SSTと観測マスクは保持します。詳細は [論文対応変更](paper_revision.md) を参照してください。

学習・検証の分割は自動ではありません。指定した入力範囲をすべて一つのプールとして扱うので、評価期間が混ざらないよう入力ディレクトリを分けてください。平均の過去窓も含めて分割条件を設計してください。

## 学習

```bash
python3 -m src.train.train_average \
  --config config.yaml --data-root data/unified_sst

python3 -m src.train.train_anomaly \
  --config config.yaml --data-root data/unified_sst \
  --average-checkpoint checkpoints/average/latest.pth
```

共通ローダーが `assimilation_weekly` / `assimilation_sst` を供給するため、既存の敵対的損失の分岐が有効になります。論文の完全再現や精度を保証するものではありません。GPVとひまわりの日時・位置をペアにせず、独立にサンプリングします。初期実装ではその組み合わせもseedとサンプル番号で再現可能に固定します。

正規化は既存と同じ℃からの線形変換で、クリップしません。既定0〜35℃の範囲外は[-1,1]の範囲外になります。範囲を変える場合は前処理・学習両方のconfigを合わせてください。旧NPZ出力を指定した場合は `olds/` のローダーで従来どおり読み込み、GPVは付きません。旧GPVのNPZと新しいNPY形式を混在させないでください。

## 色彩構造などの拡張

前処理モジュールを変更せず、ローダーの**正規化後**のサンプルにフックを追加できます。

```python
# 例: src/datasets/color_features.py （必要になった時点で実装）
class AddColorStructure:
    def __call__(self, sample, context):
        # sample: Tensorの辞書（水温、マスク、GPVなど）
        # context: 位置、緯度経度、時刻、学習段階
        # sample['color_structure'] = ...
        return sample
```

```yaml
data:
  # 既存の他のdata設定は維持
  sample_transforms:
    - src.datasets.color_features:AddColorStructure
```

クラスは引数なしで生成し、`__call__(sample, context)` がTensor辞書を返します。順番に適用します。プログラムからは `create_unified_loader(..., transforms=[...])` でも渡せます。実際に色彩構造をモデルの入力や損失として使用する場合は、対応するモデル・学習側の変更も必要です。追加キーを返すだけで自動的に学習へ反映されるわけではありません。
