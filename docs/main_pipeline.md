# 全工程の管理入口 `main.py`

前処理 → 平均推定学習 → 偏差補完学習 → パッチ推論を順に呼びます。元の処理は各モジュールに残し、main.pyは設定・実行順序・ログ・成果物の管理を担当します。前処理の完了を待ってから実行してください。

## 前処理済みデータでの実行

```bash
cd /workspace/sst-inpainting-code

# まず両学習段階を1バッチずつ実行し、1パッチ復元まで確認
python3 main.py \
  --data-root data/unified_sst \
  --run-dir outputs/smoke_001 \
  --smoke

# 本学習（前処理は再実行しない）
python3 main.py \
  --data-root data/unified_sst \
  --run-dir outputs/experiment_001
```

`--data-root` があれば既定で平均推定から開始します。平均推定のlatest.pthを偏差補完へ自動的に引き継ぎます。設定の既定はconfig.yamlの30エポック・バッチ32・CUDAです。`--device cpu`、`--epochs 10`、`--batch-size 4`、`--num-workers 2` で上書きできます。`--smoke` は1エポック・1バッチ・バッチサイズ1・worker0を優先し、本学習の代わりにはなりません。

推論入力の指定がなければ、**学習用索引の0番目のパッチを取り出した動作確認用の復元**を行います。`--sample-index 5` などで変更できます。これは未知データの評価でも、全海域の復元・合成でもありません。学習用GPVプールを利用できる統合データが必要です。

## 前処理からすべて実行

```bash
python3 main.py \
  --himawari-input /workspace/sst-inpainting-code/data_raw/Himawari_raw \
  --gpv-input /datasets/jmagpv/jpn \
  --run-dir outputs/experiment_from_raw
```

前処理結果は実験ディレクトリの `data/` に生成されます。`--patches-per-frame`、`--bounds SOUTH NORTH WEST EAST`、`--min-quality` も指定できます。smokeでも前処理の対象数は減らしません。少量の読み出し確認は、別途前処理CLIの `--max-frames` / `--max-archives` を使ってください（履歴不足の出力は学習できません）。

## 工程を限定する

工程名は `preprocess` → `average` → `anomaly` → `reconstruct` です。

```bash
# 2段階学習まで実行（推論しない）
python3 main.py --data-root data/unified_sst \
  --run-dir outputs/train_only --stop-after anomaly

# 既存の平均推定重みを使い、偏差補完から開始
python3 main.py --data-root data/unified_sst \
  --run-dir outputs/anomaly_only --start-at anomaly --stop-after anomaly \
  --average-checkpoint outputs/experiment_001/checkpoints/average/latest.pth

# 用意済みの正規化NPZを復元（この場合データローダーの入力は不要）
python3 main.py --run-dir outputs/inference_001 --start-at reconstruct \
  --average-checkpoint outputs/experiment_001/checkpoints/average/latest.pth \
  --anomaly-checkpoint outputs/experiment_001/checkpoints/anomaly/latest.pth \
  --inference-input data/inference_input.npz
```

省略した学習段階の重みは明示指定が必要です。外部チェックポイントはconfigを含む本プロジェクトの形式を使用し、温度範囲・時系列・パッチサイズ等を一致させてください。外部入力NPZは正規化済みで、既存推論コードの4キー（水温時系列、有効マスク、月平均、月平均マスク）が必要です。入力のサイズ・有限値・マスクを確認しますが、入力がどの温度範囲で正規化されたかまでは数値だけでは判定できないため、生成元設定を合わせてください。

途中工程からの開始は可能ですが、**中断エポックのoptimizer状態を復元するresumeではありません**。同じ実験ディレクトリへの再実行は拒否し、新しい名前を指定します。完了済み工程の重みを渡して後続工程だけを実行できます。

## 保存内容

```text
outputs/experiment_001/
├── config.yaml                  # 上書き設定を含む実行時設定
├── run.json                     # 工程、実行コマンド、完了/失敗状態
├── logs/
│   ├── average.log
│   ├── anomaly.log
│   └── reconstruct.log
├── checkpoints/
│   ├── average/latest.pth
│   └── anomaly/latest.pth
└── outputs/
    ├── preview_input.npz        # 自動生成した場合
    ├── preview_input.json       # 時刻・緯度経度・学習サンプル使用の記録
    └── reconstruction.npz
```

復元NPZの `pred_sst` と `pred_weekly` は正規化SSTです。℃にするには `(値+1)*(max_temp-min_temp)/2+min_temp` を使います。`pred_anomaly` は偏差の正規化値で、℃の偏差は `anomaly_range` を掛けます。

エラー時は後続を停止し、run.jsonに失敗工程を記録します。学習で非有限のエポック損失が検出された場合は保存前に停止します。推論結果のNaN/Infも検査します。元のNetCDF、tar.gz、前処理済みデータは削除しません。

人工雲マスク適用、学習/検証の自動分割、全海域推論・合成、色彩構造を使うモデルは今回の管理入口には追加していません。色彩構造などのsample_transformsを使う場合、自動プレビューは拒否します。対応済みの入力を明示するか学習段階で停止してください。既存推論モデルは追加特徴量を自動利用しません。
