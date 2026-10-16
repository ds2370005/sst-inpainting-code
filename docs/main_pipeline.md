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
├── run.json                     # 工程、実行コマンド、完了/失敗状態、実行時間
├── timings.csv                  # 全体・各工程の秒数と開始/終了日時
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

## 復元プレビューの入力と元画像を可視化

```bash
python3 visualize/visualize_epochsmoke.py \
  --run-dir outputs/smoke_001 \
  --data-root data/unified_sst
```

既存の5枚比較に加えて、入力時系列SST、欠損マスク、観測との比較、元の前処理済み全域SSTと切り取り枠、保存済み雲マスク、観測週平均、確認用JSONを出力します。元画像関連には `preview_input.json` と前処理済みデータが必要です。元のNetCDF全域ではなく、前処理対象海域の水温を表示します。

これらは**復元入力サンプル**の図です。smoke学習はシャッフルされた先頭1バッチを使うため、自動プレビューの索引0番と一致するとは限りません。今回の図は実際の最適化バッチの同一性を保証しません。観測画素上の誤差は動作確認用で、雲下の精度や未知データでの性能評価ではありません。

## 実行時間と評価図

今後の `main.py` 実行では、全体および各工程の `elapsed_seconds` を `run.json` に保存し、同じ情報を `timings.csv` にまとめます。単調増加時計で計測した実時間で、子プロセスの起動、データ読み込み、検査、保存を含みます。実験開始後の通常の例外や Ctrl+C による中断時にも記録します。開始前の引数・設定検査で失敗した場合や強制終了・電源断では保存を保証しません。過去の実行の正確な所要時間は復元できません。

学習・推論後、別スクリプトで保存済み結果を評価できます（再学習しません）。

```bash
python3 visualize/evaluate_reconstruction.py --run-dir outputs/smoke_001
```

`evaluation/` に以下を出力します。

- `metrics.json`: 有効画素数、MAE、RMSE、バイアス（推定−参照）、R²、Pearson相関係数。
- `regression.png`: 参照と推定の画素密度図、誤差ヒストグラム、水温別の誤差。
- `training_losses.png` / `.csv`: 平均推定・偏差推定の学習損失。ログがなければ図は省略。検証損失ではありません。
- `runtimes.png`: 工程別所要時間。時間記録のない過去の実行では省略。

標準では `outputs/preview_input.npz` の最新時刻の観測有効画素と比較します。**学習プレビューの整合性確認であり、雲下の復元精度や未知データの性能評価にはなりません。** 独自入力で推論した場合は、そのNPZを `--input /path/to/input.npz` で指定してください。

SST推定は回帰なのでROCは標準では生成しません。高水温の判別を評価したい場合のみ、正例の閾値（℃）を明示してください。例えば18℃以上を正例にする場合（18℃は指定方法の例です）:

```bash
python3 visualize/evaluate_reconstruction.py --run-dir outputs/smoke_001 \
  --temperature-threshold 18 --output-dir outputs/smoke_001/evaluation_18C
```

追加で `classification.png`（混同行列・ROC・PR曲線）と `roc_pr.csv` を保存し、JSONにROC-AUC、Average Precision、Accuracy、Precision、Recall、F1を追加します。ROC/PRのスコアは予測水温そのもので、確率ではありません。混同行列は予測水温にも同じ閾値を適用します。片方のクラスしかない場合、ROC-AUC/APは未定義として `null` とし、曲線CSVを生成しません。定数データの相関係数など未定義の指標も `null` です。これは雲検出ROCではありません。

独立した正解データがある場合は `--truth /path/to/truth.npz` を指定できます。必要なキーは `sst_celsius`（℃）と `valid_mask`（1=評価対象、0=除外）、形状は `[H,W]` または `[B,1,H,W]` です。予測と同じ時刻・位置・格子・順序に事前整合させてください。形状は検査しますが、座標整合や学習との独立性は自動検証しません。GPVを参照に使った場合も、観測による真値評価と同一には扱えません。

評価対象は保存された `reconstruction.npz` の範囲であり、全データセットを自動評価する処理ではありません。レポートの混在を避けるため、既存の空でない出力先への再出力は拒否します。再評価時は新しい `--output-dir` を指定してください。

## ネットワーク出力のヒストグラム診断

```bash
python3 visualize/visualize_sst_histograms.py \
  --run-dir outputs/smoke_001 \
  --data-root data/unified_sst
```

再学習せず、保存された1サンプルの復元結果を `histograms/` に出力します。`--data-root` は省略できます。指定した場合は `preview_input.json` と `dataset.json` の対応および元のSST切り取り値を検査し、観測週平均を再計算します。入力NPZを別途指定する場合は `--input` を使ってください。

- `comparison.png`: 最新時刻の観測SST・推定週平均・復元SST、および「観測SST−推定週平均」・推定アノマリーの分布。同じ観測有効画素で比較します。
- `average_comparison.png`: 観測月平均と推定週平均、さらにデータルート指定時は観測週平均と推定週平均。それぞれ参照平均が有効な同じ画素を使います。月平均と週平均は異なる期間の平均です。
- `prediction_zoom.png`: 推定週平均・推定アノマリー・復元SSTの値域を個別に自動拡大。ほぼ定数の出力に残る微小な変動を確認できます。
- `statistics.csv` / `.json`: 画素数、最小・最大・平均・標準偏差・最大−最小・1/50/99パーセンタイル。JSONには全格子の出力統計と非有限値数も保存します。全格子は陸域・欠測位置を含み得るため、精度指標には使いません。
- `histograms.csv`: 各ビンの境界・画素数・割合。各比較図内で共通ビンを使い、有限値の全範囲を含めます。縦軸はビンごとの画素割合（%）です。

SSTは℃、アノマリーは温度差の℃に戻してから集計し、マスク0の埋め値は参照ヒストグラムに含めません。`--bins 80` でビン数を調整できます。出力先が空でなければ拒否するので、再描画には新しい `--output-dir` を指定してください。

対象は復元入力と同じパッチであり、全元データや実際のsmoke最適化バッチの分布を保証する図ではありません。「観測SST−推定週平均」は推定平均を基準にした残差で、独立した真のアノマリーではありません。ヒストグラムの一致だけでは空間構造の再現や雲下の精度は判断できません。
