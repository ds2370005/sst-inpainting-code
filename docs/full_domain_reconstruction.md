# 全域反復復元

## 論文にある手順と実装上の解釈

参照：Hirahara et al., *Cloud-Free Sea-Surface-Temperature Image Reconstruction From Anomaly Inpainting Network*, DOI 10.1109/TGRS.2021.3111649、IV-C「From Pred-W-Ave」、PDF 7〜8ページ。論文内に実装コードは掲載されていないため、文章の3段階手順を実装した。

1. 現在の週平均画像で欠損の少ない256×256パッチをランダムに選ぶ。
2. **同じ位置の元の時空間SSTと月平均**を平均生成器に入力する。
3. 週平均画像の該当パッチを予測結果で置き換え、次の候補を再計算する。
4. 全海域画素が一度以上ネットワークで復元されたら、復元週平均と元の時空間SSTを偏差生成器に全域一括で入力する。週平均＋偏差で最終SSTを得る。

`src/infer/full_domain.py` に実装。推論はeval/no_grad。月平均・時空間SSTを予測値で書き換えたり、出力を同じ入力に再帰的に与えたりはしない。更新するのは週平均の復元キャンバスと候補選択用の有効性。

原著には以下の詳細は明記されていないため、設定・ログに残す実装上の選択とした。

- **推論時の閾値**：週平均の欠損海域画素率が50%以下。学習の最新フレーム雲量50%とは別の条件。原著が推論にも50%を使ったという主張ではない。
- 分母はパッチ内の既知の海域画素数。週平均がNaNの海域を欠損と数える。陸地・不明な海岸画素は除外する。海域判定は対象時刻のvalid_maskまたはcloud_mask。
- 候補位置は既定128画素間隔の重複格子。右端・下端にも候補を追加。候補からseed付き乱数で選ぶ。`--weekly-stride`で調整できる。
- 予測はパッチ内の海域全体へ書き戻す。既存の観測週平均も置き換え、重複は後勝ち。原著の「replace」に対応する解釈で、ブレンディングはしていない。
- 毎回、未復元海域を少なくとも1画素含む候補だけを選ぶ。元々観測があるだけでは「復元済み」としない。
- 適格候補がない、または最大反復数に達した場合は失敗扱い。閾値を勝手に緩めたり、月平均で埋めて完了扱いにしたりしない。途中の週平均と未復元マスクは保存し、偏差推定は実行しない。

海岸を含むパッチは学習時の全海域パッチとは条件が異なる。また、後勝ちによる継ぎ目が生じる可能性がある。これらは出力の評価で確認する必要がある。

## 入力と既存データ

全域とは、前処理済みデータに保存した緯度経度領域。元の衛星ファイル全域とは限らない。`dataset.json` の `himawari_frames` を使い、学習用ランダムパッチ索引には依存しない。学習サンプルに含まれていない日時も、必要な履歴があれば推論できる。

指定時刻を含む過去29日の同時刻、計30枚から月平均（5観測以上）・週平均（直近7枚、3観測以上）を計算する。9時刻入力は6時間間隔で過去48時間から指定時刻まで。必要ファイルの不足はエラーにする。平均の最低観測数は前処理manifestの設定を使用する。

生成器の正方形制約を外し、長方形にも対応した。全域偏差推定では縦横を8の倍数まで右・下に端値でパディングし、推論後に元のサイズへ戻す。モデルのパラメータ形状は変えていないため、**hirahara_tables_v2の既存重みを再利用できる**。それ以前の重みは引き続き拒否する。

全域偏差推定は論文の手順に合わせて一括実行するため、パッチ推論よりGPUメモリを使う。OOM時に自動タイル処理へ変更はしない。必要なら新しい出力先で `--device cpu` を明示する。

## 保存済み重みで全域推論のみ実行

リポジトリのルートから実行する。日時は例で、データと30日履歴が存在するUTC日時に変更する。新しい実験フォルダを指定する。

```bash
python3 main.py \
  --config outputs/paper_v2_epoch1_002/config.yaml \
  --data-root /workspace/sst-inpainting-code/data/unified_sst \
  --run-dir /workspace/sst-inpainting-code/outputs/full_domain_001 \
  --start-at reconstruct \
  --average-checkpoint outputs/paper_v2_epoch1_002/checkpoints/average/latest.pth \
  --anomaly-checkpoint outputs/paper_v2_epoch1_002/checkpoints/anomaly/latest.pth \
  --full-domain-time 2025-04-30T00:00:00 \
  --weekly-max-missing 0.5 \
  --weekly-stride 128 \
  --weekly-max-iterations 10000
```

今後、学習後に全域推論まで一括実行する場合は、通常の `main.py` 学習コマンドに `--full-domain-time 日時` を加える。指定しない場合は従来の1パッチプレビューのまま。`--inference-input` とは併用しない。学習サンプルの雲量選別条件は今回変更していない。

独立した入口 `python3 -m src.infer.full_domain --help` も使用できる。こちらでは同じパラメータが `--target-time`、`--max-missing`、`--stride`、`--max-iterations` となる。

## 出力

`outputs/full_domain_001/outputs/full_domain/` に保存する。

- `run.json`：対象UTC日時、入力時刻、使用重み、選択条件、全反復の位置・選択前欠損率・新規復元画素数、残り画素数、状態、実行時間。
- `weekly_reconstruction.npz`：観測週平均・月平均、更新後週平均キャンバス、ネットワーク復元済み週平均、復元済みマスク、海域マスク、緯度経度。未完了時も保存する。
- `full_reconstruction.npz`：完了時のみ。`predicted_weekly_celsius`、`predicted_anomaly_celsius`、`reconstructed_sst_celsius`、`observed_sst_celsius`、`ocean_mask`、`latitude`、`longitude`。

NPZの水温は**すでに℃**、海域外はNaN。従来のパッチ用 `reconstruction.npz` の正規化配列とは別形式。従来のパッチ評価・ヒストグラムツールへそのまま渡さない。

可視化：

```bash
python3 visualize/visualize_full_domain.py \
  --input-dir outputs/full_domain_001/outputs/full_domain
```

成功時は観測SST・復元週平均・推定偏差・復元SSTを全域表示する。反復が未完了なら観測週平均と復元済み部分を表示する。図は分布確認用であり、ブイ観測との照合や独立テスト期間の精度評価ではない。

## 検証範囲

合成データで、書き戻しによる候補の増加、全海域の被覆、端の被覆、重複の後勝ち、陸域除外、再現可能な乱数、候補なし・反復上限時の未完了処理、履歴平均、長方形の実生成器forward、全域偏差のパディングと単位換算、保存とメインからの呼出しを検査した。実データでの全域推論・学習は実行していない。
