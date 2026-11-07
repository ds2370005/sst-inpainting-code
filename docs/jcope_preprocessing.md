# JCOPE-T 1ksをひまわり格子へ揃える

`src/preprocessing/JCOPE_preprocessing.py` は、日別NetCDFの `TT(time, depth, lat, lon)` から深度0mを選び、実際の緯度・経度に基づくバイリニア補間を行う。時間平均や学習は行わず、毎時データを維持する。参照するひまわり前処理出力の `lat.npy` と `lon.npy` を出力へそのまま保存するため、画素数だけを合わせる画像縮小とは異なる。

## サーバーでの実行

プロジェクトのルートで実行する。原本のNetCDFは変更しない。出力先は新規または空のディレクトリが必要。

```bash
cd /workspace/sst-inpainting-code
python3 -m src.preprocessing.JCOPE_preprocessing \
  --input /data2/itoh/JCOPE-T \
  --satellite-root /workspace/sst-inpainting-code/data/unified_sst \
  --output /data2/itoh/JCOPE-T_aligned \
  --start 2024-08-01 --end 2025-09-30
```

最初に小規模で確認する場合は、別の出力先を指定する。

```bash
python3 -m src.preprocessing.JCOPE_preprocessing \
  --input /data2/itoh/JCOPE-T \
  --satellite-root /workspace/sst-inpainting-code/data/unified_sst \
  --output /data2/itoh/JCOPE-T_aligned_check \
  --start 2024-08-01 --end 2024-08-01 --max-frames 2
```

## 補間と検査

- `TT` の次元順序、単位℃、深度0mを検査する。
- `_FillValue`・`missing_value` をnetCDF4でマスクし、NaNに変換する。
- 補間に必要な近傍画素に欠損があれば出力もNaNとする。重みゼロの隣接点の欠損は影響させない。
- JCOPEの座標範囲外はNaNとし、外挿しない。陸域を無理に埋めない。
- ひまわり座標の昇順・降順を維持する。JCOPE側との緯度方向の違いに対応する。
- 時刻はNetCDFの時間座標から取得し、重複と正時以外を拒否する。欠けた時刻は補間せず、`simulation.json` の `gaps` に内部の時刻間隔の欠落を記録する。指定期間の先頭・末尾が揃っていることまで保証する検査ではない。
- 最初に日別ファイル名 `TT_YYYYMMDD.nc` で対象期間を絞り、その期間のファイルの時間座標だけを読み、水温は1時間分ずつ処理する。24時間分を一度に読み込まない。ただし補間用の配列も使うため、実メモリ使用量は元の1時間分より多くなる。

## 出力

```text
JCOPE-T_aligned/
  lat.npy
  lon.npy
  simulation.json
  run.json
  frames/20240801T000000/
    sst.npy
    metadata.json
  ...
```

`sst.npy` は℃のfloat32配列。`metadata.json` に元ファイル、時刻番号、製品情報、有効画素数を保存する。原本が圧縮NetCDFでも出力NPYは非圧縮となる。1650×1650なら1時間約10.9MB、2024年8月〜2025年9月の426日分なら約111GB（十進表記）の空き容量が必要。週平均等を後で作成する場合は別途容量が必要。

この出力は `python3 -m src.preprocessing.hourly_assimilation --source ...` の入力となる。GAN学習用の `--assimilation-root` には、この後に作成する学習用プールを指定する。

人工的なNetCDFで、座標・向き・既知の補間値・欠損・範囲外・深度・時刻重複をテスト済み。サーバーの実データではまだ実行していない。

## 対象外ファイルでのHDFエラーへの修正

以前は期間外のファイルも時間座標の事前検査で開いていた。現在は名前の日付で先に絞るため、8月1日だけの実行で翌年1月12日を開くことはない。選択したファイルは内部時刻と名前の日付の一致も検証する。対象期間内のファイルを開けない場合は、該当パスを明示して停止し、自動的に欠損としてスキップしない。HDFエラーだけで破損と断定せず、単独で開けるか、転送元とのサイズ・チェックサムの一致などを確認する。`--max-frames` は出力上限であり、指定期間内の時間座標検査は全対象ファイルについて行う。
