# 旧GPVを使わない再作成手順

加工済みの `unified_sst` が利用できない場合、ひまわり生データから再作成する。`src.preprocessing.pipeline` の `--gpv-input` を省略すると、旧GPVの読み出し・変換とecCodesの読み込みを行わない。ひまわりの処理・座標・サンプル索引を保存する。互換性のため空の `gpv` / `gpv_pools` 欄は残すが、学習では外部のJCOPEプールを指定する。

```bash
python3 -m src.preprocessing.pipeline \
  --himawari-input /data2/itoh/himawari/data_raw \
  --output /data2/itoh/datas/unified_sst \
  --config config.yaml
```

出力先は新規または空である必要がある。入力配下の全NetCDFを対象とするため、旧実験より生データが増えていれば、容量・サンプル数・学習時間も増える。以前の23,788サンプルと同じとは限らない。学習・検証の分割はこの変更では追加していない。`run.json` の `satellite_ready` はひまわりサンプルの有無を示す。`training_ready=false` は内蔵GPVプールがないためであり、外部JCOPEプールの準備完了後に学習できる。

## JCOPEの完全性確認

```bash
python3 check_jcope_download.py \
  --root /data2/itoh/JCOPE-T \
  --start 2024-08-01 --end 2025-12-31 \
  --output /workspace/sst-inpainting-code/outputs/jcope_check_001 \
  --deep
```

日別ファイルの存在、24時刻と日付・順序、深度0m、全24時間の水温の読み出しを確認する。陸域の欠損は異常としないが、全画素欠損の時刻は報告する。全ファイルを読むため時間がかかる。`--deep` を外すとヘッダー・時間座標中心の確認になる。提供元のチェックサムとの照合ではなく、すべての値の科学的妥当性を保証するものでもない。問題を `report.json` に保存し、問題があれば終了コード1になる。

## 座標変換

ひまわり再作成が成功し、JCOPE検査の問題を解決してから行う。

```bash
python3 -m src.preprocessing.JCOPE_preprocessing \
  --input /data2/itoh/JCOPE-T \
  --satellite-root /data2/itoh/datas/unified_sst \
  --output /data2/itoh/datas/JCOPE-T_aligned \
  --start 2024-08-01 --end 2025-09-30
```

検査は保有全期間（2025年12月まで）、補間は当初の学習用予定期間（2025年9月まで）である。加工済みデータはDATA2、検査レポート・可視化・学習結果はワークスペースに保存する。サーバーで実行する前に `df -h /data2/itoh/datas` で別ディスクを参照していることを確認する。
