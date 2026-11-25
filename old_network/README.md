# 旧実装の保管先

現方式は「9時刻の観測平均＋最近傍補間」を基準SSTにしたアノマリー学習です。
旧テストは、このフォルダ内の `old_tests/` に保存しています。

| 保存先 | 内容 |
| --- | --- |
| `old_src/train/train_average.py` | 学習型週平均推定器の学習 |
| `old_src/models/average_generator.py` | 月平均と9時刻を入力する旧平均推定器 |
| `old_src/models/physical_discriminator.py` | 旧global/local PatchGAN識別器 |
| `old_src/datasets/` | 旧インデックス形式のローダー・合成データ |
| `old_src/preprocessing/GPV_preprocessing.py` | 日平均GPVの前処理 |
| `old_src/himawaridata/FTP-Ptree.py` | 旧FTP取得スクリプト |
| `old_preprocessing/` | 従来のolds配下を保存した前処理・固定パッチ用コード |
| `old_visualize/` | 旧フレーム可視化とGPV確認コード |

## 互換性

旧設定でmain.pyから平均推定を実行すると、移動先のモジュールが呼ばれます。
旧モデル名をsrc.models.generator/discriminatorから読み込む互換窓口は残しています。
現行の共通エンコーダ・デコーダ、損失、アノマリー学習・推論の互換分岐はsrcに残します。
JCOPEでも必要なバイリニア補間はsrc/preprocessing/interpolation.pyへ独立させました。
このフォルダのコードはsrcの共通処理を利用するため、このフォルダ単独では動きません。

旧平均推定を直接起動する場合は、プロジェクトのルートで次を使用します。

```bash
python3 -m old_network.old_src.train.train_average --help
python3 -m pytest -q old_network/old_tests
```

GPV_checks.pyは以前からシェル用の貼り付けスニペットを保存したファイルであり、Pythonとして直接実行する形式ではありません。
