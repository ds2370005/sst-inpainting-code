# 旧前処理の保管

固定グリッド版のひまわり・GPV前処理と旧NPZ専用ローダーを元の構成に近い形で移動しました。アルゴリズムは保存し、移動に伴うimportのみ更新しています。

- himawaridata/crop_nc_to_1650.py
- himawaridata/frame_preprocessing.py
- himawaridata/visualize_crop.py
- data/preprocess_gpv.py
- datasets/frame_index.py
- datasets/himawari_patch_dataset.py

旧ひまわりCLIは `python -m olds.himawaridata.crop_nc_to_1650 ...`。
旧GPV CLIは `python olds/data/preprocess_gpv.py ...`。
新しい前処理は `python -m src.preprocessing.pipeline` を使用してください。
