# SST Anomaly Inpainting

PyTorch implementation scaffold for the two-stage SST reconstruction method:

1. Average Estimation Generator
2. Anomaly Inpainting Generator

The repository includes Himawari NetCDF preprocessing, direct NPZ-backed
training datasets, two-stage generators, discriminators, losses, and inference.

## Synthetic Forward Check

```bash
pytest tests/test_generators_forward.py
```

## Single-observation preprocessing (default)

```bash
python -m src.himawaridata.crop_nc_to_1650 \
  data_raw/Himawari_raw data/frames256 --max-frames 1 --skip-temporal-averages
```

Remove `--max-frames 1` to process all observation timestamps. Remove
`--skip-temporal-averages` to also calculate same-hour 7/30-day averages.
The latter requires complete 30-day history, but missing history never blocks
saving the observation frames.

Output layout:

```text
data/frames256/
  frames/himawari/YYYYmmddHHMMSS_rXX_cXX.npz
  averages/himawari/YYYYmmddHHMMSS_rXX_cXX.npz
```

Each observation NPZ (format_version=2) holds one `sst` float32 array and one
`cloud_mask` uint8 array, both `[256,256]`, plus `timestamp`, `satellite`, `lat`,
`lon`, source path, grid position, quality threshold, cloud fraction and valid
fraction. SST is Celsius, missing source SST stays NaN, and cloud mask is
1 for cloudy/missing ocean and 0 for clear ocean or land. Cloud fraction uses
all patch pixels as its denominator. All patches are saved, including cloudy
or entirely invalid patches, so future loaders can choose targets separately
from their historical context. No duplicate cloud-mask pool is generated.

Averages are separate NPZ files with `weekly_average` `[1,H,W]` and
`monthly_average` `[1,1,H,W]`, timestamp and coordinates. They do not contain
observational SST frames. Coordinates are checked across observation times.
Existing compatible frame files are skipped; `--overwrite` regenerates them.
Use a new output directory if input data or processing settings change.

**The current training loader and volume visualization scripts still expect
legacy nine-frame NPZ files. The new frame format is a preprocessing-only
migration; temporal assembly in the loader is not yet implemented.**
Satellite metadata enables later extension, but this preprocessor reads only
Himawari; GCOM-C reprojection and temporal matching are not implemented.

## Train from legacy preprocessed Himawari patches

First create legacy NPZ patches with an explicit output directory:

```bash
python -m src.himawaridata.crop_nc_to_1650 data_raw/Himawari_raw \
  src/himawaridata/himawari_sst_patches256 --format legacy-volumes
```

Then train
the average and anomaly stages from the output directory:

```bash
python -m src.train.train_average \
  --config config.yaml \
  --data-root src/himawaridata/himawari_sst_patches256

python -m src.train.train_anomaly \
  --config config.yaml \
  --data-root src/himawaridata/himawari_sst_patches256
```

Current preprocessing outputs do not contain data-assimilation SST fields.
Therefore these real-data commands train with reconstruction losses only and
report zero discriminator/adversarial loss. The adversarial physical-model loss
becomes available after aligned data-assimilation fields are added.
