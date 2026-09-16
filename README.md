# SST Anomaly Inpainting

**Reproduction status:** runnable baseline, not a complete reproduction of the
paper. See [the reproduction audit](docs/reproduction_audit.md) for verified
features, methodological differences and remaining work. Real anomaly training
uses a frozen average-generator checkpoint; random cloud masks are not yet applied.


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

## Train from single-observation products

The training dataset auto-detects the `frames/` directory. Pass the parent
`data/frames256`, not its `frames/himawari` child:

```bash
python -m src.train.train_average --config config.yaml --data-root data/frames256
python -m src.train.train_anomaly --config config.yaml --data-root data/frames256 \
  --average-checkpoint checkpoints/average/latest.pth
```

For each target, the loader selects the same grid cell at exact offsets
`[-48,-42,-36,-30,-24,-18,-12,-6,0]` hours by default. It reads those nine
observations in oldest-to-newest order and the target's separate average file.
`data.time_steps` and `data.time_interval_hours` control these offsets.
Outputs preserve the previous model interface `[1,T,H,W]` per sample.
Anomaly targets are explicitly associated with the target observation.

Targets lacking a required frame or average file are excluded. Targets above
`data.max_cloud_fraction` (default 0.5, fraction of **all** patch pixels), or
with no valid observed SST, are excluded. Cloudy historical frames are retained.
Selection counts are printed at training startup. Metadata, array shapes and
coordinate alignment are checked while reading; malformed/misaligned products
raise errors rather than being silently combined. SST/averages are normalized
using configured temperature bounds, invalid inputs are filled with zero, and
validity masks exclude cloud and non-finite SST pixels. An all-invalid historical
frame is allowed. No assembled volumes are written to disk.

Only Himawari is supported for now (`data.satellite: himawari`).
Existing volume visualization scripts still expect legacy NPZ products.

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
  --data-root src/himawaridata/himawari_sst_patches256 \
  --average-checkpoint checkpoints/average/latest.pth
```

Current preprocessing outputs do not contain data-assimilation SST fields.
Therefore these real-data commands train with reconstruction losses only and
report zero discriminator/adversarial loss. The adversarial physical-model loss
requires a data-assimilation loader integration. Exact observation/assimilation
location-time pairs are not required by the paper.

## Inference units

`src.infer.reconstruct` accepts preassembled, normalized volume NPZ inputs,
not the raw frame products. `pred_weekly` and `pred_sst` use SST normalization;
`pred_anomaly` is a unit anomaly in [-1,1]. Reconstructed normalized SST is
`pred_weekly + pred_anomaly * 2 * anomaly_range / (max_temp - min_temp)`.
Use the training config for inference. Convert final SST to Celsius with
`(pred_sst + 1) * (max_temp - min_temp) / 2 + min_temp`.

## Connecting the two training stages

Train the average generator first, then pass its checkpoint with
`--average-checkpoint`. Real-data anomaly training requires this argument.
Stage one is loaded once, set to eval mode, and frozen. Every batch feeds SST,
validity masks, monthly SST and monthly masks into it under no_grad; its predicted
weekly SST is the anomaly baseline and is used in the final SST sum. The observed
weekly SST is not used as the baseline. Only the anomaly generator (and, when
available, its discriminator) is updated. The checkpoint path is recorded in the
anomaly checkpoint config. Available temperature/time/satellite metadata is checked
for compatibility; bare state_dict checkpoints cannot provide that metadata.
Synthetic debug training without a checkpoint retains its observed-average path.
Existing frame products require no regeneration. Artificial cloud masking remains
separate unfinished work.
