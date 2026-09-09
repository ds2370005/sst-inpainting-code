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

## Train from preprocessed Himawari patches

First create NPZ patches with `src/himawaridata/crop_nc_to_1650.py`. Then train
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
