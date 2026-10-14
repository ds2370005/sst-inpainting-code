#!/usr/bin/env python3
"""Plot a smoke-run patch preview (one training batch, not accuracy evaluation).

Example:
    python3 visualize/visualize_１epochsmoke.py --run-dir outputs/smoke_001
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def plane(values, name):
    """Select the first batch/channel from a reconstructed field."""
    values = np.asarray(values)
    if values.ndim == 4 and values.shape[0] > 0 and values.shape[1] == 1:
        return values[0, 0]
    raise ValueError(f"{name}: expected [B,1,H,W], got {values.shape}")


def input_volume(values, name):
    values = np.asarray(values)
    if values.ndim == 5 and values.shape[0] > 0:
        values = values[0]
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError(f"{name}: expected [1,T,H,W] or [B,1,T,H,W], got {values.shape}")
    return values


def visualize(run_dir, output=None, input_path=None, dpi=150):
    run_dir = Path(run_dir).expanduser().resolve()
    with (run_dir / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["data"]
    low, high = float(config["min_temp"]), float(config["max_temp"])
    anomaly_range = float(config["anomaly_range"])
    if not np.isfinite([low, high, anomaly_range]).all() or high <= low or anomaly_range <= 0:
        raise ValueError("Invalid temperature normalization settings")

    def celsius(values):
        return (values + 1) * (high - low) / 2 + low

    input_path = Path(input_path).expanduser() if input_path else run_dir / "outputs/preview_input.npz"
    with np.load(input_path, allow_pickle=False) as archive:
        sst = input_volume(archive["sst_volume"], "sst_volume")
        mask = input_volume(archive["mask_volume"], "mask_volume")
        monthly = input_volume(archive["monthly_average"], "monthly_average")
        monthly_mask = input_volume(archive["monthly_mask"], "monthly_mask")
        if sst.shape != mask.shape or monthly.shape != monthly_mask.shape:
            raise ValueError("SST/mask shape mismatch")
        if monthly.shape[1] != 1 or sst.shape[1] < 1:
            raise ValueError("Invalid time dimensions")
        observed = np.where(mask[0, -1] > 0, celsius(sst[0, -1]), np.nan)
        monthly = np.where(monthly_mask[0, 0] > 0, celsius(monthly[0, 0]), np.nan)
    with np.load(run_dir / "outputs/reconstruction.npz", allow_pickle=False) as archive:
        weekly = celsius(plane(archive["pred_weekly"], "pred_weekly"))
        anomaly = plane(archive["pred_anomaly"], "pred_anomaly") * anomaly_range
        reconstructed = celsius(plane(archive["pred_sst"], "pred_sst"))
    panels = [
        ("Observed SST (latest frame)", observed),
        ("Observed monthly mean", monthly),
        ("Predicted weekly mean", weekly),
        ("Predicted anomaly", anomaly),
        ("Reconstructed SST", reconstructed),
    ]
    if any(values.shape != observed.shape for _, values in panels):
        raise ValueError("Input and reconstruction spatial shapes differ")
    output = Path(output).expanduser() if output else run_dir / "outputs/smoke_preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), constrained_layout=True)
    try:
        for index, (ax, (title, values)) in enumerate(zip(axes, panels)):
            is_anomaly = index == 3
            cmap = plt.get_cmap("coolwarm" if is_anomaly else "turbo").copy()
            cmap.set_bad("lightgray")
            image = ax.imshow(
                np.ma.masked_invalid(values), origin="upper", cmap=cmap,
                vmin=-anomaly_range if is_anomaly else low,
                vmax=anomaly_range if is_anomaly else high,
            )
            ax.set_title(title)
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            fig.colorbar(image, ax=ax, label="deg C", shrink=0.8)
        fig.suptitle("Smoke test: patch preview, not accuracy evaluation (gray = missing)")
        fig.savefig(output, dpi=dpi)
    finally:
        plt.close(fig)
    print(f"Saved: {output.resolve()}")
    for title, values in panels:
        valid = values[np.isfinite(values)]
        if valid.size:
            print(f"{title}: min={valid.min():.3f}, max={valid.max():.3f}, mean={valid.mean():.3f} deg C")
        else:
            print(f"{title}: no valid pixels")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/smoke_001")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    parser.add_argument("--input", type=Path, help="Override preview_input.npz")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    try:
        visualize(args.run_dir, args.output, args.input, args.dpi)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
