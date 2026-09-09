"""Visualize the crop used by ``crop_nc_to_1650.py``.

Example:
    python -m src.himawaridata.visualize_crop input.nc crop_overview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .crop_nc_to_1650 import (
    DEFAULT_LAT_MAX,
    DEFAULT_LAT_MIN,
    DEFAULT_LON_MAX,
    DEFAULT_LON_MIN,
    DEFAULT_PATCH_SIZE,
    build_fixed_grid,
    read_sst_and_mask_patch,
    study_area_slices,
)


def create_crop_figure(input_path: Path, output_path: Path) -> None:
    """Save a three-panel overview of the study area, grid, and one patch."""
    lat_slice, lon_slice, _, _ = study_area_slices(
        input_path,
        DEFAULT_LAT_MIN,
        DEFAULT_LAT_MAX,
        DEFAULT_LON_MIN,
        DEFAULT_LON_MAX,
    )
    grid = build_fixed_grid(lat_slice, lon_slice, DEFAULT_PATCH_SIZE)

    # Read only the 1650 x 1650 study area, not the complete satellite image.
    sst, _, lat, lon = read_sst_and_mask_patch(
        input_path, lat_slice, lon_slice, min_quality_level=4
    )
    grid_top = min(item[4] for item in grid)
    grid_left = min(item[5] for item in grid)
    grid_size = DEFAULT_PATCH_SIZE * 6
    cropped = sst[
        grid_top : grid_top + grid_size,
        grid_left : grid_left + grid_size,
    ]
    # The center patch is used as an easy-to-see example.
    patch_row, patch_column = 2, 2
    y = patch_row * DEFAULT_PATCH_SIZE
    x = patch_column * DEFAULT_PATCH_SIZE
    patch = cropped[y : y + DEFAULT_PATCH_SIZE, x : x + DEFAULT_PATCH_SIZE]

    finite = sst[np.isfinite(sst)]
    if finite.size == 0:
        raise ValueError(f"No valid SST values found in {input_path}.")
    vmin, vmax = np.percentile(finite, [2, 98])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    image_options = dict(cmap="turbo", origin="lower", vmin=vmin, vmax=vmax)

    axes[0].imshow(sst, **image_options)
    axes[0].add_patch(
        Rectangle(
            (grid_left, grid_top),
            grid_size,
            grid_size,
            fill=False,
            edgecolor="white",
            linewidth=2,
        )
    )
    axes[0].set_title("Study area: 1650 x 1650")

    axes[1].imshow(cropped, **image_options)
    for position in range(0, grid_size + 1, DEFAULT_PATCH_SIZE):
        axes[1].axhline(position - 0.5, color="white", linewidth=0.6)
        axes[1].axvline(position - 0.5, color="white", linewidth=0.6)
    axes[1].add_patch(
        Rectangle(
            (x - 0.5, y - 0.5),
            DEFAULT_PATCH_SIZE,
            DEFAULT_PATCH_SIZE,
            fill=False,
            edgecolor="magenta",
            linewidth=3,
        )
    )
    axes[1].set_title("Centered crop: 1536 x 1536 (6 x 6 grid)")

    image = axes[2].imshow(patch, **image_options)
    axes[2].set_title("Example patch: 256 x 256")
    fig.colorbar(image, ax=axes, label="Sea surface temperature (degrees C)")

    for axis in axes:
        axis.set_xlabel("pixel x")
        axis.set_ylabel("pixel y")

    fig.suptitle(
        f"Crop overview: {input_path.name}\n"
        f"{lat.min():.1f}-{lat.max():.1f} N, {lon.min():.1f}-{lon.max():.1f} E"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input Himawari NetCDF file")
    parser.add_argument("output", type=Path, help="Output PNG file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_crop_figure(args.input, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
