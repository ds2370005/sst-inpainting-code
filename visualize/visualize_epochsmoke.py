#!/usr/bin/env python3
"""Plot a smoke-run patch preview (one training batch, not accuracy evaluation).

Example:
    python3 visualize/visualize_epochsmoke.py --run-dir outputs/smoke_001
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
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


def visualize(run_dir, output=None, input_path=None, dpi=150, data_root=None):
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
    export_context(run_dir, input_path, output, config, sst, mask, observed, reconstructed, data_root, dpi)
    return output



def panel_grid(path, panels, low, high, title, dpi, columns=3, binary=False):
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns*4.5, rows*4),
                             squeeze=False, constrained_layout=True)
    try:
        cmap = plt.get_cmap("gray_r" if binary else "turbo").copy()
        cmap.set_bad("lightgray")
        for ax, (label, values) in zip(axes.flat, panels):
            im = ax.imshow(np.ma.masked_invalid(values), vmin=low, vmax=high, cmap=cmap)
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            fig.colorbar(im, ax=ax, shrink=.8, label="mask" if binary else "deg C")
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        fig.suptitle(title)
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)
    print(f"Saved: {path}")


def export_context(run_dir, input_path, output, config, sst, mask, observed, reconstructed, data_root, dpi):
    """Describe the reconstruction input, never claim it was the optimization batch."""
    low, high = float(config["min_temp"]), float(config["max_temp"])
    prefix = output.with_suffix("")
    def destination(suffix):
        return prefix.with_name(prefix.name + suffix)
    meta_path = input_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    target = datetime.fromisoformat(meta["target_time"]) if "target_time" in meta else None
    panels = []
    for i in range(sst.shape[1]):
        stamp = target - timedelta(hours=int(config.get("time_interval_hours", 6))*(sst.shape[1]-1-i)) if target else None
        label = stamp.strftime("%Y-%m-%d %H:%M UTC") if stamp else f"Input frame {i}"
        values = np.where(mask[0,i]>0, (sst[0,i]+1)*(high-low)/2+low, np.nan)
        panels.append((label, values))
    panel_grid(destination("_input_sequence.png"), panels, low, high,
               "Reconstruction input SST sequence (not necessarily the smoke training batch)", dpi)
    panel_grid(destination("_missing_masks.png"),
               [(label, 1-mask[0,i]) for i,(label,_) in enumerate(panels)], 0, 1,
               "Input missing masks: black=missing, white=valid (includes any missing cause)", dpi, binary=True)
    valid = np.isfinite(observed) & np.isfinite(reconstructed)
    difference = np.where(valid, reconstructed-observed, np.nan)
    finite = difference[valid]
    summary = dict(purpose="reconstruction_input_inspection_not_held_out_evaluation",
                   actual_training_batch_verified=False,
                   input_path=str(input_path), target_time=meta.get("target_time"),
                   observed_valid_pixels=int(valid.sum()),
                   observed_pixel_mae_C=float(np.abs(finite).mean()) if finite.size else None,
                   observed_pixel_rmse_C=float(np.sqrt((finite**2).mean())) if finite.size else None)
    panel_grid(destination("_observed_comparison.png"),
               [("Observed latest SST", observed), ("Reconstructed SST", reconstructed)],
               low, high, "Observed-pixel comparison; no ground truth under missing pixels", dpi, columns=2)
    # Source context requires retained full-frame products and preview provenance.
    root = Path(data_root).expanduser() if data_root else Path(meta["data_root"]) if meta.get("data_root") else None
    if root is None or not (root / "dataset.json").is_file() or "sample_index" not in meta:
        print("Source overview/weekly mean skipped: supply --data-root and retain preview_input.json.")
        summary["source_context"] = "unavailable"
    else:
        manifest = json.loads((root / "dataset.json").read_text())
        record = manifest["samples"][int(meta["sample_index"])]
        if record["time"] != meta.get("target_time") or record["origin"] != meta.get("origin"):
            raise ValueError("Preview provenance does not match dataset.json")
        y,x = record["origin"]
        h,w = observed.shape
        source = np.load(root / record["frames"][-1] / "sst.npy", mmap_mode="r")
        original = np.asarray(source[y:y+h,x:x+w])
        if not np.allclose(original, observed, atol=1e-4, equal_nan=True):
            raise ValueError("Original SST crop does not match the preview input")
        lat = np.load(root / "lat.npy")
        lon = np.load(root / "lon.npy")
        from matplotlib.patches import Rectangle
        fig,ax = plt.subplots(figsize=(9,7), constrained_layout=True)
        try:
            cmap=plt.get_cmap("turbo").copy();cmap.set_bad("lightgray")
            im=ax.imshow(np.ma.masked_invalid(source), cmap=cmap, vmin=low, vmax=high)
            ax.add_patch(Rectangle((x-.5,y-.5),w,h,fill=False,edgecolor="red",linewidth=2))
            ax.set_title(f"Full preprocessed SST / red: preview crop\n{record['time']} UTC")
            ax.set_xlabel("Column");ax.set_ylabel("Row")
            fig.colorbar(im,ax=ax,label="deg C")
            fig.savefig(destination("_source_overview.png"),dpi=dpi)
        finally:
            plt.close(fig)
        history=np.stack([np.array(np.load(root/path/"sst.npy",mmap_mode="r")[y:y+h,x:x+w]) for path in record["history"][:7]])
        count=np.isfinite(history).sum(0)
        weekly=np.full((h,w),np.nan,np.float32)
        np.divide(np.nansum(history,axis=0),count,out=weekly,
                  where=count>=manifest['settings']['weekly_min_observations'])
        cloud_panels=[]
        for (label,_),path in zip(panels,record['frames']):
            cloud=np.load(root/path/'cloud_mask.npy',mmap_mode='r')[y:y+h,x:x+w]
            cloud_panels.append((label,cloud))
        panel_grid(destination("_cloud_masks.png"),cloud_panels,0,1,
                   "Stored cloud masks: black=cloud-screened missing ocean",dpi,binary=True)
        panel_grid(destination("_observed_weekly.png"),[("Observed weekly mean",weekly)],low,high,
                   "Observed mean at the same crop (not the predicted weekly mean)",dpi,columns=1)
        summary.update(source_context="verified_preprocessed_source",origin=[y,x],
                       latitude_range=[float(lat[y:y+h].min()),float(lat[y:y+h].max())],
                       longitude_range=[float(lon[x:x+w].min()),float(lon[x:x+w].max())])
    destination("_inspection.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("Actual smoke optimization batch identity is not verified by these figures.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/smoke_001")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    parser.add_argument("--input", type=Path, help="Override preview_input.npz")
    parser.add_argument("--data-root", type=Path, help="Unified data root override for source context")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    try:
        visualize(args.run_dir, args.output, args.input, args.dpi, args.data_root)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
