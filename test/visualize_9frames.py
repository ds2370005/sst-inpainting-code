from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("data/patches256")
paths = sorted(root.glob("*.npz"))
if not paths:
    raise SystemExit("可視化するNPZがありません。")

best_path = None
best_count = 0

for candidate in paths:
    with np.load(candidate, allow_pickle=False) as archive:
        values = np.asarray(archive["sst_volume"])
        count = int(np.isfinite(values).sum())

    if count > best_count:
        best_path = candidate
        best_count = count

if best_path is None:
    raise SystemExit("全パッチに有効なSSTがありません。")

path = best_path
with np.load(path, allow_pickle=False) as archive:
    sst = np.asarray(archive["sst_volume"], dtype=float)

if sst.shape != (1, 9, 256, 256):
    raise SystemExit(f"想定外の形状です: {sst.shape}")

frames = sst[0]
valid = frames[np.isfinite(frames)]

print("入力:", path)
print("SST形状:", sst.shape)
print(f"有効画素率: {best_count / sst.size:.1%}")

vmin, vmax = np.percentile(valid, [2, 98])
cmap = plt.get_cmap("turbo").copy()
cmap.set_bad("lightgray")

fig, axes = plt.subplots(
    3, 3, figsize=(12, 11), constrained_layout=True
)
for i, ax in enumerate(axes.flat):
    im = ax.imshow(
        np.ma.masked_invalid(frames[i]),
        cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(f"Frame {i + 1}: {(i - 8) * 6:+d} h")
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im, ax=axes.ravel().tolist(), label="SST (deg C)")
fig.suptitle(f"{path.stem}\nGray = missing / invalid")
output = Path("outputs") / f"{path.stem}_9frames.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=150)
plt.close(fig)
print("保存先:", output.resolve())