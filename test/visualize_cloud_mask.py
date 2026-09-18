from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

root = Path(__file__).resolve().parents[1]
input_path = (
    root / "data/patches256"
    / "20250403000000_r05_c05.npz"
)

with np.load(input_path, allow_pickle=False) as archive:
    masks = archive["cloud_mask_volume"]
    timestamps = archive["timestamps"]

if masks.shape != (1, 9, 256, 256):
    raise ValueError(f"想定外の形状: {masks.shape}")
if not np.isin(masks, [0, 1]).all():
    raise ValueError("マスクに0・1以外の値があります。")

masks = masks[0]
cmap = ListedColormap(["white", "black"])
norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

fig, axes = plt.subplots(
    3, 3, figsize=(12, 11), constrained_layout=True
)

for i, ax in enumerate(axes.flat):
    mask = masks[i]
    ax.imshow(
        mask,
        cmap=cmap,
        norm=norm,
        origin="upper",
        interpolation="nearest",
    )
    ax.set_title(
        f"Frame {i + 1}: {timestamps[i]}\n"
        f"Cloud / missing ocean: {mask.mean():.1%}"
    )
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    f"{input_path.stem}\n"
    "Black (1): cloud / missing ocean | "
    "White (0): clear ocean or land"
)

output = root / "outputs" / f"{input_path.stem}_9masks.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=150)
plt.close(fig)

print("入力:", input_path)
print("保存先:", output)