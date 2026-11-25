# 役割: 256×256のSST全体を識別するネットワークを定義し、設定に応じた識別器を作る。
"""Table III discriminator: six stride-2 layers, then a 4x4 scalar head.

Returns logits. The loss applies log-sigmoid stably.
The paper's training patch size is 256x256; other sizes are rejected.
"""
# ── 使用するライブラリ・共通処理 ──────────────────
import torch
from torch import nn


# ── 処理に使用する関数・クラス ──────────────────
class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        channels = [in_channels, 64, 128, 256, 256, 256, 512]
        for i, (source, target) in enumerate(zip(channels[:-1], channels[1:]), 1):
            setattr(self, f'd{i}', nn.utils.spectral_norm(nn.Conv2d(source, target, 5, 2, 2)))
        self.d7 = nn.utils.spectral_norm(nn.Conv2d(512, 1, 4, 1, 0))
        self.activation = nn.LeakyReLU(.2, inplace=True)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (self.in_channels, 256, 256):
            raise ValueError('Table III discriminator requires [B,C,256,256] training patches')
        for i in range(1, 7):
            x = self.activation(getattr(self, f'd{i}')(x))
        return self.d7(x).flatten(1)


def build_discriminator(loss_config):
    kind = loss_config.get("discriminator", "global")
    if kind == "global_local":
        from old_network.old_src.models.physical_discriminator import PhysicalModelDiscriminator
        return PhysicalModelDiscriminator()
    if kind == "global":
        return Discriminator()
    raise ValueError(f"Unknown discriminator: {kind}")


def __getattr__(name):
    if name == "PhysicalModelDiscriminator":
        from old_network.old_src.models.physical_discriminator import PhysicalModelDiscriminator
        return PhysicalModelDiscriminator
    raise AttributeError(name)
