"""Table III discriminator: six stride-2 layers, then a 4x4 scalar head.

Returns logits. BCEWithLogits applies the paper's final sigmoid stably.
The paper's training patch size is 256x256; other sizes are rejected.
"""
import torch
from torch import nn


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


class PhysicalModelDiscriminator(nn.Module):
    """ACPR global/local principle adapted to the TGRS 256px, single-SST output.

    Global branch retains Table III. Local branch is a fully convolutional
    PatchGAN (32x32 logits for a 256x256 image). It is not an exact Fig.8 replica.
    """
    def __init__(self):
        super().__init__()
        self.global_discriminator = Discriminator()
        layers = []
        channels = [1, 32, 64, 128]
        for source, target in zip(channels[:-1], channels[1:]):
            layers += [nn.utils.spectral_norm(nn.Conv2d(source, target, 3, 2, 1)),
                       nn.LeakyReLU(.2)]
        layers += [nn.utils.spectral_norm(nn.Conv2d(128, 128, 3, 1, 1)),
                   nn.LeakyReLU(.2), nn.utils.spectral_norm(nn.Conv2d(128, 1, 1))]
        self.local_discriminator = nn.Sequential(*layers)

    def forward(self, x):
        return {"global": self.global_discriminator(x), "local": self.local_discriminator(x)}


def build_discriminator(loss_config):
    kind = loss_config.get("discriminator", "global")
    if kind == "global_local":
        return PhysicalModelDiscriminator()
    if kind == "global":
        return Discriminator()
    raise ValueError(f"Unknown discriminator: {kind}")
