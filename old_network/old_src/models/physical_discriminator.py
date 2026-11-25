"""旧方式のglobal/local PatchGAN識別器。"""
from torch import nn
from src.models.discriminator import Discriminator

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


