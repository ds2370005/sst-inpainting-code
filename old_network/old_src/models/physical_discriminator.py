# 旧実装・旧テスト: 全体識別器と局所PatchGANを組み合わせた旧識別器を定義する。
"""旧方式のglobal/local PatchGAN識別器。"""
# ── 使用するライブラリ・共通処理 ──────────────────
from torch import nn
from src.models.discriminator import Discriminator

# ── 処理に使用する関数・クラス ──────────────────
class PhysicalModelDiscriminator(nn.Module):
    """ACPRの全体・局所識別をTGRSの256画素・単一SST出力に適用した旧実装。

    全体側は表IIIを保持する。局所側は全畳み込みのPatchGANで、
    256×256入力から32×32のロジットを返す。図8の厳密な再現ではない。"""
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


