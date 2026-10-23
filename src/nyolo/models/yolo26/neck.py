import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nyolo.nn.blocks import C3k2, Conv


class Neck(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.top4 = C3k2(384, 128)
        self.top3 = C3k2(256, 64)
        self.down3 = Conv(64, 64, 3, 2)
        self.bottom4 = C3k2(192, 128)
        self.down4 = Conv(128, 128, 3, 2)
        self.bottom5 = C3k2(384, 256, attention=True)

    def forward(
        self, features: tuple[Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        p3, p4, p5 = features
        t4 = self.top4(
            torch.cat((F.interpolate(p5, scale_factor=2, mode="nearest"), p4), 1)
        )
        t3 = self.top3(
            torch.cat((F.interpolate(t4, scale_factor=2, mode="nearest"), p3), 1)
        )
        b4 = self.bottom4(torch.cat((self.down3(t3), t4), 1))
        b5 = self.bottom5(torch.cat((self.down4(b4), p5), 1))
        return t3, b4, b5
