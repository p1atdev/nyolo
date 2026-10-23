import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nyolo.models.yolo26.config import YOLO26PoseConfig
from nyolo.nn.attention import AttentionBackend
from nyolo.nn.blocks import C3k2, Conv


class Neck(nn.Module):
    def __init__(
        self, config: YOLO26PoseConfig, *, attention_backend: AttentionBackend = "sdpa"
    ) -> None:
        super().__init__()
        c256, c512, c1024 = config.head_channels
        n = config.repeats
        self.top4 = C3k2(c1024 + c512, c512, repeats=n)
        self.top3 = C3k2(2 * c512, c256, repeats=n)
        self.down3 = Conv(c256, c256, 3, 2)
        self.bottom4 = C3k2(c256 + c512, c512, repeats=n)
        self.down4 = Conv(c512, c512, 3, 2)
        self.bottom5 = C3k2(
            c512 + c1024, c1024, attention=True, attention_backend=attention_backend
        )

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
