from torch import Tensor, nn

from nyolo.models.yolo26.config import YOLO26PoseConfig
from nyolo.nn.blocks import C2PSA, SPPF, C3k2, Conv


class Backbone(nn.Module):
    """The paper's shared graph with explicit compound width/depth scaling."""

    def __init__(self, config: YOLO26PoseConfig) -> None:
        super().__init__()
        c64, c128, c256, c512, c1024 = (
            config.channels(c) for c in (64, 128, 256, 512, 1024)
        )
        n = config.repeats
        self.stem = nn.Sequential(
            Conv(3, c64, 3, 2),
            Conv(c64, c128, 3, 2),
            C3k2(c128, c256, expansion=0.25, repeats=n),
        )
        self.p3 = nn.Sequential(
            Conv(c256, c256, 3, 2), C3k2(c256, c512, expansion=0.25, repeats=n)
        )
        self.p4 = nn.Sequential(
            Conv(c512, c512, 3, 2), C3k2(c512, c512, c3k=True, repeats=n)
        )
        self.p5 = nn.Sequential(
            Conv(c512, c1024, 3, 2),
            C3k2(c1024, c1024, c3k=True, repeats=n),
            SPPF(c1024),
            C2PSA(c1024, repeats=n),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        p3 = self.p3(self.stem(x))
        p4 = self.p4(p3)
        return p3, p4, self.p5(p4)
