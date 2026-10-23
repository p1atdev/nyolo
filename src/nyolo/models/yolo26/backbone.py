from torch import Tensor, nn

from nyolo.nn.blocks import C2PSA, SPPF, C3k2, Conv


class Backbone(nn.Module):
    """Nano: width 1/4, depth 1/2; P3/P4/P5 channels 128/128/256."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            Conv(3, 16, 3, 2),
            Conv(16, 32, 3, 2),
            C3k2(32, 64, expansion=0.25),
        )
        self.p3 = nn.Sequential(Conv(64, 64, 3, 2), C3k2(64, 128, expansion=0.25))
        self.p4 = nn.Sequential(Conv(128, 128, 3, 2), C3k2(128, 128, c3k=True))
        self.p5 = nn.Sequential(
            Conv(128, 256, 3, 2),
            C3k2(256, 256, c3k=True),
            SPPF(256),
            C2PSA(256),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        p3 = self.p3(self.stem(x))
        p4 = self.p4(p3)
        return p3, p4, self.p5(p4)
