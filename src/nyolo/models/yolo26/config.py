from dataclasses import dataclass
from math import ceil

from nyolo.pose.schema import COCO17, PoseSchema

# Numeric compound-scaling specification from the public model configuration
# guide. Block wiring remains the independently implemented paper graph.
_SCALES = {
    "n": (0.5, 0.25, 1024),
    "s": (0.5, 0.5, 1024),
    "m": (0.5, 1.0, 512),
    "l": (1.0, 1.0, 512),
    "x": (1.0, 1.5, 512),
}


@dataclass(frozen=True)
class YOLO26PoseConfig:
    schema: PoseSchema = COCO17
    num_classes: int = 1
    scale: str = "n"
    architecture_version: str | None = None
    coordinate_policy: str = "unbounded-v1"

    def __post_init__(self) -> None:
        if self.scale not in _SCALES:
            raise ValueError("Supported scales are n, s, m, l, x")
        version = f"nyolo26-pose-{self.scale}-v1"
        if self.architecture_version is None:
            object.__setattr__(self, "architecture_version", version)
        elif self.architecture_version != version:
            raise ValueError("architecture_version does not match the selected scale")
        if self.coordinate_policy != "unbounded-v1":
            raise ValueError("Only unbounded-v1 coordinates are supported")
        if self.num_classes < 1:
            raise ValueError("num_classes must be positive")

    def channels(self, base: int) -> int:
        _, width, cap = _SCALES[self.scale]
        return 8 * ceil(min(base, cap) * width / 8)

    @property
    def repeats(self) -> int:
        depth, _, _ = _SCALES[self.scale]
        return max(round(2 * depth), 1)

    @property
    def head_channels(self) -> tuple[int, int, int]:
        return self.channels(256), self.channels(512), self.channels(1024)
