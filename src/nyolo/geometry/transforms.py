from dataclasses import dataclass

from torch import Tensor

from nyolo.pose.types import PoseTargets


@dataclass(frozen=True)
class ImageTransform:
    """Actual letterbox scales, including independent integer resize rounding."""

    original_size: tuple[int, int]
    input_size: tuple[int, int]
    resized_size: tuple[int, int]
    padding: tuple[int, int]  # left, top

    def __post_init__(self) -> None:
        oh, ow = self.original_size
        ih, iw = self.input_size
        rh, rw = self.resized_size
        px, py = self.padding
        if min(oh, ow, ih, iw, rh, rw) <= 0 or min(px, py) < 0:
            raise ValueError(
                "Image dimensions must be positive and padding nonnegative"
            )
        if px + rw > iw or py + rh > ih:
            raise ValueError("Resized image extends beyond input canvas")

    def points(self, xy: Tensor, *, inverse: bool = False) -> Tensor:
        oh, ow = self.original_size
        rh, rw = self.resized_size
        scale = xy.new_tensor((rw / ow, rh / oh))
        pad = xy.new_tensor(self.padding)
        return (xy - pad) / scale if inverse else xy * scale + pad

    def boxes(self, boxes: Tensor, *, inverse: bool = False) -> Tensor:
        return self.points(
            boxes.reshape(*boxes.shape[:-1], 2, 2), inverse=inverse
        ).reshape_as(boxes)

    def targets(self, targets: PoseTargets) -> PoseTargets:
        return PoseTargets(
            self.boxes(targets.boxes),
            targets.labels,
            self.points(targets.keypoints),
            targets.visibility,
            targets.valid,
        )
