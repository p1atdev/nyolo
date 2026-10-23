from dataclasses import dataclass

from nyolo.pose.schema import COCO17, PoseSchema


@dataclass(frozen=True)
class YOLO26PoseConfig:
    schema: PoseSchema = COCO17
    num_classes: int = 1
    scale: str = "n"
    architecture_version: str = "nyolo26-pose-n-v1"
    coordinate_policy: str = "unbounded-v1"

    def __post_init__(self) -> None:
        if self.scale != "n" or self.architecture_version != "nyolo26-pose-n-v1":
            raise ValueError("This release implements only nyolo26-pose-n-v1")
        if self.coordinate_policy != "unbounded-v1":
            raise ValueError("Only unbounded-v1 coordinates are supported")
        if self.num_classes < 1:
            raise ValueError("num_classes must be positive")
