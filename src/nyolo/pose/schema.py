from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class PoseSchema:
    """Joint order is part of the model, including for checkpoint loading."""

    name: str
    names: tuple[str, ...]
    flip_indices: tuple[int, ...]
    oks_sigmas: tuple[float, ...]
    skeleton: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        n = len(self.names)
        if not self.name or not n or len(set(self.names)) != n:
            raise ValueError("A schema needs a name and distinct joint names")
        if sorted(self.flip_indices) != list(range(n)):
            raise ValueError("flip_indices must be a permutation of all joints")
        if any(self.flip_indices[j] != i for i, j in enumerate(self.flip_indices)):
            raise ValueError("Applying the joint flip twice must restore joint order")
        if len(self.oks_sigmas) != n or any(
            not isfinite(s) or s <= 0 for s in self.oks_sigmas
        ):
            raise ValueError("Specify a finite positive OKS sigma for every joint")
        if any(not 0 <= i < n or not 0 <= j < n for i, j in self.skeleton):
            raise ValueError("Skeleton joint index outside schema")

    @property
    def num_keypoints(self) -> int:
        return len(self.names)


COCO17 = PoseSchema(
    name="coco17-v1",
    names=(
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ),
    flip_indices=(0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15),
    oks_sigmas=(
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    ),
    skeleton=(
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ),
)
