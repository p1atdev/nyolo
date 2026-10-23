from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor


class RawPose(NamedTuple):
    """Anchor-major tensors. Coordinates use network-input pixel units."""

    box_distances: Tensor
    class_logits: Tensor
    keypoint_offsets: Tensor
    keypoint_logits: Tensor
    sigma_logits: Tensor | None
    anchors: Tensor
    strides: Tensor
    image_size: tuple[int, int]


class DualPose(NamedTuple):
    one_to_many: RawPose
    one_to_one: RawPose


@dataclass(frozen=True)
class PoseTargets:
    """Padded B,T targets in input pixels; v=0 unknown, 1 occluded, 2 visible.

    Unknown joints may contain NaNs; known coordinates must be finite. Off-image
    joints have v=1, and remain supervised. Boxes describe the image intersection.
    """

    boxes: Tensor
    labels: Tensor
    keypoints: Tensor
    visibility: Tensor
    valid: Tensor

    def to(self, device: str | torch.device) -> "PoseTargets":
        return PoseTargets(
            *(
                x.to(device)
                for x in (
                    self.boxes,
                    self.labels,
                    self.keypoints,
                    self.visibility,
                    self.valid,
                )
            )
        )

    def validate(
        self,
        image_size: tuple[int, int],
        num_keypoints: int,
        num_classes: int,
        device: torch.device,
    ) -> None:
        if self.valid.ndim != 2:
            raise ValueError("valid must have shape B,T")
        b, t = self.valid.shape
        if (
            self.boxes.shape != (b, t, 4)
            or self.labels.shape != (b, t)
            or self.keypoints.shape != (b, t, num_keypoints, 2)
            or self.visibility.shape != (b, t, num_keypoints)
        ):
            raise ValueError("Inconsistent padded target shapes")
        tensors = (self.boxes, self.labels, self.keypoints, self.visibility, self.valid)
        if any(x.device != device for x in tensors):
            raise ValueError("All targets must be on the prediction device")
        if self.boxes.dtype != torch.float32 or self.keypoints.dtype != torch.float32:
            raise ValueError("Targets require float32 coordinates")
        if (
            self.labels.dtype != torch.int64
            or self.visibility.dtype != torch.int64
            or self.valid.dtype != torch.bool
        ):
            raise ValueError("labels/visibility must be int64 and valid must be bool")
        boxes, labels = self.boxes[self.valid], self.labels[self.valid]
        v, xy = self.visibility[self.valid], self.keypoints[self.valid]
        h, w = image_size
        if not torch.isfinite(boxes).all() or not (
            (boxes[..., 2:] > boxes[..., :2]).all()
            and (boxes[..., :2] >= 0).all()
            and (boxes[..., 2:] <= boxes.new_tensor((w, h))).all()
        ):
            raise ValueError(
                "Valid boxes must be finite, nonempty, and inside the image"
            )
        if not ((labels >= 0) & (labels < num_classes)).all():
            raise ValueError("Target class outside model class range")
        if not ((v >= 0) & (v <= 2)).all():
            raise ValueError(
                "Visibility must be 0 (unknown), 1 (occluded), or 2 (visible)"
            )
        if not torch.isfinite(xy[v > 0]).all():
            raise ValueError("Known joints require finite coordinates")
        inside = ((xy >= 0) & (xy < xy.new_tensor((w, h)))).all(-1)
        if ((v == 2) & ~inside).any():
            raise ValueError("An off-image annotated joint must be occluded (v=1)")


@dataclass(frozen=True)
class PosePredictions:
    boxes: Tensor
    scores: Tensor
    class_ids: Tensor
    keypoints: Tensor
    keypoint_scores: Tensor
    valid: Tensor

    def to_records(self) -> list[list[dict]]:
        """Explicit host conversion; negative/off-image coordinates are preserved."""
        result = []
        for i in range(self.boxes.shape[0]):
            mask = self.valid[i]
            values = [
                x[i][mask].detach().cpu().tolist()
                for x in (
                    self.boxes,
                    self.scores,
                    self.class_ids,
                    self.keypoints,
                    self.keypoint_scores,
                )
            ]
            result.append(
                [
                    dict(
                        zip(
                            (
                                "box",
                                "score",
                                "class_id",
                                "keypoints",
                                "keypoint_scores",
                            ),
                            row,
                            strict=True,
                        )
                    )
                    for row in zip(*values, strict=True)
                ]
            )
        return result
