import math

import torch
from torch import Tensor
from torch.nn import functional as F

from nyolo.pose.types import PosePredictions, RawPose


def decode_boxes(raw: RawPose) -> Tensor:
    distances = raw.box_distances * raw.strides
    return torch.cat(
        (raw.anchors - distances[..., :2], raw.anchors + distances[..., 2:]), -1
    )


def decode_keypoints(raw: RawPose) -> Tensor:
    # Affine, unconstrained coordinate regression. Never squash or clip XY.
    return raw.anchors[:, None, :] + 2 * raw.strides[:, None, :] * raw.keypoint_offsets


def decode(
    raw: RawPose, *, max_detections: int = 300, score_threshold: float = 0.25
) -> PosePredictions:
    """NMS-free one-to-one output, padded to a fixed detection dimension."""
    if (
        max_detections < 1
        or not math.isfinite(score_threshold)
        or not 0 <= score_threshold <= 1
    ):
        raise ValueError(
            "Specify a positive max_detections and a score threshold in [0,1]"
        )
    scores, classes = raw.class_logits.sigmoid().max(-1)
    boxes = decode_boxes(raw)
    points = decode_keypoints(raw)
    finite = (
        torch.isfinite(boxes).all(-1)
        & torch.isfinite(points).all((-1, -2))
        & torch.isfinite(raw.keypoint_logits).all(-1)
        & torch.isfinite(raw.class_logits).all(-1)
        & torch.isfinite(scores)
    )
    if not finite.all():
        raise FloatingPointError("Nonfinite pose predictions; decoding was stopped")
    nonempty = (boxes[..., 2:] > boxes[..., :2]).all(-1)
    ranking = scores.masked_fill(~nonempty, -1)
    count = min(max_detections, scores.shape[1])
    selected_scores, indices = ranking.topk(count, dim=1)

    def select(value: Tensor) -> Tensor:
        index = indices.reshape(*indices.shape, *((1,) * (value.ndim - 2)))
        return value.gather(1, index.expand(-1, -1, *value.shape[2:]))

    selected_boxes = select(boxes)
    h, w = raw.image_size
    limit = selected_boxes.new_tensor((w, h, w, h))
    selected_boxes = torch.minimum(selected_boxes.clamp_min(0), limit)
    valid = (selected_scores >= score_threshold) & (
        selected_boxes[..., 2:] > selected_boxes[..., :2]
    ).all(-1)
    padding = max_detections - count

    def pad(value: Tensor, fill: float = 0) -> Tensor:
        spec = [0, 0] * (value.ndim - 2) + [0, padding]
        return F.pad(value, spec, value=fill)

    return PosePredictions(
        pad(selected_boxes),
        pad(selected_scores.clamp_min(0)),
        pad(select(classes)),
        pad(select(points)),
        pad(select(raw.keypoint_logits.sigmoid())),
        pad(valid),
    )
