from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nyolo.geometry.boxes import box_area, complete_iou
from nyolo.inference.decode import decode_boxes, decode_keypoints
from nyolo.learning.assignment import task_aligned_assignment
from nyolo.learning.rle import ResidualFlow, rle_loss
from nyolo.models.yolo26 import YOLO26Pose
from nyolo.pose.schema import PoseSchema
from nyolo.pose.types import DualPose, PoseTargets, RawPose


@dataclass(frozen=True)
class LossConfig:
    box: float = 7.5
    box_l1: float = 1.0
    classification: float = 0.5
    oks: float = 24.0
    rle: float = 1.0
    keypoint_score: float = 1.0
    many_weight_start: float = 0.8
    many_weight_end: float = 0.1

    def __post_init__(self) -> None:
        gains = (
            self.box,
            self.box_l1,
            self.classification,
            self.oks,
            self.rle,
            self.keypoint_score,
        )
        if any(not isfinite(x) or x < 0 for x in gains):
            raise ValueError("Loss gains must be finite and nonnegative")
        if not 0 <= self.many_weight_end <= self.many_weight_start <= 1:
            raise ValueError("Invalid Progressive Loss weights")


@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    components: dict[str, Tensor]


class PoseCriterion(nn.Module):
    oks_sigmas: Tensor

    def __init__(self, schema: PoseSchema, config: LossConfig | None = None) -> None:
        super().__init__()
        self.schema = schema
        self.config = LossConfig() if config is None else config
        self.flow = ResidualFlow()
        self.register_buffer(
            "oks_sigmas", torch.tensor(schema.oks_sigmas, dtype=torch.float32)
        )

    def _branch(
        self, raw: RawPose, targets: PoseTargets, *, one_to_one: bool
    ) -> dict[str, Tensor]:
        if raw.sigma_logits is None:
            raise ValueError(
                "Loss requires forward_train outputs with uncertainty predictions"
            )
        boxes = decode_boxes(raw)
        assignment = task_aligned_assignment(
            boxes.detach(),
            raw.class_logits.detach(),
            raw.anchors,
            targets,
            topk=7 if one_to_one else 10,
            one_to_one=one_to_one,
        )
        fg = assignment.foreground
        normalizer = assignment.quality.sum().clamp_min(1)
        class_targets = torch.zeros_like(raw.class_logits)
        bi, ai = torch.where(fg)
        ti = assignment.target_indices[bi, ai]
        class_targets[bi, ai, targets.labels[bi, ti]] = assignment.quality[bi, ai]
        classification = (
            F.binary_cross_entropy_with_logits(
                raw.class_logits, class_targets, reduction="sum"
            )
            / normalizer
        )
        zero = (
            raw.box_distances.sum()
            + raw.keypoint_offsets.sum()
            + raw.keypoint_logits.sum()
            + raw.sigma_logits.sum()
        ) * 0
        parts = {
            "box": zero,
            "box_l1": zero,
            "classification": classification,
            "oks": zero,
            "rle": zero,
            "keypoint_score": zero,
        }
        if bi.numel() == 0:
            return parts
        weights = assignment.quality[bi, ai]
        gt_box = targets.boxes[bi, ti]
        predicted_box = boxes[bi, ai]
        parts["box"] = (
            (1 - complete_iou(predicted_box, gt_box)) * weights
        ).sum() / normalizer
        anchors, strides = raw.anchors[ai], raw.strides[ai]
        distances = (
            torch.cat((anchors - gt_box[..., :2], gt_box[..., 2:] - anchors), -1)
            / strides
        )
        parts["box_l1"] = (
            (raw.box_distances[bi, ai] - distances).abs().mean(-1) * weights
        ).sum() / normalizer
        known = targets.visibility[bi, ti] > 0
        gt_xy = torch.where(known[..., None], targets.keypoints[bi, ti], 0)
        predicted_xy = decode_keypoints(raw)[bi, ai]
        # Mask before nonlinear arithmetic: multiplying a NaN by zero is not safe.
        residual = torch.where(known[..., None], predicted_xy - gt_xy, 0)
        area = box_area(gt_box).clamp_min(1e-7)
        denominator = 2 * area[:, None] * (2 * self.oks_sigmas).square()
        oks = 1 - torch.exp(-residual.square().sum(-1) / denominator)
        counts = known.sum(-1).clamp_min(1)
        parts["oks"] = ((oks * known).sum(-1) / counts * weights).sum() / normalizer
        parts["keypoint_score"] = (
            F.binary_cross_entropy_with_logits(
                raw.keypoint_logits[bi, ai],
                known.to(raw.keypoint_logits.dtype),
                reduction="none",
            ).mean(-1)
            * weights
        ).sum() / normalizer
        rle = rle_loss(
            residual / area.sqrt()[:, None, None], raw.sigma_logits[bi, ai], self.flow
        )
        parts["rle"] = ((rle * known).sum(-1) / counts * weights).sum() / normalizer
        return parts

    def forward(
        self, predictions: DualPose, targets: PoseTargets, *, progress: float
    ) -> LossOutput:
        if not isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError("Pass explicit training progress in [0,1]")
        raw = predictions.one_to_one
        targets.validate(
            raw.image_size,
            self.schema.num_keypoints,
            raw.class_logits.shape[-1],
            raw.class_logits.device,
        )
        if targets.valid.shape[0] != raw.class_logits.shape[0]:
            raise ValueError("Target batch size differs from predictions")
        many = self._branch(predictions.one_to_many, targets, one_to_one=False)
        one = self._branch(raw, targets, one_to_one=True)
        c = self.config
        many_weight = c.many_weight_start + progress * (
            c.many_weight_end - c.many_weight_start
        )
        components = {f"many/{k}": v for k, v in many.items()}
        components.update({f"one/{k}": v for k, v in one.items()})
        total = raw.class_logits.sum() * 0
        for name in many:
            mixed = many_weight * many[name] + (1 - many_weight) * one[name]
            total = total + getattr(c, name) * mixed
        return LossOutput(total, components)


class PoseTrainingModule(nn.Module):
    """Optimizer ownership includes both the pose model and the trainable RLE flow."""

    def __init__(self, model: YOLO26Pose, config: LossConfig | None = None) -> None:
        super().__init__()
        self.model = model
        self.criterion = PoseCriterion(model.config.schema, config)

    def forward(
        self, images: Tensor, targets: PoseTargets, *, progress: float
    ) -> LossOutput:
        return self.criterion(
            self.model.forward_train(images), targets, progress=progress
        )
