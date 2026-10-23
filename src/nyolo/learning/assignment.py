from dataclasses import dataclass

import torch
from torch import Tensor

from nyolo.geometry.boxes import box_iou
from nyolo.pose.types import PoseTargets


@dataclass(frozen=True)
class Assignment:
    target_indices: Tensor
    foreground: Tensor
    quality: Tensor


@torch.no_grad()
def task_aligned_assignment(
    boxes: Tensor,
    class_logits: Tensor,
    anchors: Tensor,
    targets: PoseTargets,
    *,
    topk: int,
    one_to_one: bool = False,
    min_stride: float = 8,
    reference_stride: float = 16,
    alpha: float = 0.5,
    beta: float = 6,
) -> Assignment:
    """TAL + STAL candidate geometry; conflicts go to the highest-IoU GT.

    One-to-one selects one of each GT's top-k candidates, then resolves conflicts.
    Crowded GTs can remain unmatched; there is no hidden nearest-anchor rematch.
    """
    if topk < 1 or min_stride <= 0 or reference_stride < min_stride:
        raise ValueError("Invalid assignment configuration")
    b, a, _ = boxes.shape
    t = targets.boxes.shape[1]
    if t == 0:
        return Assignment(
            torch.zeros((b, a), dtype=torch.int64, device=boxes.device),
            torch.zeros((b, a), dtype=torch.bool, device=boxes.device),
            boxes.new_zeros((b, a)),
        )
    # Ignore padded coordinates before arithmetic, even if their values are NaN.
    gt = torch.where(targets.valid[..., None], targets.boxes, 0)
    center = (gt[..., :2] + gt[..., 2:]) * 0.5
    extent = gt[..., 2:] - gt[..., :2]
    surrogate = torch.where(extent < min_stride, reference_stride, extent)
    displacement = (anchors[None, None] - center[:, :, None]).abs()
    candidates = (displacement <= surrogate[:, :, None] * 0.5).all(-1)
    candidates &= targets.valid[..., None]
    overlap = box_iou(gt[:, :, None], boxes[:, None]).clamp_min(0)
    labels = torch.where(targets.valid, targets.labels, 0)
    probability = (
        class_logits.sigmoid()
        .transpose(1, 2)
        .gather(1, labels[..., None].expand(-1, -1, a))
    )
    metric = probability.pow(alpha) * overlap.pow(beta)
    ranking = metric.masked_fill(~candidates, -1)
    values, indices = ranking.topk(min(topk, a), dim=-1)
    if one_to_one:
        best = values.argmax(-1, keepdim=True)
        indices, values = indices.gather(-1, best), values.gather(-1, best)
    selected = torch.zeros_like(candidates).scatter(2, indices, values >= 0)
    winners = overlap.masked_fill(~selected, -1).argmax(1)
    foreground = selected.any(1)
    selected &= torch.arange(t, device=boxes.device)[None, :, None] == winners[:, None]
    positive_metric = metric * selected
    max_metric = positive_metric.amax(-1, keepdim=True).clamp_min(1e-30)
    max_overlap = (overlap * selected).amax(-1, keepdim=True)
    quality = (positive_metric / max_metric * max_overlap).amax(1)
    return Assignment(winners, foreground, quality)
