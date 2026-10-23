from math import pi

import torch
from torch import Tensor


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[..., 2:] - boxes[..., :2]).clamp_min(0).prod(-1)


def box_iou(a: Tensor, b: Tensor) -> Tensor:
    """Broadcasted xyxy IoU (pairwise callers insert singleton dimensions)."""
    low = torch.maximum(a[..., :2], b[..., :2])
    high = torch.minimum(a[..., 2:], b[..., 2:])
    intersection = (high - low).clamp_min(0).prod(-1)
    return intersection / (box_area(a) + box_area(b) - intersection).clamp_min(1e-7)


def complete_iou(a: Tensor, b: Tensor) -> Tensor:
    iou = box_iou(a, b)
    enclosing = (
        (torch.maximum(a[..., 2:], b[..., 2:]) - torch.minimum(a[..., :2], b[..., :2]))
        .square()
        .sum(-1)
        .clamp_min(1e-7)
    )
    center_distance = (
        ((a[..., :2] + a[..., 2:] - b[..., :2] - b[..., 2:]) * 0.5).square().sum(-1)
    )
    awh = (a[..., 2:] - a[..., :2]).clamp_min(1e-7)
    bwh = (b[..., 2:] - b[..., :2]).clamp_min(1e-7)
    aspect = (4 / pi**2) * (
        torch.atan(awh[..., 0] / awh[..., 1]) - torch.atan(bwh[..., 0] / bwh[..., 1])
    ).square()
    alpha = (aspect / (1 - iou + aspect).clamp_min(1e-7)).detach()
    return iou - center_distance / enclosing - alpha * aspect
