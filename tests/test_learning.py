from math import log, pi

import pytest
import torch

from nyolo.learning import PoseCriterion
from nyolo.learning.assignment import task_aligned_assignment
from nyolo.learning.rle import ResidualFlow
from nyolo.pose.schema import PoseSchema
from nyolo.pose.types import DualPose, PoseTargets, RawPose

SCHEMA = PoseSchema("two-v1", ("left", "right"), (1, 0), (0.1, 0.1))


def fixture(device, visibility=(1, 2), empty=False):
    raw = RawPose(
        torch.ones(1, 2, 4, device=device, requires_grad=True),
        torch.zeros(1, 2, 1, device=device, requires_grad=True),
        torch.zeros(1, 2, 2, 2, device=device, requires_grad=True),
        torch.zeros(1, 2, 2, device=device, requires_grad=True),
        torch.zeros(1, 2, 2, 2, device=device, requires_grad=True),
        torch.tensor([[12.0, 12.0], [20.0, 20.0]], device=device),
        torch.full((2, 1), 8.0, device=device),
        (32, 32),
    )
    targets = PoseTargets(
        torch.tensor([[[8.0, 8.0, 24.0, 24.0]]], device=device),
        torch.zeros(1, 1, dtype=torch.int64, device=device),
        torch.tensor([[[[-20.0, 12.0], [20.0, 20.0]]]], device=device),
        torch.tensor([[visibility]], dtype=torch.int64, device=device),
        torch.tensor([[not empty]], dtype=torch.bool, device=device),
    )
    return raw, targets


def test_off_image_supervision_and_flow_gradients(device):
    raw, targets = fixture(device)
    criterion = PoseCriterion(SCHEMA).to(device)
    output = criterion(DualPose(raw, raw), targets, progress=0.5)
    assert torch.isfinite(output.total)
    output.total.backward()
    assert raw.keypoint_offsets.grad[..., 0, 0].abs().sum() > 0
    assert raw.sigma_logits.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in criterion.flow.parameters()
    )


@pytest.mark.parametrize("empty", (False, True))
def test_unknown_and_empty_targets_are_finite(device, empty):
    raw, targets = fixture(device, visibility=(0, 0), empty=empty)
    targets.keypoints.fill_(float("nan"))
    criterion = PoseCriterion(SCHEMA).to(device)
    output = criterion(DualPose(raw, raw), targets, progress=0.0)
    output.total.backward()
    assert torch.isfinite(output.total)
    for value in raw[:5]:
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
    if empty:
        for value in (raw.box_distances, raw.keypoint_offsets, raw.sigma_logits):
            assert value.grad.count_nonzero() == 0
    assert torch.isfinite(raw.keypoint_offsets.grad).all()
    assert raw.keypoint_offsets.grad.abs().sum() == 0
    assert output.components["one/rle"] == 0
    assert output.components["one/oks"] == 0
    if not empty:
        # Unknown points are supervised as unlocalizable, even with NaN XY.
        assert raw.keypoint_logits.grad.abs().sum() > 0


def test_no_gt_dimension(device):
    raw, _ = fixture(device)
    targets = PoseTargets(
        torch.empty(1, 0, 4, device=device),
        torch.empty(1, 0, dtype=torch.int64, device=device),
        torch.empty(1, 0, 2, 2, device=device),
        torch.empty(1, 0, 2, dtype=torch.int64, device=device),
        torch.empty(1, 0, dtype=torch.bool, device=device),
    )
    loss = PoseCriterion(SCHEMA).to(device)(DualPose(raw, raw), targets, progress=1.0)
    loss.total.backward()
    assert loss.total > 0
    assert torch.isfinite(raw.class_logits.grad).all()


def test_stal_uses_surrogate_for_candidates_only(device):
    anchors = torch.tensor([[4.0, 4.0], [12.0, 4.0], [20.0, 4.0]], device=device)
    boxes = torch.tensor(
        [[[0.0, 0.0, 8.0, 8.0], [8.0, 0.0, 16.0, 8.0], [16.0, 0.0, 24.0, 8.0]]],
        device=device,
    )
    _, target = fixture(device)
    target.boxes[:] = torch.tensor([[[7.0, 3.0, 8.0, 4.0]]], device=device)
    original = target.boxes.clone()
    many = task_aligned_assignment(
        boxes, torch.zeros(1, 3, 1, device=device), anchors, target, topk=10
    )
    one = task_aligned_assignment(
        boxes,
        torch.zeros(1, 3, 1, device=device),
        anchors,
        target,
        topk=7,
        one_to_one=True,
    )
    assert many.foreground.sum() == 2
    assert one.foreground.sum() == 1
    torch.testing.assert_close(target.boxes, original)
    assert one.quality.max() <= 1 / 64 + 1e-6  # IoU uses the true one-pixel box.


def test_flow_starts_as_normal_density(device):
    flow = ResidualFlow().to(device)
    x = torch.tensor([[0.0, 0.0], [1.0, -2.0]], device=device)
    expected = -0.5 * (x.square() + log(2 * pi)).sum(-1)
    torch.testing.assert_close(flow.log_prob(x), expected)


def test_multi_person_assignment_and_collision_policy(device):
    anchors = torch.tensor([[4.0, 4.0], [12.0, 4.0], [20.0, 4.0]], device=device)
    boxes = torch.tensor(
        [[[0.0, 0.0, 8.0, 8.0], [8.0, 0.0, 16.0, 8.0], [16.0, 0.0, 24.0, 8.0]]],
        device=device,
    )
    gt = boxes[:, (0, 2)].clone()
    targets = PoseTargets(
        gt,
        torch.zeros(1, 2, dtype=torch.int64, device=device),
        torch.zeros(1, 2, 2, 2, device=device),
        torch.zeros(1, 2, 2, dtype=torch.int64, device=device),
        torch.ones(1, 2, dtype=torch.bool, device=device),
    )
    logits = torch.zeros(1, 3, 1, device=device)
    result = task_aligned_assignment(
        boxes, logits, anchors, targets, topk=7, one_to_one=True
    )
    assert result.foreground.tolist() == [[True, False, True]]
    assert result.target_indices[0, [0, 2]].tolist() == [0, 1]
    # Identical GTs compete for the same single anchor; no implicit rematch.
    targets.boxes[:, 1] = targets.boxes[:, 0]
    collision = task_aligned_assignment(
        boxes, logits, anchors, targets, topk=7, one_to_one=True
    )
    assert collision.foreground.tolist() == [[True, False, False]]


def test_padded_nan_boxes_and_invalid_labels_are_ignored(device):
    raw, targets = fixture(device, empty=True)
    targets.boxes.fill_(float("nan"))
    targets.labels.fill_(-100)
    targets.visibility.fill_(-100)
    loss = PoseCriterion(SCHEMA).to(device)(DualPose(raw, raw), targets, progress=0.5)
    loss.total.backward()
    assert torch.isfinite(loss.total)
    assert torch.isfinite(raw.class_logits.grad).all()


def test_reject_visible_off_image_and_missing_progress():
    raw, targets = fixture("cpu", visibility=(2, 2))
    criterion = PoseCriterion(SCHEMA)
    with pytest.raises(ValueError, match="off-image"):
        criterion(DualPose(raw, raw), targets, progress=0.5)
    with pytest.raises(ValueError, match="progress"):
        criterion(DualPose(raw, raw), targets, progress=2.0)
