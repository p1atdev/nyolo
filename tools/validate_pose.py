"""Development-only trainability probe; not a dataset trainer or package API."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from nyolo import COCO17, YOLO26Pose
from nyolo.checkpoint import load_checkpoint, save_checkpoint, save_training_state
from nyolo.geometry.boxes import box_iou
from nyolo.inference import Predictor, decode
from nyolo.learning import PoseTrainingModule
from nyolo.pose.types import PoseTargets
from nyolo.runtime import resolve_device


def synthetic_batch(size: int) -> tuple[torch.Tensor, PoseTargets, list[Image.Image]]:
    """Two identifiable stick figures, with wrists/ankles deliberately off canvas."""
    base = (
        np.array(
            [
                [0.50, 0.13],
                [0.46, 0.11],
                [0.54, 0.11],
                [0.41, 0.14],
                [0.59, 0.14],
                [0.33, 0.32],
                [0.67, 0.32],
                [0.12, 0.47],
                [0.88, 0.47],
                [-0.12, 0.62],
                [1.12, 0.62],
                [0.40, 0.62],
                [0.60, 0.62],
                [0.31, 0.87],
                [0.69, 0.87],
                [0.24, 1.12],
                [0.76, 1.12],
            ],
            dtype=np.float32,
        )
        * size
    )
    images, keypoints, visibility = [], [], []
    for i in range(2):
        points = base.copy()
        points[:, 0] += (-0.035 if i == 0 else 0.035) * size
        v = np.full(17, 2, dtype=np.int64)
        v[((points < 0) | (points >= size)).any(-1)] = 1
        image = Image.new("RGB", (size, size), (22 + i * 18, 27, 35))
        draw = ImageDraw.Draw(image)
        color = (220, 145, 80) if i == 0 else (90, 185, 220)
        for a, b in COCO17.skeleton:
            draw.line(
                [tuple(points[a]), tuple(points[b])],
                fill=color,
                width=max(size // 32, 2),
            )
        for x, y in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(235, 235, 235))
        if i == 1:
            v[3] = 0
            points[3] = np.nan
        images.append(image)
        keypoints.append(points)
        visibility.append(v)
    batch = torch.from_numpy(
        np.stack([np.array(x, dtype=np.float32) / 255 for x in images])
    )
    batch = batch.permute(0, 3, 1, 2).contiguous()
    targets = PoseTargets(
        torch.tensor([[[0.03 * size, 0.06 * size, 0.97 * size, float(size)]]] * 2),
        torch.zeros(2, 1, dtype=torch.int64),
        torch.from_numpy(np.stack(keypoints))[:, None],
        torch.from_numpy(np.stack(visibility))[:, None],
        torch.ones(2, 1, dtype=torch.bool),
    )
    return batch, targets, images


@torch.no_grad()
def measurements(
    model: YOLO26Pose, images: torch.Tensor, targets: PoseTargets
) -> dict[str, float]:
    model.eval()
    result = decode(model(images), max_detections=1, score_threshold=0.0)
    xy = result.keypoints[:, 0]
    gt, known = targets.keypoints[:, 0], targets.visibility[:, 0] > 0
    side = images.shape[-1]
    outside = ((gt < 0) | (gt >= side)).any(-1) & known
    error = (xy - gt).square().sum(-1).sqrt()
    correct_side = (
        ((gt[..., 0] < 0) & (xy[..., 0] < 0))
        | ((gt[..., 0] >= side) & (xy[..., 0] >= side))
        | ((gt[..., 1] < 0) & (xy[..., 1] < 0))
        | ((gt[..., 1] >= side) & (xy[..., 1] >= side))
    )
    return {
        "mean_keypoint_error_px": float(error[known].mean()),
        "off_image_error_px": float(error[outside].mean()),
        "off_image_correct_side_fraction": float(correct_side[outside].float().mean()),
        "mean_box_iou": float(box_iou(result.boxes[:, 0], targets.boxes[:, 0]).mean()),
        "mean_detection_score": float(result.scores[:, 0].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--size", type=int, default=96)
    args = parser.parse_args()
    if args.steps < 2 or args.size < 64 or args.size % 32:
        parser.error("Use at least 2 steps and an image size >=64 divisible by 32")
    device = resolve_device(args.device)
    torch.manual_seed(7)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    images, targets, originals = synthetic_batch(args.size)
    for i, image in enumerate(originals):
        image.save(args.output / f"synthetic-{i}.png")
    images, targets = images.to(device), targets.to(device)
    module = PoseTrainingModule(YOLO26Pose()).to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.002, weight_decay=0.0)
    before = measurements(module.model, images, targets)
    gradients: dict[str, float] = {}
    history = []
    started = time.perf_counter()
    for step in range(args.steps):
        module.train()
        optimizer.zero_grad(set_to_none=True)
        loss = module(images, targets, progress=step / (args.steps - 1))
        if not torch.isfinite(loss.total):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        loss.total.backward()
        if step == 0:
            groups = {
                "backbone": module.model.backbone,
                "neck": module.model.neck,
                "one_to_many_pose": module.model.one_to_many.joints,
                "one_to_one_pose": module.model.one_to_one.joints,
                "one_to_one_sigma": module.model.one_to_one.sigma,
                "rle_flow": module.criterion.flow,
            }
            for name, group in groups.items():
                values = [
                    p.grad.square().sum()
                    for p in group.parameters()
                    if p.grad is not None
                ]
                gradients[name] = float(torch.stack(values).sum().sqrt())
            if any(not np.isfinite(v) or v <= 0 for v in gradients.values()):
                raise AssertionError(f"Missing or invalid gradient: {gradients}")
        torch.nn.utils.clip_grad_norm_(
            module.parameters(), 10.0, error_if_nonfinite=True
        )
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == args.steps:
            item = {
                "step": step + 1,
                "loss": float(loss.total.detach()),
                **measurements(module.model, images, targets),
            }
            history.append(item)
            print(json.dumps(item), flush=True)
    after = measurements(module.model, images, targets)
    checkpoint = args.output / "synthetic-model.pt"
    save_checkpoint(checkpoint, module.model)
    save_training_state(args.output / "synthetic-training-components.pt", module)
    restored = load_checkpoint(checkpoint, device=args.device)
    with torch.inference_mode():
        expected = module.model(images)
        actual = restored(images)
        torch.testing.assert_close(
            actual.class_logits, expected.class_logits, rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual.keypoint_offsets, expected.keypoint_offsets, rtol=0, atol=0
        )
        # Full-resolution architecture check uses the same model, not a reduced surrogate.
        full = restored(torch.zeros(1, 3, 640, 640, device=device))
        assert full.keypoint_offsets.shape == (1, 8400, 17, 2)
        assert torch.isfinite(full.keypoint_offsets).all()
    predictor = Predictor(
        restored, device=args.device, image_size=(args.size, args.size)
    )
    records = predictor(originals, max_detections=1, score_threshold=0.0).to_records()
    (args.output / "predictions.json").write_text(
        json.dumps(records, allow_nan=False, indent=2) + "\n"
    )
    # Inference API and tensor path must agree, including off-image coordinates.
    exported = torch.tensor([r[0]["keypoints"] for r in records], device=device)
    reference = decode(actual, max_detections=1, score_threshold=0.0).keypoints[:, 0]
    torch.testing.assert_close(exported, reference, rtol=1e-5, atol=1e-4)
    # Exercise full-resolution training as well as the small overfit workload.
    # A separate module preserves the already-validated checkpoint and BN state.
    full_training = PoseTrainingModule(YOLO26Pose()).to(device).train()
    scale = 640 / args.size
    full_targets = PoseTargets(
        targets.boxes * scale,
        targets.labels,
        targets.keypoints * scale,
        targets.visibility,
        targets.valid,
    )
    full_images = F.interpolate(
        images, size=(640, 640), mode="bilinear", align_corners=False
    )
    full_loss = full_training(full_images, full_targets, progress=0.5)
    if not torch.isfinite(full_loss.total):
        raise FloatingPointError("Nonfinite full-resolution training loss")
    full_loss.total.backward()
    full_gradient_norm = torch.nn.utils.clip_grad_norm_(
        full_training.parameters(),
        10.0,
        error_if_nonfinite=True,
    )
    assert full_gradient_norm > 0
    passed = (
        after["mean_keypoint_error_px"]
        < min(12.0, before["mean_keypoint_error_px"] * 0.35)
        and after["off_image_error_px"] < min(12.0, before["off_image_error_px"] * 0.35)
        and after["off_image_correct_side_fraction"] == 1.0
        and after["mean_box_iou"] > 0.7
        and after["mean_detection_score"] > 0.5
    )
    report = {
        "passed": passed,
        "device": str(images.device),
        "torch": torch.__version__,
        "cpu_fallback_enabled": False,
        "steps": args.steps,
        "input_size": args.size,
        "seed": 7,
        "elapsed_seconds": time.perf_counter() - started,
        "model_parameters": sum(p.numel() for p in module.model.parameters()),
        "before": before,
        "after": after,
        "gradient_norms_first_step": gradients,
        "full_resolution_raw_shape": list(full.keypoint_offsets.shape),
        "full_resolution_backward": True,
        "full_resolution_loss": float(full_loss.total.detach()),
        "full_resolution_gradient_norm": float(full_gradient_norm),
        "checkpoint_equal": True,
        "predictor_equal": True,
        "history": history,
        "scope": "Synthetic two-image overfit only; no real-data accuracy claim",
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, allow_nan=False, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)
    if not passed:
        raise AssertionError(
            "Synthetic trainability gates failed; inspect metrics.json"
        )


if __name__ == "__main__":
    main()
