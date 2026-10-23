"""Explicit MPS/CPU shape, unbounded-pose loss, gradient and update probes."""

import argparse
import gc
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from validate_pose import synthetic_batch

from nyolo import AttentionBackend, YOLO26Pose, YOLO26PoseConfig
from nyolo.checkpoint import load_checkpoint, save_checkpoint
from nyolo.inference import decode
from nyolo.learning import PoseTrainingModule
from nyolo.runtime import resolve_device


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def run(
    scale: str,
    device: torch.device,
    inference_size: int,
    training_size: int,
    attention_backend: AttentionBackend,
) -> dict:
    torch.manual_seed(7)
    module = PoseTrainingModule(
        YOLO26Pose(YOLO26PoseConfig(scale=scale), attention_backend=attention_backend)
    ).to(device)
    module.eval()
    params = sum(p.numel() for p in module.model.parameters())
    with torch.inference_mode():
        image = torch.rand(1, 3, inference_size, inference_size, device=device)
        started = time.perf_counter()
        raw = module.model(image)
        synchronize(device)
        elapsed = time.perf_counter() - started
        expected = sum((inference_size // s) ** 2 for s in (8, 16, 32))
        assert raw.keypoint_offsets.shape == (1, expected, 17, 2)
        assert torch.isfinite(raw.keypoint_offsets).all()
        predictions = decode(raw, score_threshold=0.0)
        assert predictions.keypoints.device == image.device
    del image, raw, predictions
    images, targets, _ = synthetic_batch(training_size)
    images, targets = images.to(device), targets.to(device)
    module.train()
    optimizer = torch.optim.SGD(module.parameters(), lr=1e-4)
    started = time.perf_counter()
    result = module(images, targets, progress=0.5)
    assert torch.isfinite(result.total)
    result.total.backward()
    missing = [
        name
        for name, p in module.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not missing, missing
    groups = {
        "backbone": module.model.backbone,
        "neck": module.model.neck,
        "many_pose": module.model.one_to_many.joints,
        "one_pose": module.model.one_to_one.joints,
        "sigma": module.model.one_to_one.sigma,
        "rle": module.criterion.flow,
    }
    norms = {}
    for name, group in groups.items():
        values = [
            p.grad.square().sum() for p in group.parameters() if p.grad is not None
        ]
        norm = torch.stack(values).sum().sqrt()
        assert torch.isfinite(norm) and norm > 0, name
        norms[name] = float(norm)
    torch.nn.utils.clip_grad_norm_(module.parameters(), 10.0, error_if_nonfinite=True)
    weight = module.model.backbone.stem[0][0].weight
    before = weight.detach().clone()
    optimizer.step()
    assert not torch.equal(weight, before)
    synchronize(device)
    training_seconds = time.perf_counter() - started
    module.eval()
    with torch.inference_mode(), TemporaryDirectory(prefix="nyolo-scale-") as temporary:
        path = Path(temporary) / "model.pt"
        save_checkpoint(path, module.model)
        restored = load_checkpoint(
            path, device=str(device), attention_backend=attention_backend
        )
        assert restored.config == module.model.config
        torch.testing.assert_close(
            restored(images).keypoint_offsets,
            module.model(images).keypoint_offsets,
            rtol=0,
            atol=0,
        )
    return {
        "scale": scale,
        "architecture_version": module.model.config.architecture_version,
        "parameters": params,
        "head_channels": list(module.model.config.head_channels),
        "repeats": module.model.config.repeats,
        "inference_input": [1, 3, inference_size, inference_size],
        "raw_keypoints": [1, expected, 17, 2],
        "first_eager_forward_seconds": elapsed,
        "training_input": list(images.shape),
        "training_loss": float(result.total.detach()),
        "training_step_seconds": training_seconds,
        "checkpoint_equal": True,
        "gradient_parameter_coverage": True,
        "gradient_norms": norms,
        "optimizer_updated_parameters": True,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument(
        "--attention-backend", choices=("naive", "sdpa"), default="sdpa"
    )
    parser.add_argument(
        "--scales", choices=tuple("nsmlx"), nargs="+", default=list("nsmlx")
    )
    parser.add_argument("--inference-size", type=int, default=640)
    parser.add_argument("--training-size", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.inference_size < 32 or args.inference_size % 32:
        parser.error("inference size must be a positive multiple of 32")
    if args.training_size < 64 or args.training_size % 32:
        parser.error("training size must be >=64 and divisible by 32")
    torch.set_num_threads(4)
    device = resolve_device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "torch": torch.__version__,
        "device": str(device),
        "attention_backend": args.attention_backend,
        "dtype": "float32",
        "cpu_fallback_enabled": False,
        "results": [],
    }
    for scale in args.scales:
        result = run(
            scale,
            device,
            args.inference_size,
            args.training_size,
            args.attention_backend,
        )
        report["results"].append(result)
        args.output.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
