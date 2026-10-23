"""One explicit compile experiment per process; failures never switch execution."""

import argparse
import json
import statistics
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor
from validate_pose import synthetic_batch

from nyolo import YOLO26Pose, YOLO26PoseConfig
from nyolo.checkpoint import load_checkpoint
from nyolo.inference import decode
from nyolo.learning import PoseCriterion, PoseTrainingModule
from nyolo.runtime import resolve_device


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def tensors(output, prefix: str = "output") -> list[tuple[str, Tensor]]:
    if isinstance(output, Tensor):
        return [(prefix, output)]
    if isinstance(output, (tuple, list)):
        fields = getattr(output, "_fields", tuple(str(i) for i in range(len(output))))
        return [
            t
            for name, x in zip(fields, output, strict=True)
            for t in tensors(x, f"{prefix}.{name}")
        ]
    return []


def compare_outputs(actual, expected) -> dict:
    actual, expected = tensors(actual), tensors(expected)
    assert len(actual) == len(expected) and actual
    max_error = 0.0
    details = []
    for (name, a), (expected_name, e) in zip(actual, expected, strict=True):
        assert name == expected_name and a.shape == e.shape and a.dtype == e.dtype
        a, e = a.detach(), e.detach()
        assert torch.isfinite(a).all() and torch.isfinite(e).all()
        difference = (a - e).abs()
        error = float(difference.max())
        mismatch = ~torch.isclose(a, e, rtol=1e-3, atol=1e-4)
        details.append(
            {
                "name": name,
                "max_absolute_error": error,
                "mismatched_elements": int(mismatch.sum()),
                "elements": a.numel(),
            }
        )
        max_error = max(max_error, error)
    return {
        "passed": all(x["mismatched_elements"] == 0 for x in details),
        "rtol": 0.001,
        "atol": 0.0001,
        "max_absolute_error": max_error,
        "tensors": details,
    }


def run(args, device: torch.device) -> dict:
    torch.manual_seed(7)
    model = (
        YOLO26Pose(
            YOLO26PoseConfig(scale=args.scale), attention_backend=args.attention_backend
        ).to(device)
        if args.checkpoint is None
        else load_checkpoint(
            args.checkpoint,
            device=str(device),
            attention_backend=args.attention_backend,
        )
    )
    if model.config.scale != args.scale:
        raise ValueError("Explicit checkpoint scale differs from --scale")
    training = args.path in ("model_train", "training_module")
    model.train(training)
    images, targets, _ = synthetic_batch(args.size)
    images, targets = images.to(device), targets.to(device)
    if not training:
        images = images[:1]
    criterion = PoseCriterion(model.config.schema).to(device)
    if args.path == "inference":
        function = model
    elif args.path == "model_train":
        function = model.forward_train
    elif args.path == "training_module":
        module = PoseTrainingModule(model).to(device)

        def function(x):
            return module(x, targets, progress=0.5).total
    else:

        def function(x):
            return decode(model(x)).keypoints

    parameters = dict(model.named_parameters())
    if args.path == "model_train":
        parameters.update(
            {f"criterion.{k}": p for k, p in criterion.named_parameters()}
        )
    if args.path == "training_module":
        parameters = dict(module.named_parameters())

    def backward(output) -> Tensor:
        loss = (
            criterion(output, targets, progress=0.5).total
            if args.path == "model_train"
            else output
        )
        loss.backward()
        return loss.detach()

    def clear_gradients():
        for parameter in parameters.values():
            parameter.grad = None

    context = nullcontext if training else torch.inference_mode
    with context():
        expected = function(images)
        expected_gradients = {}
        eager_repeat = None
        if training:
            expected_loss = backward(expected)
            expected_gradients = {
                k: p.grad.detach().cpu().clone()
                for k, p in parameters.items()
                if p.grad is not None
            }
            assert expected_gradients
            clear_gradients()
            if args.repeat_reference:
                repeated = function(images)
                backward(repeated)
                eager_repeat = compare_outputs(repeated, expected)
                square_error, square_reference = 0.0, 0.0
                for name, value in expected_gradients.items():
                    gradient = parameters[name].grad.detach().cpu()
                    square_error += float((gradient - value).square().sum())
                    square_reference += float(value.square().sum())
                eager_repeat["relative_gradient_l2_error"] = (
                    square_error / max(square_reference, 1e-30)
                ) ** 0.5
                clear_gradients()
                del repeated
        # Failures propagate. This does not retry with a different backend or eager.
        options = (
            None
            if args.layout_optimization == "default"
            else {"layout_optimization": args.layout_optimization == "on"}
        )
        compiled = torch.compile(
            function,
            backend=args.backend,
            fullgraph=True,
            dynamic=False,
            options=options,
        )
        synchronize(device)
        started = time.perf_counter()
        actual = compiled(images)
        if training:
            actual_loss = backward(actual)
        synchronize(device)
        first_call = time.perf_counter() - started
        agreement = compare_outputs(actual, expected)
        gradient_error = None
        loss_agrees = True
        if training:
            loss_agrees = bool(
                torch.isclose(actual_loss, expected_loss, rtol=1e-3, atol=1e-4)
            )
            squared_error, squared_reference = 0.0, 0.0
            for name, parameter in parameters.items():
                if name not in expected_gradients:
                    assert parameter.grad is None, name
                    continue
                assert parameter.grad is not None, name
                value, reference = (
                    parameter.grad.detach().cpu(),
                    expected_gradients[name],
                )
                assert torch.isfinite(value).all(), name
                squared_error += float((value - reference).square().sum())
                squared_reference += float(reference.square().sum())
            gradient_error = (squared_error / max(squared_reference, 1e-30)) ** 0.5
            clear_gradients()
        del expected, actual, expected_gradients

        elapsed = {"eager": [], "compiled": []}
        # Alternate order to reduce thermal/order bias. No other GPU workload should
        # run alongside this process when these timings are interpreted.
        for i in range(args.iterations + 3):
            order = ("eager", "compiled") if i % 2 else ("compiled", "eager")
            for name in order:
                call = function if name == "eager" else compiled
                stance = (
                    torch.compiler.set_stance("fail_on_recompile")
                    if name == "compiled"
                    else nullcontext()
                )
                with stance:
                    clear_gradients()
                    synchronize(device)
                    started = time.perf_counter()
                    output = call(images)
                    if training:
                        backward(output)
                    synchronize(device)
                    if i >= 3:
                        elapsed[name].append((time.perf_counter() - started) * 1000)
        eager_ms, compiled_ms = elapsed["eager"], elapsed["compiled"]
    return {
        "passed": agreement["passed"]
        and loss_agrees
        and (gradient_error is None or gradient_error < 0.005),
        "execution_passed": True,
        "input_shape": list(images.shape),
        "max_output_absolute_error": agreement["max_absolute_error"],
        "output_agreement": agreement,
        "loss_agrees": loss_agrees,
        "eager_repeat": eager_repeat,
        "relative_gradient_l2_error": gradient_error,
        "first_compiled_call_seconds": first_call,
        "eager_median_ms": statistics.median(eager_ms),
        "compiled_median_ms": statistics.median(compiled_ms),
        "speedup": statistics.median(eager_ms) / statistics.median(compiled_ms),
        "eager_samples_ms": eager_ms,
        "compiled_samples_ms": compiled_ms,
        "parameters": sum(p.numel() for p in model.parameters()),
        "unique_graphs": torch._dynamo.utils.counters["stats"]["unique_graphs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--scale", choices=tuple("nsmlx"), default="n")
    parser.add_argument(
        "--attention-backend", choices=("naive", "sdpa"), default="sdpa"
    )
    parser.add_argument(
        "--backend", choices=("inductor", "aot_eager", "eager"), default="inductor"
    )
    parser.add_argument(
        "--path",
        choices=("inference", "model_train", "training_module", "inference_decode"),
        default="inference",
    )
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--layout-optimization", choices=("default", "on", "off"), default="default"
    )
    parser.add_argument("--repeat-reference", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.size < 64 or args.size % 32 or args.iterations < 1:
        parser.error("size must be >=64 divisible by 32; iterations must be positive")
    if args.backend != "inductor" and args.layout_optimization != "default":
        parser.error("layout-optimization is an Inductor-only option")
    torch.set_num_threads(4)
    device = resolve_device(args.device)
    if torch._dynamo.config.suppress_errors:
        raise RuntimeError("Dynamo suppress_errors must be False for this probe")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "torch": torch.__version__,
        "device": str(device),
        "dtype": "float32",
        "scale": args.scale,
        "path": args.path,
        "backend": args.backend,
        "attention_backend": args.attention_backend,
        "fullgraph": True,
        "dynamic": False,
        "cpu_fallback_enabled": False,
        "iterations": args.iterations,
        "run_id": args.run_id,
        "layout_optimization": args.layout_optimization,
    }
    report["weights"] = (
        "random initialization" if args.checkpoint is None else str(args.checkpoint)
    )
    try:
        report.update(run(args, device))
    except Exception as error:
        report.update(
            passed=False,
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        args.output.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in report.items() if k != "traceback"}),
            flush=True,
        )
        raise
    args.output.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise AssertionError(
            "Execution completed, but the unchanged numerical agreement gates failed"
        )


if __name__ == "__main__":
    main()
