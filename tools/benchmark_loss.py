"""Interleave current and saved baseline criteria with the same model/weights.

The baseline is an explicitly supplied local Python file containing PoseCriterion.
Both measurements include compiled forward_train, eager loss and backward.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from validate_pose import synthetic_batch

from nyolo import YOLO26Pose, YOLO26PoseConfig
from nyolo.learning import PoseCriterion
from nyolo.runtime import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--scale", choices=tuple("nsmlx"), default="n")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.size < 64 or args.size % 32 or args.iterations < 8 or args.iterations % 2:
        parser.error(
            "size must be >=64 and divisible by 32; iterations must be even and >=8"
        )
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("Set PYTORCH_ENABLE_MPS_FALLBACK=0 explicitly")
    if torch._dynamo.config.suppress_errors:
        raise RuntimeError("Dynamo suppress_errors must be False")
    if args.output.exists():
        parser.error("output exists; use a fresh path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("baseline_criterion", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    device = resolve_device("mps")
    torch.set_num_threads(4)
    torch.manual_seed(7)
    model = YOLO26Pose(YOLO26PoseConfig(scale=args.scale)).to(device).train()
    criteria = {
        "before": baseline.PoseCriterion(model.config.schema).to(device),
        "after": PoseCriterion(model.config.schema).to(device),
    }
    criteria["after"].load_state_dict(criteria["before"].state_dict())
    forward = torch.compile(
        model.forward_train,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        options={"layout_optimization": False},
    )
    images, targets, _ = synthetic_batch(args.size)
    images, targets = images.to(device), targets.to(device)

    def clear():
        model.zero_grad(set_to_none=True)
        for criterion in criteria.values():
            criterion.zero_grad(set_to_none=True)

    def invoke(name):
        raw = forward(images)
        loss = criteria[name](raw, targets, progress=0.5)
        loss.total.backward()
        return loss

    baseline_values = baseline_grads = None
    validation = {}
    for name, criterion in criteria.items():
        clear()
        loss = invoke(name)
        torch.mps.synchronize()
        values = {
            "total": loss.total.detach().cpu(),
            **{k: v.detach().cpu() for k, v in loss.components.items()},
        }
        params = {
            **dict(model.named_parameters()),
            **{f"criterion.{k}": p for k, p in criterion.named_parameters()},
        }
        grads = {
            k: p.grad.detach().cpu().clone()
            for k, p in params.items()
            if p.grad is not None
        }
        if name == "before":
            baseline_values, baseline_grads = values, grads
        value_error = max(float((values[k] - baseline_values[k]).abs()) for k in values)
        value_agreement = all(
            torch.isclose(values[k], baseline_values[k], rtol=1e-3, atol=1e-4)
            for k in values
        )
        assert grads.keys() == baseline_grads.keys()
        assert all(torch.isfinite(g).all() for g in grads.values())
        numerator = sum(
            float((grads[k] - baseline_grads[k]).square().sum()) for k in grads
        )
        denominator = sum(float(g.square().sum()) for g in baseline_grads.values())
        gradient_error = (numerator / denominator) ** 0.5
        validation[name] = {
            "passed": bool(value_agreement) and gradient_error < 0.005,
            "loss_max_abs_error": value_error,
            "gradient_relative_l2": gradient_error,
        }
    del baseline_values, baseline_grads, grads, values
    samples = {k: [] for k in criteria}
    with torch.compiler.set_stance("fail_on_recompile"):
        for i in range(args.iterations + 8):
            for name in ("before", "after") if i % 2 == 0 else ("after", "before"):
                clear()
                torch.mps.synchronize()
                start = time.perf_counter()
                invoke(name)
                torch.mps.synchronize()
                if i >= 8:
                    samples[name].append(1000 * (time.perf_counter() - start))
    report = {
        "passed": all(v["passed"] for v in validation.values()),
        "torch": torch.__version__,
        "scale": args.scale,
        "shape": list(images.shape),
        "validation": validation,
        "timings": {
            k: {"median_ms": statistics.median(v), "samples_ms": v}
            for k, v in samples.items()
        },
        "baseline": str(args.baseline),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "criterion_sha256": hashlib.sha256(
            Path("src/nyolo/learning/criterion.py").read_bytes()
        ).hexdigest(),
        "scope": "compiled NCHW model + eager pose criterion + backward; no optimizer or zero_grad",
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "timings"}, indent=2),
        flush=True,
    )
    print({k: v["median_ms"] for k, v in report["timings"].items()}, flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
