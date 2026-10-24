"""Explicit MPS comparisons; Ultralytics is an optional benchmark-only package.

Run with uv run --frozen --with ultralytics==8.4.173 python ... . No upstream
source/config/weights are copied into nyolo. Comparisons use nc=1 and COCO17;
the independent architectures and training objectives are not identical.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
import traceback
from pathlib import Path

import torch
from probe_compile import compare_outputs
from validate_pose import synthetic_batch

from nyolo import YOLO26Pose, YOLO26PoseConfig
from nyolo.checkpoint import load_checkpoint
from nyolo.learning import PoseCriterion
from nyolo.runtime import resolve_device


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return tuple(snapshot(value[k]) for k in sorted(value) if k != "feats")
    if isinstance(value, (list, tuple)):
        return tuple(snapshot(v) for v in value)
    return value


def percentile(values, q):
    ordered = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def run(args):
    from ultralytics.cfg import DEFAULT_CFG
    from ultralytics.nn.tasks import PoseModel

    device = resolve_device("mps")
    if torch._dynamo.config.suppress_errors:
        raise RuntimeError("Dynamo suppress_errors must be False")
    torch.set_num_threads(4)
    torch.manual_seed(7)
    training = args.path == "training"
    model = (
        YOLO26Pose(YOLO26PoseConfig(scale=args.scale)).to(device)
        if args.checkpoint is None
        else load_checkpoint(args.checkpoint, device="mps")
    ).train(training)
    if model.config.scale != args.scale:
        raise ValueError("Checkpoint scale differs from --scale")
    if model.config.num_classes != 1 or model.config.schema.num_keypoints != 17:
        raise ValueError("Comparison requires nc=1 and 17 keypoints")
    # The external package constructs its own reference architecture. This YAML
    # name is passed to its API, never read or translated into nyolo's model.
    reference = PoseModel(f"yolo26{args.scale}-pose.yaml", nc=1, verbose=False)
    reference.model[-1].end2end = True
    reference = reference.to(device).train(training)
    images, targets, _ = synthetic_batch(args.size)
    images = (images if training else images[:1]).to(device)
    targets = targets.to(device)
    criterion = PoseCriterion(model.config.schema).to(device)
    if training:
        reference.args = DEFAULT_CFG
        reference.criterion = reference.init_criterion()
        # Same examples, translated into the external loss's normalized format.
        valid = targets.valid
        boxes = targets.boxes[valid]
        center = (boxes[:, :2] + boxes[:, 2:]) / (2 * args.size)
        wh = (boxes[:, 2:] - boxes[:, :2]) / args.size
        official_batch = {
            "img": images,
            "batch_idx": torch.where(valid)[0],
            "cls": targets.labels[valid].unsqueeze(-1),
            "bboxes": torch.cat((center, wh), -1),
            "keypoints": torch.cat(
                (
                    torch.where(
                        targets.visibility[valid].unsqueeze(-1) > 0,
                        targets.keypoints[valid],
                        0,
                    )
                    / args.size,
                    targets.visibility[valid].unsqueeze(-1),
                ),
                -1,
            ),
        }
        models = {"nyolo": model, "ultralytics": reference}
        calls = {"nyolo": model.forward_train, "ultralytics": reference}
    else:
        reference.fuse(verbose=False)
        if args.scope == "raw":
            head = reference.model[-1]

            def raw_head(features):
                prediction = head.forward_head(features, **head.one2one)
                return prediction["boxes"], prediction["scores"], prediction["kpts"]

            # A benchmark adapter around the documented head API: omit decode
            # and top-k, just as nyolo.forward does. No upstream implementation.
            head.forward = raw_head
        models = {"nyolo": model, "nyolo_fused": model.fuse(), "ultralytics": reference}
        calls = dict(models)
    functions = {}
    owners = {}
    for name, call in calls.items():
        for execution, options in (
            ("eager", None),
            ("compiled", {}),
            ("compiled_nchw", {"layout_optimization": False}),
        ):
            if execution not in args.executions:
                continue
            # Baseline nyolo without fusion needs only default compile.
            if not training and name == "nyolo" and execution == "compiled_nchw":
                continue
            key = f"{name}_{execution}"
            functions[key] = (
                call
                if options is None
                else torch.compile(
                    call,
                    backend="inductor",
                    fullgraph=True,
                    dynamic=False,
                    options=options,
                )
            )
            owners[key] = name
    groups = {
        name: {
            **dict(m.named_parameters()),
            **(
                {f"criterion.{k}": p for k, p in criterion.named_parameters()}
                if name == "nyolo" and training
                else {}
            ),
        }
        for name, m in models.items()
    }

    def clear():
        for m in models.values():
            m.zero_grad(set_to_none=True)
        criterion.zero_grad(set_to_none=True)

    def invoke(key):
        output = functions[key](images)
        loss = None
        if training:
            if owners[key] == "nyolo":
                loss = criterion(output, targets, progress=0.5).total
            else:
                loss = reference.loss(official_batch, preds=output)[0].sum()
            loss.backward()
        elif args.scope == "decoded" and owners[key].startswith("nyolo"):
            from nyolo.inference import decode

            result = decode(output)
            # Return a tensor-only representation for numerical validation.
            output = (
                result.boxes,
                result.scores,
                result.keypoints,
                result.keypoint_scores,
            )
        return output, loss

    validation, first, baselines, baseline_gradients = {}, {}, {}, {}
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for key in functions:
            clear()
            torch.mps.synchronize()
            start = time.perf_counter()
            output, loss = invoke(key)
            torch.mps.synchronize()
            first[key] = time.perf_counter() - start
            print(f"FIRST {key}: {first[key]:.2f}s", flush=True)
            library = "ultralytics" if owners[key] == "ultralytics" else "nyolo"
            current = (snapshot(output), snapshot(loss))
            if library not in baselines:
                baselines[library] = current
                if training:
                    # The external loss initializes its RLE parameters lazily.
                    groups["ultralytics"] = dict(reference.named_parameters())
                    baseline_gradients[library] = {
                        n: p.grad.detach().cpu().clone()
                        for n, p in groups[owners[key]].items()
                        if p.grad is not None
                    }
            agreement = compare_outputs(current, baselines[library])
            gradient_error = None
            if training:
                numerator = denominator = 0.0
                for name, parameter in groups[owners[key]].items():
                    expected = baseline_gradients[library].get(name)
                    if expected is None:
                        assert parameter.grad is None, name
                        continue
                    assert parameter.grad is not None, name
                    actual = parameter.grad.detach().cpu()
                    assert torch.isfinite(actual).all(), name
                    numerator += float((actual - expected).square().sum())
                    denominator += float(expected.square().sum())
                assert denominator > 0
                gradient_error = (numerator / denominator) ** 0.5
            validation[key] = {
                "passed": agreement["passed"]
                and (gradient_error is None or gradient_error < 0.005),
                "reference": f"{library}_eager",
                "outputs_and_loss": agreement,
                "gradient_relative_l2": gradient_error,
            }
            del output, loss
        del baselines, baseline_gradients
        names = list(functions)
        samples = {k: [] for k in names}
        with torch.compiler.set_stance("fail_on_recompile"):
            for i in range(args.warmup + args.iterations):
                offset = i % len(names)
                for key in names[offset:] + names[:offset]:
                    clear()
                    torch.mps.synchronize()
                    start = time.perf_counter()
                    output, loss = invoke(key)
                    torch.mps.synchronize()
                    if i >= args.warmup:
                        samples[key].append((time.perf_counter() - start) * 1000)
                    del output, loss
        profiles = {}
        if not training:
            for name, call in calls.items():
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as prof:
                    call(images)
                    torch.mps.synchronize()
                profiles[name] = {
                    e.key: {
                        "calls": e.count,
                        "self_cpu_ms": e.self_cpu_time_total / 1000,
                    }
                    for e in prof.key_averages()
                    if any(
                        s in e.key
                        for s in ("batch_norm", "convolution", "arange", "clone")
                    )
                }
    return {
        "passed": all(v["passed"] for v in validation.values()),
        "shape": list(images.shape),
        "validation": validation,
        "first_call_seconds": first,
        "parameters": {
            k: sum(p.numel() for p in m.parameters()) for k, m in models.items()
        },
        "timings": {
            k: {
                "median_ms": statistics.median(v),
                "p10_ms": percentile(v, 0.1),
                "p90_ms": percentile(v, 0.9),
                "samples_ms": v,
            }
            for k, v in samples.items()
        },
        "cpu_operator_profiles_not_gpu_times": profiles,
        "unique_graphs": torch._dynamo.utils.counters["stats"]["unique_graphs"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=tuple("nsmlx"), default="n")
    parser.add_argument(
        "--path", choices=("inference", "training"), default="inference"
    )
    parser.add_argument("--scope", choices=("raw", "decoded"), default="raw")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--executions",
        nargs="+",
        choices=("eager", "compiled", "compiled_nchw"),
        default=["eager", "compiled", "compiled_nchw"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.size < 64 or args.size % 32 or args.iterations < 8 or args.warmup < 1:
        parser.error("size must be >=64 and divisible by 32; iterations>=8; warmup>=1")
    if args.path == "training" and args.scope != "raw":
        parser.error("training uses raw dual-head predictions")
    if "eager" not in args.executions:
        parser.error("eager is required as the numerical reference")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("Set PYTORCH_ENABLE_MPS_FALLBACK=0 explicitly")
    if args.output.exists():
        parser.error("output already exists; choose a fresh path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "torch": torch.__version__,
        "ultralytics": importlib.metadata.version("ultralytics"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "mps",
        "dtype": "float32",
        "seed": 7,
        "cpu_threads": 4,
        "scale": args.scale,
        "path": args.path,
        "scope": args.scope,
        "executions": args.executions,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "compile": {"backend": "inductor", "fullgraph": True, "dynamic": False},
        "warmup": args.warmup,
        "iterations": args.iterations,
        "timing": "rotating variants; synchronize before/after each call; first call excluded",
        "training_scope": "model + each library's eager pose criterion + backward; no optimizer or zero_grad",
        "comparison_limit": "different architectures/objectives/weights; speed comparison, not accuracy parity",
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path("src/nyolo/models/yolo26/pose.py"),
                Path("src/nyolo/learning/criterion.py"),
            )
        },
    }
    try:
        report.update(run(args))
    except Exception as error:
        report.update(passed=False, error=str(error), traceback=traceback.format_exc())
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "median_ms": {k: v["median_ms"] for k, v in report["timings"].items()},
            },
            indent=2,
        ),
        flush=True,
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
