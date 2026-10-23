# Scale and torch.compile validation — 2026-10-23

Local macOS / Apple Silicon, Python 3.12.11, PyTorch 2.14.1, float32, explicit
MPS execution. `PYTORCH_ENABLE_MPS_FALLBACK=0`. No dependency versions or uv
exclude-newer settings were changed.

## Scope and implementation

The previous model was fixed to n. `YOLO26PoseConfig(scale=...)` now expands the
same independent pose graph for n/s/m/l/x, with compound width, depth and channel
cap scaling. The n architecture ID, parameter dimensions and checkpoint format
are retained. This report concerns nyolo's graph, not equivalence to official
Ultralytics weights or accuracy.

Numeric scale settings come from the public
[model configuration guide](https://docs.ultralytics.com/tr/guides/model-yaml-config).
The construction choices are fixed in [the specification](specs/yolo26-pose.md).

## Native MPS execution

Every scale passed:

- 640x640 inference, batch 1: raw keypoint shape `[1,8400,17,2]`.
- Loss, backward and an SGD update on a 96x96 batch of 2 synthetic images,
  including off-image wrists/ankles and one unknown point.
- Finite nonzero gradients for backbone, neck, both pose heads, uncertainty and
  RLE. Every trainable parameter received a gradient tensor (some can be zero).
- Model checkpoint save/load with the scale and schema retained, and exactly
  matching inference on the same device.

| Scale | Parameters, including both heads and uncertainty | Native execution |
|---|---:|---|
| n | 3,640,624 | pass |
| s | 12,440,512 | pass |
| m | 26,964,444 | pass |
| l | 32,540,124 | pass |
| x | 73,103,292 | pass |

These are functional checks, not evidence of training convergence or generalization
for the new sizes. Full-resolution training memory/performance for s/m/l/x is not
covered by the 96x96 training probe.

The original n checkpoint reproduced its pre-refactor JSON predictions exactly.
`model.compile(...)` in-place also worked through `Predictor`, and saving/reloading
that model retained normal state keys and exactly reproduced the uncompiled
reference. These API checks are recorded in `artifacts/scale-compile/api-smoke.json`.

All 38 unit tests passed, including the existing MPS checks and new scale/version
contracts. Ruff lint/format, ty and wheel/source-distribution builds passed.

## TorchInductor inference

Each case ran in its own process, sequentially, with
`backend="inductor", fullgraph=True, dynamic=False`. Each captured exactly one
graph. After warm-up, guard misses were configured to fail rather than recompile
or enter an alternate execution path. Eager/compiled measurements alternated order,
with 3 warm-up pairs, 20 measured pairs and MPS synchronization on each call.
The table excludes compilation, PIL, decode, JSON and data loading. The initial
matrix used random initialization and a synthetic input. No simultaneous GPU
validation process was running during this matrix.

| Scale | Eager median, ms | Compiled median, ms | Speedup | Output agreement |
|---|---:|---:|---:|---|
| n | 27.97 | 22.34 | 1.25x | pass |
| s | 32.65 | 26.50 | 1.23x | pass |
| m | 64.13 | 55.17 | 1.16x | pass |
| l | 77.72 | 66.79 | 1.16x | pass |
| x | 184.82 | 155.06 | 1.19x | pass |

A separate n inference probe loaded the original synthetic-trained checkpoint:
19.03 -> 13.48 ms (1.41x), maximum absolute raw-output error 1.91e-6. This also
passed the same agreement gate. This checkpoint is trained only on the earlier
two-image synthetic probe; it is not a real-data benchmark. Timings depend on
thermal state, desktop activity, shape and weights, and are not universal speedups.
First compiled-call times were measured with the existing disk cache available;
they are not cold-cache compilation benchmarks.

## Compiled training forward

Only `model.forward_train` is compiled. The real pose criterion remains explicitly
outside the graph, and backward differentiates through the compiled model.
Input: `[2,3,96,96]`. No optimizer update is included in the timing below.

Output/loss comparison uses rtol=1e-3, atol=1e-4; the global relative gradient L2
gate is 0.005. The thresholds were not relaxed in response to failures.

| Scale | Eager / compiled median, ms | Relative gradient L2 error | Numerical gate |
|---|---:|---:|---|
| n | 73.74 / 47.73 | 0.000125 | pass |
| s | 79.28 / 54.50 | 0.000249 | pass |
| m | 95.91 / 72.73 | 0.000138 | pass |
| l | 117.28 / 86.12 | 0.001660 | pass |
| x, diagnostic repeat | 144.66 / 125.68 | 0.000289 | **output gate failed** |

x executes both forward and backward with finite values, and the loss and gradient
gates pass. Its raw output gate fails on a few near-zero values. The first matrix
stopped comparison at one mismatched element (difference 1.05e-4). The detailed
repeat measured max absolute raw difference 2.49e-4; each head had one keypoint-offset
and one uncertainty-logit element outside tolerance. Eager-to-eager repeats had
zero output and gradient differences. This separates execution capability from
the stricter numerical acceptance result; x is not marked fully validated for
compiled training under the current gate.

An explicitly separate `options={"layout_optimization": False}` experiment did
not remove those differences (relative gradient error 0.000289, output gate still
failed). It is not installed as a workaround or selected automatically. A public
[MPS layout-performance issue](https://github.com/pytorch/pytorch/issues/192551)
motivated this diagnostic, but does not establish the cause of these numerical
differences.

## Boundaries that do not compile as one full graph

Both probes fail with Dynamo `Unsupported: Data-dependent branching`:

- `decode(model(images))`: the nonfinite-output check at `if not finite.all()`.
- `PoseTrainingModule(images, targets, progress=...)`: annotation validation's
  finite/geometry checks in `PoseTargets.validate`.

The validation behavior was retained; it was not replaced by clipping, suppression
of nonfinite values, or implicit eager execution. Compile the model Tensor path
and call validated postprocessing/criterion separately.
PyTorch documents the strict behavior of
[fullgraph](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html) and
[fail_on_recompile](https://docs.pytorch.org/docs/2.14/generated/torch.compiler.set_stance.html).

## Reproduction and local evidence

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run --frozen python tools/validate_scales.py \
  --device mps --output artifacts/scale-compile/scales.json

PYTORCH_ENABLE_MPS_FALLBACK=0 uv run --frozen python tools/run_compile_matrix.py \
  --device mps --output artifacts/scale-compile/inductor

PYTORCH_ENABLE_MPS_FALLBACK=0 uv run --frozen python tools/probe_compile.py \
  --device mps --scale x --path model_train --repeat-reference \
  --output artifacts/scale-compile/x-training-diagnostics.json
```

`scales.json`, `inductor/matrix.json`, per-case JSON/logs, the x diagnostic files
and `n-trained-inference.json` are local ignored artifacts. The probe records a
failure and exits nonzero on a failed numerical gate; the matrix runner continues
independent named cases so their outcomes can all be inspected. It does not change
backend or device in response to a failure. Current child run IDs are checked to
avoid accepting a result left by an earlier process.

Real-data accuracy, dynamic-shape compilation, CUDA, Triton and TPU are outside
the evidence collected here.
