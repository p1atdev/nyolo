# SDPA validation — 2026-10-23

Local macOS / Apple Silicon, Python 3.12.11, PyTorch 2.14.1, float32.
MPS probes ran with `PYTORCH_ENABLE_MPS_FALLBACK=0`. Dependency versions and
uv exclude-newer settings were unchanged.

## Execution contract

`YOLO26Pose`, `load_checkpoint` and `load_training_state` accept the keyword-only
`attention_backend`, with `"sdpa"` as the default and `"naive"` as the other choice.
The CLI and development probes expose `--attention-backend` with the same values.
Every backbone/neck attention block receives the construction-time choice.

SDPA delegates internal kernel selection to PyTorch. It preserves unequal Q/K
and V dimensions, uses the original key-dimension scale, and disables masking,
causality and dropout. Errors propagate; there is no retry through naive or a
different device/dtype. Neither external flash-attn nor CUDA support was added.

Checkpoint format, architecture IDs, state keys and parameter shapes are unchanged.
Backend choice is not stored in the checkpoint. Loading an old checkpoint therefore
uses SDPA by default; use `attention_backend="naive"` for the original calculation
path. Different backends may differ in floating-point rounding.

## Checks performed

- All 74 tests passed. CPU and MPS checks compare attention outputs, input and
  parameter gradients in train/eval modes for the 128/256/384-channel blocks.
  Output tolerances are rtol=1e-4, atol=1e-5; gradient tolerances are 1e-3, 1e-4.
- All five model scales agree between naive and SDPA for rectangular 64x96 inputs,
  for inference and both pose heads including uncertainty, on CPU and MPS.
  The comparison uses rtol=1e-3, atol=1e-4.
- The n model's real training loss and gradients agree between backends on CPU
  and MPS, including off-image and unknown keypoints. The global relative gradient
  L2 error gate is 0.005; loss tolerances are rtol=1e-3, atol=1e-4.
- Model and training-component checkpoints load under either backend. Same-backend
  output equality, criterion state, CLI selection, rejection of unknown backends,
  and propagation of an injected SDPA error are covered.
- The pre-change synthetic-trained n checkpoint reproduced its historical PIL
  predictions exactly under naive. Its naive and SDPA raw outputs also matched
  exactly on the sampled 96x96 and 640x640 MPS inputs. This is a specific observed
  result, not a general bit-for-bit guarantee.
- All n/s/m/l/x scales passed MPS 640x640 inference and a synthetic 96x96 batch-2
  loss/backward/SGD update, with gradients reaching all trainable parameters and
  exactly matching checkpoint reload outputs.
- Ruff lint/format and ty checks passed.

## Model-only compilation

Both n-model MPS probes used Inductor, `fullgraph=True`, `dynamic=False`, with
five measured iterations after warmup and no recompilation permitted after warmup.
Each captured one graph. The criterion remains outside the compiled model path.

| Path | Input | Maximum output absolute error vs SDPA eager | Relative gradient L2 error | Result |
|---|---|---:|---:|---|
| Inference | 1x3x640x640 | 0 | n/a | pass |
| Training forward + backward | 2x3x96x96 | 4.56e-5 | 1.28e-4 | pass |

The training loss also passed rtol=1e-3, atol=1e-4. Compilation checks here cover n
only; the earlier naive compile matrix does not establish SDPA support for other
scales. These checks do not measure SDPA speedup over naive, real-data accuracy,
mixed precision, CUDA, or full-resolution training.

## Reproduction

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run --frozen pytest -q

PYTORCH_ENABLE_MPS_FALLBACK=0 uv run --frozen python tools/validate_scales.py \
  --device mps --attention-backend sdpa --output artifacts/attention-sdpa/scales.json

PYTORCH_ENABLE_MPS_FALLBACK=0 uv run --frozen python tools/probe_compile.py \
  --device mps --attention-backend sdpa --scale n --path inference \
  --size 640 --iterations 5 --output artifacts/attention-sdpa/n-inference.json

PYTORCH_ENABLE_MPS_FALLBACK=0 uv run --frozen python tools/probe_compile.py \
  --device mps --attention-backend sdpa --scale n --path model_train \
  --size 96 --iterations 5 --output artifacts/attention-sdpa/n-model-train.json
```

Local results are saved in the ignored `artifacts/attention-sdpa/` directory.
`existing-checkpoint.json` records the separate pre-change checkpoint check.
