# Initial MPS validation — 2026-10-23

Environment: local macOS arm64, Python 3.12.11, PyTorch 2.14.1, float32 eager,
device `mps:0`. `PYTORCH_ENABLE_MPS_FALLBACK=0` was set. There is no CPU
replacement for an unsupported MPS operation.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run --frozen python tools/validate_pose.py \
  --device mps --steps 240 --output artifacts/mps-smoke
```

The full n model (3,640,624 parameters including both heads and uncertainty
towers) overfit two synthetic 96x96 images. One unknown joint and eight off-image
joint annotations were included across the batch. No pretrained weights were used.

| Measurement | Before | After 240 steps |
|---|---:|---:|
| Mean known-joint error, pixels | 56.925 | 5.710 |
| Mean off-image-joint error, pixels | 88.152 | 5.894 |
| Off-image points on the correct side | 0% | 100% |
| Mean bbox IoU | .223 | .887 |
| Mean top detection score | .012 | .867 |

The gates passed. Total probe time including full-resolution checks was 39.9 s
on this run; this is not a calibrated inference-speed benchmark.

Additional evidence:

- 32 tests passed, including 14 tests executed on MPS and corresponding CPU
  checks, plus schema/runtime validation. Ruff lint/format and ty passed.
- Nonzero finite first-step gradients reached the backbone, neck, both pose
  heads, uncertainty head and trainable RLE flow.
- A 640x640 inference produced shape `[1,8400,17,2]` for raw keypoint offsets.
- A separate full-resolution 640x640 **training forward/backward** completed on
  MPS: loss 20.478, finite nonzero gradient norm 176.101 before clipping.
- Model checkpoint reloading exactly preserved outputs on the same device.
- PIL Predictor output matched direct tensor inference, including off-image XY.
- The installed CLI read the generated PNGs and checkpoint and wrote unbounded
  JSON predictions. Separate tests cover non-square letterbox inversion.
- Source and wheel distributions built successfully; importing the wheel directly
  and running its model succeeded. LICENSE and NOTICE are included in the wheel;
  specifications, tests and validation tool are included in the source distribution.

Local machine-readable results are in `artifacts/mps-smoke/metrics.json`, with
predictions and synthetic-only checkpoints beside them. These generated files
are ignored by Git. The report documents an observed run, not a cross-platform
bit-for-bit reproducibility guarantee.

This validates trainability and runtime behavior on the synthetic inputs. It does
not establish COCO AP, real-data generalization, equivalence to official weights,
or CUDA/compile/Triton/TPU support.

The subsequent [scale and torch.compile investigation](scale-compile-validation.md)
records the expanded model family and current compiler evidence separately.
