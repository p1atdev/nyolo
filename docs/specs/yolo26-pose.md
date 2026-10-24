# nyolo26-pose-{n,s,m,l,x}-v1

This document fixes the implementation contract. It is a specification for an
independent YOLO26-Pose implementation, not an official checkpoint conversion
specification. Sources and attribution are in [provenance](../provenance.md).

## Architecture

The macro graph and block compositions follow paper supplementary Figures S1/S2.
Each scale has its own explicit architecture version. The numeric compound-scale
settings are taken from the public guide; the independently implemented graph and
per-stage block choices are shared across these scales. This is not a claim of
official larger-model parameter counts or checkpoint compatibility.

| Scale | Depth | Width | Channel cap before width | Head channels | Repeated blocks |
|---|---:|---:|---:|---|---:|
| n | .5 | .25 | 1024 | 64/128/256 | 1 |
| s | .5 | .5 | 1024 | 128/256/512 | 1 |
| m | .5 | 1 | 512 | 256/512/512 | 1 |
| l | 1 | 1 | 512 | 256/512/512 | 2 |
| x | 1 | 1.5 | 512 | 384/768/768 | 2 |

Channels are rounded upward to a multiple of eight after cap/width scaling.
Repeated C3k2 and C2PSA units use max(round(2*depth),1) iterations. The final
attention neck C3k2 has one iteration. C3k has two internal bottlenecks. No generic
YAML parser is used. The table below details n; other scales apply these rules.

| Backbone stage | Output channels | Stride | Block |
|---|---:|---:|---|
| Stem 1 | 16 | 2 | 3x3 Conv |
| Stem 2 | 32 | 4 | 3x3 Conv |
| Stem 3 | 64 | 4 | C3k2, expansion .25 |
| P3 downsample | 64 | 8 | 3x3 Conv |
| P3 | 128 | 8 | C3k2, expansion .25 |
| P4 | 128 | 16 | 3x3 Conv, C3k2 with C3k |
| P5 | 256 | 32 | 3x3 Conv, C3k2 with C3k, SPPF, C2PSA |

Neck: upsample P5 and concatenate P4 -> C3k2(128); upsample and concatenate
P3 -> C3k2(64); downsample and concatenate the first neck output -> C3k2(128);
downsample and concatenate P5 -> C3k2(256) with Bottleneck + PSABlock.
The head receives channels 64/128/256 at strides 8/16/32.

Both heads have separate box, class, joint and uncertainty branches. The one-to-one
head is initialized as a copy of the one-to-many head; both propagate gradients
to the shared features. forward() runs only one-to-one without uncertainty;
forward_train() runs both heads with uncertainty, irrespective of train/eval mode.
train/eval controls BatchNorm in the normal PyTorch manner.

## Independent implementation decisions

The paper does not specify every pose tower width, normalization detail, or
checkpoint representation. This release explicitly chooses:

- Conv + BatchNorm (eps .001, momentum .03) + SiLU; residual spatial attention
  with q/k dimension half of the value dimension, and depthwise positional Conv.
- With C the first head's input channels, box towers have two Conv layers of
  width max(16,C/4) and a four-channel projection. Class towers have two
  depthwise/pointwise pairs, hidden width C.
- Joint towers have width max(box_width,3K), output 3K.
  Uncertainty towers have width max(box_width,2K), output 2K.
- Grid centers a=(x+.5,y+.5)*stride. Signed box distances d predict
  [a_x-s*d_l, a_y-s*d_t, a_x+s*d_r, a_y+s*d_b] with no DFL.
- Keypoints decode as a + 2*stride*offset. XY has no bounded activation.
- Shared 2D RealNVP: six alternating affine couplings, two hidden layers of 64,
  SiLU, bounded log scale 2*tanh(s), standard-normal base. Final coupling layers
  are initialized to zero. No flow is evaluated during inference.
- RLE residuals use sqrt(original target box area) as a common person scale.
  Uncertainty is sigmoid(logit), lower bounded at 1e-4 for logarithms/division.
- Loss gains: CIoU 7.5, direct distance L1 1, classification BCE .5, OKS 24,
  RLE 1, keypoint localizability BCE 1. These are explicit initial settings,
  not a claim to reproduce the released model's optimized recipe.

These decisions are part of architecture versions nyolo26-pose-{scale}-v1. The n
state-dictionary keys, dimensions and architecture ID retain their original form.
The default architecture ID is derived from scale; an explicitly mismatched ID
is rejected. Official
parameter names, exact numerical equivalence and weight compatibility are not
promised. Both heads and uncertainty towers are retained in saved model state.

## Assignment and loss

TAL ranks candidates by p^.5 * IoU^6. STAL independently expands a target dimension
smaller than 8 pixels to 16 pixels **for candidate selection only**. Regression and
IoU use the original image-intersection box. One-to-many selects top 10 candidates.
One-to-one takes top 7 and selects its best single candidate. Conflicts go to the
highest-IoU target; no nearest-anchor rematching is performed. A crowded target can
remain unmatched. Candidate and matching selection are outside autograd.

Quality targets use per-GT normalized alignment scores scaled by the maximum
positive IoU. Losses are quality weighted, with the summed quality lower bounded
at one for reduction. Images with no GT still train background classification.
Unknown points are masked before nonlinear arithmetic, so unknown XY may be NaN.

The total mixes both branch objectives using alpha=.8-.7*progress as the
one-to-many weight. Explicit progress in [0,1] is required. PoseTrainingModule
owns both model and criterion, so .parameters() includes trainable RLE state.
This package supplies no optimizer, Dataset, augmentation pipeline or trainer.

## Unbounded pose contract

- Targets are padded B,T tensors with a separate valid-instance mask.
- Boxes are finite nonempty xyxy image-intersection boxes in network-input pixels.
- Keypoints are K (x,y) coordinates in the same continuous pixel coordinate chart.
  Negative and beyond-image values are valid and are never clamped.
- v=0 means unlocalized/unknown; v=1 means occluded (including off image);
  v=2 means visible. Known points require finite coordinates. An off-image v=2
  annotation is an error, not an automatically changed label.
- v>0 trains XY, OKS and RLE. Keypoint score BCE also targets v>0. The score is
  localizability confidence, not a probability of physical visibility.
- A custom skeleton requires explicit joint order, flip permutation and per-joint
  OKS constants. There is no inferred joint mapping or default sigma substitution.

Letterboxing uses RGB/255, bilinear image resize, centered RGB 114 padding and the
actual independently rounded horizontal/vertical scales. Transform metadata is
used for inverse coordinates. Boxes may be clipped at inference; keypoints remain
unbounded through transforms, filtering, checkpoint use and JSON serialization.

## Runtime and scope

Supported: PyTorch float32 eager on explicit cpu or mps; rectangular input sides
divisible by 32; n/s/m/l/x pose; tensor and PIL inference; training primitives.
MPS entry points reject PYTORCH_ENABLE_MPS_FALLBACK=1. No automatic device, dtype,
alternate head or cross-backend retry is supplied. Nonfinite decoded
predictions are an error. CUDA, Triton, TPU and other tasks are pending.

Attention is an execution option, selected with attention_backend="sdpa" (default)
or "naive" at model construction and model/training-component loading. The choice
applies to every backbone and neck attention block, including repeated blocks.
It is fixed before torch.compile and exposed by model.attention_backend.
The state-dictionary keys, parameter shapes and architecture versions are unchanged.
The naive path retains the original matmul/softmax calculation order. SDPA uses
PyTorch's native kernel dispatch with no mask, no dropout, noncausal attention and
an explicit scale of key_dim**-0.5. Its B,H,N,D inputs retain the original unequal
Q/K and V dimensions; only layout is changed. SDPA failures propagate without a
naive retry. Native dispatch is not a promise to use a specific fused kernel.
The external flash-attn package is not supported by this release.

Model-only torch.compile is investigated using explicit backend, fullgraph=True
and dynamic=False. Inference means model(images); training means forward_train
compiled with criterion outside the compiled region. PIL/JSON IO and annotation
validation are separate boundaries. No automatic eager retry is provided.
Local measurements and validation reports are kept in the Git-ignored notes/
directory rather than distributed with the specification.

Checkpoints contain a format version, architecture and coordinate policy, complete
joint schema and tensor state dictionaries. Loading uses weights_only=True and
strict state matching. Training-component files additionally retain RLE/criterion
state and loss config; they contain no optimizer, RNG, sampler or trainer state
and are not advertised as full training-resume files. Attention backend selection
is not serialized: both load functions default to SDPA, including for older files.
Pass attention_backend="naive" to reproduce the previous execution path. Changing
backends can change floating-point rounding without changing the model equations.

tools/validate_pose.py is a development-only synthetic overfit probe. It checks
both branch gradients, RLE gradients, unconstrained XY, eval-mode accuracy on the
two training images, 640x640 forward, model save/load and high-level inference.
Its metrics establish trainability, not accuracy on unseen images or COCO AP.
