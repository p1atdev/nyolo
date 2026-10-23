# Implementation provenance

nyolo is an independently written PyTorch implementation. No Ultralytics source,
model YAML, serialized modules, or pretrained parameters are used in this project.

Specification sources:

- Jocher et al., *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models*,
  arXiv:2606.03748v1 (2026-06-02), https://arxiv.org/abs/2606.03748.
  Sections 3.2, 3.3, 3.4.2 and supplementary Figures S1/S2 describe the algorithms
  and architecture. The paper is published under CC BY 4.0. Its diagrams and
  text are not redistributed here.
- Li et al., *Human Pose Regression with Residual Log-likelihood Estimation*,
  ICCV 2021, https://arxiv.org/abs/2107.11291 (RLE concept).
- Dinh et al., *Density estimation using Real NVP*, ICLR 2017,
  https://arxiv.org/abs/1605.08803 (affine coupling concept).
- Ultralytics, Model YAML Configuration Guide (numeric nano scaling factors only),
  https://docs.ultralytics.com/tr/guides/model-yaml-config.
  No model YAML is copied, translated, parsed or distributed.
- COCO's standard 17-joint ordering and numerical OKS sigma data:
  https://cocodataset.org/#keypoints-eval and the original COCO evaluation reference
  https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py.
  No COCO evaluation implementation is copied.

Runtime dependencies are PyTorch (BSD-style), NumPy (BSD-style), and Pillow
(MIT-CMU), under their own licenses. They are dependencies rather than vendored
implementations. The root LICENSE applies to nyolo's independently authored code.

Implementation-specific choices and limits are recorded in specs/yolo26-pose.md.
Source-code licensing does not relicense training data or third-party weights.
Official checkpoint compatibility and official accuracy reproduction are not
claimed. Independent implementation is not a determination of third-party rights.
