from copy import deepcopy
from math import log

import torch
from torch import Tensor, nn

from nyolo.models.yolo26.backbone import Backbone
from nyolo.models.yolo26.config import YOLO26PoseConfig
from nyolo.models.yolo26.neck import Neck
from nyolo.nn.blocks import Conv
from nyolo.pose.types import DualPose, RawPose


def _tower(in_channels: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        Conv(in_channels, hidden, 3),
        Conv(hidden, hidden, 3),
        nn.Conv2d(hidden, outputs, 1),
    )


def _projection_bias(module: nn.Module) -> Tensor:
    assert isinstance(module, nn.Sequential)
    output = module[-1]
    assert isinstance(output, nn.Conv2d)
    assert output.bias is not None
    return output.bias


class PoseHead(nn.Module):
    def __init__(self, config: YOLO26PoseConfig) -> None:
        super().__init__()
        k = config.schema.num_keypoints
        self.k = k
        channels = config.head_channels
        box_width = max(16, channels[0] // 4)
        class_width = channels[0]
        self.box = nn.ModuleList(_tower(c, box_width, 4) for c in channels)
        self.classes = nn.ModuleList(
            nn.Sequential(
                Conv(c, c, 3, groups=c),
                Conv(c, class_width),
                Conv(class_width, class_width, 3, groups=class_width),
                Conv(class_width, class_width),
                nn.Conv2d(class_width, config.num_classes, 1),
            )
            for c in channels
        )
        self.joints = nn.ModuleList(
            _tower(c, max(box_width, 3 * k), 3 * k) for c in channels
        )
        self.sigma = nn.ModuleList(
            _tower(c, max(box_width, 2 * k), 2 * k) for c in channels
        )
        for level, stride in enumerate((8, 16, 32)):
            nn.init.constant_(_projection_bias(self.box[level]), 1.0)
            nn.init.constant_(
                _projection_bias(self.classes[level]),
                log(5 / config.num_classes / (640 / stride) ** 2),
            )
            nn.init.zeros_(_projection_bias(self.joints[level]))
            nn.init.zeros_(_projection_bias(self.sigma[level]))

    def forward(
        self,
        features: tuple[Tensor, Tensor, Tensor],
        image_size: tuple[int, int],
        uncertainty: bool,
    ) -> RawPose:
        boxes, classes, joints, sigmas, anchors, strides = [], [], [], [], [], []
        for level, (feature, stride) in enumerate(
            zip(features, (8, 16, 32), strict=True)
        ):
            b, _, h, w = feature.shape
            boxes.append(self.box[level](feature).flatten(2).transpose(1, 2))
            classes.append(self.classes[level](feature).flatten(2).transpose(1, 2))
            joints.append(
                self.joints[level](feature)
                .flatten(2)
                .transpose(1, 2)
                .reshape(b, h * w, self.k, 3)
            )
            if uncertainty:
                sigmas.append(
                    self.sigma[level](feature)
                    .flatten(2)
                    .transpose(1, 2)
                    .reshape(b, h * w, self.k, 2)
                )
            y, x = torch.meshgrid(
                torch.arange(h, device=feature.device, dtype=feature.dtype),
                torch.arange(w, device=feature.device, dtype=feature.dtype),
                indexing="ij",
            )
            anchors.append((torch.stack((x, y), -1).reshape(-1, 2) + 0.5) * stride)
            strides.append(feature.new_full((h * w, 1), stride))
        points = torch.cat(joints, 1)
        return RawPose(
            torch.cat(boxes, 1),
            torch.cat(classes, 1),
            points[..., :2],
            points[..., 2],
            torch.cat(sigmas, 1) if uncertainty else None,
            torch.cat(anchors),
            torch.cat(strides),
            image_size,
        )


class YOLO26Pose(nn.Module):
    """Explicit paths; train()/eval() affect normalization, not the return type."""

    def __init__(self, config: YOLO26PoseConfig | None = None) -> None:
        super().__init__()
        self.config = YOLO26PoseConfig() if config is None else config
        self.backbone = Backbone(self.config)
        self.neck = Neck(self.config)
        self.one_to_many = PoseHead(self.config)
        self.one_to_one = deepcopy(self.one_to_many)

    def _features(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Images must have shape B,3,H,W")
        h, w = images.shape[-2:]
        if min(h, w) < 32 or h % 32 or w % 32:
            raise ValueError("Input dimensions must be positive multiples of 32")
        if self.training and images.shape[0] * (h // 32) * (w // 32) <= 1:
            raise ValueError(
                "Training BatchNorm requires more than one P5 value per channel"
            )
        if images.dtype != torch.float32:
            raise ValueError(
                "This release supports float32 images and model parameters"
            )
        return self.neck(self.backbone(images))

    def forward(self, images: Tensor) -> RawPose:
        return self.one_to_one(self._features(images), tuple(images.shape[-2:]), False)

    def forward_train(self, images: Tensor) -> DualPose:
        features = self._features(images)
        size = tuple(images.shape[-2:])
        return DualPose(
            self.one_to_many(features, size, True),
            self.one_to_one(features, size, True),
        )
