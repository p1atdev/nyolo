"""Independent, trainable YOLO26 pose components for PyTorch."""

from nyolo.models.yolo26 import YOLO26Pose, YOLO26PoseConfig
from nyolo.pose.schema import COCO17, PoseSchema

__all__ = ["COCO17", "PoseSchema", "YOLO26Pose", "YOLO26PoseConfig"]
