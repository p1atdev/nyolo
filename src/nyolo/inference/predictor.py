from collections.abc import Sequence

import torch
from PIL import Image

from nyolo.inference.decode import decode
from nyolo.inference.preprocess import letterbox
from nyolo.models.yolo26 import YOLO26Pose
from nyolo.pose.types import PosePredictions
from nyolo.runtime import resolve_device


class Predictor:
    """Moves the model to eval/device and uses a fused copy by default.

    Pass fuse=False to explicitly retain separate Conv/BatchNorm execution.
    Fusion errors propagate; execution never retries an alternative path.
    """

    def __init__(
        self,
        model: YOLO26Pose,
        *,
        device: str,
        image_size: tuple[int, int] = (640, 640),
        fuse: bool = True,
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        if fuse:
            self.model = self.model.fuse()
        self.image_size = image_size

    @torch.inference_mode()
    def __call__(
        self,
        images: Sequence[Image.Image],
        *,
        score_threshold: float = 0.25,
        max_detections: int = 300,
    ) -> PosePredictions:
        if not images:
            raise ValueError("Supply at least one image")
        prepared = [letterbox(image, self.image_size) for image in images]
        batch = torch.stack([x[0] for x in prepared]).to(self.device)
        result = decode(
            self.model(batch),
            score_threshold=score_threshold,
            max_detections=max_detections,
        )
        boxes, points = [], []
        for i, (_, transform) in enumerate(prepared):
            original = transform.boxes(result.boxes[i], inverse=True)
            h, w = transform.original_size
            boxes.append(
                torch.minimum(original.clamp_min(0), original.new_tensor((w, h, w, h)))
            )
            points.append(transform.points(result.keypoints[i], inverse=True))
        restored_boxes = torch.stack(boxes)
        valid = result.valid & (restored_boxes[..., 2:] > restored_boxes[..., :2]).all(
            -1
        )
        return PosePredictions(
            restored_boxes,
            result.scores,
            result.class_ids,
            torch.stack(points),
            result.keypoint_scores,
            valid,
        )
