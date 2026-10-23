import json

import pytest
import torch
from PIL import Image

from nyolo import YOLO26Pose
from nyolo.geometry.transforms import ImageTransform
from nyolo.inference import Predictor, decode, letterbox
from nyolo.inference.decode import decode_keypoints
from nyolo.pose.types import RawPose
from nyolo.runtime import resolve_device


def raw_fixture(device):
    return RawPose(
        torch.ones(1, 2, 4, device=device),
        torch.tensor([[[8.0], [-8.0]]], device=device),
        torch.tensor(
            [[[[-3.0, 0.0], [4.0, 5.0]], [[0.0, 0.0], [1.0, 1.0]]]], device=device
        ),
        torch.full((1, 2, 2), 8.0, device=device),
        None,
        torch.tensor([[4.0, 4.0], [12.0, 12.0]], device=device),
        torch.full((2, 1), 8.0, device=device),
        (32, 32),
    )


def test_unbounded_decode_and_serialization(device):
    raw = raw_fixture(device)
    xy = decode_keypoints(raw)
    torch.testing.assert_close(xy[0, 0], xy.new_tensor([[-44.0, 4.0], [68.0, 84.0]]))
    result = decode(raw, max_detections=4, score_threshold=0.5)
    assert result.keypoints.shape == (1, 4, 2, 2)
    assert result.valid.tolist() == [[True, False, False, False]]
    records = json.loads(json.dumps(result.to_records(), allow_nan=False))
    assert records[0][0]["keypoints"] == [[-44.0, 4.0], [68.0, 84.0]]
    assert records[0][0]["keypoint_scores"][0] > 0.99


def test_letterbox_rounding_and_off_image_inverse(device):
    image = Image.new("RGB", (137, 51), (255, 0, 0))
    pixels, transform = letterbox(image, (96, 128))
    assert pixels.shape == (3, 96, 128)
    assert transform.resized_size == (48, 128)
    assert transform.padding == (0, 24)
    xy = torch.tensor([[-20.0, -10.0], [170.0, 80.0], [42.0, 27.0]], device=device)
    torch.testing.assert_close(transform.points(transform.points(xy), inverse=True), xy)
    assert transform.points(xy)[0, 0] < 0
    assert pixels.max() == 1.0
    assert pixels[:, 0, 0].tolist() == pytest.approx([114 / 255] * 3)


def test_transform_validation():
    with pytest.raises(ValueError):
        ImageTransform((10, 10), (32, 32), (40, 40), (0, 0))


def test_nonfinite_prediction_is_an_error(device):
    raw = raw_fixture(device)
    raw.keypoint_offsets[0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        decode(raw)


def test_predictor_restores_non_square_image_without_clipping_joints(device):
    class FixedModel(YOLO26Pose):
        def forward(self, images):
            return raw_fixture(images.device)

    predictor = Predictor(FixedModel(), device=device, image_size=(32, 32))
    result = predictor([Image.new("RGB", (16, 8))], max_detections=1)
    torch.testing.assert_close(
        result.keypoints[0, 0],
        result.keypoints.new_tensor([[-22.0, -2.0], [34.0, 38.0]]),
    )
    assert result.valid.item()


def test_explicit_execution_policy(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(RuntimeError, match="implicit CPU"):
        resolve_device("mps")
    with pytest.raises(ValueError, match="cpu or mps"):
        resolve_device("cuda")
