import pytest
import torch

from nyolo import COCO17, PoseSchema, YOLO26Pose, YOLO26PoseConfig
from nyolo.checkpoint import (
    load_checkpoint,
    load_training_state,
    save_checkpoint,
    save_training_state,
)
from nyolo.learning import PoseTrainingModule


def test_nano_pyramid_and_inference_contract(device):
    model = YOLO26Pose().to(device).eval()
    images = torch.zeros(1, 3, 96, 128, device=device)
    with torch.no_grad():
        features = model.neck(model.backbone(images))
        assert [tuple(x.shape) for x in features] == [
            (1, 64, 12, 16),
            (1, 128, 6, 8),
            (1, 256, 3, 4),
        ]
        raw = model(images)
        assert raw.class_logits.shape == (1, 252, 1)
        assert raw.keypoint_offsets.shape == (1, 252, 17, 2)
        assert raw.sigma_logits is None
        dual = model.forward_train(images)
        assert dual.one_to_one.sigma_logits.shape == (1, 252, 17, 2)
        assert dual.one_to_many.class_logits.shape == raw.class_logits.shape


def test_custom_schema_checkpoint_and_rle_state(tmp_path, device):
    schema = PoseSchema("custom3", ("a", "b", "c"), (0, 2, 1), (0.1, 0.2, 0.2))
    module = (
        PoseTrainingModule(YOLO26Pose(YOLO26PoseConfig(schema=schema)))
        .to(device)
        .eval()
    )
    with torch.no_grad():
        module.criterion.flow.layers[0].net[-1].bias.fill_(0.3)
        images = torch.rand(1, 3, 64, 64, device=device)
        expected = module.model(images)
    path = tmp_path / "training.pt"
    save_training_state(path, module)
    restored = load_training_state(path, device=device).eval()
    assert restored.model.config == module.model.config
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, module.state_dict()[name], atol=0, rtol=0)
    with torch.no_grad():
        actual = restored.model(images)
    torch.testing.assert_close(
        actual.keypoint_offsets, expected.keypoint_offsets, atol=0, rtol=0
    )
    save_checkpoint(tmp_path / "model.pt", module.model)
    assert load_checkpoint(tmp_path / "model.pt", device=device).config.schema == schema
    with pytest.raises(ValueError, match="no criterion"):
        load_training_state(tmp_path / "model.pt", device=device)


def test_reject_unsupported_contracts():
    with pytest.raises(ValueError, match="only"):
        YOLO26PoseConfig(scale="s")
    with pytest.raises(ValueError, match="sigma"):
        PoseSchema("bad", ("a",), (0,), ())
    with pytest.raises(ValueError, match="multiples"):
        YOLO26Pose().eval()(torch.zeros(1, 3, 65, 64))
    assert COCO17.num_keypoints == 17
