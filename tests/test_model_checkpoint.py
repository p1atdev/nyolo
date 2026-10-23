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
    with pytest.raises(ValueError, match="scales"):
        YOLO26PoseConfig(scale="tiny")
    with pytest.raises(ValueError, match="sigma"):
        PoseSchema("bad", ("a",), (0,), ())
    with pytest.raises(ValueError, match="multiples"):
        YOLO26Pose().eval()(torch.zeros(1, 3, 65, 64))
    assert COCO17.num_keypoints == 17


@pytest.mark.parametrize(
    "scale,channels,repeats",
    (
        ("n", (64, 128, 256), 1),
        ("s", (128, 256, 512), 1),
        ("m", (256, 512, 512), 1),
        ("l", (256, 512, 512), 2),
        ("x", (384, 768, 768), 2),
    ),
)
def test_compound_scale_contract(scale, channels, repeats):
    config = YOLO26PoseConfig(scale=scale)
    assert config.head_channels == channels
    assert config.repeats == repeats
    assert config.architecture_version == f"nyolo26-pose-{scale}-v1"
    model = YOLO26Pose(config).eval()
    with torch.no_grad():
        images = torch.rand(1, 3, 64, 96)
        features = model.neck(model.backbone(images))
        assert [x.shape[1] for x in features] == list(channels)
        assert model(images).keypoint_offsets.shape == (1, 126, 17, 2)
        dual = model.forward_train(images)
        assert dual.one_to_one.sigma_logits.shape == (1, 126, 17, 2)
    if scale == "n":
        # The refactor must retain the original n architecture/state dimensions.
        assert sum(p.numel() for p in model.parameters()) == 3640624


def test_reject_scale_version_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        YOLO26PoseConfig(scale="x", architecture_version="nyolo26-pose-n-v1")
    assert (
        YOLO26PoseConfig(architecture_version="nyolo26-pose-n-v1") == YOLO26PoseConfig()
    )
